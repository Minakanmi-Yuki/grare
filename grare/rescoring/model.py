"""GraRe rescorer.

The published model uses candidate attributes, local geometry, object context,
and a multi-task head bundle:

- Local tier: 4 cm gripper-aligned point cloud encoded by ShellAttn (the
  stratified-shell encoder with cross-shell self-attention).
- Object tier: 512 camera-frame xyz points cut from the depth image by a
  MobileSAM mask whose point prompt is the candidate's projected 2D
  pixel; encoded by a frozen Point-MAE backbone and pose-conditioned via FiLM.
  Supervised at train time by an obj-id CE head over the GraspNet
  88-class object bank.
- Aux heads: collision (BCE on is_collision), empty (BCE on is_empty),
  obj_id (CE with ignore_index=-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShellAttnEncoder(nn.Module):
    """Stratified-shell point encoder with masked per-shell pooling and
    cross-shell self-attention.

    Inputs (passed by GraspRescorer.forward):
        local_cloud : (B, P, 3) xyz in gripper-local frame
        cloud_mask  : (B, P) bool — True for real, False for zero-pad

    Shell membership is computed at forward time from ``r = ‖xyz‖`` against
    ``shell_edges_m`` (e.g. (0, 0.005, 0.015, 0.025, 0.040)) — no auxiliary
    feature channels are required at inference, so the relabel archive only
    needs to store xyz.
    """

    def __init__(
        self,
        per_point_dim: int,
        hidden_dim: int,
        n_shells: int,
        n_heads: int,
        n_attn_layers: int,
        *,
        shell_edges_m: tuple[float, ...] = (0.0, 0.005, 0.015, 0.025, 0.040),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_shells = n_shells
        if len(shell_edges_m) != n_shells + 1:
            raise ValueError(
                f"shell_edges_m must have len(n_shells)+1 = {n_shells + 1}; got {len(shell_edges_m)}"
            )
        self.register_buffer(
            "shell_edges",
            torch.tensor(shell_edges_m, dtype=torch.float32),
            persistent=False,
        )
        self.point_mlp = nn.Sequential(
            MLPBlock(per_point_dim, 64, dropout=dropout),
            MLPBlock(64, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )
        self.shell_pos = nn.Parameter(torch.zeros(n_shells, hidden_dim))
        nn.init.normal_(self.shell_pos, std=0.02)
        attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shell_attn = nn.TransformerEncoder(
            attn_layer,
            num_layers=n_attn_layers,
            enable_nested_tensor=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _masked_max(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.full_like(feats[..., :1], -1e4)
        masked = torch.where(mask.unsqueeze(-1), feats, neg_inf)
        return masked.max(dim=1).values

    def _masked_mean(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        return (feats * m).sum(dim=1) / denom

    def _zero_empty_rows(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.any(dim=1, keepdim=True)
        return torch.where(valid, output, torch.zeros_like(output))

    def _shell_one_hot(self, local_cloud: torch.Tensor) -> torch.Tensor:
        # r = ‖xyz‖; bucketize into [edge_0, edge_1, ..., edge_n_shells].
        # `right=False` → left-closed/right-open buckets; r==edge_0 (=0) lands
        # in shell 0. `bucketize` returns indices in [0, n_shells]; clamp the
        # rightmost edge into the last shell.
        edges = self.shell_edges.to(local_cloud.device, dtype=local_cloud.dtype)
        radii = torch.linalg.vector_norm(local_cloud, dim=-1)
        ids = torch.bucketize(radii, edges[1:-1], right=False)
        ids = ids.clamp_(0, self.n_shells - 1)
        return torch.nn.functional.one_hot(ids, num_classes=self.n_shells).to(local_cloud.dtype)

    def forward(
        self,
        local_cloud: torch.Tensor,           # (B, P, 3)
        cloud_mask: torch.Tensor,            # (B, P) bool
    ) -> torch.Tensor:
        feats = self.point_mlp(local_cloud)  # (B, P, H)

        shell_one_hot = self._shell_one_hot(local_cloud)
        per_shell: list[torch.Tensor] = []
        for s in range(self.n_shells):
            shell_mask = cloud_mask & (shell_one_hot[..., s] > 0.5)
            per_shell.append(self._masked_max(feats, shell_mask))
        S = torch.stack(per_shell, dim=1)        # (B, n_shells, H)
        S = S + self.shell_pos.unsqueeze(0)
        shell_alive = (cloud_mask.unsqueeze(-1) & (shell_one_hot > 0.5)).any(dim=1)
        safe_shell_alive = shell_alive.clone()
        all_empty = ~safe_shell_alive.any(dim=1)
        if bool(all_empty.any()):
            safe_shell_alive[all_empty, 0] = True
        attn_mask = ~safe_shell_alive
        S = self.shell_attn(S, src_key_padding_mask=attn_mask)
        denom = shell_alive.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        z_attn = (S * shell_alive.unsqueeze(-1).float()).sum(dim=1) / denom
        z_max = self._masked_max(feats, cloud_mask)
        z_mean = self._masked_mean(feats, cloud_mask)
        output = self.proj(torch.cat([z_attn, z_max, z_mean], dim=-1))
        return self._zero_empty_rows(output, cloud_mask)


class PoseMLP(nn.Module):
    def __init__(
        self,
        pose_dim: int = 14,
        hidden_dim: int = 128,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(pose_dim, 64, dropout=dropout),
            MLPBlock(64, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        return self.net(pose_features)


@dataclass(frozen=True)
class RescorerConfig:
    """Configuration for the published candidate + local + object GraRe model."""

    pose_dim: int = 14
    point_dim: int = 3
    hidden_dim: int = 128
    dropout: float = 0.1
    shell_attn_n_shells: int = 4
    shell_attn_heads: int = 4
    shell_attn_layers: int = 1
    shell_attn_per_point_dim: int = 3
    shell_edges_m: tuple[float, ...] = (0.0, 0.005, 0.015, 0.025, 0.040)

    object_cloud_points: int = 512
    object_hidden_dim: int = 128
    object_pmae_ckpt: str = ""             # path to point_mae pretrain.pth ("" = random)
    object_pmae_num_group: int = 32        # patches per object cloud
    object_pmae_group_size: int = 32       # points per patch

    fusion_layers: int = 1
    fusion_heads: int = 4
    fusion_ffn_mult: int = 2

    num_object_classes: int = 88


class FiLMModulator(nn.Module):
    """(1 + γ) ⊙ x + β with γ, β predicted from a conditioning vector.

    Both projection layers are zero-initialised, so the modulator starts as
    the identity. Gradients only flow through γ, β once the optimiser learns
    to use them.
    """

    def __init__(self, target_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, target_dim)
        self.to_beta = nn.Linear(cond_dim, target_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return x * (1.0 + gamma) + beta


def _fps_batched(points: torch.Tensor, num: int) -> torch.Tensor:
    B, N, _ = points.shape
    if N == 0:
        return torch.zeros((B, num), dtype=torch.long, device=points.device)
    if N <= num:
        idx = torch.arange(N, device=points.device).unsqueeze(0).expand(B, -1)
        if N < num:
            pad = idx[:, -1:].expand(-1, num - N)
            idx = torch.cat([idx, pad], dim=1)
        return idx
    out = torch.empty((B, num), dtype=torch.long, device=points.device)
    distances = torch.full((B, N), float("inf"), device=points.device)
    radii = (points * points).sum(dim=2)
    seed = torch.argmax(radii, dim=1)
    out[:, 0] = seed
    last = points[torch.arange(B, device=points.device), seed]
    for s in range(1, num):
        diff = points - last.unsqueeze(1)
        d_new = (diff * diff).sum(dim=2)
        distances = torch.minimum(distances, d_new)
        idx = torch.argmax(distances, dim=1)
        out[:, s] = idx
        last = points[torch.arange(B, device=points.device), idx]
    return out


class ObjectEncoderPointMAE(nn.Module):
    """Object tier with a frozen-pretrained Point-MAE backbone + adapter.

    The SAM single-object cloud (B, P, 3), candidate-centred, is normalized
    to the unit sphere (matching ShapeNet pretraining), grouped into
    ``num_group`` patches (FPS + kNN), embedded by the pretrained Point-MAE
    patch encoder + Transformer, mean+max pooled, and projected to
    ``object_hidden_dim``.

    The Point-MAE patch encoder, positional embedding, Transformer blocks,
    and final normalization are frozen. Only the projection adapter is
    trainable, matching the published model.
    """

    def __init__(self, cfg: RescorerConfig) -> None:
        super().__init__()
        from grare.rescoring.point_mae import (
            PointMAEBlock,
            PointMAEPatchEncoder,
        )
        embed_dim, depth, num_heads = 384, 12, 6
        self.num_group = int(cfg.object_pmae_num_group)
        self.group_size = int(cfg.object_pmae_group_size)
        self.embed_dim = embed_dim

        self.patch_encoder = PointMAEPatchEncoder(encoder_channel=embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim),
        )
        self.blocks = nn.ModuleList(
            [PointMAEBlock(embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # adapter: pooled (mean+max = 2*embed) → object_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * embed_dim, cfg.object_hidden_dim),
            nn.LayerNorm(cfg.object_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
        )
        self._out_dim = int(cfg.object_hidden_dim)
        if cfg.object_pmae_ckpt:
            self._load_pretrained(cfg.object_pmae_ckpt)
        self._freeze_backbone()

    def _load_pretrained(self, path: str) -> None:
        """Map official ``module.MAE_encoder.*`` keys onto this layout."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("base_model", ckpt.get("state_dict", ckpt))
        prefix = "module.MAE_encoder."
        own = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            sub = k[len(prefix):].replace("blocks.blocks.", "blocks.")
            if sub.startswith("encoder."):
                sub = "patch_encoder." + sub[len("encoder."):]
            if sub in own and own[sub].shape == v.shape:
                own[sub] = v
                loaded += 1
        self.load_state_dict(own, strict=False)
        print(f"[ObjectEncoderPointMAE] loaded {loaded} pretrained tensors from {path}")

    def _freeze_backbone(self) -> None:
        for module in (self.patch_encoder, self.pos_embed, self.blocks, self.norm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> ObjectEncoderPointMAE:
        super().train(mode)
        # BatchNorm statistics are part of the frozen Point-MAE backbone.
        self.patch_encoder.eval()
        self.pos_embed.eval()
        self.blocks.eval()
        self.norm.eval()
        self.proj.train(mode)
        return self

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward_pooled(self, object_cloud: torch.Tensor) -> torch.Tensor:
        """Backbone forward up to (but not including) the proj adapter.

        Returns the (B, 2*embed_dim) mean+max pooled feature. When the whole
        backbone is frozen, this is a pure deterministic function of
        object_cloud — FPS uses a fixed
        farthest-point seed and the transformer is in eval-equivalent frozen
        state — so it can be precomputed offline and cached. The trainable
        proj adapter is applied separately in forward().
        """
        B = object_cloud.shape[0]
        device = object_cloud.device
        if object_cloud.numel() == 0 or object_cloud.shape[1] == 0:
            return torch.zeros((B, 2 * self.embed_dim), device=device, dtype=object_cloud.dtype)
        groups, centroids = self._group_normalized(object_cloud)   # (B,G,M,3),(B,G,3)
        Bf, G, M, _ = groups.shape
        x = self.patch_encoder(groups.reshape(Bf * G, M, 3).unsqueeze(0)).reshape(Bf, G, -1)
        x = x + self.pos_embed(centroids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                           # (B,G,embed)
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)

    def forward(
        self,
        object_cloud: torch.Tensor,
        *,
        pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the object cloud to (B, out_dim).

        When ``pooled`` is provided it is used directly (skipping the frozen
        backbone) — this is the cached fast path. Otherwise the backbone runs
        on ``object_cloud``. The proj adapter is always applied here so it
        stays trainable in both paths.
        """
        if pooled is not None:
            return self.from_pooled(pooled)
        B = object_cloud.shape[0]
        device = object_cloud.device
        if object_cloud.numel() == 0 or object_cloud.shape[1] == 0:
            return torch.zeros((B, self._out_dim), device=device, dtype=object_cloud.dtype)
        return self.proj(self.forward_pooled(object_cloud))

    def from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


    def _group_normalized(self, cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit-sphere normalize per object, then FPS+kNN group (pure torch)."""
        B, P, _ = cloud.shape
        # mask zero-pad rows; normalize by farthest real point from centroid
        mask = cloud.abs().sum(-1) > 0                              # (B,P)
        cen = (cloud * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        pts = cloud - cen.unsqueeze(1)
        scale = (pts.norm(dim=-1) * mask).amax(dim=1, keepdim=True).clamp(min=1e-6)
        pts = pts / scale.unsqueeze(-1)
        idx = _fps_batched(pts, self.num_group)                    # (B,G)
        centroids = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        d2 = torch.cdist(centroids, pts)                           # (B,G,P)
        k = min(self.group_size, P)
        knn = d2.topk(k, dim=-1, largest=False).indices            # (B,G,k)
        nbr = torch.gather(
            pts.unsqueeze(1).expand(-1, self.num_group, -1, -1), 2,
            knn.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        groups = nbr - centroids.unsqueeze(2)                      # local coords
        if k < self.group_size:
            pad = torch.zeros(B, self.num_group, self.group_size - k, 3, device=cloud.device, dtype=cloud.dtype)
            groups = torch.cat([groups, pad], dim=2)
        return groups, centroids


class TierTransformerFusion(nn.Module):
    """Fuse the three published feature types as Transformer tokens,
    add a learned tier embedding, run a small Transformer encoder over the
    set, then mean-pool to (B, H/2).

    Per-tier identity is encoded explicitly via ``tier_embedding`` and
    cross-tier self-attention models interactions between candidate, local,
    and object descriptors.
    """

    def __init__(self, cfg: RescorerConfig, num_tiers: int) -> None:
        super().__init__()
        H = cfg.hidden_dim
        self.num_tiers = int(num_tiers)
        self.tier_embedding = nn.Embedding(self.num_tiers, H)
        layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg.fusion_heads,
            dim_feedforward=H * cfg.fusion_ffn_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.fusion_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(H)
        self.proj = nn.Linear(H, H // 2)
        # Zero-init the tier embedding so an untrained block starts from a
        # symmetric mean of the three feature tokens.
        nn.init.zeros_(self.tier_embedding.weight)

    def forward(self, tier_feats: list[torch.Tensor]) -> torch.Tensor:
        if len(tier_feats) != self.num_tiers:
            raise ValueError(
                f"TierTransformerFusion expected {self.num_tiers} tiers, got {len(tier_feats)}"
            )
        tokens = torch.stack(tier_feats, dim=1)            # (B, T, H)
        ids = torch.arange(self.num_tiers, device=tokens.device)
        tokens = tokens + self.tier_embedding(ids).unsqueeze(0)
        tokens = self.encoder(tokens)                       # (B, T, H)
        pooled = tokens.mean(dim=1)                          # (B, H)
        return self.proj(self.norm(pooled))                 # (B, H/2)


class GraspRescorer(nn.Module):
    """Published candidate + local ShellAttn + Point-MAE rescorer."""

    def __init__(self, config: RescorerConfig) -> None:
        super().__init__()
        if config.object_hidden_dim != config.hidden_dim:
            raise ValueError(
                "the published tier Transformer requires object_hidden_dim == hidden_dim"
            )
        self.config = config
        self.pose_encoder = PoseMLP(
            config.pose_dim, config.hidden_dim,
            dropout=config.dropout,
        )
        self.geometry_encoder = ShellAttnEncoder(
            per_point_dim=config.shell_attn_per_point_dim,
            hidden_dim=config.hidden_dim,
            n_shells=config.shell_attn_n_shells,
            n_heads=config.shell_attn_heads,
            n_attn_layers=config.shell_attn_layers,
            shell_edges_m=tuple(config.shell_edges_m),
            dropout=config.dropout,
        )
        self.film_local = FiLMModulator(config.hidden_dim, config.hidden_dim)
        self.object_encoder = ObjectEncoderPointMAE(config)
        self.film_object = FiLMModulator(config.object_hidden_dim, config.hidden_dim)
        # The published architecture always fuses candidate, local, and
        # object descriptors with a single tier Transformer.
        self.fusion = TierTransformerFusion(config, num_tiers=3)
        head_in = config.hidden_dim // 2
        self.score_head = nn.Linear(head_in, 1)
        self.coll_head = nn.Linear(head_in, 1)
        self.empty_head = nn.Linear(head_in, 1)
        self.obj_head = nn.Linear(head_in, config.num_object_classes)

    def forward(
        self,
        pose_features: torch.Tensor,
        local_cloud: torch.Tensor | None = None,
        *,
        cloud_mask: torch.Tensor | None = None,
        object_cloud: torch.Tensor | None = None,
        object_pooled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        h = self.encode_features(
            pose_features,
            local_cloud,
            cloud_mask=cloud_mask,
            object_cloud=object_cloud,
            object_pooled=object_pooled,
        )
        return {
            "score": self.score_head(h).squeeze(-1),
            "is_collision_logit": self.coll_head(h).squeeze(-1),
            "is_empty_logit": self.empty_head(h).squeeze(-1),
            "obj_id_logit": self.obj_head(h),
        }

    def encode_features(
        self,
        pose_features: torch.Tensor,
        local_cloud: torch.Tensor | None = None,
        *,
        cloud_mask: torch.Tensor | None = None,
        object_cloud: torch.Tensor | None = None,
        object_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_cloud is None:
            raise ValueError("local_cloud is required by the published GraRe model")
        if pose_features.ndim != 2 or pose_features.shape[-1] != self.config.pose_dim:
            raise ValueError(
                f"pose_features must have shape (B, {self.config.pose_dim})"
            )
        if cloud_mask is None:
            cloud_mask = torch.any(local_cloud != 0, dim=-1)

        candidate_feat = self.pose_encoder(pose_features)
        local_feat = self.geometry_encoder(local_cloud, cloud_mask)
        local_feat = self.film_local(local_feat, candidate_feat)

        if object_pooled is not None:
            # The frozen backbone makes cached pooled Point-MAE features exact;
            # the trainable adapter and FiLM remain on the path.
            object_feat = self.object_encoder.from_pooled(object_pooled)
        elif object_cloud is not None:
            object_feat = self.object_encoder(object_cloud)
        else:
            raise ValueError(
                "object_cloud or object_pooled is required by the published GraRe model"
            )
        object_feat = self.film_object(object_feat, candidate_feat)

        return self.fusion([candidate_feat, local_feat, object_feat])
