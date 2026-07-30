from __future__ import annotations

import torch
from torch import nn


class PointMAEPatchEncoder(nn.Module):
    """Point-MAE per-patch shared MLP used by the object tier."""

    def __init__(self, encoder_channel: int = 384) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, n, 3)
        f = self.first_conv(pg.transpose(2, 1))
        f_global = torch.max(f, dim=2, keepdim=True)[0]
        f = torch.cat([f_global.expand(-1, -1, n), f], dim=1)
        f = self.second_conv(f)
        f_global = torch.max(f, dim=2, keepdim=False)[0]
        return f_global.reshape(bs, g, -1)


class PointMAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz, num_tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class PointMAEMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PointMAEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PointMAEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = PointMAEMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
