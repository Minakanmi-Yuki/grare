"""Run a self-contained CPU smoke test with generated candidate features."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from grare.rescoring.checkpoints import load_model_checkpoint
from grare.rescoring.inference import rerank_archive
from grare.rescoring.model import GraspRescorer, RescorerConfig
from grare.rescoring.training_core import MultiTaskCriteria, run_epoch


class _BatchLoader:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def _synthetic_archive(path: Path, count: int = 8) -> dict[str, torch.Tensor]:
    generator = np.random.default_rng(7)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    translations = generator.normal(0.0, 0.02, (count, 3)).astype(np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    widths = generator.uniform(0.02, 0.08, count).astype(np.float32)
    scores = np.linspace(1.0, 0.1, count, dtype=np.float32)
    local = generator.normal(0.0, 0.015, (count, 32, 3)).astype(np.float32)
    object_pooled = generator.normal(0.0, 0.1, (count, 768)).astype(np.float32)
    grasp_group = np.zeros((count, 17), dtype=np.float32)
    grasp_group[:, 0] = scores
    grasp_group[:, 1] = widths
    grasp_group[:, 4:13] = rotations.reshape(count, 9)
    grasp_group[:, 13:16] = translations
    np.savez_compressed(
        path,
        grasp_group_array=grasp_group,
        base_scores=scores,
        grasp_widths=widths,
        grasp_poses=poses,
        local_cloud=local,
        cloud_mask=np.ones((count, 32), dtype=np.bool_),
        object_pooled=object_pooled,
        mu_min=generator.uniform(0.2, 1.2, count).astype(np.float32),
        is_collision=np.zeros(count, dtype=np.bool_),
        is_empty=np.zeros(count, dtype=np.bool_),
        object_assignments=np.full(count, -1, dtype=np.int64),
    )
    pose_features = np.concatenate(
        [rotations.reshape(count, 9), translations, widths[:, None], scores[:, None]],
        axis=1,
    )
    return {
        "pose_features": torch.from_numpy(pose_features),
        "local_cloud": torch.from_numpy(local),
        "cloud_mask": torch.ones(count, 32, dtype=torch.bool),
        "mu_min": torch.from_numpy(generator.uniform(0.2, 1.2, count).astype(np.float32)),
        "is_collision": torch.zeros(count),
        "is_empty": torch.zeros(count),
        "object_assignments": torch.full((count,), -1, dtype=torch.long),
        "object_pooled": torch.from_numpy(object_pooled),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=None)
    return parser.parse_args()


def _run(work_dir: Path) -> None:
    torch.manual_seed(7)
    archive = work_dir / "frame.npz"
    batch = _synthetic_archive(archive)
    config = RescorerConfig(
        pose_dim=14,
        hidden_dim=32,
        dropout=0.0,
        shell_attn_n_shells=4,
        shell_attn_heads=2,
        shell_attn_layers=1,
        shell_attn_per_point_dim=3,
        object_hidden_dim=32,
        object_pmae_num_group=4,
        object_pmae_group_size=8,
    )
    model = GraspRescorer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    metrics = run_epoch(
        model=model,
        loader=_BatchLoader(batch),
        criteria=MultiTaskCriteria(score=nn.SmoothL1Loss(reduction="none")),
        optimizer=optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=None,
        scaler=None,
        training=True,
        grad_clip_norm=1.0,
        success_mu_thresh=0.4,
        score_collapse_std_threshold=1e-4,
    )
    if not np.isfinite(float(metrics["loss"])):
        raise RuntimeError("synthetic training produced a non-finite loss")
    checkpoint = work_dir / "smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "trainer_config": {"seed": 7},
        },
        checkpoint,
    )
    loaded = load_model_checkpoint(checkpoint, device="cpu")
    result = rerank_archive(
        archive,
        loaded,
        device="cpu",
        rescoring_score_weight=1.0,
        score_normalization="zscore",
    )
    original = np.load(archive)["grasp_group_array"]
    ranked = result["grasp_group_array"]
    if ranked.shape != original.shape:
        raise RuntimeError("re-ranking changed the candidate-set shape")
    if not np.all(np.diff(ranked[:, 0]) < 0):
        raise RuntimeError("exported scores are not strictly descending")
    print(
        f"GraRe smoke test passed: loss={float(metrics['loss']):.6f}, "
        f"candidates={len(ranked)}, checkpoint_reload=ok, rerank=ok"
    )


def main() -> int:
    args = parse_args()
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _run(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="grare-smoke-") as directory:
            _run(Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
