"""Smoke test for training_core.run_epoch.

Builds a tiny synthetic batch, runs one training and one eval epoch on CPU,
and verifies the metrics dict is populated. The point is to exercise the
on-device accumulators that replaced the per-batch .item() syncs.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from grare.rescoring.model import GraspRescorer, RescorerConfig
from grare.rescoring.training_core import (
    MultiTaskCriteria,
    _build_quality_target,
    run_epoch,
)


class _SyntheticBatchLoader:
    def __init__(self, num_batches: int, batch_size: int, num_points: int, pose_dim: int) -> None:
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_points = num_points
        self.pose_dim = pose_dim
        torch.manual_seed(0)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            torch.manual_seed(i + 1)
            yield {
                "pose_features": torch.randn(self.batch_size, self.pose_dim),
                "local_cloud": torch.randn(self.batch_size, self.num_points, 3),
                "cloud_mask": torch.ones(self.batch_size, self.num_points, dtype=torch.bool),
                "mu_min": torch.rand(self.batch_size) * 1.5,
                "is_collision": torch.zeros(self.batch_size, dtype=torch.float32),
                "is_empty": torch.zeros(self.batch_size, dtype=torch.float32),
                "object_assignments": torch.full((self.batch_size,), -1, dtype=torch.long),
                "object_pooled": torch.randn(self.batch_size, 768),
            }


class RunEpochSmokeTests(unittest.TestCase):
    def _make_setup(self) -> tuple:
        config = RescorerConfig(
            pose_dim=14,
            hidden_dim=32,
            object_hidden_dim=32,
            shell_attn_heads=2,
            object_pmae_num_group=4,
            object_pmae_group_size=8,
        )
        model = GraspRescorer(config)
        device = torch.device("cpu")
        model.to(device)
        criteria = MultiTaskCriteria(score=nn.MSELoss(reduction="none"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = _SyntheticBatchLoader(num_batches=4, batch_size=8, num_points=64, pose_dim=14)
        return model, loader, criteria, optimizer, device

    def _common_kwargs(self):
        return dict(
            grad_clip_norm=None,
            success_mu_thresh=0.4,
            score_collapse_std_threshold=1e-4,
        )

    def test_training_epoch_returns_finite_metrics(self) -> None:
        model, loader, criteria, optimizer, device = self._make_setup()
        metrics = run_epoch(
            model=model,
            loader=loader,
            criteria=criteria,
            optimizer=optimizer,
            device=device,
            amp_enabled=False,
            amp_dtype=None,
            scaler=None,
            training=True,
            **self._common_kwargs(),
        )
        for key in (
            "loss", "primary_loss", "epoch_sec", "samples_per_sec",
            "pred_mean", "pred_std", "pred_target_corr",
            "score_collapse_flag",
        ):
            self.assertIn(key, metrics)
        for key in ("loss", "primary_loss", "pred_mean", "pred_std"):
            value = metrics[key]
            self.assertTrue(
                value is None or isinstance(value, (int, float)),
                f"{key} = {value!r} is not a python scalar",
            )
        self.assertTrue(metrics["loss"] == metrics["loss"], "loss is NaN")

    def test_eval_epoch_no_optimizer(self) -> None:
        model, loader, criteria, _opt, device = self._make_setup()
        metrics = run_epoch(
            model=model,
            loader=loader,
            criteria=criteria,
            optimizer=None,
            device=device,
            amp_enabled=False,
            amp_dtype=None,
            scaler=None,
            training=False,
            **self._common_kwargs(),
        )
        self.assertGreater(metrics["epoch_sec"], 0.0)

    def test_quality_target_matches_signed_margin(self) -> None:
        mu_min = torch.tensor([0.0, 0.2, 0.4, 0.8, float("inf")])
        target = _build_quality_target(mu_min, success_mu_thresh=0.4)
        expected = torch.tensor([0.4, 0.2, 0.0, -0.4, -1.6])
        self.assertTrue(torch.allclose(target, expected))

if __name__ == "__main__":
    unittest.main()
