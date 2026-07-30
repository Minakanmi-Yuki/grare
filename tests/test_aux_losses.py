"""Multi-task losses fire and contribute to total."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from grare.rescoring.model import GraspRescorer, RescorerConfig
from grare.rescoring.training_core import MultiTaskCriteria, run_epoch


class AuxLossesTests(unittest.TestCase):
    def _model_and_loader(self, *, B=4, P=8, P_obj=32):
        cfg = RescorerConfig(
            pose_dim=14, hidden_dim=32, dropout=0.0,
            shell_attn_per_point_dim=3,
            shell_attn_n_shells=4, shell_attn_heads=2, shell_attn_layers=1,
            object_cloud_points=P_obj, object_hidden_dim=32,
            object_pmae_num_group=4, object_pmae_group_size=8,
        )
        model = GraspRescorer(cfg)
        torch.manual_seed(7)
        batch = {
            "pose_features": torch.randn(B, 14),
            "local_cloud": torch.randn(B, P, 3),
            "cloud_mask": torch.ones(B, P, dtype=torch.bool),
            "object_pooled": torch.randn(B, 768),
            "object_assignments": torch.randint(0, 88, (B,)),
            "is_collision": torch.randint(0, 2, (B,)).float(),
            "is_empty": torch.randint(0, 2, (B,)).float(),
            "mu_min": torch.rand(B) * 0.6 + 0.1,
        }
        return model, [batch] * 2

    def test_all_three_aux_losses_fire(self):
        model, loader = self._model_and_loader()
        criteria = MultiTaskCriteria(
            score=nn.MSELoss(reduction="none"),
            lambda_collision=0.1, lambda_empty=0.05, lambda_obj_id=0.05,
        )
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        metrics = run_epoch(
            model=model, loader=loader, criteria=criteria, optimizer=opt,
            device=torch.device("cpu"),
            amp_enabled=False, amp_dtype=None, scaler=None,
            training=True, grad_clip_norm=1.0, success_mu_thresh=0.4,
            score_collapse_std_threshold=1e-4,
        )
        for key in ("loss_score", "loss_coll", "loss_empty", "loss_obj"):
            value = metrics[key]
            self.assertIsNotNone(value, f"{key} is None")
            self.assertTrue(torch.isfinite(torch.tensor(value)).item(),
                            f"{key} is not finite: {value}")
        self.assertGreater(metrics["loss"], metrics["loss_score"])

    def test_zero_lambdas_match_score_only(self):
        model, loader = self._model_and_loader()
        criteria = MultiTaskCriteria(
            score=nn.MSELoss(reduction="none"),
            lambda_collision=0.0, lambda_empty=0.0, lambda_obj_id=0.0,
        )
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        metrics = run_epoch(
            model=model, loader=loader, criteria=criteria, optimizer=opt,
            device=torch.device("cpu"),
            amp_enabled=False, amp_dtype=None, scaler=None,
            training=True, grad_clip_norm=None, success_mu_thresh=0.4,
            score_collapse_std_threshold=1e-4,
        )
        self.assertIsNone(metrics["loss_coll"])
        self.assertIsNone(metrics["loss_empty"])
        self.assertIsNone(metrics["loss_obj"])
        self.assertAlmostEqual(metrics["loss"], metrics["loss_score"], places=6)


if __name__ == "__main__":
    unittest.main()
