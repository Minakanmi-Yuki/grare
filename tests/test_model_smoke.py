"""Model smoke: forward 4 heads with right shapes; grads flow."""
from __future__ import annotations

import unittest

import torch

from grare.rescoring.model import GraspRescorer, RescorerConfig


class ModelSmokeTests(unittest.TestCase):
    def _make(self, **overrides):
        defaults = dict(
            pose_dim=14, hidden_dim=32, dropout=0.0,
            shell_attn_per_point_dim=3,
            shell_attn_n_shells=4, shell_attn_heads=2, shell_attn_layers=1,
            object_cloud_points=32, object_hidden_dim=32,
            object_pmae_num_group=4, object_pmae_group_size=8,
        )
        defaults.update(overrides)
        return RescorerConfig(**defaults)

    def _batch(self, B=2, P=8):
        torch.manual_seed(0)
        pose = torch.randn(B, 14)
        lc = torch.randn(B, P, 3)
        mask = torch.ones(B, P, dtype=torch.bool)
        pooled = torch.randn(B, 768)
        return pose, lc, mask, pooled

    def test_full_forward_yields_4_heads(self):
        m = GraspRescorer(self._make())
        m.eval()
        pose, lc, mask, pooled = self._batch()
        out = m(pose, lc, cloud_mask=mask, object_pooled=pooled)
        self.assertEqual(out["score"].shape, (2,))
        self.assertEqual(out["is_collision_logit"].shape, (2,))
        self.assertEqual(out["is_empty_logit"].shape, (2,))
        self.assertEqual(out["obj_id_logit"].shape, (2, 88))
        for v in out.values():
            self.assertTrue(torch.isfinite(v).all())

    def test_score_head_grad_reaches_every_branch(self):
        m = GraspRescorer(self._make())
        pose, lc, mask, pooled = self._batch()
        out = m(pose, lc, cloud_mask=mask, object_pooled=pooled)
        out["score"].sum().backward()
        self.assertIsNotNone(m.pose_encoder.net[0].net[0].weight.grad)
        self.assertIsNotNone(m.geometry_encoder.point_mlp[0].net[0].weight.grad)
        self.assertIsNotNone(m.object_encoder.proj[0].weight.grad)
        self.assertIsNotNone(m.fusion.proj.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in m.object_encoder.blocks.parameters())
        )

    def test_object_context_is_required(self):
        m = GraspRescorer(self._make())
        m.eval()
        pose, lc, mask, _ = self._batch()
        with self.assertRaisesRegex(ValueError, "object_cloud or object_pooled"):
            m(pose, lc, cloud_mask=mask)


if __name__ == "__main__":
    unittest.main()
