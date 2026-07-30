"""Tests for the frozen-Point-MAE object-pooled cache fast path.

Verifies that precomputing the backbone's pooled output and feeding it via
``object_pooled`` produces numerically identical results to running the full
backbone — the correctness guarantee that makes the offline cache safe.
"""
from __future__ import annotations

import unittest

import torch

from grare.rescoring.model import (
    GraspRescorer,
    RescorerConfig,
    ObjectEncoderPointMAE,
)


def _cfg(**kw) -> RescorerConfig:
    base = dict(
        object_cloud_points=128,
        object_pmae_num_group=8,
        object_pmae_group_size=8,
        object_hidden_dim=64,
        hidden_dim=64,
        object_pmae_ckpt="",  # random init is fine; determinism is what we test
    )
    base.update(kw)
    return RescorerConfig(**base)


class TestObjectPooledEncoder(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.enc = ObjectEncoderPointMAE(_cfg()).eval()
        self.oc = torch.randn(5, 128, 3)

    def test_forward_pooled_dim(self) -> None:
        pooled = self.enc.forward_pooled(self.oc)
        self.assertEqual(pooled.shape, (5, 2 * self.enc.embed_dim))

    def test_proj_of_pooled_equals_forward(self) -> None:
        with torch.no_grad():
            direct = self.enc(self.oc)                      # backbone + proj
            pooled = self.enc.forward_pooled(self.oc)        # backbone only
            cached = self.enc(self.oc, pooled=pooled)        # proj on cached pooled
        torch.testing.assert_close(direct, cached, rtol=1e-5, atol=1e-6)

    def test_pooled_is_deterministic(self) -> None:
        with torch.no_grad():
            a = self.enc.forward_pooled(self.oc)
            b = self.enc.forward_pooled(self.oc.clone())
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_point_mae_backbone_is_frozen(self) -> None:
        self.enc.train()
        self.assertFalse(self.enc.patch_encoder.training)
        self.assertTrue(self.enc.proj.training)
        backbone = (
            self.enc.patch_encoder,
            self.enc.pos_embed,
            self.enc.blocks,
            self.enc.norm,
        )
        self.assertTrue(
            all(not parameter.requires_grad for module in backbone for parameter in module.parameters())
        )


class TestModelForwardCachedPath(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.model = GraspRescorer(_cfg()).eval()
        self.pose = torch.randn(4, 14)
        self.local = torch.randn(4, 64, 3)
        self.oc = torch.randn(4, 128, 3)

    def test_cached_path_matches_backbone_path(self) -> None:
        enc = self.model.object_encoder
        with torch.no_grad():
            pooled = enc.forward_pooled(self.oc)
            out_backbone = self.model(self.pose, self.local, object_cloud=self.oc)
            out_cached = self.model(
                self.pose, self.local, object_cloud=self.oc, object_pooled=pooled
            )
        torch.testing.assert_close(
            out_backbone["score"], out_cached["score"], rtol=1e-5, atol=1e-6
        )

    def test_cached_path_ignores_object_cloud_when_pooled_present(self) -> None:
        enc = self.model.object_encoder
        with torch.no_grad():
            pooled = enc.forward_pooled(self.oc)
            # Pass garbage object_cloud; pooled must dominate.
            garbage = torch.randn(4, 128, 3) * 100
            out_pooled_real = self.model(
                self.pose, self.local, object_cloud=garbage, object_pooled=pooled
            )
            out_clean = self.model(
                self.pose, self.local, object_cloud=self.oc, object_pooled=pooled
            )
        torch.testing.assert_close(
            out_pooled_real["score"], out_clean["score"], rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
