from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from grare.rescoring.checkpoints import load_model_checkpoint
from grare.rescoring.model import GraspRescorer, RescorerConfig


def _published_config() -> RescorerConfig:
    return RescorerConfig(
        pose_dim=14,
        hidden_dim=128,
        dropout=0.1,
        shell_attn_n_shells=4,
        shell_attn_heads=4,
        shell_attn_layers=1,
        shell_attn_per_point_dim=3,
        object_cloud_points=512,
        object_hidden_dim=128,
        object_pmae_ckpt="",
        object_pmae_num_group=32,
        object_pmae_group_size=32,
        fusion_layers=1,
        fusion_heads=4,
        fusion_ffn_mult=2,
        num_object_classes=88,
    )


_HEADS = ("score_head", "coll_head", "empty_head", "obj_head")


class CheckpointHeadCompatTest(unittest.TestCase):
    """Published checkpoints stored output heads under an inner ``body`` module."""

    def test_loads_checkpoint_with_body_prefixed_head_keys(self) -> None:
        config = _published_config()
        model = GraspRescorer(config)
        state_dict = model.state_dict()

        legacy_state_dict = {}
        for key, value in state_dict.items():
            for head in _HEADS:
                if key.startswith(head + "."):
                    key = f"{head}.body.{key[len(head) + 1:]}"
                    break
            legacy_state_dict[key] = value

        # The legacy layout must actually differ, otherwise this test is vacuous.
        self.assertTrue(any(".body." in key for key in legacy_state_dict))
        self.assertNotEqual(sorted(legacy_state_dict), sorted(state_dict))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {
                    "model_state_dict": legacy_state_dict,
                    "model_config": config.__dict__,
                    "trainer_config": {},
                },
                path,
            )
            loaded = load_model_checkpoint(path, device="cpu")

        reloaded = loaded.state_dict()
        self.assertEqual(sorted(reloaded), sorted(state_dict))
        for key, value in state_dict.items():
            self.assertTrue(torch.equal(reloaded[key], value), msg=key)

    def test_current_checkpoint_layout_still_loads(self) -> None:
        config = _published_config()
        model = GraspRescorer(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config.__dict__,
                    "trainer_config": {},
                },
                path,
            )
            loaded = load_model_checkpoint(path, device="cpu")
        self.assertEqual(sorted(loaded.state_dict()), sorted(model.state_dict()))

    def test_genuinely_incompatible_checkpoint_still_raises(self) -> None:
        config = _published_config()
        model = GraspRescorer(config)
        state_dict = model.state_dict()
        state_dict["score_head.weight"] = torch.zeros(1, 999)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "model_config": config.__dict__,
                    "trainer_config": {},
                },
                path,
            )
            with self.assertRaises(RuntimeError):
                load_model_checkpoint(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
