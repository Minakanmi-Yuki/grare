from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from grare.rescoring.model import GraspRescorer, RescorerConfig
from grare.rescoring.trainer import TrainerConfig, train_model


class _TinyLoader:
    def __init__(self, *, num_batches: int = 2, batch_size: int = 4) -> None:
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            torch.manual_seed(batch_idx + 1)
            yield {
                "pose_features": torch.randn(self.batch_size, 14),
                "local_cloud": torch.randn(self.batch_size, 8, 3),
                "cloud_mask": torch.ones(self.batch_size, 8, dtype=torch.bool),
                "object_pooled": torch.randn(self.batch_size, 768),
                "mu_min": torch.linspace(0.0, 0.8, self.batch_size),
            }


class TrainerSummaryTests(unittest.TestCase):
    def _model(self) -> GraspRescorer:
        return GraspRescorer(
            RescorerConfig(
                pose_dim=14,
                hidden_dim=16,
                object_hidden_dim=16,
                shell_attn_heads=4,
                shell_attn_layers=1,
                shell_attn_per_point_dim=3,
            )
        )

    def _config(self) -> TrainerConfig:
        return TrainerConfig(
            batch_size=4,
            max_epochs=1,
            min_epochs=0,
            early_stop_patience=20,
            device="cpu",
            amp="off",
        )

    def test_train_model_writes_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_model(
                self._model(),
                _TinyLoader(num_batches=2),
                None,
                self._config(),
                tmpdir,
                summary_extra={"experiment": "unit"},
            )

            summary_path = Path(result["summary_path"])
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["num_epochs"], 1)
            self.assertEqual(summary["num_steps"], 2)
            self.assertEqual(summary["experiment"], "unit")
            self.assertIsNotNone(summary["best_train_loss"])

    def test_interrupted_train_model_writes_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "grare.rescoring.trainer.run_epoch",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train_model(
                        self._model(),
                        _TinyLoader(num_batches=2),
                        None,
                        self._config(),
                        tmpdir,
                    )

            summary_path = Path(tmpdir) / "summary.json"
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["stop_reason"], "keyboard_interrupt")
            self.assertEqual(summary["num_epochs"], 0)


if __name__ == "__main__":
    unittest.main()
