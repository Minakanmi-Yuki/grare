"""Tests for deterministic DataLoader runtime configuration."""
from __future__ import annotations

import unittest

import torch

from grare.rescoring import data_prep


class ConfigureLoaderRuntimeTests(unittest.TestCase):
    def _call(self, requested: int, **overrides) -> data_prep.LoaderRuntimeConfig:
        kwargs = dict(
            batch_size=8,
            requested_num_workers=requested,
            prefetch_factor=2,
            disable_pin_memory=False,
            disable_persistent_workers=False,
            train_device=torch.device("cpu"),
            dataset_device=torch.device("cpu"),
        )
        kwargs.update(overrides)
        return data_prep.configure_loader_runtime(**kwargs)

    def test_explicit_zero_disables_workers(self) -> None:
        config = self._call(requested=0)
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.persistent_workers)
        self.assertIsNone(config.prefetch_factor)

    def test_explicit_positive_pins_workers(self) -> None:
        config = self._call(requested=12)
        self.assertEqual(config.num_workers, 12)
        self.assertTrue(config.persistent_workers)
        self.assertEqual(config.prefetch_factor, 2)

    def test_negative_worker_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._call(requested=-1)

    def test_dataset_on_gpu_zeros_workers(self) -> None:
        config = self._call(
            requested=4,
            train_device=torch.device("cuda"),
            dataset_device=torch.device("cuda"),
        )
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.pin_memory)


if __name__ == "__main__":
    unittest.main()
