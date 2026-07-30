"""Archive schema: object_cloud + object_assignments load correctly."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grare.relabeling.dataset_builder import _load_archive_payload


def _write_archive(path: Path, *, n: int, P: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grasp_group_array=np.zeros((n, 17), dtype=np.float32),
        base_scores=np.linspace(0, 1, n).astype(np.float32),
        grasp_widths=np.zeros((n,), dtype=np.float32),
        grasp_poses=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        local_cloud=np.zeros((n, P, 3), dtype=np.float32),
        cloud_mask=np.ones((n, P), dtype=np.bool_),
        mu_min=np.full((n,), 0.3, dtype=np.float32),
        is_collision=np.zeros((n,), dtype=np.bool_),
        is_empty=np.zeros((n,), dtype=np.bool_),
        object_cloud=np.full((n, 64, 3), 0.5, dtype=np.float32),
        object_assignments=np.array([0, 5, -1] + [42] * (n - 3), dtype=np.int32)[:n],
    )


class ArchiveSchemaTests(unittest.TestCase):
    def test_archive_loads_with_object_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "archive.npz"
            _write_archive(p, n=5)
            payload = _load_archive_payload(p, success_mu_thresh=0.4)
        self.assertIn("object_cloud", payload)
        self.assertIn("object_assignments", payload)
        self.assertEqual(payload["object_cloud"].shape, (5, 64, 3))
        self.assertEqual(payload["object_cloud"].dtype, np.float32)
        self.assertEqual(payload["object_assignments"].shape, (5,))
        self.assertEqual(payload["object_assignments"].dtype, np.int64)

    def test_obj_id_values_are_valid_class_ids_or_neg_one(self):
        # The obj_id head expects values in [-1, num_object_classes); the
        # frame-local-to-global mapping in scene_labeling must respect this.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "archive.npz"
            _write_archive(p, n=5)
            payload = _load_archive_payload(p, success_mu_thresh=0.4)
        oa = payload["object_assignments"]
        self.assertTrue(np.all((oa == -1) | ((oa >= 0) & (oa < 88))))


if __name__ == "__main__":
    unittest.main()
