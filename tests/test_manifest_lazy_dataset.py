from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from grare.relabeling.dataset_builder import (
    ArchiveRelabeledCandidateDataset,
    PackedRelabeledCandidateDataset,
    RelabeledCandidateDataset,
    collate_archive_batches,
    collate_rescoring_batch,
)
from grare.relabeling.archive_io import save_npz_archive
from grare.relabeling.manifest import (
    build_archive_manifest,
    default_manifest_path,
    load_manifest_records,
    manifest_records_by_archive_path,
)
from grare.cli.train import ArchiveIndexBatchSampler


class ManifestLazyDatasetTests(unittest.TestCase):
    def test_manifest_lazy_dataset_matches_eager_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "train_top8"
            self._write_archive(root / "scene_0000" / "realsense" / "000000.npz", scene_id=0, frame_id=0)
            self._write_archive(root / "scene_0000" / "realsense" / "000001.npz", scene_id=0, frame_id=1)
            build_archive_manifest(root, num_workers=1)

            archives = sorted(root.glob("**/*.npz"))
            records_by_path = manifest_records_by_archive_path(
                load_manifest_records(default_manifest_path(root)),
                root=root,
            )
            records = [records_by_path[path.resolve()] for path in archives]
            eager = RelabeledCandidateDataset(archives, success_mu_thresh=0.4)
            lazy = ArchiveRelabeledCandidateDataset(archives, success_mu_thresh=0.4, manifest_records=records)

            self.assertEqual(len(eager), lazy.__len_samples__())
            lazy_batch = next(iter(DataLoader(
                lazy,
                batch_sampler=ArchiveIndexBatchSampler(
                    lazy.archive_counts,
                    batch_size=6,
                    seed=7,
                    shuffle=False,
                ),
                collate_fn=collate_archive_batches,
                num_workers=0,
            )))
            eager_batch = next(iter(DataLoader(
                eager,
                batch_size=6,
                shuffle=False,
                collate_fn=collate_rescoring_batch,
                num_workers=0,
            )))
            for key in (
                "pose_features",
                "base_score",
                "local_cloud",
                "cloud_mask",
                "mu_min",
                "success_label",
                "is_collision",
                "is_empty",
                "archive_index",
                "local_index",
            ):
                self.assertEqual(tuple(lazy_batch[key].shape), tuple(eager_batch[key].shape), key)
                if lazy_batch[key].dtype == torch.bool:
                    self.assertTrue(torch.equal(lazy_batch[key], eager_batch[key]), key)
                else:
                    self.assertTrue(torch.allclose(lazy_batch[key], eager_batch[key], equal_nan=True), key)

    def test_training_dataset_reads_stored_npz_and_compressed_npz_equally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            compressed = root / "compressed.npz"
            stored = root / "stored.npz"
            payload = self._sample_payload(scene_id=0, frame_id=0)
            np.savez_compressed(compressed, **payload)
            save_npz_archive(stored, archive_format="stored", **payload)

            ds_compressed = RelabeledCandidateDataset([compressed], success_mu_thresh=0.4)
            ds_stored = RelabeledCandidateDataset([stored], success_mu_thresh=0.4)

            self.assertEqual(len(ds_compressed), len(ds_stored))
            self.assertTrue(torch.allclose(ds_compressed.pose_features, ds_stored.pose_features))
            self.assertTrue(torch.allclose(ds_compressed.base_score, ds_stored.base_score))
            self.assertTrue(torch.allclose(ds_compressed.local_cloud, ds_stored.local_cloud))
            self.assertTrue(torch.equal(ds_compressed.cloud_mask, ds_stored.cloud_mask))
            self.assertTrue(torch.allclose(ds_compressed.mu_min, ds_stored.mu_min))
            self.assertTrue(torch.allclose(ds_compressed.success_label, ds_stored.success_label))
            self.assertTrue(torch.equal(ds_compressed.is_collision, ds_stored.is_collision))
            self.assertTrue(torch.equal(ds_compressed.is_empty, ds_stored.is_empty))

    def test_lazy_dataset_reads_object_pooled_sidecar_without_object_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "train_all"
            pooled_root = Path(tmpdir) / "object_pooled" / "train_all"
            archive_path = root / "scene_0000" / "realsense" / "000000.npz"
            pooled_path = pooled_root / archive_path.relative_to(root)
            self._write_archive(archive_path, scene_id=0, frame_id=0)
            pooled_path.parent.mkdir(parents=True, exist_ok=True)
            object_pooled = np.arange(3 * 8, dtype=np.float16).reshape(3, 8)
            np.savez_compressed(pooled_path, object_pooled=object_pooled)

            ds = ArchiveRelabeledCandidateDataset(
                [archive_path],
                success_mu_thresh=0.4,
                object_pooled_roots=[pooled_root],
                input_roots=[root],
                require_object_pooled=True,
            )
            item = ds[0]
            self.assertIn("object_pooled", item)
            self.assertNotIn("object_cloud", item)
            self.assertTrue(torch.allclose(item["object_pooled"].float(), torch.from_numpy(object_pooled).float()))

            batch = next(iter(DataLoader(
                ds,
                batch_sampler=ArchiveIndexBatchSampler(
                    ds.archive_counts,
                    batch_size=6,
                    seed=7,
                    shuffle=False,
                ),
                collate_fn=collate_archive_batches,
                num_workers=0,
            )))
            self.assertIn("object_pooled", batch)
            self.assertNotIn("object_cloud", batch)
            self.assertEqual(tuple(batch["object_pooled"].shape), (3, 8))

    def test_packed_dataset_matches_eager_and_uses_object_pooled_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "train_all"
            pooled_root = Path(tmpdir) / "object_pooled" / "train_all"
            packed_root = Path(tmpdir) / "packed" / "train_all"
            archive0 = root / "scene_0000" / "realsense" / "000000.npz"
            archive1 = root / "scene_0001" / "realsense" / "000001.npz"
            self._write_archive(archive0, scene_id=0, frame_id=0)
            self._write_archive(archive1, scene_id=1, frame_id=1)
            for archive_path in (archive0, archive1):
                pooled_path = pooled_root / archive_path.relative_to(root)
                pooled_path.parent.mkdir(parents=True, exist_ok=True)
                base = 100 if archive_path == archive1 else 0
                object_pooled = (base + np.arange(3 * 8, dtype=np.float16)).reshape(3, 8)
                np.savez_compressed(pooled_path, object_pooled=object_pooled)
            build_archive_manifest(root, num_workers=1)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grare.cli.pack",
                    "--input-root",
                    str(root),
                    "--output-root",
                    str(packed_root),
                    "--object-pooled-root",
                    str(pooled_root),
                    "--require-object-pooled",
                    "--archive-manifest",
                    "auto",
                    "--require-archive-manifest",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )

            archives = sorted(root.glob("**/*.npz"))
            records_by_path = manifest_records_by_archive_path(
                load_manifest_records(default_manifest_path(root)),
                root=root,
            )
            records = [records_by_path[path.resolve()] for path in archives]
            eager = RelabeledCandidateDataset(
                archives,
                success_mu_thresh=0.4,
                object_pooled_roots=[pooled_root],
                input_roots=[root],
                require_object_pooled=True,
            )
            packed = PackedRelabeledCandidateDataset(
                archives,
                packed_root=packed_root,
                input_roots=[root],
                manifest_records=records,
                require_object_pooled=True,
            )

            self.assertEqual(len(eager), len(packed))

            packed_batch = next(iter(DataLoader(
                packed,
                batch_size=6,
                shuffle=False,
                collate_fn=collate_rescoring_batch,
                num_workers=0,
            )))
            eager_batch = next(iter(DataLoader(
                eager,
                batch_size=6,
                shuffle=False,
                collate_fn=collate_rescoring_batch,
                num_workers=0,
            )))
            for key in (
                "pose_features",
                "base_score",
                "local_cloud",
                "cloud_mask",
                "mu_min",
                "success_label",
                "is_collision",
                "is_empty",
                "object_assignments",
                "archive_index",
                "local_index",
                "object_pooled",
            ):
                self.assertEqual(tuple(packed_batch[key].shape), tuple(eager_batch[key].shape), key)
                if packed_batch[key].dtype == torch.bool:
                    self.assertTrue(torch.equal(packed_batch[key], eager_batch[key]), key)
                else:
                    self.assertTrue(torch.allclose(packed_batch[key], eager_batch[key], equal_nan=True), key)

    @staticmethod
    def _sample_payload(*, scene_id: int, frame_id: int) -> dict[str, np.ndarray]:
        n = 3
        grasp_poses = np.tile(np.eye(4, dtype=np.float32).reshape(1, 4, 4), (n, 1, 1))
        grasp_poses[:, :3, 3] = np.arange(n, dtype=np.float32).reshape(n, 1)
        return {
            "grasp_group_array": np.zeros((n, 17), dtype=np.float32),
            "base_scores": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            "grasp_widths": np.full((n,), 0.05, dtype=np.float32),
            "grasp_poses": grasp_poses,
            "local_cloud": np.arange(n * 4 * 3, dtype=np.float32).reshape(n, 4, 3),
            "cloud_mask": np.ones((n, 4), dtype=bool),
            "mu_min": np.asarray([0.2, 0.5, np.inf], dtype=np.float32),
            "is_collision": np.asarray([False, True, False]),
            "is_empty": np.asarray([False, False, False]),
            "object_cloud": np.zeros((n, 0, 3), dtype=np.float32),
            "object_assignments": np.full((n,), -1, dtype=np.int32),
            "meta_json": np.array(
                json.dumps(
                    {
                        "benchmark": "graspnet",
                        "detector": "economicgrasp",
                        "split": "train",
                        "camera": "realsense",
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                    }
                ),
                dtype=object,
            ),
        }

    @staticmethod
    def _write_archive(path: Path, *, scene_id: int, frame_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **ManifestLazyDatasetTests._sample_payload(scene_id=scene_id, frame_id=frame_id))


if __name__ == "__main__":
    unittest.main()
