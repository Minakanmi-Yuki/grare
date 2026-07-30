from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from grare.rescoring.inference import rerank_archive


class _DummyModel(torch.nn.Module):
    def __init__(self, outputs: list[float]) -> None:
        super().__init__()
        self.register_buffer("_outputs", torch.tensor(outputs, dtype=torch.float32))

    def forward(
        self,
        pose_features: torch.Tensor,
        local_cloud: torch.Tensor | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        del pose_features, local_cloud
        return {"score": self._outputs.clone()}


class _ObjectPooledModel(torch.nn.Module):
    def forward(
        self,
        pose_features: torch.Tensor,
        local_cloud: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del pose_features, local_cloud
        object_pooled = kwargs.get("object_pooled")
        if object_pooled is None:
            raise AssertionError("object_pooled was not passed to the model")
        return {"score": object_pooled[:, 0].float()}


def _zscore(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return (array - np.mean(array)) / np.std(array)


class RerankProtocolTests(unittest.TestCase):
    def _write_archive(self, root: Path, *, base_scores: list[float]) -> Path:
        archive_path = root / "candidate.npz"
        num_grasps = len(base_scores)
        grasp_group_array = np.zeros((num_grasps, 17), dtype=np.float32)
        grasp_group_array[:, 0] = np.asarray(base_scores, dtype=np.float32)
        grasp_group_array[:, 1] = 0.05
        grasp_group_array[:, 4:13] = np.tile(
            np.eye(3, dtype=np.float32).reshape(1, 9),
            (num_grasps, 1),
        )
        grasp_group_array[:, 13:16] = np.arange(
            num_grasps,
            dtype=np.float32,
        ).reshape(-1, 1)
        grasp_poses = np.tile(
            np.eye(4, dtype=np.float32).reshape(1, 4, 4),
            (num_grasps, 1, 1),
        )
        grasp_poses[:, :3, 3] = np.arange(
            num_grasps,
            dtype=np.float32,
        ).reshape(-1, 1)
        np.savez_compressed(
            archive_path,
            grasp_group_array=grasp_group_array,
            base_scores=np.asarray(base_scores, dtype=np.float32),
            grasp_widths=np.full((num_grasps,), 0.05, dtype=np.float32),
            grasp_poses=grasp_poses,
            local_cloud=np.zeros((num_grasps, 4, 3), dtype=np.float32),
            meta_json=np.array(
                json.dumps(
                    {
                        "benchmark": "graspnet",
                        "scene_id": 1,
                        "frame_id": 0,
                        "camera": "kinect",
                    }
                ),
                dtype=object,
            ),
        )
        return archive_path

    def test_lambda_one_uses_zscore_and_keeps_every_candidate(self) -> None:
        base_scores = [0.9, 0.8, 0.7]
        model_scores = [0.1, 0.5, 0.2]
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = self._write_archive(Path(tmpdir), base_scores=base_scores)
            result = rerank_archive(
                archive_path,
                _DummyModel(model_scores),
                device="cpu",
                rescoring_score_weight=1.0,
            )

        expected_order = np.argsort(-_zscore(model_scores), kind="stable")
        np.testing.assert_array_equal(result["original_indices"], expected_order)
        np.testing.assert_allclose(
            result["exported_scores"],
            _zscore(model_scores)[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            result["rescoring_scores"],
            np.asarray(model_scores, dtype=np.float32)[expected_order],
        )
        self.assertEqual(result["candidate_count"], len(base_scores))
        self.assertEqual(result["grasp_group_array"].shape[0], len(base_scores))
        self.assertEqual(result["score_normalization"], "zscore")
        self.assertEqual(result["lambda"], 1.0)
        self.assertTrue(result["score_column_overwritten"])

    def test_lambda_fuses_normalized_detector_and_rescorer_scores(self) -> None:
        base_scores = np.asarray([0.9, 0.8, 0.1], dtype=np.float32)
        model_scores = np.asarray([0.0, 1.0, 0.2], dtype=np.float32)
        weight = 0.7
        expected = (1.0 - weight) * _zscore(base_scores) + weight * _zscore(model_scores)
        expected_order = np.argsort(-expected, kind="stable")

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = self._write_archive(
                Path(tmpdir),
                base_scores=base_scores.tolist(),
            )
            result = rerank_archive(
                archive_path,
                _DummyModel(model_scores.tolist()),
                device="cpu",
                rescoring_score_weight=weight,
            )

        np.testing.assert_array_equal(result["original_indices"], expected_order)
        np.testing.assert_allclose(
            result["exported_scores"],
            expected[expected_order],
            rtol=1e-6,
        )
        np.testing.assert_allclose(result["base_scores"], base_scores[expected_order])
        self.assertEqual(result["candidate_count"], len(base_scores))
        self.assertEqual(result["final_top1_idx"], int(expected_order[0]))
        self.assertTrue(result["top1_changed_vs_base"])

    def test_lambda_zero_preserves_detector_order(self) -> None:
        base_scores = np.asarray([0.2, 0.9, 0.4, 0.1], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = self._write_archive(
                Path(tmpdir),
                base_scores=base_scores.tolist(),
            )
            result = rerank_archive(
                archive_path,
                _DummyModel([10.0, -2.0, 3.0, 8.0]),
                device="cpu",
                rescoring_score_weight=0.0,
            )

        expected_order = np.argsort(-base_scores, kind="stable")
        np.testing.assert_array_equal(result["original_indices"], expected_order)
        np.testing.assert_allclose(
            result["exported_scores"],
            _zscore(base_scores)[expected_order],
            rtol=1e-6,
        )

    def test_non_zscore_normalization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = self._write_archive(
                Path(tmpdir),
                base_scores=[0.9, 0.8, 0.1],
            )
            with self.assertRaisesRegex(ValueError, "published GraRe protocol"):
                rerank_archive(
                    archive_path,
                    _DummyModel([0.0, 1.0, 0.2]),
                    device="cpu",
                    score_normalization="none",
                )

    def test_object_pooled_sidecar_is_used_for_rerank(self) -> None:
        pooled_scores = np.asarray([0.4, 0.9, 0.1], dtype=np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_root = root / "local"
            local_root.mkdir(parents=True, exist_ok=True)
            archive_path = self._write_archive(
                local_root,
                base_scores=[0.1, 0.2, 0.3],
            )
            pooled_path = root / "pooled" / "candidate.npz"
            pooled_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pooled_path,
                object_pooled=np.column_stack(
                    [pooled_scores, np.zeros_like(pooled_scores)]
                ),
            )
            result = rerank_archive(
                archive_path,
                _ObjectPooledModel(),
                device="cpu",
                object_pooled_archive_path=pooled_path,
                require_object_pooled=True,
                rescoring_score_weight=1.0,
            )

        expected_scores = pooled_scores.astype(np.float32)
        expected_order = np.argsort(-expected_scores, kind="stable")
        np.testing.assert_array_equal(result["original_indices"], expected_order)
        np.testing.assert_allclose(
            result["rescoring_scores"],
            expected_scores[expected_order],
        )
        np.testing.assert_allclose(
            result["exported_scores"],
            _zscore(expected_scores)[expected_order],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
