from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from grare.cli import prepare
from grare.relabeling.scene_labeling import (
    BatchAnalyticRelabeler,
    SceneLabelingConfig,
    _BaseLabelBackend,
)


def _prediction_array(count: int) -> np.ndarray:
    prediction = np.zeros((count, 17), dtype=np.float32)
    prediction[:, 0] = np.linspace(0.1, 0.9, count, dtype=np.float32)[::-1]
    prediction[:, 1] = np.arange(count, dtype=np.float32)
    return prediction


def _relabeler(dataset_root: Path, *, split: str = "test") -> BatchAnalyticRelabeler:
    return BatchAnalyticRelabeler(
        SceneLabelingConfig(
            detector="baseline",
            benchmark="graspnet",
            split=split,
            camera="kinect",
            dataset_root=str(dataset_root),
        )
    )


class CandidatePreservationTests(unittest.TestCase):
    def test_scene_assets_cache_keeps_only_the_current_scene(self) -> None:
        config = SceneLabelingConfig(
            detector="baseline",
            benchmark="graspnet",
            split="test",
            camera="kinect",
            dataset_root="unused",
        )
        backend = _BaseLabelBackend(config)
        backend._scene_cache = {
            1: ([], [], ()),
            2: ([], [], ()),
        }
        backend._scene_cache_order = [1, 2]
        backend._scene_object_cache = {
            1: ([], ()),
            2: ([], ()),
        }
        backend._scene_object_cache_order = [1, 2]

        backend._prune_scene_cache()

        self.assertEqual(list(backend._scene_cache), [2])
        self.assertEqual(backend._scene_cache_order, [2])
        self.assertEqual(list(backend._scene_object_cache), [2])
        self.assertEqual(backend._scene_object_cache_order, [2])

    def test_prediction_file_passes_every_detector_candidate_to_labeler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prediction_path = root / "scene_0001" / "kinect" / "0000.npy"
            prediction_path.parent.mkdir(parents=True)
            expected = _prediction_array(7)
            np.save(prediction_path, expected)
            relabeler = _relabeler(root)
            captured: dict[str, object] = {}

            def capture_prediction(prediction, save_path, _label_fn, **kwargs):
                captured["array"] = prediction.grasp_group_array.copy()
                captured["meta"] = kwargs["extra_meta"]
                return Path(save_path)

            relabeler.labeler.label_prediction = Mock(side_effect=capture_prediction)
            save_path = root / "output" / "candidate.npz"
            result = relabeler.relabel_prediction_file(prediction_path, save_path)

        self.assertEqual(result, save_path)
        np.testing.assert_array_equal(captured["array"], expected)
        self.assertEqual(
            captured["meta"],
            {"source_prediction_file": str(prediction_path)},
        )

    def test_tree_limit_caps_frames_without_changing_worker_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "input"
            for frame_id in range(3):
                frame_path = input_root / "scene_0001" / "kinect" / f"{frame_id:04d}.npy"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(frame_path, _prediction_array(4 + frame_id))

            relabeler = _relabeler(root)
            relabeler._backend_impl = Mock()
            relabeler._process_archive = Mock()
            with contextlib.redirect_stdout(io.StringIO()):
                summary = relabeler.relabel_dump_tree(
                    input_root,
                    root / "output",
                    limit=2,
                    skip_existing=False,
                    num_workers=1,
                )

        self.assertEqual(summary["num_archives"], 2)
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(relabeler._process_archive.call_count, 2)
        processed_names = [
            call.args[0].name for call in relabeler._process_archive.call_args_list
        ]
        self.assertEqual(processed_names, ["0000.npy", "0001.npy"])

    def test_prepare_accepts_test_features_and_forwards_frame_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            argv = [
                "grare-prepare",
                "--input-root",
                str(root / "input"),
                "--output-root",
                str(root / "output"),
                "--detector",
                "baseline",
                "--dataset-root",
                str(root / "dataset"),
                "--camera",
                "kinect",
                "--split",
                "test",
                "--input-format",
                "detector-dump",
                "--limit",
                "2",
                "--no-manifest",
            ]
            relabeler = Mock()
            relabeler.relabel_dump_tree.return_value = {
                "num_archives": 0,
                "processed": 0,
                "skipped": 0,
            }
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "grare.relabeling.scene_labeling.BatchAnalyticRelabeler",
                    return_value=relabeler,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = prepare.main()

        self.assertEqual(status, 0)
        relabeler.relabel_dump_tree.assert_called_once()
        self.assertEqual(relabeler.relabel_dump_tree.call_args.kwargs["limit"], 2)


if __name__ == "__main__":
    unittest.main()
