from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from grare.cli.demo import (
    _apply_baseline_visualization_steps,
    _build_pose_features,
    _config_name_for,
    _load_graspnet_frame,
    _parse_scene_id,
    _restore_original_candidate_order,
    _score_color_values,
    _split_for_scene,
    rerank_candidates,
)
from grare.relabeling.scene_labeling import (
    OnlineFeatureExtractor,
    SamObjectCloudConfig,
    SceneLabelingConfig,
)


def _grasps() -> np.ndarray:
    grasps = np.zeros((3, 17), dtype=np.float32)
    grasps[:, 0] = [0.2, 0.8, 0.4]
    grasps[:, 1] = 0.06
    grasps[:, 3] = 0.02
    grasps[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
    grasps[:, 13:16] = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [-0.1, 0.0, 1.0]]
    return grasps


def test_demo_pose_features_match_published_candidate_layout() -> None:
    grasps = _grasps()
    features = _build_pose_features(grasps)
    assert features.shape == (3, 14)
    assert np.array_equal(features[:, :9], grasps[:, 4:13])
    assert np.array_equal(features[:, 9:12], grasps[:, 13:16])
    assert np.array_equal(features[:, 12], grasps[:, 1])
    assert np.array_equal(features[:, 13], grasps[:, 0])


def test_demo_selection_resolves_the_supported_configs_and_splits() -> None:
    assert _config_name_for("graspnet_baseline", "realsense") == "gn_realsense"
    assert _config_name_for("graspnet_baseline", "kinect") == "gn_kinect"
    assert _config_name_for("scale_balanced_grasp", "realsense") == "sbg_realsense"
    assert _config_name_for("economicgrasp", "kinect") == "eg_kinect"
    assert _config_name_for("hggd", "realsense") == "hggd_realsense"
    assert _config_name_for("rngnet", "kinect") == "rngnet_kinect"
    assert _parse_scene_id("scene_0100") == 100
    assert _split_for_scene(99) == "train"
    assert _split_for_scene(100) == "test"
    with pytest.raises(SystemExit, match="0-189"):
        _split_for_scene(190)
    with pytest.raises(SystemExit, match="no GraRe configuration"):
        _config_name_for("scale_balanced_grasp", "kinect")


def test_demo_reranking_preserves_candidates_and_replaces_only_score() -> None:
    original = _grasps()
    reranked, summary = rerank_candidates(original, np.array([0.9, 0.1, 0.3], dtype=np.float32), score_weight=1.0)
    assert np.array_equal(np.sort(reranked[:, 1:], axis=0), np.sort(original[:, 1:], axis=0))
    assert np.all(reranked[:-1, 0] > reranked[1:, 0])
    assert summary["candidate_count"] == 3
    assert summary["top1_changed_vs_base"] is True


def test_demo_score_toggle_restores_grare_scores_to_detector_pose_order() -> None:
    detector = _grasps()
    order = np.array([2, 0, 1], dtype=np.int64)
    reranked = detector[order].copy()
    reranked[:, 0] = [0.9, 0.8, 0.7]
    restored = _restore_original_candidate_order(reranked, order)
    assert np.array_equal(restored[:, 1:], detector[:, 1:])
    assert np.allclose(restored[:, 0], [0.8, 0.7, 0.9])
    assert np.allclose(_score_color_values(restored[:, 0]), [0.5, 0.0, 1.0])


def test_demo_score_colors_normalize_out_of_range_grare_scores() -> None:
    assert np.allclose(_score_color_values(np.array([-2.0, 0.0, 2.0], dtype=np.float32)), [0.0, 0.5, 1.0])
    assert np.allclose(_score_color_values(np.array([3.0, 3.0], dtype=np.float32)), [0.5, 0.5])


@pytest.mark.parametrize("mask_kind", ["missing", "valid", "wrong_shape", "empty"])
def test_demo_frame_workspace_mask(tmp_path: Path, mask_kind: str) -> None:
    import cv2
    from scipy.io import savemat

    frame_dir = tmp_path / "scenes" / "scene_0100" / "realsense"
    for folder in ("rgb", "depth", "meta", "workspace_mask"):
        (frame_dir / folder).mkdir(parents=True)
    rgb = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.array([[1000, 0], [1000, 1000]], dtype=np.uint16)
    assert cv2.imwrite(str(frame_dir / "rgb" / "0000.png"), rgb)
    assert cv2.imwrite(str(frame_dir / "depth" / "0000.png"), depth)
    savemat(
        frame_dir / "meta" / "0000.mat",
        {"intrinsic_matrix": np.diag([2.0, 2.0, 1.0]), "factor_depth": 1000.0},
    )
    masks = {
        "valid": np.array([[255, 255], [0, 255]], dtype=np.uint8),
        "wrong_shape": np.ones((3, 2), dtype=np.uint8),
        "empty": np.zeros((2, 2), dtype=np.uint8),
    }
    if mask_kind in masks:
        assert cv2.imwrite(str(frame_dir / "workspace_mask" / "0000.png"), masks[mask_kind])
    resolved = SimpleNamespace(dataset_root=tmp_path, scene_id=100, frame_id=0, camera="realsense")
    if mask_kind in {"wrong_shape", "empty"}:
        message = "does not match depth shape" if mask_kind == "wrong_shape" else "no valid depth points"
        with pytest.raises(ValueError, match=message):
            _load_graspnet_frame(resolved)
        return

    frame = _load_graspnet_frame(resolved)
    expected = [[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]]
    if mask_kind == "missing":
        expected.insert(1, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(frame.observed_points, expected)
    np.testing.assert_array_equal(frame.display_points, frame.observed_points)
    np.testing.assert_array_equal(frame.display_colors, frame.observed_colors)
    np.testing.assert_allclose(frame.observed_colors, np.tile(np.array([30, 20, 10]) / 255, (len(expected), 1)))
    assert frame.valid_grid.sum() == 3
    assert frame.workspace_grid.sum() == len(expected)


def test_online_feature_extractor_needs_no_graspnet_labels() -> None:
    config = SceneLabelingConfig(
        detector="graspnet_baseline",
        benchmark="graspnet",
        split="demo",
        camera="realsense",
        dataset_root="",
        sam=SamObjectCloudConfig(enabled=False),
    )
    extractor = OnlineFeatureExtractor(config)
    grasps = _grasps()[:1]
    points_grid = np.zeros((4, 4, 3), dtype=np.float32)
    points_grid[..., 2] = 1.0
    points_grid[2, 2, :2] = [0.01, -0.01]
    valid = np.ones((4, 4), dtype=bool)
    output = extractor.extract(
        grasps,
        observed_points=points_grid.reshape(-1, 3),
        points_grid=points_grid,
        valid_grid=valid,
        intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        rgb_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    assert output["local_cloud"].shape == (1, 512, 3)
    assert output["cloud_mask"].shape == (1, 512)
    assert output["cloud_mask"].any()
    assert output["object_cloud"].shape == (1, 512, 3)
    assert not output["object_cloud"].any()


def test_visualization_selection_accepts_returning_nms_api() -> None:
    class Group:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def nms(self) -> "Group":
            self.calls.append("nms")
            return self

        def sort_by_score(self) -> "Group":
            self.calls.append("sort_by_score")
            return self

        def __getitem__(self, item: object) -> str:
            self.calls.append(item)
            return "top-grasps"

    group = Group()
    assert _apply_baseline_visualization_steps(group, top_k=50) == "top-grasps"
    assert group.calls == ["nms", "sort_by_score", slice(None, 50, None)]


def test_visualization_selection_accepts_in_place_nms_api() -> None:
    class Group:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def nms(self) -> None:
            self.calls.append("nms")
            return None

        def sort_by_score(self) -> "Group":
            self.calls.append("sort_by_score")
            return self

        def __getitem__(self, item: object) -> str:
            self.calls.append(item)
            return "top-grasps"

    group = Group()
    assert _apply_baseline_visualization_steps(group, top_k=50) == "top-grasps"
    assert group.calls == ["nms", "sort_by_score", slice(None, 50, None)]
