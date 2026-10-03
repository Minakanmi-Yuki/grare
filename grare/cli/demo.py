#!/usr/bin/env python3
"""Visualize a frozen detector dump and its GraRe re-ranking on one frame.

``grare-demo`` selects one existing GraspNet detector dump with
``--detector``, ``--camera``, ``--scene``, and ``--frame``.  It loads the
matching RGB-D frame from ``$GRASPNET_ROOT``, applies the GraRe checkpoint
selected by that detector/camera configuration, and writes a self-contained
demo directory under ``$GRARE_OUTPUT_ROOT/demos``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from grare.rescoring.inference import _build_export_scores, _strictly_descending
from grare.config import config_name_for_selection, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DETECTORS = (
    "graspnet_baseline",
    "scale_balanced_grasp",
    "economicgrasp",
    "hggd",
    "rngnet",
)
CAMERAS = ("realsense", "kinect")


@dataclass(frozen=True)
class DemoFrame:
    """One calibrated RGB-D frame plus detector and display point clouds."""

    rgb_image: np.ndarray
    intrinsics: np.ndarray
    depth_scale: float
    points_grid: np.ndarray
    valid_grid: np.ndarray
    workspace_grid: np.ndarray
    observed_points: np.ndarray
    observed_colors: np.ndarray
    display_points: np.ndarray
    display_colors: np.ndarray


@dataclass(frozen=True)
class ResolvedDemo:
    """All config-derived paths for one detector/camera/scene/frame selection."""

    detector: str
    camera: str
    scene_id: int
    frame_id: int
    split: str
    config_name: str
    config: dict[str, Any]
    dataset_root: Path
    dump_path: Path
    grare_checkpoint: Path
    sam_checkpoint: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--detector", choices=DETECTORS, default="graspnet_baseline")
    parser.add_argument("--camera", choices=CAMERAS, default="realsense")
    parser.add_argument("--scene", type=_parse_scene_id, default=100, help="GraspNet scene ID, 0-189 (default: 100).")
    parser.add_argument("--frame", type=_parse_frame_id, default=0, help="Frame ID within the scene, 0-255 (default: 0).")
    parser.add_argument("--device", default="cuda", help="Torch device for MobileSAM and GraRe (default: cuda).")
    parser.add_argument("--rerank-batch-size", type=int, default=128, help="Candidates per GraRe forward pass.")
    parser.add_argument("--preview-top-k", type=int, default=50, help="Maximum ranked grasps shown in Open3D assets/viewer.")
    parser.add_argument("--show", action="store_true", help="Open an Open3D viewer after saving outputs; requires a display server.")
    parser.add_argument(
        "--render-png",
        action="store_true",
        help=(
            "Save detector_view.png and grare_view.png with the same fixed 1280x720 "
            "Open3D camera as --show. Requires a display server; use xvfb-run on a headless host."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print the resolved plan without loading models.")
    # These explicit paths remain available for development and recovery, but
    # ordinary usage is configuration-driven and does not expose them.
    parser.add_argument("--grare-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sam-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dump-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--score-weight", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _parse_index(value: str, *, option: str, prefix: str = "") -> int:
    text = str(value).strip()
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    if not text.isdecimal():
        raise argparse.ArgumentTypeError(f"{option} must be a non-negative integer, got {value!r}")
    return int(text)


def _parse_scene_id(value: str) -> int:
    return _parse_index(value, option="--scene", prefix="scene_")


def _parse_frame_id(value: str) -> int:
    return _parse_index(value, option="--frame")


def _split_for_scene(scene_id: int) -> str:
    if 0 <= scene_id <= 99:
        return "train"
    if 100 <= scene_id <= 189:
        return "test"
    raise SystemExit("--scene must be in the GraspNet-1Billion range 0-189")


def _required_path(value: str | None, *, env_name: str, option: str) -> Path:
    raw = value or os.environ.get(env_name)
    if not raw:
        raise SystemExit(f"{env_name} is not set and {option} was not provided. Source grare_paths.env first.")
    return Path(raw).expanduser()


def _config_name_for(detector: str, camera: str) -> str:
    try:
        return config_name_for_selection(detector, camera)
    except ValueError as exc:
        raise SystemExit(f"{exc}. Select a detector/camera pair listed in the Train and Evaluate table.") from exc


def _resolve_demo(args: argparse.Namespace) -> ResolvedDemo:
    if args.frame < 0 or args.frame > 255:
        raise SystemExit("--frame must be in the GraspNet-1Billion range 0-255")
    if args.rerank_batch_size <= 0:
        raise SystemExit("--rerank-batch-size must be positive")
    if args.preview_top_k < 0:
        raise SystemExit("--preview-top-k must be non-negative")
    config_name = _config_name_for(args.detector, args.camera)
    try:
        config = load_config(CONFIG_ROOT / f"{config_name}.yaml")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"failed to load demo configuration {config_name!r}: {exc}") from exc
    split = _split_for_scene(int(args.scene))
    dataset_root = _required_path(args.dataset_root, env_name="GRASPNET_ROOT", option="--dataset-root")
    dump_root = _required_path(args.dump_root, env_name="GRARE_DUMP_ROOT", option="--dump-root")
    scene_name = f"scene_{int(args.scene):04d}"
    frame_name = f"{int(args.frame):04d}.npy"
    dump_path = dump_root / args.detector / split / scene_name / args.camera / frame_name
    grare_checkpoint = Path(args.grare_checkpoint).expanduser() if args.grare_checkpoint else Path(
        config["paths"]["checkpoint_dir"]
    ) / "best.pt"
    sam_checkpoint = _required_path(args.sam_checkpoint, env_name="GRARE_SAM_CKPT", option="--sam-checkpoint")
    output_root = Path(config["paths"]["checkpoint_dir"]).parents[1]
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        output_root / "demos" / config_name / scene_name / f"{int(args.frame):04d}"
    )
    resolved = ResolvedDemo(
        detector=args.detector,
        camera=args.camera,
        scene_id=int(args.scene),
        frame_id=int(args.frame),
        split=split,
        config_name=config_name,
        config=config,
        dataset_root=dataset_root,
        dump_path=dump_path,
        grare_checkpoint=grare_checkpoint,
        sam_checkpoint=sam_checkpoint,
        output_dir=output_dir,
    )
    if not args.dry_run:
        _validate_resolved_demo(resolved)
    return resolved


def _validate_resolved_demo(resolved: ResolvedDemo) -> None:
    frame_dir = resolved.dataset_root / "scenes" / f"scene_{resolved.scene_id:04d}" / resolved.camera
    required = (
        frame_dir / "rgb" / f"{resolved.frame_id:04d}.png",
        frame_dir / "depth" / f"{resolved.frame_id:04d}.png",
        frame_dir / "meta" / f"{resolved.frame_id:04d}.mat",
        resolved.dump_path,
        resolved.grare_checkpoint,
        resolved.sam_checkpoint,
    )
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return
    missing_path = missing[0]
    if missing_path == resolved.dump_path:
        raise SystemExit(
            f"detector dump is missing: {missing_path}\n"
            "Run grare-dump for this detector/camera/split before using grare-demo."
        )
    raise SystemExit(f"required demo input is missing: {missing_path}")


def _load_graspnet_frame(resolved: ResolvedDemo) -> DemoFrame:
    """Load the selected raw RGB-D frame without requiring annotations."""
    import cv2
    import scipy.io as scio

    frame_dir = resolved.dataset_root / "scenes" / f"scene_{resolved.scene_id:04d}" / resolved.camera
    rgb_path = frame_dir / "rgb" / f"{resolved.frame_id:04d}.png"
    depth_path = frame_dir / "depth" / f"{resolved.frame_id:04d}.png"
    workspace_mask_path = frame_dir / "workspace_mask" / f"{resolved.frame_id:04d}.png"
    meta_path = frame_dir / "meta" / f"{resolved.frame_id:04d}.mat"
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise ValueError(f"failed to read RGB-D files for scene={resolved.scene_id}, frame={resolved.frame_id}")
    if depth.ndim != 2:
        raise ValueError(f"depth.png must be a single-channel image, got shape={depth.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta = scio.loadmat(str(meta_path))
    if "intrinsic_matrix" not in meta:
        raise KeyError(f"{meta_path} is missing intrinsic_matrix")
    intrinsics = np.asarray(meta["intrinsic_matrix"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsic_matrix must have shape (3, 3), got {intrinsics.shape}")
    if "factor_depth" not in meta:
        raise KeyError(f"{meta_path} is missing factor_depth")
    depth_scale = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"depth scale must be positive, got {depth_scale}")

    from grare.relabeling.sam_masks import points_grid_from_depth

    points_grid, valid_grid = points_grid_from_depth(depth, intrinsics, depth_scale=depth_scale)
    # Match graspnet-baseline/demo.py: visualize only the workspace-masked
    # valid depth points, rather than the complete camera field of view.
    # This removes the bright background/table points that make the GUI view
    # look overexposed compared with the upstream demo.
    workspace_mask = (
        cv2.imread(str(workspace_mask_path), cv2.IMREAD_UNCHANGED)
        if workspace_mask_path.is_file()
        else None
    )
    if workspace_mask is not None:
        if workspace_mask.shape != depth.shape:
            raise ValueError(
                f"workspace_mask.png shape {workspace_mask.shape} does not match depth shape {depth.shape}"
            )
        workspace_grid = (workspace_mask > 0) & valid_grid
    else:
        # Keep RGB-D-only recovery usable when an older asset bundle omits the
        # optional workspace mask.
        workspace_grid = valid_grid.copy()
    observed_points = points_grid[workspace_grid].astype(np.float32, copy=False)
    observed_colors = (rgb[workspace_grid].astype(np.float32) / 255.0).astype(np.float32, copy=False)
    # The upstream demo renders the same masked cloud used above.
    display_points = observed_points
    display_colors = observed_colors
    if len(observed_points) == 0:
        raise ValueError("the workspace has no valid depth points")
    return DemoFrame(
        rgb_image=rgb,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        points_grid=points_grid,
        valid_grid=valid_grid,
        workspace_grid=workspace_grid,
        observed_points=observed_points,
        observed_colors=observed_colors,
        display_points=display_points,
        display_colors=display_colors,
    )


def _load_detector_dump(path: Path) -> np.ndarray:
    """Load one standard ``(K, 17)`` detector dump without altering candidates."""
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim != 2 or (array.size and array.shape[1] < 17):
        raise ValueError(f"detector dump must have shape (K, >=17), got {array.shape}: {path}")
    if array.size == 0:
        return np.empty((0, 17), dtype=np.float32)
    return array


def _build_pose_features(grasp_group_array: np.ndarray) -> np.ndarray:
    array = np.asarray(grasp_group_array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 16:
        raise ValueError(f"GraspGroup array must have shape (K, >=16), got {array.shape}")
    rotations = array[:, 4:13].reshape(-1, 9)
    return np.concatenate([rotations, array[:, 13:16], array[:, 1:2], array[:, :1]], axis=1).astype(np.float32)


def _score_candidates(
    model: Any,
    grasp_group_array: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    count = len(grasp_group_array)
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    pose_features = _build_pose_features(grasp_group_array)
    score_parts: list[np.ndarray] = []
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            heads = model(
                torch.from_numpy(pose_features[start:stop]).to(device_obj),
                torch.from_numpy(features["local_cloud"][start:stop]).to(device_obj),
                cloud_mask=torch.from_numpy(features["cloud_mask"][start:stop]).to(device_obj),
                object_cloud=torch.from_numpy(features["object_cloud"][start:stop]).to(device_obj),
            )
            score_parts.append(heads["score"].float().cpu().numpy())
    return np.concatenate(score_parts).astype(np.float32, copy=False)


def rerank_candidates(
    grasp_group_array: np.ndarray,
    model_scores: np.ndarray,
    *,
    score_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the same z-score fusion and stable ordering as ``grare-rerank``."""
    input_grasps = np.asarray(grasp_group_array, dtype=np.float32)
    scores = np.asarray(model_scores, dtype=np.float32)
    if input_grasps.ndim != 2 or input_grasps.shape[1] < 1:
        raise ValueError("grasp_group_array must have shape (K, >=1)")
    if scores.shape != (len(input_grasps),):
        raise ValueError(f"model_scores shape {scores.shape} does not match {len(input_grasps)} candidates")
    exported, metadata = _build_export_scores(
        base_scores=input_grasps[:, 0],
        rescoring_scores=scores,
        rescoring_score_weight=float(score_weight),
    )
    order = np.argsort(-exported, kind="stable")
    reranked = input_grasps[order].copy()
    reranked[:, 0] = _strictly_descending(exported[order])
    return reranked, {
        **metadata,
        "model_scores_raw": scores[order].astype(np.float32, copy=False),
        "base_scores": input_grasps[order, 0].astype(np.float32, copy=False),
        "original_indices": order.astype(np.int64, copy=False),
    }


def _make_online_extractor(
    model: Any,
    *,
    resolved: ResolvedDemo,
    device: str,
) -> Any:
    from grare.relabeling.scene_labeling import (
        OnlineFeatureExtractor,
        SamObjectCloudConfig,
        SceneLabelingConfig,
    )

    shell_edges = tuple(float(value) for value in model.config.shell_edges_m)
    shell_budgets = (64, 128, 128, 192)
    if len(shell_edges) != len(shell_budgets) + 1:
        raise ValueError(
            "this GraRe checkpoint has an unsupported shell configuration; "
            f"expected 5 shell edges, got {len(shell_edges)}"
        )
    construction = resolved.config["feature_construction"]
    return OnlineFeatureExtractor(
        SceneLabelingConfig(
            detector=resolved.detector,
            benchmark="graspnet",
            split="demo",
            camera=resolved.camera,
            dataset_root="",
            local_cloud_radius=float(shell_edges[-1]),
            cloud_sampler="stratified_fps",
            shell_edges_m=shell_edges,
            shell_budgets=shell_budgets,
            object_cloud_points=int(model.config.object_cloud_points),
            sam=SamObjectCloudConfig(
                enabled=True,
                checkpoint=str(resolved.sam_checkpoint),
                device=device,
                cluster_radius_m=float(construction["sam_prompt_cluster_radius_m"]),
                prompt_batch_size=64,
                min_area_pixels=int(construction["sam_min_area_pixels"]),
                max_area_ratio=float(construction["sam_max_area_ratio"]),
            ),
        )
    )


def _apply_baseline_visualization_steps(group: Any, *, top_k: int) -> Any:
    """Apply GraspNet's intended NMS/sort/top-k visualization selection.

    The upstream demo calls ``gg.nms()`` before sorting, but versions of
    ``graspnetAPI`` differ: some modify the group in place and the currently
    packaged version returns a new group.  Handle both so duplicate detector
    grasps do not leak into the native GraspNet-style visualization.
    """
    nms_result = group.nms()
    if nms_result is not None:
        group = nms_result
    group.sort_by_score()
    return group[:top_k]


def _baseline_style_grasp_group(grasps: np.ndarray, *, top_k: int) -> Any:
    from graspnetAPI import GraspGroup

    group = GraspGroup(np.asarray(grasps, dtype=np.float32).copy())
    return _apply_baseline_visualization_steps(group, top_k=top_k)


def _open3d_cloud(frame: DemoFrame) -> Any:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(frame.display_points)
    cloud.colors = o3d.utility.Vector3dVector(frame.display_colors)
    return cloud


def _merge_gripper_meshes(grippers: list[Any]) -> Any:
    import open3d as o3d

    merged = o3d.geometry.TriangleMesh()
    for gripper in grippers:
        merged += gripper
    return merged


def _restore_original_candidate_order(
    reranked_grasps: np.ndarray,
    original_indices: np.ndarray,
) -> np.ndarray:
    """Put a reranked ``(K, 17)`` array back in detector candidate order."""
    reranked = np.asarray(reranked_grasps, dtype=np.float32)
    indices = np.asarray(original_indices, dtype=np.int64)
    if reranked.ndim != 2 or indices.shape != (len(reranked),):
        raise ValueError("reranked_grasps and original_indices have incompatible shapes")
    if len(reranked) and not np.array_equal(np.sort(indices), np.arange(len(reranked))):
        raise ValueError("original_indices must be a permutation of candidate indices")
    restored = np.empty_like(reranked)
    restored[indices] = reranked
    return restored


def _score_color_values(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores for GraspNet's blue-to-red color map."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {values.shape}")
    if not len(values):
        return np.empty((0,), dtype=np.float32)
    finite = np.isfinite(values)
    if not np.all(finite):
        values = values.copy()
        values[~finite] = 0.0
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.full_like(values, 0.5)
    return ((values - low) / (high - low)).astype(np.float32, copy=False)


def _colored_gripper_mesh(grasps: np.ndarray) -> Any:
    """Build one mesh whose vertex colors encode the supplied grasp scores."""
    import open3d as o3d
    from graspnetAPI import Grasp

    array = np.asarray(grasps, dtype=np.float32)
    scores = array[:, 0] if len(array) else np.empty((0,), dtype=np.float32)
    colors = _score_color_values(scores)
    mesh = o3d.geometry.TriangleMesh()
    for grasp, color_value in zip(array, colors):
        mesh += Grasp(grasp).to_open3d_geometry(
            color=(float(color_value), 0.0, float(1.0 - color_value))
        )
    # Keep the mesh topology and normals exactly like
    # graspnetAPI.plot_gripper_pro_max(); the upstream demo passes these
    # meshes directly to Open3D without an explicit normal computation.
    return mesh


def _selected_row_indices(source: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Match selected detector rows to their original candidate indices.

    The complete row is used so duplicate poses with different detector scores
    remain distinguishable. Identical duplicate rows are interchangeable for
    visualization and are consumed in source order.
    """
    source_array = np.ascontiguousarray(np.asarray(source, dtype=np.float32))
    selected_array = np.ascontiguousarray(np.asarray(selected, dtype=np.float32))
    buckets: dict[bytes, list[int]] = {}
    for index, row in enumerate(source_array):
        buckets.setdefault(row.tobytes(), []).append(index)
    indices: list[int] = []
    for row in selected_array:
        matches = buckets.get(row.tobytes())
        if not matches:
            raise ValueError("failed to match a displayed grasp to the detector candidates")
        indices.append(matches.pop(0))
    return np.asarray(indices, dtype=np.int64)


def save_open3d_assets(
    output_dir: Path,
    *,
    frame: DemoFrame,
    detector_grasps: np.ndarray,
    grare_grasps: np.ndarray,
    top_k: int,
) -> dict[str, Path]:
    """Save the exact upstream Open3D geometry for later desktop viewing."""
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = output_dir / "scene_cloud.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), _open3d_cloud(frame), write_ascii=False):
        raise OSError(f"failed to write point cloud: {cloud_path}")
    output_paths = {"scene_cloud": cloud_path}
    for name, grasps in (("detector", detector_grasps), ("grare", grare_grasps)):
        group = _baseline_style_grasp_group(grasps, top_k=top_k)
        mesh_path = output_dir / f"{name}_grippers.ply"
        # Use the same score-to-color path as the interactive viewer so both
        # Detector and GraRe assets use a consistent min-max color scale.
        mesh = _colored_gripper_mesh(np.asarray(group.grasp_group_array, dtype=np.float32))
        if not o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False):
            raise OSError(f"failed to write gripper mesh: {mesh_path}")
        output_paths[f"{name}_grippers"] = mesh_path
    return output_paths


OPEN3D_RENDER_WIDTH = 1280
OPEN3D_RENDER_HEIGHT = 720
OPEN3D_VIEW_WIDTH = OPEN3D_RENDER_WIDTH
OPEN3D_VIEW_HEIGHT = OPEN3D_RENDER_HEIGHT


def _configure_reference_camera(
    visualizer: Any,
    frame: DemoFrame,
    *,
    width: int = OPEN3D_RENDER_WIDTH,
    height: int = OPEN3D_RENDER_HEIGHT,
) -> None:
    """Use the RGB-D camera as the reference Open3D viewer camera."""
    import open3d as o3d

    intrinsics = frame.intrinsics
    scale_x = width / OPEN3D_RENDER_WIDTH
    scale_y = height / OPEN3D_RENDER_HEIGHT
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsics[0, 0]) * scale_x,
        float(intrinsics[1, 1]) * scale_y,
        float(intrinsics[0, 2]) * scale_x,
        float(intrinsics[1, 2]) * scale_y,
    )
    parameters.extrinsic = np.eye(4, dtype=np.float64)
    if not visualizer.get_view_control().convert_from_pinhole_camera_parameters(
        parameters,
        allow_arbitrary=True,
    ):
        raise RuntimeError("Open3D rejected the RGB-D reference camera parameters")


def _open3d_visualizer(
    frame: DemoFrame,
    grasps: np.ndarray,
    *,
    top_k: int,
    visible: bool,
    window_name: str = "GraRe demo",
    width: int = OPEN3D_RENDER_WIDTH,
    height: int = OPEN3D_RENDER_HEIGHT,
    left: int = 50,
    top: int = 50,
) -> Any:
    import open3d as o3d

    cloud = _open3d_cloud(frame)
    group = _baseline_style_grasp_group(grasps, top_k=top_k)
    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        visible=visible,
    )
    if not created:
        raise RuntimeError(
            "Open3D could not create a display window. This host is headless; "
            "omit --show and open scene_cloud.ply with grare_grippers.ply on a desktop host instead."
        )
    try:
        visualizer.add_geometry(cloud)
        # Keep saved renders consistent with the interactive viewer and PLY
        # assets: all displayed grasp scores use the same min-max color map.
        colored_grippers = _colored_gripper_mesh(
            np.asarray(group.grasp_group_array, dtype=np.float32)
        )
        if len(colored_grippers.vertices):
            visualizer.add_geometry(colored_grippers)
        visualizer.poll_events()
        visualizer.update_renderer()
        _configure_reference_camera(visualizer, frame, width=width, height=height)
        visualizer.poll_events()
        visualizer.update_renderer()
        return visualizer
    except Exception:
        visualizer.destroy_window()
        raise


def save_open3d_render(
    path: Path,
    *,
    frame: DemoFrame,
    grasps: np.ndarray,
    top_k: int,
) -> Path:
    """Capture the same native Open3D layout as ``--show`` to a PNG."""
    visualizer = _open3d_visualizer(frame, grasps, top_k=top_k, visible=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open3D 0.19 writes the file successfully but returns False under
        # Xvfb, so validate the output path rather than the return value.
        visualizer.capture_screen_image(str(path), do_render=True)
    finally:
        visualizer.destroy_window()
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Open3D failed to capture image: {path}")
    return path


def _show_open3d(
    frame: DemoFrame,
    detector_grasps: np.ndarray,
    grare_grasps: np.ndarray,
    *,
    top_k: int,
) -> None:
    """Show one interactive scene with a Detector/GraRe score toggle.

    ``grare_grasps`` is expected to be in detector candidate order.  The
    detector's NMS/top-k selection is used for both modes so pressing the
    button changes only the vertex colors, never the displayed poses.
    """
    import open3d as o3d
    from open3d.visualization import gui, rendering

    detector_array = np.asarray(detector_grasps, dtype=np.float32)
    grare_array = np.asarray(grare_grasps, dtype=np.float32)
    if detector_array.shape != grare_array.shape:
        raise ValueError("detector and GraRe grasp arrays must have the same shape")

    display_group = _baseline_style_grasp_group(detector_array, top_k=top_k)
    display_detector = np.asarray(display_group.grasp_group_array, dtype=np.float32).copy()
    display_indices = _selected_row_indices(detector_array, display_detector)
    display_grare = grare_array[display_indices].copy()
    if not np.array_equal(display_detector[:, 1:], display_grare[:, 1:]):
        raise ValueError("Detector and GraRe display candidates do not have identical poses")

    app = gui.Application.instance
    try:
        app.initialize()
        window = app.create_window(
            "GraRe demo — score view",
            OPEN3D_VIEW_WIDTH,
            OPEN3D_VIEW_HEIGHT,
            50,
            50,
        )
    except Exception as exc:
        raise RuntimeError(
            "Open3D could not create the interactive viewer. "
            "This host may be headless; omit --show and open the saved PLY assets "
            "on a desktop host instead."
        ) from exc

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    # Match the legacy Visualizer defaults used by the original GraspNet demo:
    # white background, vertex colors, and the legacy 5 px point size.
    scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene_widget.scene.set_lighting(rendering.Open3DScene.NO_SHADOWS, [0.577, -0.577, -0.577])
    scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    cloud_material = rendering.MaterialRecord()
    cloud_material.shader = "defaultUnlit"
    cloud_material.point_size = 5.0
    gripper_material = rendering.MaterialRecord()
    # Display the same score colors encoded by graspnetAPI without the GUI
    # renderer's specular/directional-light boost.  ``defaultLit`` makes the
    # small gripper boxes look overexposed compared with the upstream legacy
    # Visualizer, especially against the white background.
    gripper_material.shader = "defaultUnlit"

    scene_widget.scene.add_geometry("scene_cloud", _open3d_cloud(frame), cloud_material)
    detector_mesh = _colored_gripper_mesh(display_detector)
    grare_mesh = _colored_gripper_mesh(display_grare)

    def set_gripper_mesh(mesh: Any) -> None:
        if scene_widget.scene.has_geometry("grippers"):
            scene_widget.scene.remove_geometry("grippers")
        if len(mesh.vertices):
            scene_widget.scene.add_geometry("grippers", mesh, gripper_material)
        scene_widget.force_redraw()

    set_gripper_mesh(detector_mesh)

    button = gui.Button("Scores: Detector")
    button.tooltip = "Switch between Detector and GraRe grasp scores"
    showing_grare = False

    def on_button_clicked() -> None:
        nonlocal showing_grare
        showing_grare = not showing_grare
        set_gripper_mesh(grare_mesh if showing_grare else detector_mesh)
        button.text = "Scores: GraRe" if showing_grare else "Scores: Detector"

    button.set_on_clicked(on_button_clicked)

    def on_layout(layout_context: Any) -> None:
        content = window.content_rect
        scene_widget.frame = content
        preferred = button.calc_preferred_size(layout_context, button.Constraints())
        button.frame = gui.Rect(
            content.x + 12,
            content.y + 12,
            max(preferred.width + 20, 180),
            preferred.height + 10,
        )

    window.set_on_layout(on_layout)
    window.add_child(scene_widget)
    window.add_child(button)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        OPEN3D_VIEW_WIDTH,
        OPEN3D_VIEW_HEIGHT,
        float(frame.intrinsics[0, 0]) * OPEN3D_VIEW_WIDTH / OPEN3D_RENDER_WIDTH,
        float(frame.intrinsics[1, 1]) * OPEN3D_VIEW_HEIGHT / OPEN3D_RENDER_HEIGHT,
        float(frame.intrinsics[0, 2]) * OPEN3D_VIEW_WIDTH / OPEN3D_RENDER_WIDTH,
        float(frame.intrinsics[1, 2]) * OPEN3D_VIEW_HEIGHT / OPEN3D_RENDER_HEIGHT,
    )
    scene_widget.setup_camera(
        intrinsic,
        np.eye(4, dtype=np.float64),
        scene_widget.scene.bounding_box,
    )
    app.run()


def _summary_path_values(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path.resolve()) for name, path in paths.items()}


def main() -> int:
    args = parse_args()
    resolved = _resolve_demo(args)
    score_weight = float(
        resolved.config["rerank"]["lambda"] if args.score_weight is None else args.score_weight
    )
    if not 0.0 <= score_weight <= 1.0:
        raise SystemExit("--score-weight must be in [0, 1]")
    plan = {
        "stage": "grare_demo",
        "detector": resolved.detector,
        "camera": resolved.camera,
        "scene": resolved.scene_id,
        "frame": resolved.frame_id,
        "split": resolved.split,
        "config": str((CONFIG_ROOT / f"{resolved.config_name}.yaml").resolve()),
        "detector_dump": str(resolved.dump_path.resolve()),
        "grare_checkpoint": str(resolved.grare_checkpoint.resolve()),
        "sam_checkpoint": str(resolved.sam_checkpoint.resolve()),
        "output_dir": str(resolved.output_dir.resolve()),
        "headless_open3d_assets": str(resolved.output_dir.resolve()),
        "interactive_viewer": bool(args.show),
        "render_png": bool(args.render_png),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    started = time.perf_counter()
    frame = _load_graspnet_frame(resolved)
    detector_grasps = _load_detector_dump(resolved.dump_path)

    output_dir = resolved.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_path = output_dir / "detector_grasps.npy"
    grare_path = output_dir / "grare_grasps.npy"
    np.save(detector_path, detector_grasps)

    rerank_summary: dict[str, Any] = {"candidate_count": int(len(detector_grasps))}
    grare_grasps = detector_grasps.copy()
    if len(detector_grasps):
        from grare.rescoring import load_model_checkpoint

        model = load_model_checkpoint(resolved.grare_checkpoint, device=str(args.device))
        extractor = _make_online_extractor(model, resolved=resolved, device=str(args.device))
        features = extractor.extract(
            detector_grasps,
            observed_points=frame.observed_points,
            points_grid=frame.points_grid,
            valid_grid=frame.valid_grid,
            intrinsics=frame.intrinsics,
            rgb_image=frame.rgb_image,
        )
        model_scores = _score_candidates(
            model,
            detector_grasps,
            features,
            device=str(args.device),
            batch_size=int(args.rerank_batch_size),
        )
        grare_grasps, rerank_summary = rerank_candidates(
            detector_grasps,
            model_scores,
            score_weight=score_weight,
        )
        rerank_summary = {
            **rerank_summary,
            "object_cloud_nonempty": int(np.count_nonzero(np.any(features["object_cloud"] != 0, axis=(1, 2)))),
            "local_cloud_nonempty": int(np.count_nonzero(np.any(features["cloud_mask"], axis=1))),
        }
    np.save(grare_path, grare_grasps)
    grare_detector_order = grare_grasps
    if len(grare_grasps) and "original_indices" in rerank_summary:
        grare_detector_order = _restore_original_candidate_order(
            grare_grasps,
            rerank_summary["original_indices"],
        )
    open3d_assets = save_open3d_assets(
        output_dir,
        frame=frame,
        detector_grasps=detector_grasps,
        grare_grasps=grare_grasps,
        top_k=int(args.preview_top_k),
    )
    render_paths: dict[str, Path] = {}
    if args.render_png:
        render_paths = {
            "detector_view": save_open3d_render(
                output_dir / "detector_view.png",
                frame=frame,
                grasps=detector_grasps,
                top_k=int(args.preview_top_k),
            ),
            "grare_view": save_open3d_render(
                output_dir / "grare_view.png",
                frame=frame,
                grasps=grare_grasps,
                top_k=int(args.preview_top_k),
            ),
        }
    if args.show:
        _show_open3d(
            frame,
            detector_grasps,
            grare_detector_order,
            top_k=int(args.preview_top_k),
        )

    summary = {
        **plan,
        "candidates_from_dump": int(len(detector_grasps)),
        "depth_scale": frame.depth_scale,
        "valid_depth_points": int(len(frame.observed_points)),
        "output_files": _summary_path_values(
            {
                "detector_grasps": detector_path,
                "grare_grasps": grare_path,
                **open3d_assets,
                **render_paths,
            }
        ),
        "rerank": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in rerank_summary.items()
            if key not in {"model_scores_raw", "base_scores", "original_indices"}
        },
        "runtime_sec": round(time.perf_counter() - started, 4),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
