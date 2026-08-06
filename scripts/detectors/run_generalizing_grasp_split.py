#!/usr/bin/env python3
"""Run Generalizing-Grasp per RGB-D frame for GraRe's canonical dump layout.

The upstream evaluator consumes one fused point cloud per scene, whereas
GraRe's labels, features, training, and official evaluator operate on the 256
camera frames in every scene.  This adapter uses the released frozen model on
each original RGB-D frame, deriving its normal features directly from that
frame, and writes the shared ``scene_*/camera/####.npy`` contract.
"""

from __future__ import annotations

import argparse
import collections.abc
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from frame_dump_utils import (
    ProgressReporter,
    configure_cuda,
    configure_seed,
    dump_path,
    iter_frame_indices,
    prefetch_items,
    points_from_depth,
    read_intrinsics,
    read_rgbd,
    save_grasp_array,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERALIZING_ROOT = Path(
    os.environ.get("GRARE_GENERALIZING_GRASP_ROOT", str(PROJECT_ROOT / "external" / "Generalizing-Grasp"))
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--camera", choices=("realsense", "kinect"), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--data_workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--index-shard-count", type=int, default=1)
    parser.add_argument("--index-shard-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser.parse_args()


def _install_torch_six_shim() -> None:
    if "torch._six" not in sys.modules:
        shim = types.ModuleType("torch._six")
        shim.container_abcs = collections.abc
        sys.modules["torch._six"] = shim


def _load_modules():
    if not (GENERALIZING_ROOT / "models" / "graspnet_sparseconv.py").is_file():
        raise SystemExit(
            f"Generalizing-Grasp source is missing: {GENERALIZING_ROOT}\n"
            "Clone https://github.com/mahaoxiang822/Generalizing-Grasp into "
            "external/Generalizing-Grasp or set GRARE_GENERALIZING_GRASP_ROOT."
        )
    _install_torch_six_shim()
    for path in (
        GENERALIZING_ROOT,
        GENERALIZING_ROOT / "models",
        GENERALIZING_ROOT / "utils",
        GENERALIZING_ROOT / "pointnet2",
        GENERALIZING_ROOT / "knn",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import MinkowskiEngine as ME
        from graspnet_sparseconv import GraspNet_MSCQ, pred_decode
        from collision_detector import ModelFreeCollisionDetector
        from graspnetAPI import GraspGroup
    except ImportError as error:
        raise SystemExit(
            "Generalizing-Grasp needs MinkowskiEngine and PointNet++ operators. "
            "Run ./scripts/build_detector_extensions.sh after cloning the detector sources. "
            f"Original import error: {error}"
        ) from error
    return ME, GraspNet_MSCQ, pred_decode, ModelFreeCollisionDetector, GraspGroup


def _frame_points_and_normals(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    num_point: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a raw-frame cloud and consistently oriented finite-difference normals."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    z = depth / 1000.0
    points = np.stack(
        (
            (cols - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0]),
            (rows - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1]),
            z,
        ),
        axis=-1,
    )
    # Central finite differences avoid an Open3D normal-estimation pass per
    # frame, which would otherwise dominate frozen-detector dumping.
    tangent_u = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    tangent_v = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normals = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    valid = (z > 0) & np.isfinite(normals).all(axis=-1) & (norm[..., 0] > 1e-7)
    # Point normals conventionally face the camera in the released sparse
    # backbone input.
    facing_away = np.sum(normals * points, axis=-1) > 0
    normals[facing_away] *= -1
    point_values = points[valid]
    normal_values = normals[valid]
    if len(point_values) == 0:
        raise ValueError("RGB-D frame has no valid points")
    replace = len(point_values) < num_point
    indices = rng.choice(len(point_values), size=num_point, replace=replace)
    return point_values[indices].astype(np.float32), normal_values[indices].astype(np.float32)


def _minkowski_batch(ME, cloud: np.ndarray, normals: np.ndarray, *, voxel_size: float, device: torch.device) -> dict[str, torch.Tensor]:
    coors, feats = ME.utils.sparse_collate([cloud / voxel_size], [normals])
    coors, feats, _, inverse = ME.utils.sparse_quantize(
        coors.float(), feats.float(), return_index=True, return_inverse=True
    )
    return {
        "point_clouds": torch.from_numpy(cloud)[None].to(device),
        "coors": coors.to(device),
        "feats": feats.to(device),
        "quantize2original": inverse.to(device),
    }


def _filter_and_save(
    predictions: np.ndarray,
    *,
    collision_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float,
    GraspGroup,
) -> np.ndarray:
    group = GraspGroup(np.asarray(predictions, dtype=np.float32))
    if collision_thresh > 0 and len(group):
        # Imported after the source root is set, so this is Generalizing-
        # Grasp's published model-free collision geometry.
        from collision_detector import ModelFreeCollisionDetector

        detector = ModelFreeCollisionDetector(collision_cloud, voxel_size=voxel_size)
        group = group[~detector.detect(group, approach_dist=0.05, collision_thresh=collision_thresh)]
    nms_result = group.nms(0.03, np.pi / 6)
    if nms_result is not None:
        group = nms_result
    group.sort_by_score()
    return np.asarray(group.grasp_group_array, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if args.num_point <= 0 or args.num_view <= 0 or args.voxel_size <= 0:
        raise SystemExit("--num_point, --num_view, and --voxel_size must be positive")
    if args.data_workers < 0 or args.prefetch_factor <= 0:
        raise SystemExit("--data_workers must be non-negative and --prefetch-factor must be positive")
    if args.max_batches is not None and args.max_batches < 0:
        raise SystemExit("--max_batches must be non-negative")
    try:
        selected = list(iter_frame_indices(split=args.split, shard_count=args.index_shard_count, shard_id=args.index_shard_id))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    dataset_root = Path(args.dataset_root).expanduser()
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Generalizing-Grasp checkpoint does not exist: {checkpoint_path}")
    pending, skipped = [], 0
    for item in selected:
        path = dump_path(dump_dir, scene_id=item[0], camera=args.camera, frame_id=item[1])
        if args.skip_existing and path.is_file():
            skipped += 1
        else:
            pending.append(item)
    if args.max_batches is not None:
        pending = pending[: args.max_batches]

    configure_seed(args.seed, deterministic=bool(args.deterministic))
    device = configure_cuda(tf32=bool(args.tf32), cudnn_benchmark=bool(args.cudnn_benchmark))
    if device.type != "cuda":
        raise SystemExit("Generalizing-Grasp requires CUDA.")
    ME, GraspNet_MSCQ, pred_decode, ModelFreeCollisionDetector, GraspGroup = _load_modules()
    del ModelFreeCollisionDetector  # Imported for an early, actionable dependency check.
    model = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=args.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.08,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    started = time.perf_counter()
    print(json.dumps({
        "stage": "generalizing_grasp_dump_setup", "detector": "generalizing_grasp",
        "dataset_root": str(dataset_root.resolve()), "checkpoint_path": str(checkpoint_path.resolve()),
        "dump_dir": str(dump_dir.resolve()), "camera": args.camera, "split": args.split,
        "selected_frames": len(selected), "pending_frames": len(pending), "skipped_existing_files": skipped,
        "num_point": args.num_point, "num_view": args.num_view, "voxel_size": args.voxel_size,
        "data_workers": args.data_workers, "prefetch_factor": args.prefetch_factor,
    }, ensure_ascii=False), flush=True)
    progress = ProgressReporter(detector="generalizing_grasp", total=len(pending))
    def load_frame(item: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene_id, frame_id = item
        rgb, depth = read_rgbd(
            dataset_root, scene_id=scene_id, camera=args.camera, frame_id=frame_id
        )
        intrinsics = read_intrinsics(dataset_root, scene_id=scene_id, camera=args.camera)
        return rgb, depth, intrinsics

    with torch.inference_mode():
        for (scene_id, frame_id), (_rgb, depth, intrinsics) in prefetch_items(
            pending, load_frame, workers=args.data_workers, factor=args.prefetch_factor
        ):
            rng = np.random.default_rng(args.seed + scene_id * 256 + frame_id)
            cloud, normals = _frame_points_and_normals(depth, intrinsics, num_point=args.num_point, rng=rng)
            end_points = _minkowski_batch(ME, cloud, normals, voxel_size=args.voxel_size, device=device)
            predictions, _saved = pred_decode(model(end_points))
            array = _filter_and_save(
                predictions[0].detach().cpu().numpy(),
                collision_cloud=cloud,
                collision_thresh=args.collision_thresh,
                voxel_size=args.voxel_size,
                GraspGroup=GraspGroup,
            )
            save_grasp_array(dump_path(dump_dir, scene_id=scene_id, camera=args.camera, frame_id=frame_id), array)
            progress.advance(scene_id=scene_id, frame_id=frame_id)

    print(json.dumps({
        "stage": "generalizing_grasp_dump", "detector": "generalizing_grasp", "camera": args.camera,
        "split": args.split, "processed_frames": len(pending), "skipped_existing_files": skipped,
        "runtime_sec": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
