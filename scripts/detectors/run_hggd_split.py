#!/usr/bin/env python3
"""Run HGGD on raw GraspNet RGB-D frames and write canonical GraRe dumps.

The official HGGD evaluation program is frame-based but imports CUDA-era
PyTorch3D/CuPoch utilities at module import time.  This adapter keeps the
published networks and decoding code intact while providing the small tensor
operations it uses through modern PyTorch.  Collision filtering is performed
with the same model-free GraspNet implementation used by the GN adapter.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from frame_dump_utils import (
    ProgressReporter,
    configure_cuda,
    configure_seed,
    dump_path,
    grasp_group_array,
    iter_frame_indices,
    prefetch_items,
    read_rgbd,
    save_grasp_array,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HGGD_ROOT = Path(os.environ.get("GRARE_HGGD_ROOT", str(PROJECT_ROOT / "external" / "HGGD"))).expanduser()
GN_ROOT = Path(
    os.environ.get("GRARE_GN_BASELINE_ROOT", str(PROJECT_ROOT / "external" / "graspnet-baseline"))
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--camera", choices=("realsense", "kinect"), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--data_workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--num_point", type=int, default=25600)
    parser.add_argument("--center-num", type=int, default=48)
    parser.add_argument("--group-num", type=int, default=512)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--voxel_size", type=float, default=0.01)
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


def _install_pytorch3d_compat() -> None:
    """Supply the HGGD inference subset when PyTorch3D has no CUDA 13 wheel.

    HGGD uses KNN interpolation and a point-set downsampling helper.  The
    network is permutation invariant, so a deterministic evenly-spaced subset
    is sufficient for its local PointNet input and avoids a legacy extension
    build solely for dumping frozen candidates.
    """
    try:
        import pytorch3d.ops  # noqa: F401
        return
    except ImportError:
        pass

    p3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    ops_utils = types.ModuleType("pytorch3d.ops.utils")
    transforms = types.ModuleType("pytorch3d.transforms")

    KNN = collections.namedtuple("KNN", ("dists", "idx", "knn"))

    def knn_points(p1, p2, K=1, **_kwargs):
        if p1.ndim != 3 or p2.ndim != 3:
            raise ValueError("knn_points expects (B, N, D) tensors")
        k = min(int(K), p2.shape[1])
        distances, indices = torch.topk(torch.cdist(p1, p2).square(), k=k, dim=-1, largest=False)
        expanded = p2[:, None].expand(-1, p1.shape[1], -1, -1)
        neighbours = torch.gather(
            expanded,
            2,
            indices[..., None].expand(-1, -1, -1, p2.shape[-1]),
        )
        # PyTorch3D's KNN result supports both attribute access and tuple
        # unpacking; HGGD uses both forms in its published utilities.
        return KNN(distances, indices, neighbours)

    def masked_gather(points, indices):
        expanded = points[:, None].expand(-1, indices.shape[1], -1, -1)
        return torch.gather(
            expanded,
            2,
            indices[..., None].expand(-1, -1, -1, points.shape[-1]),
        )

    def sample_farthest_points(points, lengths=None, K=1, **_kwargs):
        batch, count = points.shape[:2]
        k = int(K)
        if lengths is None:
            lengths = torch.full((batch,), count, dtype=torch.long, device=points.device)
        selected: list[torch.Tensor] = []
        for batch_id in range(batch):
            valid = max(1, int(lengths[batch_id]))
            if valid >= k:
                indices = torch.linspace(0, valid - 1, k, device=points.device).round().long()
            else:
                indices = torch.arange(k, device=points.device) % valid
            selected.append(indices)
        indices = torch.stack(selected)
        sampled = torch.gather(points, 1, indices[..., None].expand(-1, -1, points.shape[-1]))
        return sampled, indices

    def ball_query(p1, p2, K=1, **_kwargs):
        return knn_points(p1, p2, K=K)

    def euler_angles_to_matrix(euler, convention="XYZ"):
        if convention != "XYZ":
            raise ValueError("compatibility path supports XYZ Euler angles only")
        x, y, z = euler.unbind(-1)
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        rx = torch.stack((one, zero, zero, zero, x.cos(), -x.sin(), zero, x.sin(), x.cos()), -1).reshape(*x.shape, 3, 3)
        ry = torch.stack((y.cos(), zero, y.sin(), zero, one, zero, -y.sin(), zero, y.cos()), -1).reshape(*x.shape, 3, 3)
        rz = torch.stack((z.cos(), -z.sin(), zero, z.sin(), z.cos(), zero, zero, zero, one), -1).reshape(*x.shape, 3, 3)
        return rx @ ry @ rz

    def matrix_to_quaternion(matrix):
        # Numerically stable enough for HGGD's unused training-only metric path.
        trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
        qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2
        denom = 4 * qw.clamp_min(1e-8)
        qx = (matrix[..., 2, 1] - matrix[..., 1, 2]) / denom
        qy = (matrix[..., 0, 2] - matrix[..., 2, 0]) / denom
        qz = (matrix[..., 1, 0] - matrix[..., 0, 1]) / denom
        return torch.stack((qw, qx, qy, qz), -1)

    ops.knn_points = knn_points
    ops.ball_query = ball_query
    ops.sample_farthest_points = sample_farthest_points
    ops_utils.masked_gather = masked_gather
    transforms.euler_angles_to_matrix = euler_angles_to_matrix
    transforms.matrix_to_quaternion = matrix_to_quaternion
    p3d.ops = ops
    p3d.transforms = transforms
    sys.modules.update(
        {
            "pytorch3d": p3d,
            "pytorch3d.ops": ops,
            "pytorch3d.ops.utils": ops_utils,
            "pytorch3d.transforms": transforms,
        }
    )


def _load_modules(camera: str):
    if not (HGGD_ROOT / "models" / "anchornet.py").is_file():
        raise SystemExit(
            f"HGGD source is missing: {HGGD_ROOT}\n"
            "Clone https://github.com/THU-VCLab/HGGD into external/HGGD "
            "or set GRARE_HGGD_ROOT."
        )
    _install_torch_six_shim()
    _install_pytorch3d_compat()
    # HGGD imports CuPoch only inside its unused upstream collision class.
    sys.modules.setdefault("cupoch", types.ModuleType("cupoch"))
    for path in (HGGD_ROOT, GN_ROOT / "utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from dataset import config as hggd_config

    intrinsic = hggd_config.get_camera_intrinsic(camera)
    hggd_config.camera = camera
    from dataset import evaluation as hggd_evaluation
    from dataset import grasp as hggd_grasp
    from dataset import graspnet_utils as hggd_graspnet_utils
    from dataset import pc_dataset_tools
    from dataset import utils as hggd_utils
    from models.anchornet import AnchorGraspNet
    from models.localgraspnet import PointMultiGraspNet
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup

    # Several upstream modules captured the default RealSense camera at import
    # time. Route their inference calls through the selected frame intrinsics.
    hggd_utils.get_camera_intrinsic = lambda *_args, **_kwargs: intrinsic
    hggd_evaluation.get_camera_intrinsic = lambda *_args, **_kwargs: intrinsic
    hggd_grasp.get_camera_intrinsic = lambda *_args, **_kwargs: intrinsic
    hggd_graspnet_utils.get_camera_intrinsic = lambda *_args, **_kwargs: intrinsic
    pc_dataset_tools.get_camera_intrinsic = lambda *_args, **_kwargs: intrinsic
    return (
        hggd_evaluation,
        pc_dataset_tools,
        hggd_utils.PointCloudHelper,
        AnchorGraspNet,
        PointMultiGraspNet,
        ModelFreeCollisionDetector,
        GraspGroup,
    )


def _model_input(rgb: np.ndarray, depth: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror HGGD's W×H image convention and depth normalization."""
    rgb_wh = torch.from_numpy(rgb).permute(2, 1, 0).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    depth_wh = torch.from_numpy(depth.T.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)
    rgb_small = F.interpolate(rgb_wh, size=(640, 360), mode="bilinear", align_corners=False)
    depth_small = F.interpolate(depth_wh[:, None], size=(640, 360), mode="nearest").squeeze(1)
    depth_small = torch.clamp(depth_small / 1000.0 - (depth_small / 1000.0).mean(), -1, 1)
    return torch.cat((depth_small[:, None], rgb_small), dim=1), rgb_wh, depth_wh


def _standardize(group, *, collision_cloud: np.ndarray, collision_thresh: float, voxel_size: float, GraspGroup, ModelFreeCollisionDetector) -> np.ndarray:
    array = grasp_group_array(group)
    if not len(array):
        return array
    standard = GraspGroup(array)
    if collision_thresh > 0:
        detector = ModelFreeCollisionDetector(collision_cloud, voxel_size=voxel_size)
        standard = standard[~detector.detect(standard, approach_dist=0.05, collision_thresh=collision_thresh)]
    nms_result = standard.nms(0.03, np.pi / 6)
    if nms_result is not None:
        standard = nms_result
    standard.sort_by_score()
    return np.asarray(standard.grasp_group_array, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if min(args.num_point, args.center_num, args.group_num, args.local_k) <= 0:
        raise SystemExit("--num_point, --center-num, --group-num, and --local-k must be positive")
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
        raise SystemExit(f"HGGD checkpoint does not exist: {checkpoint_path}")

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
        raise SystemExit("HGGD requires CUDA.")
    (
        hggd_evaluation,
        pc_dataset_tools,
        PointCloudHelper,
        AnchorGraspNet,
        PointMultiGraspNet,
        ModelFreeCollisionDetector,
        GraspGroup,
    ) = _load_modules(args.camera)

    anchornet = AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6).to(device).eval()
    localnet = PointMultiGraspNet(info_size=3, k_cls=49).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    anchornet.load_state_dict(checkpoint["anchor"])
    localnet.load_state_dict(checkpoint["local"])
    anchors = {"gamma": checkpoint["gamma"].to(device), "beta": checkpoint["beta"].to(device)}
    helper = PointCloudHelper(args.num_point)

    started = time.perf_counter()
    print(json.dumps({
        "stage": "hggd_dump_setup", "detector": "hggd", "dataset_root": str(dataset_root.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()), "dump_dir": str(dump_dir.resolve()),
        "camera": args.camera, "split": args.split, "selected_frames": len(selected),
        "pending_frames": len(pending), "skipped_existing_files": skipped, "num_point": args.num_point,
        "center_num": args.center_num, "group_num": args.group_num, "local_k": args.local_k,
        "data_workers": args.data_workers, "prefetch_factor": args.prefetch_factor,
    }, ensure_ascii=False), flush=True)
    progress = ProgressReporter(detector="hggd", total=len(pending))
    def load_frame(item: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return read_rgbd(dataset_root, scene_id=item[0], camera=args.camera, frame_id=item[1])

    with torch.inference_mode():
        for (scene_id, frame_id), (rgb, depth) in prefetch_items(
            pending, load_frame, workers=args.data_workers, factor=args.prefetch_factor
        ):
            model_input, rgb_wh, depth_wh = _model_input(rgb, depth, device=device)
            view_points, _, _ = helper.to_scene_points(rgb_wh, depth_wh, include_rgb=True)
            points = view_points[..., :3]
            xyzs = helper.to_xyz_maps(depth_wh)
            pred_2d, perpoint_features = anchornet(model_input)
            outputs = hggd_evaluation.anchor_output_process(*pred_2d, sigma=10)
            rect_group = hggd_evaluation.detect_2d_grasp(
                *outputs, ratio=8, anchor_k=6, anchor_w=50.0, anchor_z=20.0,
                mask_thre=0.01, center_num=args.center_num, grid_size=8, grasp_nms=8,
            )
            if len(rect_group):
                points_all = pc_dataset_tools.feature_fusion(points, perpoint_features, xyzs)
                rect_groups = [rect_group]
                groups, local_centers = pc_dataset_tools.data_process(
                    points_all, depth_wh, rect_groups, args.center_num, args.group_num,
                    (640, 360), min_points=32, is_training=False,
                )
                rect_group = rect_groups[0]
                if len(groups) and len(rect_group):
                    grasp_info = torch.from_numpy(
                        np.stack((rect_group.thetas, rect_group.widths, rect_group.depths), axis=1).astype(np.float32)
                    ).to(device)
                    _, predicted, offsets = localnet(groups, grasp_info)
                    _, rectangles = hggd_evaluation.detect_6d_grasp_multi(
                        rect_group, predicted, offsets, local_centers, (640, 360), anchors, k=args.local_k
                    )
                    hggd_group = rectangles.to_6d_grasp_group(depth=0.02)
                    array = _standardize(
                        hggd_group,
                        collision_cloud=view_points[0, :, :3].detach().cpu().numpy(),
                        collision_thresh=args.collision_thresh,
                        voxel_size=args.voxel_size,
                        GraspGroup=GraspGroup,
                        ModelFreeCollisionDetector=ModelFreeCollisionDetector,
                    )
                else:
                    array = np.empty((0, 17), dtype=np.float32)
            else:
                array = np.empty((0, 17), dtype=np.float32)
            save_grasp_array(dump_path(dump_dir, scene_id=scene_id, camera=args.camera, frame_id=frame_id), array)
            progress.advance(scene_id=scene_id, frame_id=frame_id)

    print(json.dumps({
        "stage": "hggd_dump", "detector": "hggd", "camera": args.camera, "split": args.split,
        "processed_frames": len(pending), "skipped_existing_files": skipped,
        "runtime_sec": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
