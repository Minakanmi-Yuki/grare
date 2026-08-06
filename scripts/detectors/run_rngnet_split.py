#!/usr/bin/env python3
"""Dump one GraspNet split with the official standalone RNGNet implementation.

RNGNet is distributed as a self-contained ``RNGNet.py`` module.  This adapter
keeps that code untouched and translates its ``GraspGroup`` output into the
same per-frame ``(K, 17)`` layout used by every GraRe detector adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from frame_dump_utils import (
    ProgressReporter,
    configure_cuda,
    configure_seed,
    dump_path,
    grasp_group_array,
    iter_frame_indices,
    prefetch_items,
    points_from_depth,
    read_intrinsics,
    read_rgbd,
    save_grasp_array,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RNGNET_ROOT = Path(
    os.environ.get("GRARE_RNGNET_ROOT", str(PROJECT_ROOT / "external" / "RNGNet"))
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
    parser.add_argument("--center-num", type=int, default=48)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument(
        "--collision_thresh",
        type=float,
        default=0.0,
        help="Enable RNGNet collision filtering when positive; zero keeps its raw candidates.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--index-shard-count", type=int, default=1)
    parser.add_argument("--index-shard-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser.parse_args()


def _load_rngnet() -> object:
    if not (RNGNET_ROOT / "RNGNet.py").is_file():
        raise SystemExit(
            f"RNGNet source is missing: {RNGNET_ROOT}\n"
            "Clone https://github.com/THU-VCLab/RNGNet into external/RNGNet "
            "or set GRARE_RNGNET_ROOT."
        )
    if str(RNGNET_ROOT) not in sys.path:
        sys.path.insert(0, str(RNGNET_ROOT))
    from RNGNet import RngNet  # type: ignore[import-not-found]

    return RngNet


def main() -> int:
    args = parse_args()
    if args.center_num <= 0 or args.local_k <= 0:
        raise SystemExit("--center-num and --local-k must be positive")
    if args.data_workers < 0 or args.prefetch_factor <= 0:
        raise SystemExit("--data_workers must be non-negative and --prefetch-factor must be positive")
    if args.max_batches is not None and args.max_batches < 0:
        raise SystemExit("--max_batches must be non-negative")
    try:
        selected = list(
            iter_frame_indices(
                split=args.split,
                shard_count=args.index_shard_count,
                shard_id=args.index_shard_id,
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    dataset_root = Path(args.dataset_root).expanduser()
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"RNGNet checkpoint does not exist: {checkpoint_path}")
    if not dataset_root.is_dir():
        raise SystemExit(f"GraspNet dataset root does not exist: {dataset_root}")

    pending: list[tuple[int, int]] = []
    skipped = 0
    for scene_id, frame_id in selected:
        path = dump_path(dump_dir, scene_id=scene_id, camera=args.camera, frame_id=frame_id)
        if args.skip_existing and path.is_file():
            skipped += 1
        else:
            pending.append((scene_id, frame_id))
    if args.max_batches is not None:
        pending = pending[: args.max_batches]

    configure_seed(args.seed, deterministic=bool(args.deterministic))
    device = configure_cuda(tf32=bool(args.tf32), cudnn_benchmark=bool(args.cudnn_benchmark))
    if str(device) != "cuda:0":
        raise SystemExit("RNGNet requires a CUDA device.")
    RngNet = _load_rngnet()
    detector = RngNet(
        checkpoint_path=str(checkpoint_path),
        camera=args.camera,
        device="cuda",
        use_anchornet=True,
        params={"center_num": args.center_num, "local_k": args.local_k},
    )

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "stage": "rngnet_dump_setup",
                "detector": "rngnet",
                "dataset_root": str(dataset_root.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "dump_dir": str(dump_dir.resolve()),
                "camera": args.camera,
                "split": args.split,
                "selected_frames": len(selected),
                "pending_frames": len(pending),
                "skipped_existing_files": skipped,
                "center_num": args.center_num,
                "local_k": args.local_k,
                "data_workers": args.data_workers,
                "prefetch_factor": args.prefetch_factor,
                "collision_filter": bool(args.collision_thresh > 0),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    progress = ProgressReporter(detector="rngnet", total=len(pending))
    def load_frame(
        item: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        scene_id, frame_id = item
        rgb, depth = read_rgbd(
            dataset_root, scene_id=scene_id, camera=args.camera, frame_id=frame_id
        )
        intrinsics = (
            read_intrinsics(dataset_root, scene_id=scene_id, camera=args.camera)
            if args.collision_thresh > 0
            else None
        )
        return rgb, depth, intrinsics

    for (scene_id, frame_id), (rgb, depth, intrinsics) in prefetch_items(
        pending, load_frame, workers=args.data_workers, factor=args.prefetch_factor
    ):
        grasps = detector.infer_from_rgbd_heatmap(rgb.astype(np.float32) / 255.0, depth)
        if args.collision_thresh > 0 and len(grasps):
            assert intrinsics is not None
            grasps = detector.postprocess(
                grasps,
                scene_points=points_from_depth(depth, intrinsics),
                nms_translation_thresh=0.03,
                nms_rotation_thresh=float(np.pi / 6),
            )
        elif len(grasps):
            grasps = detector.postprocess(
                grasps,
                nms_translation_thresh=0.03,
                nms_rotation_thresh=float(np.pi / 6),
            )
        save_grasp_array(
            dump_path(dump_dir, scene_id=scene_id, camera=args.camera, frame_id=frame_id),
            grasp_group_array(grasps),
        )
        progress.advance(scene_id=scene_id, frame_id=frame_id)

    print(
        json.dumps(
            {
                "stage": "rngnet_dump",
                "detector": "rngnet",
                "camera": args.camera,
                "split": args.split,
                "processed_frames": len(pending),
                "skipped_existing_files": skipped,
                "runtime_sec": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
