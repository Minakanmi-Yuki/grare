#!/usr/bin/env python3
"""Run the official GraspNet1B evaluator on a dump folder and persist the
per-scene raw accuracy tensor alongside the JSON summary.

Output: a (n_scenes, top_k, n_grasps, n_mu) tensor (typically (90, 256, 50, 6))
plus a JSON summary with overall / split-level (seen/similar/novel) AP.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from grare.evaluation.graspnet_eval_adapter import GraspNetEvalAdapter
from grare.utils.experiment_logging import timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-scene raw eval on GraspNet1B.")
    p.add_argument("--benchmark", default="graspnet")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dump-folder", required=True)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--proc",
        type=int,
        default=_default_proc(),
        help="Number of evaluator worker processes. Defaults to min(40, max(1, os.cpu_count()//2)) "
             "so the cap matches the official graspnetAPI ceiling but adapts to the host.",
    )
    p.add_argument("--save-raw", required=True, help="Path to output .npy for the per-scene tensor")
    p.add_argument("--save-summary", required=True, help="Path to output summary json")
    p.add_argument("--tag", default=None, help="Optional tag recorded in the summary json")
    return p.parse_args()


def _default_proc() -> int:
    cpu = os.cpu_count() or 1
    return min(40, max(1, cpu // 2))


def _metric_triplet(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"AP": None, "AP@0.8": None, "AP@0.4": None}
    return {
        "AP": round(float(np.mean(arr)) * 100.0, 4),
        "AP@0.8": round(float(np.mean(arr[..., 3])) * 100.0, 4) if arr.ndim >= 4 and arr.shape[-1] >= 4 else None,
        "AP@0.4": round(float(np.mean(arr[..., 1])) * 100.0, 4) if arr.ndim >= 4 and arr.shape[-1] >= 4 else None,
    }


def _validate_graspnet_test_dump_complete(dump_folder: str | Path, camera: str) -> None:
    root = Path(dump_folder)
    missing: list[str] = []
    for scene_id in range(100, 190):
        scene_dir = root / f"scene_{scene_id:04d}"
        legacy_scene_dir = scene_dir / camera
        for ann_id in range(256):
            path = scene_dir / f"{ann_id:04d}.npy"
            if not path.is_file():
                path = legacy_scene_dir / f"{ann_id:04d}.npy"
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 10:
                    raise FileNotFoundError(
                        "GraspNet test dump is incomplete; missing examples: "
                        + ", ".join(missing)
                    )


@contextlib.contextmanager
def _official_eval_dump_view(dump_folder: str | Path, camera: str):
    root = Path(dump_folder)
    first_flat = root / "scene_0100" / "0000.npy"
    first_legacy = root / "scene_0100" / camera / "0000.npy"
    if first_legacy.is_file() or not first_flat.is_file():
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="grare_graspnet_eval_") as tmp:
        view = Path(tmp)
        for scene_id in range(100, 190):
            src_scene = root / f"scene_{scene_id:04d}"
            dst_camera = view / f"scene_{scene_id:04d}" / camera
            dst_camera.parent.mkdir(parents=True, exist_ok=True)
            dst_camera.symlink_to(src_scene, target_is_directory=True)
        yield view


def main() -> int:
    args = parse_args()
    if args.benchmark != "graspnet":
        raise SystemExit(f"grare supports only benchmark='graspnet'; got {args.benchmark!r}")
    t0 = time.perf_counter()
    if args.split != "test":
        raise SystemExit(f"unsupported split: {args.split}")
    _validate_graspnet_test_dump_complete(args.dump_folder, args.camera)
    adapter = GraspNetEvalAdapter(
        dataset_root=args.dataset_root,
        camera=args.camera,
        split=args.split,
    )
    with _official_eval_dump_view(args.dump_folder, args.camera) as eval_dump_folder:
        res, ap_values = adapter.evaluator.eval_all(str(eval_dump_folder), proc=args.proc)
    res_arr = np.asarray(res, dtype=np.float32)
    raw_path = Path(args.save_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, res_arr)

    split_breakdown = {
        "overall": _metric_triplet(res_arr),
        "seen": _metric_triplet(res_arr[:30]) if res_arr.shape[0] >= 30 else None,
        "similar": _metric_triplet(res_arr[30:60]) if res_arr.shape[0] >= 60 else None,
        "novel": _metric_triplet(res_arr[60:90]) if res_arr.shape[0] >= 90 else None,
    }
    payload = {
        "tag": args.tag,
        "dump_folder": str(Path(args.dump_folder).resolve()),
        "camera": args.camera,
        "split": args.split,
        "raw_path": str(raw_path.resolve()),
        "shape": list(res_arr.shape),
        "official_ap_values": [float(x) for x in ap_values],
        "official_ap_percent_values": [round(float(x) * 100.0, 4) for x in ap_values],
        "overall": split_breakdown["overall"],
        "seen": split_breakdown["seen"],
        "similar": split_breakdown["similar"],
        "novel": split_breakdown["novel"],
        "proc": args.proc,
        "runtime_sec": time.perf_counter() - t0,
        "timestamp": timestamp(),
    }
    Path(args.save_summary).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.save_summary).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
