#!/usr/bin/env python3
"""Run the official GraspNet1B evaluator on a dump folder and persist the
per-scene raw accuracy tensor alongside the JSON summary.

Output: a (n_scenes, top_k, n_grasps, n_mu) tensor (typically (90, 256, 50, 6))
plus a JSON summary with overall / split-level (seen/similar/novel) AP.

By default each scene is evaluated in its own subprocess and every annotation is
written to disk as soon as it is scored, so an interrupted run resumes from the
last completed annotation and a wedged scene is retried under a timeout. Pass
``--no-checkpoint`` to call the official ``eval_all`` in one shot instead, which
keeps all results in memory and discards them if the run does not finish.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from grare.evaluation.checkpointed import (
    ANN_SHAPE,
    SCENE_SHAPE,
    TOTAL_ANN,
    ann_path,
    assemble_scene,
    atomic_save_npy,
    evaluate_annotation,
    scene_path,
    valid_npy,
)
from grare.evaluation.graspnet_eval_adapter import GraspNetEvalAdapter
from grare.utils.cpu import effective_cpu_count
from grare.utils.experiment_logging import timestamp


TEST_SCENES = tuple(range(100, 190))


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
        help="Number of scenes evaluated concurrently. Defaults to half the usable "
             "cores, respecting any container CPU quota.",
    )
    p.add_argument("--save-raw", required=True, help="Path to output .npy for the per-scene tensor")
    p.add_argument("--save-summary", required=True, help="Path to output summary json")
    p.add_argument("--tag", default=None, help="Optional tag recorded in the summary json")
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Call the official eval_all in one shot instead of evaluating scene by scene. "
             "Nothing is written until all scenes finish.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Where per-annotation shards are written. Defaults to <save-raw>.shards/.",
    )
    p.add_argument(
        "--scene-timeout-sec",
        type=float,
        default=1800.0,
        help="Restart a scene that produces no new annotation within this many seconds.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="How many times a scene may be restarted before the run fails.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Discard existing shards and outputs before evaluating.",
    )
    p.add_argument("--worker-scene-id", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _default_proc() -> int:
    # effective_cpu_count() honours a container CPU quota, which os.cpu_count()
    # ignores; on a limited cgroup the host count oversubscribes badly.
    return min(40, max(1, effective_cpu_count() // 2))


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
    for scene_id in TEST_SCENES:
        scene_dir = root / f"scene_{scene_id:04d}"
        legacy_scene_dir = scene_dir / camera
        for ann_id in range(TOTAL_ANN):
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
        for scene_id in TEST_SCENES:
            src_scene = root / f"scene_{scene_id:04d}"
            dst_camera = view / f"scene_{scene_id:04d}" / camera
            dst_camera.parent.mkdir(parents=True, exist_ok=True)
            dst_camera.symlink_to(src_scene, target_is_directory=True)
        yield view


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return Path(str(args.save_raw) + ".shards")


def _annotation_progress(checkpoint: Path) -> int:
    return sum(
        1
        for scene_id in TEST_SCENES
        for ann_id in range(TOTAL_ANN)
        if valid_npy(ann_path(checkpoint, scene_id, ann_id), ANN_SHAPE)
    )


def _run_scene_worker(args: argparse.Namespace) -> int:
    """Evaluate one scene, writing each annotation as soon as it is scored."""
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.eval_utils import create_table_points, voxel_sample_points

    scene_id = int(args.worker_scene_id)
    checkpoint = _checkpoint_dir(args)
    adapter = GraspNetEvalAdapter(
        dataset_root=args.dataset_root,
        camera=args.camera,
        split=args.split,
    )
    evaluator = adapter.evaluator
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    model_list, dexmodel_list, _ = evaluator.get_scene_models(scene_id, ann_id=0)
    model_sampled_list = [voxel_sample_points(model, 0.008) for model in model_list]

    for ann_id in range(TOTAL_ANN):
        out = ann_path(checkpoint, scene_id, ann_id)
        if valid_npy(out, ANN_SHAPE):
            continue
        accuracy = evaluate_annotation(
            evaluator,
            scene_id,
            ann_id,
            args.dump_folder,
            model_sampled_list,
            dexmodel_list,
            table,
            config,
        )
        atomic_save_npy(out, accuracy.astype(np.float32, copy=False))

    if not assemble_scene(checkpoint, scene_id):
        raise RuntimeError(f"failed to assemble scene_{scene_id:04d}")
    return 0


def _evaluate_checkpointed(args: argparse.Namespace) -> np.ndarray:
    """Evaluate every scene in its own subprocess, resuming completed shards."""
    checkpoint = _checkpoint_dir(args)
    if args.force:
        shutil.rmtree(checkpoint, ignore_errors=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        scene_id
        for scene_id in TEST_SCENES
        if not valid_npy(scene_path(checkpoint, scene_id), SCENE_SHAPE)
    ]
    if pending:
        print(
            json.dumps(
                {
                    "stage": "eval_start",
                    "scenes_pending": len(pending),
                    "scenes_done": len(TEST_SCENES) - len(pending),
                    "annotations_done": _annotation_progress(checkpoint),
                    "checkpoint_dir": str(checkpoint),
                    "concurrency": args.proc,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    retries = {scene_id: 0 for scene_id in TEST_SCENES}
    active: dict[int, tuple[subprocess.Popen, float, int]] = {}
    queue = list(pending)

    def launch(scene_id: int) -> None:
        log = (log_dir / f"scene_{scene_id:04d}.log").open("a", encoding="utf-8")
        command = [
            sys.executable, "-m", "grare.cli.evaluate",
            "--dataset-root", str(args.dataset_root),
            "--dump-folder", str(args.dump_folder),
            "--camera", str(args.camera),
            "--split", str(args.split),
            "--save-raw", str(args.save_raw),
            "--save-summary", str(args.save_summary),
            "--checkpoint-dir", str(checkpoint),
            "--worker-scene-id", str(scene_id),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        done = sum(
            1 for ann_id in range(TOTAL_ANN)
            if valid_npy(ann_path(checkpoint, scene_id, ann_id), ANN_SHAPE)
        )
        active[scene_id] = (process, time.time(), done)

    while queue or active:
        while queue and len(active) < max(1, args.proc):
            launch(queue.pop(0))

        time.sleep(5.0)

        for scene_id in list(active):
            process, last_change, last_done = active[scene_id]
            done = sum(
                1 for ann_id in range(TOTAL_ANN)
                if valid_npy(ann_path(checkpoint, scene_id, ann_id), ANN_SHAPE)
            )
            if done != last_done:
                active[scene_id] = (process, time.time(), done)
                last_change, last_done = time.time(), done

            finished = process.poll() is not None
            stalled = (time.time() - last_change) > args.scene_timeout_sec

            if finished and valid_npy(scene_path(checkpoint, scene_id), SCENE_SHAPE):
                active.pop(scene_id)
                continue

            if finished or stalled:
                if stalled and not finished:
                    process.kill()
                    process.wait(timeout=30)
                active.pop(scene_id)
                retries[scene_id] += 1
                if retries[scene_id] > args.max_retries:
                    raise RuntimeError(
                        f"scene_{scene_id:04d} did not finish after {args.max_retries} retries; "
                        f"see {log_dir / f'scene_{scene_id:04d}.log'}"
                    )
                reason = "stalled" if stalled else f"exit={process.returncode}"
                print(
                    f"[eval] retrying scene_{scene_id:04d} ({reason}, "
                    f"attempt {retries[scene_id]}/{args.max_retries}, {done}/{TOTAL_ANN} annotations)",
                    flush=True,
                )
                queue.append(scene_id)

    return np.stack(
        [np.load(scene_path(checkpoint, scene_id)) for scene_id in TEST_SCENES],
        axis=0,
    ).astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    if args.benchmark != "graspnet":
        raise SystemExit(f"grare supports only benchmark='graspnet'; got {args.benchmark!r}")
    if args.split != "test":
        raise SystemExit(f"unsupported split: {args.split}")

    if args.worker_scene_id is not None:
        return _run_scene_worker(args)

    t0 = time.perf_counter()
    _validate_graspnet_test_dump_complete(args.dump_folder, args.camera)

    official_ap_values: list[float] = []
    if args.no_checkpoint:
        adapter = GraspNetEvalAdapter(
            dataset_root=args.dataset_root,
            camera=args.camera,
            split=args.split,
        )
        with _official_eval_dump_view(args.dump_folder, args.camera) as eval_dump_folder:
            res, ap_values = adapter.evaluator.eval_all(str(eval_dump_folder), proc=args.proc)
        res_arr = np.asarray(res, dtype=np.float32)
        official_ap_values = [float(x) for x in ap_values]
    else:
        res_arr = _evaluate_checkpointed(args)

    raw_path = Path(args.save_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, res_arr)

    split_breakdown = {
        "overall": _metric_triplet(res_arr),
        "seen": _metric_triplet(res_arr[:30]) if res_arr.shape[0] >= 30 else None,
        "similar": _metric_triplet(res_arr[30:60]) if res_arr.shape[0] >= 60 else None,
        "novel": _metric_triplet(res_arr[60:90]) if res_arr.shape[0] >= 90 else None,
    }
    if not official_ap_values and res_arr.shape[0] >= 90:
        official_ap_values = [
            float(np.mean(res_arr[:30])),
            float(np.mean(res_arr[30:60])),
            float(np.mean(res_arr[60:90])),
        ]
    payload = {
        "tag": args.tag,
        "dump_folder": str(Path(args.dump_folder).resolve()),
        "camera": args.camera,
        "split": args.split,
        "raw_path": str(raw_path.resolve()),
        "shape": list(res_arr.shape),
        "official_ap_values": official_ap_values,
        "official_ap_percent_values": [round(float(x) * 100.0, 4) for x in official_ap_values],
        "overall": split_breakdown["overall"],
        "seen": split_breakdown["seen"],
        "similar": split_breakdown["similar"],
        "novel": split_breakdown["novel"],
        "proc": args.proc,
        "checkpointed": not args.no_checkpoint,
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
