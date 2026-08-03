#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from grare.rescoring import load_model_checkpoint, rerank_archive
from grare.utils.benchmark_protocol import filter_archive_paths_by_camera
from grare.utils.experiment_logging import append_jsonl, numeric_stats, reset_jsonl, timestamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank every candidate with the published GraRe z-score fusion "
            "and export evaluator-compatible .npy files."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--camera",
        default=None,
        help="Optional camera filter, e.g. kinect or realsense. Archives from other cameras are ignored.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-root", default=None, help="Optional relabeled archive root that provides local_cloud features.")
    parser.add_argument(
        "--object-pooled-root",
        default=None,
        help=(
            "Optional sidecar root holding object_pooled npz files with the "
            "same relative layout as --input-root."
        ),
    )
    parser.add_argument(
        "--require-object-pooled",
        action="store_true",
        help="Fail if a test archive is missing its object_pooled sidecar/cache.",
    )
    parser.add_argument(
        "--scene-list-json",
        default=None,
        help="Optional JSON file containing scene names like scene_0018 used to filter the input archives.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated list of CUDA devices (e.g. 'cuda:0,cuda:1') for "
             "multi-GPU rerank. When set with --num-workers > 1, workers are "
             "round-robin assigned to devices. Default: --device value applied "
             "to every worker.",
    )
    parser.add_argument(
        "--rescoring-score-weight",
        type=float,
        default=0.5,
        help=(
            "Lambda in [0, 1] for fusing z-score-normalized detector and "
            "GraRe scores."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes. Each worker loads the checkpoint once and reranks a disjoint archive shard.",
    )
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--save-records-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = timestamp()
    started_perf = time.perf_counter()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    scene_name_filter = _load_scene_name_filter(args.scene_list_json)
    rescoring_score_weight = float(args.rescoring_score_weight)
    if not 0.0 <= rescoring_score_weight <= 1.0:
        raise ValueError("--rescoring-score-weight must be in [0, 1]")

    rescoring_parts: list[np.ndarray] = []
    exported_score_parts: list[np.ndarray] = []
    base_score_parts: list[np.ndarray] = []
    raw_model_score_parts: list[np.ndarray] = []
    num_grasps_parts: list[int] = []
    top1_rescoring_scores: list[float] = []
    top1_exported_scores: list[float] = []
    top1_model_scores_raw: list[float] = []
    top1_base_scores_after: list[float] = []
    top1_base_scores_before: list[float] = []
    top1_changed_vs_base_count = 0
    candidate_count_parts: list[int] = []
    sort_descending: bool | None = None
    per_scene: dict[str, dict[str, int]] = {}

    if args.save_records_path:
        reset_jsonl(args.save_records_path)

    archive_paths = filter_archive_paths_by_camera(sorted(input_root.glob("**/*.npz")), args.camera)
    if scene_name_filter is not None:
        archive_paths = [
            archive_path
            for archive_path in archive_paths
            if _archive_scene_name(input_root, archive_path) in scene_name_filter
        ]
    if args.num_workers > 1:
        summary = _run_parallel(
            args=args,
            archive_paths=archive_paths,
            started_at=started_at,
            started_perf=started_perf,
            rescoring_score_weight=rescoring_score_weight,
            scene_name_filter=scene_name_filter,
        )
        if args.save_path:
            write_json(args.save_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    model = load_model_checkpoint(args.checkpoint, device=args.device if torch.cuda.is_available() else "cpu")

    for archive_path in archive_paths:
        features_archive_path = None
        if args.features_root is not None:
            features_archive_path = Path(args.features_root) / archive_path.relative_to(input_root)
        object_pooled_archive_path = None
        if args.object_pooled_root is not None:
            object_pooled_archive_path = Path(args.object_pooled_root) / archive_path.relative_to(input_root)
        result = rerank_archive(
            archive_path,
            model,
            device=args.device,
            features_archive_path=features_archive_path,
            object_pooled_archive_path=object_pooled_archive_path,
            require_object_pooled=args.require_object_pooled,
            rescoring_score_weight=rescoring_score_weight,
            score_normalization="zscore",
        )
        save_path = _resolve_output_path(
            output_root=output_root,
            input_root=input_root,
            archive_path=archive_path,
            meta=result["meta"],
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result["grasp_group_array"])

        exported_scores = result["exported_scores"].astype(np.float32, copy=False)
        rescoring_scores = result["rescoring_scores"].astype(np.float32, copy=False)
        model_scores_raw = result["model_scores_raw"].astype(np.float32, copy=False)
        base_scores = result["base_scores"].astype(np.float32, copy=False)
        base_scores_input = result["base_scores_input"].astype(np.float32, copy=False)
        if sort_descending is None:
            sort_descending = bool(result.get("sort_descending", True))
        meta = result["meta"]
        scene_name = f"scene_{int(meta['scene_id']):04d}"
        per_scene_stats = per_scene.setdefault(scene_name, {"num_files": 0, "num_grasps": 0})
        per_scene_stats["num_files"] += 1
        per_scene_stats["num_grasps"] += int(len(rescoring_scores))

        exported_score_parts.append(exported_scores)
        rescoring_parts.append(rescoring_scores)
        raw_model_score_parts.append(model_scores_raw)
        base_score_parts.append(base_scores)
        num_grasps_parts.append(int(len(rescoring_scores)))
        if bool(result.get("top1_changed_vs_base", False)):
            top1_changed_vs_base_count += 1
        candidate_count_parts.append(int(result.get("candidate_count", len(rescoring_scores))))
        if len(rescoring_scores) > 0:
            top1_exported_scores.append(float(exported_scores[0]))
            top1_rescoring_scores.append(float(rescoring_scores[0]))
            top1_model_scores_raw.append(float(model_scores_raw[0]))
            top1_base_scores_after.append(float(base_scores[0]))
            top1_base_scores_before.append(float(np.max(base_scores_input)))

        if args.save_records_path:
            append_jsonl(
                args.save_records_path,
                {
                    "scene_id": int(meta["scene_id"]),
                    "frame_id": int(meta["frame_id"]),
                    "scene_name": scene_name,
                    "archive_path": str(archive_path),
                    "output_path": str(save_path),
                    "num_grasps": int(len(rescoring_scores)),
                    "exported_scores": numeric_stats(exported_scores),
                    "rescoring_scores": numeric_stats(rescoring_scores),
                    "model_scores_raw": numeric_stats(model_scores_raw),
                    "base_scores_after_rerank": numeric_stats(base_scores),
                    "base_scores_before_rerank": numeric_stats(base_scores_input),
                    "top1_exported_score": float(exported_scores[0]) if len(exported_scores) > 0 else None,
                    "top1_rescoring_score": float(rescoring_scores[0]) if len(rescoring_scores) > 0 else None,
                    "top1_model_score_raw": float(model_scores_raw[0]) if len(model_scores_raw) > 0 else None,
                    "top1_base_score_before": float(np.max(base_scores_input)) if len(base_scores_input) > 0 else None,
                    "top1_base_score_after": float(base_scores[0]) if len(base_scores) > 0 else None,
                    "score_normalization": result.get("score_normalization"),
                    "rescoring_score_weight": result.get("rescoring_score_weight"),
                    "candidate_count": result.get("candidate_count"),
                    "base_top1_idx": result.get("base_top1_idx"),
                    "proposed_top1_idx": result.get("proposed_top1_idx"),
                    "final_top1_idx": result.get("final_top1_idx"),
                    "top1_changed_vs_base": bool(result.get("top1_changed_vs_base", False)),
                    "score_column_index": result.get("score_column_index"),
                    "score_column_overwritten": bool(result.get("score_column_overwritten", False)),
                },
            )

    summary = {
        "stage": "rerank",
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "features_root": None if args.features_root is None else str(Path(args.features_root).resolve()),
        "object_pooled_root": None if args.object_pooled_root is None else str(Path(args.object_pooled_root).resolve()),
        "require_object_pooled": bool(args.require_object_pooled),
        "scene_list_json": None if args.scene_list_json is None else str(Path(args.scene_list_json).resolve()),
        "scene_name_filter": None if scene_name_filter is None else sorted(scene_name_filter),
        "device": args.device,
        "score_normalization": "zscore",
        "rescoring_score_weight": rescoring_score_weight,
        "save_records_path": None if args.save_records_path is None else str(Path(args.save_records_path).resolve()),
        "sort_descending": sort_descending,
        "score_column_index": 0,
        "score_column_overwritten": True,
        "num_files": len(num_grasps_parts),
        "num_scenes": len(per_scene),
        "num_grasps_total": int(sum(num_grasps_parts)),
        "num_grasps_per_file": numeric_stats(num_grasps_parts),
        "exported_scores": numeric_stats(np.concatenate(exported_score_parts, axis=0) if exported_score_parts else np.empty((0,), dtype=np.float32)),
        "rescoring_scores": numeric_stats(np.concatenate(rescoring_parts, axis=0) if rescoring_parts else np.empty((0,), dtype=np.float32)),
        "model_scores_raw": numeric_stats(np.concatenate(raw_model_score_parts, axis=0) if raw_model_score_parts else np.empty((0,), dtype=np.float32)),
        "base_scores_after_rerank": numeric_stats(np.concatenate(base_score_parts, axis=0) if base_score_parts else np.empty((0,), dtype=np.float32)),
        "top1_exported_scores": numeric_stats(top1_exported_scores),
        "top1_rescoring_scores": numeric_stats(top1_rescoring_scores),
        "top1_model_scores_raw": numeric_stats(top1_model_scores_raw),
        "top1_base_scores_before_rerank": numeric_stats(top1_base_scores_before),
        "top1_base_scores_after_rerank": numeric_stats(top1_base_scores_after),
        "top1_changed_vs_base_count": top1_changed_vs_base_count,
        "candidate_count_per_file": numeric_stats(candidate_count_parts),
        "per_scene": per_scene,
    }
    if args.save_path:
        write_json(args.save_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_parallel(
    *,
    args: argparse.Namespace,
    archive_paths: list[Path],
    started_at: str,
    started_perf: float,
    rescoring_score_weight: float,
    scene_name_filter: set[str] | None,
) -> dict[str, Any]:
    workers = max(1, min(int(args.num_workers), len(archive_paths) or 1))
    shards = _shard_paths(archive_paths, workers)
    devices = _resolve_worker_devices(args, workers)
    options = {
        "input_root": str(Path(args.input_root).resolve()),
        "camera": args.camera,
        "output_root": str(Path(args.output_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "features_root": None if args.features_root is None else str(Path(args.features_root).resolve()),
        "object_pooled_root": None if args.object_pooled_root is None else str(Path(args.object_pooled_root).resolve()),
        "require_object_pooled": bool(args.require_object_pooled),
        "device": args.device,
        "rescoring_score_weight": rescoring_score_weight,
        "save_records_path": args.save_records_path,
    }
    print(
        json.dumps(
            {
                "stage": "rerank_parallel_start",
                "num_workers": workers,
                "num_archives": len(archive_paths),
                "device": args.device,
                "devices": devices,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    ctx = mp.get_context("spawn")
    worker_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _worker_main,
                {
                    "worker_id": worker_id,
                    "archive_paths": [str(path) for path in shard],
                    "options": {**options, "device": devices[worker_id % len(devices)]},
                },
            )
            for worker_id, shard in enumerate(shards)
            if shard
        ]
        for future in as_completed(futures):
            worker_results.append(future.result())

    if args.save_records_path:
        _merge_worker_record_files(Path(args.save_records_path), worker_count=workers)

    return _merge_worker_summaries(
        worker_results=worker_results,
        args=args,
        started_at=started_at,
        started_perf=started_perf,
        rescoring_score_weight=rescoring_score_weight,
        scene_name_filter=scene_name_filter,
    )


def _worker_main(payload: dict[str, Any]) -> dict[str, Any]:
    worker_id = int(payload["worker_id"])
    paths = [Path(path) for path in payload["archive_paths"]]
    options = payload["options"]
    model = load_model_checkpoint(
        options["checkpoint"],
        device=options["device"] if torch.cuda.is_available() else "cpu",
    )
    input_root = Path(options["input_root"])
    output_root = Path(options["output_root"])
    features_root = None if options["features_root"] is None else Path(options["features_root"])
    object_pooled_root = None if options.get("object_pooled_root") is None else Path(options["object_pooled_root"])
    records_path = _worker_records_path(options.get("save_records_path"), worker_id)
    if records_path is not None:
        reset_jsonl(records_path)

    accum = _WorkerAccumulator(worker_id=worker_id)
    started = time.perf_counter()
    for index, archive_path in enumerate(paths, start=1):
        features_archive_path = None
        if features_root is not None:
            features_archive_path = features_root / archive_path.relative_to(input_root)
        object_pooled_archive_path = None
        if object_pooled_root is not None:
            object_pooled_archive_path = object_pooled_root / archive_path.relative_to(input_root)
        result = rerank_archive(
            archive_path,
            model,
            device=options["device"],
            features_archive_path=features_archive_path,
            object_pooled_archive_path=object_pooled_archive_path,
            require_object_pooled=bool(options.get("require_object_pooled", False)),
            rescoring_score_weight=float(options["rescoring_score_weight"]),
            score_normalization="zscore",
        )
        save_path = _resolve_output_path(
            output_root=output_root,
            input_root=input_root,
            archive_path=archive_path,
            meta=result["meta"],
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result["grasp_group_array"])
        accum.add(archive_path=archive_path, save_path=save_path, result=result, records_path=records_path)
        if index % 100 == 0 or index == len(paths):
            print(
                json.dumps(
                    {
                        "stage": "rerank_worker_progress",
                        "worker": worker_id,
                        "processed": index,
                        "total": len(paths),
                        "output_count": accum.num_files,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = accum.to_summary()
    summary["runtime_sec"] = time.perf_counter() - started
    return summary


class _RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.finite_count = 0
        self.nonfinite_count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min: float | None = None
        self.max: float | None = None

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        self.count += int(arr.size)
        finite = arr[np.isfinite(arr)]
        self.finite_count += int(finite.size)
        self.nonfinite_count += int(arr.size - finite.size)
        if finite.size == 0:
            return
        local_min = float(np.min(finite))
        local_max = float(np.max(finite))
        self.min = local_min if self.min is None else min(self.min, local_min)
        self.max = local_max if self.max is None else max(self.max, local_max)
        self.sum += float(np.sum(finite, dtype=np.float64))
        self.sumsq += float(np.sum(finite * finite, dtype=np.float64))

    def merge(self, other: dict[str, Any]) -> None:
        self.count += int(other.get("count", 0))
        self.finite_count += int(other.get("finite_count", 0))
        self.nonfinite_count += int(other.get("nonfinite_count", 0))
        if other.get("min") is not None:
            value = float(other["min"])
            self.min = value if self.min is None else min(self.min, value)
        if other.get("max") is not None:
            value = float(other["max"])
            self.max = value if self.max is None else max(self.max, value)
        self.sum += float(other.get("sum", 0.0))
        self.sumsq += float(other.get("sumsq", 0.0))

    def to_raw(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "min": self.min,
            "max": self.max,
            "sum": self.sum,
            "sumsq": self.sumsq,
        }

    def to_numeric_stats(self) -> dict[str, Any]:
        if self.finite_count == 0:
            return {
                "count": self.count,
                "finite_count": self.finite_count,
                "nonfinite_count": self.nonfinite_count,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "p05": None,
                "p50": None,
                "p95": None,
            }
        mean = self.sum / self.finite_count
        variance = max(self.sumsq / self.finite_count - mean * mean, 0.0)
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "p05": None,
            "p50": None,
            "p95": None,
        }


class _WorkerAccumulator:
    def __init__(self, *, worker_id: int) -> None:
        self.worker_id = worker_id
        self.num_files = 0
        self.num_grasps_total = 0
        self.num_grasps_parts: list[int] = []
        self.candidate_count_parts: list[int] = []
        self.top1_rescoring_scores: list[float] = []
        self.top1_exported_scores: list[float] = []
        self.top1_model_scores_raw: list[float] = []
        self.top1_base_scores_after: list[float] = []
        self.top1_base_scores_before: list[float] = []
        self.top1_changed_vs_base_count = 0
        self.sort_descending: bool | None = None
        self.per_scene: dict[str, dict[str, int]] = {}
        self.score_stats = {
            "exported_scores": _RunningStats(),
            "rescoring_scores": _RunningStats(),
            "model_scores_raw": _RunningStats(),
            "base_scores_after_rerank": _RunningStats(),
        }

    def add(
        self,
        *,
        archive_path: Path,
        save_path: Path,
        result: dict[str, Any],
        records_path: Path | None,
    ) -> None:
        exported_scores = result["exported_scores"].astype(np.float32, copy=False)
        rescoring_scores = result["rescoring_scores"].astype(np.float32, copy=False)
        model_scores_raw = result["model_scores_raw"].astype(np.float32, copy=False)
        base_scores = result["base_scores"].astype(np.float32, copy=False)
        base_scores_input = result["base_scores_input"].astype(np.float32, copy=False)
        if self.sort_descending is None:
            self.sort_descending = bool(result.get("sort_descending", True))
        meta = result["meta"]
        scene_name = f"scene_{int(meta['scene_id']):04d}"
        per_scene_stats = self.per_scene.setdefault(scene_name, {"num_files": 0, "num_grasps": 0})
        per_scene_stats["num_files"] += 1
        per_scene_stats["num_grasps"] += int(len(rescoring_scores))

        self.num_files += 1
        self.num_grasps_total += int(len(rescoring_scores))
        self.num_grasps_parts.append(int(len(rescoring_scores)))
        self.candidate_count_parts.append(int(result.get("candidate_count", len(rescoring_scores))))
        self.score_stats["exported_scores"].add(exported_scores)
        self.score_stats["rescoring_scores"].add(rescoring_scores)
        self.score_stats["model_scores_raw"].add(model_scores_raw)
        self.score_stats["base_scores_after_rerank"].add(base_scores)
        if bool(result.get("top1_changed_vs_base", False)):
            self.top1_changed_vs_base_count += 1
        if len(rescoring_scores) > 0:
            self.top1_exported_scores.append(float(exported_scores[0]))
            self.top1_rescoring_scores.append(float(rescoring_scores[0]))
            self.top1_model_scores_raw.append(float(model_scores_raw[0]))
            self.top1_base_scores_after.append(float(base_scores[0]))
            self.top1_base_scores_before.append(float(np.max(base_scores_input)))
        if records_path is not None:
            append_jsonl(
                records_path,
                {
                    "worker": self.worker_id,
                    "scene_id": int(meta["scene_id"]),
                    "frame_id": int(meta["frame_id"]),
                    "scene_name": scene_name,
                    "archive_path": str(archive_path),
                    "output_path": str(save_path),
                    "num_grasps": int(len(rescoring_scores)),
                    "exported_scores": numeric_stats(exported_scores),
                    "rescoring_scores": numeric_stats(rescoring_scores),
                    "model_scores_raw": numeric_stats(model_scores_raw),
                    "base_scores_after_rerank": numeric_stats(base_scores),
                    "base_scores_before_rerank": numeric_stats(base_scores_input),
                    "top1_exported_score": float(exported_scores[0]) if len(exported_scores) > 0 else None,
                    "top1_rescoring_score": float(rescoring_scores[0]) if len(rescoring_scores) > 0 else None,
                    "top1_model_score_raw": float(model_scores_raw[0]) if len(model_scores_raw) > 0 else None,
                    "top1_base_score_before": float(np.max(base_scores_input)) if len(base_scores_input) > 0 else None,
                    "top1_base_score_after": float(base_scores[0]) if len(base_scores) > 0 else None,
                    "score_normalization": result.get("score_normalization"),
                    "rescoring_score_weight": result.get("rescoring_score_weight"),
                    "candidate_count": result.get("candidate_count"),
                    "base_top1_idx": result.get("base_top1_idx"),
                    "proposed_top1_idx": result.get("proposed_top1_idx"),
                    "final_top1_idx": result.get("final_top1_idx"),
                    "top1_changed_vs_base": bool(result.get("top1_changed_vs_base", False)),
                    "score_column_index": result.get("score_column_index"),
                    "score_column_overwritten": bool(result.get("score_column_overwritten", False)),
                },
            )

    def to_summary(self) -> dict[str, Any]:
        return {
            "worker": self.worker_id,
            "num_files": self.num_files,
            "num_scenes": len(self.per_scene),
            "num_grasps_total": self.num_grasps_total,
            "num_grasps_parts": self.num_grasps_parts,
            "candidate_count_parts": self.candidate_count_parts,
            "top1_rescoring_scores": self.top1_rescoring_scores,
            "top1_exported_scores": self.top1_exported_scores,
            "top1_model_scores_raw": self.top1_model_scores_raw,
            "top1_base_scores_after": self.top1_base_scores_after,
            "top1_base_scores_before": self.top1_base_scores_before,
            "top1_changed_vs_base_count": self.top1_changed_vs_base_count,
            "sort_descending": self.sort_descending,
            "per_scene": self.per_scene,
            "score_stats": {key: stats.to_raw() for key, stats in self.score_stats.items()},
        }


def _merge_worker_summaries(
    *,
    worker_results: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: str,
    started_perf: float,
    rescoring_score_weight: float,
    scene_name_filter: set[str] | None,
) -> dict[str, Any]:
    score_stats = {
        "exported_scores": _RunningStats(),
        "rescoring_scores": _RunningStats(),
        "model_scores_raw": _RunningStats(),
        "base_scores_after_rerank": _RunningStats(),
    }
    per_scene: dict[str, dict[str, int]] = {}
    num_grasps_parts: list[int] = []
    candidate_count_parts: list[int] = []
    top1_rescoring_scores: list[float] = []
    top1_exported_scores: list[float] = []
    top1_model_scores_raw: list[float] = []
    top1_base_scores_after: list[float] = []
    top1_base_scores_before: list[float] = []
    counters = {
        "num_files": 0,
        "num_grasps_total": 0,
        "top1_changed_vs_base_count": 0,
    }
    sort_descending: bool | None = None
    for result in worker_results:
        counters["num_files"] += int(result["num_files"])
        counters["num_grasps_total"] += int(result["num_grasps_total"])
        for key in counters:
            if key in {"num_files", "num_grasps_total"}:
                continue
            counters[key] += int(result.get(key, 0))
        num_grasps_parts.extend(result.get("num_grasps_parts", []))
        candidate_count_parts.extend(result.get("candidate_count_parts", []))
        top1_rescoring_scores.extend(result.get("top1_rescoring_scores", []))
        top1_exported_scores.extend(result.get("top1_exported_scores", []))
        top1_model_scores_raw.extend(result.get("top1_model_scores_raw", []))
        top1_base_scores_after.extend(result.get("top1_base_scores_after", []))
        top1_base_scores_before.extend(result.get("top1_base_scores_before", []))
        if sort_descending is None and result.get("sort_descending") is not None:
            sort_descending = bool(result["sort_descending"])
        for scene_name, stats in result.get("per_scene", {}).items():
            dst = per_scene.setdefault(scene_name, {"num_files": 0, "num_grasps": 0})
            dst["num_files"] += int(stats.get("num_files", 0))
            dst["num_grasps"] += int(stats.get("num_grasps", 0))
        for key, stats in result.get("score_stats", {}).items():
            if key in score_stats:
                score_stats[key].merge(stats)

    summary = {
        "stage": "rerank",
        "parallel": True,
        "num_workers": int(args.num_workers),
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "input_root": str(Path(args.input_root).resolve()),
        "camera": args.camera,
        "output_root": str(Path(args.output_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "features_root": None if args.features_root is None else str(Path(args.features_root).resolve()),
        "object_pooled_root": None if args.object_pooled_root is None else str(Path(args.object_pooled_root).resolve()),
        "require_object_pooled": bool(args.require_object_pooled),
        "scene_list_json": None if args.scene_list_json is None else str(Path(args.scene_list_json).resolve()),
        "scene_name_filter": None if scene_name_filter is None else sorted(scene_name_filter),
        "device": args.device,
        "score_normalization": "zscore",
        "rescoring_score_weight": rescoring_score_weight,
        "save_records_path": None if args.save_records_path is None else str(Path(args.save_records_path).resolve()),
        "sort_descending": sort_descending,
        "score_column_index": 0,
        "score_column_overwritten": True,
        "num_files": counters["num_files"],
        "num_scenes": len(per_scene),
        "num_grasps_total": counters["num_grasps_total"],
        "num_grasps_per_file": numeric_stats(num_grasps_parts),
        "exported_scores": score_stats["exported_scores"].to_numeric_stats(),
        "rescoring_scores": score_stats["rescoring_scores"].to_numeric_stats(),
        "model_scores_raw": score_stats["model_scores_raw"].to_numeric_stats(),
        "base_scores_after_rerank": score_stats["base_scores_after_rerank"].to_numeric_stats(),
        "top1_exported_scores": numeric_stats(top1_exported_scores),
        "top1_rescoring_scores": numeric_stats(top1_rescoring_scores),
        "top1_model_scores_raw": numeric_stats(top1_model_scores_raw),
        "top1_base_scores_before_rerank": numeric_stats(top1_base_scores_before),
        "top1_base_scores_after_rerank": numeric_stats(top1_base_scores_after),
        "top1_changed_vs_base_count": counters["top1_changed_vs_base_count"],
        "candidate_count_per_file": numeric_stats(candidate_count_parts),
        "per_scene": per_scene,
        "worker_summaries": [
            {
                "worker": int(result["worker"]),
                "num_files": int(result["num_files"]),
                "num_scenes": int(result["num_scenes"]),
                "num_grasps_total": int(result["num_grasps_total"]),
                "runtime_sec": float(result["runtime_sec"]),
                "per_scene": result["per_scene"],
            }
            for result in sorted(worker_results, key=lambda item: int(item["worker"]))
        ],
    }
    return summary


def _shard_paths(paths: list[Path], workers: int) -> list[list[Path]]:
    shards = [[] for _ in range(workers)]
    for index, path in enumerate(paths):
        shards[index % workers].append(path)
    return shards


def _resolve_worker_devices(args: argparse.Namespace, workers: int) -> list[str]:
    """Build the per-worker device list for parallel rerank.

    With --devices='cuda:0,cuda:1' workers are round-robin assigned. Without
    --devices the existing single-device flag is broadcast, preserving
    backward-compatible behaviour when only one GPU is available.
    """
    raw = getattr(args, "devices", None)
    if not raw:
        return [args.device] * max(1, workers)
    devices = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not devices:
        return [args.device] * max(1, workers)
    return devices


def _worker_records_path(path: str | None, worker_id: int) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p.with_name(f"{p.stem}.worker{worker_id}{p.suffix}")


def _merge_worker_record_files(path: Path, *, worker_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for worker_id in range(int(worker_count)):
            worker_path = _worker_records_path(str(path), worker_id)
            if worker_path is None or not worker_path.is_file():
                continue
            with worker_path.open("r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)
            worker_path.unlink()

def _resolve_output_path(
    *,
    output_root: Path,
    input_root: Path,
    archive_path: Path,
    meta: dict[str, object],
) -> Path:
    benchmark = str(meta.get("benchmark", "") or "")
    camera = str(meta.get("camera", archive_path.parent.name) or archive_path.parent.name)
    scene_id = meta.get("scene_id")
    frame_id = meta.get("frame_id")
    if scene_id is not None and frame_id is not None:
        frame_width = 6 if benchmark == "gc6d" else 4
        return (
            output_root
            / f"scene_{int(scene_id):04d}"
            / camera
            / f"{int(frame_id):0{frame_width}d}.npy"
        )
    return output_root / archive_path.relative_to(input_root).with_suffix(".npy")


def _load_scene_name_filter(scene_list_json: str | None) -> set[str] | None:
    if scene_list_json is None:
        return None
    payload = json.loads(Path(scene_list_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {scene_list_json}")
    return {str(item) for item in payload}


def _archive_scene_name(input_root: Path, archive_path: Path) -> str:
    rel_path = archive_path.relative_to(input_root)
    if len(rel_path.parts) < 2:
        raise ValueError(f"Unexpected archive layout under {input_root}: {archive_path}")
    return rel_path.parts[0]


if __name__ == "__main__":
    raise SystemExit(main())
