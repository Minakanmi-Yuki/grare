#!/usr/bin/env python3
"""Apply analytic relabeling to candidate archives (GraspNet1B)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from grare.utils.experiment_logging import (
    append_jsonl,
    boolean_stats,
    numeric_stats,
    reset_jsonl,
    timestamp,
    write_json,
)
from grare.relabeling.manifest import build_archive_manifest
from grare.relabeling.archive_io import normalize_archive_format
from grare.utils.cpu import effective_cpu_count


def _scene_name_from_archive_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("scene_"):
            return part
    return path.parent.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply analytic relabeling to candidate archives.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--benchmark", default="graspnet")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--stage",
        choices=("all", "labels", "object"),
        default="all",
        help=(
            "all: analytic labels, local geometry, and the SAM object cloud in one pass. "
            "labels: the CPU-bound labels and local geometry only, skipping SAM. "
            "object: add the GPU-bound SAM object cloud to archives written by --stage labels. "
            "Splitting the two lets each run at its own --num-workers, since one is CPU-bound "
            "and the other is GPU-bound."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=("candidate-archive", "detector-dump", "auto"),
        default="candidate-archive",
        help="candidate-archive reads prebuilt .npz files; detector-dump reads raw detector .npy files directly.",
    )
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-cloud-radius", type=float, default=0.04)
    parser.add_argument("--local-cloud-max-points", type=int, default=512)
    parser.add_argument(
        "--cloud-sampler",
        choices=("stratified_fps",),
        default="stratified_fps",
        help="Stratified-FPS over shell-radius buckets.",
    )
    parser.add_argument(
        "--shell-edges",
        type=str,
        default="0.0,0.005,0.015,0.025,0.040",
        help="Comma-separated shell radius edges in meters.",
    )
    parser.add_argument(
        "--shell-budgets",
        type=str,
        default="64,128,128,192",
        help="Comma-separated per-shell budgets (sum overrides --local-cloud-max-points).",
    )
    parser.add_argument("--voxel-size", type=float, default=0.008)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-mu-thresh", type=float, default=0.4)
    parser.add_argument("--object-cloud-points", type=int, default=512)
    parser.add_argument("--sam-checkpoint", type=str, default=None,
        help="Path to MobileSAM checkpoint. When unset, the SAM-prompted "
             "object cloud is disabled and object_cloud is emitted as zeros.")
    parser.add_argument("--sam-model-type", default="vit_t",
        choices=("vit_t", "vit_b", "vit_l", "vit_h"))
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--sam-cluster-radius-m", type=float, default=0.03,
        help="3D radius for greedy candidate clustering. Candidates within this "
             "distance share a SAM mask to amortise inference cost.")
    parser.add_argument("--sam-min-area-pixels", type=int, default=200)
    parser.add_argument("--sam-max-area-ratio", type=float, default=0.4)
    parser.add_argument("--sam-iou-score-floor", type=float, default=0.0)
    parser.add_argument("--sam-multimask-pick", default="smallest_valid",
        choices=("smallest_valid", "best_iou"))
    parser.add_argument(
        "--sam-prompt-batch-size",
        type=int,
        default=int(os.environ.get("GRARE_SAM_PROMPT_BATCH_SIZE", "64")),
        help="Number of point prompts decoded per MobileSAM mask-decoder batch.",
    )
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--save-records-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--manifest-summary-path", default=None)
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument(
        "--flatten-camera-dir",
        action="store_true",
        help="Drop scene_XXXX/<camera>/ from output relative paths so archives use scene_XXXX/FFFF.npz.",
    )
    parser.add_argument("--object-cloud-root", default=None,
        help="Optional sidecar root for object_cloud files with the same relative layout as --output-root.")
    parser.add_argument("--omit-object-cloud", action="store_true",
        help="Do not store object_cloud in the main output archive. If --object-cloud-root is also set, "
             "object_cloud is written there as a sidecar; otherwise this is a local-only archive.")
    parser.add_argument(
        "--archive-format",
        choices=("compressed", "stored"),
        default="compressed",
        help="Output npz storage. 'stored' skips zlib compression for faster training reads.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=("quick", "full"),
        default="quick",
        help="quick counts output archives only; full scans every output archive for dataset statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.flatten_camera_dir:
        os.environ["GRARE_FLATTEN_CAMERA_DIR"] = "1"
    if args.benchmark != "graspnet":
        raise SystemExit(f"grare supports only benchmark='graspnet'; got {args.benchmark!r}")
    from grare.relabeling.scene_labeling import (
        BatchAnalyticRelabeler,
        SamObjectCloudConfig,
        SceneLabelingConfig,
    )

    os.environ["GRARE_ARCHIVE_FORMAT"] = normalize_archive_format(args.archive_format)
    started_at = timestamp()
    started_perf = time.perf_counter()
    input_format = _resolve_input_format(args.input_root, args.input_format)
    input_pattern = args.pattern or ("**/*.npy" if input_format == "detector-dump" else "**/*.npz")
    output_pattern = _output_pattern_for_input_pattern(input_pattern)
    shell_edges = tuple(float(s) for s in args.shell_edges.split(","))
    shell_budgets = tuple(int(s) for s in args.shell_budgets.split(","))
    # --stage labels is the CPU-bound pass: keep SAM off even when a checkpoint is
    # supplied, so it can run at a high --num-workers without touching the GPU.
    sam_enabled = bool(args.sam_checkpoint) and args.stage != "labels"
    if args.stage == "object" and not sam_enabled:
        raise SystemExit("--stage object requires --sam-checkpoint")
    sam_cfg = SamObjectCloudConfig(
        enabled=sam_enabled,
        checkpoint=str(args.sam_checkpoint or ""),
        model_type=str(args.sam_model_type),
        device=str(args.sam_device),
        multimask_pick=str(args.sam_multimask_pick),
        min_area_pixels=int(args.sam_min_area_pixels),
        max_area_ratio=float(args.sam_max_area_ratio),
        iou_score_floor=float(args.sam_iou_score_floor),
        cluster_radius_m=float(args.sam_cluster_radius_m),
        prompt_batch_size=int(args.sam_prompt_batch_size),
    )
    effective_num_workers = _effective_prepare_workers(
        args.num_workers,
        stage=args.stage,
        sam_device=args.sam_device,
        sam_enabled=sam_enabled,
    )
    relabeler = BatchAnalyticRelabeler(
        SceneLabelingConfig(
            detector=args.detector,
            benchmark=args.benchmark,
            split=args.split,
            camera=args.camera,
            dataset_root=args.dataset_root,
            local_cloud_radius=args.local_cloud_radius,
            local_cloud_max_points=args.local_cloud_max_points,
            voxel_size=args.voxel_size,
            cloud_sampler=args.cloud_sampler,
            shell_edges_m=shell_edges,
            shell_budgets=shell_budgets,
            object_cloud_points=int(args.object_cloud_points),
            sam=sam_cfg,
        )
    )
    if args.stage == "object":
        # GPU stage: add the SAM object cloud to archives that already carry the
        # analytic labels and local geometry, without redoing the CPU work.
        summary = relabeler.augment_tree(
            args.input_root,
            args.output_root,
            pattern=input_pattern,
            limit=args.limit,
            skip_existing=not args.overwrite,
            num_workers=effective_num_workers,
            object_cloud_root=args.object_cloud_root,
            include_object_cloud=not args.omit_object_cloud,
        )
    elif input_format == "detector-dump":
        summary = relabeler.relabel_dump_tree(
            args.input_root,
            args.output_root,
            pattern=input_pattern,
            limit=args.limit,
            skip_existing=not args.overwrite,
            num_workers=effective_num_workers,
            object_cloud_root=args.object_cloud_root,
            include_object_cloud=not args.omit_object_cloud,
        )
    else:
        summary = relabeler.relabel_tree(
            args.input_root,
            args.output_root,
            pattern=input_pattern,
            limit=args.limit,
            skip_existing=not args.overwrite,
            num_workers=effective_num_workers,
        )
    manifest_payload = None
    if not args.no_manifest:
        manifest = build_archive_manifest(
            args.output_root,
            manifest_path=args.manifest_path,
            summary_path=args.manifest_summary_path,
            pattern=output_pattern,
            success_mu_thresh=args.success_mu_thresh,
            num_workers=max(1, min(effective_num_workers, 16)),
            show_progress=True,
        )
        manifest_payload = {
            "manifest_path": str(manifest.manifest_path),
            "summary_path": str(manifest.summary_path),
            "summary": manifest.summary,
        }
    if args.summary_mode == "full" or args.save_records_path:
        detailed_summary = _summarize_relabeled_outputs(
            output_root=args.output_root,
            pattern=output_pattern,
            success_mu_thresh=args.success_mu_thresh,
            save_records_path=args.save_records_path,
        )
    else:
        detailed_summary = _quick_summary_outputs(args.output_root, output_pattern)
    payload = {
        "stage": "relabel",
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "input_root": str(Path(args.input_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "object_cloud_root": None if args.object_cloud_root is None else str(Path(args.object_cloud_root).resolve()),
        "omit_object_cloud": bool(args.omit_object_cloud),
        "detector": args.detector,
        "benchmark": args.benchmark,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "camera": args.camera,
        "split": args.split,
        "input_format": input_format,
        "pattern": input_pattern,
        "output_pattern": output_pattern,
        "limit": args.limit,
        "num_workers_requested": args.num_workers,
        "num_workers": effective_num_workers,
        "overwrite": bool(args.overwrite),
        "archive_format": normalize_archive_format(args.archive_format),
        "summary_mode": args.summary_mode,
        "success_mu_thresh": args.success_mu_thresh,
        "sam_prompt_batch_size": int(args.sam_prompt_batch_size),
        "tree_summary": summary,
        "manifest": manifest_payload,
        "detailed_summary": detailed_summary,
    }
    if args.save_path:
        write_json(args.save_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _resolve_input_format(input_root: str | Path, requested: str) -> str:
    if requested != "auto":
        return requested
    root = Path(input_root)
    if any(root.glob("**/*.npz")):
        return "candidate-archive"
    if any(root.glob("**/*.npy")):
        return "detector-dump"
    return "candidate-archive"


def _output_pattern_for_input_pattern(pattern: str) -> str:
    if pattern.endswith(".npy"):
        return f"{pattern[:-4]}.npz"
    return pattern


def _visible_cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [token.strip() for token in visible.split(",") if token.strip()]
        if devices:
            return len(devices)
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _label_worker_cap() -> int:
    """Memory-safe default for GraspNet/Dex-Net relabel workers.

    A relabel worker owns a GraspNetEval instance and a scene/Dex-Net cache;
    CPU quota alone is therefore not a safe upper bound. The cap can be raised
    deliberately for a larger-memory host with GRARE_PREPARE_MAX_LABEL_WORKERS.
    """
    raw = os.environ.get("GRARE_PREPARE_MAX_LABEL_WORKERS", "20")
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


def _effective_prepare_workers(requested: int, *, stage: str, sam_device: str, sam_enabled: bool) -> int:
    """Prevent process oversubscription and one-GPU MobileSAM replication."""
    requested = max(1, int(requested))
    if sam_enabled and str(sam_device).strip().lower().startswith("cuda"):
        # Each worker owns a complete MobileSAM model and CUDA context. More
        # workers than visible GPUs replicate the model and commonly OOM or
        # deadlock on a single-GPU container.
        limit = max(1, _visible_cuda_device_count())
        reason = "visible GPU count for CUDA-backed SAM"
    else:
        limit = max(1, effective_cpu_count())
        reason = "effective CPU quota"
        if stage == "labels":
            limit = min(limit, _label_worker_cap())
            reason += " and memory-safe labels cap"
    effective = min(requested, limit)
    if effective < requested:
        print(
            f"[prepare] limiting --num-workers from {requested} to {effective} "
            f"({reason}; stage={stage})",
            flush=True,
        )
    return effective


def _summarize_relabeled_outputs(
    output_root: str | Path,
    pattern: str,
    success_mu_thresh: float,
    save_records_path: str | None,
) -> dict[str, object]:
    output_root = Path(output_root)
    archive_paths = sorted(output_root.glob(pattern))
    if save_records_path:
        reset_jsonl(save_records_path)

    num_grasps_parts: list[int] = []
    base_score_parts: list[np.ndarray] = []
    mu_parts: list[np.ndarray] = []
    finite_mu_parts: list[np.ndarray] = []
    success_parts: list[np.ndarray] = []
    collision_parts: list[np.ndarray] = []
    empty_parts: list[np.ndarray] = []
    local_cloud_points_parts: list[np.ndarray] = []
    per_scene: dict[str, dict[str, int]] = {}

    for archive_path in archive_paths:
        with np.load(archive_path, allow_pickle=True) as archive:
            base_scores = archive["base_scores"].astype(np.float32, copy=False)
            mu_min = archive["mu_min"].astype(np.float32, copy=False)
            is_collision = archive["is_collision"].astype(bool, copy=False)
            is_empty = archive["is_empty"].astype(bool, copy=False)
            local_cloud = archive["local_cloud"].astype(np.float32, copy=False)
        finite_mu = np.isfinite(mu_min)
        success = finite_mu & (mu_min <= success_mu_thresh)
        local_cloud_points = np.count_nonzero(np.any(local_cloud != 0, axis=-1), axis=1).astype(np.int32, copy=False)

        scene_name = _scene_name_from_archive_path(archive_path)
        frame_id = int(archive_path.stem)
        per_scene_stats = per_scene.setdefault(
            scene_name,
            {
                "num_archives": 0,
                "num_grasps": 0,
                "success_count": 0,
                "collision_count": 0,
                "empty_count": 0,
            },
        )
        per_scene_stats["num_archives"] += 1
        per_scene_stats["num_grasps"] += int(len(base_scores))
        per_scene_stats["success_count"] += int(np.count_nonzero(success))
        per_scene_stats["collision_count"] += int(np.count_nonzero(is_collision))
        per_scene_stats["empty_count"] += int(np.count_nonzero(is_empty))

        num_grasps_parts.append(int(len(base_scores)))
        base_score_parts.append(base_scores)
        mu_parts.append(mu_min)
        finite_mu_parts.append(mu_min[finite_mu])
        success_parts.append(success.astype(bool, copy=False))
        collision_parts.append(is_collision)
        empty_parts.append(is_empty)
        local_cloud_points_parts.append(local_cloud_points)

        if save_records_path:
            append_jsonl(
                save_records_path,
                {
                    "scene_name": scene_name,
                    "frame_id": frame_id,
                    "archive_path": str(archive_path),
                    "num_grasps": int(len(base_scores)),
                    "base_scores": numeric_stats(base_scores),
                    "mu_min": numeric_stats(mu_min),
                    "mu_min_finite": numeric_stats(mu_min[finite_mu]),
                    "success": boolean_stats(success),
                    "collision": boolean_stats(is_collision),
                    "empty": boolean_stats(is_empty),
                    "local_cloud_points": numeric_stats(local_cloud_points),
                },
            )

    base_scores_all = np.concatenate(base_score_parts, axis=0) if base_score_parts else np.empty((0,), dtype=np.float32)
    mu_all = np.concatenate(mu_parts, axis=0) if mu_parts else np.empty((0,), dtype=np.float32)
    finite_mu_all = np.concatenate(finite_mu_parts, axis=0) if finite_mu_parts else np.empty((0,), dtype=np.float32)
    success_all = np.concatenate(success_parts, axis=0) if success_parts else np.empty((0,), dtype=bool)
    collision_all = np.concatenate(collision_parts, axis=0) if collision_parts else np.empty((0,), dtype=bool)
    empty_all = np.concatenate(empty_parts, axis=0) if empty_parts else np.empty((0,), dtype=bool)
    local_cloud_points_all = (
        np.concatenate(local_cloud_points_parts, axis=0)
        if local_cloud_points_parts
        else np.empty((0,), dtype=np.int32)
    )

    return {
        "num_archives": len(archive_paths),
        "num_scenes": len(per_scene),
        "num_grasps_total": int(sum(num_grasps_parts)),
        "num_grasps_per_archive": numeric_stats(num_grasps_parts),
        "base_scores": numeric_stats(base_scores_all),
        "mu_min": numeric_stats(mu_all),
        "mu_min_finite": numeric_stats(finite_mu_all),
        "success": boolean_stats(success_all),
        "collision": boolean_stats(collision_all),
        "empty": boolean_stats(empty_all),
        "local_cloud_points": numeric_stats(local_cloud_points_all),
        "per_scene": per_scene,
    }


def _quick_summary_outputs(output_root: str | Path, pattern: str) -> dict[str, object]:
    output_root = Path(output_root)
    archive_paths = sorted(output_root.glob(pattern))
    per_scene: dict[str, int] = {}
    total_bytes = 0
    for archive_path in archive_paths:
        total_bytes += archive_path.stat().st_size
        scene_name = _scene_name_from_archive_path(archive_path)
        per_scene[scene_name] = per_scene.get(scene_name, 0) + 1
    return {
        "summary_mode": "quick",
        "num_archives": len(archive_paths),
        "num_scenes": len(per_scene),
        "bytes_total": total_bytes,
        "per_scene_num_archives": per_scene,
    }


if __name__ == "__main__":
    raise SystemExit(main())
