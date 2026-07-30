from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

MANIFEST_FILENAME = "manifest.jsonl"
MANIFEST_SUMMARY_FILENAME = "manifest.summary.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestBuildResult:
    manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]


def default_manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_FILENAME


def default_manifest_summary_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_SUMMARY_FILENAME


def build_archive_manifest(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    pattern: str = "**/*.npz",
    success_mu_thresh: float = 0.4,
    num_workers: int = 1,
    show_progress: bool = False,
) -> ManifestBuildResult:
    root = Path(root)
    manifest_path = Path(manifest_path) if manifest_path is not None else default_manifest_path(root)
    summary_path = Path(summary_path) if summary_path is not None else default_manifest_summary_path(root)
    archive_paths = sorted(root.glob(pattern))
    tasks = [(str(root), str(path.relative_to(root)), float(success_mu_thresh)) for path in archive_paths]

    if num_workers > 1 and len(tasks) > 1:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=min(int(num_workers), len(tasks))) as pool:
            iterator = pool.imap_unordered(_record_from_task, tasks, chunksize=max(1, len(tasks) // (num_workers * 8)))
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="manifest", unit="archive", dynamic_ncols=True)
            records = list(iterator)
    else:
        iterator = tasks
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="manifest", unit="archive", dynamic_ncols=True)
        records = [_record_from_task(task) for task in iterator]

    records.sort(key=lambda record: str(record["relative_path"]))
    summary = summarize_manifest_records(records, root=root, success_mu_thresh=success_mu_thresh)
    write_manifest(manifest_path, records)
    write_manifest_summary(summary_path, summary)
    return ManifestBuildResult(manifest_path=manifest_path, summary_path=summary_path, summary=summary)


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_no}: unsupported manifest schema_version={record.get('schema_version')!r}")
            records.append(record)
    return records


def write_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def write_manifest_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def summarize_manifest_records(
    records: Iterable[dict[str, Any]],
    *,
    root: str | Path | None = None,
    success_mu_thresh: float | None = None,
) -> dict[str, Any]:
    records = list(records)
    total_grasps = sum(int(record.get("num_grasps", 0)) for record in records)
    success_count = sum(int(record.get("success_count", 0)) for record in records)
    collision_count = sum(int(record.get("collision_count", 0)) for record in records)
    empty_count = sum(int(record.get("empty_count", 0)) for record in records)
    finite_mu_count = sum(int(record.get("mu_min_finite_count", 0)) for record in records)
    finite_mu_sum = sum(float(record.get("mu_min_finite_sum", 0.0)) for record in records)
    finite_mu_sumsq = sum(float(record.get("mu_min_finite_sumsq", 0.0)) for record in records)

    per_scene: dict[str, dict[str, int]] = {}
    for record in records:
        scene_name = str(record.get("scene_name") or "unknown")
        scene = per_scene.setdefault(
            scene_name,
            {
                "num_archives": 0,
                "num_grasps": 0,
                "success_count": 0,
                "collision_count": 0,
                "empty_count": 0,
            },
        )
        scene["num_archives"] += 1
        scene["num_grasps"] += int(record.get("num_grasps", 0))
        scene["success_count"] += int(record.get("success_count", 0))
        scene["collision_count"] += int(record.get("collision_count", 0))
        scene["empty_count"] += int(record.get("empty_count", 0))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "root": None if root is None else str(Path(root).resolve()),
        "success_mu_thresh": success_mu_thresh,
        "num_archives": len(records),
        "num_scenes": len(per_scene),
        "num_grasps_total": int(total_grasps),
        "success_count": int(success_count),
        "negative_count": int(total_grasps - success_count),
        "collision_count": int(collision_count),
        "empty_count": int(empty_count),
        "mu_min_finite_count": int(finite_mu_count),
        "mu_min_nonfinite_count": int(total_grasps - finite_mu_count),
        "mu_min_finite_sum": float(finite_mu_sum),
        "mu_min_finite_sumsq": float(finite_mu_sumsq),
        "per_scene": per_scene,
    }


def manifest_records_by_archive_path(
    records: Iterable[dict[str, Any]],
    *,
    root: str | Path,
) -> dict[Path, dict[str, Any]]:
    root = Path(root)
    return {(root / str(record["relative_path"])).resolve(): record for record in records}


def validate_manifest_coverage(
    records_by_path: dict[Path, dict[str, Any]],
    archive_paths: Iterable[str | Path],
    *,
    success_mu_thresh: float | None = None,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for archive_path in archive_paths:
        path = Path(archive_path).resolve()
        record = records_by_path.get(path)
        if record is None:
            problems.append(f"missing manifest record: {path}")
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            problems.append(f"archive missing on disk: {path}")
            continue
        recorded_size = record.get("archive_size")
        recorded_mtime_ns = record.get("archive_mtime_ns")
        if recorded_size is not None and int(recorded_size) != int(stat.st_size):
            problems.append(f"stale manifest size: {path}")
        if recorded_mtime_ns is not None and int(recorded_mtime_ns) != int(stat.st_mtime_ns):
            problems.append(f"stale manifest mtime: {path}")
        recorded_thresh = record.get("success_mu_thresh")
        if success_mu_thresh is not None and recorded_thresh is not None:
            if abs(float(recorded_thresh) - float(success_mu_thresh)) > 1e-12:
                problems.append(
                    f"manifest success_mu_thresh mismatch: {path} "
                    f"record={recorded_thresh} expected={success_mu_thresh}"
                )
    return not problems, problems


def _record_from_task(task: tuple[str, str, float]) -> dict[str, Any]:
    root_str, relative_str, success_mu_thresh = task
    root = Path(root_str)
    relative_path = Path(relative_str)
    archive_path = root / relative_path
    stat = archive_path.stat()
    with np.load(archive_path, allow_pickle=True) as archive:
        base_scores = archive["base_scores"].astype(np.float64, copy=False)
        if "mu_min" in archive.files:
            mu_min = archive["mu_min"].astype(np.float64, copy=False)
        else:
            mu_min = np.full((len(base_scores),), np.inf, dtype=np.float64)
        if "is_collision" in archive.files:
            is_collision = archive["is_collision"].astype(bool, copy=False)
        else:
            is_collision = np.zeros((len(base_scores),), dtype=bool)
        if "is_empty" in archive.files:
            is_empty = archive["is_empty"].astype(bool, copy=False)
        else:
            is_empty = np.zeros((len(base_scores),), dtype=bool)
        meta = _load_meta(archive)

    num_grasps = int(len(base_scores))
    finite_mu = np.isfinite(mu_min)
    success = finite_mu & (mu_min <= float(success_mu_thresh))
    finite_values = mu_min[finite_mu]
    scene_name = _scene_name_from_meta_or_path(meta, relative_path)
    frame_id = _frame_id_from_meta_or_path(meta, relative_path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "relative_path": relative_path.as_posix(),
        "archive_size": int(stat.st_size),
        "archive_mtime_ns": int(stat.st_mtime_ns),
        "scene_name": scene_name,
        "frame_id": frame_id,
        "detector": meta.get("detector"),
        "benchmark": meta.get("benchmark"),
        "split": meta.get("split"),
        "camera": meta.get("camera"),
        "num_grasps": num_grasps,
        "success_mu_thresh": float(success_mu_thresh),
        "success_count": int(np.count_nonzero(success)),
        "collision_count": int(np.count_nonzero(is_collision)),
        "empty_count": int(np.count_nonzero(is_empty)),
        "mu_min_finite_count": int(finite_values.size),
        "mu_min_nonfinite_count": int(num_grasps - finite_values.size),
        "mu_min_finite_sum": float(np.sum(finite_values)) if finite_values.size else 0.0,
        "mu_min_finite_sumsq": float(np.sum(finite_values * finite_values)) if finite_values.size else 0.0,
        "base_score_sum": float(np.sum(base_scores)) if base_scores.size else 0.0,
        "base_score_sumsq": float(np.sum(base_scores * base_scores)) if base_scores.size else 0.0,
    }


def _load_meta(npz_file: np.lib.npyio.NpzFile) -> dict[str, Any]:
    meta_json = npz_file.get("meta_json")
    if meta_json is None:
        return {}
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    return json.loads(str(meta_json))


def _scene_name_from_meta_or_path(meta: dict[str, Any], relative_path: Path) -> str:
    scene_id = meta.get("scene_id")
    if scene_id is not None:
        return f"scene_{int(scene_id):04d}"
    for part in relative_path.parts:
        if part.startswith("scene_"):
            return part
    if len(relative_path.parts) >= 3:
        return relative_path.parts[-3]
    return "unknown"


def _frame_id_from_meta_or_path(meta: dict[str, Any], relative_path: Path) -> int | None:
    frame_id = meta.get("frame_id")
    if frame_id is not None:
        return int(frame_id)
    try:
        return int(relative_path.stem)
    except ValueError:
        return None
