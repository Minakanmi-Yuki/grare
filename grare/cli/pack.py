#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from collections.abc import Sequence

import numpy as np

from grare.relabeling.dataset_builder import (
    _archive_sample_count,
    _copy_payload_slice,
    _load_archive_payload,
    _object_pooled_path_for_archive,
)
from grare.relabeling.manifest import (
    default_manifest_path,
    load_manifest_records,
    manifest_records_by_archive_path,
    validate_manifest_coverage,
)
from grare.utils.benchmark_protocol import filter_archive_paths_by_camera
from grare.utils.experiment_logging import timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack relabeled local_cloud/object_pooled archives into read-only "
            "NPY arrays for mmap-backed training."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        nargs="+",
        help="One or more relabeled local_cloud roots to pack.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Optional camera filter, e.g. kinect or realsense. Archives from other cameras are ignored.",
    )
    parser.add_argument("--output-root", required=True, help="Packed dataset output directory.")
    parser.add_argument(
        "--object-pooled-root",
        nargs="+",
        default=None,
        help="Optional object_pooled sidecar root(s), one per input root or one shared root.",
    )
    parser.add_argument(
        "--require-object-pooled",
        action="store_true",
        help="Fail if an input archive is missing an object_pooled sidecar/cache.",
    )
    parser.add_argument(
        "--include-object-cloud",
        action="store_true",
        help="Also pack raw object_cloud. Omit for frozen-PMAE training.",
    )
    parser.add_argument("--success-mu-thresh", type=float, default=0.4)
    parser.add_argument(
        "--archive-manifest",
        nargs="+",
        default=None,
        help="Optional manifest.jsonl path(s). Use 'auto' for each input root.",
    )
    parser.add_argument(
        "--require-archive-manifest",
        action="store_true",
        help="Fail when an input root has no fresh manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing packed dataset directory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N non-empty archives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = timestamp()
    started_perf = time.perf_counter()
    input_roots = [Path(path) for path in args.input_root]
    object_pooled_roots = _normalize_optional_roots(
        args.object_pooled_root,
        input_roots,
        option_name="--object-pooled-root",
    )
    output_root = Path(args.output_root)
    arrays_root = output_root / "arrays"
    index_path = output_root / "index.json"
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        _remove_packed_outputs(output_root)
    arrays_root.mkdir(parents=True, exist_ok=True)

    records_by_root = _load_manifest_records_by_root(
        input_roots,
        args.archive_manifest,
        require_manifest=args.require_archive_manifest,
        success_mu_thresh=args.success_mu_thresh,
        camera=args.camera,
    )
    archive_entries = _collect_archive_entries(
        input_roots,
        records_by_root=records_by_root,
        require_manifest=args.require_archive_manifest,
        camera=args.camera,
    )
    if not archive_entries:
        raise FileNotFoundError(f"no relabeled .npz archives found under {input_roots}")
    counts = [entry["num_samples"] for entry in archive_entries]
    total_samples = int(sum(counts))
    if total_samples <= 0:
        raise ValueError("No relabeled entries were found")

    first_idx = next(idx for idx, count in enumerate(counts) if count > 0)
    first_entry = archive_entries[first_idx]
    first_payload = _load_pack_archive_payload(
        first_entry["path"],
        success_mu_thresh=float(args.success_mu_thresh),
        object_pooled_roots=object_pooled_roots,
        input_roots=input_roots,
        require_object_pooled=bool(args.require_object_pooled),
        include_object_cloud=bool(args.include_object_cloud),
    )
    if int(first_payload["base_score"].shape[0]) != int(first_entry["num_samples"]):
        raise ValueError(
            f"{first_entry['path']}: manifest/count hint says {first_entry['num_samples']} "
            f"samples but archive contains {first_payload['base_score'].shape[0]}"
        )

    arrays = _create_output_arrays(
        arrays_root,
        total_samples=total_samples,
        sample=first_payload,
    )
    archive_records: list[dict] = []
    loaded_archives = 0
    loaded_samples = 0

    for archive_idx, entry in enumerate(archive_entries):
        expected = int(entry["num_samples"])
        start = loaded_samples
        end = start + expected
        archive_record = {
            "input_root_index": int(entry["input_root_index"]),
            "relative_path": str(entry["relative_path"]),
            "absolute_path": str(Path(entry["path"]).resolve()),
            "start": int(start),
            "end": int(end),
            "num_samples": int(expected),
            "manifest_record": entry.get("manifest_record"),
        }
        if expected <= 0:
            archive_records.append(archive_record)
            continue

        payload = (
            first_payload
            if archive_idx == first_idx
            else _load_pack_archive_payload(
                entry["path"],
                success_mu_thresh=float(args.success_mu_thresh),
                object_pooled_roots=object_pooled_roots,
                input_roots=input_roots,
                require_object_pooled=bool(args.require_object_pooled),
                include_object_cloud=bool(args.include_object_cloud),
            )
        )
        n = int(payload["base_score"].shape[0])
        if n != expected:
            raise ValueError(
                f"{entry['path']}: manifest/count hint says {expected} samples but archive contains {n}"
            )
        _validate_payload_capacity(arrays, payload, archive_path=entry["path"])
        _clear_padded_slice(arrays, payload, start=start, end=end)
        _copy_payload_slice(arrays, payload, archive_idx=archive_idx, start=start, end=end)
        loaded_archives += 1
        loaded_samples += n
        archive_records.append(archive_record)
        if (
            loaded_archives == 1
            or loaded_archives == len([count for count in counts if count > 0])
            or (args.progress_every > 0 and loaded_archives % int(args.progress_every) == 0)
        ):
            print(
                json.dumps(
                    {
                        "stage": "pack_progress",
                        "archives": loaded_archives,
                        "archives_total": int(sum(1 for count in counts if count > 0)),
                        "samples": loaded_samples,
                        "samples_total": total_samples,
                        "output_root": str(output_root),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if loaded_samples != total_samples:
        raise RuntimeError(f"packed {loaded_samples} samples, expected {total_samples}")
    for array in arrays.values():
        if isinstance(array, np.memmap):
            array.flush()
    arrays_meta = _array_metadata(output_root, arrays)
    index = {
        "schema_version": 1,
        "created_at": timestamp(),
        "started_at": started_at,
        "runtime_sec": float(time.perf_counter() - started_perf),
        "input_roots": [str(path.resolve()) for path in input_roots],
        "camera": args.camera,
        "object_pooled_roots": (
            None
            if object_pooled_roots is None
            else [str(path.resolve()) for path in object_pooled_roots]
        ),
        "success_mu_thresh": float(args.success_mu_thresh),
        "require_object_pooled": bool(args.require_object_pooled),
        "include_object_cloud": bool(args.include_object_cloud),
        "num_archives": len(archive_records),
        "num_samples": int(total_samples),
        "arrays": arrays_meta,
        "archives": archive_records,
    }
    tmp_index = index_path.with_name(f".{index_path.name}.tmp")
    tmp_index.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_index.replace(index_path)
    print(json.dumps({"stage": "pack_done", "output_root": str(output_root), "num_samples": total_samples}, ensure_ascii=False), flush=True)
    return 0


def _load_pack_archive_payload(
    archive_path: str | Path,
    *,
    success_mu_thresh: float,
    object_pooled_roots: Sequence[Path] | None,
    input_roots: Sequence[Path] | None,
    require_object_pooled: bool,
    include_object_cloud: bool,
) -> dict[str, np.ndarray]:
    object_pooled_path = _object_pooled_path_for_archive(
        archive_path,
        input_roots=input_roots,
        object_pooled_roots=object_pooled_roots,
        require=require_object_pooled,
    )
    return _load_archive_payload(
        archive_path,
        success_mu_thresh=success_mu_thresh,
        object_pooled_path=object_pooled_path,
        skip_object_cloud=not include_object_cloud,
        require_object_pooled=require_object_pooled,
    )


def _create_output_arrays(
    arrays_root: Path,
    *,
    total_samples: int,
    sample: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        "pose_features": ((total_samples, sample["pose_features"].shape[1]), np.dtype("float32")),
        "base_score": ((total_samples,), np.dtype("float32")),
        "local_cloud": ((total_samples, sample["local_cloud"].shape[1], 3), np.dtype("float32")),
        "cloud_mask": ((total_samples, sample["cloud_mask"].shape[1]), np.dtype("bool")),
        "mu_min": ((total_samples,), np.dtype("float32")),
        "success_label": ((total_samples,), np.dtype("float32")),
        "is_collision": ((total_samples,), np.dtype("float32")),
        "is_empty": ((total_samples,), np.dtype("float32")),
        "object_assignments": ((total_samples,), np.dtype("int64")),
        "archive_indices": ((total_samples,), np.dtype("int32")),
        "local_indices": ((total_samples,), np.dtype("int32")),
    }
    if "object_cloud" in sample:
        specs["object_cloud"] = ((total_samples, sample["object_cloud"].shape[1], 3), np.dtype("float32"))
    if "object_pooled" in sample:
        specs["object_pooled"] = (
            (total_samples, sample["object_pooled"].shape[1]),
            np.dtype(sample["object_pooled"].dtype),
        )
    return {
        field: np.lib.format.open_memmap(
            arrays_root / _array_filename(field),
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for field, (shape, dtype) in specs.items()
    }


def _validate_payload_capacity(
    arrays: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    *,
    archive_path: str | Path,
) -> None:
    local_cap = int(arrays["local_cloud"].shape[1])
    mask_cap = int(arrays["cloud_mask"].shape[1])
    if int(payload["local_cloud"].shape[1]) > local_cap:
        raise ValueError(
            f"{archive_path}: local_cloud has {payload['local_cloud'].shape[1]} points, "
            f"but packed dataset was initialized with capacity {local_cap}. "
            "Regenerate relabel assets with a fixed local_cloud point count before packing."
        )
    if int(payload["cloud_mask"].shape[1]) > mask_cap:
        raise ValueError(
            f"{archive_path}: cloud_mask has {payload['cloud_mask'].shape[1]} points, "
            f"but packed dataset was initialized with capacity {mask_cap}."
        )
    if "object_cloud" in payload:
        if "object_cloud" not in arrays:
            raise ValueError(f"{archive_path}: object_cloud present but output array was not initialized")
        object_cap = int(arrays["object_cloud"].shape[1])
        if int(payload["object_cloud"].shape[1]) > object_cap:
            raise ValueError(
                f"{archive_path}: object_cloud has {payload['object_cloud'].shape[1]} points, "
                f"but packed dataset was initialized with capacity {object_cap}."
            )
    if "object_pooled" in payload:
        if "object_pooled" not in arrays:
            raise ValueError(f"{archive_path}: object_pooled present but output array was not initialized")
        if int(payload["object_pooled"].shape[1]) != int(arrays["object_pooled"].shape[1]):
            raise ValueError(
                f"{archive_path}: object_pooled dim {payload['object_pooled'].shape[1]} "
                f"!= packed dim {arrays['object_pooled'].shape[1]}"
            )


def _clear_padded_slice(
    arrays: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
) -> None:
    if int(payload["local_cloud"].shape[1]) < int(arrays["local_cloud"].shape[1]):
        arrays["local_cloud"][start:end] = 0
    if int(payload["cloud_mask"].shape[1]) < int(arrays["cloud_mask"].shape[1]):
        arrays["cloud_mask"][start:end] = False
    if "object_cloud" in payload and "object_cloud" in arrays:
        if int(payload["object_cloud"].shape[1]) < int(arrays["object_cloud"].shape[1]):
            arrays["object_cloud"][start:end] = 0


def _array_metadata(output_root: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for field, array in arrays.items():
        public_field = _public_field_name(field)
        path = Path("arrays") / _array_filename(field)
        file_path = output_root / path
        out[public_field] = {
            "path": path.as_posix(),
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(np.dtype(array.dtype)),
            "nbytes": int(array.nbytes),
            "file_size": int(file_path.stat().st_size),
        }
    return out


def _array_filename(field: str) -> str:
    return f"{_public_field_name(field)}.npy"


def _public_field_name(field: str) -> str:
    if field == "archive_indices":
        return "archive_index"
    if field == "local_indices":
        return "local_index"
    return field


def _collect_archive_entries(
    input_roots: Sequence[Path],
    *,
    records_by_root: dict[Path, dict[Path, dict]],
    require_manifest: bool,
    camera: str | None,
) -> list[dict]:
    entries: list[dict] = []
    for root_idx, root in enumerate(input_roots):
        root_records = records_by_root.get(root.resolve())
        for archive_path in filter_archive_paths_by_camera(sorted(root.glob("**/*.npz")), camera):
            manifest_record = None if root_records is None else root_records.get(archive_path.resolve())
            if require_manifest and manifest_record is None:
                raise KeyError(f"missing fresh manifest record for {archive_path}")
            count = (
                int(manifest_record.get("num_grasps", 0))
                if manifest_record is not None
                else _archive_sample_count(archive_path)
            )
            entries.append(
                {
                    "input_root_index": int(root_idx),
                    "relative_path": archive_path.relative_to(root).as_posix(),
                    "path": archive_path,
                    "num_samples": max(int(count), 0),
                    "manifest_record": manifest_record,
                }
            )
    return entries


def _load_manifest_records_by_root(
    input_roots: Sequence[Path],
    manifest_args: Sequence[str] | None,
    *,
    require_manifest: bool,
    success_mu_thresh: float,
    camera: str | None,
) -> dict[Path, dict[Path, dict]]:
    if manifest_args is None:
        if not require_manifest:
            return {}
        manifest_args = ["auto"]
    if len(manifest_args) == 1 and len(input_roots) > 1:
        manifest_args = [manifest_args[0]] * len(input_roots)
    if len(manifest_args) != len(input_roots):
        raise ValueError(
            "--archive-manifest must provide exactly one value per --input-root "
            f"({len(manifest_args)} != {len(input_roots)})"
        )
    out: dict[Path, dict[Path, dict]] = {}
    for root, manifest_arg in zip(input_roots, manifest_args):
        manifest_path = default_manifest_path(root) if manifest_arg == "auto" else Path(manifest_arg)
        if not manifest_path.is_file():
            if require_manifest:
                raise FileNotFoundError(f"required archive manifest not found: {manifest_path}")
            continue
        records = load_manifest_records(manifest_path)
        by_path = manifest_records_by_archive_path(records, root=root)
        archive_paths = filter_archive_paths_by_camera(sorted(root.glob("**/*.npz")), camera)
        ok, problems = validate_manifest_coverage(
            by_path,
            archive_paths,
            success_mu_thresh=float(success_mu_thresh),
        )
        if not ok:
            if require_manifest:
                raise ValueError(f"{manifest_path} is stale: {problems[0]}")
            continue
        out[root.resolve()] = by_path
    return out


def _normalize_optional_roots(
    roots: Sequence[str] | None,
    input_roots: Sequence[Path],
    *,
    option_name: str,
) -> list[Path] | None:
    if roots is None:
        return None
    if len(roots) == 0:
        return None
    if len(roots) == 1 and len(input_roots) > 1:
        roots = list(roots) * len(input_roots)
    if len(roots) != len(input_roots):
        raise ValueError(
            f"{option_name} must provide one path or exactly one path per --input-root "
            f"({len(roots)} != {len(input_roots)})"
        )
    return [Path(path) for path in roots]


def _remove_packed_outputs(output_root: Path) -> None:
    allowed = {"index.json", "arrays"}
    if output_root.exists():
        for child in output_root.iterdir():
            if child.name not in allowed:
                raise ValueError(
                    f"{output_root} contains unexpected entry {child.name!r}; refusing --overwrite"
                )
    arrays = output_root / "arrays"
    if arrays.exists():
        for path in arrays.glob("*.npy"):
            path.unlink()
        try:
            arrays.rmdir()
        except OSError:
            pass
    index = output_root / "index.json"
    if index.exists():
        index.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
