"""Download published GraRe dumps, prepared features, or trained checkpoints.

Dump and feature archives are downloaded from the GraRe Hugging Face Dataset
repository and extracted into their matching asset roots. Checkpoints are
downloaded from the matching Model repository into ``$GRARE_OUTPUT_ROOT``.

Examples:
    grare-fetch features --detector economicgrasp --camera realsense
    grare-fetch dumps --detector economicgrasp --camera realsense
    grare-fetch checkpoint --detector economicgrasp --camera realsense
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from typing import Any, Iterable

from grare.config import config_name_for_selection, load_config


DEFAULT_DATASET_REPO = "jibaoyuan/grare-graspnet"
DEFAULT_MODEL_REPO = "jibaoyuan/grare-graspnet"
INDEX_FILENAME = "metadata/index.json"
FEATURE_STAGES = ("local_cloud", "object_cloud", "object_pooled")
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser(
        "features", help="Download prepared features into $GRARE_DATA_ROOT."
    )
    features.add_argument("--detector", required=True)
    features.add_argument("--camera", required=True)
    features.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Download train, test, or both splits (default: all).",
    )
    features.add_argument(
        "--stages",
        nargs="+",
        choices=(*FEATURE_STAGES, "all"),
        default=["all"],
        help="Feature stages to fetch (default: all).",
    )
    features.add_argument(
        "--output-root",
        default=None,
        help="Override $GRARE_DATA_ROOT; intended for an alternate asset workspace.",
    )
    features.add_argument(
        "--repo-id",
        default=os.environ.get("GRARE_HF_DATASET_REPO", DEFAULT_DATASET_REPO),
        help="Hugging Face Dataset repository.",
    )
    features.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace local files that do not match a downloaded shard.",
    )
    features.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip archive-count and manifest verification after extraction.",
    )
    features.add_argument("--dry-run", action="store_true")

    dumps = subparsers.add_parser(
        "dumps", help="Download frozen-detector dumps into $GRARE_DUMP_ROOT."
    )
    dumps.add_argument("--detector", required=True)
    dumps.add_argument("--camera", required=True)
    dumps.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Download train, test, or both splits (default: all).",
    )
    dumps.add_argument(
        "--output-root",
        default=None,
        help="Override $GRARE_DUMP_ROOT; intended for an alternate asset workspace.",
    )
    dumps.add_argument(
        "--repo-id",
        default=os.environ.get("GRARE_HF_DATASET_REPO", DEFAULT_DATASET_REPO),
        help="Hugging Face Dataset repository.",
    )
    dumps.add_argument("--overwrite", action="store_true")
    dumps.add_argument("--no-verify", action="store_true")
    dumps.add_argument("--dry-run", action="store_true")

    checkpoint = subparsers.add_parser(
        "checkpoint", help="Download a trained checkpoint into $GRARE_OUTPUT_ROOT."
    )
    selection = checkpoint.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Config path (kept for backward compatibility).")
    selection.add_argument("--detector", help="Frozen detector used to train the checkpoint.")
    checkpoint.add_argument("--camera", help="Camera paired with --detector.")
    checkpoint.add_argument(
        "--output-root",
        default=None,
        help="Override $GRARE_OUTPUT_ROOT; intended for an alternate output directory.",
    )
    checkpoint.add_argument(
        "--repo-id",
        default=os.environ.get("GRARE_HF_MODEL_REPO", DEFAULT_MODEL_REPO),
        help="Hugging Face Model repository.",
    )
    checkpoint.add_argument("--overwrite", action="store_true")
    checkpoint.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _required_env(name: str) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value)
    raise SystemExit(
        f"{name} is not set. Source $GRARE_ASSET_WORKSPACE/grare_paths.env first."
    )


def _hf_hub_download(*, repo_id: str, repo_type: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - exercised by end users
        raise SystemExit(
            "grare-fetch requires huggingface_hub. Install it with "
            "python -m pip install -e '.[hub]'."
        ) from error
    token = os.environ.get("HF_TOKEN")
    def download() -> str:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            token=token,
        )

    return Path(_retry_hub(download, f"download {repo_type} artifact {filename}"))


def _retry_hub(operation: Any, description: str) -> Any:
    """Retry transient Hugging Face service failures without hiding real errors."""
    for attempt in range(1, 6):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"transient Hugging Face HTTP {status} while attempting to {description}; "
                f"retrying in {delay}s ({attempt}/5)",
                flush=True,
            )
            time.sleep(delay)


def _load_index(*, repo_id: str, repo_type: str) -> dict[str, Any]:
    path = _hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=INDEX_FILENAME)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(
            f"unsupported GraRe Hugging Face index schema: {payload.get('schema_version')!r}"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_download(path: Path, entry: dict[str, Any]) -> None:
    expected_size = entry.get("size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(
            f"downloaded size mismatch for {entry['path']}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    expected_hash = entry.get("sha256")
    if expected_hash and _sha256(path) != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {entry['path']}")


def _stage_selection(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return FEATURE_STAGES
    return tuple(dict.fromkeys(values))


def _safe_member_destination(root: Path, member: tarfile.TarInfo) -> Path:
    name = Path(member.name)
    if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
        raise RuntimeError(f"unsafe member in published asset shard: {member.name!r}")
    destination = root / name
    if root.resolve() not in {destination.resolve(), *destination.resolve().parents}:
        raise RuntimeError(f"asset shard member escapes output root: {member.name!r}")
    return destination


def _extract_shard(*, source: Path, output_root: Path, overwrite: bool) -> None:
    with tarfile.open(source, mode="r") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported member in published asset shard: {member.name!r}")
            destination = _safe_member_destination(output_root, member)
            if destination.exists():
                if destination.stat().st_size == member.size:
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"existing file differs from {source.name}: {destination}; "
                        "pass --overwrite to replace it"
                    )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as input_stream:
                if input_stream is None:
                    raise RuntimeError(f"could not read {member.name!r} from {source}")
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
                ) as output_stream:
                    tmp_path = Path(output_stream.name)
                    shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            tmp_path.replace(destination)


def _verify_feature_tree(
    *,
    output_root: Path,
    detector: str,
    camera: str,
    splits: Iterable[str],
    stages: Iterable[str],
    expected_counts: dict[str, Any],
) -> None:
    base = output_root / "relabeled" / detector / camera
    for split in splits:
        expected = int(expected_counts[split])
        for stage in stages:
            root = base / stage / split
            actual = sum(1 for _ in root.rglob("*.npz")) if root.is_dir() else 0
            if actual != expected:
                raise RuntimeError(
                    f"incomplete {detector} {camera} {split} {stage}: "
                    f"{actual} / {expected} archives"
                )
        if "local_cloud" in stages:
            manifest = base / "local_cloud" / split / "manifest.jsonl"
            summary = base / "local_cloud" / split / "manifest.summary.json"
            if not manifest.is_file() or not summary.is_file():
                raise RuntimeError(f"missing {detector} {camera} {split} manifest files")


def _verify_dump_tree(
    *,
    output_root: Path,
    detector: str,
    camera: str,
    splits: Iterable[str],
    expected_counts: dict[str, Any],
) -> None:
    for split in splits:
        expected = int(expected_counts[split])
        root = output_root / detector / split
        actual = sum(1 for _ in root.glob(f"scene_*/{camera}/*.npy")) if root.is_dir() else 0
        if actual != expected:
            raise RuntimeError(
                f"incomplete {detector} {camera} {split} dumps: {actual} / {expected} frames"
            )


def _fetch_features(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else _required_env("GRARE_DATA_ROOT")
    splits = SPLITS if args.split == "all" else (args.split,)
    stages = _stage_selection(args.stages)
    index = _load_index(repo_id=args.repo_id, repo_type="dataset")
    key = f"{args.detector}/{args.camera}"
    feature_set = index.get("feature_sets", {}).get(key)
    if not isinstance(feature_set, dict):
        available = ", ".join(sorted(index.get("feature_sets", {}))) or "none"
        raise RuntimeError(f"no published features for {key}; available: {available}")
    if feature_set.get("complete") is not True:
        raise RuntimeError(
            f"published features for {key} are still being uploaded; retry after the release is complete"
        )
    expected_counts = feature_set.get("archive_counts", {})
    if any(split not in expected_counts for split in splits):
        raise RuntimeError(f"published index lacks archive counts for {key}")
    shards = [
        entry
        for entry in feature_set.get("shards", [])
        if entry.get("split") in splits and entry.get("stage") in stages
    ]
    if not shards:
        raise RuntimeError(f"published index has no matching feature shards for {key}")
    total_bytes = sum(int(entry.get("size", 0)) for entry in shards)
    print(
        json.dumps(
            {
                "stage": "fetch_features",
                "repo_id": args.repo_id,
                "detector": args.detector,
                "camera": args.camera,
                "splits": list(splits),
                "stages": list(stages),
                "shards": len(shards),
                "bytes": total_bytes,
                "output_root": str(output_root),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for entry in shards:
            print(entry["path"], flush=True)
        return 0
    for number, entry in enumerate(shards, start=1):
        source = _hf_hub_download(
            repo_id=args.repo_id, repo_type="dataset", filename=str(entry["path"])
        )
        _verify_download(source, entry)
        _extract_shard(source=source, output_root=output_root, overwrite=bool(args.overwrite))
        print(
            json.dumps(
                {"stage": "fetch_features_progress", "shard": number, "shards": len(shards), "path": entry["path"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not args.no_verify:
        _verify_feature_tree(
            output_root=output_root,
            detector=args.detector,
            camera=args.camera,
            splits=splits,
            stages=stages,
            expected_counts=expected_counts,
        )
    return 0


def _fetch_dumps(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else _required_env("GRARE_DUMP_ROOT")
    splits = SPLITS if args.split == "all" else (args.split,)
    index = _load_index(repo_id=args.repo_id, repo_type="dataset")
    key = f"{args.detector}/{args.camera}"
    dump_set = index.get("dump_sets", {}).get(key)
    if not isinstance(dump_set, dict):
        available = ", ".join(sorted(index.get("dump_sets", {}))) or "none"
        raise RuntimeError(f"no published dumps for {key}; available: {available}")
    if dump_set.get("complete") is not True:
        raise RuntimeError(
            f"published dumps for {key} are still being uploaded; retry after the release is complete"
        )
    expected_counts = dump_set.get("frame_counts", {})
    if any(split not in expected_counts for split in splits):
        raise RuntimeError(f"published index lacks dump frame counts for {key}")
    shards = [
        entry
        for entry in dump_set.get("shards", [])
        if entry.get("split") in splits and entry.get("stage") == "dumps"
    ]
    if not shards:
        raise RuntimeError(f"published index has no matching dump shards for {key}")
    total_bytes = sum(int(entry.get("size", 0)) for entry in shards)
    print(
        json.dumps(
            {
                "stage": "fetch_dumps",
                "repo_id": args.repo_id,
                "detector": args.detector,
                "camera": args.camera,
                "splits": list(splits),
                "shards": len(shards),
                "bytes": total_bytes,
                "output_root": str(output_root),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for entry in shards:
            print(entry["path"], flush=True)
        return 0
    for number, entry in enumerate(shards, start=1):
        source = _hf_hub_download(
            repo_id=args.repo_id, repo_type="dataset", filename=str(entry["path"])
        )
        _verify_download(source, entry)
        _extract_shard(source=source, output_root=output_root, overwrite=bool(args.overwrite))
        print(
            json.dumps(
                {"stage": "fetch_dumps_progress", "shard": number, "shards": len(shards), "path": entry["path"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not args.no_verify:
        _verify_dump_tree(
            output_root=output_root,
            detector=args.detector,
            camera=args.camera,
            splits=splits,
            expected_counts=expected_counts,
        )
    return 0


def _atomic_copy(*, source: Path, destination: Path, overwrite: bool, entry: dict[str, Any]) -> None:
    if destination.exists():
        matches_size = destination.stat().st_size == source.stat().st_size
        matches_hash = not entry.get("sha256") or _sha256(destination) == entry["sha256"]
        if matches_size and matches_hash:
            return
        if not overwrite:
            raise FileExistsError(
                f"existing checkpoint differs from published artifact: {destination}; "
                "pass --overwrite to replace it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        tmp_path = Path(stream.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, stream, length=8 * 1024 * 1024)
    tmp_path.replace(destination)


def _checkpoint_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.config:
        if args.camera:
            raise SystemExit("--camera cannot be combined with --config")
        path = Path(args.config)
    else:
        if not args.camera:
            raise SystemExit("--camera is required with --detector")
        try:
            name = config_name_for_selection(args.detector, args.camera)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        path = Path(__file__).resolve().parents[2] / "configs" / f"{name}.yaml"
    return load_config(path, []), path


def _fetch_checkpoint(args: argparse.Namespace) -> int:
    config, config_path = _checkpoint_config(args)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_root = Path(args.output_root) if args.output_root else checkpoint_dir.parent.parent
    name = str(config["name"])
    index = _load_index(repo_id=args.repo_id, repo_type="model")
    checkpoint_set = index.get("checkpoints", {}).get(name)
    if not isinstance(checkpoint_set, dict):
        available = ", ".join(sorted(index.get("checkpoints", {}))) or "none"
        raise RuntimeError(f"no published checkpoint for {name}; available: {available}")
    files = checkpoint_set.get("files", [])
    if not files:
        raise RuntimeError(f"published checkpoint index has no files for {name}")
    print(
        json.dumps(
            {
                "stage": "fetch_checkpoint",
                "repo_id": args.repo_id,
                "config": str(config_path),
                "name": name,
                "files": len(files),
                "output_root": str(output_root),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for entry in files:
            print(entry["path"], flush=True)
        return 0
    for entry in files:
        source = _hf_hub_download(repo_id=args.repo_id, repo_type="model", filename=str(entry["path"]))
        _verify_download(source, entry)
        destination = output_root / str(entry["path"])
        _atomic_copy(
            source=source,
            destination=destination,
            overwrite=bool(args.overwrite),
            entry=entry,
        )
    best = output_root / "checkpoints" / name / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"published checkpoint for {name} did not provide {best}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "features":
        return _fetch_features(args)
    if args.command == "dumps":
        return _fetch_dumps(args)
    if args.command == "checkpoint":
        return _fetch_checkpoint(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
