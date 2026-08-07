#!/usr/bin/env python3
"""Publish completed GraRe dumps, feature shards, or checkpoints to Hugging Face.

This maintainer tool creates one uncompressed tar shard per five-scene group.
The contained ``.npz`` files are already compressed, so tar reduces Hub file
count without wasting CPU or sacrificing resumable uploads.

Set ``HF_TOKEN`` before running. The token is deliberately never accepted as
a command-line argument, where it could be recorded in shell history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from typing import Any, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from grare.config import load_config


DEFAULT_DATASET_REPO = "jibaoyuan/grare-graspnet"
DEFAULT_MODEL_REPO = "jibaoyuan/grare-graspnet"
INDEX_FILENAME = "metadata/index.json"
FEATURE_STAGES = ("local_cloud", "object_cloud", "object_pooled")
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser("features", help="Publish one prepared detector-camera tree.")
    features.add_argument("--detector", required=True)
    features.add_argument("--camera", required=True)
    features.add_argument("--data-root", default=os.environ.get("GRARE_DATA_ROOT"))
    features.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    features.add_argument(
        "--scenes-per-shard",
        type=int,
        default=5,
        help="Scenes per tar shard (default: 5).",
    )
    features.add_argument(
        "--uploads-per-commit",
        type=int,
        default=5,
        help="Tar shards to send in each Hugging Face commit (default: 5).",
    )
    features.add_argument("--dry-run", action="store_true")

    dumps = subparsers.add_parser("dumps", help="Publish one frozen-detector dump tree.")
    dumps.add_argument("--detector", required=True)
    dumps.add_argument("--camera", required=True)
    dumps.add_argument("--dump-root", default=os.environ.get("GRARE_DUMP_ROOT"))
    dumps.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    dumps.add_argument("--scenes-per-shard", type=int, default=5, help="Scenes per tar shard (default: 5).")
    dumps.add_argument(
        "--uploads-per-commit",
        type=int,
        default=5,
        help="Tar shards to send in each Hugging Face commit (default: 5).",
    )
    dumps.add_argument("--dry-run", action="store_true")

    checkpoint = subparsers.add_parser("checkpoint", help="Publish the best checkpoint for one config.")
    checkpoint.add_argument("--config", required=True)
    checkpoint.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    checkpoint.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set.")
    return token


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_revision": _source_revision(),
        "feature_sets": {},
        "dump_sets": {},
        "checkpoints": {},
    }


def _retry_hub(operation: Any, description: str) -> Any:
    """Retry transient Hugging Face failures, which are common for huge uploads."""
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


def _load_index(*, repo_id: str, repo_type: str, token: str) -> dict[str, Any]:
    api = HfApi(token=token)
    remote_files = _retry_hub(
        lambda: api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token),
        f"list {repo_type} repository {repo_id}",
    )
    if INDEX_FILENAME not in set(remote_files):
        return _empty_index()
    try:
        path = _retry_hub(
            lambda: hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=INDEX_FILENAME,
                token=token,
                force_download=True,
            ),
            f"download {repo_type} index from {repo_id}",
        )
    except EntryNotFoundError:
        return _empty_index()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"unsupported remote index schema: {payload.get('schema_version')!r}")
    payload.setdefault("feature_sets", {})
    payload.setdefault("dump_sets", {})
    payload.setdefault("checkpoints", {})
    return payload


def _upload_index(*, api: HfApi, repo_id: str, repo_type: str, token: str, index: dict[str, Any]) -> None:
    index["source_revision"] = _source_revision()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as stream:
        path = Path(stream.name)
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        _retry_hub(
            lambda: api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=INDEX_FILENAME,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message="Update GraRe published asset index",
            ),
            f"upload {repo_type} index to {repo_id}",
        )
    finally:
        path.unlink(missing_ok=True)


def _upload_file(
    *,
    api: HfApi,
    source: Path,
    path_in_repo: str,
    repo_id: str,
    repo_type: str,
    token: str,
) -> dict[str, Any]:
    _retry_hub(
        lambda: api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Add {path_in_repo}",
        ),
        f"upload {path_in_repo}",
    )
    return {
        "path": path_in_repo,
        "size": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _upload_files(
    *,
    api: HfApi,
    sources: list[tuple[Path, str]],
    repo_id: str,
    repo_type: str,
    token: str,
) -> list[dict[str, Any]]:
    """Upload a small batch of completed tar shards in one Hub commit."""
    paths = [path_in_repo for _, path_in_repo in sources]
    _retry_hub(
        lambda: api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=[
                CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(source))
                for source, path_in_repo in sources
            ],
            commit_message=f"Add {len(sources)} GraRe asset shard(s): {paths[0]}",
        ),
        f"upload {len(sources)} dataset shard(s) to {repo_id}",
    )
    return [
        {"path": path_in_repo, "size": source.stat().st_size, "sha256": _sha256(source)}
        for source, path_in_repo in sources
    ]


def _scene_groups(scenes: list[Path], size: int) -> Iterable[list[Path]]:
    for offset in range(0, len(scenes), size):
        yield scenes[offset : offset + size]


def _create_tar(*, source_paths: Iterable[Path], data_root: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w") as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path.relative_to(data_root)), recursive=False)


def _stage_scene_paths(stage_root: Path, scenes: Iterable[Path]) -> Iterable[Path]:
    for scene in scenes:
        for path in sorted(scene.rglob("*")):
            if path.is_file():
                yield path


def _dump_scene_paths(scenes: Iterable[Path], camera: str) -> Iterable[Path]:
    for scene in scenes:
        for path in sorted((scene / camera).rglob("*.npy")):
            if path.is_file():
                yield path


def _feature_entry_index(
    index: dict[str, Any], *, detector: str, camera: str, data_root: Path
) -> dict[str, Any]:
    key = f"{detector}/{camera}"
    entry = index["feature_sets"].setdefault(
        key,
        {
            "detector": detector,
            "camera": camera,
            "archive_counts": {},
            "shards": [],
            "complete": False,
        },
    )
    source_root = data_root / "relabeled" / detector / camera
    if entry.get("detector") != detector or entry.get("camera") != camera:
        raise RuntimeError(f"remote index collision for {key}")
    for split in SPLITS:
        local_root = source_root / "local_cloud" / split
        entry["archive_counts"][split] = sum(1 for _ in local_root.rglob("*.npz"))
    return entry


def _dump_entry_index(
    index: dict[str, Any], *, detector: str, camera: str, dump_root: Path
) -> dict[str, Any]:
    key = f"{detector}/{camera}"
    entry = index["dump_sets"].setdefault(
        key,
        {
            "detector": detector,
            "camera": camera,
            "frame_counts": {},
            "shards": [],
            "complete": False,
        },
    )
    if entry.get("detector") != detector or entry.get("camera") != camera:
        raise RuntimeError(f"remote index collision for {key}")
    for split in SPLITS:
        split_root = dump_root / detector / split
        entry["frame_counts"][split] = sum(
            1 for _ in split_root.glob(f"scene_*/{camera}/*.npy")
        )
    return entry


def _upsert_shard(feature_set: dict[str, Any], shard: dict[str, Any]) -> bool:
    shards = feature_set["shards"]
    for idx, existing in enumerate(shards):
        if existing["path"] == shard["path"]:
            if existing.get("sha256") == shard["sha256"] and existing.get("size") == shard["size"]:
                return False
            shards[idx] = shard
            return True
    shards.append(shard)
    shards.sort(key=lambda item: item["path"])
    return True


def _published_shard(
    feature_set: dict[str, Any], *, path: str, scenes: list[str]
) -> dict[str, Any] | None:
    """Return a compatible previously indexed shard, if one exists.

    A finished shard is immutable: its path encodes the detector, camera,
    stage, split, and scene range.  Matching its explicit scene list lets an
    interrupted publish resume without rebuilding or uploading that shard.
    """
    for existing in feature_set["shards"]:
        if (
            existing.get("path") == path
            and existing.get("scenes") == scenes
            and existing.get("size") is not None
            and existing.get("sha256")
        ):
            return existing
    return None


def _publish_features(args: argparse.Namespace) -> int:
    if not args.data_root:
        raise SystemExit("--data-root is required or GRARE_DATA_ROOT must be set.")
    if args.scenes_per_shard < 1:
        raise SystemExit("--scenes-per-shard must be positive.")
    if args.uploads_per_commit < 1:
        raise SystemExit("--uploads-per-commit must be positive.")
    data_root = Path(args.data_root).resolve()
    source_root = data_root / "relabeled" / args.detector / args.camera
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    token = _require_token()
    api = HfApi(token=token)
    index = _load_index(repo_id=args.repo_id, repo_type="dataset", token=token)
    feature_set = _feature_entry_index(
        index, detector=args.detector, camera=args.camera, data_root=data_root
    )
    # Object clouds are intentionally removable after a verified upload because
    # the frozen Point-MAE stage has already consumed them. A complete remote
    # set is immutable, so do not require those reclaimed local intermediates
    # when resuming the release queue.
    if feature_set.get("complete") is True:
        print(
            json.dumps(
                {
                    "stage": "publish_features_done",
                    "repo_id": args.repo_id,
                    "detector": args.detector,
                    "camera": args.camera,
                    "archive_counts": feature_set["archive_counts"],
                    "shards": len(feature_set["shards"]),
                    "already_complete": True,
                    "dry_run": bool(args.dry_run),
                },
                ensure_ascii=False,
            )
        )
        return 0
    changed = False
    for stage in FEATURE_STAGES:
        for split in SPLITS:
            stage_root = source_root / stage / split
            scenes = sorted(path for path in stage_root.glob("scene_*") if path.is_dir())
            if not scenes:
                raise RuntimeError(f"no scenes found under {stage_root}")
            # Keep only one small batch of temporary tars on disk.  This is
            # important for the multi-hundred-GB prepared feature trees.
            with tempfile.TemporaryDirectory(prefix="grare-hf-") as temp_dir:
                pending: list[tuple[Path, str, list[str]]] = []

                def flush_pending() -> None:
                    nonlocal changed
                    if not pending:
                        return
                    sources = [(path, remote_path) for path, remote_path, _ in pending]
                    artifacts = (
                        _upload_files(
                            api=api,
                            sources=sources,
                            repo_id=args.repo_id,
                            repo_type="dataset",
                            token=token,
                        )
                        if not args.dry_run
                        else [
                            {
                                "path": remote_path,
                                "size": path.stat().st_size,
                                "sha256": _sha256(path),
                            }
                            for path, remote_path, _ in pending
                        ]
                    )
                    for (_, _, scene_names), artifact in zip(pending, artifacts, strict=True):
                        shard = {"stage": stage, "split": split, "scenes": scene_names, **artifact}
                        changed = _upsert_shard(feature_set, shard) or changed
                    pending.clear()

                for group in _scene_groups(scenes, args.scenes_per_shard):
                    first = group[0].name.removeprefix("scene_")
                    last = group[-1].name.removeprefix("scene_")
                    remote_path = (
                        f"shards/v1/{args.detector}/{args.camera}/{stage}/{split}/"
                        f"scenes-{first}-{last}.tar"
                    )
                    scene_names = [scene.name for scene in group]
                    if _published_shard(feature_set, path=remote_path, scenes=scene_names):
                        print(
                            json.dumps(
                                {"stage": "already_published", "path": remote_path},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        continue
                    print(json.dumps({"stage": "package", "path": remote_path}, ensure_ascii=False), flush=True)
                    shard_path = Path(temp_dir) / f"{len(pending):03d}.tar"
                    _create_tar(
                        source_paths=_stage_scene_paths(stage_root, group),
                        data_root=data_root,
                        destination=shard_path,
                    )
                    pending.append((shard_path, remote_path, scene_names))
                    if len(pending) == args.uploads_per_commit:
                        flush_pending()

                if stage == "local_cloud":
                    manifests = sorted(
                        path for path in stage_root.iterdir()
                        if path.is_file() and path.name.startswith("manifest")
                    )
                    if not manifests:
                        raise RuntimeError(f"missing manifest files under {stage_root}")
                    remote_path = (
                        f"shards/v1/{args.detector}/{args.camera}/{stage}/{split}/manifests.tar"
                    )
                    if not _published_shard(feature_set, path=remote_path, scenes=[]):
                        shard_path = Path(temp_dir) / f"{len(pending):03d}-manifests.tar"
                        _create_tar(source_paths=manifests, data_root=data_root, destination=shard_path)
                        pending.append((shard_path, remote_path, []))
                flush_pending()
    if feature_set.get("complete") is not True:
        feature_set["complete"] = True
        changed = True
    if changed and not args.dry_run:
        _upload_index(
            api=api, repo_id=args.repo_id, repo_type="dataset", token=token, index=index
        )
    print(
        json.dumps(
            {
                "stage": "publish_features_done",
                "repo_id": args.repo_id,
                "detector": args.detector,
                "camera": args.camera,
                "archive_counts": feature_set["archive_counts"],
                "shards": len(feature_set["shards"]),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _publish_dumps(args: argparse.Namespace) -> int:
    if not args.dump_root:
        raise SystemExit("--dump-root is required or GRARE_DUMP_ROOT must be set.")
    if args.scenes_per_shard < 1:
        raise SystemExit("--scenes-per-shard must be positive.")
    if args.uploads_per_commit < 1:
        raise SystemExit("--uploads-per-commit must be positive.")
    dump_root = Path(args.dump_root).resolve()
    source_root = dump_root / args.detector
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    token = _require_token()
    api = HfApi(token=token)
    index = _load_index(repo_id=args.repo_id, repo_type="dataset", token=token)
    dump_set = _dump_entry_index(
        index, detector=args.detector, camera=args.camera, dump_root=dump_root
    )
    if any(int(dump_set["frame_counts"][split]) == 0 for split in SPLITS):
        raise RuntimeError(f"no {args.camera} dump frames found under {source_root}")
    changed = False
    for split in SPLITS:
        split_root = source_root / split
        scenes = sorted(
            path for path in split_root.glob("scene_*") if (path / args.camera).is_dir()
        )
        if not scenes:
            raise RuntimeError(f"no scenes found under {split_root} for camera={args.camera}")
        with tempfile.TemporaryDirectory(prefix="grare-hf-") as temp_dir:
            pending: list[tuple[Path, str, list[str]]] = []

            def flush_pending() -> None:
                nonlocal changed
                if not pending:
                    return
                sources = [(path, remote_path) for path, remote_path, _ in pending]
                artifacts = (
                    _upload_files(
                        api=api,
                        sources=sources,
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        token=token,
                    )
                    if not args.dry_run
                    else [
                        {"path": remote_path, "size": path.stat().st_size, "sha256": _sha256(path)}
                        for path, remote_path, _ in pending
                    ]
                )
                for (_, _, scene_names), artifact in zip(pending, artifacts, strict=True):
                    shard = {"stage": "dumps", "split": split, "scenes": scene_names, **artifact}
                    changed = _upsert_shard(dump_set, shard) or changed
                pending.clear()

            for group in _scene_groups(scenes, args.scenes_per_shard):
                first = group[0].name.removeprefix("scene_")
                last = group[-1].name.removeprefix("scene_")
                remote_path = (
                    f"shards/v1/{args.detector}/{args.camera}/dumps/{split}/"
                    f"scenes-{first}-{last}.tar"
                )
                scene_names = [scene.name for scene in group]
                if _published_shard(dump_set, path=remote_path, scenes=scene_names):
                    print(
                        json.dumps({"stage": "already_published", "path": remote_path}, ensure_ascii=False),
                        flush=True,
                    )
                    continue
                print(json.dumps({"stage": "package", "path": remote_path}, ensure_ascii=False), flush=True)
                shard_path = Path(temp_dir) / f"{len(pending):03d}.tar"
                _create_tar(
                    source_paths=_dump_scene_paths(group, args.camera),
                    data_root=dump_root,
                    destination=shard_path,
                )
                pending.append((shard_path, remote_path, scene_names))
                if len(pending) == args.uploads_per_commit:
                    flush_pending()
            flush_pending()
    if dump_set.get("complete") is not True:
        dump_set["complete"] = True
        changed = True
    if changed and not args.dry_run:
        _upload_index(api=api, repo_id=args.repo_id, repo_type="dataset", token=token, index=index)
    print(
        json.dumps(
            {
                "stage": "publish_dumps_done",
                "repo_id": args.repo_id,
                "detector": args.detector,
                "camera": args.camera,
                "frame_counts": dump_set["frame_counts"],
                "shards": len(dump_set["shards"]),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _checkpoint_sources(output_root: Path, name: str) -> list[tuple[Path, str]]:
    candidates = [
        (output_root / "checkpoints" / name / "best.pt", f"checkpoints/{name}/best.pt"),
        (output_root / "checkpoints" / name / "history.json", f"checkpoints/{name}/history.json"),
        (output_root / "checkpoints" / name / "summary.json", f"checkpoints/{name}/summary.json"),
        (output_root / "predictions" / name / "rerank_summary.json", f"predictions/{name}/rerank_summary.json"),
        (output_root / "evaluation" / name / "per_scene_raw.json", f"evaluation/{name}/per_scene_raw.json"),
    ]
    if not candidates[0][0].is_file():
        raise FileNotFoundError(candidates[0][0])
    return [(source, destination) for source, destination in candidates if source.is_file()]


def _publish_checkpoint(args: argparse.Namespace) -> int:
    token = _require_token()
    config = load_config(args.config, [])
    name = str(config["name"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_root = checkpoint_dir.parent.parent
    api = HfApi(token=token)
    index = _load_index(repo_id=args.repo_id, repo_type="model", token=token)
    files = []
    for source, destination in _checkpoint_sources(output_root, name):
        print(json.dumps({"stage": "upload_checkpoint", "path": destination}, ensure_ascii=False), flush=True)
        entry = (
            _upload_file(
                api=api,
                source=source,
                path_in_repo=destination,
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
            )
            if not args.dry_run
            else {"path": destination, "size": source.stat().st_size, "sha256": _sha256(source)}
        )
        files.append(entry)
    index["checkpoints"][name] = {
        "name": name,
        "detector": config["detector"],
        "camera": config["camera"],
        "files": files,
    }
    if not args.dry_run:
        _upload_index(api=api, repo_id=args.repo_id, repo_type="model", token=token, index=index)
    print(
        json.dumps(
            {"stage": "publish_checkpoint_done", "repo_id": args.repo_id, "name": name, "files": len(files)},
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "features":
        return _publish_features(args)
    if args.command == "dumps":
        return _publish_dumps(args)
    if args.command == "checkpoint":
        return _publish_checkpoint(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
