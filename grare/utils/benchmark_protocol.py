from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

_SCENE_RE = re.compile(r"^scene_(\d{4})$")
_KNOWN_GRASPNET_CAMERAS = {"kinect", "realsense"}


def read_archive_meta(archive_path: str | Path) -> dict[str, Any]:
    import numpy as np

    with np.load(Path(archive_path), allow_pickle=True) as archive:
        meta_json = archive.get("meta_json")
        if meta_json is None:
            return {}
        if isinstance(meta_json, np.ndarray):
            meta_json = meta_json.item()
    return json.loads(meta_json)


def archive_camera_from_path(archive_path: str | Path) -> str | None:
    for part in reversed(Path(archive_path).parts):
        if part in _KNOWN_GRASPNET_CAMERAS:
            return part
    return None


def archive_matches_camera(archive_path: str | Path, camera: str | None) -> bool:
    if not camera:
        return True
    expected = str(camera)
    path_camera = archive_camera_from_path(archive_path)
    if path_camera is not None:
        return path_camera == expected
    meta_camera = read_archive_meta(archive_path).get("camera")
    return meta_camera is None or str(meta_camera) == expected


def filter_archive_paths_by_camera(
    archive_paths: Iterable[str | Path],
    camera: str | None,
) -> list[Path]:
    return [Path(path) for path in archive_paths if archive_matches_camera(path, camera)]


def archive_scene_key(archive_path: str | Path) -> str:
    archive_path = Path(archive_path)
    meta = read_archive_meta(archive_path)
    scene_id = meta.get("scene_id")
    if scene_id is not None:
        return f"scene_{int(scene_id):04d}"
    parent = archive_path.parent.name
    if not parent.startswith("scene_"):
        parent = archive_path.parent.parent.name
    if parent.startswith("scene_"):
        return parent
    return f"scene_{int(parent):04d}"


def archive_scene_key_from_path(archive_path: str | Path) -> str | None:
    for part in Path(archive_path).parts:
        if _SCENE_RE.match(part):
            return part
    return None


def archive_split_from_path(archive_path: str | Path) -> str | None:
    for part in Path(archive_path).parts:
        if part in {"train", "test"}:
            return part
        if part.startswith("train_top") or part.startswith("train_"):
            return "train"
        if part.startswith("test_top") or part.startswith("test_"):
            return "test"
    return None
