from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

import numpy as np


ARCHIVE_FORMATS = ("compressed", "stored")


def archive_format_from_env(default: str = "compressed") -> str:
    value = os.environ.get("GRARE_ARCHIVE_FORMAT", default)
    return normalize_archive_format(value)


def normalize_archive_format(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "compress": "compressed",
        "compressed": "compressed",
        "zip_deflated": "compressed",
        "deflated": "compressed",
        "uncompressed": "stored",
        "store": "stored",
        "stored": "stored",
        "zip_stored": "stored",
    }
    if normalized not in aliases:
        raise ValueError(
            f"unsupported archive format {value!r}; expected one of {ARCHIVE_FORMATS}"
        )
    return aliases[normalized]


def save_npz_archive(
    path: str | Path,
    *,
    archive_format: str = "compressed",
    atomic: bool = True,
    **payload: Any,
) -> Path:
    """Write an npz archive with a selectable zip compression mode.

    `stored` keeps the `.npz` layout but skips zlib compression. It is larger on
    disk, yet substantially cheaper to read during training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_format = normalize_archive_format(archive_format)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp") if atomic else path
    try:
        with tmp_path.open("wb") as handle:
            if archive_format == "compressed":
                np.savez_compressed(handle, **payload)
            elif archive_format == "stored":
                np.savez(handle, **payload)
            else:  # pragma: no cover - guarded by normalize_archive_format.
                raise ValueError(f"unsupported archive format: {archive_format}")
        if atomic:
            tmp_path.replace(path)
    except Exception:
        if atomic:
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_npz_payload(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def load_meta(npz_file: np.lib.npyio.NpzFile) -> dict[str, Any]:
    """Decode the ``meta_json`` entry of a relabel archive into a plain dict.

    Returns ``{}`` when the archive predates the current schema and has no meta entry.
    """
    meta_json = npz_file.get("meta_json")
    if meta_json is None:
        return {}
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    return json.loads(meta_json)
