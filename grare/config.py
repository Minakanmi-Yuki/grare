"""YAML configuration loader for grare-run."""

from __future__ import annotations

import os
from pathlib import Path
import re
import math
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


# One GraRe configuration is released for every entry in this table. Keep the
# detector/camera selection independent of a local asset-root layout so the
# Hub downloader and the demo resolve exactly the same checkpoint.
CONFIG_NAME_BY_SELECTION = {
    ("graspnet_baseline", "realsense"): "gn_realsense",
    ("graspnet_baseline", "kinect"): "gn_kinect",
    ("scale_balanced_grasp", "realsense"): "sbg_realsense",
    ("economicgrasp", "realsense"): "eg_realsense",
    ("economicgrasp", "kinect"): "eg_kinect",
    ("hggd", "realsense"): "hggd_realsense",
    ("hggd", "kinect"): "hggd_kinect",
    ("rngnet", "realsense"): "rngnet_realsense",
    ("rngnet", "kinect"): "rngnet_kinect",
}


def config_name_for_selection(detector: str, camera: str) -> str:
    """Return the released config name for one frozen-detector setting."""
    name = CONFIG_NAME_BY_SELECTION.get((detector, camera))
    if name is None:
        raise ValueError(
            f"no GraRe configuration is available for detector={detector!r}, "
            f"camera={camera!r}"
        )
    return name


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"configuration inheritance cycle: {resolved}")
    seen.add(resolved)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    return _deep_merge(_load_yaml(resolved.parent / str(parent), seen), payload)


def _expand_environment(value: Any, *, allow_missing: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, allow_missing=allow_missing)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment(item, allow_missing=allow_missing) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            if allow_missing:
                return match.group(0)
            raise ValueError(f"required environment variable is not set: {name}")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override through non-mapping key: {part}")
        node = child
    node[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    allow_missing_environment: bool = False,
) -> dict[str, Any]:
    config = _load_yaml(Path(path), set())
    for specification in overrides or []:
        if "=" not in specification:
            raise ValueError(f"override must be key=value: {specification!r}")
        key, raw = specification.split("=", 1)
        _set_dotted(config, key, yaml.safe_load(raw))
    config = _expand_environment(config, allow_missing=allow_missing_environment)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for key in ("name", "detector", "camera", "paths", "train", "rerank", "eval"):
        if key not in config:
            raise ValueError(f"configuration is missing top-level key: {key}")
    if config["detector"] not in {
        "graspnet_baseline",
        "scale_balanced_grasp",
        "economicgrasp",
        "hggd",
        "rngnet",
    }:
        raise ValueError(f"unsupported detector: {config['detector']!r}")
    if config["camera"] not in {"realsense", "kinect"}:
        raise ValueError(f"unsupported camera: {config['camera']!r}")
    if int(config["train"]["batch_size"]) != 2048:
        raise ValueError("the reported settings require train.batch_size=2048")
    # lambda=1.0 is the GraRe ranking; lambda=0.0 keeps the detector's own
    # ranking and produces the detector baseline AP that README compares
    # against. Intermediate values are valid ablations and get their own
    # rerank/evaluation artifact directories in grare-run.
    rerank_lambda = float(config["rerank"]["lambda"])
    if not math.isfinite(rerank_lambda) or not 0.0 <= rerank_lambda <= 1.0:
        raise ValueError(f"rerank.lambda must be a finite value in [0, 1]; got {rerank_lambda}")
