from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Callable

import numpy as np

from .archive_io import archive_format_from_env, save_npz_archive


LabelFn = Callable[[np.ndarray, dict[str, Any]], dict[str, np.ndarray | float | int | bool]]


@dataclass(frozen=True)
class AnalyticLabelConfig:
    detector: str
    benchmark: str
    split: str
    camera: str


class AnalyticLabeler:
    """Callback-driven labeler used to turn candidate archives into training labels.

    The actual physics / evaluator call is dataset-specific, so the core pipeline stays
    generic and expects the caller to provide a `label_fn`.
    """

    def __init__(self, config: AnalyticLabelConfig) -> None:
        self.config = config

    def label_archive(
        self,
        candidate_path: str | Path,
        save_path: str | Path,
        label_fn: LabelFn,
        extra_meta: dict[str, Any] | None = None,
        archive_format: str | None = None,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> Path:
        candidate_path = Path(candidate_path)
        with np.load(candidate_path, allow_pickle=True) as candidate:
            meta = _load_meta(candidate)
            grasp_group_array = np.asarray(candidate["grasp_group_array"], dtype=np.float32)
            labels = label_fn(grasp_group_array, meta)

            payload = {
                "grasp_group_array": grasp_group_array,
                "base_scores": np.asarray(candidate["base_scores"], dtype=np.float32),
                "grasp_widths": np.asarray(candidate["grasp_widths"], dtype=np.float32),
                "grasp_poses": np.asarray(candidate["grasp_poses"], dtype=np.float32),
                "mu_min": _coerce_vector(labels.get("mu_min"), len(grasp_group_array), fill_value=np.inf),
                "is_collision": _coerce_bool_vector(labels.get("is_collision"), len(grasp_group_array)),
                "is_empty": _coerce_bool_vector(labels.get("is_empty"), len(grasp_group_array)),
                "local_cloud": _coerce_local_cloud(labels.get("local_cloud"), len(grasp_group_array)),
                "cloud_mask": _coerce_cloud_mask(labels.get("cloud_mask"), len(grasp_group_array)),
                # -1 sentinel for object_assignments marks "no GT object
                # assigned" (background, far from any model).
                "object_assignments": _coerce_object_assignments(
                    labels.get("object_assignments"), len(grasp_group_array)
                ),
            }
            object_cloud = _coerce_object_cloud(
                labels.get("object_cloud"), len(grasp_group_array)
            )
            if include_object_cloud:
                payload["object_cloud"] = object_cloud
        merged_meta = {
            **meta,
            "detector": self.config.detector,
            "benchmark": self.config.benchmark,
            "split": self.config.split,
            "camera": self.config.camera,
        }
        if extra_meta:
            merged_meta.update(extra_meta)
        if object_cloud_save_path is not None:
            _save_object_cloud_sidecar(
                object_cloud_save_path,
                object_cloud=object_cloud,
                object_assignments=payload["object_assignments"],
                meta=merged_meta,
                archive_format=archive_format or archive_format_from_env(),
            )

        return save_npz_archive(
            save_path,
            archive_format=archive_format or archive_format_from_env(),
            **payload,
            meta_json=np.array(json.dumps(merged_meta), dtype=object),
        )

    def label_prediction(
        self,
        prediction: Any,
        save_path: str | Path,
        label_fn: LabelFn,
        extra_meta: dict[str, Any] | None = None,
        archive_format: str | None = None,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> Path:
        """Label a detector prediction directly without a pre-relabel archive."""
        key = prediction.key
        grasp_group_array = np.asarray(prediction.grasp_group_array, dtype=np.float32)
        meta = {
            "detector": key.detector,
            "benchmark": key.benchmark,
            "split": key.split,
            "camera": key.camera,
            "scene_id": key.scene_id,
            "frame_id": key.frame_id,
        }
        if extra_meta:
            meta.update(extra_meta)
        labels = label_fn(grasp_group_array, meta)
        payload = {
            "grasp_group_array": grasp_group_array,
            "base_scores": _extract_base_scores(grasp_group_array),
            "grasp_widths": _extract_widths(grasp_group_array),
            "grasp_poses": _extract_poses(grasp_group_array),
            "mu_min": _coerce_vector(labels.get("mu_min"), len(grasp_group_array), fill_value=np.inf),
            "is_collision": _coerce_bool_vector(labels.get("is_collision"), len(grasp_group_array)),
            "is_empty": _coerce_bool_vector(labels.get("is_empty"), len(grasp_group_array)),
            "local_cloud": _coerce_local_cloud(labels.get("local_cloud"), len(grasp_group_array)),
            "cloud_mask": _coerce_cloud_mask(labels.get("cloud_mask"), len(grasp_group_array)),
            "object_assignments": _coerce_object_assignments(
                labels.get("object_assignments"), len(grasp_group_array)
            ),
        }
        object_cloud = _coerce_object_cloud(
            labels.get("object_cloud"), len(grasp_group_array)
        )
        if include_object_cloud:
            payload["object_cloud"] = object_cloud
        if object_cloud_save_path is not None:
            _save_object_cloud_sidecar(
                object_cloud_save_path,
                object_cloud=object_cloud,
                object_assignments=payload["object_assignments"],
                meta=meta,
                archive_format=archive_format or archive_format_from_env(),
            )
        return save_npz_archive(
            save_path,
            archive_format=archive_format or archive_format_from_env(),
            **payload,
            meta_json=np.array(json.dumps(meta), dtype=object),
        )

    def augment_archive(
        self,
        candidate_path: str | Path,
        save_path: str | Path,
        augment_fn: LabelFn,
        archive_format: str | None = None,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> Path:
        """Upgrade a legacy archive to the current schema.

        The legacy archive already carries the expensive analytic labels
        (mu_min / is_collision / is_empty) and local_cloud. We reuse those
        verbatim and only call ``augment_fn`` to compute the two object-tier
        fields the v2 schema lacked. The deprecated ``local_features`` channel
        is dropped (ShellAttn consumes xyz only).
        """
        candidate_path = Path(candidate_path)
        with np.load(candidate_path, allow_pickle=True) as candidate:
            meta = _load_meta(candidate)
            grasp_group_array = np.asarray(candidate["grasp_group_array"], dtype=np.float32)
            n = len(grasp_group_array)
            extra = augment_fn(grasp_group_array, meta)

            payload = {
                "grasp_group_array": grasp_group_array,
                "base_scores": np.asarray(candidate["base_scores"], dtype=np.float32),
                "grasp_widths": np.asarray(candidate["grasp_widths"], dtype=np.float32),
                "grasp_poses": np.asarray(candidate["grasp_poses"], dtype=np.float32),
                "mu_min": _coerce_vector(candidate["mu_min"], n, fill_value=np.inf),
                "is_collision": _coerce_bool_vector(candidate["is_collision"], n),
                "is_empty": _coerce_bool_vector(candidate["is_empty"], n),
                "local_cloud": _coerce_local_cloud(candidate["local_cloud"], n),
                "cloud_mask": _coerce_cloud_mask(candidate["cloud_mask"], n),
                "object_assignments": _coerce_object_assignments(
                    extra.get("object_assignments"), n
                ),
            }
            object_cloud = _coerce_object_cloud(extra.get("object_cloud"), n)
            if include_object_cloud:
                payload["object_cloud"] = object_cloud
        merged_meta = {
            **meta,
            "detector": self.config.detector,
            "benchmark": self.config.benchmark,
            "split": self.config.split,
            "camera": self.config.camera,
            "schema_upgrade": "legacy_to_object_tier",
        }
        if object_cloud_save_path is not None:
            _save_object_cloud_sidecar(
                object_cloud_save_path,
                object_cloud=object_cloud,
                object_assignments=payload["object_assignments"],
                meta=merged_meta,
                archive_format=archive_format or archive_format_from_env(),
            )

        return save_npz_archive(
            save_path,
            archive_format=archive_format or archive_format_from_env(),
            **payload,
            meta_json=np.array(json.dumps(merged_meta), dtype=object),
        )


def _load_meta(npz_file: np.lib.npyio.NpzFile) -> dict[str, Any]:
    meta_json = npz_file.get("meta_json")
    if meta_json is None:
        return {}
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    return json.loads(meta_json)


def _coerce_vector(value: Any, n: int, fill_value: float) -> np.ndarray:
    if value is None:
        return np.full((n,), fill_value, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"Expected vector of length {n}, got {len(arr)}")
    return arr


def _coerce_bool_vector(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros((n,), dtype=bool)
    arr = np.asarray(value, dtype=bool).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"Expected bool vector of length {n}, got {len(arr)}")
    return arr


def _coerce_local_cloud(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros((n, 0, 3), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != n or arr.shape[-1] != 3:
        raise ValueError("local_cloud must have shape [N, P, 3]")
    return arr


def _coerce_cloud_mask(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros((n, 0), dtype=np.bool_)
    arr = np.asarray(value, dtype=np.bool_)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("cloud_mask must have shape [N, P]")
    return arr


def _coerce_object_assignments(value: Any, n: int) -> np.ndarray:
    """Per-candidate global GraspNet object id (0..87) or -1 for background.

    The relabel backend computes a frame-local index and maps it to the
    GraspNet model bank via ``evaluator.get_scene_models``. The -1 sentinel
    marks candidates whose nearest model exceeds ``background_max_dist`` or
    that fail collision validation.
    """
    if value is None:
        return np.full((n,), -1, dtype=np.int32)
    arr = np.asarray(value, dtype=np.int32).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"Expected vector of length {n}, got {len(arr)}")
    return arr


def _coerce_object_cloud(value: Any, n: int) -> np.ndarray:
    """Object tier cloud: (N, P_obj, 3) camera-aligned points within radius."""
    if value is None:
        return np.zeros((n, 0, 3), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != n or arr.shape[-1] != 3:
        raise ValueError("object_cloud must have shape [N, P, 3]")
    return arr


def _save_object_cloud_sidecar(
    path: str | Path,
    *,
    object_cloud: np.ndarray,
    object_assignments: np.ndarray,
    meta: dict[str, Any],
    archive_format: str,
) -> Path:
    return save_npz_archive(
        path,
        archive_format=archive_format,
        object_cloud=object_cloud,
        object_assignments=np.asarray(object_assignments, dtype=np.int32),
        meta_json=np.array(json.dumps(meta), dtype=object),
    )


def _extract_base_scores(grasp_group_array: np.ndarray) -> np.ndarray:
    if grasp_group_array.shape[1] == 0:
        return np.empty((len(grasp_group_array),), dtype=np.float32)
    return grasp_group_array[:, 0].astype(np.float32, copy=False)


def _extract_widths(grasp_group_array: np.ndarray) -> np.ndarray:
    if grasp_group_array.shape[1] < 2:
        return np.empty((len(grasp_group_array),), dtype=np.float32)
    return grasp_group_array[:, 1].astype(np.float32, copy=False)


def _extract_poses(grasp_group_array: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(grasp_group_array), axis=0)
    if grasp_group_array.shape[1] < 16:
        return poses
    rotations = grasp_group_array[:, 4:13].reshape(-1, 3, 3).astype(np.float32, copy=False)
    translations = grasp_group_array[:, 13:16].astype(np.float32, copy=False)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    return poses
