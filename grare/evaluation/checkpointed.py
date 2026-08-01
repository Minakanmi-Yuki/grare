"""Checkpointed per-scene evaluation for the official GraspNet protocol.

The official ``eval_all`` returns its tensor only after all 90 scenes finish and
keeps every intermediate result in worker memory, so an interruption discards
the whole run and a wedged worker stalls it silently. This module keeps the
official per-annotation scoring but writes each annotation to disk as soon as it
is computed, so a run resumes from the last completed annotation and a scene can
be retried under a timeout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import (
    eval_grasp,
    get_scene_name,
    transform_points,
)

TOTAL_ANN = 256
TOP_K = 50
FRICTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
ANN_SHAPE = (TOP_K, len(FRICTIONS))
SCENE_SHAPE = (TOTAL_ANN, TOP_K, len(FRICTIONS))

__all__ = [
    "TOTAL_ANN",
    "TOP_K",
    "FRICTIONS",
    "ANN_SHAPE",
    "SCENE_SHAPE",
    "ann_path",
    "scene_path",
    "atomic_save_npy",
    "valid_npy",
    "evaluate_annotation",
    "assemble_scene",
    "scene_is_complete",
]


def ann_path(ckpt: Path, scene_id: int, ann_id: int) -> Path:
    return ckpt / "ann" / f"scene_{scene_id:04d}" / f"{ann_id:04d}.npy"


def scene_path(ckpt: Path, scene_id: int) -> Path:
    return ckpt / "scene" / f"scene_{scene_id:04d}.npy"


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def valid_npy(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return tuple(arr.shape) == shape

def evaluate_annotation(
    evaluator: GraspNetEval,
    scene_id: int,
    ann_id: int,
    dump_folder: str | Path,
    model_sampled_list: list[np.ndarray],
    dexmodel_list: list,
    table: np.ndarray,
    config,
    *,
    max_width: float = 0.1,
) -> np.ndarray:
    dump_root = Path(dump_folder)
    grasp_path = dump_root / get_scene_name(scene_id) / f"{ann_id:04d}.npy"
    if not grasp_path.is_file():
        grasp_path = dump_root / get_scene_name(scene_id) / evaluator.camera / f"{ann_id:04d}.npy"
    grasp_group = GraspGroup().from_npy(str(grasp_path))
    if len(grasp_group) == 0:
        return np.zeros(ANN_SHAPE, dtype=np.float32)
    _, pose_list, camera_pose, align_mat = evaluator.get_model_poses(scene_id, ann_id)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

    gg_array = grasp_group.grasp_group_array
    gg_array[gg_array[:, 1] < 0, 1] = 0
    gg_array[gg_array[:, 1] > max_width, 1] = max_width
    grasp_group.grasp_group_array = gg_array

    grasp_list, score_list, collision_mask_list = eval_grasp(
        grasp_group,
        model_sampled_list,
        dexmodel_list,
        pose_list,
        config,
        table=table_trans,
        voxel_size=0.008,
        TOP_K=TOP_K,
    )

    grasp_list = [x for x in grasp_list if len(x) != 0]
    score_list = [x for x in score_list if len(x) != 0]
    collision_mask_list = [x for x in collision_mask_list if len(x) != 0]
    if len(grasp_list) == 0:
        return np.zeros(ANN_SHAPE, dtype=np.float32)

    grasp_list = np.concatenate(grasp_list)
    score_list = np.concatenate(score_list)
    collision_mask_list = np.concatenate(collision_mask_list)
    indices = np.argsort(-grasp_list[:, 0])
    score_list = score_list[indices]
    collision_mask_list = collision_mask_list[indices]

    grasp_accuracy = np.zeros(ANN_SHAPE, dtype=np.float32)
    valid_scores = (score_list > 0)
    for fric_idx, fric in enumerate(FRICTIONS):
        positives = ((score_list <= fric) & valid_scores).astype(np.int32)
        cumsum = np.cumsum(positives)
        limit = min(TOP_K, len(score_list))
        if limit:
            denom = np.arange(1, limit + 1, dtype=np.float32)
            grasp_accuracy[:limit, fric_idx] = cumsum[:limit] / denom
        if limit < TOP_K:
            total = int(cumsum[-1]) if cumsum.size else 0
            for k in range(limit, TOP_K):
                grasp_accuracy[k, fric_idx] = total / float(k + 1)
    _ = collision_mask_list
    return grasp_accuracy


def assemble_scene(ckpt: Path, scene_id: int) -> bool:
    anns: list[np.ndarray] = []
    for ann_id in range(TOTAL_ANN):
        path = ann_path(ckpt, scene_id, ann_id)
        if not valid_npy(path, ANN_SHAPE):
            return False
        anns.append(np.load(path).astype(np.float32, copy=False))
    atomic_save_npy(scene_path(ckpt, scene_id), np.stack(anns, axis=0))
    return True


def scene_is_complete(checkpoint_dir: Path, scene_id: int) -> bool:
    """True when a scene shard exists, or every annotation shard does."""
    if valid_npy(scene_path(checkpoint_dir, scene_id), SCENE_SHAPE):
        return True
    return all(
        valid_npy(ann_path(checkpoint_dir, scene_id, ann_id), ANN_SHAPE)
        for ann_id in range(TOTAL_ANN)
    )
