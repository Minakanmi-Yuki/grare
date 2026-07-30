"""Inference and candidate-set re-ranking for the published GraRe protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import GraspRescorer
from grare.relabeling.archive_io import load_meta as _load_meta


def rerank_archive(
    archive_path: str | Path,
    model: GraspRescorer,
    *,
    device: str = "cpu",
    features_archive_path: str | Path | None = None,
    object_pooled_archive_path: str | Path | None = None,
    require_object_pooled: bool = False,
    rescoring_score_weight: float = 1.0,
    score_normalization: str = "zscore",
) -> dict[str, Any]:
    """Score and reorder every unchanged candidate in one relabeled archive.

    GraRe preserves the candidate set and writes the fused score back to
    column zero, which is the field used by the official GraspNet evaluator.
    The paper protocol uses z-score normalization and retains every candidate.
    """
    archive_path = Path(archive_path)
    with np.load(archive_path, allow_pickle=True) as archive:
        meta = _load_meta(archive)
        features_archive = archive
        if "local_cloud" not in archive.files:
            if features_archive_path is None:
                raise KeyError(
                    f"local_cloud is missing from {archive_path}; provide the matching "
                    "relabeled archive with --features-root"
                )
            features_path = Path(features_archive_path)
            if not features_path.is_file():
                raise FileNotFoundError(f"features archive not found: {features_path}")
            features_archive = np.load(features_path, allow_pickle=True)

        base_scores = archive["base_scores"].astype(np.float32, copy=False)
        pose_features = _build_pose_features(
            archive["grasp_poses"].astype(np.float32, copy=False),
            archive["grasp_widths"].astype(np.float32, copy=False),
            base_scores,
        )
        local_cloud = features_archive["local_cloud"].astype(np.float32, copy=False)
        if "cloud_mask" in features_archive.files:
            cloud_mask = features_archive["cloud_mask"].astype(np.bool_, copy=False)
        else:
            cloud_mask = np.any(local_cloud != 0, axis=-1)

        object_cloud = None
        if "object_cloud" in features_archive.files:
            candidate_object_cloud = features_archive["object_cloud"].astype(np.float32, copy=False)
            if candidate_object_cloud.ndim == 3 and candidate_object_cloud.shape[1] > 0:
                object_cloud = candidate_object_cloud

        object_pooled = None
        if "object_pooled" in features_archive.files:
            candidate_pooled = features_archive["object_pooled"].astype(np.float32, copy=False)
            if candidate_pooled.ndim == 2 and candidate_pooled.shape[0] == local_cloud.shape[0]:
                object_pooled = candidate_pooled
        elif object_pooled_archive_path is not None:
            pooled_path = Path(object_pooled_archive_path)
            if not pooled_path.is_file():
                if require_object_pooled:
                    raise FileNotFoundError(f"object_pooled archive not found: {pooled_path}")
            else:
                with np.load(pooled_path, allow_pickle=True) as pooled_archive:
                    if "object_pooled" not in pooled_archive.files:
                        raise KeyError(f"{pooled_path} does not contain object_pooled")
                    candidate_pooled = pooled_archive["object_pooled"].astype(np.float32, copy=False)
                    if candidate_pooled.ndim != 2 or candidate_pooled.shape[0] != local_cloud.shape[0]:
                        raise ValueError(
                            f"{pooled_path}: object_pooled shape={candidate_pooled.shape} "
                            f"does not match local candidates={local_cloud.shape[0]}"
                        )
                    object_pooled = candidate_pooled.copy()
        elif require_object_pooled:
            raise KeyError(f"object_pooled is required but absent for: {archive_path}")

        input_grasps = archive["grasp_group_array"].astype(np.float32, copy=False)

    if score_normalization != "zscore":
        raise ValueError("the published GraRe protocol uses score_normalization='zscore'")

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device_obj)
    model.eval()
    with torch.inference_mode():
        model_kwargs: dict[str, torch.Tensor] = {
            "cloud_mask": torch.from_numpy(cloud_mask).to(device_obj),
        }
        if object_cloud is not None:
            model_kwargs["object_cloud"] = torch.from_numpy(object_cloud).to(device_obj)
        if object_pooled is not None:
            model_kwargs["object_pooled"] = torch.from_numpy(object_pooled).to(device_obj)
        heads = model(
            torch.from_numpy(pose_features).to(device_obj),
            torch.from_numpy(local_cloud).to(device_obj),
            **model_kwargs,
        )
        model_scores = heads["score"].detach().cpu().numpy().astype(np.float32, copy=False)

    exported_scores, score_meta = _build_export_scores(
        base_scores=base_scores,
        rescoring_scores=model_scores,
        rescoring_score_weight=rescoring_score_weight,
    )
    order = np.argsort(-exported_scores, kind="stable")
    ranked_exported_scores = _strictly_descending(exported_scores[order])
    reranked = input_grasps[order].copy()
    reranked[:, 0] = ranked_exported_scores
    return {
        "grasp_group_array": reranked,
        "grasp_group_array_input": input_grasps,
        "exported_scores": ranked_exported_scores,
        "exported_scores_input": exported_scores.astype(np.float32, copy=False),
        "rescoring_scores": model_scores[order].astype(np.float32, copy=False),
        "rescoring_scores_input": model_scores,
        "model_scores_raw": model_scores[order].astype(np.float32, copy=False),
        "model_scores_raw_input": model_scores,
        "base_scores": base_scores[order].astype(np.float32, copy=False),
        "base_scores_input": base_scores,
        "original_indices": order.astype(np.int64, copy=False),
        "meta": meta,
        "sort_descending": True,
        "score_column_index": 0,
        "score_column_overwritten": True,
        **score_meta,
    }


def export_reranked_prediction(
    archive_path: str | Path,
    model: GraspRescorer,
    save_path: str | Path,
    *,
    device: str = "cpu",
    features_archive_path: str | Path | None = None,
    object_pooled_archive_path: str | Path | None = None,
    require_object_pooled: bool = False,
    rescoring_score_weight: float = 1.0,
) -> Path:
    result = rerank_archive(
        archive_path,
        model,
        device=device,
        features_archive_path=features_archive_path,
        object_pooled_archive_path=object_pooled_archive_path,
        require_object_pooled=require_object_pooled,
        rescoring_score_weight=rescoring_score_weight,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, result["grasp_group_array"])
    return save_path


def _build_pose_features(
    grasp_poses: np.ndarray,
    widths: np.ndarray,
    base_scores: np.ndarray,
) -> np.ndarray:
    if len(grasp_poses) == 0:
        return np.zeros((0, 14), dtype=np.float32)
    rotations = grasp_poses[:, :3, :3].reshape(len(grasp_poses), -1)
    translations = grasp_poses[:, :3, 3]
    return np.concatenate(
        [
            rotations.astype(np.float32, copy=False),
            translations.astype(np.float32, copy=False),
            widths.reshape(-1, 1).astype(np.float32, copy=False),
            base_scores.reshape(-1, 1).astype(np.float32, copy=False),
        ],
        axis=1,
    )


def _build_export_scores(
    *,
    base_scores: np.ndarray,
    rescoring_scores: np.ndarray,
    rescoring_score_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= float(rescoring_score_weight) <= 1.0:
        raise ValueError("rescoring_score_weight must be in [0, 1]")
    if base_scores.shape != rescoring_scores.shape:
        raise ValueError("base_scores and rescoring_scores must have the same shape")
    base_component = _normalize_zscore(base_scores)
    rescoring_component = _normalize_zscore(rescoring_scores)
    rescoring_weight = float(rescoring_score_weight)
    exported_scores = (
        (1.0 - rescoring_weight) * base_component
        + rescoring_weight * rescoring_component
    ).astype(np.float32, copy=False)
    base_top1_idx = _argmax_or_none(base_scores)
    proposed_top1_idx = _argmax_or_none(exported_scores)
    return exported_scores, {
        "lambda": rescoring_weight,
        "score_normalization": "zscore",
        "rescoring_score_weight": rescoring_weight,
        "candidate_count": int(base_scores.size),
        "base_top1_idx": base_top1_idx,
        "proposed_top1_idx": proposed_top1_idx,
        "final_top1_idx": proposed_top1_idx,
        "top1_changed_vs_base": _top1_changed(base_top1_idx, proposed_top1_idx),
    }


def _normalize_zscore(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    if scores.size == 0:
        return scores.copy()
    std = float(np.std(scores))
    if std <= 1e-6:
        return np.zeros_like(scores)
    return (scores - float(np.mean(scores))) / std


def _strictly_descending(scores: np.ndarray) -> np.ndarray:
    ranked = np.asarray(scores, dtype=np.float32).copy()
    for index in range(1, len(ranked)):
        if ranked[index] >= ranked[index - 1]:
            ranked[index] = np.nextafter(ranked[index - 1], np.float32(-np.inf))
    return ranked


def _argmax_or_none(scores: np.ndarray) -> int | None:
    if scores.size == 0:
        return None
    return int(np.argmax(scores))


def _top1_changed(base_top1_idx: int | None, final_top1_idx: int | None) -> bool:
    if base_top1_idx is None or final_top1_idx is None:
        return False
    return base_top1_idx != final_top1_idx
