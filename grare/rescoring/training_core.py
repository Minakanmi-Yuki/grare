from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import os
import time

import torch
from torch import nn

from .model import GraspRescorer


@dataclass
class MultiTaskCriteria:
    """Bundle of head-specific losses + their fixed weights.

    Constructed in trainer.train_model() and threaded into ``run_epoch``.
    """

    score: nn.Module
    coll: nn.Module = field(default_factory=lambda: nn.BCEWithLogitsLoss(reduction="none"))
    empty: nn.Module = field(default_factory=lambda: nn.BCEWithLogitsLoss(reduction="none"))
    obj_id: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss(ignore_index=-1, reduction="none"))
    lambda_collision: float = 0.0
    lambda_empty: float = 0.0
    lambda_obj_id: float = 0.0


def _build_quality_target(
    mu_min: torch.Tensor,
    success_mu_thresh: float,
) -> torch.Tensor:
    tau = float(success_mu_thresh)
    if tau <= 0:
        raise ValueError(f"success_mu_thresh must be positive; got {success_mu_thresh!r}")
    worst_mu = max(tau, 1.0) + 1.0
    filled_mu = torch.where(torch.isfinite(mu_min), mu_min, torch.full_like(mu_min, worst_mu))
    return torch.full_like(filled_mu, tau) - filled_mu


def _mean_std(total_sum: float, total_sq_sum: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    mean = total_sum / float(count)
    variance = max(total_sq_sum / float(count) - mean * mean, 0.0)
    return float(mean), float(variance ** 0.5)


def _correlation(
    x_sum: float,
    x_sq_sum: float,
    y_sum: float,
    y_sq_sum: float,
    xy_sum: float,
    count: int,
) -> float | None:
    if count <= 1:
        return None
    x_mean = x_sum / float(count)
    y_mean = y_sum / float(count)
    x_var = max(x_sq_sum / float(count) - x_mean * x_mean, 0.0)
    y_var = max(y_sq_sum / float(count) - y_mean * y_mean, 0.0)
    denom = (x_var * y_var) ** 0.5
    if denom <= 1e-12:
        return None
    covariance = xy_sum / float(count) - x_mean * y_mean
    return float(covariance / denom)


def _resolve_amp(amp: str, device: torch.device) -> tuple[bool, torch.dtype | None]:
    if device.type != "cuda" or amp == "off":
        return False, None
    if amp == "bf16":
        return True, torch.bfloat16
    if amp == "fp16":
        return True, torch.float16
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criteria: MultiTaskCriteria,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    training: bool,
    grad_clip_norm: float | None,
    success_mu_thresh: float,
    score_collapse_std_threshold: float,
) -> dict[str, float | bool | None]:
    """Multi-task epoch loop.

    Forward returns a dict with score / collision / empty / obj_id heads.
    Total loss is a weighted sum of per-head losses; missing aux signals
    are masked out per-sample so the loop is robust to mixed batches.
    """
    if training:
        model.train()
    else:
        model.eval()

    epoch_start = time.perf_counter()
    zero64 = lambda: torch.zeros((), dtype=torch.float64, device=device)
    total_loss_t = zero64()
    total_score_t = zero64()
    total_coll_t = zero64()
    total_empty_t = zero64()
    total_obj_t = zero64()
    total_pred_sum_t = zero64()
    total_pred_sq_sum_t = zero64()
    total_target_sum_t = zero64()
    total_target_sq_sum_t = zero64()
    total_pred_target_sum_t = zero64()
    obj_correct_t = zero64()
    obj_count_t = zero64()
    coll_count_t = zero64()
    empty_count_t = zero64()
    total_count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress_every_batches = max(int(os.environ.get("GRARE_TRAIN_LOG_EVERY_BATCHES", "50")), 0)
    progress_every_sec = max(float(os.environ.get("GRARE_TRAIN_LOG_EVERY_SEC", "30")), 0.0)
    last_progress = epoch_start
    processed_batches = 0

    has_obj = criteria.lambda_obj_id > 0
    has_coll = criteria.lambda_collision > 0
    has_empty = criteria.lambda_empty > 0

    for batch_idx, batch in enumerate(loader, start=1):
        processed_batches = batch_idx
        pose_features = batch["pose_features"].to(device, non_blocking=True)
        local_cloud = batch["local_cloud"].to(device, non_blocking=True)
        cloud_mask = batch.get("cloud_mask")
        object_cloud = batch.get("object_cloud")
        object_pooled = batch.get("object_pooled")
        object_assignments = batch.get("object_assignments")
        is_collision = batch.get("is_collision")
        is_empty = batch.get("is_empty")
        if cloud_mask is not None:
            cloud_mask = cloud_mask.to(device, non_blocking=True)
        if object_cloud is not None and object_cloud.numel() > 0:
            object_cloud = object_cloud.to(device, non_blocking=True)
        else:
            object_cloud = None
        if object_pooled is not None and object_pooled.numel() > 0:
            object_pooled = object_pooled.to(device, non_blocking=True).float()
        else:
            object_pooled = None
        if object_assignments is not None:
            object_assignments = object_assignments.to(device, non_blocking=True).long()
        if is_collision is not None:
            is_collision = is_collision.to(device, non_blocking=True).float()
        if is_empty is not None:
            is_empty = is_empty.to(device, non_blocking=True).float()
        mu_min = batch["mu_min"].to(device, non_blocking=True)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context:
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled and amp_dtype is not None
                else nullcontext()
            )
            with autocast_context:
                model_kwargs: dict = {}
                if cloud_mask is not None and cloud_mask.shape[-1] > 0:
                    model_kwargs["cloud_mask"] = cloud_mask
                if object_cloud is not None:
                    model_kwargs["object_cloud"] = object_cloud
                if object_pooled is not None:
                    model_kwargs["object_pooled"] = object_pooled
                heads = model(pose_features, local_cloud, **model_kwargs)
                preds = heads["score"]
                target = _build_quality_target(
                    mu_min,
                    success_mu_thresh,
                )
                target_finite = torch.isfinite(target)
                if not bool(torch.all(target_finite)):
                    keep = target_finite
                    preds = preds[keep]
                    target = target[keep]
                    if preds.numel() == 0:
                        continue
                    if heads.get("is_collision_logit") is not None:
                        heads["is_collision_logit"] = heads["is_collision_logit"][keep]
                    if heads.get("is_empty_logit") is not None:
                        heads["is_empty_logit"] = heads["is_empty_logit"][keep]
                    if heads.get("obj_id_logit") is not None:
                        heads["obj_id_logit"] = heads["obj_id_logit"][keep]
                    if is_collision is not None:
                        is_collision = is_collision[keep]
                    if is_empty is not None:
                        is_empty = is_empty[keep]
                    if object_assignments is not None:
                        object_assignments = object_assignments[keep]

                loss_score = criteria.score(preds, target).mean()
                loss = loss_score
                loss_coll_value: torch.Tensor | None = None
                loss_empty_value: torch.Tensor | None = None
                loss_obj_value: torch.Tensor | None = None
                if has_coll and is_collision is not None and heads.get("is_collision_logit") is not None:
                    loss_coll_value = criteria.coll(heads["is_collision_logit"], is_collision).mean()
                    loss = loss + criteria.lambda_collision * loss_coll_value
                if has_empty and is_empty is not None and heads.get("is_empty_logit") is not None:
                    loss_empty_value = criteria.empty(heads["is_empty_logit"], is_empty).mean()
                    loss = loss + criteria.lambda_empty * loss_empty_value
                if has_obj and object_assignments is not None and heads.get("obj_id_logit") is not None:
                    obj_loss_per_sample = criteria.obj_id(heads["obj_id_logit"], object_assignments)
                    valid_obj = (object_assignments >= 0)
                    n_valid = int(valid_obj.sum().item())
                    if n_valid > 0:
                        loss_obj_value = obj_loss_per_sample[valid_obj].mean()
                        loss = loss + criteria.lambda_obj_id * loss_obj_value
            if training and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        batch_size = int(target.numel())
        total_count += batch_size
        bs = float(batch_size)
        with torch.no_grad():
            total_loss_t += loss.detach().to(torch.float64) * bs
            total_score_t += loss_score.detach().to(torch.float64) * bs
            if loss_coll_value is not None:
                total_coll_t += loss_coll_value.detach().to(torch.float64) * bs
                coll_count_t += bs
            if loss_empty_value is not None:
                total_empty_t += loss_empty_value.detach().to(torch.float64) * bs
                empty_count_t += bs
            if loss_obj_value is not None:
                if heads.get("obj_id_logit") is not None and object_assignments is not None:
                    valid_obj = (object_assignments >= 0)
                    if valid_obj.any():
                        valid_obj_count = valid_obj.sum().to(torch.float64)
                        total_obj_t += loss_obj_value.detach().to(torch.float64) * valid_obj_count
                        obj_count_t += valid_obj_count
                        pred_obj = heads["obj_id_logit"].argmax(dim=-1)
                        obj_correct_t += (pred_obj[valid_obj] == object_assignments[valid_obj]).sum().to(torch.float64)
            pred_stats = preds.detach().to(torch.float64)
            target_stats = target.detach().to(torch.float64)
            total_pred_sum_t += pred_stats.sum()
            total_pred_sq_sum_t += torch.sum(pred_stats * pred_stats)
            total_target_sum_t += target_stats.sum()
            total_target_sq_sum_t += torch.sum(target_stats * target_stats)
            total_pred_target_sum_t += torch.sum(pred_stats * target_stats)

        now = time.perf_counter()
        if (
            (progress_every_batches > 0 and batch_idx % progress_every_batches == 0)
            or (progress_every_sec > 0 and now - last_progress >= progress_every_sec)
            or (total_batches is not None and batch_idx >= total_batches)
        ):
            elapsed = now - epoch_start
            print(json.dumps({
                "stage": "train_progress" if training else "val_progress",
                "batch": batch_idx,
                "batches": total_batches,
                "samples": total_count,
                "samples_per_sec": float(total_count) / max(elapsed, 1e-9),
            }, ensure_ascii=False), flush=True)
            last_progress = now

    denom = max(total_count, 1)
    epoch_sec = time.perf_counter() - epoch_start
    accum = torch.stack([
        total_loss_t, total_score_t, total_coll_t, total_empty_t, total_obj_t,
        total_pred_sum_t, total_pred_sq_sum_t, total_target_sum_t,
        total_target_sq_sum_t, total_pred_target_sum_t,
        obj_correct_t, obj_count_t, coll_count_t, empty_count_t,
    ]).cpu().tolist()
    (
        total_loss, total_score, total_coll, total_empty, total_obj,
        ps, psq, ts, tsq, pts, obj_correct, obj_count, coll_count, empty_count,
    ) = accum
    pred_mean, pred_std = _mean_std(ps, psq, denom)
    pred_target_corr = _correlation(ps, psq, ts, tsq, pts, denom)
    return {
        "loss": total_loss / denom,
        "primary_loss": total_score / denom,
        "loss_score": total_score / denom,
        "loss_coll": (total_coll / coll_count) if coll_count > 0 else None,
        "loss_empty": (total_empty / empty_count) if empty_count > 0 else None,
        "loss_obj": (total_obj / obj_count) if obj_count > 0 else None,
        "obj_acc_top1": (obj_correct / obj_count) if obj_count > 0 else None,
        "mae": None,
        "epoch_sec": epoch_sec,
        "samples_per_sec": float(total_count) / max(epoch_sec, 1e-9),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_target_corr": pred_target_corr,
        "score_collapse_flag": bool(total_count > 0 and pred_std <= score_collapse_std_threshold),
        "num_batches": processed_batches,
        "num_samples": total_count,
    }
