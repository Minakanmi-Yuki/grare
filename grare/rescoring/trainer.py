from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .checkpoints import _save_checkpoint, load_model_checkpoint, load_training_state
from .model import GraspRescorer
from .training_core import MultiTaskCriteria, _resolve_amp, run_epoch
from ..utils.experiment_logging import append_jsonl


@dataclass(frozen=True)
class TrainerConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 40
    min_epochs: int = 20
    min_optimizer_steps: int = 0
    min_checkpoint_steps: int = 0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0
    lr_scheduler_factor: float = 0.7
    lr_scheduler_patience: int = 4
    min_lr: float = 1e-5
    scheduler_warmup_steps: int = 0
    checkpoint_interval_steps: int = 0
    local_health_max_batches: int = 0
    local_health_seed: int = 20260710
    min_local_raw_rms: float = 0.0
    min_local_shuffle_ratio: float = 0.0
    grad_clip_norm: float | None = 1.0
    score_collapse_std_threshold: float = 1e-4
    success_mu_thresh: float = 0.4
    seed: int = 7
    device: str = "cuda"
    amp: str = "auto"
    resume_from: str | None = None
    lambda_collision: float = 0.10
    lambda_empty: float = 0.05
    lambda_obj_id: float = 0.05
    num_object_classes: int = 88
    monitor_metric: str = "loss_score"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _local_health_requested(config: TrainerConfig) -> bool:
    return bool(
        config.local_health_max_batches > 0
        or config.min_local_raw_rms > 0
        or config.min_local_shuffle_ratio > 0
    )


def _evaluate_local_health(
    model: GraspRescorer,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    seed: int,
) -> dict[str, float | int]:
    """Evaluate the GN-Kinect checkpoint gate reported in the supplement."""
    if max_batches <= 0:
        raise ValueError("local_health_max_batches must be positive when the gate is enabled")

    raw_features: list[torch.Tensor] = []
    normal_scores: list[torch.Tensor] = []
    score_deltas: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                pose_features = batch["pose_features"].to(device, non_blocking=True)
                local_cloud = batch["local_cloud"].to(device, non_blocking=True)
                cloud_mask = batch.get("cloud_mask")
                if cloud_mask is None:
                    cloud_mask = torch.any(local_cloud != 0, dim=-1)
                else:
                    cloud_mask = cloud_mask.to(device, non_blocking=True)
                if pose_features.shape[0] < 2:
                    continue

                model_kwargs: dict[str, torch.Tensor] = {"cloud_mask": cloud_mask}
                object_pooled = batch.get("object_pooled")
                if object_pooled is not None and object_pooled.numel() > 0:
                    model_kwargs["object_pooled"] = object_pooled.to(
                        device, non_blocking=True
                    ).float()
                else:
                    object_cloud = batch.get("object_cloud")
                    if object_cloud is not None and object_cloud.numel() > 0:
                        model_kwargs["object_cloud"] = object_cloud.to(
                            device, non_blocking=True
                        )

                raw_geometry = model.geometry_encoder(local_cloud, cloud_mask)
                normal = model(pose_features, local_cloud, **model_kwargs)["score"]
                permutation = torch.randperm(
                    pose_features.shape[0], generator=generator
                ).to(device)
                shuffled_kwargs = dict(model_kwargs)
                shuffled_kwargs["cloud_mask"] = cloud_mask[permutation]
                shuffled = model(
                    pose_features,
                    local_cloud[permutation],
                    **shuffled_kwargs,
                )["score"]
                raw_features.append(raw_geometry.detach().float().cpu())
                normal_scores.append(normal.detach().float().cpu())
                score_deltas.append((normal - shuffled).detach().float().cpu())
    finally:
        model.train(was_training)

    if not raw_features:
        raise RuntimeError("local-health gate did not observe a batch with at least two samples")
    raw = torch.cat(raw_features, dim=0)
    scores = torch.cat(normal_scores, dim=0)
    deltas = torch.cat(score_deltas, dim=0)
    raw_centered_rms = torch.sqrt(torch.mean((raw - raw.mean(dim=0, keepdim=True)).square()))
    score_std = torch.std(scores, unbiased=False)
    score_delta_rms = torch.sqrt(torch.mean(deltas.square()))
    shuffle_ratio = score_delta_rms / score_std.clamp_min(1e-12)
    return {
        "local_raw_centered_rms": float(raw_centered_rms.item()),
        "local_score_std": float(score_std.item()),
        "local_score_shuffle_delta_rms": float(score_delta_rms.item()),
        "local_score_shuffle_ratio": float(shuffle_ratio.item()),
        "local_health_num_samples": int(raw.shape[0]),
    }


def train_model(
    model: GraspRescorer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TrainerConfig,
    save_dir: str | Path,
    *,
    tensorboard_dir: str | Path | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    if _local_health_requested(config) and val_loader is None:
        raise ValueError("local-health checkpoint gate requires a validation loader")
    if _local_health_requested(config) and config.local_health_max_batches <= 0:
        raise ValueError("local_health_max_batches must be positive when the gate is enabled")

    tb_writer = None
    if tensorboard_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        except Exception as exc:  # missing tensorboard package, etc.
            print(f"[trainer] TensorBoard disabled: {exc}", flush=True)
            tb_writer = None

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Per-batch tensor shapes are stable (fixed point cloud size, fixed
        # batch size with drop_last semantics), so cuDNN's autotuner can lock
        # in the fastest kernel after the first few batches.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model.to(device)

    criterion: nn.Module = nn.SmoothL1Loss(reduction="none")

    multi_task_criteria = MultiTaskCriteria(
        score=criterion,
        lambda_collision=float(config.lambda_collision),
        lambda_empty=float(config.lambda_empty),
        lambda_obj_id=float(config.lambda_obj_id),
    )

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
    )
    amp_enabled, amp_dtype = _resolve_amp(config.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    history_path = save_dir / "history.json"
    history_jsonl_path = save_dir / "history.jsonl"
    summary_path = save_dir / "summary.json"
    history_jsonl_path.write_text("", encoding="utf-8")
    max_epoch_limit = None if config.max_epochs <= 0 else int(config.max_epochs)
    stop_reason = "max_epochs_reached" if max_epoch_limit is not None else "early_stop"
    epoch = 0
    train_started_perf = time.perf_counter()
    total_train_epoch_sec = 0.0
    total_val_epoch_sec = 0.0
    total_train_steps = 0
    total_val_steps = 0
    status = "running"

    if config.resume_from is not None:
        resume_path = Path(config.resume_from)
        if resume_path.is_file():
            state = load_training_state(
                resume_path,
                model,
                optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                map_location=str(device),
            )
            best_metric = state["best_metric"]
            best_epoch = state["best_epoch"]
            epoch = state["epoch"] + 1  # next epoch to run
            total_train_steps = state["global_step"]
            print(
                json.dumps(
                    {
                        "stage": "train_resume",
                        "checkpoint": str(resume_path),
                        "resume_from_epoch": epoch,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "global_step": total_train_steps,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    _write_training_summary(
        summary_path,
        status=status,
        history=history,
        config=config,
        device=device,
        started_perf=train_started_perf,
        total_train_epoch_sec=total_train_epoch_sec,
        total_val_epoch_sec=total_val_epoch_sec,
        total_train_steps=total_train_steps,
        total_val_steps=total_val_steps,
        best_metric=best_metric,
        best_epoch=best_epoch,
        stop_reason=None,
        tensorboard_dir=tensorboard_dir,
        history_path=history_path,
        history_jsonl_path=history_jsonl_path,
        summary_extra=summary_extra,
    )

    try:
        while True:
            if max_epoch_limit is not None and epoch >= max_epoch_limit:
                break
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criteria=multi_task_criteria,
                optimizer=optimizer,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                training=True,
                grad_clip_norm=config.grad_clip_norm,
                success_mu_thresh=config.success_mu_thresh,
                score_collapse_std_threshold=config.score_collapse_std_threshold,
            )
            val_metrics = None
            if val_loader is not None:
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criteria=multi_task_criteria,
                    optimizer=None,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=None,
                    training=False,
                    grad_clip_norm=None,
                    success_mu_thresh=config.success_mu_thresh,
                    score_collapse_std_threshold=config.score_collapse_std_threshold,
                )
            epoch_train_steps = int(train_metrics.get("num_batches") or 0)
            completed_train_steps = total_train_steps + epoch_train_steps
            local_health_metrics: dict[str, float | int] = {}
            if val_loader is not None and _local_health_requested(config):
                local_health_metrics = _evaluate_local_health(
                    model,
                    val_loader,
                    device=device,
                    max_batches=int(config.local_health_max_batches),
                    seed=int(config.local_health_seed),
                )
            local_raw_rms = float(local_health_metrics.get("local_raw_centered_rms", 0.0))
            local_shuffle_ratio = float(
                local_health_metrics.get("local_score_shuffle_ratio", 0.0)
            )
            local_health_pass = bool(
                local_raw_rms >= float(config.min_local_raw_rms)
                and local_shuffle_ratio >= float(config.min_local_shuffle_ratio)
            )
            checkpoint_steps_ready = completed_train_steps >= max(int(config.min_checkpoint_steps), 0)
            checkpoint_eligible = bool(checkpoint_steps_ready and local_health_pass)

            metric_key = config.monitor_metric
            if val_metrics is not None:
                if metric_key not in val_metrics or val_metrics[metric_key] is None:
                    raise KeyError(
                        f"monitor_metric={metric_key!r} not present in val_metrics; "
                        f"available keys: {sorted(val_metrics.keys())}"
                    )
                monitor_name = f"val_{metric_key}"
                monitor_value = float(val_metrics[metric_key])
            else:
                if metric_key not in train_metrics or train_metrics[metric_key] is None:
                    raise KeyError(
                        f"monitor_metric={metric_key!r} not present in train_metrics; "
                        f"available keys: {sorted(train_metrics.keys())}"
                    )
                monitor_name = f"train_{metric_key}"
                monitor_value = float(train_metrics[metric_key])
            checkpoint_metrics = dict(val_metrics if val_metrics is not None else train_metrics)
            checkpoint_metrics.update(local_health_metrics)
            metric_improved = monitor_value < (best_metric - config.early_stop_min_delta)
            improved = bool(metric_improved and checkpoint_eligible)
            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                epochs_without_improve = 0
                _save_checkpoint(
                    save_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    checkpoint_metrics,
                    scheduler=scheduler,
                    scaler=scaler if scaler.is_enabled() else None,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    global_step=completed_train_steps,
                )
            elif checkpoint_eligible:
                epochs_without_improve += 1
            else:
                epochs_without_improve = 0

            periodic_checkpoint_path = None
            checkpoint_interval = max(int(config.checkpoint_interval_steps), 0)
            if (
                checkpoint_interval > 0
                and completed_train_steps // checkpoint_interval
                > total_train_steps // checkpoint_interval
            ):
                periodic_checkpoint_path = save_dir / f"step_{completed_train_steps:08d}.pt"
                _save_checkpoint(
                    periodic_checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    config,
                    checkpoint_metrics,
                    scheduler=scheduler,
                    scaler=scaler if scaler.is_enabled() else None,
                    best_metric=None if best_metric == float("inf") else best_metric,
                    best_epoch=None if best_epoch < 0 else best_epoch,
                    global_step=completed_train_steps,
                )

            scheduler_steps_ready = completed_train_steps >= max(int(config.scheduler_warmup_steps), 0)
            current_lr = float(optimizer.param_groups[0]["lr"])
            record = {
                "epoch": epoch,
                "global_step": completed_train_steps,
                "lr": current_lr,
                "monitor_name": monitor_name,
                "monitor_value": monitor_value,
                "metric_improved": metric_improved,
                "improved": improved,
                "checkpoint_steps_ready": checkpoint_steps_ready,
                "checkpoint_eligible": checkpoint_eligible,
                "local_health_pass": local_health_pass,
                "scheduler_stepped": scheduler_steps_ready,
                "periodic_checkpoint": (
                    None if periodic_checkpoint_path is None else str(periodic_checkpoint_path)
                ),
                "best_epoch": best_epoch,
                "best_monitor_value_so_far": (
                    None if best_metric == float("inf") else float(best_metric)
                ),
                "epochs_without_improve": epochs_without_improve,
                "early_stop_patience": config.early_stop_patience,
                "score_collapse_std_threshold": config.score_collapse_std_threshold,
                **local_health_metrics,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
            if val_metrics is not None:
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)
            total_train_epoch_sec += float(train_metrics.get("epoch_sec") or 0.0)
            total_train_steps = completed_train_steps
            if val_metrics is not None:
                total_val_epoch_sec += float(val_metrics.get("epoch_sec") or 0.0)
                total_val_steps += int(val_metrics.get("num_batches") or 0)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with history_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)
            append_jsonl(history_jsonl_path, record)
            _write_training_summary(
                summary_path,
                status=status,
                history=history,
                config=config,
                device=device,
                started_perf=train_started_perf,
                total_train_epoch_sec=total_train_epoch_sec,
                total_val_epoch_sec=total_val_epoch_sec,
                total_train_steps=total_train_steps,
                total_val_steps=total_val_steps,
                best_metric=best_metric,
                best_epoch=best_epoch,
                stop_reason=None,
                tensorboard_dir=tensorboard_dir,
                history_path=history_path,
                history_jsonl_path=history_jsonl_path,
                summary_extra=summary_extra,
            )
            if tb_writer is not None:
                for k, v in record.items():
                    if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                        tb_writer.add_scalar(k, float(v), epoch)
                    elif isinstance(v, bool):
                        tb_writer.add_scalar(k, 1.0 if v else 0.0, epoch)
                tb_writer.flush()
            if scheduler_steps_ready:
                scheduler.step(monitor_value)

            if (
                config.early_stop_patience >= 0
                and not improved
                and epoch + 1 >= max(config.min_epochs, 1)
                and completed_train_steps >= max(config.min_optimizer_steps, 0)
                and best_epoch >= 0
                and epochs_without_improve >= config.early_stop_patience
            ):
                stop_reason = f"no_{monitor_name}_improvement_for_{config.early_stop_patience}_epochs"
                break
            epoch += 1
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        _write_training_summary(
            summary_path,
            status=status,
            history=history,
            config=config,
            device=device,
            started_perf=train_started_perf,
            total_train_epoch_sec=total_train_epoch_sec,
            total_val_epoch_sec=total_val_epoch_sec,
            total_train_steps=total_train_steps,
            total_val_steps=total_val_steps,
            best_metric=best_metric,
            best_epoch=best_epoch,
            stop_reason=stop_reason,
            tensorboard_dir=tensorboard_dir,
            history_path=history_path,
            history_jsonl_path=history_jsonl_path,
            summary_extra=summary_extra,
        )
        _close_tb(tb_writer)
        raise
    except BaseException as exc:
        status = "failed"
        stop_reason = f"{type(exc).__name__}: {exc}"
        _write_training_summary(
            summary_path,
            status=status,
            history=history,
            config=config,
            device=device,
            started_perf=train_started_perf,
            total_train_epoch_sec=total_train_epoch_sec,
            total_val_epoch_sec=total_val_epoch_sec,
            total_train_steps=total_train_steps,
            total_val_steps=total_val_steps,
            best_metric=best_metric,
            best_epoch=best_epoch,
            stop_reason=stop_reason,
            tensorboard_dir=tensorboard_dir,
            history_path=history_path,
            history_jsonl_path=history_jsonl_path,
            summary_extra=summary_extra,
        )
        _close_tb(tb_writer)
        raise
    _close_tb(tb_writer)
    final_summary = _write_training_summary(
        summary_path,
        status=status,
        history=history,
        config=config,
        device=device,
        started_perf=train_started_perf,
        total_train_epoch_sec=total_train_epoch_sec,
        total_val_epoch_sec=total_val_epoch_sec,
        total_train_steps=total_train_steps,
        total_val_steps=total_val_steps,
        best_metric=best_metric,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        tensorboard_dir=tensorboard_dir,
        history_path=history_path,
        history_jsonl_path=history_jsonl_path,
        summary_extra=summary_extra,
    )
    return {
        "best_monitor_name": (f"val_{config.monitor_metric}" if val_loader is not None else f"train_{config.monitor_metric}"),
        "best_val_loss": (
            None if val_loader is None or best_metric == float("inf") else best_metric
        ),
        "best_train_loss": (
            None if val_loader is not None or best_metric == float("inf") else best_metric
        ),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "stopped_epoch": epoch,
        "stop_reason": stop_reason,
        "early_stop_triggered": stop_reason != "max_epochs_reached",
        "history_path": str(history_path),
        "history_jsonl_path": str(history_jsonl_path),
        "summary_path": str(summary_path),
        "summary": final_summary,
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else None,
    }


def _close_tb(tb_writer: Any | None) -> None:
    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:
            pass


def _write_training_summary(
    path: Path,
    *,
    status: str,
    history: list[dict[str, Any]],
    config: TrainerConfig,
    device: torch.device,
    started_perf: float,
    total_train_epoch_sec: float,
    total_val_epoch_sec: float,
    total_train_steps: int,
    total_val_steps: int,
    best_metric: float,
    best_epoch: int,
    stop_reason: str | None,
    tensorboard_dir: str | Path | None,
    history_path: Path,
    history_jsonl_path: Path,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last = history[-1] if history else {}
    has_val = any("val_loss" in row for row in history)
    best_monitor_name = (f"val_{config.monitor_metric}" if has_val else f"train_{config.monitor_metric}")
    elapsed_sec = time.perf_counter() - started_perf
    train_samples = sum(int(row.get("train_num_samples") or 0) for row in history)
    val_samples = sum(int(row.get("val_num_samples") or 0) for row in history)
    summary = {
        "status": status,
        "stop_reason": stop_reason,
        "training_seconds": float(total_train_epoch_sec),
        "validation_seconds": float(total_val_epoch_sec),
        "total_seconds": float(elapsed_sec),
        "num_epochs": len(history),
        "num_steps": int(total_train_steps),
        "num_val_steps": int(total_val_steps),
        "num_train_samples_seen": int(train_samples),
        "num_val_samples_seen": int(val_samples),
        "best_epoch": int(best_epoch),
        "best_monitor_name": best_monitor_name,
        "best_loss": None if best_metric == float("inf") else float(best_metric),
        "best_val_loss": None if not has_val or best_metric == float("inf") else float(best_metric),
        "best_train_loss": None if has_val or best_metric == float("inf") else float(best_metric),
        "final_epoch": last.get("epoch"),
        "final_train_loss": last.get("train_loss"),
        "final_val_loss": last.get("val_loss"),
        "final_lr": last.get("lr"),
        "epochs_without_improve": last.get("epochs_without_improve"),
        "early_stop_patience": config.early_stop_patience,
        "min_epochs": config.min_epochs,
        "min_optimizer_steps": config.min_optimizer_steps,
        "min_checkpoint_steps": config.min_checkpoint_steps,
        "max_epochs": config.max_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm,
        "scheduler_warmup_steps": config.scheduler_warmup_steps,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "local_health_max_batches": config.local_health_max_batches,
        "min_local_raw_rms": config.min_local_raw_rms,
        "min_local_shuffle_ratio": config.min_local_shuffle_ratio,
        "amp": config.amp,
        "device": str(device),
        "success_mu_thresh": config.success_mu_thresh,
        "history_path": str(history_path),
        "history_jsonl_path": str(history_jsonl_path),
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else None,
    }
    if summary_extra:
        summary.update(summary_extra)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
