from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .model import GraspRescorer, RescorerConfig

if TYPE_CHECKING:
    from .trainer import TrainerConfig


def _is_readable_file(path: str | Path) -> bool:
    try:
        return Path(path).expanduser().is_file()
    except OSError:
        return False


def _resolve_object_pmae_ckpt(path: object) -> str:
    """Return a usable Point-MAE ckpt path for old published checkpoints.

    Published rescorer checkpoints may contain the absolute Point-MAE path from
    the training machine.  At inference time the full rescorer state_dict is
    loaded immediately afterwards, so the pretrained path is only a convenient
    initializer.  Prefer the saved path when it exists; otherwise use the local
    environment path if available, and finally disable the initializer instead
    of failing before state_dict loading.
    """
    saved = str(path or "")
    if saved and _is_readable_file(saved):
        return saved
    for env_name in ("GRARE_POINT_MAE_CKPT",):
        candidate = os.environ.get(env_name, "")
        if candidate and _is_readable_file(candidate):
            return candidate
    return ""


_HEAD_PREFIXES = ("score_head.", "coll_head.", "empty_head.", "obj_head.")


def _normalize_head_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept output-head keys saved with an inner ``body`` submodule.

    Checkpoints produced before the heads were simplified to a plain
    ``nn.Linear`` store them as ``score_head.body.weight``. The tensor shapes
    are unchanged, so dropping the ``body.`` level is an exact remapping.
    """
    remapped = {}
    for key, value in state_dict.items():
        for prefix in _HEAD_PREFIXES:
            if key.startswith(prefix + "body."):
                key = prefix + key[len(prefix) + len("body.") :]
                break
        remapped[key] = value
    return remapped


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> GraspRescorer:
    """Load a rescorer checkpoint.

    The checkpoint stores its own ``model_config`` dict; the loader filters
    it against ``RescorerConfig`` fields so old checkpoints with extra keys
    still load.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer_config_dict = checkpoint.get("trainer_config", {}) or {}
    ckpt_model_config = checkpoint.get("model_config") or {}
    valid_keys = set(RescorerConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in ckpt_model_config.items() if k in valid_keys}
    if "object_pmae_ckpt" in filtered:
        filtered["object_pmae_ckpt"] = _resolve_object_pmae_ckpt(filtered["object_pmae_ckpt"])
    config = RescorerConfig(**filtered)
    model = GraspRescorer(config)
    state_dict = _normalize_head_keys(checkpoint["model_state_dict"])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"failed to load rescorer checkpoint {checkpoint_path!s}. "
            "The checkpoint state_dict does not match the published GraRe "
            "candidate + local + object architecture."
        ) from exc
    model._checkpoint_trainer_config = trainer_config_dict
    model.to(device)
    model.eval()
    return model


def _save_checkpoint(
    checkpoint_path: Path,
    model: GraspRescorer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainerConfig,
    metrics: dict[str, float],
    scheduler: object | None = None,
    scaler: object | None = None,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    global_step: int | None = None,
) -> None:
    payload: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "trainer_config": asdict(config),
        "metrics": metrics,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_metric is not None:
        payload["best_metric"] = float(best_metric)
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if global_step is not None:
        payload["global_step"] = int(global_step)
    torch.save(payload, checkpoint_path)


def load_training_state(
    checkpoint_path: str | Path,
    model: GraspRescorer,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: object | None = None,
    scaler: object | None = None,
    map_location: str = "cpu",
) -> dict:
    """Restore mid-training state in-place; returns metadata for the loop.

    Older checkpoints lacking scheduler/scaler/best_metric fall back to
    sensible defaults so a fresh-clone user can still resume an old `best.pt`.
    The trainer is the only caller and gates this behind an explicit
    `resume=True` flag — verifying that resume is the user's intent.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"failed to resume from checkpoint {checkpoint_path!s}. "
            "Checkpoints trained with the removed global/scene-token tier are "
            "not compatible with the current Pose + Local + Object model."
        ) from exc
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_metric": float(ckpt["best_metric"]) if "best_metric" in ckpt else float("inf"),
        "best_epoch": int(ckpt["best_epoch"]) if "best_epoch" in ckpt else -1,
        "global_step": int(ckpt.get("global_step", 0)),
        "metrics": ckpt.get("metrics", {}),
    }
