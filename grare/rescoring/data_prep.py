from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..relabeling import RelabeledCandidateDataset
from ..relabeling.manifest import summarize_manifest_records
from ..utils.benchmark_protocol import (
    archive_scene_key,
    archive_scene_key_from_path,
    archive_split_from_path,
    read_archive_meta,
)
from ..utils.experiment_logging import boolean_stats, numeric_stats


@dataclass(frozen=True)
class LoaderRuntimeConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None

    def dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


def make_worker_init_fn(seed: int):
    """Factory for a DataLoader `worker_init_fn` that re-seeds RNGs per worker.

    Without this, fork()-based workers inherit the parent's RNG state and
    spawn()-based workers start from undefined state — both can yield
    correlated batches across workers and irreproducible runs.
    """
    base_seed = int(seed)

    def _init(worker_id: int) -> None:  # pragma: no cover - exercised via DataLoader
        import random as _random
        worker_seed = base_seed + int(worker_id)
        _random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


CACHE_DEVICES = ("cpu", "cuda")


def resolve_cache_device(
    requested: str,
    train_device: torch.device,
) -> torch.device:
    """Resolve where eager dataset tensors are cached.

    The reported settings cache on CPU: the training features are far larger
    than GPU memory for EG, and CPU caching keeps DataLoader workers usable.
    """
    if requested not in CACHE_DEVICES:
        raise ValueError(f"cache_device must be one of {CACHE_DEVICES}; got {requested!r}")
    if requested == "cuda" and train_device.type == "cuda" and torch.cuda.is_available():
        return train_device
    return torch.device("cpu")


def configure_loader_runtime(
    *,
    batch_size: int,
    requested_num_workers: int,
    prefetch_factor: int,
    disable_pin_memory: bool,
    disable_persistent_workers: bool,
    train_device: torch.device,
    dataset_device: torch.device,
) -> LoaderRuntimeConfig:
    if requested_num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    num_workers = requested_num_workers
    pin_memory = (not disable_pin_memory) and train_device.type == "cuda" and dataset_device.type == "cpu"
    persistent_workers = num_workers > 0 and not disable_persistent_workers
    if dataset_device.type != "cpu":
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    return LoaderRuntimeConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def summarize_dataset(dataset: RelabeledCandidateDataset) -> dict[str, object]:
    if getattr(dataset, "is_lazy_archive_dataset", False) or getattr(dataset, "is_packed_dataset", False):
        manifest_records = getattr(dataset, "manifest_records", None)
        manifest_summary = (
            summarize_manifest_records(manifest_records)
            if manifest_records is not None
            else None
        )
        return {
            "summary_mode": "packed_mmap" if getattr(dataset, "is_packed_dataset", False) else "lazy_archive",
            "num_archives": len(getattr(dataset, "archive_paths", [])),
            "num_samples": (
                dataset.__len_samples__() if hasattr(dataset, "__len_samples__") else None
            ),
            "note": "statistics are skipped to avoid scanning all archives at setup",
            "manifest_summary": manifest_summary,
        }
    base_scores = dataset.base_score.detach().cpu().numpy()
    mu_min = dataset.mu_min.detach().cpu().numpy()
    success_label = dataset.success_label.detach().cpu().numpy() >= 0.5
    is_collision = dataset.is_collision.detach().cpu().numpy() >= 0.5
    is_empty = dataset.is_empty.detach().cpu().numpy() >= 0.5
    local_cloud = dataset.local_cloud.detach().cpu().numpy()
    local_cloud_points = np.count_nonzero(np.any(local_cloud != 0, axis=-1), axis=1)
    finite_mu = mu_min[np.isfinite(mu_min)]
    return {
        "base_scores": numeric_stats(base_scores),
        "mu_min": numeric_stats(mu_min),
        "mu_min_finite": numeric_stats(finite_mu),
        "success_label": boolean_stats(success_label),
        "is_collision": boolean_stats(is_collision),
        "is_empty": boolean_stats(is_empty),
        "local_cloud_points": numeric_stats(local_cloud_points),
    }


def split_archives(
    archives: list[Path],
    *,
    val_ratio: float,
    mode: str,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, object]]:
    if val_ratio <= 0 or len(archives) <= 1:
        scene_keys = sorted({_fast_archive_scene_key(path) for path in archives})
        return archives, [], {
            "num_train_scenes": len(scene_keys),
            "num_val_scenes": 0,
            "train_scene_keys": scene_keys,
            "val_scene_keys": [],
        }

    if mode == "scene":
        groups: dict[str, list[Path]] = {}
        for archive_path in archives:
            groups.setdefault(_fast_archive_scene_key(archive_path), []).append(archive_path)
        group_keys = sorted(groups)
        if len(group_keys) <= 1:
            return archives, [], {
                "num_train_scenes": len(group_keys),
                "num_val_scenes": 0,
                "train_scene_keys": group_keys,
                "val_scene_keys": [],
            }
        rng = np.random.default_rng(seed)
        shuffled = list(group_keys)
        rng.shuffle(shuffled)
        val_group_count = min(max(int(round(len(group_keys) * val_ratio)), 1), len(group_keys) - 1)
        val_keys = set(shuffled[:val_group_count])
        train_archives = [path for key in shuffled[val_group_count:] for path in groups[key]]
        val_archives = [path for key in shuffled[:val_group_count] for path in groups[key]]
        train_scene_keys = sorted(set(shuffled[val_group_count:]))
        val_scene_keys = sorted(val_keys)
        return train_archives, val_archives, {
            "num_train_scenes": len(train_scene_keys),
            "num_val_scenes": len(val_scene_keys),
            "train_scene_keys": train_scene_keys,
            "val_scene_keys": val_scene_keys,
        }

    if mode == "archive":
        rng = np.random.default_rng(seed)
        shuffled = list(archives)
        rng.shuffle(shuffled)
        val_archive_count = min(max(int(round(len(shuffled) * val_ratio)), 1), len(shuffled) - 1)
        val_archives = sorted(shuffled[:val_archive_count])
        train_archives = sorted(shuffled[val_archive_count:])
        val_scene_keys = sorted({_fast_archive_scene_key(path) for path in val_archives})
        train_scene_keys = sorted({_fast_archive_scene_key(path) for path in train_archives})
        return train_archives, val_archives, {
            "num_train_scenes": len(train_scene_keys),
            "num_val_scenes": len(val_scene_keys),
            "train_scene_keys": train_scene_keys,
            "val_scene_keys": val_scene_keys,
        }

    raise ValueError(f"Unsupported val split mode: {mode}")




def validate_training_archives(
    archives: list[Path],
) -> None:
    for archive_path in archives:
        split = archive_split_from_path(archive_path)
        if split == "train":
            continue
        if split == "test":
            raise ValueError(
                "training archives must come from the benchmark training split; "
                f"test-split archive found: {archive_path}"
            )
        meta = read_archive_meta(archive_path)
        if str(meta.get("split", "")) == "test":
            raise ValueError(
                "training archives must come from the benchmark training split; "
                f"test-split metadata found in: {archive_path}"
            )


def _fast_archive_scene_key(archive_path: str | Path) -> str:
    scene_key = archive_scene_key_from_path(archive_path)
    return scene_key if scene_key is not None else archive_scene_key(archive_path)
