#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from grare.relabeling import (
    ArchiveRelabeledCandidateDataset,
    PackedRelabeledCandidateDataset,
    RelabeledCandidateDataset,
    collate_archive_batches,
)
from grare.relabeling.manifest import (
    default_manifest_path,
    load_manifest_records,
    manifest_records_by_archive_path,
    validate_manifest_coverage,
)
from grare.rescoring.data_prep import (
    configure_loader_runtime,
    make_worker_init_fn,
    resolve_cache_device,
    split_archives,
    summarize_dataset,
    validate_training_archives,
)
from grare.rescoring import GraspRescorer, RescorerConfig, TrainerConfig, train_model
from grare.rescoring.trainer import set_seed
from grare.utils.experiment_logging import timestamp
from grare.utils.benchmark_protocol import filter_archive_paths_by_camera

class ArchiveIndexBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        archive_counts: list[int],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.archive_counts = [int(count) for count in archive_counts]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        order = list(range(len(self.archive_counts)))
        if self.shuffle and len(order) > 1:
            order = torch.randperm(len(order), generator=generator).tolist()

        batch: list[int] = []
        batch_size = 0
        for archive_idx in order:
            count = max(int(self.archive_counts[int(archive_idx)]), 0)
            if count <= 0:
                continue
            if batch and batch_size + count > self.batch_size:
                yield batch
                batch = []
                batch_size = 0
            batch.append(int(archive_idx))
            batch_size += count
            if count >= self.batch_size:
                yield batch
                batch = []
                batch_size = 0
        if batch:
            yield batch
        self._epoch += 1

    def __len__(self) -> int:
        count = 0
        batch_size = 0
        for archive_count in self.archive_counts:
            archive_count = max(int(archive_count), 0)
            if archive_count <= 0:
                continue
            if batch_size > 0 and batch_size + archive_count > self.batch_size:
                count += 1
                batch_size = 0
            batch_size += archive_count
            if archive_count >= self.batch_size:
                count += 1
                batch_size = 0
        if batch_size > 0:
            count += 1
        return count


class PackedContiguousBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        archive_counts: list[int],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.archive_counts = [int(count) for count in archive_counts]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._length = sum(max(int(count), 0) for count in self.archive_counts)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        chunk_starts = list(range(0, self._length, self.batch_size))
        if self.shuffle and len(chunk_starts) > 1:
            perm = torch.randperm(len(chunk_starts), generator=generator).tolist()
            chunk_starts = [chunk_starts[i] for i in perm]
        for chunk_start in chunk_starts:
            chunk_stop = min(chunk_start + self.batch_size, self._length)
            if self.drop_last and chunk_stop - chunk_start < self.batch_size:
                continue
            yield list(range(chunk_start, chunk_stop))
        self._epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return self._length // self.batch_size
        if self._length <= 0:
            return 0
        return (self._length + self.batch_size - 1) // self.batch_size


class PackedArchiveMixedBatchSampler(Sampler[list[int]]):
    """Mix complete archive ranges while retaining vectorized mmap fetches.

    Packed logical indices are contiguous within each archive. Randomizing the
    archive order, then filling fixed-size batches from those ranges, mixes
    frames/scenes without constructing a multi-million-element sample
    permutation or falling back to per-sample Python dataset access.
    """

    def __init__(
        self,
        archive_counts: list[int],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.archive_counts = [max(int(count), 0) for count in archive_counts]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._offsets = [0]
        for count in self.archive_counts:
            self._offsets.append(self._offsets[-1] + count)
        self._length = int(self._offsets[-1])

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        order = list(range(len(self.archive_counts)))
        if self.shuffle and len(order) > 1:
            order = torch.randperm(len(order), generator=generator).tolist()

        batch: list[int] = []
        for archive_idx in order:
            archive_start = self._offsets[archive_idx]
            archive_stop = self._offsets[archive_idx + 1]
            cursor = archive_start
            while cursor < archive_stop:
                take = min(self.batch_size - len(batch), archive_stop - cursor)
                batch.extend(range(cursor, cursor + take))
                cursor += take
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch
        self._epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return self._length // self.batch_size
        if self._length <= 0:
            return 0
        return (self._length + self.batch_size - 1) // self.batch_size


def _identity_batch(batch):
    return batch


def _build_seeded_model(config: RescorerConfig, seed: int) -> GraspRescorer:
    set_seed(int(seed))
    return GraspRescorer(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rescoring model from relabeled candidate archives.")
    parser.add_argument("--input-root", required=True, nargs="+",
                        help="One or more relabeled-candidate directories; when multiple are passed the archives are concatenated.")
    parser.add_argument(
        "--camera",
        default=None,
        help="Optional camera filter, e.g. kinect or realsense. Archives from other cameras are ignored.",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--tensorboard-dir", default=None,
                        help="If set, write per-epoch SummaryWriter scalars (loss, "
                             "learning rate, and score correlation) under this dir.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-split-mode", choices=("scene", "archive"), default="scene")
    parser.add_argument("--val-split-seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=40, help="Use 0 or a negative value for no hard epoch limit.")
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument(
        "--min-optimizer-steps",
        type=int,
        default=0,
        help="Do not allow early stopping before this many optimizer steps.",
    )
    parser.add_argument(
        "--min-checkpoint-steps",
        type=int,
        default=0,
        help="Do not select best.pt before this many optimizer steps.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.7)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler-warmup-steps",
        type=int,
        default=0,
        help="Do not step ReduceLROnPlateau before this optimizer-step count.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=0,
        help="Save a resumable step_*.pt checkpoint at approximately this interval; 0 disables.",
    )
    parser.add_argument("--local-health-max-batches", type=int, default=0)
    parser.add_argument("--local-health-seed", type=int, default=20260710)
    parser.add_argument("--min-local-raw-rms", type=float, default=0.0)
    parser.add_argument("--min-local-shuffle-ratio", type=float, default=0.0)
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="loss_score",
        choices=["loss", "primary_loss", "loss_score"],
        help="Validation metric used for checkpoint selection and scheduling.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint (typically a previous best.pt) to resume "
             "training from. Restores model + optimizer + scheduler + scaler "
             "state and the best-metric tracker. Cold-start when omitted.",
    )
    parser.add_argument("--shell-attn-n-shells", type=int, default=4)
    parser.add_argument("--shell-attn-heads", type=int, default=4)
    parser.add_argument("--shell-attn-layers", type=int, default=1)
    parser.add_argument("--shell-attn-per-point-dim", type=int, default=3,
                        help="Per-point input dim fed to ShellAttn. xyz only (=3) by default.")
    parser.add_argument("--score-collapse-std-threshold", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--success-mu-thresh", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker count. Use 0 to load in the main process.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto")
    parser.add_argument("--cache-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--archive-sample-counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional per-archive candidate-count hint used by lazy archive batching. "
            "Pass one value per --input-root; the value is applied to every archive "
            "under that root. Prefer --archive-manifest for exact counts."
        ),
    )
    parser.add_argument(
        "--archive-manifest",
        nargs="+",
        default=None,
        help=(
            "Optional per-input-root manifest.jsonl paths. Use 'auto' to read "
            "manifest.jsonl from each input root. Manifest records provide true "
            "per-archive counts and label statistics without scanning every npz. "
            "In eager tensor mode they are also used to preallocate the RAM cache "
            "and avoid a high temporary concatenate peak."
        ),
    )
    parser.add_argument(
        "--require-archive-manifest",
        action="store_true",
        help="Fail instead of falling back when a lazy archive manifest is missing or stale.",
    )
    parser.add_argument(
        "--archive-batch-sampling",
        action="store_true",
        help="Pack complete relabeled archives into large batches to reduce lazy-loading overhead.",
    )
    parser.add_argument(
        "--packed-dataset-root",
        default=None,
        help=(
            "Optional root produced by grare-pack. "
            "When set, sample tensors are read from mmap-backed NPY arrays "
            "instead of eager private RAM tensors or lazy NPZ archives."
        ),
    )
    parser.add_argument(
        "--packed-batch-sampling",
        choices=("archive_mixed", "contiguous"),
        default="archive_mixed",
        help=(
            "Batch sampler for mmap-packed datasets. archive_mixed randomizes "
            "archive/frame ranges before filling batches; contiguous preserves "
            "scene-homogeneous chunks."
        ),
    )
    parser.add_argument(
        "--object-pooled-root",
        nargs="+",
        default=None,
        help=(
            "Optional sidecar root(s) holding object_pooled npz files with the "
            "same relative layout as --input-root. When set with lazy archive "
            "batching, training reads object_pooled from the sidecar and skips "
            "object_cloud from the main relabel archive."
        ),
    )
    parser.add_argument(
        "--require-object-pooled",
        action="store_true",
        help="Fail if any archive is missing its object_pooled sidecar/cache.",
    )
    parser.add_argument("--lambda-collision", type=float, default=0.10)
    parser.add_argument("--lambda-empty", type=float, default=0.05)
    parser.add_argument("--lambda-obj-id", type=float, default=0.05)
    parser.add_argument("--num-object-classes", type=int, default=88)
    parser.add_argument("--object-cloud-points", type=int, default=512)
    parser.add_argument("--object-hidden-dim", type=int, default=128)
    parser.add_argument("--object-pmae-ckpt", type=str, default="",
                        help="Point-MAE pretrain.pth for the frozen object encoder.")
    parser.add_argument("--object-pmae-num-group", type=int, default=32)
    parser.add_argument("--object-pmae-group-size", type=int, default=32)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-ffn-mult", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.packed_dataset_root and args.archive_batch_sampling:
        raise ValueError("--packed-dataset-root and --archive-batch-sampling are mutually exclusive")
    started_at = timestamp()
    started_perf = time.perf_counter()
    input_roots = [Path(p) for p in args.input_root]
    object_pooled_roots = _normalize_optional_roots(
        args.object_pooled_root,
        input_roots,
        option_name="--object-pooled-root",
    )
    if args.archive_sample_counts is not None and len(args.archive_sample_counts) != len(input_roots):
        raise ValueError(
            "--archive-sample-counts must provide exactly one value per --input-root "
            f"({len(args.archive_sample_counts)} != {len(input_roots)})"
        )
    if args.archive_manifest is not None and len(args.archive_manifest) not in {1, len(input_roots)}:
        raise ValueError(
            "--archive-manifest must provide one value or exactly one value per --input-root "
            f"({len(args.archive_manifest)} not in {{1, {len(input_roots)}}})"
        )
    if args.packed_dataset_root:
        archives, archive_count_hints, archive_manifest_records = _load_packed_archive_catalog(
            Path(args.packed_dataset_root),
            input_roots=input_roots,
            camera=args.camera,
        )
    else:
        archives = []
        archive_count_hints = [] if args.archive_sample_counts is not None else None
        archive_manifest_records = [] if args.archive_manifest is not None or args.require_archive_manifest else None
        records_by_root = _load_manifest_records_by_root(
            input_roots,
            args.archive_manifest,
            require_manifest=args.require_archive_manifest,
            success_mu_thresh=args.success_mu_thresh,
        )
        for root_idx, root in enumerate(input_roots):
            root_archives = filter_archive_paths_by_camera(sorted(root.glob("**/*.npz")), args.camera)
            archives.extend(root_archives)
            if archive_count_hints is not None:
                hint = int(args.archive_sample_counts[root_idx])
                archive_count_hints.extend([hint] * len(root_archives))
            if archive_manifest_records is not None:
                root_records = records_by_root.get(root.resolve())
                if root_records is None:
                    archive_manifest_records.extend([None] * len(root_archives))
                else:
                    archive_manifest_records.extend([root_records[path.resolve()] for path in root_archives])
    validate_training_archives(archives)
    train_archives, val_archives, split_meta = split_archives(
        archives,
        val_ratio=args.val_ratio,
        mode=args.val_split_mode,
        seed=args.val_split_seed,
    )
    train_count_hints, val_count_hints = _split_count_hints(
        archives,
        archive_count_hints,
        train_archives,
        val_archives,
    )
    train_manifest_records, val_manifest_records = _split_manifest_records(
        archives,
        archive_manifest_records,
        train_archives,
        val_archives,
    )
    if args.packed_dataset_root:
        dataset_mode = "packed_mmap"
        dataset_cls = PackedRelabeledCandidateDataset
    elif args.archive_batch_sampling:
        dataset_mode = "lazy_archive"
        dataset_cls = ArchiveRelabeledCandidateDataset
    else:
        dataset_mode = "eager_tensor"
        dataset_cls = RelabeledCandidateDataset
    dataset_extra: dict = {
        "load_object_cloud": True,
        "input_roots": input_roots,
    }
    if args.packed_dataset_root:
        dataset_extra["packed_root"] = Path(args.packed_dataset_root)
        dataset_extra["require_object_pooled"] = bool(args.require_object_pooled)
    elif object_pooled_roots is not None:
        dataset_extra["object_pooled_roots"] = object_pooled_roots
        dataset_extra["require_object_pooled"] = bool(args.require_object_pooled)

    print(
        json.dumps(
            {
                "stage": "train_load_start",
                "dataset_mode": dataset_mode,
                "camera": args.camera,
                "num_archives": len(archives),
                "num_train_archives": len(train_archives),
                "num_val_archives": len(val_archives),
                "archive_count_hints": args.archive_sample_counts,
                "archive_manifest": args.archive_manifest,
                "object_pooled_roots": (
                    None
                    if object_pooled_roots is None
                    else [str(path) for path in object_pooled_roots]
                ),
                "load_object_cloud": True,
                "packed_dataset_root": args.packed_dataset_root,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_dataset = dataset_cls(
        train_archives,
        success_mu_thresh=args.success_mu_thresh,
        **(
            {"manifest_records": train_manifest_records}
            if train_manifest_records is not None
            else (
                {"archive_sample_counts": train_count_hints}
                if args.archive_batch_sampling and train_count_hints is not None
                else {}
            )
        ),
            **(
                {
                    "eager_load_workers": max(1, int(args.num_workers)),
                    "progress_label": "train",
                }
                if dataset_mode == "eager_tensor"
                else {}
            ),
        **dataset_extra,
    )
    val_dataset = (
        dataset_cls(
            val_archives,
            success_mu_thresh=args.success_mu_thresh,
            **(
                {"manifest_records": val_manifest_records}
                if val_manifest_records is not None
                else (
                    {"archive_sample_counts": val_count_hints}
                    if args.archive_batch_sampling and val_count_hints is not None
                    else {}
                )
            ),
            **(
                {
                    "eager_load_workers": max(1, int(args.num_workers)),
                    "progress_label": "val",
                }
                if dataset_mode == "eager_tensor"
                else {}
            ),
            **dataset_extra,
        )
        if val_archives
        else None
    )
    dataset_bytes = train_dataset.tensor_bytes() + (val_dataset.tensor_bytes() if val_dataset is not None else 0)
    train_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_device = resolve_cache_device(args.cache_device, train_device)
    if dataset_mode == "eager_tensor" and cache_device != train_dataset.device:
        train_dataset.cache_tensors_(cache_device)
    if val_dataset is not None and dataset_mode == "eager_tensor" and cache_device != val_dataset.device:
        val_dataset.cache_tensors_(cache_device)

    train_size = (
        train_dataset.__len_samples__()
        if hasattr(train_dataset, "__len_samples__")
        else len(train_dataset)
    )
    val_size = (
        val_dataset.__len_samples__()
        if val_dataset is not None and hasattr(val_dataset, "__len_samples__")
        else (len(val_dataset) if val_dataset is not None else 0)
    )
    loader_runtime = configure_loader_runtime(
        batch_size=args.batch_size,
        requested_num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        disable_pin_memory=args.no_pin_memory,
        disable_persistent_workers=args.no_persistent_workers,
        train_device=train_device,
        dataset_device=train_dataset.device,
    )

    train_loader_kwargs = loader_runtime.dataloader_kwargs()
    worker_init_fn = (
        make_worker_init_fn(int(args.seed)) if loader_runtime.num_workers > 0 else None
    )
    packed_sampler_cls = (
        PackedArchiveMixedBatchSampler
        if args.packed_batch_sampling == "archive_mixed"
        else PackedContiguousBatchSampler
    )
    drop_last_val = bool(args.archive_batch_sampling) is False  # only safe without batch sampler
    if args.archive_batch_sampling:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ArchiveIndexBatchSampler(
                train_dataset.archive_counts,
                batch_size=args.batch_size,
                seed=args.seed,
                shuffle=True,
            ),
            collate_fn=collate_archive_batches,
            num_workers=loader_runtime.num_workers,
            pin_memory=loader_runtime.pin_memory,
            persistent_workers=loader_runtime.persistent_workers,
            worker_init_fn=worker_init_fn,
            **({"prefetch_factor": loader_runtime.prefetch_factor} if loader_runtime.prefetch_factor is not None else {}),
        )
    elif dataset_mode == "packed_mmap":
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=packed_sampler_cls(
                train_dataset.archive_counts,
                batch_size=args.batch_size,
                seed=args.seed,
                shuffle=True,
                drop_last=True,
            ),
            collate_fn=_identity_batch,
            num_workers=loader_runtime.num_workers,
            pin_memory=loader_runtime.pin_memory,
            persistent_workers=loader_runtime.persistent_workers,
            worker_init_fn=worker_init_fn,
            **({"prefetch_factor": loader_runtime.prefetch_factor} if loader_runtime.prefetch_factor is not None else {}),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            **train_loader_kwargs,
        )
    val_loader = None
    if val_dataset is not None:
        if args.archive_batch_sampling:
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=ArchiveIndexBatchSampler(
                    val_dataset.archive_counts,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    shuffle=False,
                ),
                collate_fn=collate_archive_batches,
                num_workers=loader_runtime.num_workers,
                pin_memory=loader_runtime.pin_memory,
                persistent_workers=loader_runtime.persistent_workers,
                worker_init_fn=worker_init_fn,
                **({"prefetch_factor": loader_runtime.prefetch_factor} if loader_runtime.prefetch_factor is not None else {}),
            )
        elif dataset_mode == "packed_mmap":
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=packed_sampler_cls(
                    val_dataset.archive_counts,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    shuffle=False,
                    drop_last=False,
                ),
                collate_fn=_identity_batch,
                num_workers=loader_runtime.num_workers,
                pin_memory=loader_runtime.pin_memory,
                persistent_workers=loader_runtime.persistent_workers,
                worker_init_fn=worker_init_fn,
                **({"prefetch_factor": loader_runtime.prefetch_factor} if loader_runtime.prefetch_factor is not None else {}),
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=drop_last_val,
                worker_init_fn=worker_init_fn,
                **loader_runtime.dataloader_kwargs(),
            )

    sample = train_dataset[0]
    sample_pose = sample["pose_features"]  # type: ignore[assignment]
    pose_dim = int(sample_pose.shape[-1]) if sample_pose.ndim > 1 else int(sample_pose.numel())
    config = RescorerConfig(
        pose_dim=pose_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        shell_attn_n_shells=args.shell_attn_n_shells,
        shell_attn_heads=args.shell_attn_heads,
        shell_attn_layers=args.shell_attn_layers,
        shell_attn_per_point_dim=args.shell_attn_per_point_dim,
        object_cloud_points=int(args.object_cloud_points),
        object_hidden_dim=int(args.object_hidden_dim),
        object_pmae_ckpt=str(args.object_pmae_ckpt),
        object_pmae_num_group=int(args.object_pmae_num_group),
        object_pmae_group_size=int(args.object_pmae_group_size),
        fusion_layers=int(args.fusion_layers),
        fusion_heads=int(args.fusion_heads),
        fusion_ffn_mult=int(args.fusion_ffn_mult),
        num_object_classes=int(args.num_object_classes),
    )
    trainer_config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        min_optimizer_steps=args.min_optimizer_steps,
        min_checkpoint_steps=args.min_checkpoint_steps,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        lr_scheduler_factor=args.scheduler_factor,
        lr_scheduler_patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        local_health_max_batches=args.local_health_max_batches,
        local_health_seed=args.local_health_seed,
        min_local_raw_rms=args.min_local_raw_rms,
        min_local_shuffle_ratio=args.min_local_shuffle_ratio,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        score_collapse_std_threshold=float(args.score_collapse_std_threshold),
        success_mu_thresh=args.success_mu_thresh,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        resume_from=args.resume_from,
        lambda_collision=float(args.lambda_collision),
        lambda_empty=float(args.lambda_empty),
        lambda_obj_id=float(args.lambda_obj_id),
        num_object_classes=int(args.num_object_classes),
        monitor_metric=str(args.monitor_metric),
    )
    print(
        json.dumps(
            {
                "stage": "train_setup",
                "num_archives": len(archives),
                "num_samples": train_size + val_size,
                "num_train_archives": len(train_archives),
                "num_val_archives": len(val_archives),
                "train_size": train_size,
                "val_size": val_size,
                "num_train_scenes": split_meta["num_train_scenes"],
                "num_val_scenes": split_meta["num_val_scenes"],
                "val_split_mode": args.val_split_mode,
                "val_split_seed": int(args.val_split_seed),
                "target": "quality",
                "score_collapse_std_threshold": args.score_collapse_std_threshold,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "dataset_bytes": dataset_bytes,
                "dataset_gb": round(dataset_bytes / 1024**3, 4),
                "train_device": str(train_device),
                "cache_device": str(train_dataset.device),
                "archive_batch_sampling": args.archive_batch_sampling,
                "packed_dataset_root": args.packed_dataset_root,
                "packed_batch_sampling": args.packed_batch_sampling,
                "archive_count_hints": args.archive_sample_counts,
                "archive_manifest": args.archive_manifest,
                "archive_manifest_loaded": train_manifest_records is not None,
                "num_workers": loader_runtime.num_workers,
                "pin_memory": loader_runtime.pin_memory,
                "persistent_workers": loader_runtime.persistent_workers,
                "amp": args.amp,
                "max_epochs": args.max_epochs,
                "min_epochs": args.min_epochs,
                "min_optimizer_steps": args.min_optimizer_steps,
                "min_checkpoint_steps": args.min_checkpoint_steps,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "weight_decay": args.weight_decay,
                "grad_clip_norm": None if args.grad_clip_norm <= 0 else args.grad_clip_norm,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "scheduler_min_lr": args.scheduler_min_lr,
                "scheduler_warmup_steps": args.scheduler_warmup_steps,
                "checkpoint_interval_steps": args.checkpoint_interval_steps,
                "model_init_seed": int(args.seed),
                "val_split_seed": int(args.val_split_seed),
                "save_dir": args.save_dir,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = _build_seeded_model(config, args.seed)
    num_params = sum(param.numel() for param in model.parameters())
    num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary_extra = {
        "num_params": int(num_params),
        "num_params_M": float(num_params) / 1_000_000.0,
        "num_trainable_params": int(num_trainable_params),
        "num_trainable_params_M": float(num_trainable_params) / 1_000_000.0,
        "depth": int(args.shell_attn_layers),
        "hidden_dim": int(args.hidden_dim),
        "detector_train_size": int(train_size),
        "detector_val_size": int(val_size),
        "num_archives": int(len(archives)),
        "num_train_archives": int(len(train_archives)),
        "num_val_archives": int(len(val_archives)),
        "num_train_scenes": int(split_meta["num_train_scenes"]),
        "num_val_scenes": int(split_meta["num_val_scenes"]),
        "num_workers": int(loader_runtime.num_workers),
        "prefetch_factor": loader_runtime.prefetch_factor,
        "pin_memory": bool(loader_runtime.pin_memory),
        "persistent_workers": bool(loader_runtime.persistent_workers),
        "archive_batch_sampling": bool(args.archive_batch_sampling),
        "packed_dataset_root": args.packed_dataset_root,
        "packed_batch_sampling": str(args.packed_batch_sampling),
        "model_init_seed": int(args.seed),
        "val_split_seed": int(args.val_split_seed),
        "cache_device": str(train_dataset.device),
        "train_device": str(train_device),
        "dataset_gb": round(dataset_bytes / 1024**3, 4),
        "target": "quality",
        "summary_schema_version": 1,
    }
    result = train_model(
        model, train_loader, val_loader, trainer_config, args.save_dir,
        tensorboard_dir=args.tensorboard_dir,
        summary_extra=summary_extra,
    )

    payload = {
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "input_root": [str(path) for path in input_roots],
        "save_dir": str(Path(args.save_dir)),
        "num_archives": len(archives),
        "num_samples": train_size + val_size,
        "dataset_bytes": dataset_bytes,
        "dataset_gb": round(dataset_bytes / 1024**3, 4),
        "split_config": {
            "val_ratio": args.val_ratio,
            "val_split_mode": args.val_split_mode,
            "val_split_seed": int(args.val_split_seed),
            "num_train_archives": len(train_archives),
            "num_val_archives": len(val_archives),
            "num_train_scenes": split_meta["num_train_scenes"],
            "num_val_scenes": split_meta["num_val_scenes"],
            "train_scene_keys": split_meta["train_scene_keys"],
            "val_scene_keys": split_meta["val_scene_keys"],
        },
        "target_config": {
            "success_mu_thresh": float(args.success_mu_thresh),
        },
        "train_dataset_summary": summarize_dataset(train_dataset),
        "val_dataset_summary": summarize_dataset(val_dataset) if val_dataset is not None else None,
        "train_size": train_size,
        "val_size": val_size,
        "loader_config": {
            "archive_batch_sampling": args.archive_batch_sampling,
            "packed_dataset_root": args.packed_dataset_root,
            "archive_count_hints": args.archive_sample_counts,
            "archive_manifest": args.archive_manifest,
            "archive_manifest_loaded": train_manifest_records is not None,
            "object_pooled_root": (
                None
                if object_pooled_roots is None
                else [str(path) for path in object_pooled_roots]
            ),
            "require_object_pooled": bool(args.require_object_pooled),
            "num_workers": loader_runtime.num_workers,
            "prefetch_factor": loader_runtime.prefetch_factor,
            "pin_memory": loader_runtime.pin_memory,
            "persistent_workers": loader_runtime.persistent_workers,
            "cache_device": str(train_dataset.device),
            "train_device": str(train_device),
        },
        "cli_args": vars(args),
        "model_config": config.__dict__,
        "trainer_config": trainer_config.__dict__,
        "result": result,
    }
    save_path = Path(args.save_dir) / "train_run.json"
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _finalize_summary(Path(args.save_dir) / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def _finalize_summary(summary_path: Path, payload: dict) -> None:
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "status": "completed",
            "total_seconds": float(payload["runtime_sec"]),
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "input_root": payload["input_root"],
            "save_dir": payload["save_dir"],
            "train_run_path": str(summary_path.parent / "train_run.json"),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _split_count_hints(
    archives: Sequence[Path],
    count_hints: Sequence[int] | None,
    train_archives: Sequence[Path],
    val_archives: Sequence[Path],
) -> tuple[list[int] | None, list[int] | None]:
    if count_hints is None:
        return None, None
    by_path = {path: int(count) for path, count in zip(archives, count_hints)}
    return [by_path[path] for path in train_archives], [by_path[path] for path in val_archives]


def _split_manifest_records(
    archives: Sequence[Path],
    records: Sequence[dict | None] | None,
    train_archives: Sequence[Path],
    val_archives: Sequence[Path],
) -> tuple[list[dict] | None, list[dict] | None]:
    if records is None:
        return None, None
    by_path = {path: record for path, record in zip(archives, records)}
    train_records = [by_path[path] for path in train_archives]
    val_records = [by_path[path] for path in val_archives]
    if any(record is None for record in train_records) or any(record is None for record in val_records):
        return None, None
    return train_records, val_records


def _load_manifest_records_by_root(
    input_roots: Sequence[Path],
    manifest_args: Sequence[str] | None,
    *,
    require_manifest: bool,
    success_mu_thresh: float,
) -> dict[Path, dict[Path, dict]]:
    if manifest_args is None:
        if not require_manifest:
            return {}
        manifest_args = ["auto"]
    if len(manifest_args) == 1 and len(input_roots) > 1:
        manifest_args = [manifest_args[0]] * len(input_roots)
    if len(manifest_args) != len(input_roots):
        raise ValueError(
            "--archive-manifest must provide exactly one value per --input-root "
            f"({len(manifest_args)} != {len(input_roots)})"
        )

    loaded: dict[Path, dict[Path, dict]] = {}
    for root, manifest_arg in zip(input_roots, manifest_args):
        root = root.resolve()
        manifest_path = default_manifest_path(root) if manifest_arg == "auto" else Path(manifest_arg)
        if not manifest_path.is_file():
            if require_manifest:
                raise FileNotFoundError(f"archive manifest not found: {manifest_path}")
            continue
        records = load_manifest_records(manifest_path)
        by_path = manifest_records_by_archive_path(records, root=root)
        archive_paths = sorted(root.glob("**/*.npz"))
        ok, problems = validate_manifest_coverage(
            by_path,
            archive_paths,
            success_mu_thresh=success_mu_thresh,
        )
        if not ok:
            message = f"archive manifest is incomplete or stale: {manifest_path}; first issue: {problems[0]}"
            if require_manifest:
                raise ValueError(message)
            print(f"[manifest] {message}; falling back", flush=True)
            continue
        loaded[root] = by_path
    return loaded


def _load_packed_archive_catalog(
    packed_root: Path,
    *,
    input_roots: Sequence[Path],
    camera: str | None,
) -> tuple[list[Path], list[int], list[dict]]:
    index_path = packed_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"packed dataset index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"{index_path}: unsupported packed schema_version={payload.get('schema_version')!r}")
    recorded_camera = payload.get("camera")
    if camera and recorded_camera is not None and str(recorded_camera) != str(camera):
        raise ValueError(
            f"{index_path}: packed camera={recorded_camera!r} does not match requested camera={camera!r}"
        )
    packed_records = list(payload.get("archives") or [])
    if not packed_records:
        raise ValueError(f"{index_path}: no archive records")

    archives: list[Path] = []
    counts: list[int] = []
    manifest_records: list[dict] = []
    for record in packed_records:
        root_index = int(record.get("input_root_index", 0))
        if root_index < 0 or root_index >= len(input_roots):
            raise ValueError(
                f"{index_path}: input_root_index={root_index} is outside configured roots={len(input_roots)}"
            )
        relative_path = str(record.get("relative_path") or "")
        if not relative_path:
            raise ValueError(f"{index_path}: packed archive record is missing relative_path")
        manifest_record = dict(record.get("manifest_record") or {})
        if not manifest_record:
            raise ValueError(f"{index_path}: packed archive {relative_path} is missing manifest_record")
        manifest_camera = manifest_record.get("camera")
        if camera and manifest_camera is not None and str(manifest_camera) != str(camera):
            continue
        archive_path = input_roots[root_index] / relative_path
        if camera and manifest_camera is None:
            if not filter_archive_paths_by_camera([archive_path], camera):
                continue
        archives.append(archive_path)
        counts.append(int(record["num_samples"]))
        manifest_records.append(manifest_record)

    if not archives:
        raise ValueError(f"{index_path}: no archive records matched camera={camera!r}")
    expected_count = payload.get("num_archives")
    if recorded_camera in {None, camera} and expected_count is not None and len(archives) != int(expected_count):
        raise ValueError(
            f"{index_path}: selected archives={len(archives)} does not match num_archives={expected_count}"
        )
    return archives, counts, manifest_records


def _normalize_optional_roots(
    roots: Sequence[str] | None,
    input_roots: Sequence[Path],
    *,
    option_name: str,
) -> list[Path] | None:
    if roots is None:
        return None
    if len(roots) == 1 and len(input_roots) > 1:
        roots = [roots[0]] * len(input_roots)
    if len(roots) != len(input_roots):
        raise ValueError(
            f"{option_name} must provide one value or exactly one value per --input-root "
            f"({len(roots)} != {len(input_roots)})"
        )
    return [Path(root) for root in roots]


if __name__ == "__main__":
    raise SystemExit(main())
