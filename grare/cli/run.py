"""Run the optional packing, training, re-ranking, and evaluation stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from grare.config import load_config
from grare.utils.runtime import configure_thread_pools


STAGES = ("pack", "train", "rerank", "eval")

configure_thread_pools()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-from", choices=STAGES, default="pack")
    parser.add_argument("--stop-after", choices=STAGES, default="eval")
    parser.add_argument(
        "--repack",
        action="store_true",
        help="Rebuild the configured packed training dataset before training.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Discard completed evaluation shards before running eval.",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def _train_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    train = config["train"]
    command = _module_command("grare.cli.train") + [
        "--input-root", paths["local_cloud_train"],
        "--camera", config["camera"],
        "--save-dir", paths["checkpoint_dir"],
        "--val-ratio", str(train["val_ratio"]),
        "--val-split-mode", "scene",
        "--val-split-seed", str(train["val_split_seed"]),
        "--batch-size", str(train["batch_size"]),
        "--min-epochs", str(train["min_epochs"]),
        "--max-epochs", str(train["max_epochs"]),
        "--min-optimizer-steps", str(train.get("min_optimizer_steps", 0)),
        "--min-checkpoint-steps", str(train.get("min_checkpoint_steps", 0)),
        "--early-stop-patience", str(train["early_stop_patience"]),
        "--lr", str(train["lr"]),
        "--weight-decay", str(train["weight_decay"]),
        "--grad-clip-norm", str(train["grad_clip_norm"]),
        "--scheduler-patience", str(train["scheduler_patience"]),
        "--scheduler-factor", str(train["scheduler_factor"]),
        "--scheduler-min-lr", str(train["scheduler_min_lr"]),
        "--scheduler-warmup-steps", str(train.get("scheduler_warmup_steps", 0)),
        "--checkpoint-interval-steps", str(train.get("checkpoint_interval_steps", 0)),
        "--monitor-metric", "loss_score",
        "--hidden-dim", "128",
        "--dropout", "0.1",
        "--success-mu-thresh", "0.4",
        "--device", train["device"],
        "--num-workers", str(train["num_workers"]),
        "--prefetch-factor", "4",
        "--amp", train["amp"],
        "--cache-device", "cpu",
        "--shell-attn-n-shells", "4",
        "--shell-attn-heads", "4",
        "--shell-attn-layers", "1",
        "--shell-attn-per-point-dim", "3",
        "--score-collapse-std-threshold", "0.0001",
        "--seed", str(train["seed"]),
        "--lambda-collision", "0.10",
        "--lambda-empty", "0.05",
        "--lambda-obj-id", "0.05",
        "--num-object-classes", "88",
        "--object-cloud-points", "512",
        "--object-hidden-dim", "128",
        "--object-pmae-ckpt", paths["point_mae_checkpoint"],
        "--object-pmae-num-group", "32",
        "--object-pmae-group-size", "32",
        "--fusion-layers", "1",
        "--fusion-heads", "4",
        "--fusion-ffn-mult", "2",
    ]
    packed_train = paths.get("packed_train")
    if packed_train:
        command.extend([
            "--packed-dataset-root", str(packed_train),
            "--packed-batch-sampling", "archive_mixed",
        ])
    else:
        command.extend([
            "--archive-batch-sampling",
            "--archive-manifest", "auto",
            "--require-archive-manifest",
            "--object-pooled-root", paths["object_pooled_train"],
            "--require-object-pooled",
        ])
    if int(train.get("local_health_max_batches", 0)) > 0:
        command.extend([
            "--local-health-max-batches", str(train["local_health_max_batches"]),
            "--local-health-seed", str(train.get("local_health_seed", 20260710)),
            "--min-local-raw-rms", str(train.get("min_local_raw_rms", 0.0)),
            "--min-local-shuffle-ratio", str(train.get("min_local_shuffle_ratio", 0.0)),
        ])
    if train.get("resume_from"):
        command.extend(["--resume-from", str(train["resume_from"])])
    return command


def _packed_index_path(config: dict[str, Any]) -> Path | None:
    """Return the configured packed-dataset index, when this config uses one."""
    packed_train = config["paths"].get("packed_train")
    return None if not packed_train else Path(packed_train) / "index.json"


def _pack_command(config: dict[str, Any]) -> list[str] | None:
    """Build the pack command required by packed-mmap training configs.

    Configurations without ``paths.packed_train`` intentionally retain lazy
    archive loading, so they have no packing stage.
    """
    paths = config["paths"]
    packed_train = paths.get("packed_train")
    if not packed_train:
        return None
    return _module_command("grare.cli.pack") + [
        "--input-root",
        paths["local_cloud_train"],
        "--camera",
        config["camera"],
        "--output-root",
        str(packed_train),
        "--object-pooled-root",
        paths["object_pooled_train"],
        "--require-object-pooled",
        "--archive-manifest",
        "auto",
        "--require-archive-manifest",
    ]


def _is_detector_baseline(config: dict[str, Any]) -> bool:
    """A lambda=0.0 run keeps the detector's own ranking (the AP baseline)."""
    return float(config["rerank"]["lambda"]) == 0.0


def _stage_dir(config: dict[str, Any], key: str) -> Path:
    """Keep detector-baseline artifacts out of the GraRe artifact directories.

    Both runs consume the same features and checkpoint, so writing them to the
    same rerank/eval directories would silently overwrite the GraRe results the
    baseline is meant to be compared against.
    """
    directory = Path(config["paths"][key])
    if _is_detector_baseline(config):
        return directory.with_name(directory.name + "_detector_baseline")
    return directory


def _rerank_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    rerank = config["rerank"]
    output = _stage_dir(config, "rerank_dir")
    return _module_command("grare.cli.rerank") + [
        "--input-root", paths["local_cloud_test"],
        "--camera", config["camera"],
        "--output-root", str(output),
        "--checkpoint", str(Path(paths["checkpoint_dir"]) / "best.pt"),
        "--object-pooled-root", paths["object_pooled_test"],
        "--require-object-pooled",
        "--device", rerank["device"],
        "--num-workers", str(rerank["num_workers"]),
        "--rescoring-score-weight", str(rerank["lambda"]),
        "--save-path", str(output / "rerank_summary.json"),
        "--save-records-path", str(output / "rerank_records.jsonl"),
    ]


def _eval_command(config: dict[str, Any], *, force: bool = False) -> list[str]:
    paths = config["paths"]
    output = _stage_dir(config, "eval_dir")
    tag = config["name"]
    if _is_detector_baseline(config):
        tag = f"{tag}_detector_baseline"
    command = _module_command("grare.cli.evaluate") + [
        "--dataset-root", paths["graspnet_root"],
        "--dump-folder", str(_stage_dir(config, "rerank_dir")),
        "--camera", config["camera"],
        "--split", "test",
        "--proc", str(config["eval"]["proc"]),
        "--save-raw", str(output / "per_scene_raw.npy"),
        "--save-summary", str(output / "per_scene_raw.json"),
        "--tag", tag,
    ]
    if force:
        command.append("--force")
    return command


def _pack_execution(
    config: dict[str, Any],
    *,
    repack: bool,
) -> tuple[list[str] | None, str | None]:
    """Return a pack command or the reason that packing is unnecessary.

    A completed pack is immutable input to mmap-backed training, so the normal
    path reuses it.  A partial output directory is safe to replace because
    ``grare-pack --overwrite`` refuses unknown files before deleting anything.
    """
    command = _pack_command(config)
    index_path = _packed_index_path(config)
    if command is None or index_path is None:
        return None, "no packed_train configured"
    if index_path.is_file() and not repack:
        return None, f"existing packed dataset at {index_path.parent}"
    if repack or index_path.parent.exists():
        command.append("--overwrite")
    return command, None


def main() -> int:
    args = parse_args()
    config = load_config(
        args.config,
        args.overrides,
        allow_missing_environment=args.dry_run,
    )
    start = STAGES.index(args.start_from)
    stop = STAGES.index(args.stop_after)
    if stop < start:
        raise ValueError("--stop-after precedes --start-from")
    builders = {"train": _train_command, "rerank": _rerank_command}
    for stage in STAGES[start : stop + 1]:
        if stage == "pack":
            if args.dry_run:
                command = _pack_command(config)
                skipped_reason = "no packed_train configured" if command is None else None
                index_path = _packed_index_path(config)
                if (
                    command is not None
                    and args.repack
                    and index_path is not None
                    and index_path.parent.exists()
                ):
                    command.append("--overwrite")
            else:
                command, skipped_reason = _pack_execution(config, repack=args.repack)
            if command is None:
                print(f"[pack] skipped: {skipped_reason}", flush=True)
                continue
        else:
            command = (
                _eval_command(config, force=args.force_eval)
                if stage == "eval"
                else builders[stage](config)
            )
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
