"""Run the paper training, re-ranking, and official-evaluation stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from grare.config import load_config


STAGES = ("train", "rerank", "eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-from", choices=STAGES, default="train")
    parser.add_argument("--stop-after", choices=STAGES, default="eval")
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


def _rerank_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    rerank = config["rerank"]
    output = Path(paths["rerank_dir"])
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


def _eval_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    output = Path(paths["eval_dir"])
    return _module_command("grare.cli.evaluate") + [
        "--dataset-root", paths["graspnet_root"],
        "--dump-folder", paths["rerank_dir"],
        "--camera", config["camera"],
        "--split", "test",
        "--proc", str(config["eval"]["proc"]),
        "--save-raw", str(output / "per_scene_raw.npy"),
        "--save-summary", str(output / "per_scene_raw.json"),
        "--tag", config["name"],
    ]


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
    builders = {"train": _train_command, "rerank": _rerank_command, "eval": _eval_command}
    for stage in STAGES[start : stop + 1]:
        command = builders[stage](config)
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
