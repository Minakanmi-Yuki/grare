"""Build GraRe features for one frozen-detector setting.

Runs the resumable labels, MobileSAM object-cloud, and frozen Point-MAE stages
for GraspNet train and test in order. Detector dumps must already exist.

Example:
    grare-feature --detector graspnet_baseline --camera realsense
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


DETECTORS = (
    "graspnet_baseline",
    "scale_balanced_grasp",
    "economicgrasp",
    "hggd",
    "rngnet",
    "generalizing_grasp",
)
CAMERAS = ("realsense", "kinect")
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detector", required=True, choices=DETECTORS)
    parser.add_argument("--camera", required=True, choices=CAMERAS)
    parser.add_argument(
        "--split",
        default="all",
        choices=(*SPLITS, "all"),
        help="Build features for train, test, or both splits in order (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(
        f"{name} is not set. Source $GRARE_ASSET_WORKSPACE/grare_paths.env first."
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    dump_root = Path(_required_env("GRARE_DUMP_ROOT"))
    data_root = Path(_required_env("GRARE_DATA_ROOT"))
    dataset_root = _required_env("GRASPNET_ROOT")
    sam_checkpoint = _required_env("GRARE_SAM_CKPT")
    point_mae_checkpoint = _required_env("GRARE_POINT_MAE_CKPT")
    splits = SPLITS if args.split == "all" else (args.split,)

    for split in splits:
        local_root = data_root / "relabeled" / args.detector / args.camera / "local_cloud" / split
        object_root = data_root / "relabeled" / args.detector / args.camera / "object_cloud" / split
        pooled_root = data_root / "relabeled" / args.detector / args.camera / "object_pooled" / split

        _run(
            [
                sys.executable,
                "-m",
                "grare.cli.prepare",
                "--stage",
                "labels",
                "--input-root",
                str(dump_root / args.detector / split),
                "--pattern",
                f"scene_*/{args.camera}/*.npy",
                "--input-format",
                "detector-dump",
                "--output-root",
                str(local_root),
                "--detector",
                args.detector,
                "--dataset-root",
                dataset_root,
                "--camera",
                args.camera,
                "--split",
                split,
                "--omit-object-cloud",
            ],
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                "-m",
                "grare.cli.prepare",
                "--stage",
                "object",
                "--input-root",
                str(local_root),
                "--output-root",
                str(local_root),
                "--object-cloud-root",
                str(object_root),
                "--detector",
                args.detector,
                "--dataset-root",
                dataset_root,
                "--camera",
                args.camera,
                "--split",
                split,
                "--sam-checkpoint",
                sam_checkpoint,
                "--omit-object-cloud",
            ],
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                "-m",
                "grare.cli.precompute_object",
                "--archive-root",
                str(local_root),
                "--object-cloud-root",
                str(object_root),
                "--output-root",
                str(pooled_root),
                "--pmae-ckpt",
                point_mae_checkpoint,
            ],
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
