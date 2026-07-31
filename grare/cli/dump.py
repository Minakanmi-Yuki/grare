#!/usr/bin/env python3
"""Generate frozen-detector candidate dumps for a GraspNet split.

Drives an upstream detector checkpoint over the GraspNet train or test scenes
and writes one raw ``(K, 17)`` GraspGroup array per frame. These dumps are the
input to ``grare-prepare``; GraRe never changes the candidates themselves.

Clone the detector sources into ``external/`` and build their CUDA extensions
first (see the README Installation section):

    ./scripts/build_detector_extensions.sh
    ./scripts/verify_detector_extensions.sh

Examples:
    # GraspNet-Baseline, RealSense test split.
    grare-dump --detector graspnet_baseline --camera realsense --split test

    # Scale-Balanced-Grasp, RealSense train split.
    grare-dump --detector scale_balanced_grasp --camera realsense --split train

    # Preview the upstream command without running it.
    grare-dump --detector economicgrasp --camera kinect --split test --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

from grare.detectors.economicgrasp_wrapper import (
    EconomicGraspConfig,
    EconomicGraspWrapper,
)
from grare.detectors.graspnet_baseline_wrapper import (
    GraspNetBaselineConfig,
    GraspNetBaselineWrapper,
)
from grare.detectors.scale_balanced_grasp_wrapper import (
    ScaleBalancedGraspConfig,
    ScaleBalancedGraspWrapper,
)
from grare.utils.experiment_logging import timestamp, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETECTORS = ("graspnet_baseline", "scale_balanced_grasp", "economicgrasp")
CAMERAS = ("realsense", "kinect")
SPLITS = ("train", "test")

# Checkpoint filenames as published by each upstream project. These match the
# layout documented in the README Downloads section.
DEFAULT_CKPT_BY_CAMERA = {
    "realsense": {
        "graspnet_baseline": "graspnet_baseline/checkpoint-rs.tar",
        "scale_balanced_grasp": "scale_balanced_grasp/log_full_model/checkpoint.tar",
        "economicgrasp": "economicgrasp/economicgrasp_realsense.tar",
    },
    "kinect": {
        "graspnet_baseline": "graspnet_baseline/checkpoint-kn.tar",
        "economicgrasp": "economicgrasp/economicgrasp_kinect.tar",
    },
}


def _default_dump_root() -> Path:
    env_dump = os.environ.get("GRARE_DUMP_ROOT")
    if env_dump:
        return Path(env_dump).expanduser()
    raise SystemExit(
        "GRARE_DUMP_ROOT is not set and --dump-root was not provided. "
        "Source the environment file written by scripts/prepare_data_assets.sh."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--detector", required=True, choices=DETECTORS)
    parser.add_argument("--camera", default="realsense", choices=CAMERAS)
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLITS,
        help="Generate dumps for GraspNet train or test scenes.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Detector checkpoint. Default: $GRARE_DETECTOR_CKPT_ROOT/<published name>.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Directory the adapter runs from. Default: the GraRe repository root.",
    )
    parser.add_argument(
        "--dump-root",
        default=None,
        help="Dump root. Default: $GRARE_DUMP_ROOT.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override $GRASPNET_ROOT.")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--save-summary", default=None, help="Persist run summary JSON to this path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the upstream command, do not execute.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip frames whose .npy dump already exists.")
    parser.add_argument("--batch-size", type=int, default=None, help="Detector inference batch size.")
    parser.add_argument("--data-workers", type=int, default=None, help="DataLoader workers for detector inference.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=0,
        help="CPU threads for collision filtering and saving in the GN and SBG adapters.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory when CUDA is available.")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent DataLoader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches. Intended for smoke tests.")
    parser.add_argument("--index-shard-count", type=int, default=1, help="Split dump indices into N shards.")
    parser.add_argument("--index-shard-id", type=int, default=0, help="Run only this zero-based shard.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by detector inference.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Prefer deterministic kernels where supported. Slower, but more reproducible.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    dataset_root_raw = args.dataset_root or os.environ.get("GRASPNET_ROOT", "")
    if not dataset_root_raw:
        raise SystemExit("GRASPNET_ROOT is not set and --dataset-root was not provided.")
    dataset_root = Path(dataset_root_raw).expanduser()

    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser()
    else:
        ckpt_root_raw = os.environ.get("GRARE_DETECTOR_CKPT_ROOT", "")
        if not ckpt_root_raw:
            raise SystemExit(
                "GRARE_DETECTOR_CKPT_ROOT is not set and --ckpt was not provided. "
                "Source the environment file written by scripts/prepare_data_assets.sh."
            )
        rel_ckpt = DEFAULT_CKPT_BY_CAMERA.get(args.camera, {}).get(args.detector)
        if rel_ckpt is None:
            raise SystemExit(
                f"no published checkpoint for detector={args.detector!r}, camera={args.camera!r}. "
                "Scale-Balanced-Grasp publishes a RealSense checkpoint only; "
                "pass --ckpt explicitly to use another one."
            )
        ckpt = Path(ckpt_root_raw).expanduser() / rel_ckpt

    repo_root = Path(args.repo_root).expanduser() if args.repo_root else PROJECT_ROOT
    dump_root = Path(args.dump_root).expanduser() if args.dump_root else _default_dump_root()
    return dataset_root, ckpt, repo_root, dump_root


def _build_wrapper(args: argparse.Namespace, dataset_root: Path, ckpt: Path, repo_root: Path):
    if args.index_shard_count < 1:
        raise SystemExit("--index-shard-count must be >= 1")
    if not 0 <= args.index_shard_id < args.index_shard_count:
        raise SystemExit("--index-shard-id must satisfy 0 <= id < count")
    if args.detector == "economicgrasp" and args.index_shard_count > 1:
        raise SystemExit(
            "index sharding is supported for graspnet_baseline and scale_balanced_grasp only"
        )

    persistent_workers = not args.no_persistent_workers
    common = {
        "dataset_root": str(dataset_root),
        "checkpoint_path": str(ckpt),
        "camera": args.camera,
        "split": args.split,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": args.pin_memory,
        "persistent_workers": persistent_workers,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "skip_existing": args.skip_existing,
        "max_batches": args.max_batches,
    }
    wrapper_kwargs = {
        "repo_root": repo_root,
        "python_bin": args.python_bin,
        "cuda_device": args.cuda_device,
    }

    if args.detector == "graspnet_baseline":
        config = GraspNetBaselineConfig(
            batch_size=args.batch_size if args.batch_size is not None else 1,
            num_workers=args.data_workers if args.data_workers is not None else 6,
            postprocess_workers=args.postprocess_workers,
            index_shard_count=args.index_shard_count,
            index_shard_id=args.index_shard_id,
            **common,
        )
        return GraspNetBaselineWrapper(config, **wrapper_kwargs)

    if args.detector == "scale_balanced_grasp":
        config = ScaleBalancedGraspConfig(
            batch_size=args.batch_size if args.batch_size is not None else 1,
            data_workers=args.data_workers if args.data_workers is not None else 4,
            postprocess_workers=args.postprocess_workers,
            index_shard_count=args.index_shard_count,
            index_shard_id=args.index_shard_id,
            **common,
        )
        return ScaleBalancedGraspWrapper(config, **wrapper_kwargs)

    config = EconomicGraspConfig(
        batch_size=args.batch_size if args.batch_size is not None else 4,
        data_workers=args.data_workers if args.data_workers is not None else 4,
        **common,
    )
    return EconomicGraspWrapper(config, **wrapper_kwargs)


def _validate_dataset_root(dataset_root: Path, *, split: str, camera: str) -> None:
    scene = "scene_0000" if split == "train" else "scene_0100"
    expected = [
        dataset_root / "scenes" / scene / camera / "rgb" / "0000.png",
        dataset_root / "scenes" / scene / camera / "depth" / "0000.png",
        dataset_root / "scenes" / scene / camera / "label" / "0000.png",
        dataset_root / "scenes" / scene / camera / "meta" / "0000.mat",
    ]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return
    raise SystemExit(
        "GraspNet-1Billion appears to be missing or not extracted.\n"
        f"  GRASPNET_ROOT: {dataset_root}\n"
        f"  missing example: {missing[0]}\n"
        "Download and extract the dataset before generating detector dumps."
    )


def main() -> int:
    args = parse_args()
    dataset_root, ckpt, repo_root, dump_root = _resolve_paths(args)
    wrapper = _build_wrapper(args, dataset_root, ckpt, repo_root)

    if args.dry_run:
        for command in wrapper.build_inference_commands(dump_root):
            print(" ".join(shlex.quote(str(part)) for part in command))
        return 0

    if not ckpt.is_file():
        raise SystemExit(
            f"detector checkpoint does not exist: {ckpt}\n"
            "Download it from the upstream project (see the README Downloads section)."
        )
    if not repo_root.is_dir():
        raise SystemExit(f"repo root does not exist: {repo_root}")
    _validate_dataset_root(dataset_root, split=args.split, camera=args.camera)

    started_at = timestamp()
    started_perf = time.perf_counter()
    setup = {
        "stage": "detector_dump_setup",
        "detector": args.detector,
        "camera": args.camera,
        "split": args.split,
        "dataset_root": str(dataset_root.resolve()),
        "checkpoint_path": str(ckpt.resolve()),
        "repo_root": str(repo_root.resolve()),
        "dump_root": str(dump_root.resolve()),
        "cuda_device": args.cuda_device,
        "batch_size": args.batch_size,
        "data_workers": args.data_workers,
        "postprocess_workers": args.postprocess_workers,
        "max_batches": args.max_batches,
        "index_shard_count": args.index_shard_count,
        "index_shard_id": args.index_shard_id,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "started_at": started_at,
    }
    print(json.dumps(setup, ensure_ascii=False), flush=True)

    wrapper.run_inference(dump_root, check=True)

    payload = {
        **setup,
        "stage": "detector_dump",
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
    }
    if args.save_summary:
        write_json(args.save_summary, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
