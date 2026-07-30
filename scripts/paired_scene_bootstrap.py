#!/usr/bin/env python3
"""Paired scene-bootstrap comparison for official GraspNet tensors.

Both inputs must be ``per_scene_raw.npy`` tensors produced by
``grare-evaluate`` with shape ``(90, 256, 50, 6)``. The script resamples the
paired test scenes, so each bootstrap draw preserves the frame-level pairing
between a frozen detector and its GraRe re-ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SHAPE = (90, 256, 50, 6)
SPLITS = {
    "overall": slice(0, 90),
    "seen": slice(0, 30),
    "similar": slice(30, 60),
    "novel": slice(60, 90),
}
METRICS: dict[str, int | None] = {"AP": None, "AP@0.4": 1, "AP@0.8": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Detector per_scene_raw.npy")
    parser.add_argument("--treatment", type=Path, required=True, help="GraRe per_scene_raw.npy")
    parser.add_argument("--bootstraps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    return parser.parse_args()


def load_raw(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(f"expected {EXPECTED_SHAPE} at {path}, found {array.shape}")
    return np.asarray(array, dtype=np.float32)


def scene_values(raw: np.ndarray, friction_index: int | None) -> np.ndarray:
    if friction_index is None:
        return raw.mean(axis=(1, 2, 3), dtype=np.float64) * 100.0
    return raw[..., friction_index].mean(axis=(1, 2), dtype=np.float64) * 100.0


def paired_interval(delta: np.ndarray, *, rng: np.random.Generator, count: int) -> dict[str, float]:
    if delta.ndim != 1 or delta.size == 0:
        raise ValueError("paired scene values must be a nonempty vector")
    if count <= 0:
        samples = np.asarray([delta.mean()], dtype=np.float64)
    else:
        indices = rng.integers(0, delta.size, size=(count, delta.size), endpoint=False)
        samples = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "positive_scene_fraction": float((delta > 0).mean()),
        "positive_scene_count": int((delta > 0).sum()),
        "scene_count": int(delta.size),
    }


def main() -> int:
    args = parse_args()
    if args.bootstraps < 0:
        raise ValueError("--bootstraps must be nonnegative")
    baseline = load_raw(args.baseline)
    treatment = load_raw(args.treatment)
    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, scene_slice in SPLITS.items():
        comparisons[split_name] = {}
        for metric_name, friction_index in METRICS.items():
            delta = scene_values(treatment, friction_index)[scene_slice] - scene_values(
                baseline, friction_index
            )[scene_slice]
            comparisons[split_name][metric_name] = paired_interval(
                delta, rng=rng, count=args.bootstraps
            )
    payload = {
        "baseline": str(args.baseline.resolve()),
        "treatment": str(args.treatment.resolve()),
        "expected_shape": list(EXPECTED_SHAPE),
        "bootstrap_count": args.bootstraps,
        "bootstrap_seed": args.seed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
