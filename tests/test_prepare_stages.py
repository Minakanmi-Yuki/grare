from __future__ import annotations

import subprocess
import sys


def _help() -> str:
    out = subprocess.run(
        [sys.executable, "-m", "grare.cli.prepare", "--help"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def test_stage_flag_offers_the_split_passes() -> None:
    text = _help()
    assert "--stage" in text
    for choice in ("all", "labels", "object"):
        assert choice in text


def test_stage_object_requires_a_sam_checkpoint() -> None:
    out = subprocess.run(
        [
            sys.executable, "-m", "grare.cli.prepare",
            "--stage", "object",
            "--input-root", "/nonexistent",
            "--output-root", "/nonexistent",
            "--detector", "graspnet_baseline",
            "--dataset-root", "/nonexistent",
            "--camera", "realsense",
        ],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "--stage object requires --sam-checkpoint" in (out.stdout + out.stderr)
