from __future__ import annotations

import subprocess
import sys

from grare.cli.prepare import _default_prepare_workers


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


def test_default_labels_workers_are_resource_aware() -> None:
    from grare.cli.prepare import _label_worker_cap
    from grare.utils.cpu import effective_cpu_count

    assert _default_prepare_workers(
        stage="labels", sam_device="cuda", sam_enabled=False
    ) == min(max(1, effective_cpu_count()), _label_worker_cap())


def test_default_object_workers_use_the_sam_cap(monkeypatch) -> None:
    monkeypatch.setattr("grare.cli.prepare._visible_cuda_device_count", lambda: 2)
    monkeypatch.setattr("grare.cli.prepare._sam_worker_cap", lambda: 4)
    assert _default_prepare_workers(
        stage="object", sam_device="cuda", sam_enabled=True
    ) == 8
