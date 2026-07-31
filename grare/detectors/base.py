"""Shared driver for running a frozen upstream detector over a GraspNet split.

The detector implementations themselves are not vendored. Each wrapper builds
the command line for the matching adapter under ``scripts/detectors/``, which
imports the upstream source tree cloned into ``external/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import shlex
import subprocess

from grare.candidates import (
    DetectorPrediction,
    PredictionKey,
    iter_prediction_files,
    load_prediction_from_file,
    parse_prediction_scene_frame,
)

__all__ = [
    "BaseDetectorWrapper",
    "DetectorPrediction",
    "PredictionKey",
    "iter_prediction_files",
    "load_prediction_from_file",
    "parse_prediction_scene_frame",
]


class BaseDetectorWrapper(ABC):
    def __init__(
        self,
        *,
        repo_root: str | Path,
        python_bin: str = "python",
        cuda_device: int = 0,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.python_bin = python_bin
        self.cuda_device = cuda_device

    @abstractmethod
    def build_inference_commands(self, output_root: str | Path) -> list[list[str]]:
        raise NotImplementedError

    def run_inference(self, output_root: str | Path, check: bool = True) -> None:
        for command in self.build_inference_commands(output_root):
            self._run_command(command, check=check)
        self.postprocess_dumps(output_root)

    def postprocess_dumps(self, output_root: str | Path) -> None:  # noqa: B027
        """Hook for consolidating per-mode dump trees into the canonical
        ``<root>/<detector>/<split>/scene_*/<camera>/*.npy`` layout consumed by
        ``grare-prepare``. Default: no-op."""

    def _run_command(self, command: list[str], check: bool = True) -> None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
        print(
            json.dumps(
                {
                    "stage": "detector_dump_command",
                    "repo_root": str(self.repo_root.resolve()),
                    "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                    "command": " ".join(shlex.quote(str(part)) for part in command),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        subprocess.run(command, cwd=self.repo_root, env=env, check=check)
