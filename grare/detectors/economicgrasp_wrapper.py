from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .base import BaseDetectorWrapper


@dataclass(frozen=True)
class EconomicGraspConfig:
    dataset_root: str
    checkpoint_path: str
    camera: str = "kinect"
    split: str = "test"
    modes: tuple[str, ...] = ("seen", "similar", "novel")
    batch_size: int = 4
    data_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = False
    persistent_workers: bool = True
    num_point: int = 20000
    collision_thresh: float = 0.0
    voxel_size: float = 0.005
    inference: bool = True
    evaluate: bool = False
    skip_existing: bool = False
    max_batches: int | None = None
    tf32: bool = True
    cudnn_benchmark: bool = True
    seed: int = 0
    deterministic: bool = False


class EconomicGraspWrapper(BaseDetectorWrapper):
    def __init__(
        self,
        config: EconomicGraspConfig,
        *,
        repo_root: str | Path,
        python_bin: str = "python",
        cuda_device: int = 0,
    ) -> None:
        super().__init__(repo_root=repo_root, python_bin=python_bin, cuda_device=cuda_device)
        self.config = config

    def build_inference_commands(self, output_root: str | Path) -> list[list[str]]:
        output_root = Path(output_root)
        dump_dir = output_root / "economicgrasp" / self.config.split
        dump_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "scripts/detectors/run_economicgrasp_split.py",
            "--dataset_root",
            self.config.dataset_root,
            "--checkpoint_path",
            self.config.checkpoint_path,
            "--dump_dir",
            str(dump_dir),
            "--camera",
            self.config.camera,
            "--split",
            self.config.split,
            "--batch_size",
            str(self.config.batch_size),
            "--data_workers",
            str(self.config.data_workers),
            "--prefetch-factor",
            str(self.config.prefetch_factor),
            "--collision_thresh",
            str(self.config.collision_thresh),
            "--voxel_size",
            str(self.config.voxel_size),
            "--num_point",
            str(self.config.num_point),
            "--seed",
            str(self.config.seed),
        ]
        if self.config.deterministic:
            command.append("--deterministic")
        if self.config.pin_memory:
            command.append("--pin-memory")
        if not self.config.persistent_workers:
            command.append("--no-persistent-workers")
        if self.config.skip_existing:
            command.append("--skip-existing")
        if self.config.max_batches is not None:
            command += ["--max_batches", str(self.config.max_batches)]
        if self.config.tf32:
            command.append("--tf32")
        if self.config.cudnn_benchmark:
            command.append("--cudnn-benchmark")
        return [command]

    def _modes_for_split(self) -> tuple[str, ...]:
        if self.config.split == "train":
            return ("train",)
        return self.config.modes

    def _dataset_split_for_mode(self, mode: str) -> str:
        if mode == "train":
            return "train"
        return f"test_{mode}"

    def postprocess_dumps(self, output_root: str | Path) -> None:
        """Clean up legacy per-mode subdirs if they exist.

        Current OPAL commands write EconomicGrasp train/test modes directly to
        <root>/economicgrasp/<split>/scene_XXXX/<camera>/*.npy. This hook keeps
        older <camera>_<mode>/ outputs readable by moving their scene folders
        into the split directory when present.
        """
        output_root = Path(output_root)
        det_root = output_root / "economicgrasp" / self.config.split
        for mode in self._modes_for_split():
            mode_dir = det_root / f"{self.config.camera}_{mode}"
            if not mode_dir.is_dir():
                continue
            for scene_dir in sorted(mode_dir.glob("scene_*")):
                target = det_root / scene_dir.name
                if target.exists():
                    continue
                # Rename so we keep one logical copy; falls back to symlink
                # across filesystems.
                try:
                    os.rename(scene_dir, target)
                except OSError:
                    target.symlink_to(scene_dir.resolve())
            for ap_artifact in mode_dir.glob(f"ap_{self.config.camera}_*.npy"):
                ap_artifact.unlink(missing_ok=True)
            try:
                mode_dir.rmdir()
            except OSError:
                # Non-empty (e.g. orphaned ap_*.npy or unexpected files);
                # leave alone rather than risk losing data.
                pass
