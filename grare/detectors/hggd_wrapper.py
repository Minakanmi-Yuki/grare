from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BaseDetectorWrapper


@dataclass(frozen=True)
class HGGDConfig:
    """Published HGGD inference settings exposed through ``grare-dump``."""

    dataset_root: str
    checkpoint_path: str
    camera: str = "realsense"
    split: str = "test"
    # HGGD's two-stage detector is frame-oriented, so its upstream model still
    # runs one frame at a time. Keep batch_size=1 but use the same input
    # pipeline depth as the other detectors to hide RGB-D file latency.
    batch_size: int = 1
    data_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    # Match the released HGGD test command exactly.
    num_point: int = 25600
    center_num: int = 48
    group_num: int = 512
    local_k: int = 10
    collision_thresh: float = 0.01
    voxel_size: float = 0.01
    skip_existing: bool = True
    max_batches: int | None = None
    postprocess_workers: int = 0
    index_shard_count: int = 1
    index_shard_id: int = 0
    tf32: bool = True
    cudnn_benchmark: bool = True
    seed: int = 0
    deterministic: bool = False


class HGGDWrapper(BaseDetectorWrapper):
    def __init__(
        self,
        config: HGGDConfig,
        *,
        repo_root: str | Path,
        python_bin: str = "python",
        cuda_device: int = 0,
    ) -> None:
        super().__init__(repo_root=repo_root, python_bin=python_bin, cuda_device=cuda_device)
        self.config = config

    def build_inference_commands(self, output_root: str | Path) -> list[list[str]]:
        dump_dir = Path(output_root) / "hggd" / self.config.split
        dump_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_bin,
            "scripts/detectors/run_hggd_split.py",
            "--dataset_root", self.config.dataset_root,
            "--checkpoint_path", self.config.checkpoint_path,
            "--dump_dir", str(dump_dir),
            "--camera", self.config.camera,
            "--split", self.config.split,
            "--data_workers", str(self.config.data_workers),
            "--prefetch-factor", str(self.config.prefetch_factor),
            "--num_point", str(self.config.num_point),
            "--center-num", str(self.config.center_num),
            "--group-num", str(self.config.group_num),
            "--local-k", str(self.config.local_k),
            "--collision_thresh", str(self.config.collision_thresh),
            "--voxel_size", str(self.config.voxel_size),
            "--seed", str(self.config.seed),
            "--index-shard-count", str(self.config.index_shard_count),
            "--index-shard-id", str(self.config.index_shard_id),
        ]
        if self.config.skip_existing:
            command.append("--skip-existing")
        if self.config.max_batches is not None:
            command += ["--max_batches", str(self.config.max_batches)]
        if self.config.deterministic:
            command.append("--deterministic")
        if self.config.tf32:
            command.append("--tf32")
        if self.config.cudnn_benchmark:
            command.append("--cudnn-benchmark")
        return [command]
