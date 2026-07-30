"""Detector-agnostic I/O for GraspNet-format candidate dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictionKey:
    detector: str
    benchmark: str
    split: str
    scene_id: int
    frame_id: int
    camera: str

    @property
    def scene_name(self) -> str:
        return f"scene_{self.scene_id:04d}"


@dataclass
class DetectorPrediction:
    key: PredictionKey
    grasp_group_array: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.grasp_group_array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("grasp_group_array must be rank-2")
        self.grasp_group_array = array.reshape(0, 17) if array.size == 0 else array

    @property
    def base_scores(self) -> np.ndarray:
        if self.grasp_group_array.shape[1] == 0:
            return np.empty((0,), dtype=np.float32)
        return self.grasp_group_array[:, 0].astype(np.float32, copy=False)

def parse_prediction_scene_frame(path: str | Path, camera: str) -> tuple[int, int]:
    path = Path(path)
    frame_id = int(path.stem)
    parent = path.parent
    if parent.name == camera:
        parent = parent.parent
    name = parent.name
    if name.startswith("scene_"):
        return int(name.rsplit("_", 1)[1]), frame_id
    if name.isdigit():
        return int(name), frame_id
    raise ValueError(f"cannot parse scene/frame from prediction path: {path}")


def load_prediction_from_file(
    path: str | Path,
    *,
    detector: str,
    benchmark: str,
    split: str,
    camera: str,
) -> DetectorPrediction:
    scene_id, frame_id = parse_prediction_scene_frame(path, camera)
    return DetectorPrediction(
        key=PredictionKey(detector, benchmark, split, scene_id, frame_id, camera),
        grasp_group_array=np.load(path),
    )
