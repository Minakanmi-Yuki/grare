"""Small, dependency-light helpers shared by per-frame detector adapters.

The public GraspNet detector repositories disagree on their dataset wrappers.
GraRe adapters use this module only for deterministic frame enumeration and
the canonical GraspGroup ``(K, 17)`` output contract; model-specific code
remains in its own adapter.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterator, TypeVar
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np


FRAMES_PER_SCENE = 256
SCENES_BY_SPLIT = {
    "train": range(0, 100),
    "test": range(100, 190),
}
T = TypeVar("T")
R = TypeVar("R")


def configure_seed(seed: int, *, deterministic: bool) -> None:
    """Seed Python/NumPy/Torch without importing Torch for dry utilities."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def configure_cuda(*, tf32: bool, cudnn_benchmark: bool) -> object:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if tf32:
            torch.set_float32_matmul_precision("high")
    return device


def iter_frame_indices(
    *, split: str, shard_count: int, shard_id: int
) -> Iterator[tuple[int, int]]:
    if split not in SCENES_BY_SPLIT:
        raise ValueError(f"unsupported split: {split!r}")
    if shard_count < 1 or not 0 <= shard_id < shard_count:
        raise ValueError("invalid index shard")
    ordinal = 0
    for scene_id in SCENES_BY_SPLIT[split]:
        for frame_id in range(FRAMES_PER_SCENE):
            if ordinal % shard_count == shard_id:
                yield scene_id, frame_id
            ordinal += 1


def dump_path(dump_root: Path, *, scene_id: int, camera: str, frame_id: int) -> Path:
    return dump_root / f"scene_{scene_id:04d}" / camera / f"{frame_id:04d}.npy"


def read_rgbd(dataset_root: Path, *, scene_id: int, camera: str, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    root = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera
    rgb_bgr = cv2.imread(str(root / "rgb" / f"{frame_id:04d}.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "depth" / f"{frame_id:04d}.png"), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None:
        raise FileNotFoundError(
            f"could not read RGB-D frame scene_{scene_id:04d}/{camera}/{frame_id:04d}"
        )
    if depth.ndim != 2:
        raise ValueError(f"expected a single-channel depth image, got {depth.shape}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), np.asarray(depth)


def prefetch_items(
    items: list[T],
    loader: Callable[[T], R],
    *,
    workers: int,
    factor: int,
) -> Iterator[tuple[T, R]]:
    """Load a bounded, ordered window of CPU inputs ahead of GPU inference."""
    if workers <= 0:
        for item in items:
            yield item, loader(item)
        return
    window = max(workers, workers * max(1, factor))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grare-rgbd") as executor:
        iterator = iter(items)
        queue: list[tuple[T, Future[R]]] = []

        def submit_until_full() -> None:
            while len(queue) < window:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                queue.append((item, executor.submit(loader, item)))

        submit_until_full()
        while queue:
            item, future = queue.pop(0)
            yield item, future.result()
            submit_until_full()


@lru_cache(maxsize=None)
def read_intrinsics(dataset_root: Path, *, scene_id: int, camera: str) -> np.ndarray:
    path = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera / "camK.npy"
    intrinsics = np.asarray(np.load(path), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"expected 3x3 intrinsics in {path}, got {intrinsics.shape}")
    return intrinsics


def points_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project valid millimetre depth pixels to the camera frame."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols] / 1000.0
    x = (cols.astype(np.float32) - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0])
    y = (rows.astype(np.float32) - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1])
    return np.stack((x, y, z), axis=1).astype(np.float32, copy=False)


def grasp_group_array(group: object) -> np.ndarray:
    """Convert a compatible upstream GraspGroup into the standard array."""
    scores = np.asarray(getattr(group, "scores"), dtype=np.float32).reshape(-1, 1)
    count = len(scores)
    if count == 0:
        return np.empty((0, 17), dtype=np.float32)
    rotations = getattr(group, "rotation_matrices", None)
    if rotations is None:
        rotations = getattr(group, "rotations")
    object_ids = np.asarray(
        getattr(group, "object_ids", -np.ones((count,), dtype=np.float32)), dtype=np.float32
    ).reshape(-1, 1)
    array = np.concatenate(
        [
            scores,
            np.asarray(getattr(group, "widths"), dtype=np.float32).reshape(-1, 1),
            np.asarray(getattr(group, "heights"), dtype=np.float32).reshape(-1, 1),
            np.asarray(getattr(group, "depths"), dtype=np.float32).reshape(-1, 1),
            np.asarray(rotations, dtype=np.float32).reshape(-1, 9),
            np.asarray(getattr(group, "translations"), dtype=np.float32).reshape(-1, 3),
            object_ids,
        ],
        axis=1,
    )
    if array.shape != (count, 17):
        raise ValueError(f"invalid canonical grasp array shape: {array.shape}")
    return array


def save_grasp_array(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 17:
        raise ValueError(f"expected a (K, 17) grasp array, got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


class ProgressReporter:
    """Emit stable JSON progress without a TTY-dependent progress bar."""

    def __init__(self, *, detector: str, total: int, interval_sec: float = 30.0) -> None:
        self.detector = detector
        self.total = total
        self.completed = 0
        self.started = time.perf_counter()
        self.last_report = self.started
        self.interval_sec = interval_sec

    def advance(self, *, scene_id: int, frame_id: int) -> None:
        self.completed += 1
        now = time.perf_counter()
        if self.completed != self.total and now - self.last_report < self.interval_sec:
            return
        elapsed = now - self.started
        per_item = elapsed / self.completed if self.completed else 0.0
        remaining = max(0, self.total - self.completed) * per_item
        print(
            json.dumps(
                {
                    "stage": "detector_dump_progress",
                    "detector": self.detector,
                    "completed": self.completed,
                    "total": self.total,
                    "scene": f"scene_{scene_id:04d}",
                    "frame": frame_id,
                    "elapsed_sec": round(elapsed, 2),
                    "eta_sec": round(remaining, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        self.last_report = now
