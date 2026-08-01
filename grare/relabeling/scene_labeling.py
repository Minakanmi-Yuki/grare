from __future__ import annotations

import contextlib
import gc
import json
from dataclasses import dataclass, replace
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from grare.candidates import load_prediction_from_file

_GRASPNET_API_IMPORT_ERROR: Exception | None = None

try:
    from graspnetAPI import GraspNetEval
    from graspnetAPI.utils.config import get_config as get_graspnet_config
    from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import (
        GraspQualityConfigFactory as GraspNetGraspQualityConfigFactory,
    )
    from graspnetAPI.utils.dexnet.grasping.quality import (
        PointGraspMetrics3D as GraspNetPointGraspMetrics3D,
    )
    from graspnetAPI.utils.eval_utils import (
        collision_detection as graspnet_collision_detection,
        create_table_points,
        get_grasp_score as graspnet_get_grasp_score,
        transform_points as graspnet_transform_points,
        voxel_sample_points as graspnet_voxel_sample_points,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by import-only smoke tests
    _GRASPNET_API_IMPORT_ERROR = exc
    GraspNetEval = None  # type: ignore[assignment]
    get_graspnet_config = None  # type: ignore[assignment]
    GraspNetGraspQualityConfigFactory = None  # type: ignore[assignment]
    GraspNetPointGraspMetrics3D = None  # type: ignore[assignment]

    def _missing_graspnet_api(*args: Any, **kwargs: Any) -> Any:
        _require_graspnet_api()

    graspnet_collision_detection = _missing_graspnet_api
    create_table_points = _missing_graspnet_api
    graspnet_get_grasp_score = _missing_graspnet_api
    graspnet_transform_points = _missing_graspnet_api
    graspnet_voxel_sample_points = _missing_graspnet_api

from .analytic_labeler import AnalyticLabelConfig, AnalyticLabeler


FC_LIST = (1.2, 1.0, 0.8, 0.6, 0.4, 0.2)
SCENE_CACHE_SIZE = 1
RELABEL_MAX_CHUNKS_PER_CHILD = 16
FEATURE_MAX_CHUNKS_PER_CHILD = 64
MEMORY_RELEASE_INTERVAL = 8
_PROGRESS_LOG_INTERVAL_SEC = float(os.environ.get("GRARE_PREPARE_LOG_EVERY_SEC", "30"))
_PROGRESS_POLL_SEC = 5.0
# Frames per pool task. GraspNet scenes contain 256 frames per camera. Keeping
# a complete scene in one task is important: each worker owns a scene/Dex-Net
# cache, so splitting one scene into many small tasks makes different workers
# repeatedly voxelize and reload the same meshes, multiplying CPU, RAM and I/O.
# Override for unusual datasets or for a small smoke test.
RELABEL_CHUNK_SIZE = int(os.environ.get("GRARE_PREPARE_CHUNK_SIZE", "256"))
FEATURE_CHUNK_SIZE = int(os.environ.get("GRARE_PREPARE_CHUNK_SIZE", "256"))


def _require_graspnet_api() -> None:
    if _GRASPNET_API_IMPORT_ERROR is None:
        return
    raise ModuleNotFoundError(
        "graspnetAPI is required for GraspNet relabel/eval stages. "
        "Install the public graspnetAPI package before running this stage."
    ) from _GRASPNET_API_IMPORT_ERROR


@dataclass(frozen=True)
class SamObjectCloudConfig:
    """Settings for the SAM-prompted object cloud at relabel time.

    SAM is used in ``SamPredictor`` mode: each candidate's translation is
    projected to a 2D pixel and used as a point prompt; SAM returns three
    multi-scale masks and ``SamCandidatePredictor`` picks the smallest
    valid one. Candidates with similar 3D translations share masks via
    greedy clustering at radius ``cluster_radius_m`` to amortise SAM cost.
    """

    enabled: bool = False
    checkpoint: str = ""
    model_type: str = "vit_t"
    device: str = "cuda"
    multimask_pick: str = "smallest_valid"
    min_area_pixels: int = 200
    max_area_ratio: float = 0.4
    iou_score_floor: float = 0.0
    cluster_radius_m: float = 0.03
    prompt_batch_size: int = 64


@dataclass(frozen=True)
class SceneLabelingConfig:
    """Configuration for scene relabeling on GraspNet1B.

    The object tier cloud is produced by SAM (per-candidate prompt) and
    back-projected via depth + intrinsics; FPS to ``object_cloud_points``.
    """

    detector: str
    benchmark: str  # always "graspnet"
    split: str
    camera: str
    dataset_root: str
    local_cloud_radius: float = 0.04
    local_cloud_max_points: int = 256  # only consulted when sampler != stratified_fps
    voxel_size: float = 0.008
    cloud_sampler: str = "stratified_fps"
    shell_edges_m: tuple[float, ...] = (0.0, 0.005, 0.015, 0.025, 0.040)
    shell_budgets: tuple[int, ...] = (64, 128, 128, 192)
    object_cloud_points: int = 512
    sam: SamObjectCloudConfig = SamObjectCloudConfig()


@dataclass
class FrameContext:
    model_points: list[np.ndarray]
    dexmodel_list: list[Any]
    pose_list: list[np.ndarray]
    collision_scene_points: np.ndarray
    observed_points: np.ndarray
    model_tree: cKDTree | None
    model_tree_labels: np.ndarray
    local_cloud_tree: cKDTree | None
    background_max_dist: float | None
    # Per-pixel 3D points + intrinsics for SAM mask back-projection.
    points_grid: np.ndarray | None = None       # (H, W, 3) float32
    valid_grid: np.ndarray | None = None        # (H, W) bool
    intrinsics: np.ndarray | None = None        # (3, 3) float32
    rgb_image: np.ndarray | None = None         # (H, W, 3) uint8
    # Per-frame mapping from local model index (the position in
    # ``model_points`` / ``dexmodel_list`` / ``model_tree_labels``) to the
    # global GraspNet object id (0..87). Used by the obj-id auxiliary CE
    # head; -1 marks "no GT object" for that index slot.
    global_obj_ids: tuple[int, ...] = ()


@dataclass
class LocalCloudContext:
    observed_points: np.ndarray
    local_cloud_tree: cKDTree | None
    points_grid: np.ndarray | None = None
    valid_grid: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    rgb_image: np.ndarray | None = None


@dataclass
class ObjectAugmentContext:
    model_tree: cKDTree | None
    model_tree_labels: np.ndarray
    background_max_dist: float | None
    points_grid: np.ndarray | None = None
    valid_grid: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    rgb_image: np.ndarray | None = None
    global_obj_ids: tuple[int, ...] = ()


class BatchAnalyticRelabeler:
    """Per-scene analytic relabeler for GraspNet1B."""

    def __init__(self, config: SceneLabelingConfig) -> None:
        if config.benchmark != "graspnet":
            raise ValueError(
                f"grare only supports benchmark='graspnet'; got {config.benchmark!r}"
            )
        if config.local_cloud_max_points < 0:
            raise ValueError("local_cloud_max_points must be non-negative")
        if config.local_cloud_radius <= 0:
            raise ValueError("local_cloud_radius must be positive")

        sampler = config.cloud_sampler
        shell_edges = tuple(config.shell_edges_m)
        shell_budgets = tuple(config.shell_budgets)
        if sampler == "stratified_fps":
            if len(shell_edges) < 2 or any(
                shell_edges[i] >= shell_edges[i + 1] for i in range(len(shell_edges) - 1)
            ):
                raise ValueError("shell_edges_m must be strictly increasing")
            if len(shell_budgets) != len(shell_edges) - 1:
                raise ValueError("shell_budgets length must equal len(shell_edges)-1")
            if any(b <= 0 for b in shell_budgets):
                raise ValueError("each shell budget must be positive")
            if shell_edges[-1] > config.local_cloud_radius + 1e-9:
                raise ValueError("outermost shell edge exceeds local_cloud_radius")

        self.config = SceneLabelingConfig(
            detector=config.detector,
            benchmark=config.benchmark,
            split=config.split,
            camera=config.camera,
            dataset_root=config.dataset_root,
            local_cloud_radius=config.local_cloud_radius,
            local_cloud_max_points=config.local_cloud_max_points,
            voxel_size=config.voxel_size,
            cloud_sampler=sampler,
            shell_edges_m=shell_edges,
            shell_budgets=shell_budgets,
            object_cloud_points=config.object_cloud_points,
            sam=config.sam,
        )
        self.labeler = AnalyticLabeler(
            AnalyticLabelConfig(
                detector=self.config.detector,
                benchmark=self.config.benchmark,
                split=self.config.split,
                camera=self.config.camera,
            )
        )
        self._backend_impl: _GraspNetLabelBackend | None = None

    @property
    def _backend(self) -> _GraspNetLabelBackend:
        if self._backend_impl is None:
            self._backend_impl = _GraspNetLabelBackend(self.config)
        return self._backend_impl

    def relabel_archive(self, candidate_path: str | Path, save_path: str | Path) -> Path:
        return self.labeler.label_archive(
            candidate_path,
            save_path,
            self._label_fn,
        )

    def extract_features_archive(self, candidate_path: str | Path, save_path: str | Path) -> Path:
        return self.labeler.label_archive(
            candidate_path,
            save_path,
            self._feature_fn,
            extra_meta={
                "labels_available": False,
                "feature_stage": "local_cloud_only",
            },
        )

    def relabel_prediction_file(
        self,
        prediction_path: str | Path,
        save_path: str | Path,
        *,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> Path:
        prediction = load_prediction_from_file(
            prediction_path,
            detector=self.config.detector,
            benchmark=self.config.benchmark,
            split=self.config.split,
            camera=self.config.camera,
        )
        return self.labeler.label_prediction(
            prediction,
            save_path,
            self._label_fn,
            extra_meta={
                "source_prediction_file": str(prediction_path),
            },
            include_object_cloud=include_object_cloud,
            object_cloud_save_path=object_cloud_save_path,
        )

    def extract_features_prediction_file(
        self,
        prediction_path: str | Path,
        save_path: str | Path,
    ) -> Path:
        prediction = load_prediction_from_file(
            prediction_path,
            detector=self.config.detector,
            benchmark=self.config.benchmark,
            split=self.config.split,
            camera=self.config.camera,
        )
        return self.labeler.label_prediction(
            prediction,
            save_path,
            self._feature_fn,
            extra_meta={
                "labels_available": False,
                "feature_stage": "local_cloud_only",
                "source_prediction_file": str(prediction_path),
            },
        )

    def augment_archive(
        self,
        candidate_path: str | Path,
        save_path: str | Path,
        *,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> Path:
        """Upgrade one legacy archive to the current schema in place of a relabel.

        Reuses the legacy analytic labels + local_cloud verbatim and only
        computes the two missing object-tier fields. Drops the deprecated
        ``local_features`` channel (ShellAttn consumes xyz only).
        """
        return self.labeler.augment_archive(
            candidate_path,
            save_path,
            self._augment_fn,
            include_object_cloud=include_object_cloud,
            object_cloud_save_path=object_cloud_save_path,
        )

    def relabel_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str = "**/*.npz",
        limit: int | None = None,
        skip_existing: bool = True,
        num_workers: int = 1,
    ) -> dict[str, int]:
        return self._process_tree(
            input_root,
            output_root,
            pattern=pattern,
            limit=limit,
            skip_existing=skip_existing,
            num_workers=num_workers,
            worker_mode="relabel",
        )

    def relabel_dump_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str = "**/*.npy",
        limit: int | None = None,
        skip_existing: bool = True,
        num_workers: int = 1,
        object_cloud_root: str | Path | None = None,
        include_object_cloud: bool = True,
    ) -> dict[str, int]:
        return self._process_tree(
            input_root,
            output_root,
            pattern=pattern,
            limit=limit,
            skip_existing=skip_existing,
            num_workers=num_workers,
            worker_mode="relabel_dump",
            object_cloud_root=object_cloud_root,
            include_object_cloud=include_object_cloud,
        )

    def extract_features_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str = "**/*.npz",
        limit: int | None = None,
        skip_existing: bool = True,
        num_workers: int = 1,
    ) -> dict[str, int]:
        return self._process_tree(
            input_root,
            output_root,
            pattern=pattern,
            limit=limit,
            skip_existing=skip_existing,
            num_workers=num_workers,
            worker_mode="features",
        )

    def extract_features_dump_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str = "**/*.npy",
        limit: int | None = None,
        skip_existing: bool = True,
        num_workers: int = 1,
    ) -> dict[str, int]:
        return self._process_tree(
            input_root,
            output_root,
            pattern=pattern,
            limit=limit,
            skip_existing=skip_existing,
            num_workers=num_workers,
            worker_mode="features_dump",
        )

    def augment_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str = "**/*.npz",
        limit: int | None = None,
        skip_existing: bool = True,
        num_workers: int = 1,
        object_cloud_root: str | Path | None = None,
        include_object_cloud: bool = True,
    ) -> dict[str, int]:
        return self._process_tree(
            input_root,
            output_root,
            pattern=pattern,
            limit=limit,
            skip_existing=skip_existing,
            num_workers=num_workers,
            worker_mode="augment",
            object_cloud_root=object_cloud_root,
            include_object_cloud=include_object_cloud,
        )

    def _process_tree(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        pattern: str,
        limit: int | None,
        skip_existing: bool,
        num_workers: int,
        worker_mode: str,
        object_cloud_root: str | Path | None = None,
        include_object_cloud: bool = True,
    ) -> dict[str, int]:
        input_root = Path(input_root)
        output_root = Path(output_root)
        object_cloud_root = Path(object_cloud_root) if object_cloud_root is not None else None
        candidate_paths = sorted(input_root.glob(pattern))
        if limit is not None:
            candidate_paths = candidate_paths[:limit]

        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        skipped = 0
        pending_paths: list[Path] = []
        validate_existing = True
        for candidate_path in candidate_paths:
            save_path = _output_path_for_input(input_root, output_root, candidate_path)
            object_cloud_save_path = (
                _output_path_for_input(input_root, object_cloud_root, candidate_path)
                if object_cloud_root is not None
                else None
            )
            if skip_existing and _existing_output_is_reusable(
                save_path,
                worker_mode,
                validate=validate_existing,
                object_cloud_path=object_cloud_save_path,
                include_object_cloud=include_object_cloud,
            ):
                skipped += 1
                continue
            pending_paths.append(candidate_path)

        total_pending = len(pending_paths)
        max_chunks_per_child = _max_chunks_per_child(worker_mode)
        release_interval = MEMORY_RELEASE_INTERVAL
        print(
            (
                f"[{worker_mode}] input={input_root.resolve()} output={output_root.resolve()} "
                f"total={len(candidate_paths)} skipped={skipped} pending={total_pending} "
                f"workers={num_workers} max_chunks_per_child={max_chunks_per_child} "
                f"validate_existing={validate_existing}"
            ),
            flush=True,
        )

        if total_pending == 0:
            print(
                f"[{worker_mode}] complete processed=0 skipped={skipped} total={len(candidate_paths)}",
                flush=True,
            )
            return {
                "num_archives": len(candidate_paths),
                "processed": 0,
                "skipped": skipped,
            }

        if num_workers == 1 or total_pending <= 1:
            processed = 0
            with tqdm(
                total=total_pending,
                desc=worker_mode,
                unit="archive",
                dynamic_ncols=True,
                mininterval=2.0,
            ) as progress:
                for candidate_path in pending_paths:
                    save_path = _output_path_for_input(input_root, output_root, candidate_path)
                    object_cloud_save_path = (
                        _output_path_for_input(input_root, object_cloud_root, candidate_path)
                        if object_cloud_root is not None
                        else None
                    )
                    self._process_archive(
                        candidate_path,
                        save_path,
                        worker_mode=worker_mode,
                        include_object_cloud=include_object_cloud,
                        object_cloud_save_path=object_cloud_save_path,
                    )
                    processed += 1
                    self._backend.release_after_archive(
                        worker_mode=worker_mode,
                        release_memory=(processed % release_interval == 0),
                    )
                    progress.update(1)
            self._backend.release_after_archive(
                worker_mode=worker_mode,
                release_memory=True,
            )
            print(f"[{worker_mode}] complete processed={processed} skipped={skipped} total={len(candidate_paths)}", flush=True)
            return {
                "num_archives": len(candidate_paths),
                "processed": processed,
                "skipped": skipped,
            }

        relative_paths = [candidate_path.relative_to(input_root) for candidate_path in pending_paths]
        chunk_size = _chunk_size(worker_mode)
        relative_chunks = _chunk_relative_paths(
            relative_paths,
            chunk_size,
            group_by_scene=True,
        )
        # CUDA-backed SAM workers must initialize independent device contexts.
        sam_uses_cuda = bool(self.config.sam.enabled) and str(self.config.sam.device).startswith("cuda")
        if sam_uses_cuda:
            start_method = "spawn"
        else:
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        processed = 0
        with ctx.Pool(
            processes=min(num_workers, len(relative_chunks)),
            initializer=_init_relabel_worker,
            initargs=(
                self.config,
                str(input_root),
                str(output_root),
                worker_mode,
                None if object_cloud_root is None else str(object_cloud_root),
                include_object_cloud,
            ),
            maxtasksperchild=max_chunks_per_child,
        ) as pool:
            with tqdm(
                total=total_pending,
                desc=worker_mode,
                unit="archive",
                dynamic_ncols=True,
                mininterval=2.0,
            ) as progress:
                progress.set_postfix(chunks=f"0/{len(relative_chunks)}", chunk_size=chunk_size)
                chunk_done = 0
                started = time.perf_counter()
                last_report = started
                results = pool.imap_unordered(
                    _process_relabel_chunk, enumerate(relative_chunks, start=1)
                )
                # Report on a timer rather than per completed chunk. A chunk is
                # many frames and one frame can take seconds, so waiting for a
                # chunk to finish leaves minutes with no output at all. Counting
                # the archives already on disk shows progress within a chunk.
                while True:
                    try:
                        chunk_result = results.next(timeout=_PROGRESS_POLL_SEC)
                    except mp.TimeoutError:
                        chunk_result = None
                    except StopIteration:
                        break

                    if chunk_result is not None:
                        chunk_processed = int(chunk_result["processed"])
                        processed += chunk_processed
                        chunk_done += 1
                        progress.update(chunk_processed)
                        progress.set_postfix(
                            chunks=f"{chunk_done}/{len(relative_chunks)}",
                            chunk_size=chunk_size,
                        )

                    now = time.perf_counter()
                    if now - last_report < _PROGRESS_LOG_INTERVAL_SEC:
                        continue
                    last_report = now
                    written = _count_written_outputs(output_root)
                    done = min(max(written - skipped, 0), total_pending)
                    elapsed = now - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = max(total_pending - done, 0)
                    print(
                        json.dumps(
                            {
                                "stage": f"{worker_mode}_progress",
                                "archives": done,
                                "archives_total": total_pending,
                                "percent": round(100.0 * done / max(total_pending, 1), 2),
                                "chunks": chunk_done,
                                "chunks_total": len(relative_chunks),
                                "archives_per_sec": round(rate, 3),
                                "elapsed_sec": round(elapsed, 1),
                                "eta_sec": round(remaining / rate, 1) if rate > 0 else None,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

        print(f"[{worker_mode}] complete processed={processed} skipped={skipped} total={len(candidate_paths)}", flush=True)

        return {
            "num_archives": len(candidate_paths),
            "processed": processed,
            "skipped": skipped,
        }

    def _label_fn(self, grasp_group_array: np.ndarray, meta: dict[str, Any]) -> dict[str, np.ndarray]:
        scene_id = int(meta["scene_id"])
        frame_id = int(meta["frame_id"])
        return self._backend.label(scene_id, frame_id, grasp_group_array)

    def _feature_fn(self, grasp_group_array: np.ndarray, meta: dict[str, Any]) -> dict[str, np.ndarray]:
        scene_id = int(meta["scene_id"])
        frame_id = int(meta["frame_id"])
        return self._backend.extract_features(scene_id, frame_id, grasp_group_array)

    def _augment_fn(self, grasp_group_array: np.ndarray, meta: dict[str, Any]) -> dict[str, np.ndarray]:
        scene_id = int(meta["scene_id"])
        frame_id = int(meta["frame_id"])
        return self._backend.augment_object_fields(scene_id, frame_id, grasp_group_array)

    def _process_archive(
        self,
        candidate_path: Path,
        save_path: Path,
        *,
        worker_mode: str,
        include_object_cloud: bool = True,
        object_cloud_save_path: str | Path | None = None,
    ) -> None:
        if worker_mode == "relabel":
            self.relabel_archive(candidate_path, save_path)
            return
        if worker_mode.startswith("features"):
            self.extract_features_archive(candidate_path, save_path)
            return
        if worker_mode == "relabel_dump":
            self.relabel_prediction_file(
                candidate_path,
                save_path,
                include_object_cloud=include_object_cloud,
                object_cloud_save_path=object_cloud_save_path,
            )
            return
        if worker_mode == "features_dump":
            self.extract_features_prediction_file(candidate_path, save_path)
            return
        if worker_mode == "augment":
            self.augment_archive(
                candidate_path,
                save_path,
                include_object_cloud=include_object_cloud,
                object_cloud_save_path=object_cloud_save_path,
            )
            return
        raise ValueError(f"Unsupported worker_mode: {worker_mode}")


class _BaseLabelBackend:
    def __init__(self, config: SceneLabelingConfig) -> None:
        self.config = config
        self._scene_cache: dict[int, tuple[list[np.ndarray], list[Any], tuple[int, ...]]] = {}
        self._scene_cache_order: list[int] = []
        self._scene_object_cache: dict[int, tuple[list[np.ndarray], tuple[int, ...]]] = {}
        self._scene_object_cache_order: list[int] = []
        self._frame_cache: tuple[int, int, FrameContext] | None = None
        self._label_context_cache: tuple[int, int, FrameContext] | None = None
        self._local_cloud_cache: tuple[int, int, LocalCloudContext] | None = None
        self._object_context_cache: tuple[int, int, ObjectAugmentContext] | None = None
        self._sam_predictor = None  # type: ignore[assignment]

    def label(self, scene_id: int, frame_id: int, grasp_group_array: np.ndarray) -> dict[str, np.ndarray]:
        grasp_group_array = np.asarray(grasp_group_array, dtype=np.float32)
        if grasp_group_array.ndim != 2:
            raise ValueError("grasp_group_array must have rank 2")

        num_grasps = len(grasp_group_array)
        frame = self._get_frame_context(scene_id, frame_id)
        local_cloud, cloud_mask = self._extract_local_cloud(
            grasp_group_array,
            observed_points=frame.observed_points,
            local_cloud_tree=frame.local_cloud_tree,
        )
        # SAM-prompted object cloud per candidate, fed to the PointNet++
        # object encoder at train time. Output is in the camera frame.
        object_cloud = self._extract_object_cloud(
            grasp_group_array,
            points_grid=frame.points_grid,
            valid_grid=frame.valid_grid,
            intrinsics=frame.intrinsics,
            rgb_image=frame.rgb_image,
        )
        labels = self._compute_analytic_labels(frame, grasp_group_array)
        labels.update(
            {
                "local_cloud": local_cloud,
                "cloud_mask": cloud_mask,
                "object_cloud": object_cloud,
            }
        )
        return labels

    def label_only(self, scene_id: int, frame_id: int, grasp_group_array: np.ndarray) -> dict[str, np.ndarray]:
        grasp_group_array = np.asarray(grasp_group_array, dtype=np.float32)
        if grasp_group_array.ndim != 2:
            raise ValueError("grasp_group_array must have rank 2")

        frame = self._get_label_context(scene_id, frame_id)
        return self._compute_analytic_labels(frame, grasp_group_array)

    def _compute_analytic_labels(
        self,
        frame: FrameContext,
        grasp_group_array: np.ndarray,
    ) -> dict[str, np.ndarray]:
        grasp_group_array = np.asarray(grasp_group_array, dtype=np.float32)
        num_grasps = len(grasp_group_array)
        if num_grasps == 0:
            return {
                "mu_min": np.zeros((0,), dtype=np.float32),
                "is_collision": np.zeros((0,), dtype=bool),
                "is_empty": np.zeros((0,), dtype=bool),
                "object_assignments": np.zeros((0,), dtype=np.int32),
            }

        if frame.model_tree is None:
            return {
                "mu_min": np.full((num_grasps,), np.inf, dtype=np.float32),
                "is_collision": np.zeros((num_grasps,), dtype=bool),
                "is_empty": np.ones((num_grasps,), dtype=bool),
                "object_assignments": np.full((num_grasps,), -1, dtype=np.int32),
            }

        clipped = grasp_group_array.copy()
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, self.max_width)
        translations = clipped[:, 13:16]

        nearest_dist, nearest_idx = frame.model_tree.query(translations, k=1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
        nearest_dist = np.asarray(nearest_dist, dtype=np.float32).reshape(-1)
        object_assignments = frame.model_tree_labels[nearest_idx]

        valid_mask = np.ones((num_grasps,), dtype=bool)
        if frame.background_max_dist is not None:
            valid_mask &= nearest_dist < frame.background_max_dist

        grasp_list = [
            clipped[(object_assignments == object_idx) & valid_mask]
            for object_idx in range(len(frame.model_points))
        ]
        collision_mask_list, empty_mask_list, dexgrasp_list = self.collision_detection(
            grasp_list,
            frame.model_points,
            frame.dexmodel_list,
            frame.pose_list,
            frame.collision_scene_points,
            return_dexgrasps=True,
        )

        mu_min = np.full((num_grasps,), np.inf, dtype=np.float32)
        is_collision = np.zeros((num_grasps,), dtype=bool)
        is_empty = np.zeros((num_grasps,), dtype=bool)

        for object_idx in range(len(frame.model_points)):
            archive_indices = np.flatnonzero((object_assignments == object_idx) & valid_mask)
            if len(archive_indices) == 0:
                continue

            collision_mask = np.asarray(collision_mask_list[object_idx], dtype=bool)
            empty_mask = np.asarray(empty_mask_list[object_idx], dtype=bool)
            is_empty[archive_indices] = empty_mask
            is_collision[archive_indices] = collision_mask & ~empty_mask

            for local_idx, dexgrasp in enumerate(dexgrasp_list[object_idx]):
                if dexgrasp is None or collision_mask[local_idx]:
                    continue
                score = self._get_grasp_score_impl(
                    dexgrasp,
                    frame.dexmodel_list[object_idx],
                    self.force_closure_quality_config,
                )
                if score > 0:
                    mu_min[archive_indices[local_idx]] = float(score)

        if frame.background_max_dist is not None:
            is_empty[~valid_mask] = True

        # obj-id supervision: map per-frame model index -> global GraspNet
        # object id (0..87). Falls back to -1 when metadata is unavailable;
        # CrossEntropyLoss uses ignore_index=-1.
        global_obj_ids = frame.global_obj_ids
        if global_obj_ids:
            global_assignments = np.full((num_grasps,), -1, dtype=np.int32)
            valid_local = (object_assignments >= 0) & (object_assignments < len(global_obj_ids)) & valid_mask
            local_idx_array = object_assignments.astype(np.int64, copy=False)
            global_lookup = np.asarray(global_obj_ids, dtype=np.int32)
            global_assignments[valid_local] = global_lookup[local_idx_array[valid_local]]
        else:
            global_assignments = np.full((num_grasps,), -1, dtype=np.int32)

        return {
            "mu_min": mu_min,
            "is_collision": is_collision,
            "is_empty": is_empty,
            "object_assignments": global_assignments,
        }

    def augment_object_fields(
        self, scene_id: int, frame_id: int, grasp_group_array: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute only the object-tier fields (object_assignments + object_cloud).

        Used to upgrade a legacy v2 candidate archive (which already carries
        the expensive analytic labels mu_min / is_collision / is_empty and the
        local_cloud) to the current schema without re-running the BLAS-bound dexnet
        force-closure + collision pass. We reuse the cached frame context for
        the GT-model KD-tree (object_assignments) and the depth/RGB grids
        (SAM object_cloud); neither depends on collision_detection.
        """
        grasp_group_array = np.asarray(grasp_group_array, dtype=np.float32)
        if grasp_group_array.ndim != 2:
            raise ValueError("grasp_group_array must have rank 2")

        num_grasps = len(grasp_group_array)
        frame = self._get_object_context(scene_id, frame_id)
        object_cloud = self._extract_object_cloud(
            grasp_group_array,
            points_grid=frame.points_grid,
            valid_grid=frame.valid_grid,
            intrinsics=frame.intrinsics,
            rgb_image=frame.rgb_image,
        )

        if num_grasps == 0 or frame.model_tree is None:
            return {
                "object_assignments": np.full((num_grasps,), -1, dtype=np.int32),
                "object_cloud": object_cloud,
            }

        clipped = grasp_group_array.copy()
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, self.max_width)
        translations = clipped[:, 13:16]

        nearest_dist, nearest_idx = frame.model_tree.query(translations, k=1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
        nearest_dist = np.asarray(nearest_dist, dtype=np.float32).reshape(-1)
        object_assignments = frame.model_tree_labels[nearest_idx]

        valid_mask = np.ones((num_grasps,), dtype=bool)
        if frame.background_max_dist is not None:
            valid_mask &= nearest_dist < frame.background_max_dist

        # Map frame-local model index -> global GraspNet object id (0..87).
        # Identical to label()'s obj-id supervision so the upgraded archive
        # matches a freshly-relabeled archive bit-for-bit on this field.
        global_obj_ids = frame.global_obj_ids
        if global_obj_ids:
            global_assignments = np.full((num_grasps,), -1, dtype=np.int32)
            valid_local = (
                (object_assignments >= 0)
                & (object_assignments < len(global_obj_ids))
                & valid_mask
            )
            local_idx_array = object_assignments.astype(np.int64, copy=False)
            global_lookup = np.asarray(global_obj_ids, dtype=np.int32)
            global_assignments[valid_local] = global_lookup[local_idx_array[valid_local]]
        else:
            global_assignments = np.full((num_grasps,), -1, dtype=np.int32)

        return {
            "object_assignments": global_assignments,
            "object_cloud": object_cloud,
        }

    def extract_features(self, scene_id: int, frame_id: int, grasp_group_array: np.ndarray) -> dict[str, np.ndarray]:
        grasp_group_array = np.asarray(grasp_group_array, dtype=np.float32)
        if grasp_group_array.ndim != 2:
            raise ValueError("grasp_group_array must have rank 2")

        context = self._get_local_cloud_context(scene_id, frame_id)
        local_cloud, cloud_mask = self._extract_local_cloud(
            grasp_group_array,
            observed_points=context.observed_points,
            local_cloud_tree=context.local_cloud_tree,
        )
        # SAM-prompted object cloud at test time. extract_features() is the
        # test-side path so it must NOT touch any GT label (no obj_id, no
        # mu_min) -- only the geometry.
        object_cloud = self._extract_object_cloud(
            grasp_group_array,
            points_grid=context.points_grid,
            valid_grid=context.valid_grid,
            intrinsics=context.intrinsics,
            rgb_image=context.rgb_image,
        )
        return {
            "local_cloud": local_cloud,
            "cloud_mask": cloud_mask,
            "object_cloud": object_cloud,
        }

    def _extract_local_cloud(
        self,
        grasp_group_array: np.ndarray,
        *,
        observed_points: np.ndarray,
        local_cloud_tree: cKDTree | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (local_cloud, cloud_mask).

        local_cloud : (N, P, 3) float32  point xyz in gripper-local frame
        cloud_mask  : (N, P) bool        True for real points, False for zero-pad
        """
        cfg = self.config
        sampler = cfg.cloud_sampler
        num_grasps = len(grasp_group_array)

        if sampler == "stratified_fps":
            P = int(sum(cfg.shell_budgets))
        else:
            P = int(cfg.local_cloud_max_points)

        local_cloud = np.zeros((num_grasps, P, 3), dtype=np.float32)
        cloud_mask = np.zeros((num_grasps, P), dtype=np.bool_)
        if num_grasps == 0 or local_cloud_tree is None or P == 0:
            return local_cloud, cloud_mask

        translations = grasp_group_array[:, 13:16]
        rotations = grasp_group_array[:, 4:13].reshape(-1, 3, 3)
        neighbor_indices = local_cloud_tree.query_ball_point(translations, r=cfg.local_cloud_radius)

        for i, indices in enumerate(neighbor_indices):
            if not indices:
                continue
            arr = observed_points[np.asarray(indices, dtype=np.int64)]
            centred = arr - translations[i]
            local_pts = np.matmul(centred, rotations[i]).astype(np.float32, copy=False)

            if sampler == "stratified_fps":
                kept_idx, slot_offsets = _stratified_fps_select(
                    local_pts, cfg.shell_edges_m, cfg.shell_budgets
                )
                kept_pts = local_pts[kept_idx] if len(kept_idx) else local_pts[:0]
                local_cloud[i, slot_offsets] = kept_pts
                cloud_mask[i, slot_offsets] = True
            else:
                if len(local_pts) > P:
                    distances = (local_pts * local_pts).sum(axis=1)
                    keep = np.argsort(distances)[:P]
                    local_pts = local_pts[keep]
                m = len(local_pts)
                local_cloud[i, :m] = local_pts
                cloud_mask[i, :m] = True

        return local_cloud, cloud_mask

    def _extract_object_cloud(
        self,
        grasp_group_array: np.ndarray,
        *,
        points_grid: np.ndarray | None,
        valid_grid: np.ndarray | None,
        intrinsics: np.ndarray | None,
        rgb_image: np.ndarray | None,
    ) -> np.ndarray:
        """SAM-prompted object cloud: per-candidate point prompt → smallest
        valid mask → back-project depth pixels in mask → FPS to budget.

        Candidates with similar 3D translations share masks via greedy
        clustering at radius ``cfg.sam.cluster_radius_m`` to amortise SAM cost.

        Returns ``(N, P_obj, 3)`` float32 (camera frame). Output is zero-padded
        for candidates whose SAM call yields no valid mask.
        """
        cfg = self.config
        num_grasps = len(grasp_group_array)
        P = int(cfg.object_cloud_points)
        out = np.zeros((num_grasps, P, 3), dtype=np.float32)
        if (
            num_grasps == 0
            or P == 0
            or not cfg.sam.enabled
            or points_grid is None
            or valid_grid is None
            or intrinsics is None
            or rgb_image is None
        ):
            return out

        from .sam_masks import (
            SamCandidatePredictor,
            SamPredictorConfig,
            project_3d_to_pixel,
            back_project_mask_to_points,
        )

        if self._sam_predictor is None:
            self._sam_predictor = SamCandidatePredictor(
                SamPredictorConfig(
                    checkpoint=cfg.sam.checkpoint,
                    model_type=cfg.sam.model_type,
                    device=cfg.sam.device,
                    multimask_pick=cfg.sam.multimask_pick,
                    min_area_pixels=cfg.sam.min_area_pixels,
                    max_area_ratio=cfg.sam.max_area_ratio,
                    iou_score_floor=cfg.sam.iou_score_floor,
                    prompt_batch_size=cfg.sam.prompt_batch_size,
                )
            )
        predictor = self._sam_predictor
        predictor.set_image(rgb_image)

        translations = grasp_group_array[:, 13:16].astype(np.float32, copy=False)
        # Greedy 3D clustering to share masks across nearby candidates.
        cluster_radius = float(cfg.sam.cluster_radius_m)
        cluster_owner = np.full((num_grasps,), -1, dtype=np.int32)
        unique_centroids: list[np.ndarray] = []
        for i in range(num_grasps):
            if cluster_owner[i] >= 0:
                continue
            ti = translations[i]
            cluster_owner[i] = len(unique_centroids)
            unique_centroids.append(ti)
            unassigned = cluster_owner < 0
            if not np.any(unassigned):
                continue
            d = np.linalg.norm(translations - ti, axis=1)
            cluster_owner[unassigned & (d < cluster_radius)] = len(unique_centroids) - 1

        # Run SAM once per cluster centroid. Cache (mask -> FPS subset).
        cluster_object_pts: list[np.ndarray | None] = [None] * len(unique_centroids)
        prompt_pixels: list[tuple[int, int]] = []
        prompt_cluster_ids: list[int] = []
        for cid, centroid in enumerate(unique_centroids):
            u, v = project_3d_to_pixel(centroid, intrinsics)
            if u < 0 or v < 0:
                continue
            prompt_pixels.append((u, v))
            prompt_cluster_ids.append(cid)

        masks = predictor.predict_many_at_pixels(prompt_pixels)
        for cid, mask in zip(prompt_cluster_ids, masks):
            if mask is None:
                continue
            obj_pts = back_project_mask_to_points(mask, points_grid, valid_grid)
            if len(obj_pts) == 0:
                continue
            if len(obj_pts) > P:
                kept = _fps_indices(obj_pts, P, seed_strategy="farthest")
                obj_pts = obj_pts[kept]
            cluster_object_pts[cid] = obj_pts.astype(np.float32, copy=False)

        for i in range(num_grasps):
            cid = int(cluster_owner[i])
            pts = cluster_object_pts[cid]
            if pts is None:
                continue
            m = len(pts)
            out[i, :m] = pts
        return out

    def _get_frame_context(self, scene_id: int, frame_id: int) -> FrameContext:
        if self._frame_cache is not None and self._frame_cache[:2] == (scene_id, frame_id):
            return self._frame_cache[2]
        context = self._load_frame_context(scene_id, frame_id)
        self._frame_cache = (scene_id, frame_id, context)
        return context

    def _get_label_context(self, scene_id: int, frame_id: int) -> FrameContext:
        if self._label_context_cache is not None and self._label_context_cache[:2] == (scene_id, frame_id):
            return self._label_context_cache[2]
        context = self._load_label_context(scene_id, frame_id)
        self._label_context_cache = (scene_id, frame_id, context)
        return context

    def _get_local_cloud_context(self, scene_id: int, frame_id: int) -> LocalCloudContext:
        if self._local_cloud_cache is not None and self._local_cloud_cache[:2] == (scene_id, frame_id):
            return self._local_cloud_cache[2]
        context = self._load_local_cloud_context(scene_id, frame_id)
        self._local_cloud_cache = (scene_id, frame_id, context)
        return context

    def _get_object_context(self, scene_id: int, frame_id: int) -> ObjectAugmentContext:
        if self._object_context_cache is not None and self._object_context_cache[:2] == (scene_id, frame_id):
            return self._object_context_cache[2]
        context = self._load_object_context(scene_id, frame_id)
        self._object_context_cache = (scene_id, frame_id, context)
        return context

    def _get_scene_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], list[Any], tuple[int, ...]]:
        if scene_id not in self._scene_cache:
            assets = self._load_scene_assets(scene_id, ann_id)
            self._scene_cache[scene_id] = assets
            self._scene_cache_order.append(scene_id)
            self._prune_scene_cache()
        return self._scene_cache[scene_id]

    def _get_scene_object_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], tuple[int, ...]]:
        if scene_id not in self._scene_object_cache:
            assets = self._load_scene_object_assets(scene_id, ann_id)
            self._scene_object_cache[scene_id] = assets
            self._scene_object_cache_order.append(scene_id)
            self._prune_scene_cache()
        return self._scene_object_cache[scene_id]

    def release_after_archive(
        self,
        *,
        worker_mode: str,
        release_memory: bool,
    ) -> None:
        self._frame_cache = None
        self._label_context_cache = None
        self._local_cloud_cache = None
        self._object_context_cache = None
        if worker_mode == "features":
            # Feature extraction reads only observed point clouds. It does not
            # need scene model/dex caches, so keep memory flat in long runs.
            self._scene_cache.clear()
            self._scene_cache_order.clear()
            self._scene_object_cache.clear()
            self._scene_object_cache_order.clear()
        if release_memory:
            # Labels are CPU-only. Avoid importing/probing torch.cuda in every
            # labels worker: that initializes CUDA (and can emit warnings) even
            # though this stage never uses the GPU. The object/SAM stage opts
            # into CUDA cache release below.
            sam_device = str(self.config.sam.device).strip().lower()
            _release_python_and_torch_memory(
                release_cuda=self.config.sam.enabled and sam_device.startswith("cuda")
            )

    def _prune_scene_cache(self) -> None:
        for cache, order in (
            (self._scene_cache, self._scene_cache_order),
            (self._scene_object_cache, self._scene_object_cache_order),
        ):
            while len(order) > SCENE_CACHE_SIZE:
                old_scene_id = order.pop(0)
                cache.pop(old_scene_id, None)

    def _load_frame_context(self, scene_id: int, frame_id: int) -> FrameContext:
        raise NotImplementedError

    def _load_label_context(self, scene_id: int, frame_id: int) -> FrameContext:
        raise NotImplementedError

    def _load_local_cloud_context(self, scene_id: int, frame_id: int) -> LocalCloudContext:
        raise NotImplementedError

    def _load_object_context(self, scene_id: int, frame_id: int) -> ObjectAugmentContext:
        raise NotImplementedError

    def _load_scene_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], list[Any], tuple[int, ...]]:
        raise NotImplementedError

    def _load_scene_object_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], tuple[int, ...]]:
        raise NotImplementedError


class _GraspNetLabelBackend(_BaseLabelBackend):
    max_width = 0.10
    collision_detection = staticmethod(graspnet_collision_detection)

    def __init__(self, config: SceneLabelingConfig) -> None:
        _require_graspnet_api()
        super().__init__(config)
        with _suppress_external_progress():
            self.evaluator = GraspNetEval(root=config.dataset_root, camera=config.camera, split=config.split)
        self.force_closure_quality_config = _build_force_closure_config(
            get_graspnet_config,
            GraspNetGraspQualityConfigFactory,
        )
        score_impl = _fast_grasp_score if _fast_force_closure_enabled() else graspnet_get_grasp_score
        self._get_grasp_score_impl = staticmethod(
            lambda dexgrasp, dexmodel, quality_config: score_impl(
                dexgrasp,
                dexmodel,
                FC_LIST,
                quality_config,
            )
        )

    def _load_scene_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], list[Any], tuple[int, ...]]:
        model_list, dexmodel_list, obj_list = self.evaluator.get_scene_models(scene_id, ann_id)
        sampled_models = [graspnet_voxel_sample_points(model, self.config.voxel_size) for model in model_list]
        # obj_list contains the global GraspNet object ids (0..87) for the
        # frame's local index slots; the obj-id aux head consumes this.
        global_obj_ids = tuple(int(x) for x in obj_list)
        return sampled_models, dexmodel_list, global_obj_ids

    def _load_scene_object_assets(self, scene_id: int, ann_id: int) -> tuple[list[np.ndarray], tuple[int, ...]]:
        """Load only object point clouds needed by object-only augmentation.

        Unlike ``get_scene_models()``, this intentionally does not load Dex-Net
        graspable objects. Object augmentation needs a KD-tree over posed model
        points for object_assignments plus RGB/depth for SAM; Dex-Net models are
        only needed by analytic collision and force-closure labels.
        """
        import open3d as o3d

        obj_list, _, _, _ = self.evaluator.get_model_poses(scene_id, ann_id)
        model_root = Path(self.config.dataset_root) / "models"
        sampled_models: list[np.ndarray] = []
        for obj_idx in obj_list:
            model_path = model_root / f"{int(obj_idx):03d}" / "nontextured.ply"
            model = o3d.io.read_point_cloud(str(model_path))
            points = np.asarray(model.points)
            sampled_models.append(graspnet_voxel_sample_points(points, self.config.voxel_size))
        return sampled_models, tuple(int(x) for x in obj_list)

    def _load_sam_inputs(self, scene_id: int, frame_id: int) -> dict[str, np.ndarray | None]:
        """Load RGB + per-pixel xyz grid + intrinsics for SAM mask back-projection.

        Returns empty/None entries when SAM is disabled in config.
        """
        if not self.config.sam.enabled:
            return {"rgb_image": None, "points_grid": None, "valid_grid": None, "intrinsics": None}
        import os
        import cv2
        from .sam_masks import points_grid_from_depth

        scene_dir = os.path.join(
            self.config.dataset_root, "scenes", f"scene_{scene_id:04d}", self.config.camera,
        )
        rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{frame_id:04d}.png"))
        if rgb is None:
            return {"rgb_image": None, "points_grid": None, "valid_grid": None, "intrinsics": None}
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(scene_dir, "depth", f"{frame_id:04d}.png"), cv2.IMREAD_UNCHANGED)
        intrinsics = np.load(os.path.join(scene_dir, "camK.npy"))
        points_grid, valid_grid = points_grid_from_depth(depth, intrinsics)
        return {
            "rgb_image": rgb,
            "points_grid": points_grid,
            "valid_grid": valid_grid,
            "intrinsics": intrinsics.astype(np.float32, copy=False),
        }

    def _load_frame_context(self, scene_id: int, frame_id: int) -> FrameContext:
        model_list, dexmodel_list, global_obj_ids = self._get_scene_assets(scene_id, ann_id=0)
        _, pose_list, camera_pose, align_mat = self.evaluator.get_model_poses(scene_id, frame_id)
        model_points = [
            graspnet_transform_points(model, pose)
            for model, pose in zip(model_list, pose_list)
        ]
        collision_scene_points = np.concatenate(model_points, axis=0)
        table = create_table_points(
            1.0,
            1.0,
            0.05,
            dx=-0.5,
            dy=-0.5,
            dz=-0.05,
            grid_size=self.config.voxel_size,
        )
        table_transform = graspnet_transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        collision_scene_points = np.concatenate([collision_scene_points, table_transform], axis=0)

        observed_points, _ = self.evaluator.loadScenePointCloud(
            scene_id,
            self.config.camera,
            frame_id,
            align=False,
            format="numpy",
            use_workspace=False,
            use_mask=True,
        )
        sam_inputs = self._load_sam_inputs(scene_id, frame_id)
        return _build_frame_context(
            model_points=model_points,
            dexmodel_list=dexmodel_list,
            pose_list=pose_list,
            collision_scene_points=collision_scene_points,
            observed_points=observed_points,
            background_max_dist=None,
            global_obj_ids=global_obj_ids,
            **sam_inputs,
        )

    def _load_label_context(self, scene_id: int, frame_id: int) -> FrameContext:
        model_list, dexmodel_list, global_obj_ids = self._get_scene_assets(scene_id, ann_id=0)
        _, pose_list, camera_pose, align_mat = self.evaluator.get_model_poses(scene_id, frame_id)
        model_points = [
            graspnet_transform_points(model, pose)
            for model, pose in zip(model_list, pose_list)
        ]
        collision_scene_points = np.concatenate(model_points, axis=0)
        table = create_table_points(
            1.0,
            1.0,
            0.05,
            dx=-0.5,
            dy=-0.5,
            dz=-0.05,
            grid_size=self.config.voxel_size,
        )
        table_transform = graspnet_transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        collision_scene_points = np.concatenate([collision_scene_points, table_transform], axis=0)

        return _build_frame_context(
            model_points=model_points,
            dexmodel_list=dexmodel_list,
            pose_list=pose_list,
            collision_scene_points=collision_scene_points,
            observed_points=np.zeros((0, 3), dtype=np.float32),
            background_max_dist=None,
            global_obj_ids=global_obj_ids,
            points_grid=None,
            valid_grid=None,
            intrinsics=None,
            rgb_image=None,
        )

    def _load_local_cloud_context(self, scene_id: int, frame_id: int) -> LocalCloudContext:
        observed_points, _ = self.evaluator.loadScenePointCloud(
            scene_id,
            self.config.camera,
            frame_id,
            align=False,
            format="numpy",
            use_workspace=False,
            use_mask=True,
        )
        sam_inputs = self._load_sam_inputs(scene_id, frame_id)
        return _build_local_cloud_context(observed_points, **sam_inputs)

    def _load_object_context(self, scene_id: int, frame_id: int) -> ObjectAugmentContext:
        model_list, global_obj_ids = self._get_scene_object_assets(scene_id, ann_id=0)
        _, pose_list, _, _ = self.evaluator.get_model_poses(scene_id, frame_id)
        model_points = [
            graspnet_transform_points(model, pose)
            for model, pose in zip(model_list, pose_list)
        ]
        sam_inputs = self._load_sam_inputs(scene_id, frame_id)
        return _build_object_context(
            model_points=model_points,
            background_max_dist=None,
            global_obj_ids=global_obj_ids,
            **sam_inputs,
        )


def _build_frame_context(
    *,
    model_points: list[np.ndarray],
    dexmodel_list: list[Any],
    pose_list: list[np.ndarray],
    collision_scene_points: np.ndarray,
    observed_points: np.ndarray,
    background_max_dist: float | None,
    global_obj_ids: tuple[int, ...] = (),
    points_grid: np.ndarray | None = None,
    valid_grid: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    rgb_image: np.ndarray | None = None,
) -> FrameContext:
    observed_points = np.asarray(observed_points, dtype=np.float32)
    local_cloud_tree = cKDTree(observed_points) if len(observed_points) > 0 else None

    if model_points:
        merged_points = np.concatenate(model_points, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(
            [
                np.full((len(points),), object_idx, dtype=np.int32)
                for object_idx, points in enumerate(model_points)
            ],
            axis=0,
        )
        model_tree = cKDTree(merged_points) if len(merged_points) > 0 else None
    else:
        labels = np.zeros((0,), dtype=np.int32)
        model_tree = None

    return FrameContext(
        model_points=[np.asarray(points, dtype=np.float32) for points in model_points],
        dexmodel_list=dexmodel_list,
        pose_list=pose_list,
        collision_scene_points=np.asarray(collision_scene_points, dtype=np.float32),
        observed_points=observed_points,
        model_tree=model_tree,
        model_tree_labels=labels,
        local_cloud_tree=local_cloud_tree,
        background_max_dist=background_max_dist,
        points_grid=points_grid,
        valid_grid=valid_grid,
        intrinsics=intrinsics,
        rgb_image=rgb_image,
        global_obj_ids=global_obj_ids,
    )


def _build_local_cloud_context(
    observed_points: np.ndarray,
    *,
    points_grid: np.ndarray | None = None,
    valid_grid: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    rgb_image: np.ndarray | None = None,
) -> LocalCloudContext:
    observed_points = np.asarray(observed_points, dtype=np.float32)
    return LocalCloudContext(
        observed_points=observed_points,
        local_cloud_tree=cKDTree(observed_points) if len(observed_points) > 0 else None,
        points_grid=points_grid,
        valid_grid=valid_grid,
        intrinsics=intrinsics,
        rgb_image=rgb_image,
    )


def _build_object_context(
    *,
    model_points: list[np.ndarray],
    background_max_dist: float | None,
    global_obj_ids: tuple[int, ...] = (),
    points_grid: np.ndarray | None = None,
    valid_grid: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    rgb_image: np.ndarray | None = None,
) -> ObjectAugmentContext:
    if model_points:
        merged_points = np.concatenate(model_points, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(
            [
                np.full((len(points),), object_idx, dtype=np.int32)
                for object_idx, points in enumerate(model_points)
            ],
            axis=0,
        )
        model_tree = cKDTree(merged_points) if len(merged_points) > 0 else None
    else:
        labels = np.zeros((0,), dtype=np.int32)
        model_tree = None
    return ObjectAugmentContext(
        model_tree=model_tree,
        model_tree_labels=labels,
        background_max_dist=background_max_dist,
        points_grid=points_grid,
        valid_grid=valid_grid,
        intrinsics=intrinsics,
        rgb_image=rgb_image,
        global_obj_ids=global_obj_ids,
    )


def _shell_ids_for(points: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=np.int32)
    radii = np.linalg.norm(points, axis=1)
    edges_arr = np.asarray(edges, dtype=np.float32)
    ids = np.searchsorted(edges_arr, radii, side="right") - 1
    np.clip(ids, 0, len(edges_arr) - 2, out=ids)
    return ids.astype(np.int32, copy=False)


def _fps_indices(points: np.ndarray, budget: int, seed_strategy: str = "farthest") -> np.ndarray:
    n = len(points)
    if n == 0 or budget == 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    # Avoid einsum dispatch in the inner loop. Component-wise float32
    # arithmetic retains the original recurrence and tie-breaking while
    # being considerably cheaper for these 3D vectors.
    radii = np.linalg.norm(points, axis=1)
    if seed_strategy == "nearest":
        seed = int(np.argmin(radii))
    else:
        seed = int(np.argmax(radii))
    selected = np.empty((budget,), dtype=np.int64)
    selected[0] = seed
    distances = np.full((n,), np.inf, dtype=np.float32)
    for s in range(1, budget):
        last = points[selected[s - 1]]
        diff = points - last
        d_new = (
            diff[:, 0] * diff[:, 0]
            + diff[:, 1] * diff[:, 1]
            + diff[:, 2] * diff[:, 2]
        )
        np.minimum(distances, d_new, out=distances)
        selected[s] = int(np.argmax(distances))
    return selected


def _stratified_fps_select(
    points: np.ndarray,
    shell_edges: tuple[float, ...],
    shell_budgets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    radii = np.linalg.norm(points, axis=1)
    kept_idx_parts: list[np.ndarray] = []
    slot_offsets_parts: list[np.ndarray] = []
    cursor = 0
    for s, budget in enumerate(shell_budgets):
        lo = shell_edges[s]
        hi = shell_edges[s + 1]
        if s == len(shell_budgets) - 1:
            in_shell = (radii >= lo) & (radii <= hi + 1e-9)
        else:
            in_shell = (radii >= lo) & (radii < hi)
        shell_global = np.flatnonzero(in_shell)
        if shell_global.size:
            shell_pts = points[shell_global]
            local_kept = _fps_indices(shell_pts, budget, seed_strategy="farthest")
            picked_global = shell_global[local_kept]
            kept_idx_parts.append(picked_global)
            slot_offsets_parts.append(cursor + np.arange(picked_global.size, dtype=np.int64))
        cursor += budget
    if not kept_idx_parts:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(kept_idx_parts), np.concatenate(slot_offsets_parts)


def _build_force_closure_config(get_config_fn: Any, quality_factory: Any) -> dict[float, Any]:
    config = get_config_fn()
    quality_config: dict[float, Any] = {}
    for value_fc in FC_LIST:
        value = float(round(float(value_fc), 2))
        config["metrics"]["force_closure"]["friction_coef"] = value
        quality_config[value] = quality_factory.create_config(config["metrics"]["force_closure"])
    return quality_config


def _fast_grasp_score(
    grasp: Any,
    obj: Any,
    fc_list: tuple[float, ...],
    force_closure_quality_config: dict[float, Any],
) -> float:
    """Equivalent GraspNet force-closure score with contacts computed once.

    GraspNetAPI's get_grasp_score calls grasp_quality once per friction
    coefficient. For the default force_closure metric that recomputes the same
    finger contacts up to six times, even though the contacts do not depend on
    the friction coefficient. Reusing contacts keeps the benchmark definition
    unchanged while removing the largest relabel CPU hotspot.
    """
    if not fc_list:
        return -1.0

    first_fc = float(round(float(fc_list[0]), 2))
    first_config = force_closure_quality_config[first_fc]
    if getattr(first_config, "quality_method", None) != "force_closure":
        return float(graspnet_get_grasp_score(grasp, obj, fc_list, force_closure_quality_config))

    try:
        contacts_found, contacts = grasp.close_fingers(
            obj,
            check_approach=first_config.check_approach,
            vis=False,
        )
    except Exception:
        return float(graspnet_get_grasp_score(grasp, obj, fc_list, force_closure_quality_config))

    if not contacts_found:
        return -1.0
    if len(contacts) != 2:
        return float(graspnet_get_grasp_score(grasp, obj, fc_list, force_closure_quality_config))

    quality = -1.0
    was_force_closure = False
    last_fc = float(round(float(fc_list[-1]), 2))
    for idx, fc in enumerate(fc_list):
        value_fc = float(round(float(fc), 2))
        previous_force_closure = was_force_closure
        was_force_closure = bool(
            GraspNetPointGraspMetrics3D.force_closure(
                contacts[0],
                contacts[1],
                value_fc,
            )
        )
        if previous_force_closure and not was_force_closure:
            return float(round(float(fc_list[idx - 1]), 2))
        if was_force_closure and value_fc == last_fc:
            return value_fc
        if value_fc == first_fc and not was_force_closure:
            return quality
    return quality


def _fast_force_closure_enabled() -> bool:
    raw = os.environ.get("GRARE_RELABEL_FAST_FC", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_WORKER_RELABELER: BatchAnalyticRelabeler | None = None
_WORKER_INPUT_ROOT: Path | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_OBJECT_CLOUD_ROOT: Path | None = None
_WORKER_MODE: str | None = None
_WORKER_INCLUDE_OBJECT_CLOUD = True
_WORKER_ARCHIVES_PROCESSED = 0


def _init_relabel_worker(
    config: SceneLabelingConfig,
    input_root: str,
    output_root: str,
    worker_mode: str,
    object_cloud_root: str | None = None,
    include_object_cloud: bool = True,
) -> None:
    global _WORKER_RELABELER, _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT, _WORKER_OBJECT_CLOUD_ROOT, _WORKER_MODE, _WORKER_INCLUDE_OBJECT_CLOUD, _WORKER_ARCHIVES_PROCESSED
    config = _assign_worker_sam_device(config)
    _WORKER_RELABELER = BatchAnalyticRelabeler(config)
    _WORKER_INPUT_ROOT = Path(input_root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_OBJECT_CLOUD_ROOT = Path(object_cloud_root) if object_cloud_root is not None else None
    _WORKER_MODE = worker_mode
    _WORKER_INCLUDE_OBJECT_CLOUD = bool(include_object_cloud)
    _WORKER_ARCHIVES_PROCESSED = 0


def _assign_worker_sam_device(config: SceneLabelingConfig) -> SceneLabelingConfig:
    sam = config.sam
    if not sam.enabled:
        return config
    if str(sam.device).strip().lower() not in {"cuda", "auto"}:
        return config
    device_count = _visible_cuda_device_count()
    if device_count <= 1:
        return config
    identity = mp.current_process()._identity
    raw_index = int(identity[0] - 1) if identity else os.getpid()
    worker_device = f"cuda:{raw_index % device_count}"
    return replace(config, sam=replace(sam, device=worker_device))


def _visible_cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [token.strip() for token in visible.split(",") if token.strip()]
        if devices:
            return len(devices)
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _process_relabel_chunk(payload: tuple[int, list[Path]]) -> dict[str, Any]:
    global _WORKER_ARCHIVES_PROCESSED
    if (
        _WORKER_RELABELER is None
        or _WORKER_INPUT_ROOT is None
        or _WORKER_OUTPUT_ROOT is None
        or _WORKER_MODE is None
    ):
        raise RuntimeError("relabel worker is not initialized")

    chunk_index, relative_paths = payload

    processed = 0
    release_interval = MEMORY_RELEASE_INTERVAL
    for relative_path in relative_paths:
        candidate_path = _WORKER_INPUT_ROOT / relative_path
        save_path = _output_path_for_input(_WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT, candidate_path)
        object_cloud_save_path = (
            _output_path_for_input(_WORKER_INPUT_ROOT, _WORKER_OBJECT_CLOUD_ROOT, candidate_path)
            if _WORKER_OBJECT_CLOUD_ROOT is not None
            else None
        )
        _WORKER_RELABELER._process_archive(
            candidate_path,
            save_path,
            worker_mode=_WORKER_MODE,
            include_object_cloud=_WORKER_INCLUDE_OBJECT_CLOUD,
            object_cloud_save_path=object_cloud_save_path,
        )
        processed += 1
        _WORKER_ARCHIVES_PROCESSED += 1
        _WORKER_RELABELER._backend.release_after_archive(
            worker_mode=_WORKER_MODE,
            release_memory=(_WORKER_ARCHIVES_PROCESSED % release_interval == 0),
        )
    return {
        "chunk_index": chunk_index,
        "processed": processed,
        "chunk_size": len(relative_paths),
    }


def _chunk_relative_paths(
    relative_paths: list[Path],
    chunk_size: int,
    *,
    group_by_scene: bool,
) -> list[list[Path]]:
    if not relative_paths:
        return []

    chunk_size = max(1, int(chunk_size))
    if group_by_scene:
        scene_groups: dict[str, list[Path]] = {}
        for relative_path in relative_paths:
            scene_groups.setdefault(_scene_key(relative_path), []).append(relative_path)
        chunks: list[list[Path]] = []
        for _scene, paths in sorted(scene_groups.items()):
            chunks.extend(
                paths[idx : idx + chunk_size]
                for idx in range(0, len(paths), chunk_size)
            )
        return chunks

    return [
        relative_paths[idx : idx + chunk_size]
        for idx in range(0, len(relative_paths), chunk_size)
    ]


def _scene_key(relative_path: Path) -> str:
    parts = relative_path.parts
    for part in parts:
        if part.startswith("scene_"):
            return part
    return parts[0] if parts else ""



def _count_written_outputs(output_root: Path) -> int:
    """Count archives already on disk, so progress advances within a chunk."""
    try:
        return sum(1 for _ in output_root.glob("**/*.npz"))
    except OSError:
        return 0

def _chunk_size(worker_mode: str) -> int:
    if worker_mode.startswith("features"):
        return FEATURE_CHUNK_SIZE
    return RELABEL_CHUNK_SIZE


def _max_chunks_per_child(worker_mode: str) -> int:
    if worker_mode.startswith("features"):
        return FEATURE_MAX_CHUNKS_PER_CHILD
    return RELABEL_MAX_CHUNKS_PER_CHILD


def _existing_output_is_reusable(
    path: Path,
    worker_mode: str,
    *,
    validate: bool,
    object_cloud_path: Path | None = None,
    include_object_cloud: bool = True,
) -> bool:
    if not path.exists():
        return False
    if object_cloud_path is not None and not object_cloud_path.exists():
        return False
    if not validate:
        return True
    required = {
        "grasp_group_array",
        "base_scores",
        "grasp_widths",
        "grasp_poses",
        "local_cloud",
        "cloud_mask",
        "mu_min",
        "is_collision",
        "is_empty",
        "meta_json",
        "object_assignments",
    }
    if include_object_cloud:
        required.add("object_cloud")
    try:
        with np.load(path, allow_pickle=True) as archive:
            if not required.issubset(set(archive.files)):
                return False
        if object_cloud_path is not None:
            with np.load(object_cloud_path, allow_pickle=True) as archive:
                return "object_cloud" in archive.files
        return True
    except Exception:
        return False


def _output_path_for_input(input_root: Path, output_root: Path, input_path: Path) -> Path:
    relative = input_path.relative_to(input_root)
    if _flatten_camera_dir_enabled():
        parts = relative.parts
        if len(parts) >= 3 and parts[0].startswith("scene_") and parts[1] in {"realsense", "kinect"}:
            relative = Path(parts[0], *parts[2:])
    if relative.suffix == ".npy":
        relative = relative.with_suffix(".npz")
    return output_root / relative


def _flatten_camera_dir_enabled() -> bool:
    raw = os.environ.get("GRARE_FLATTEN_CAMERA_DIR", "0")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _release_python_and_torch_memory(*, release_cuda: bool = False) -> None:
    gc.collect()
    if not release_cuda:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _suppress_external_progress():
    raw = os.environ.get("GRARE_SUPPRESS_GRASPNET_PROGRESS", "1")
    enabled = raw.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
