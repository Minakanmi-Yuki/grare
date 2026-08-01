#!/usr/bin/env python3
"""Run the GraspNet-Baseline checkpoint on an arbitrary GraspNet split."""
from __future__ import annotations

import argparse
import collections.abc
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRASPNET_BASELINE_ROOT = Path(
    os.environ.get("GRARE_GN_BASELINE_ROOT", str(PROJECT_ROOT / "external" / "graspnet-baseline"))
).expanduser()


def _install_torch_six_shim() -> None:
    """Support upstream GraspNet code on PyTorch versions without torch._six."""
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.container_abcs = collections.abc
    sys.modules["torch._six"] = module


def _graspnet_api_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("GRARE_GRASPNET_API_ROOT")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.extend((PROJECT_ROOT / "graspnetAPI", PROJECT_ROOT.parent / "graspnetAPI"))
    return paths


for path in (
    *[
        candidate
        for candidate in _graspnet_api_paths()
        if (candidate / "graspnetAPI" / "__init__.py").exists()
    ],
    GRASPNET_BASELINE_ROOT / "models",
    GRASPNET_BASELINE_ROOT / "dataset",
    GRASPNET_BASELINE_ROOT / "utils",
    GRASPNET_BASELINE_ROOT / "pointnet2",
    GRASPNET_BASELINE_ROOT / "knn",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

_install_torch_six_shim()

from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnet_dataset import GraspNetDataset, collate_fn  # noqa: E402
from graspnetAPI import GraspGroup  # noqa: E402

from grare.utils.experiment_logging import timestamp, write_json  # noqa: E402


def _voxel_downsample_cloud(scene_points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelize a collision cloud in the DataLoader worker.

    The old path regenerated the RGB-D cloud in the postprocess thread and
    then voxelized it there.  Returning this compact cloud with the sampled
    network input lets the two stages share one frame decode and keeps the
    postprocess pool focused on collision math and file output.
    """
    import open3d as o3d

    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
    scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
    return np.asarray(scene_cloud.points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--split", default="test", choices=("train", "test", "test_seen", "test_similar", "test_novel"))
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--index-shard-count", type=int, default=1)
    parser.add_argument("--index-shard-id", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--profile-latency", action="store_true", help="Record synchronized detector forward+decode latency.")
    parser.add_argument("--latency-warmup-batches", type=int, default=5)
    parser.add_argument("--latency-output", default=None, help="Optional JSON path for detector latency statistics.")
    parser.add_argument("--latency-only", action="store_true", help="Skip prediction saves/postprocess; useful with --profile-latency.")
    parser.add_argument("--save_summary", default=None)
    return parser.parse_args()


_WORKER_BASE_SEED = 0


def my_worker_init_fn(worker_id: int) -> None:
    worker_seed = _WORKER_BASE_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ResumeSubset(Dataset):
    """Load only selected source indices and seed sampling by source index."""

    def __init__(
        self,
        dataset,
        indices: list[int],
        seed: int,
        collision_voxel_size: float | None = None,
        share_collision_cloud: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.seed = int(seed)
        self.collision_voxel_size = collision_voxel_size
        self.share_collision_cloud = bool(share_collision_cloud)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = self.indices[position]
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        sample_seed = (self.seed + int(source_index)) % (2**32 - 1)
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        try:
            if not self.share_collision_cloud:
                return self.dataset[source_index]
            sample = self.dataset.get_data(
                source_index,
                return_raw_cloud_with_sample=True,
            )
            if self.collision_voxel_size is not None:
                sample["_collision_cloud"] = _voxel_downsample_cloud(
                    sample["_collision_cloud"],
                    self.collision_voxel_size,
                )
            return sample
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)


def _collate_dump_batch(batch):
    """Collate tensors normally while preserving variable-size clouds as a list."""
    collision_clouds = [sample.pop("_collision_cloud") for sample in batch]
    collated = collate_fn(batch)
    collated["_collision_clouds"] = collision_clouds
    return collated


def _move_batch_to_device(batch_data: dict, device: torch.device) -> None:
    for key in batch_data:
        if key == "_collision_clouds":
            continue
        value = batch_data[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device, non_blocking=True)
        else:
            batch_data[key] = value.to(device, non_blocking=True)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_latency(values_ms: list[float]) -> dict:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "std_ms": None,
        }
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": round(float(np.mean(arr)), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "std_ms": round(float(np.std(arr)), 4),
    }


def _complete_postprocess(
    futures: set,
    *,
    block: bool,
    progress,
    saved_scenes: set[str],
) -> int:
    if not futures:
        return 0
    done = {future for future in futures if future.done()}
    if block and not done:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
    completed = 0
    for future in done:
        futures.remove(future)
        scene_name = future.result()
        saved_scenes.add(scene_name)
        progress.update(1)
        completed += 1
    return completed


def main() -> int:
    args = parse_args()
    if args.index_shard_count < 1:
        raise SystemExit("--index-shard-count must be >= 1")
    if not 0 <= args.index_shard_id < args.index_shard_count:
        raise SystemExit("--index-shard-id must satisfy 0 <= id < count")
    started_at = timestamp()
    started_perf = time.perf_counter()
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    dataset = GraspNetDataset(
        args.dataset_root,
        valid_obj_idxs=None,
        grasp_labels=None,
        split=args.split,
        camera=args.camera,
        num_points=args.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    scene_names = dataset.scene_list()
    source_indices = list(range(len(dataset)))
    if args.index_shard_count > 1:
        source_indices = [
            data_idx
            for data_idx in source_indices
            if data_idx % args.index_shard_count == args.index_shard_id
        ]

    def target_for_index(data_idx: int) -> tuple[str, Path, Path]:
        scene_name = scene_names[data_idx]
        save_dir = dump_dir / scene_name / args.camera
        save_path = save_dir / f"{data_idx % 256:04d}.npy"
        return scene_name, save_dir, save_path

    pending_indices = list(source_indices)
    existing_indices: list[int] = []
    existing_scenes: set[str] = set()
    if args.skip_existing:
        pending_indices = []
        for data_idx in source_indices:
            scene_name, _, save_path = target_for_index(data_idx)
            if save_path.exists():
                existing_indices.append(data_idx)
                existing_scenes.add(scene_name)
            else:
                pending_indices.append(data_idx)

    supports_shared_collision_cloud = (
        "return_raw_cloud_with_sample"
        in inspect.signature(dataset.get_data).parameters
    )
    supports_pre_downsampled_detector = (
        "downsample" in inspect.signature(ModelFreeCollisionDetector).parameters
    )
    share_collision_cloud = (
        args.collision_thresh > 0
        and supports_shared_collision_cloud
        and supports_pre_downsampled_detector
    )
    if args.collision_thresh > 0 and not share_collision_cloud:
        print(
            "warning: GraspNet-Baseline dump fast-path patch is not applied; "
            "falling back to a second frame read for collision filtering. "
            "Run ./scripts/build_detector_extensions.sh to apply it.",
            flush=True,
        )
    active_dataset = ResumeSubset(
        dataset,
        pending_indices,
        args.seed,
        collision_voxel_size=args.voxel_size if share_collision_cloud else None,
        share_collision_cloud=share_collision_cloud,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.data_workers,
        "worker_init_fn": my_worker_init_fn,
        "collate_fn": _collate_dump_batch if share_collision_cloud else collate_fn,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.data_workers > 0 and not args.no_persistent_workers,
    }
    if args.data_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(active_dataset, **loader_kwargs)

    net = GraspNet(
        input_feature_dim=0,
        num_view=args.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        if args.tf32:
            torch.set_float32_matmul_precision("high")
    net.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    loaded_epoch = int(checkpoint["epoch"])
    print(
        json.dumps(
            {
                "stage": "baseline_dump_setup",
                "detector": "graspnet_baseline",
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "dump_dir": str(dump_dir.resolve()),
                "camera": args.camera,
                "split": args.split,
                "num_dataset_samples": len(dataset),
                "num_samples": len(source_indices),
                "num_pending_samples": len(pending_indices),
                "num_batches": len(dataloader),
                "skipped_existing_files": len(existing_indices),
                "batch_size": args.batch_size,
                "data_workers": args.data_workers,
                "prefetch_factor": args.prefetch_factor if args.data_workers > 0 else None,
                "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
                "persistent_workers": bool(args.data_workers > 0 and not args.no_persistent_workers),
                "postprocess_workers": args.postprocess_workers,
                "shared_collision_cloud": bool(share_collision_cloud),
                "index_shard_count": args.index_shard_count,
                "index_shard_id": args.index_shard_id,
                "collision_thresh": args.collision_thresh,
                "voxel_size": args.voxel_size,
                "seed": args.seed,
                "deterministic": bool(args.deterministic),
                "tf32": bool(args.tf32),
                "cudnn_benchmark": bool(args.cudnn_benchmark),
                "profile_latency": bool(args.profile_latency),
                "latency_warmup_batches": int(args.latency_warmup_batches),
                "latency_only": bool(args.latency_only),
                "loaded_epoch": loaded_epoch,
                "started_at": started_at,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "stage": "baseline_dump_resume_scan",
                "detector": "graspnet_baseline",
                "split": args.split,
                "camera": args.camera,
                "dump_dir": str(dump_dir.resolve()),
                "num_dataset_samples": len(dataset),
                "num_samples": len(source_indices),
                "num_pending_samples": len(pending_indices),
                "num_batches": len(dataloader),
                "skipped_existing_files": len(existing_indices),
                "completed_files": len(existing_indices),
                "saved_scenes": len(existing_scenes),
                "skip_existing": bool(args.skip_existing),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    net.eval()
    batch_interval = 20
    tic = time.time()
    saved_files = 0
    skipped_existing_files = len(existing_indices)
    saved_scenes: set[str] = set(existing_scenes)
    progress_total = len(source_indices)
    if args.max_batches is not None:
        progress_total = min(
            progress_total,
            skipped_existing_files + min(len(pending_indices), args.max_batches * args.batch_size),
        )
    progress = tqdm(
        total=progress_total,
        initial=skipped_existing_files,
        desc=f"graspnet_baseline {args.split} {args.camera}",
        unit="frame",
    )
    progress.set_postfix(saved=saved_files, skipped=skipped_existing_files, scenes=len(saved_scenes), refresh=False)

    pending_cursor = 0
    postprocess_pool = (
        ThreadPoolExecutor(max_workers=args.postprocess_workers)
        if args.postprocess_workers > 0
        else None
    )
    postprocess_futures: set = set()
    max_pending_postprocess = max(args.batch_size, args.postprocess_workers * 4)
    latency_batch_ms: list[float] = []
    latency_frame_ms: list[float] = []

    def save_prediction(
        data_idx: int,
        scene_name: str,
        save_dir: Path,
        save_path: Path,
        preds: np.ndarray,
        collision_cloud: np.ndarray | None = None,
    ) -> str:
        grasp_group = GraspGroup(preds)

        if args.collision_thresh > 0:
            if collision_cloud is None:
                # Backward-compatible fallback for callers that do not use
                # the shared-cloud DataLoader path.
                collision_cloud, _ = dataset.get_data(data_idx, return_raw_cloud=True)
                already_downsampled = False
            else:
                already_downsampled = True
            if supports_pre_downsampled_detector:
                detector = ModelFreeCollisionDetector(
                    collision_cloud,
                    voxel_size=args.voxel_size,
                    downsample=not already_downsampled,
                )
            else:
                detector = ModelFreeCollisionDetector(
                    collision_cloud,
                    voxel_size=args.voxel_size,
                )
            collision_mask = detector.detect(
                grasp_group,
                approach_dist=0.05,
                collision_thresh=args.collision_thresh,
            )
            grasp_group = grasp_group[~collision_mask]

        save_dir.mkdir(parents=True, exist_ok=True)
        grasp_group.save_npy(str(save_path))
        return scene_name

    try:
        for batch_idx, batch_data in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            saved_files += _complete_postprocess(
                postprocess_futures,
                block=False,
                progress=progress,
                saved_scenes=saved_scenes,
            )
            progress.set_postfix(
                saved=saved_files,
                skipped=skipped_existing_files,
                scenes=len(saved_scenes),
                refresh=False,
            )

            point_clouds = batch_data["point_clouds"]
            actual_batch_size = int(point_clouds.shape[0])
            batch_indices = pending_indices[pending_cursor : pending_cursor + actual_batch_size]
            pending_cursor += actual_batch_size
            batch_targets = []
            existing_offsets: set[int] = set()
            for offset, data_idx in enumerate(batch_indices):
                scene_name, save_dir, save_path = target_for_index(data_idx)
                batch_targets.append((data_idx, scene_name, save_dir, save_path))
                if args.skip_existing and save_path.exists():
                    existing_offsets.add(offset)
                    skipped_existing_files += 1
                    saved_scenes.add(scene_name)

            if len(existing_offsets) == actual_batch_size:
                progress.update(actual_batch_size)
                progress.set_postfix(
                    saved=saved_files,
                    skipped=skipped_existing_files,
                    scenes=len(saved_scenes),
                    refresh=False,
                )
                continue

            _move_batch_to_device(batch_data, device)
            if args.profile_latency:
                _sync_if_cuda(device)
                latency_t0 = time.perf_counter()
            with torch.inference_mode():
                end_points = net(batch_data)
                grasp_preds = pred_decode(end_points)
            if args.profile_latency:
                _sync_if_cuda(device)
                elapsed_ms = (time.perf_counter() - latency_t0) * 1000.0
                if batch_idx >= args.latency_warmup_batches:
                    latency_batch_ms.append(elapsed_ms)
                    latency_frame_ms.append(elapsed_ms / max(actual_batch_size, 1))

            if args.latency_only:
                progress.update(actual_batch_size)
                progress.set_postfix(
                    saved=saved_files,
                    skipped=skipped_existing_files,
                    scenes=len(saved_scenes),
                    refresh=False,
                )
                continue

            for offset, preds_tensor in enumerate(grasp_preds):
                if offset in existing_offsets or offset >= len(batch_targets):
                    progress.update(1)
                    continue
                data_idx, scene_name, save_dir, save_path = batch_targets[offset]
                preds = preds_tensor.detach().cpu().numpy()
                collision_cloud = (
                    batch_data["_collision_clouds"][offset]
                    if share_collision_cloud
                    else None
                )
                if postprocess_pool is None:
                    saved_scenes.add(
                        save_prediction(
                            data_idx,
                            scene_name,
                            save_dir,
                            save_path,
                            preds,
                            collision_cloud,
                        )
                    )
                    saved_files += 1
                    progress.update(1)
                else:
                    postprocess_futures.add(
                        postprocess_pool.submit(
                            save_prediction,
                            data_idx,
                            scene_name,
                            save_dir,
                            save_path,
                            preds,
                            collision_cloud,
                        )
                    )
                    while len(postprocess_futures) >= max_pending_postprocess:
                        saved_files += _complete_postprocess(
                            postprocess_futures,
                            block=True,
                            progress=progress,
                            saved_scenes=saved_scenes,
                        )

            saved_files += _complete_postprocess(
                postprocess_futures,
                block=False,
                progress=progress,
                saved_scenes=saved_scenes,
            )
            progress.set_postfix(
                saved=saved_files,
                skipped=skipped_existing_files,
                scenes=len(saved_scenes),
                refresh=False,
            )

            if batch_idx % batch_interval == 0:
                elapsed = max(time.time() - tic, 1e-6)
                print(
                    json.dumps(
                        {
                            "stage": "baseline_dump_progress",
                            "detector": "graspnet_baseline",
                            "split": args.split,
                            "batch_idx": batch_idx,
                            "num_batches": len(dataloader),
                            "saved_files": saved_files,
                            "skipped_existing_files": skipped_existing_files,
                            "completed_files": saved_files + skipped_existing_files,
                            "saved_scenes": len(saved_scenes),
                            "sec_per_batch_window": elapsed / max(batch_interval, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                tic = time.time()
        while postprocess_futures:
            saved_files += _complete_postprocess(
                postprocess_futures,
                block=True,
                progress=progress,
                saved_scenes=saved_scenes,
            )
    finally:
        if postprocess_pool is not None:
            postprocess_pool.shutdown(wait=True)

    progress.close()
    payload = {
        "stage": "baseline_dump",
        "detector": "graspnet_baseline",
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dump_dir": str(dump_dir.resolve()),
        "camera": args.camera,
        "split": args.split,
        "num_dataset_samples": len(dataset),
        "num_samples": len(source_indices),
        "num_batches": len(dataloader),
        "saved_files": saved_files,
        "skipped_existing_files": skipped_existing_files,
        "completed_files": saved_files + skipped_existing_files,
        "saved_scenes": len(saved_scenes),
        "index_shard_count": args.index_shard_count,
        "index_shard_id": args.index_shard_id,
        "loaded_epoch": loaded_epoch,
    }
    if args.profile_latency:
        payload["latency"] = {
            "stage": "detector_forward_decode_latency",
            "detector": "graspnet_baseline",
            "unit": "milliseconds",
            "measured_scope": "torch model forward + pred_decode, synchronized on CUDA",
            "excluded_scope": "DataLoader, host-to-device copy timing before the synchronized block, collision filtering, NMS save, and file I/O",
            "warmup_batches": int(args.latency_warmup_batches),
            "batch_size": int(args.batch_size),
            "latency_only": bool(args.latency_only),
            "batch_ms": _summarize_latency(latency_batch_ms),
            "per_frame_ms": _summarize_latency(latency_frame_ms),
        }
        if args.latency_output:
            write_json(args.latency_output, payload)
    if args.save_summary:
        write_json(args.save_summary, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
