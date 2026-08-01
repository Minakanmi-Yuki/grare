#!/usr/bin/env python3
"""Run the EconomicGrasp checkpoint on an arbitrary GraspNet split."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECONOMICGRASP_ROOT = Path(
    os.environ.get("GRARE_ECONOMICGRASP_ROOT", str(PROJECT_ROOT / "external" / "EconomicGrasp"))
).expanduser()


def _graspnet_api_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("GRARE_GRASPNET_API_ROOT")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.extend((PROJECT_ROOT / "graspnetAPI", PROJECT_ROOT.parent / "graspnetAPI"))
    return paths


def _install_import_paths() -> None:
    for path in (
        *[
            candidate
            for candidate in _graspnet_api_paths()
            if (candidate / "graspnetAPI" / "__init__.py").exists()
        ],
        ECONOMICGRASP_ROOT,
        ECONOMICGRASP_ROOT / "libs",
        ECONOMICGRASP_ROOT / "libs" / "MinkowskiEngine",
        ECONOMICGRASP_ROOT / "libs" / "pointnet2",
        ECONOMICGRASP_ROOT / "libs" / "knn",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--split", default="test", choices=("train", "test", "test_seen", "test_similar", "test_novel"))
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--num_angle", type=int, default=12)
    parser.add_argument("--num_depth", type=int, default=4)
    parser.add_argument("--m_point", type=int, default=1024)
    parser.add_argument("--grasp_max_width", type=float, default=0.1)
    parser.add_argument("--graspness_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--data_workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=32)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--collision_thresh", type=float, default=0.0)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
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


def _load_economicgrasp_modules(args: argparse.Namespace):
    """Import EconomicGrasp after giving its global argparse module safe args."""
    _install_import_paths()
    old_argv = sys.argv[:]
    sys.argv = [
        old_argv[0],
        "--dataset_root",
        args.dataset_root,
        "--camera",
        args.camera,
        "--checkpoint_path",
        args.checkpoint_path,
        "--save_dir",
        args.dump_dir,
        "--test_mode",
        "seen",
        "--num_point",
        str(args.num_point),
        "--num_view",
        str(args.num_view),
        "--num_angle",
        str(args.num_angle),
        "--num_depth",
        str(args.num_depth),
        "--m_point",
        str(args.m_point),
        "--grasp_max_width",
        str(args.grasp_max_width),
        "--graspness_threshold",
        str(args.graspness_threshold),
        "--batch_size",
        str(args.batch_size),
        "--collision_thresh",
        str(args.collision_thresh),
        "--voxel_size",
        str(args.voxel_size),
        "--inference",
    ]
    try:
        from dataset.graspnet_dataset import GraspNetDataset, collate_fn
        from models.economicgrasp import economicgrasp, pred_decode
        from utils.collision_detector import ModelFreeCollisionDetector
        from graspnetAPI import GraspGroup
    finally:
        sys.argv = old_argv
    return GraspNetDataset, collate_fn, economicgrasp, pred_decode, ModelFreeCollisionDetector, GraspGroup


from grare.utils.experiment_logging import timestamp, write_json  # noqa: E402


_WORKER_BASE_SEED = 0


def my_worker_init_fn(worker_id: int) -> None:
    worker_seed = _WORKER_BASE_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ResumeSubset(Dataset):
    """Load only selected source indices and seed sampling by source index."""

    def __init__(self, dataset, indices: list[int], seed: int) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.seed = int(seed)

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
            return self.dataset[source_index]
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)


def _move_batch_to_device(batch_data: dict, device: torch.device) -> None:
    for key in batch_data:
        value = batch_data[key]
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device, non_blocking=True)
        elif "graph" in key:
            for i in range(len(value)):
                value[i] = value[i].to(device, non_blocking=True)
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
        saved_scenes.add(future.result())
        progress.update(1)
        completed += 1
    return completed


def main() -> int:
    args = parse_args()
    (
        GraspNetDataset,
        collate_fn,
        economicgrasp,
        pred_decode,
        ModelFreeCollisionDetector,
        GraspGroup,
    ) = _load_economicgrasp_modules(args)

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
        split=args.split,
        camera=args.camera,
        num_points=args.num_point,
        voxel_size=args.voxel_size,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    scene_names = dataset.scene_list()

    def target_for_index(data_idx: int) -> tuple[str, Path, Path]:
        scene_name = scene_names[data_idx]
        save_dir = dump_dir / scene_name / args.camera
        save_path = save_dir / f"{data_idx % 256:04d}.npy"
        return scene_name, save_dir, save_path

    pending_indices = list(range(len(dataset)))
    existing_indices: list[int] = []
    existing_scenes: set[str] = set()
    if args.skip_existing:
        pending_indices = []
        for data_idx in range(len(dataset)):
            scene_name, _, save_path = target_for_index(data_idx)
            if save_path.exists():
                existing_indices.append(data_idx)
                existing_scenes.add(scene_name)
            else:
                pending_indices.append(data_idx)

    active_dataset = ResumeSubset(dataset, pending_indices, args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.data_workers,
        "worker_init_fn": my_worker_init_fn,
        "collate_fn": collate_fn,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.data_workers > 0 and not args.no_persistent_workers,
    }
    if args.data_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(active_dataset, **loader_kwargs)

    net = economicgrasp(seed_feat_dim=512, is_training=False, voxel_size=args.voxel_size)
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
                "detector": "economicgrasp",
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "dump_dir": str(dump_dir.resolve()),
                "camera": args.camera,
                "split": args.split,
                "num_samples": len(dataset),
                "num_pending_samples": len(pending_indices),
                "num_batches": len(dataloader),
                "skipped_existing_files": len(existing_indices),
                "batch_size": args.batch_size,
                "data_workers": args.data_workers,
                "prefetch_factor": args.prefetch_factor if args.data_workers > 0 else None,
                "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
                "persistent_workers": bool(args.data_workers > 0 and not args.no_persistent_workers),
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
                "detector": "economicgrasp",
                "split": args.split,
                "camera": args.camera,
                "dump_dir": str(dump_dir.resolve()),
                "num_samples": len(dataset),
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
    progress_total = len(dataset)
    if args.max_batches is not None:
        progress_total = min(
            progress_total,
            skipped_existing_files + min(len(pending_indices), args.max_batches * args.batch_size),
        )
    progress = tqdm(
        total=progress_total,
        initial=skipped_existing_files,
        desc=f"economicgrasp {args.split} {args.camera}",
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
    ) -> str:
        grasp_group = GraspGroup(preds)
        if args.collision_thresh > 0:
            cloud, _ = dataset.get_data(data_idx, return_raw_cloud=True)
            detector = ModelFreeCollisionDetector(cloud, voxel_size=args.voxel_size)
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
            progress.set_postfix(saved=saved_files, skipped=skipped_existing_files, scenes=len(saved_scenes), refresh=False)

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
                continue

            for offset, preds_tensor in enumerate(grasp_preds):
                if offset in existing_offsets or offset >= len(batch_targets):
                    progress.update(1)
                    continue
                data_idx, scene_name, save_dir, save_path = batch_targets[offset]
                preds = preds_tensor.detach().cpu().numpy()
                if postprocess_pool is None:
                    saved_scenes.add(save_prediction(data_idx, scene_name, save_dir, save_path, preds))
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
            progress.set_postfix(saved=saved_files, skipped=skipped_existing_files, scenes=len(saved_scenes), refresh=False)

            if batch_idx % batch_interval == 0:
                elapsed = max(time.time() - tic, 1e-6)
                print(
                    json.dumps(
                        {
                            "stage": "baseline_dump_progress",
                            "detector": "economicgrasp",
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
    finally:
        while postprocess_futures:
            saved_files += _complete_postprocess(
                postprocess_futures,
                block=True,
                progress=progress,
                saved_scenes=saved_scenes,
            )
        if postprocess_pool is not None:
            postprocess_pool.shutdown(wait=True)

    progress.close()
    payload = {
        "stage": "baseline_dump",
        "detector": "economicgrasp",
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_sec": time.perf_counter() - started_perf,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dump_dir": str(dump_dir.resolve()),
        "camera": args.camera,
        "split": args.split,
        "num_samples": len(dataset),
        "num_batches": len(dataloader),
        "saved_files": saved_files,
        "skipped_existing_files": skipped_existing_files,
        "completed_files": saved_files + skipped_existing_files,
        "saved_scenes": len(saved_scenes),
        "loaded_epoch": loaded_epoch,
    }
    if args.profile_latency:
        payload["latency"] = {
            "stage": "detector_forward_decode_latency",
            "detector": "economicgrasp",
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
