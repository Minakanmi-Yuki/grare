#!/usr/bin/env python3
"""Precompute frozen Point-MAE object-tier pooled features.

The object tier's Point-MAE backbone is frozen, so its pre-projection pooled
output is a deterministic function of object_cloud (FPS uses a fixed
farthest-point seed and the Transformer weights never change). We can run the
backbone once offline and cache the (N, 2*embed_dim) pooled vector. Training
then skips the 12-layer Transformer and learns only the projection adapter.

The preferred layout is a sidecar tree whose files mirror the relabel archive
relative paths and contain only ``object_pooled`` plus small metadata. This
keeps the large SAM ``object_cloud`` out of the training read path. Source
archives are left unchanged.

Usage:
    grare-precompute-object \
        --archive-root /path/to/local_cloud/train \
        --object-cloud-root /path/to/object_cloud/train \
        --output-root /path/to/object_pooled/train \
        --pmae-ckpt /path/to/point_mae_pretrain.pth \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from grare.utils.runtime import configure_thread_pools

configure_thread_pools()

import numpy as np

from grare.relabeling.archive_io import (
    load_npz_payload,
    normalize_archive_format,
    save_npz_archive,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute frozen Point-MAE object pooled features.")
    p.add_argument("--archive-root", required=True,
        help="Relabel archive root holding scene_*/<camera>/*.npz with object_cloud.")
    p.add_argument("--object-cloud-root", default=None,
        help="Optional sidecar root holding object_cloud files with the same relative layout as --archive-root.")
    p.add_argument("--output-root", required=True,
        help="Sidecar output root. Files mirror --archive-root relative paths and contain object_pooled.")
    p.add_argument("--pmae-ckpt", required=True,
        help="Point-MAE pretrain checkpoint (must match training config).")
    p.add_argument("--pattern", default="**/*.npz")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1,
        help="Split the sorted archive list into N strided, disjoint shards for "
             "multi-process parallelism. Each process picks one --shard.")
    p.add_argument("--shard", type=int, default=0,
        help="This process's shard index in [0, num_shards). Strided slice "
             "archives[shard::num_shards] — disjoint across shards.")
    p.add_argument("--batch-size", type=int, default=4096,
        help="Candidates per backbone forward (independent of training batch).")
    p.add_argument("--object-cloud-points", type=int, default=512)
    p.add_argument("--object-hidden-dim", type=int, default=128)
    p.add_argument("--object-pmae-num-group", type=int, default=32)
    p.add_argument("--object-pmae-group-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true",
        help="Recompute even if object_pooled already present.")
    p.add_argument("--archive-format", choices=("compressed", "stored"), default="compressed")
    p.add_argument("--save-path", default=None)
    return p.parse_args()


def build_encoder(args, device):
    import torch
    from grare.rescoring.model import RescorerConfig, ObjectEncoderPointMAE
    cfg = RescorerConfig(
        object_pmae_ckpt=args.pmae_ckpt,
        object_hidden_dim=args.object_hidden_dim,
        object_cloud_points=args.object_cloud_points,
        object_pmae_num_group=args.object_pmae_num_group,
        object_pmae_group_size=args.object_pmae_group_size,
    )
    enc = ObjectEncoderPointMAE(cfg).to(device).eval()
    return enc


def main() -> int:
    import torch

    args = parse_args()
    os.environ["GRARE_ARCHIVE_FORMAT"] = normalize_archive_format(args.archive_format)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = build_encoder(args, device)

    root = Path(args.archive_root)
    object_cloud_root = Path(args.object_cloud_root) if args.object_cloud_root else None
    output_root = Path(args.output_root)
    archives = sorted(root.glob(args.pattern))
    if args.limit is not None:
        archives = archives[: args.limit]
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            raise SystemExit(f"--shard {args.shard} out of range [0, {args.num_shards})")
        archives = archives[args.shard :: args.num_shards]

    started = time.perf_counter()
    processed = skipped = total_rows = 0
    pending: list[tuple[Path, Path, dict[str, np.ndarray], np.ndarray, int]] = []
    pending_rows = 0

    def flush_pending() -> None:
        """Run one fused Point-MAE batch, then write each archive sidecar.

        Detector archives are usually only a few hundred candidates, so the
        old loop launched one GPU forward per archive and rarely reached the
        configured batch size. Accumulating adjacent archives keeps the frozen
        backbone on large kernels while retaining the same per-archive output
        layout and resume semantics.
        """
        nonlocal pending, pending_rows, processed, total_rows
        if not pending:
            return

        valid_items = [
            item
            for item in pending
            if item[4] > 0 and item[3].ndim == 3 and item[3].shape[1] > 0
        ]
        pooled_parts: list[np.ndarray] = []
        if valid_items:
            merged = np.concatenate([item[3] for item in valid_items], axis=0)
            with torch.inference_mode():
                for start_idx in range(0, len(merged), args.batch_size):
                    chunk = torch.from_numpy(merged[start_idx : start_idx + args.batch_size]).to(
                        device, non_blocking=device.type == "cuda"
                    )
                    pooled_parts.append(enc.forward_pooled(chunk).float().cpu().numpy())
            pooled_merged = np.concatenate(pooled_parts, axis=0)
        else:
            pooled_merged = np.zeros((0, 2 * enc.embed_dim), dtype=np.float32)

        pooled_cursor = 0
        for archive_path, save_path, payload, object_cloud, n in pending:
            if n == 0 or object_cloud.ndim != 3 or object_cloud.shape[1] == 0:
                pooled = np.zeros((n, 2 * enc.embed_dim), dtype=np.float16)
            else:
                pooled = pooled_merged[pooled_cursor : pooled_cursor + n].astype(np.float16)
                pooled_cursor += n
            sidecar_payload = {
                "object_pooled": pooled,
                "source_relative_path": np.array(str(archive_path.relative_to(root)), dtype=object),
            }
            if "meta_json" in payload:
                sidecar_payload["meta_json"] = payload["meta_json"]
            save_npz_archive(save_path, archive_format=args.archive_format, **sidecar_payload)
            processed += 1
            total_rows += n
            if processed % 200 == 0:
                rate = total_rows / max(time.perf_counter() - started, 1e-6)
                print(
                    f"[precompute] {processed} archives, {total_rows} rows, {rate:.0f} rows/s",
                    flush=True,
                )
        pending = []
        pending_rows = 0

    for ap in archives:
        save_ap = output_root / ap.relative_to(root)
        if save_ap.is_file() and not args.overwrite:
            skipped += 1
            continue
        payload = load_npz_payload(ap)
        object_payload = payload
        if object_cloud_root is not None:
            object_path = object_cloud_root / ap.relative_to(root)
            object_payload = load_npz_payload(object_path)
        oc = np.asarray(object_payload["object_cloud"], dtype=np.float32)
        n = int(oc.shape[0])
        pending.append((ap, save_ap, payload, oc, n))
        pending_rows += n
        if pending_rows >= args.batch_size:
            flush_pending()
    flush_pending()

    summary = {
        "stage": "precompute_object_pooled",
        "archive_root": str(root.resolve()),
        "object_cloud_root": None if object_cloud_root is None else str(object_cloud_root.resolve()),
        "output_root": str(output_root.resolve()),
        "num_shards": args.num_shards,
        "shard": args.shard,
        "num_archives": len(archives),
        "processed": processed,
        "skipped": skipped,
        "total_rows": total_rows,
        "pooled_dim": 2 * enc.embed_dim,
        "runtime_sec": time.perf_counter() - started,
        "device": str(device),
    }
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
