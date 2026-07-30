from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from grare.relabeling.archive_io import load_meta as _load_meta


@dataclass(frozen=True)
class RescoringSample:
    pose_features: np.ndarray
    base_score: float
    local_cloud: np.ndarray
    cloud_mask: np.ndarray
    mu_min: float
    success_label: float
    is_collision: bool
    is_empty: bool
    meta: dict[str, Any]


class RelabeledCandidateDataset(Dataset):
    _TENSOR_FIELDS = (
        "pose_features",
        "base_score",
        "local_cloud",
        "cloud_mask",
        "mu_min",
        "success_label",
        "is_collision",
        "is_empty",
        "object_cloud",
        "object_assignments",
    )

    def __init__(
        self,
        archives: Iterable[str | Path],
        success_mu_thresh: float = 0.4,
        include_meta: bool = False,
        camera: str | None = None,
        object_pooled_roots: Sequence[str | Path] | None = None,
        input_roots: Sequence[str | Path] | None = None,
        require_object_pooled: bool = False,
        load_object_cloud: bool = True,
        manifest_records: Sequence[dict[str, Any]] | None = None,
        eager_load_workers: int = 1,
        progress_label: str | None = None,
    ) -> None:
        self.archive_paths = [str(Path(path)) for path in archives]
        self.include_meta = include_meta
        self.camera = camera
        self.object_pooled_roots = (
            [Path(path) for path in object_pooled_roots]
            if object_pooled_roots is not None
            else None
        )
        self.input_roots = (
            [Path(path) for path in input_roots]
            if input_roots is not None
            else None
        )
        self.require_object_pooled = bool(require_object_pooled)
        self.load_object_cloud = bool(load_object_cloud)
        self.manifest_records = (
            list(manifest_records) if manifest_records is not None else None
        )
        payload = _load_dataset_payload(
            self.archive_paths,
            success_mu_thresh=success_mu_thresh,
            object_pooled_roots=self.object_pooled_roots,
            input_roots=self.input_roots,
            require_object_pooled=self.require_object_pooled,
            load_object_cloud=self.load_object_cloud,
            manifest_records=self.manifest_records,
            num_workers=eager_load_workers,
            progress_label=progress_label,
        )
        if payload is None:
            raise ValueError("No relabeled entries were found")
        self.pose_features = torch.from_numpy(payload["pose_features"]).contiguous()
        self.base_score = torch.from_numpy(payload["base_score"]).contiguous()
        self.local_cloud = torch.from_numpy(payload["local_cloud"]).contiguous()
        self.cloud_mask = torch.from_numpy(payload["cloud_mask"]).contiguous()
        self.mu_min = torch.from_numpy(payload["mu_min"]).contiguous()
        self.success_label = torch.from_numpy(payload["success_label"]).contiguous()
        self.is_collision = torch.from_numpy(payload["is_collision"]).contiguous()
        self.is_empty = torch.from_numpy(payload["is_empty"]).contiguous()
        self.object_cloud = (
            torch.from_numpy(payload["object_cloud"]).contiguous()
            if "object_cloud" in payload
            else None
        )
        self.object_pooled = (
            torch.from_numpy(payload["object_pooled"]).contiguous()
            if "object_pooled" in payload
            else None
        )
        self.object_assignments = torch.from_numpy(payload["object_assignments"]).contiguous()
        self.archive_indices = torch.from_numpy(payload["archive_indices"])
        self.local_indices = torch.from_numpy(payload["local_indices"])
        self._archive_meta: list[dict[str, Any]] = list(payload.get("archive_metas") or [])

    def __len__(self) -> int:
        return int(self.pose_features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        sample: dict[str, torch.Tensor | dict[str, Any]] = {
            "pose_features": self.pose_features[index],
            "base_score": self.base_score[index],
            "local_cloud": self.local_cloud[index],
            "cloud_mask": self.cloud_mask[index],
            "mu_min": self.mu_min[index],
            "success_label": self.success_label[index],
            "is_collision": self.is_collision[index],
            "is_empty": self.is_empty[index],
            "object_assignments": self.object_assignments[index],
            "archive_index": self.archive_indices[index],
            "local_index": self.local_indices[index],
        }
        if self.object_cloud is not None:
            sample["object_cloud"] = self.object_cloud[index]
        if self.object_pooled is not None:
            sample["object_pooled"] = self.object_pooled[index]
        if self.include_meta:
            archive_idx = int(self.archive_indices[index].item())
            local_idx = int(self.local_indices[index].item())
            sample["meta"] = {
                "archive_path": self.archive_paths[archive_idx],
                "index": local_idx,
            }
        return sample

    @property
    def device(self) -> torch.device:
        return self.pose_features.device

    def tensor_bytes(self) -> int:
        total = 0
        for field in self._tensor_field_names():
            tensor = getattr(self, field)
            total += int(tensor.numel() * tensor.element_size())
        return total

    def cache_tensors_(self, device: str | torch.device) -> None:
        target = torch.device(device)
        for field in self._tensor_field_names():
            tensor = getattr(self, field)
            setattr(self, field, tensor.to(target, non_blocking=target.type == "cuda"))

    def _tensor_field_names(self) -> tuple[str, ...]:
        fields = [field for field in self._TENSOR_FIELDS if getattr(self, field, None) is not None]
        if self.object_pooled is not None:
            fields.append("object_pooled")
        return tuple(fields)


class ArchiveRelabeledCandidateDataset(Dataset):
    """Lazy archive-level dataset for large relabel archives.

    Each item is one relabeled `.npz` archive containing all candidates from a
    detector frame. This avoids eagerly decompressing the full archive tree
    before training starts, and lets DataLoader workers load archives in
    parallel.
    """

    is_lazy_archive_dataset = True

    def __init__(
        self,
        archives: Iterable[str | Path],
        success_mu_thresh: float = 0.4,
        archive_sample_counts: Sequence[int] | None = None,
        manifest_records: Sequence[dict[str, Any]] | None = None,
        camera: str | None = None,
        object_pooled_roots: Sequence[str | Path] | None = None,
        input_roots: Sequence[str | Path] | None = None,
        require_object_pooled: bool = False,
        load_object_cloud: bool = True,
    ) -> None:
        self.archive_paths = [str(Path(path)) for path in archives]
        if not self.archive_paths:
            raise ValueError("No relabeled entries were found")
        self.success_mu_thresh = float(success_mu_thresh)
        self.camera = camera
        self.object_pooled_roots = (
            [Path(path) for path in object_pooled_roots]
            if object_pooled_roots is not None
            else None
        )
        self.input_roots = (
            [Path(path) for path in input_roots]
            if input_roots is not None
            else None
        )
        self.require_object_pooled = bool(require_object_pooled)
        self.load_object_cloud = bool(load_object_cloud)
        self.manifest_records = list(manifest_records) if manifest_records is not None else None
        if self.manifest_records is not None:
            if len(self.manifest_records) != len(self.archive_paths):
                raise ValueError(
                    "manifest_records must have the same length as archives "
                    f"({len(self.manifest_records)} != {len(self.archive_paths)})"
                )
            self.archive_counts = [max(int(record.get("num_grasps", 0)), 0) for record in self.manifest_records]
            self.archive_counts_are_hints = False
        elif archive_sample_counts is None:
            self.archive_counts = [_archive_sample_count(path) for path in self.archive_paths]
            self.archive_counts_are_hints = False
        else:
            self.archive_counts = [max(int(count), 0) for count in archive_sample_counts]
            if len(self.archive_counts) != len(self.archive_paths):
                raise ValueError(
                    "archive_sample_counts must have the same length as archives "
                    f"({len(self.archive_counts)} != {len(self.archive_paths)})"
                )
            self.archive_counts_are_hints = True
        if sum(self.archive_counts) <= 0:
            raise ValueError("No relabeled entries were found")
        self._tensor_bytes_estimate = _estimate_archive_tensor_bytes(
            self.archive_paths,
            self.archive_counts,
            object_pooled_roots=self.object_pooled_roots,
            input_roots=self.input_roots,
            require_object_pooled=self.require_object_pooled,
            load_object_cloud=self.load_object_cloud,
        )

    def __len__(self) -> int:
        return len(self.archive_paths)

    def __getitem__(self, archive_index: int) -> dict[str, torch.Tensor]:
        path = self.archive_paths[int(archive_index)]
        object_pooled_path = self._object_pooled_path(path)
        payload = _load_archive_payload(
            path,
            success_mu_thresh=self.success_mu_thresh,
            object_pooled_path=object_pooled_path,
            skip_object_cloud=_should_skip_object_cloud(
                object_pooled_path,
                require_object_pooled=self.require_object_pooled,
                load_object_cloud=self.load_object_cloud,
            ),
            require_object_pooled=self.require_object_pooled,
        )
        payload.pop("_meta", None)
        n = int(payload["base_score"].shape[0])
        payload["archive_index"] = np.full((n,), int(archive_index), dtype=np.int32)
        payload["local_index"] = np.arange(n, dtype=np.int32)
        out = {key: torch.from_numpy(value).contiguous() for key, value in payload.items()}
        return out

    def _object_pooled_path(self, archive_path: str | Path) -> Path | None:
        return _object_pooled_path_for_archive(
            archive_path,
            input_roots=self.input_roots,
            object_pooled_roots=self.object_pooled_roots,
            require=self.require_object_pooled,
        )

    def __len_samples__(self) -> int:
        return int(sum(self.archive_counts))

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def tensor_bytes(self) -> int:
        return int(self._tensor_bytes_estimate)

    def cache_tensors_(self, device: str | torch.device) -> None:
        # Lazy archives are intentionally loaded per batch by DataLoader workers.
        # There is no full in-memory tensor cache to move.
        return None


class PackedRelabeledCandidateDataset(Dataset):
    """Sample-level dataset backed by `.npy` memmaps.

    The packed format stores the same per-candidate fields as
    RelabeledCandidateDataset, but each field is one contiguous NPY array. When
    several training processes map the same files, the OS can share clean pages
    across processes instead of each process building a private decompressed
    tensor cache.
    """

    is_packed_dataset = True

    _OPTIONAL_FIELDS = ("object_cloud", "object_pooled")
    _REQUIRED_FIELDS = (
        "pose_features",
        "base_score",
        "local_cloud",
        "cloud_mask",
        "mu_min",
        "success_label",
        "is_collision",
        "is_empty",
        "object_assignments",
        "archive_index",
        "local_index",
    )

    def __init__(
        self,
        archives: Iterable[str | Path],
        *,
        packed_root: str | Path,
        input_roots: Sequence[str | Path] | None = None,
        include_meta: bool = False,
        require_object_pooled: bool = False,
        load_object_cloud: bool = True,
        manifest_records: Sequence[dict[str, Any]] | None = None,
        mmap_mode: str = "r",
        success_mu_thresh: float = 0.4,
        **_: Any,
    ) -> None:
        self.packed_root = Path(packed_root)
        self.include_meta = bool(include_meta)
        self.input_roots = [Path(path) for path in input_roots] if input_roots is not None else None
        self.archive_paths = [str(Path(path)) for path in archives]
        del success_mu_thresh
        if not self.archive_paths:
            raise ValueError("No relabeled entries were found")

        index_path = self.packed_root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"packed dataset index not found: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        schema_version = int(self.index.get("schema_version", -1))
        if schema_version != 1:
            raise ValueError(f"{index_path}: unsupported packed schema_version={schema_version!r}")

        arrays_meta = dict(self.index.get("arrays") or {})
        self._arrays: dict[str, np.ndarray] = {}
        for field in self._REQUIRED_FIELDS:
            if field not in arrays_meta:
                raise KeyError(f"{index_path}: missing packed array metadata for {field!r}")
            self._arrays[field] = self._load_array(field, arrays_meta[field], mmap_mode=mmap_mode)
        for field in self._OPTIONAL_FIELDS:
            meta = arrays_meta.get(field)
            if meta is not None:
                self._arrays[field] = self._load_array(field, meta, mmap_mode=mmap_mode)

        if require_object_pooled and "object_pooled" not in self._arrays:
            raise KeyError(f"{self.packed_root}: object_pooled is required but absent")
        if not load_object_cloud:
            self._arrays.pop("object_cloud", None)
            self._arrays.pop("object_pooled", None)

        packed_records = list(self.index.get("archives") or [])
        if not packed_records:
            raise ValueError(f"{index_path}: no archive records")
        selected = self._select_archive_records(packed_records)
        if not selected:
            raise ValueError("No relabeled entries were found")
        self._records = selected
        self.manifest_records = (
            list(manifest_records)
            if manifest_records is not None
            else [dict(record.get("manifest_record") or {}) for record in selected]
        )
        self.archive_counts = [int(record["num_samples"]) for record in selected]
        self._record_starts = np.asarray(
            [int(record["start"]) for record in selected],
            dtype=np.int64,
        )
        self._record_offsets = np.zeros((len(selected) + 1,), dtype=np.int64)
        self._record_offsets[1:] = np.cumsum(np.asarray(self.archive_counts, dtype=np.int64))
        self._length = int(self._record_offsets[-1])
        if self._length <= 0:
            raise ValueError("No relabeled entries were found")
        self._tensor_bytes = self._estimate_selected_bytes()

    def __len__(self) -> int:
        return self._length

    def __len_samples__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        packed_index, archive_pos, local_index = self._packed_index(index)
        sample: dict[str, torch.Tensor | dict[str, Any]] = {
            "pose_features": self._tensor_at("pose_features", packed_index),
            "base_score": self._tensor_at("base_score", packed_index),
            "local_cloud": self._tensor_at("local_cloud", packed_index),
            "cloud_mask": self._tensor_at("cloud_mask", packed_index),
            "mu_min": self._tensor_at("mu_min", packed_index),
            "success_label": self._tensor_at("success_label", packed_index),
            "is_collision": self._tensor_at("is_collision", packed_index),
            "is_empty": self._tensor_at("is_empty", packed_index),
            "object_assignments": self._tensor_at("object_assignments", packed_index),
            "archive_index": torch.tensor(archive_pos, dtype=torch.int32),
            "local_index": self._tensor_at("local_index", packed_index),
        }
        if "object_cloud" in self._arrays:
            sample["object_cloud"] = self._tensor_at("object_cloud", packed_index)
        if "object_pooled" in self._arrays:
            sample["object_pooled"] = self._tensor_at("object_pooled", packed_index)
        if self.include_meta:
            sample["meta"] = {
                "archive_path": self.archive_paths[archive_pos],
                "index": local_index,
            }
        return sample

    def __getitems__(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        """Return an already-collated batch for PyTorch's batched fetch path.

        DataLoader calls ``__getitems__`` once per batch when available. For the
        packed mmap format this avoids thousands of Python ``__getitem__`` calls
        per batch and lets contiguous archive chunks be copied as vectorized
        NumPy slices.
        """
        if isinstance(indices, torch.Tensor):
            index_array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim == 0:
            return self[int(index_array)]  # type: ignore[return-value]

        packed_indices, archive_pos, local_offsets = self._packed_indices(index_array.reshape(-1))
        batch: dict[str, torch.Tensor | list[dict[str, Any]]] = {
            "pose_features": self._tensor_batch("pose_features", packed_indices),
            "base_score": self._tensor_batch("base_score", packed_indices),
            "local_cloud": self._tensor_batch("local_cloud", packed_indices),
            "cloud_mask": self._tensor_batch("cloud_mask", packed_indices),
            "mu_min": self._tensor_batch("mu_min", packed_indices),
            "success_label": self._tensor_batch("success_label", packed_indices),
            "is_collision": self._tensor_batch("is_collision", packed_indices),
            "is_empty": self._tensor_batch("is_empty", packed_indices),
            "object_assignments": self._tensor_batch("object_assignments", packed_indices),
            "archive_index": torch.from_numpy(archive_pos.astype(np.int32, copy=False)),
            "local_index": self._tensor_batch("local_index", packed_indices),
        }
        if "object_cloud" in self._arrays:
            batch["object_cloud"] = self._tensor_batch("object_cloud", packed_indices)
        if "object_pooled" in self._arrays:
            batch["object_pooled"] = self._tensor_batch("object_pooled", packed_indices)
        if self.include_meta:
            batch["meta"] = [
                {
                    "archive_path": self.archive_paths[int(pos)],
                    "index": int(local_idx),
                }
                for pos, local_idx in zip(archive_pos, local_offsets, strict=True)
            ]
        return batch

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def tensor_bytes(self) -> int:
        return int(self._tensor_bytes)

    def cache_tensors_(self, device: str | torch.device) -> None:
        # Packed datasets are deliberately mmap-backed. Moving the full dataset
        # into a private tensor cache defeats the sharing property.
        return None

    def _load_array(self, field: str, meta: dict[str, Any], *, mmap_mode: str) -> np.ndarray:
        rel_path = meta.get("path") or f"arrays/{field}.npy"
        path = self.packed_root / str(rel_path)
        if not path.is_file():
            raise FileNotFoundError(f"packed array not found: {path}")
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        expected_shape = tuple(int(v) for v in meta.get("shape", ()))
        if expected_shape and tuple(array.shape) != expected_shape:
            raise ValueError(f"{path}: shape={array.shape} expected={expected_shape}")
        expected_dtype = meta.get("dtype")
        if expected_dtype is not None and np.dtype(array.dtype) != np.dtype(str(expected_dtype)):
            raise ValueError(f"{path}: dtype={array.dtype} expected={expected_dtype}")
        return array

    def _select_archive_records(self, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        by_abs: dict[str, dict[str, Any]] = {}
        by_rel: dict[str, list[dict[str, Any]]] = {}
        by_root_rel: dict[tuple[int, str], dict[str, Any]] = {}
        for record in records:
            abs_path = record.get("absolute_path")
            if abs_path:
                by_abs[str(Path(abs_path).resolve())] = record
            rel_path = str(record.get("relative_path") or "")
            if rel_path:
                by_rel.setdefault(rel_path, []).append(record)
                by_root_rel[(int(record.get("input_root_index", 0)), rel_path)] = record

        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for archive_path in self.archive_paths:
            path = Path(archive_path)
            record = by_abs.get(str(path.resolve()))
            if record is None:
                key = self._relative_key(path)
                if key is not None:
                    record = by_root_rel.get(key)
                    if record is None:
                        rel_matches = by_rel.get(key[1], [])
                        if len(rel_matches) == 1:
                            record = rel_matches[0]
            if record is None:
                missing.append(str(path))
            else:
                selected.append(record)
        if missing:
            raise KeyError(
                f"{self.packed_root}: {len(missing)} requested archives are not in the packed index; "
                f"first missing: {missing[0]}"
            )
        return selected

    def _relative_key(self, path: Path) -> tuple[int, str] | None:
        if self.input_roots is None:
            return None
        for root_idx, root in enumerate(self.input_roots):
            try:
                rel = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            return root_idx, rel.as_posix()
        return None

    def _packed_index(self, index: int) -> tuple[int, int, int]:
        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(index)
        record_pos = int(np.searchsorted(self._record_offsets, idx, side="right") - 1)
        record = self._records[record_pos]
        offset = idx - int(self._record_offsets[record_pos])
        packed_index = int(record["start"]) + int(offset)
        local_index = int(offset)
        return packed_index, record_pos, local_index

    def _packed_indices(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1).copy()
        idx[idx < 0] += self._length
        if idx.size == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )
        invalid = (idx < 0) | (idx >= self._length)
        if bool(np.any(invalid)):
            raise IndexError(int(idx[int(np.flatnonzero(invalid)[0])]))
        archive_pos = np.searchsorted(self._record_offsets, idx, side="right") - 1
        local_offsets = idx - self._record_offsets[archive_pos]
        packed_indices = self._record_starts[archive_pos] + local_offsets
        return (
            packed_indices.astype(np.int64, copy=False),
            archive_pos.astype(np.int64, copy=False),
            local_offsets.astype(np.int64, copy=False),
        )

    def _tensor_at(self, field: str, packed_index: int) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            return torch.from_numpy(np.asarray(self._arrays[field][packed_index]))

    def _tensor_batch(
        self,
        field: str,
        packed_indices: np.ndarray,
    ) -> torch.Tensor:
        data = _read_indexed_runs(self._arrays[field], packed_indices)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            return torch.from_numpy(data)

    def _estimate_selected_bytes(self) -> int:
        if not self._arrays:
            return 0
        total = 0
        for array in self._arrays.values():
            if array.shape[:1] != (int(self.index.get("num_samples", array.shape[0])),):
                continue
            per_sample = int(array.dtype.itemsize)
            for dim in array.shape[1:]:
                per_sample *= int(dim)
            total += per_sample * self._length
        return int(total)


def _read_indexed_runs(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        data = array[idx]
    else:
        breakpoints = np.flatnonzero(np.diff(idx) != 1) + 1
        if breakpoints.size == 0:
            start = int(idx[0])
            data = array[start : start + int(idx.size)]
        else:
            pieces = []
            start_pos = 0
            for stop_pos in [*breakpoints.tolist(), int(idx.size)]:
                start = int(idx[start_pos])
                stop = int(idx[stop_pos - 1]) + 1
                pieces.append(array[start:stop])
                start_pos = stop_pos
            data = np.concatenate(pieces, axis=0)
    data = np.asarray(data)
    if not data.flags.c_contiguous or not data.flags.writeable:
        data = np.array(data, copy=True)
    return data


def collate_archive_batches(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty archive batch")
    keys = (
        "pose_features",
        "base_score",
        "mu_min",
        "success_label",
        "is_collision",
        "is_empty",
        "object_assignments",
        "archive_index",
        "local_index",
    )
    out = {key: torch.cat([item[key] for item in batch], dim=0) for key in keys}
    out["local_cloud"] = _cat_padded_3d([item["local_cloud"] for item in batch])
    out["cloud_mask"] = _cat_padded_2d([item["cloud_mask"] for item in batch])
    if "object_cloud" in batch[0]:
        out["object_cloud"] = _cat_padded_3d([item["object_cloud"] for item in batch])
    if "object_pooled" in batch[0]:
        # Per-candidate cached pooled feature (N, 2*embed_dim) — concat like
        # pose_features. Only present when every archive carries the cache.
        out["object_pooled"] = torch.cat([item["object_pooled"] for item in batch], dim=0)
    return out


def collate_rescoring_batch(
    batch: list[dict[str, torch.Tensor | dict[str, Any]]] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    local_cloud = _pad_local_cloud([item["local_cloud"] for item in batch])  # type: ignore[arg-type]
    cloud_mask = _pad_cloud_mask([item["cloud_mask"] for item in batch])  # type: ignore[arg-type]
    out: dict[str, Any] = {
        "pose_features": torch.stack([item["pose_features"] for item in batch]),  # type: ignore[index]
        "base_score": torch.stack([item["base_score"] for item in batch]),  # type: ignore[index]
        "local_cloud": local_cloud,
        "cloud_mask": cloud_mask,
        "mu_min": torch.stack([item["mu_min"] for item in batch]),  # type: ignore[index]
        "success_label": torch.stack([item["success_label"] for item in batch]),  # type: ignore[index]
        "is_collision": torch.stack([item["is_collision"] for item in batch]),  # type: ignore[index]
        "is_empty": torch.stack([item["is_empty"] for item in batch]),  # type: ignore[index]
        "object_assignments": torch.stack([item["object_assignments"] for item in batch]),  # type: ignore[index]
        "archive_index": torch.stack([item["archive_index"] for item in batch]),  # type: ignore[index]
        "local_index": torch.stack([item["local_index"] for item in batch]),  # type: ignore[index]
        "meta": [item.get("meta") for item in batch],
    }
    if "object_cloud" in batch[0]:
        out["object_cloud"] = _pad_local_cloud([item["object_cloud"] for item in batch])  # type: ignore[arg-type]
    if "object_pooled" in batch[0]:
        out["object_pooled"] = torch.stack([item["object_pooled"] for item in batch])  # type: ignore[index]
    return out


def build_manifest(
    input_root: str | Path,
    save_path: str | Path,
    pattern: str = "**/*.npz",
) -> Path:
    input_root = Path(input_root)
    records = sorted(str(path) for path in input_root.glob(pattern))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump({"archives": records}, f, indent=2, ensure_ascii=False)
    return save_path

def _load_dataset_payload(
    archives: Iterable[str | Path],
    success_mu_thresh: float,
    *,
    object_pooled_roots: Sequence[Path] | None = None,
    input_roots: Sequence[Path] | None = None,
    require_object_pooled: bool = False,
    load_object_cloud: bool = True,
    manifest_records: Sequence[dict[str, Any]] | None = None,
    num_workers: int = 1,
    progress_label: str | None = None,
) -> dict[str, np.ndarray] | None:
    archive_paths = [str(Path(path)) for path in archives]
    if manifest_records is not None and len(manifest_records) != len(archive_paths):
        raise ValueError(
            "manifest_records must have the same length as archives "
            f"({len(manifest_records)} != {len(archive_paths)})"
        )
    counts = _archive_sample_counts_for_eager(
        archive_paths,
        manifest_records=manifest_records,
        num_workers=num_workers,
    )
    total_samples = int(sum(counts))
    if total_samples <= 0:
        return None
    non_empty_archives_total = int(sum(1 for count in counts if count > 0))

    offsets = np.zeros((len(counts) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    first_archive_idx = next(idx for idx, count in enumerate(counts) if count > 0)
    _, first_payload = _load_eager_archive_payload(
        first_archive_idx,
        archive_paths[first_archive_idx],
        success_mu_thresh=success_mu_thresh,
        object_pooled_roots=object_pooled_roots,
        input_roots=input_roots,
        require_object_pooled=require_object_pooled,
        load_object_cloud=load_object_cloud,
    )
    first_n = int(first_payload["base_score"].shape[0])
    if first_n != int(counts[first_archive_idx]):
        raise ValueError(
            f"{archive_paths[first_archive_idx]}: manifest/count hint says "
            f"{counts[first_archive_idx]} samples but archive contains {first_n}"
        )
    out = _allocate_eager_payload(total_samples, first_payload)
    archive_metas: list[dict[str, Any]] = [{} for _ in archive_paths]
    loaded_archives = 0
    loaded_samples = 0
    last_log = time.perf_counter()

    def _write_payload(archive_idx: int, payload: dict[str, np.ndarray]) -> None:
        nonlocal out, loaded_archives, loaded_samples, last_log
        expected = int(counts[archive_idx])
        n = int(payload["base_score"].shape[0])
        if n != expected:
            raise ValueError(
                f"{archive_paths[archive_idx]}: manifest/count hint says "
                f"{expected} samples but archive contains {n}"
            )
        if n <= 0:
            return
        start = int(offsets[archive_idx])
        end = start + n
        out = _ensure_eager_capacity(out, payload)
        _copy_payload_slice(out, payload, archive_idx=archive_idx, start=start, end=end)
        archive_metas[archive_idx] = payload.get("_meta") or {}
        loaded_archives += 1
        loaded_samples += n
        now = time.perf_counter()
        should_log = (
            loaded_archives == 1
            or loaded_archives == non_empty_archives_total
            or loaded_archives % 1000 == 0
            or now - last_log >= 30.0
        )
        if should_log:
            print(
                json.dumps(
                    {
                        "stage": "eager_load_progress",
                        "label": progress_label,
                        "archives": loaded_archives,
                        "archives_total": non_empty_archives_total,
                        "samples": loaded_samples,
                        "samples_total": total_samples,
                        "workers": max(1, int(num_workers)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_log = now

    _write_payload(first_archive_idx, first_payload)
    remaining = [
        (idx, path)
        for idx, path in enumerate(archive_paths)
        if idx != first_archive_idx and counts[idx] > 0
    ]
    workers = max(1, int(num_workers))
    if workers <= 1 or not remaining:
        for archive_idx, archive_path in remaining:
            loaded_archive_idx, payload = _load_eager_archive_payload(
                archive_idx,
                archive_path,
                success_mu_thresh=success_mu_thresh,
                object_pooled_roots=object_pooled_roots,
                input_roots=input_roots,
                require_object_pooled=require_object_pooled,
                load_object_cloud=load_object_cloud,
            )
            _write_payload(loaded_archive_idx, payload)
    else:
        max_pending = max(1, workers * 2)
        task_iter = iter(remaining)
        pending: set[Future[tuple[int, dict[str, np.ndarray]]]] = set()

        def _submit_until_full(executor: ThreadPoolExecutor) -> None:
            while len(pending) < max_pending:
                try:
                    archive_idx, archive_path = next(task_iter)
                except StopIteration:
                    break
                pending.add(
                    executor.submit(
                        _load_eager_archive_payload,
                        archive_idx,
                        archive_path,
                        success_mu_thresh=success_mu_thresh,
                        object_pooled_roots=object_pooled_roots,
                        input_roots=input_roots,
                        require_object_pooled=require_object_pooled,
                        load_object_cloud=load_object_cloud,
                    )
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            _submit_until_full(executor)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    archive_idx, payload = future.result()
                    _write_payload(archive_idx, payload)
                _submit_until_full(executor)

    out["archive_metas"] = archive_metas
    return out


def _archive_sample_counts_for_eager(
    archive_paths: Sequence[str],
    *,
    manifest_records: Sequence[dict[str, Any]] | None,
    num_workers: int,
) -> list[int]:
    if manifest_records is not None:
        return [max(int(record.get("num_grasps", 0)), 0) for record in manifest_records]
    workers = max(1, int(num_workers))
    if workers <= 1 or len(archive_paths) <= 1:
        return [_archive_sample_count(path) for path in archive_paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_archive_sample_count, archive_paths))


def _load_eager_archive_payload(
    archive_idx: int,
    archive_path: str | Path,
    *,
    success_mu_thresh: float,
    object_pooled_roots: Sequence[Path] | None,
    input_roots: Sequence[Path] | None,
    require_object_pooled: bool,
    load_object_cloud: bool,
) -> tuple[int, dict[str, np.ndarray]]:
    object_pooled_path = _object_pooled_path_for_archive(
        archive_path,
        input_roots=input_roots,
        object_pooled_roots=object_pooled_roots,
        require=require_object_pooled,
    )
    payload = _load_archive_payload(
        archive_path,
        success_mu_thresh=success_mu_thresh,
        object_pooled_path=object_pooled_path,
        skip_object_cloud=_should_skip_object_cloud(
            object_pooled_path,
            require_object_pooled=require_object_pooled,
            load_object_cloud=load_object_cloud,
        ),
        require_object_pooled=require_object_pooled,
    )
    return archive_idx, payload


def _allocate_eager_payload(
    total_samples: int,
    sample: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "pose_features": np.empty((total_samples, sample["pose_features"].shape[1]), dtype=np.float32),
        "base_score": np.empty((total_samples,), dtype=np.float32),
        "local_cloud": np.zeros((total_samples, sample["local_cloud"].shape[1], 3), dtype=np.float32),
        "cloud_mask": np.zeros((total_samples, sample["cloud_mask"].shape[1]), dtype=np.bool_),
        "mu_min": np.empty((total_samples,), dtype=np.float32),
        "success_label": np.empty((total_samples,), dtype=np.float32),
        "is_collision": np.empty((total_samples,), dtype=np.float32),
        "is_empty": np.empty((total_samples,), dtype=np.float32),
        "object_assignments": np.empty((total_samples,), dtype=np.int64),
        "archive_indices": np.empty((total_samples,), dtype=np.int32),
        "local_indices": np.empty((total_samples,), dtype=np.int32),
    }
    if "object_cloud" in sample:
        out["object_cloud"] = np.zeros((total_samples, sample["object_cloud"].shape[1], 3), dtype=np.float32)
    if "object_pooled" in sample:
        out["object_pooled"] = np.empty(
            (total_samples, sample["object_pooled"].shape[1]),
            dtype=sample["object_pooled"].dtype,
        )
    return out


def _ensure_eager_capacity(
    out: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if payload["local_cloud"].shape[1] > out["local_cloud"].shape[1]:
        out["local_cloud"] = _grow_padded_3d(out["local_cloud"], payload["local_cloud"].shape[1])
    if payload["cloud_mask"].shape[1] > out["cloud_mask"].shape[1]:
        out["cloud_mask"] = _grow_padded_2d(out["cloud_mask"], payload["cloud_mask"].shape[1])
    if "object_cloud" in payload:
        if "object_cloud" not in out:
            out["object_cloud"] = np.zeros(
                (out["base_score"].shape[0], payload["object_cloud"].shape[1], 3),
                dtype=np.float32,
            )
        elif payload["object_cloud"].shape[1] > out["object_cloud"].shape[1]:
            out["object_cloud"] = _grow_padded_3d(out["object_cloud"], payload["object_cloud"].shape[1])
    if "object_pooled" in payload:
        if "object_pooled" not in out:
            out["object_pooled"] = np.empty(
                (out["base_score"].shape[0], payload["object_pooled"].shape[1]),
                dtype=payload["object_pooled"].dtype,
            )
        elif payload["object_pooled"].shape[1] != out["object_pooled"].shape[1]:
            raise ValueError(
                "object_pooled feature dimension changed within one dataset "
                f"({payload['object_pooled'].shape[1]} != {out['object_pooled'].shape[1]})"
            )
    return out


def _copy_payload_slice(
    out: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    *,
    archive_idx: int,
    start: int,
    end: int,
) -> None:
    out["pose_features"][start:end] = payload["pose_features"]
    out["base_score"][start:end] = payload["base_score"]
    local_points = int(payload["local_cloud"].shape[1])
    out["local_cloud"][start:end, :local_points, :] = payload["local_cloud"]
    mask_points = int(payload["cloud_mask"].shape[1])
    out["cloud_mask"][start:end, :mask_points] = payload["cloud_mask"]
    out["mu_min"][start:end] = payload["mu_min"]
    out["success_label"][start:end] = payload["success_label"]
    out["is_collision"][start:end] = payload["is_collision"]
    out["is_empty"][start:end] = payload["is_empty"]
    out["object_assignments"][start:end] = payload["object_assignments"]
    if "object_cloud" in payload:
        object_points = int(payload["object_cloud"].shape[1])
        out["object_cloud"][start:end, :object_points, :] = payload["object_cloud"]
    if "object_pooled" in payload:
        out["object_pooled"][start:end] = payload["object_pooled"]
    out["archive_indices"][start:end] = np.int32(archive_idx)
    out["local_indices"][start:end] = np.arange(end - start, dtype=np.int32)


def _grow_padded_3d(array: np.ndarray, points: int) -> np.ndarray:
    grown = np.zeros((array.shape[0], int(points), array.shape[2]), dtype=array.dtype)
    grown[:, : array.shape[1], :] = array
    return grown


def _grow_padded_2d(array: np.ndarray, points: int) -> np.ndarray:
    grown = np.zeros((array.shape[0], int(points)), dtype=array.dtype)
    grown[:, : array.shape[1]] = array
    return grown


def _load_archive_payload(
    archive_path: str | Path,
    *,
    success_mu_thresh: float,
    object_pooled_path: str | Path | None = None,
    skip_object_cloud: bool = False,
    require_object_pooled: bool = False,
) -> dict[str, np.ndarray]:
    with np.load(Path(archive_path), allow_pickle=True) as archive:
        grasp_poses = archive["grasp_poses"].astype(np.float32, copy=False)
        base_scores = archive["base_scores"].astype(np.float32, copy=False)
        local_cloud = archive["local_cloud"].astype(np.float32, copy=False)
        mu_min = archive["mu_min"].astype(np.float32, copy=False)
        is_collision = archive["is_collision"].astype(np.float32, copy=False)
        is_empty = archive["is_empty"].astype(np.float32, copy=False)
        widths = archive["grasp_widths"].astype(np.float32, copy=False)
        if "cloud_mask" in archive.files:
            cloud_mask = archive["cloud_mask"].astype(np.bool_, copy=False)
        else:
            cloud_mask = np.any(local_cloud != 0, axis=-1)
        object_assignments = archive["object_assignments"].astype(np.int64, copy=False)
        object_cloud = None
        if not skip_object_cloud:
            object_cloud = archive["object_cloud"].astype(np.float32, copy=False)
        object_pooled = _load_object_pooled(
            archive,
            object_pooled_path=object_pooled_path,
            require=require_object_pooled,
        )
        meta = _load_meta(archive)

    payload = {
        "pose_features": _build_pose_features(grasp_poses, widths, base_scores),
        "base_score": base_scores,
        "local_cloud": local_cloud,
        "cloud_mask": cloud_mask,
        "mu_min": mu_min,
        "success_label": (np.isfinite(mu_min) & (mu_min <= success_mu_thresh)).astype(np.float32, copy=False),
        "is_collision": is_collision,
        "is_empty": is_empty,
        "object_assignments": object_assignments,
        "_meta": meta,
    }
    if object_cloud is not None:
        payload["object_cloud"] = object_cloud
    if object_pooled is not None:
        payload["object_pooled"] = object_pooled
    return payload


def _load_object_pooled(
    archive: np.lib.npyio.NpzFile,
    *,
    object_pooled_path: str | Path | None,
    require: bool,
) -> np.ndarray | None:
    if object_pooled_path is not None:
        path = Path(object_pooled_path)
        if path.is_file():
            with np.load(path, allow_pickle=True) as pooled_archive:
                if "object_pooled" not in pooled_archive.files:
                    raise KeyError(f"{path} does not contain object_pooled")
                return np.asarray(pooled_archive["object_pooled"])
        if require:
            raise FileNotFoundError(f"missing object_pooled sidecar: {path}")
    if "object_pooled" in archive.files:
        return np.asarray(archive["object_pooled"])
    if require:
        raise KeyError("object_pooled is required but absent")
    return None


def _archive_sample_count(archive_path: str | Path) -> int:
    with np.load(Path(archive_path), allow_pickle=True) as archive:
        return int(len(archive["base_scores"]))


def _estimate_archive_tensor_bytes(
    archive_paths: list[str],
    archive_counts: list[int],
    *,
    object_pooled_roots: Sequence[Path] | None = None,
    input_roots: Sequence[Path] | None = None,
    require_object_pooled: bool = False,
    load_object_cloud: bool = True,
) -> int:
    for path, count in zip(archive_paths, archive_counts):
        if count <= 0:
            continue
        object_pooled_path = _object_pooled_path_for_archive(
            path,
            input_roots=input_roots,
            object_pooled_roots=object_pooled_roots,
            require=require_object_pooled,
        )
        payload = _load_archive_payload(
            path,
            success_mu_thresh=0.4,
            object_pooled_path=object_pooled_path,
            skip_object_cloud=_should_skip_object_cloud(
                object_pooled_path,
                require_object_pooled=require_object_pooled,
                load_object_cloud=load_object_cloud,
            ),
            require_object_pooled=require_object_pooled,
        )
        per_archive = 0
        for key, value in payload.items():
            if key == "_meta" or not isinstance(value, np.ndarray):
                continue
            per_archive += int(value.nbytes)
        per_archive += int(np.empty((count,), dtype=np.int32).nbytes) * 2
        return int(round((per_archive / max(count, 1)) * sum(archive_counts)))
    return 0


def _object_pooled_path_for_archive(
    archive_path: str | Path,
    *,
    input_roots: Sequence[Path] | None,
    object_pooled_roots: Sequence[Path] | None,
    require: bool,
) -> Path | None:
    if not object_pooled_roots:
        return None
    if not input_roots:
        raise ValueError("input_roots are required when object_pooled_roots are used")
    path = Path(archive_path)
    for input_root, pooled_root in zip(input_roots, object_pooled_roots):
        try:
            rel = path.relative_to(input_root)
        except ValueError:
            continue
        return pooled_root / rel
    if require:
        raise FileNotFoundError(f"no object pooled root matches archive path {path}")
    return None


def _should_skip_object_cloud(
    object_pooled_path: Path | None,
    *,
    require_object_pooled: bool,
    load_object_cloud: bool = True,
) -> bool:
    if not load_object_cloud:
        return True
    if object_pooled_path is None:
        return False
    return bool(require_object_pooled or object_pooled_path.is_file())


def _build_pose_features(grasp_poses: np.ndarray, widths: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    rotations = grasp_poses[:, :3, :3].reshape(len(grasp_poses), -1)
    translations = grasp_poses[:, :3, 3]
    return np.concatenate(
        [
            rotations.astype(np.float32, copy=False),
            translations.astype(np.float32, copy=False),
            widths.reshape(-1, 1).astype(np.float32, copy=False),
            base_scores.reshape(-1, 1).astype(np.float32, copy=False),
        ],
        axis=1,
    )


def _concat_local_cloud_parts(parts: list[np.ndarray]) -> np.ndarray:
    max_points = max((part.shape[1] for part in parts), default=0)
    if max_points == 0:
        total = sum(part.shape[0] for part in parts)
        return np.zeros((total, 0, 3), dtype=np.float32)

    padded_parts = []
    for part in parts:
        if part.shape[1] == max_points:
            padded_parts.append(part)
            continue
        padded = np.zeros((part.shape[0], max_points, 3), dtype=np.float32)
        padded[:, : part.shape[1]] = part
        padded_parts.append(padded)
    return np.concatenate(padded_parts, axis=0).astype(np.float32, copy=False)


def _concat_cloud_mask_parts(parts: list[np.ndarray]) -> np.ndarray:
    max_points = max((part.shape[1] for part in parts), default=0)
    total = sum(part.shape[0] for part in parts)
    if max_points == 0:
        return np.zeros((total, 0), dtype=np.bool_)
    padded_parts = []
    for part in parts:
        if part.shape[1] == max_points:
            padded_parts.append(part)
            continue
        padded = np.zeros((part.shape[0], max_points), dtype=np.bool_)
        padded[:, : part.shape[1]] = part
        padded_parts.append(padded)
    return np.concatenate(padded_parts, axis=0).astype(np.bool_, copy=False)


def _pad_local_cloud(clouds: list[torch.Tensor]) -> torch.Tensor:
    max_points = max((cloud.shape[0] for cloud in clouds), default=0)
    if max_points == 0:
        return torch.zeros((len(clouds), 0, 3), dtype=torch.float32)
    padded = []
    for cloud in clouds:
        if cloud.shape[0] == max_points:
            padded.append(cloud)
            continue
        pad = torch.zeros((max_points - cloud.shape[0], 3), dtype=cloud.dtype)
        padded.append(torch.cat([cloud, pad], dim=0))
    return torch.stack(padded, dim=0)


def _pad_cloud_mask(masks: list[torch.Tensor]) -> torch.Tensor:
    max_points = max((m.shape[0] for m in masks), default=0)
    if max_points == 0:
        return torch.zeros((len(masks), 0), dtype=torch.bool)
    padded = []
    for m in masks:
        if m.shape[0] == max_points:
            padded.append(m)
            continue
        pad = torch.zeros((max_points - m.shape[0],), dtype=torch.bool)
        padded.append(torch.cat([m, pad], dim=0))
    return torch.stack(padded, dim=0)


def _cat_padded_3d(parts: list[torch.Tensor]) -> torch.Tensor:
    max_points = max((int(part.shape[1]) for part in parts), default=0)
    feat_dim = max((int(part.shape[2]) for part in parts), default=0)
    padded = []
    for part in parts:
        if tuple(part.shape[1:]) == (max_points, feat_dim):
            padded.append(part)
            continue
        out = torch.zeros((part.shape[0], max_points, feat_dim), dtype=part.dtype)
        out[:, : part.shape[1], : part.shape[2]] = part
        padded.append(out)
    return torch.cat(padded, dim=0)


def _cat_padded_2d(parts: list[torch.Tensor]) -> torch.Tensor:
    max_points = max((int(part.shape[1]) for part in parts), default=0)
    padded = []
    for part in parts:
        if int(part.shape[1]) == max_points:
            padded.append(part)
            continue
        out = torch.zeros((part.shape[0], max_points), dtype=part.dtype)
        out[:, : part.shape[1]] = part
        padded.append(out)
    return torch.cat(padded, dim=0)
