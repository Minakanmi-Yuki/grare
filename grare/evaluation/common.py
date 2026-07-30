from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np


@dataclass
class EvaluationSummary:
    benchmark: str
    camera: str
    split: str
    dump_folder: str
    ap: float
    ap_04: float | None
    ap_08: float | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_topk_accuracy(
    res: np.ndarray,
    *,
    benchmark: str,
    camera: str,
    split: str,
    dump_folder: str | Path,
    ap_values: list[float] | tuple[float, ...] | np.ndarray | float | None = None,
) -> EvaluationSummary:
    arr = np.asarray(res, dtype=np.float32)
    ap_values_list = _coerce_ap_values(ap_values)
    ap = float(np.mean(arr))
    ap_04 = None
    ap_08 = None
    if arr.ndim >= 4 and arr.shape[-1] >= 4:
        ap_04 = float(np.mean(arr[..., 1]))
        ap_08 = float(np.mean(arr[..., 3]))
    elif ap_values_list is not None and len(ap_values_list) >= 3:
        ap_04 = float(ap_values_list[1])
        ap_08 = float(ap_values_list[2])

    extra = {
        "shape": list(arr.shape),
        "ap_percent": round(ap * 100.0, 4),
        "ap_04_percent": round(ap_04 * 100.0, 4) if ap_04 is not None else None,
        "ap_08_percent": round(ap_08 * 100.0, 4) if ap_08 is not None else None,
        "mean_percent": round(ap * 100.0, 4),
    }
    if ap_values_list is not None:
        extra["official_ap_values"] = ap_values_list
        extra["official_ap_percent_values"] = [round(float(x) * 100.0, 4) for x in ap_values_list]

    split_breakdown = _split_breakdown(arr, split=split)
    if split_breakdown:
        extra["split_breakdown"] = split_breakdown

    return EvaluationSummary(
        benchmark=benchmark,
        camera=camera,
        split=split,
        dump_folder=str(dump_folder),
        ap=ap,
        ap_04=ap_04,
        ap_08=ap_08,
        extra=extra,
    )


def _split_breakdown(arr: np.ndarray, *, split: str) -> dict[str, dict[str, float | None]]:
    if split != "test" or arr.ndim < 1 or arr.shape[0] < 90:
        return {}
    return {
        "overall": _metric_triplet(arr),
        "seen": _metric_triplet(arr[:30]),
        "similar": _metric_triplet(arr[30:60]),
        "novel": _metric_triplet(arr[60:90]),
    }


def _metric_triplet(arr: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(arr, dtype=np.float32)
    if values.size == 0:
        return {"ap": None, "ap_04": None, "ap_08": None}
    ap = float(np.mean(values))
    ap_04 = None
    ap_08 = None
    if values.ndim >= 4 and values.shape[-1] >= 4:
        ap_04 = float(np.mean(values[..., 1]))
        ap_08 = float(np.mean(values[..., 3]))
    return {"ap": ap, "ap_04": ap_04, "ap_08": ap_08}


def _coerce_ap_values(ap_values: list[float] | tuple[float, ...] | np.ndarray | float | None) -> list[float] | None:
    if ap_values is None:
        return None
    arr = np.asarray(ap_values, dtype=np.float32)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr.reshape(-1)]


def save_summary(summary: EvaluationSummary, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    return save_path
