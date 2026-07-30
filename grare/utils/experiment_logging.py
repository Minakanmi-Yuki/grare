from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def append_jsonl(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    return path


def reset_jsonl(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def numeric_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    payload: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(arr.size - finite.size),
    }
    if finite.size == 0:
        payload.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "p05": None,
                "p50": None,
                "p95": None,
            }
        )
        return payload

    payload.update(
        {
            "min": _safe_float(np.min(finite)),
            "max": _safe_float(np.max(finite)),
            "mean": _safe_float(np.mean(finite)),
            "std": _safe_float(np.std(finite)),
            "p05": _safe_float(np.percentile(finite, 5)),
            "p50": _safe_float(np.percentile(finite, 50)),
            "p95": _safe_float(np.percentile(finite, 95)),
        }
    )
    return payload


def boolean_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=bool).reshape(-1)
    total = int(arr.size)
    true_count = int(np.count_nonzero(arr))
    false_count = total - true_count
    return {
        "count": total,
        "true_count": true_count,
        "false_count": false_count,
        "true_rate": _safe_rate(true_count, total),
        "false_rate": _safe_rate(false_count, total),
    }


def _safe_rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return _safe_float(count / total)


def _safe_float(value: Any) -> float | None:
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric
