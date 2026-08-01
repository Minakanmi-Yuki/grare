"""SAM-based per-candidate object segmentation for the object tier.

Each grasp candidate's translation is projected to the camera image plane
and used as a SAM point prompt. SAM returns three multi-scale masks; we
pick the smallest mask whose area falls in [min_area, max_area_ratio * H*W],
which heuristically corresponds to the candidate's target object (rather
than the table or the whole image).

Train, eval and deployment all share this same forward path, so the
ObjectEncoder sees a consistent mask distribution regardless of stage.
GraspNet1B's GT segLabel is intentionally NOT used.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamPredictorConfig:
    checkpoint: str
    model_type: str = "vit_t"           # MobileSAM uses "vit_t"
    device: str = "cuda"
    multimask_pick: str = "smallest_valid"  # "smallest_valid" | "best_iou"
    min_area_pixels: int = 200
    max_area_ratio: float = 0.4
    iou_score_floor: float = 0.0
    prompt_batch_size: int = 64


class SamCandidatePredictor:
    """Lazy-loaded MobileSAM SamPredictor wrapper.

    Usage per frame:
        predictor.set_image(rgb)
        for c in candidates:
            mask = predictor.predict_at_pixel(u, v)   # bool (H, W) or None
    """

    def __init__(self, config: SamPredictorConfig) -> None:
        self.config = config
        # Lazy import: callers that never relabel never pay the import cost.
        from mobile_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
        sam.to(config.device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self._H = 0
        self._W = 0
        self._prompt_batch_size = _prompt_batch_size(config.prompt_batch_size)

    def set_image(self, rgb: np.ndarray) -> None:
        """Encode a new RGB frame for subsequent point prompts."""
        self._predictor.set_image(rgb)
        self._H, self._W = int(rgb.shape[0]), int(rgb.shape[1])

    @property
    def image_hw(self) -> tuple[int, int]:
        return self._H, self._W

    def predict_at_pixel(self, u: int, v: int) -> np.ndarray | None:
        """Return a bool (H, W) mask, or None if no valid mask found.

        ``u, v`` are in pixel coords (col, row). When the prompt is invalid
        or all 3 candidate masks fail the area filter, returns None and the
        caller should fall back to an empty point cloud.
        """
        if self._H == 0:
            raise RuntimeError("call set_image before predict_at_pixel")
        if not (0 <= u < self._W and 0 <= v < self._H):
            return None
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[int(u), int(v)]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        return self._select_mask(masks, scores)

    def predict_many_at_pixels(self, pixels: list[tuple[int, int]]) -> list[np.ndarray | None]:
        """Return one selected mask per point prompt.

        MobileSAM's image encoder is already cached by ``set_image``. Batching
        point prompts keeps the mask decoder on larger CUDA kernels instead of
        launching one tiny prediction per cluster.
        """
        if self._H == 0:
            raise RuntimeError("call set_image before predict_many_at_pixels")
        results: list[np.ndarray | None] = [None] * len(pixels)
        valid: list[tuple[int, int, int]] = [
            (idx, int(u), int(v))
            for idx, (u, v) in enumerate(pixels)
            if 0 <= int(u) < self._W and 0 <= int(v) < self._H
        ]
        if not valid:
            return results

        import torch

        for start in range(0, len(valid), self._prompt_batch_size):
            chunk = valid[start : start + self._prompt_batch_size]
            coords = np.array([(u, v) for _, u, v in chunk], dtype=np.float32)
            coords = self._predictor.transform.apply_coords(coords, self._predictor.original_size)
            coords_torch = torch.as_tensor(
                coords,
                dtype=torch.float,
                device=self._predictor.device,
            )[:, None, :]
            labels_torch = torch.ones(
                (len(chunk), 1),
                dtype=torch.int,
                device=self._predictor.device,
            )
            masks, scores, low_res_masks = self._predictor.predict_torch(
                coords_torch,
                labels_torch,
                boxes=None,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
            )
            masks_np = masks.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            del masks, scores, low_res_masks
            for row, (idx, _, _) in enumerate(chunk):
                results[idx] = self._select_mask(masks_np[row], scores_np[row])
        return results

    def _select_mask(self, masks: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        cfg = self.config
        HW = self._H * self._W
        max_area = int(cfg.max_area_ratio * HW)

        valid: list[tuple[int, int]] = []  # (area, idx)
        for k in range(masks.shape[0]):
            if scores[k] < cfg.iou_score_floor:
                continue
            area = int(masks[k].sum())
            if area < cfg.min_area_pixels or area > max_area:
                continue
            valid.append((area, k))
        if not valid:
            return None
        if cfg.multimask_pick == "best_iou":
            best = max(valid, key=lambda t: float(scores[t[1]]))
        else:  # smallest_valid (default — biases toward instance vs table)
            best = min(valid, key=lambda t: t[0])
        return masks[best[1]].astype(bool, copy=False)


def _prompt_batch_size(default: int) -> int:
    raw = os.environ.get("GRARE_SAM_PROMPT_BATCH_SIZE")
    if raw is not None and raw.strip() != "":
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(default))


def project_3d_to_pixel(translation: np.ndarray, intrinsics: np.ndarray) -> tuple[int, int]:
    """Pinhole projection. Returns (u, v) = (col, row), rounded to nearest int.

    Returns (-1, -1) when the 3D point is behind the camera.
    """
    x, y, z = float(translation[0]), float(translation[1]), float(translation[2])
    if z <= 1e-6:
        return -1, -1
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = int(round(x * fx / z + cx))
    v = int(round(y * fy / z + cy))
    return u, v


def back_project_mask_to_points(
    mask: np.ndarray,
    points_grid: np.ndarray,
    valid_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Pull 3D points whose pixel falls inside ``mask``.

    Args:
        mask: (H, W) bool — SAM-predicted instance mask
        points_grid: (H, W, 3) float32 — per-pixel 3D point in camera frame
            (z=0 marks "no depth"; caller can supply ``valid_grid`` to skip).
        valid_grid: (H, W) bool — True for pixels with a valid 3D point.
            Defaults to ``points_grid[..., 2] > 0``.

    Returns:
        (M, 3) float32 array of 3D points inside the mask. M can be 0.
    """
    if valid_grid is None:
        valid_grid = points_grid[..., 2] > 0
    pick = mask & valid_grid
    return points_grid[pick].astype(np.float32, copy=False)


def points_grid_from_depth(
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_scale: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a (H, W) depth image into a (H, W, 3) per-pixel xyz grid.

    Returns (points_grid, valid_grid). ``valid_grid[v, u] == True`` exactly
    when ``depth_mm[v, u] > 0``.
    """
    H, W = depth_mm.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    z = depth_mm.astype(np.float32) / float(depth_scale)
    xmap = np.arange(W, dtype=np.float32)
    ymap = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xmap, ymap)
    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    valid = z > 0
    return points, valid
