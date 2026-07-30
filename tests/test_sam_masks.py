"""SAM mask helpers: projection, back-projection, candidate predictor stub."""
from __future__ import annotations

import unittest
import numpy as np

from grare.relabeling.sam_masks import (
    project_3d_to_pixel,
    back_project_mask_to_points,
    points_grid_from_depth,
)


class ProjectionTests(unittest.TestCase):
    def test_pinhole_round_trip(self):
        K = np.array([[500., 0, 300], [0, 500, 200], [0, 0, 1]])
        u, v = project_3d_to_pixel(np.array([0.1, 0.05, 1.0]), K)
        self.assertEqual((u, v), (350, 225))

    def test_z_zero_returns_invalid(self):
        K = np.array([[500., 0, 300], [0, 500, 200], [0, 0, 1]])
        u, v = project_3d_to_pixel(np.array([0.1, 0.0, 0.0]), K)
        self.assertEqual((u, v), (-1, -1))


class BackProjectTests(unittest.TestCase):
    def test_picks_only_masked_points(self):
        H, W = 4, 5
        pts = np.zeros((H, W, 3), dtype=np.float32)
        pts[1, 2, :] = [0.1, 0.2, 0.5]
        pts[2, 3, :] = [0.4, 0.5, 0.6]
        pts[3, 0, :] = [0.0, 0.0, 0.7]   # outside mask
        mask = np.zeros((H, W), dtype=bool)
        mask[1:3, :] = True
        out = back_project_mask_to_points(mask, pts)
        self.assertEqual(out.shape, (2, 3))

    def test_invalid_depth_skipped(self):
        H, W = 3, 3
        pts = np.zeros((H, W, 3), dtype=np.float32)
        pts[1, 1, :] = [0.0, 0.0, 0.0]   # z=0 → invalid
        valid = pts[..., 2] > 0
        mask = np.ones((H, W), dtype=bool)
        out = back_project_mask_to_points(mask, pts, valid)
        self.assertEqual(len(out), 0)


class PointsGridTests(unittest.TestCase):
    def test_basic_consistency(self):
        H, W = 4, 5
        depth_mm = np.full((H, W), 1000, dtype=np.uint16)  # 1.0 m everywhere
        K = np.array([[500., 0, 2.0], [0, 500, 1.5], [0, 0, 1]])
        pts, valid = points_grid_from_depth(depth_mm, K)
        self.assertEqual(pts.shape, (H, W, 3))
        self.assertTrue(valid.all())
        # The pixel at (col=2, row=1) should reproject to (0,...,1)
        self.assertAlmostEqual(float(pts[1, 2, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(pts[1, 2, 2]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
