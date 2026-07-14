import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gaussian_splatting_2d"
    / "scene"
    / "dense_prior_utils.py"
)
SPEC = importlib.util.spec_from_file_location("dense_prior_utils", MODULE_PATH)
utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = utils
SPEC.loader.exec_module(utils)


class ArrayImage:
    def __init__(self, array):
        self.array = np.asarray(array)

    def convert(self, _mode):
        return self

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype=dtype)


class DensePriorUtilsTests(unittest.TestCase):
    def test_validates_image_id_order_and_colors(self):
        image_a = np.zeros((8, 10, 3), dtype=np.uint8)
        image_a[..., 0] = 200
        image_b = np.zeros((8, 10, 3), dtype=np.uint8)
        image_b[..., 1] = 150
        cameras = [
            SimpleNamespace(image_id=2, image=ArrayImage(image_b)),
            SimpleNamespace(image_id=1, image=ArrayImage(image_a)),
        ]
        dense_colors = np.stack((image_a, image_b))

        indices, errors, order = utils.validate_dense_image_order(
            cameras, dense_colors
        )

        self.assertEqual(indices.tolist(), [1, 0])
        self.assertEqual(order, "RGB")
        self.assertLess(float(np.max(errors)), 1e-6)

    def test_estimates_robust_depth_scale(self):
        world = np.zeros((12, 16, 3), dtype=np.float32)
        world[..., 2] = 5.0
        depth = np.full((24, 32), 2.0, dtype=np.float32)
        confidence = np.arange(192, dtype=np.float32).reshape(12, 16)
        world[0, 0] = np.nan

        scale, count, relative_mad = utils.estimate_depth_scale(
            world,
            depth,
            np.eye(3),
            np.array([0.0, 0.0, 1.0]),
            confidence=confidence,
            min_valid=32,
        )

        self.assertAlmostEqual(scale, 3.0, places=6)
        self.assertGreaterEqual(count, 32)
        self.assertAlmostEqual(relative_mad, 0.0, places=6)

    def test_validates_ply_alignment_and_sanitizes_normals(self):
        dense = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
        ply = dense.reshape(-1, 3).copy()
        self.assertEqual(
            utils.validate_dense_points_match_ply(dense, ply, sample_count=24),
            0.0,
        )

        fallback = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
        normals, invalid = utils.sanitize_normals(
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]),
            fallback_normals=fallback,
        )
        self.assertEqual(invalid, 2)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(normals[1:], fallback[1:])

    def test_validates_sampled_ply_alignment(self):
        dense = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
        sample_indices = np.array([0, 5, 11, 23], dtype=np.int64)
        ply = dense.reshape(-1, 3)[sample_indices].copy()

        self.assertEqual(
            utils.validate_dense_points_match_ply(
                dense,
                ply,
                point_indices=sample_indices,
                sample_count=len(sample_indices),
            ),
            0.0,
        )

    def test_rejects_invalid_sample_indices(self):
        dense = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
        ply = dense.reshape(-1, 3)[[0, 5]].copy()

        with self.assertRaisesRegex(ValueError, "unique, strictly increasing"):
            utils.validate_dense_points_match_ply(
                dense, ply, point_indices=np.array([5, 5], dtype=np.int64)
            )

    def test_reshapes_flat_confidence(self):
        confidence = np.arange(24).reshape(2, 12)
        reshaped = utils.reshape_confidence(confidence, (2, 3, 4, 3))
        self.assertEqual(reshaped.shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
