import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gaussian_splatting_2d"
    / "scene"
    / "point_generator.py"
)
SPEC = importlib.util.spec_from_file_location("point_generator", MODULE_PATH)
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class PointGeneratorTests(unittest.TestCase):
    def test_scales_intrinsics_and_aligns_priors_to_image_grid(self):
        image = np.zeros((384, 512, 3), dtype=np.uint8)
        image[192, 256] = [10, 20, 30]
        camera = SimpleNamespace(
            uid=1,
            image_id=7,
            width=4000,
            height=3000,
            image=image,
            depth_map=np.full((768, 1024), 2.0, dtype=np.float32),
            world_normal_map=np.dstack([
                np.zeros((192, 256), dtype=np.float32),
                np.zeros((192, 256), dtype=np.float32),
                np.ones((192, 256), dtype=np.float32),
            ]),
            R=np.eye(3),
            T=np.zeros(3),
        )
        intrinsics = {
            1: SimpleNamespace(params=np.array([1000, 1000, 2000, 1500]))
        }
        scales = {7: {"estimated_scale": 1.0}}

        with mock.patch.object(
            generator.np.random,
            "normal",
            side_effect=[np.array([256.0]), np.array([192.0])],
        ):
            xyz, rgb, normals = generator.gaussian_generator(
                [camera], intrinsics, 1, scales
            )

        np.testing.assert_allclose(xyz, [[0.0, 0.0, 2.0]], atol=1e-6)
        np.testing.assert_array_equal(rgb, [[10, 20, 30]])
        np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]], atol=1e-6)

    def test_supports_simple_pinhole_intrinsics(self):
        camera = SimpleNamespace(width=4000, height=3000)
        intrinsic = SimpleNamespace(params=np.array([1000, 2000, 1500]))

        scaled = generator._scaled_intrinsics(camera, intrinsic, 512, 384)

        np.testing.assert_allclose(scaled, [128.0, 128.0, 256.0, 192.0])

    def test_skips_cameras_without_depth_scales(self):
        image = np.zeros((384, 512, 3), dtype=np.uint8)
        image[192, 256] = [10, 20, 30]
        camera = SimpleNamespace(
            uid=1,
            image_id=7,
            width=512,
            height=384,
            image=image,
            depth_map=np.full((384, 512), 2.0, dtype=np.float32),
            world_normal_map=np.dstack([
                np.zeros((384, 512), dtype=np.float32),
                np.zeros((384, 512), dtype=np.float32),
                np.ones((384, 512), dtype=np.float32),
            ]),
            R=np.eye(3),
            T=np.zeros(3),
        )
        camera_without_scale = SimpleNamespace(**vars(camera))
        camera_without_scale.image_id = 8
        intrinsics = {
            1: SimpleNamespace(params=np.array([128, 128, 256, 192]))
        }
        scales = {7: {"estimated_scale": 1.0}}

        with mock.patch.object(
            generator.np.random,
            "normal",
            side_effect=[
                np.array([256.0, 256.0]),
                np.array([192.0, 192.0]),
            ],
        ):
            xyz, rgb, normals = generator.gaussian_generator(
                [camera_without_scale, camera], intrinsics, 2, scales
            )

        self.assertEqual(xyz.shape, (2, 3))
        np.testing.assert_array_equal(rgb, [[10, 20, 30], [10, 20, 30]])
        np.testing.assert_allclose(
            normals, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1e-6
        )

    def test_rejects_when_no_camera_has_a_valid_depth_scale(self):
        camera = SimpleNamespace(image_id=7)

        with self.assertRaisesRegex(
            ValueError, "No cameras have a finite, positive depth scale"
        ):
            generator.gaussian_generator(
                [camera], {}, 1, {7: {"estimated_scale": np.nan}}
            )


if __name__ == "__main__":
    unittest.main()
