import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gaussian_splatting_2d"
    / "utils"
    / "evaluation_utils.py"
)
SPEC = importlib.util.spec_from_file_location("evaluation_utils", MODULE_PATH)
utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = utils
SPEC.loader.exec_module(utils)


class EvaluationUtilsTests(unittest.TestCase):
    def test_pairs_render_and_gt_by_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            method_dir = Path(directory)
            (method_dir / "renders").mkdir()
            (method_dir / "gt").mkdir()
            for name in ("00001.png", "00000.png"):
                (method_dir / "renders" / name).write_bytes(b"")
                (method_dir / "gt" / name).write_bytes(b"")

            pairs = utils.pair_rendered_images(method_dir)

            self.assertEqual(
                [render.name for render, _ in pairs],
                ["00000.png", "00001.png"],
            )

    def test_rejects_mismatched_render_and_gt_names(self):
        with tempfile.TemporaryDirectory() as directory:
            method_dir = Path(directory)
            (method_dir / "renders").mkdir()
            (method_dir / "gt").mkdir()
            (method_dir / "renders" / "00000.png").write_bytes(b"")
            (method_dir / "gt" / "00001.png").write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "do not match"):
                utils.pair_rendered_images(method_dir)

    def test_basic_metrics_include_bright_region(self):
        gt = np.zeros((4, 4, 3), dtype=np.float32)
        gt[0, 0] = 1.0
        render = gt.copy()
        render[0, 0] = 0.5

        metrics = utils.basic_rgb_metrics(
            render, gt, bright_fraction=1.0 / 16.0
        )

        self.assertGreater(metrics["bright_l1"], metrics["l1"])
        self.assertLess(metrics["bright_psnr"], metrics["psnr"])
        self.assertAlmostEqual(metrics["bright_pixel_fraction"], 1.0 / 16.0)

    def test_summarizes_metric_distributions(self):
        summary = utils.summarize_records(
            [{"psnr": 10.0}, {"psnr": 20.0}, {"psnr": 30.0}],
            ["psnr"],
        )

        self.assertEqual(summary["psnr"]["mean"], 20.0)
        self.assertEqual(summary["psnr"]["median"], 20.0)
        self.assertEqual(summary["psnr"]["min"], 10.0)
        self.assertEqual(summary["psnr"]["max"], 30.0)


if __name__ == "__main__":
    unittest.main()
