import math
from pathlib import Path

import numpy as np


def pair_rendered_images(method_dir):
    method_dir = Path(method_dir)
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    render_paths = sorted(render_dir.glob("*.png"))
    gt_by_name = {path.name: path for path in gt_dir.glob("*.png")}

    if not render_paths:
        raise FileNotFoundError(f"No rendered PNG files found in {render_dir}")

    render_names = {path.name for path in render_paths}
    gt_names = set(gt_by_name)
    if render_names != gt_names:
        missing_gt = sorted(render_names - gt_names)
        missing_render = sorted(gt_names - render_names)
        raise ValueError(
            "Rendered/GT filenames do not match: "
            f"missing_gt={missing_gt[:5]}, missing_render={missing_render[:5]}"
        )

    return [(path, gt_by_name[path.name]) for path in render_paths]


def basic_rgb_metrics(render, gt, bright_fraction=0.05, eps=1e-12):
    render = np.asarray(render, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if render.shape != gt.shape or render.ndim != 3 or render.shape[-1] != 3:
        raise ValueError(
            f"Expected matching HxWx3 arrays, got render={render.shape}, gt={gt.shape}"
        )
    if not 0 < bright_fraction <= 1:
        raise ValueError("bright_fraction must be in (0, 1]")
    if not np.isfinite(render).all() or not np.isfinite(gt).all():
        raise ValueError("RGB arrays must contain only finite values")

    error = render - gt
    l1 = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    psnr = float(-10.0 * math.log10(max(mse, eps)))

    luminance = (
        0.2126 * gt[..., 0]
        + 0.7152 * gt[..., 1]
        + 0.0722 * gt[..., 2]
    )
    threshold = float(np.quantile(luminance, 1.0 - bright_fraction))
    bright_mask = luminance >= threshold
    bright_error = error[bright_mask]
    bright_l1 = float(np.mean(np.abs(bright_error)))
    bright_mse = float(np.mean(bright_error ** 2))
    bright_psnr = float(-10.0 * math.log10(max(bright_mse, eps)))

    return {
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "bright_l1": bright_l1,
        "bright_mse": bright_mse,
        "bright_psnr": bright_psnr,
        "bright_threshold": threshold,
        "bright_pixel_fraction": float(np.mean(bright_mask)),
    }


def summarize_records(records, metric_names):
    if not records:
        raise ValueError("Cannot summarize an empty record list")

    summary = {}
    for metric_name in metric_names:
        values = np.asarray(
            [record[metric_name] for record in records], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {metric_name} contains non-finite values")
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary
