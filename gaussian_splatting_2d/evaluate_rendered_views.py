import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.evaluation_utils import (
    basic_rgb_metrics,
    pair_rendered_images,
    summarize_records,
)
from utils.loss_utils import ssim


METRIC_NAMES = [
    "l1",
    "psnr",
    "ssim",
    "lpips",
    "bright_l1",
    "bright_psnr",
    "bright_pixel_fraction",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exported train/test RGB views with identical metrics for "
            "multiple 2DGS-compatible model outputs."
        )
    )
    parser.add_argument("--model_paths", "-m", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--expected_views", type=int)
    parser.add_argument("--bright_fraction", type=float, default=0.05)
    parser.add_argument(
        "--lpips_net", choices=("alex", "squeeze", "vgg"), default="vgg"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "model"


def load_camera_names(model_path):
    cameras_path = model_path / "cameras.json"
    if not cameras_path.exists():
        return []
    with cameras_path.open(encoding="utf-8") as handle:
        cameras = json.load(handle)
    cameras.sort(key=lambda camera: int(camera.get("id", 0)))
    return [camera.get("img_name", "") for camera in cameras]


def source_name_for_render(render_name, camera_names):
    try:
        index = int(Path(render_name).stem)
    except ValueError:
        return ""
    if 0 <= index < len(camera_names):
        return camera_names[index]
    return ""


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = tf.to_tensor(image).unsqueeze(0)
    return array, tensor


def evaluate_model(
    model_path,
    label,
    args,
    lpips_model,
    device,
    output_dir,
):
    method_dir = (
        model_path / args.split / f"ours_{args.iteration}"
    )
    pairs = pair_rendered_images(method_dir)
    if args.expected_views is not None and len(pairs) != args.expected_views:
        raise ValueError(
            f"{label}: expected {args.expected_views} views, found {len(pairs)}"
        )

    camera_names = load_camera_names(model_path)
    records = []
    gt_hasher = hashlib.sha256()
    for render_path, gt_path in tqdm(pairs, desc=f"Evaluating {label}"):
        render_array, render_tensor = load_rgb(render_path)
        gt_array, gt_tensor = load_rgb(gt_path)
        if render_array.shape != gt_array.shape:
            raise ValueError(
                f"{label}/{render_path.name}: render shape {render_array.shape} "
                f"does not match GT shape {gt_array.shape}"
            )

        basic = basic_rgb_metrics(
            render_array,
            gt_array,
            bright_fraction=args.bright_fraction,
        )
        gt_uint8 = np.rint(gt_array * 255.0).astype(np.uint8)
        gt_hasher.update(render_path.name.encode("utf-8"))
        gt_hasher.update(np.asarray(gt_uint8.shape, dtype=np.int64).tobytes())
        gt_hasher.update(gt_uint8.tobytes())
        render_tensor = render_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        with torch.no_grad():
            ssim_value = float(ssim(render_tensor, gt_tensor).item())
            lpips_value = float(lpips_model(render_tensor, gt_tensor).item())

        records.append(
            {
                "render_name": render_path.name,
                "source_name": source_name_for_render(
                    render_path.name, camera_names
                ),
                **basic,
                "ssim": ssim_value,
                "lpips": lpips_value,
            }
        )

    point_cloud_path = (
        model_path
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Missing point cloud: {point_cloud_path}")
    gaussian_count = len(PlyData.read(point_cloud_path)["vertex"])

    summary = {
        "label": label,
        "model_path": str(model_path),
        "split": args.split,
        "iteration": args.iteration,
        "view_count": len(records),
        "image_width": int(render_array.shape[1]),
        "image_height": int(render_array.shape[0]),
        "lpips_net": args.lpips_net,
        "bright_fraction": args.bright_fraction,
        "gaussian_count": gaussian_count,
        "point_cloud_bytes": point_cloud_path.stat().st_size,
        "gt_pixel_fingerprint_sha256": gt_hasher.hexdigest(),
        "metrics": summarize_records(records, METRIC_NAMES),
    }

    label_slug = safe_label(label)
    per_view_path = output_dir / f"{label_slug}_per_view.csv"
    with per_view_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_dir / f"{label_slug}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n===== {label} =====")
    print(f"views={summary['view_count']}")
    print(f"resolution={summary['image_width']}x{summary['image_height']}")
    print(f"gaussians={summary['gaussian_count']}")
    print(f"gt_pixel_fingerprint_sha256={summary['gt_pixel_fingerprint_sha256']}")
    for metric_name in METRIC_NAMES:
        metric = summary["metrics"][metric_name]
        print(
            f"{metric_name}: mean={metric['mean']:.7f}, "
            f"median={metric['median']:.7f}, min={metric['min']:.7f}, "
            f"max={metric['max']:.7f}"
        )
    print(f"summary_json={summary_path}")
    print(f"per_view_csv={per_view_path}")
    return summary


def write_comparison_csv(summaries, output_path):
    fieldnames = [
        "label",
        "model_path",
        "view_count",
        "resolution",
        "gaussian_count",
        "point_cloud_bytes",
        "gt_pixel_fingerprint_sha256",
    ]
    for metric_name in METRIC_NAMES:
        fieldnames.extend(
            [f"{metric_name}_mean", f"{metric_name}_median"]
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "label": summary["label"],
                "model_path": summary["model_path"],
                "view_count": summary["view_count"],
                "resolution": (
                    f"{summary['image_width']}x{summary['image_height']}"
                ),
                "gaussian_count": summary["gaussian_count"],
                "point_cloud_bytes": summary["point_cloud_bytes"],
                "gt_pixel_fingerprint_sha256": summary[
                    "gt_pixel_fingerprint_sha256"
                ],
            }
            for metric_name in METRIC_NAMES:
                metric = summary["metrics"][metric_name]
                row[f"{metric_name}_mean"] = metric["mean"]
                row[f"{metric_name}_median"] = metric["median"]
            writer.writerow(row)


def main():
    args = parse_args()
    model_paths = [Path(path).resolve() for path in args.model_paths]
    labels = args.labels or [path.name for path in model_paths]
    if len(labels) != len(model_paths):
        raise ValueError("--labels must have the same count as --model_paths")
    if len(set(labels)) != len(labels):
        raise ValueError("--labels must be unique")
    label_slugs = [safe_label(label) for label in labels]
    if len(set(label_slugs)) != len(label_slugs):
        raise ValueError("--labels become non-unique after filename sanitization")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Loading LPIPS network: {args.lpips_net}")
    lpips_model = LPIPS(args.lpips_net, version="0.1").to(device).eval()

    summaries = []
    for model_path, label in zip(model_paths, labels):
        summaries.append(
            evaluate_model(
                model_path,
                label,
                args,
                lpips_model,
                device,
                output_dir,
            )
        )

    if len(summaries) > 1:
        reference = summaries[0]
        comparable_fields = [
            "view_count",
            "image_width",
            "image_height",
            "gt_pixel_fingerprint_sha256",
        ]
        for summary in summaries[1:]:
            mismatches = {
                field: (reference[field], summary[field])
                for field in comparable_fields
                if reference[field] != summary[field]
            }
            if mismatches:
                raise ValueError(
                    f"{summary['label']} is not directly comparable with "
                    f"{reference['label']}: {mismatches}"
                )

    combined_json = output_dir / "comparison_summary.json"
    with combined_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    combined_csv = output_dir / "comparison_summary.csv"
    write_comparison_csv(summaries, combined_csv)
    print(f"\ncomparison_json={combined_json}")
    print(f"comparison_csv={combined_csv}")
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
