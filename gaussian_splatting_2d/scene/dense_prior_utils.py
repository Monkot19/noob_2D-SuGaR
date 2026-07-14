import cv2
import numpy as np


def _unit_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    is_integer = np.issubdtype(image.dtype, np.integer)
    image = image[..., :3].astype(np.float32)
    if is_integer or np.nanmax(image) > 1.5:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def validate_dense_image_order(cam_infos, dense_colors, max_mean_error=0.25):
    """Validate that dense map index equals COLMAP image_id - 1."""
    dense_colors = np.asarray(dense_colors)
    if dense_colors.ndim != 4 or dense_colors.shape[-1] < 3:
        raise ValueError(
            f"pointsColor_all.npy must have shape (N,H,W,3), got {dense_colors.shape}"
        )
    if dense_colors.shape[0] != len(cam_infos):
        raise ValueError(
            f"Dense maps ({dense_colors.shape[0]}) do not match cameras ({len(cam_infos)})"
        )

    dense_indices = np.array([cam.image_id - 1 for cam in cam_infos], dtype=np.int64)
    if set(dense_indices.tolist()) != set(range(len(cam_infos))):
        raise ValueError(
            "Trackless dense fallback requires contiguous COLMAP image IDs 1..N; "
            f"got {[cam.image_id for cam in cam_infos]}"
        )

    rgb_errors = []
    bgr_errors = []
    for cam_info, dense_index in zip(cam_infos, dense_indices):
        source = _unit_rgb(np.asarray(cam_info.image.convert("RGB")))
        dense = _unit_rgb(dense_colors[dense_index])
        source = cv2.resize(
            source, (dense.shape[1], dense.shape[0]), interpolation=cv2.INTER_AREA
        )
        rgb_errors.append(float(np.mean(np.abs(source - dense))))
        bgr_errors.append(float(np.mean(np.abs(source - dense[..., ::-1]))))

    if np.mean(bgr_errors) < np.mean(rgb_errors):
        errors, channel_order = np.asarray(bgr_errors), "BGR"
    else:
        errors, channel_order = np.asarray(rgb_errors), "RGB"

    if np.max(errors) > max_mean_error:
        raise ValueError(
            "Dense map order could not be validated against source images: "
            f"maximum normalized color MAE {np.max(errors):.4f} exceeds "
            f"the safety limit {max_mean_error:.4f}"
        )
    return dense_indices, errors, channel_order


def reshape_confidence(confidence, dense_shape):
    confidence = np.asarray(confidence)
    n_images, height, width, _ = dense_shape
    if confidence.size != n_images * height * width:
        raise ValueError(
            f"Confidence shape {confidence.shape} is incompatible with dense XYZ "
            f"shape {dense_shape}"
        )
    return confidence.reshape(n_images, height, width)


def estimate_depth_scale(world_points, metric_depth, R, T, confidence=None,
                         min_valid=128):
    """Estimate per-image scale using camera-Z / Metric3D depth."""
    world_points = np.asarray(world_points, dtype=np.float64)
    metric_depth = np.asarray(metric_depth, dtype=np.float64)
    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3) world points, got {world_points.shape}")
    if metric_depth.ndim != 2:
        raise ValueError(f"Expected (H,W) Metric3D depth, got {metric_depth.shape}")

    height, width = world_points.shape[:2]
    depth = cv2.resize(metric_depth, (width, height), interpolation=cv2.INTER_LINEAR)
    world_flat = world_points.reshape(-1, 3)
    camera_z = (
        world_flat @ np.asarray(R, dtype=np.float64)
        + np.asarray(T, dtype=np.float64).reshape(1, 3)
    )[:, 2]
    depth_flat = depth.reshape(-1)

    valid = (
        np.isfinite(world_flat).all(axis=1)
        & np.isfinite(camera_z)
        & np.isfinite(depth_flat)
        & (camera_z > 1e-6)
        & (depth_flat > 1e-6)
    )
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
        if confidence.size != valid.size:
            raise ValueError("Confidence and dense XYZ sample counts differ")
        finite_conf = np.isfinite(confidence)
        candidates = confidence[valid & finite_conf]
        if candidates.size >= min_valid:
            threshold = np.median(candidates)
            valid &= finite_conf & (confidence >= threshold)

    ratios = camera_z[valid] / depth_flat[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < min_valid:
        raise ValueError(
            f"Only {ratios.size} valid dense depth correspondences; need {min_valid}"
        )

    low, high = np.quantile(ratios, [0.05, 0.95])
    trimmed = ratios[(ratios >= low) & (ratios <= high)]
    if trimmed.size >= min_valid:
        ratios = trimmed

    scale = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - scale)))
    return scale, len(ratios), mad / max(abs(scale), 1e-12)


def validate_dense_points_match_ply(dense_points, ply_points, sample_count=4096,
                                    tolerance=1e-4):
    dense = np.asarray(dense_points).reshape(-1, 3)
    ply = np.asarray(ply_points).reshape(-1, 3)
    if dense.shape != ply.shape:
        raise ValueError(
            f"Dense XYZ has {len(dense)} points but points3D.ply has {len(ply)}"
        )
    indices = np.linspace(0, len(dense) - 1, min(sample_count, len(dense)), dtype=np.int64)
    errors = np.linalg.norm(
        dense[indices].astype(np.float64) - ply[indices].astype(np.float64), axis=1
    )
    if not np.isfinite(errors).all() or np.max(errors) > tolerance:
        raise ValueError(
            "points3D_all.npy and points3D.ply are not aligned: sampled max "
            f"XYZ error is {np.nanmax(errors):.6g}"
        )
    return float(np.max(errors))


def sanitize_normals(normals, fallback_normals=None, eps=1e-8):
    normals = np.asarray(normals, dtype=np.float32).copy()
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & np.isfinite(lengths) & (lengths > eps)
    normals[valid] /= lengths[valid, None]

    if fallback_normals is None:
        normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fallback, _ = sanitize_normals(fallback_normals, None, eps)
        normals[~valid] = fallback[~valid]
    return normals, int((~valid).sum())
