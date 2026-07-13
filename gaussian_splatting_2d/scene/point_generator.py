import cv2
import numpy as np


def _scaled_intrinsics(cam_info, intrinsic, image_width, image_height):
    params = np.asarray(intrinsic.params, dtype=np.float64)
    if params.size == 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif params.size >= 4:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported camera intrinsics: {params}")

    calibration_width = float(cam_info.width)
    calibration_height = float(cam_info.height)
    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError(
            f"Invalid calibration size {cam_info.width}x{cam_info.height}"
        )

    scale_x = image_width / calibration_width
    scale_y = image_height / calibration_height
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def _resize_priors(cam_info, image_width, image_height):
    depth = np.asarray(cam_info.depth_map, dtype=np.float32).squeeze()
    normals = np.asarray(cam_info.world_normal_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}")
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"Expected a (H,W,3) normal map, got {normals.shape}")

    target_size = (image_width, image_height)
    if depth.shape != (image_height, image_width):
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
    if normals.shape[:2] != (image_height, image_width):
        normals = cv2.resize(normals, target_size, interpolation=cv2.INTER_LINEAR)

    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    valid = np.isfinite(normals).all(axis=2, keepdims=True) & (lengths > 1e-8)
    normals = np.divide(
        normals,
        lengths,
        out=np.zeros_like(normals),
        where=valid,
    )
    normals[~valid[..., 0]] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return depth, normals


def gaussian_generator(cam_infos, cam_intrinsics, num_points, image_depth_scale_map):
    """
    Note: Not guaranteed to return exactly num_points
    Usually, number of returned points is 33% lower than num_points
    """

    num_images = len(cam_infos)
    num_points_per_image = num_points // num_images

    assert num_points >= num_images, "Generate more points than number of images"

    xyzs, rgbs, normals = [], [], []
    for cam_info in cam_infos:
        image = np.asarray(cam_info.image)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected an RGB image, got {image.shape}")
        image_height, image_width = image.shape[:2]
        fx, fy, cx, cy = _scaled_intrinsics(
            cam_info,
            cam_intrinsics[cam_info.uid],
            image_width,
            image_height,
        )
        depth_map, normal_map = _resize_priors(
            cam_info, image_width, image_height
        )

        # Sample from 2D gaussian with 50% samples coming from central portion of image bounded by ([-w/4, w/4], [-h/4, h/4])
        xs = np.random.normal(
            loc=cx, scale=image_width / 2.696, size=num_points_per_image
        ).astype(int)
        ys = np.random.normal(
            loc=cy, scale=image_height / 2.696, size=num_points_per_image
        ).astype(int)

        valid_mask = (
            (xs >= 0) & (xs < image_width)
            & (ys >= 0) & (ys < image_height)
        )

        xs = xs[valid_mask]
        ys = ys[valid_mask]

        rgb = image[ys, xs, :3]
        normal = normal_map[ys, xs, :]
        depth = depth_map[ys, xs]

        depth = depth * image_depth_scale_map[cam_info.image_id]["estimated_scale"]

        camera_coords = np.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], axis=0)

        # Camera to world transform
        world_coords = cam_info.R @ (camera_coords - cam_info.T[:, None])
        world_coords = world_coords.T

        xyzs.append(world_coords)
        rgbs.append(rgb)
        normals.append(normal)

    xyzs = np.vstack(xyzs)
    rgbs = np.vstack(rgbs)
    normals = np.vstack(normals)

    return xyzs, rgbs, normals
