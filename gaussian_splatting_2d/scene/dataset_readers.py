#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from collections import defaultdict

from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
    read_points3D_text_with_tracks, read_points3D_binary_with_tracks
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.point_generator import gaussian_generator
from scene.dense_prior_utils import (
    estimate_depth_scale,
    reshape_confidence,
    sanitize_normals,
    validate_dense_image_order,
    validate_dense_points_match_ply,
)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_id: int
    image_path: str
    image_name: str
    width: int
    height: int
    depth_map: np.array = None
    normal_map: np.array = None
    world_normal_map: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def transform_normal_map_to_world(normal_map, R):
    H, W, _ = normal_map.shape
    normals_flat = normal_map.reshape(-1, 3)  # (H*W, 3)

    # Apply rotation (Rn for each normal n in N): (H*W, 3) @ (3, 3).T => (H*W, 3)
    normals_world_flat = normals_flat @ R.T  # (H*W, 3)

    # Reshape back
    normals_world = normals_world_flat.reshape(H, W, 3)

    return normals_world

def estimate_subpixel_normal(normal_map, x, y):
    normal = np.zeros(3, dtype=np.float32)
    for c in range(3):
        normal[c] = cv2.remap(normal_map[..., c],
                               np.array([[x]], dtype=np.float32),
                               np.array([[y]], dtype=np.float32),
                               interpolation=cv2.INTER_LINEAR)[0, 0]
    return normal / np.linalg.norm(normal)

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_path = os.path.join(os.path.dirname(images_folder), "estimated_dense_depth", f"{image_name}.npy")
        depth_map = None
        if os.path.exists(depth_path):
            depth_map = np.load(depth_path)

        normal_path = os.path.join(os.path.dirname(images_folder), "estimated_dense_normal", f"{image_name}.npy")
        normal_map = None
        world_normal_map = None
        if os.path.exists(normal_path):
            normal_map = np.load(normal_path)
            normal_map = normal_map / np.linalg.norm(normal_map, axis=2, keepdims=True)

            world_normal_map = transform_normal_map_to_world(normal_map, R)
            world_normal_map = world_normal_map / np.linalg.norm(world_normal_map, axis=2, keepdims=True)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_id=extr.id,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_map=depth_map, normal_map=normal_map, world_normal_map=world_normal_map)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readTracklessDensePrior(path, cam_infos, cam_intrinsics, ply_path,
                            initialization_prior):
    if initialization_prior != "monocular_depth_normal":
        raise ValueError(
            "Trackless dense data currently supports only "
            "--initialization_prior monocular_depth_normal"
        )

    sparse_path = os.path.join(path, "sparse", "0")
    points_path = os.path.join(sparse_path, "points3D_all.npy")
    colors_path = os.path.join(sparse_path, "pointsColor_all.npy")
    sample_indices_path = os.path.join(
        sparse_path, "points3D_sample_indices.npy"
    )
    confidence_paths = [
        os.path.join(sparse_path, "confidence.npy"),
        os.path.join(sparse_path, "confidence_dsp.npy"),
    ]

    if not os.path.exists(colors_path):
        raise FileNotFoundError(
            f"Trackless dense initialization requires {colors_path} to validate "
            "the image-to-dense-map order"
        )

    print(f"No COLMAP tracks found; using trackless dense points from {points_path}")
    source_pcd = fetchPly(ply_path)
    dense_points = np.load(points_path, mmap_mode="r")
    dense_colors = np.load(colors_path, mmap_mode="r")
    if dense_points.ndim != 4 or dense_points.shape[-1] != 3:
        raise ValueError(
            f"points3D_all.npy must have shape (N,H,W,3), got {dense_points.shape}"
        )

    dense_point_count = int(np.prod(dense_points.shape[:-1]))
    source_point_count = len(source_pcd.points)
    source_indices = None
    if source_point_count != dense_point_count:
        if not os.path.exists(sample_indices_path):
            raise FileNotFoundError(
                f"points3D.ply contains {source_point_count} points while dense XYZ "
                f"contains {dense_point_count}; sampled initialization requires "
                f"{sample_indices_path}"
            )
        source_indices = np.load(sample_indices_path, mmap_mode="r")
        print(
            f"Using {source_point_count} sampled source points via "
            f"{sample_indices_path}"
        )

    max_xyz_error = validate_dense_points_match_ply(
        dense_points, source_pcd.points, point_indices=source_indices
    )
    print(f"Dense XYZ/PLY validation passed; sampled max error={max_xyz_error:.3g}")

    dense_indices, color_errors, channel_order = validate_dense_image_order(
        cam_infos, dense_colors
    )
    print(f"Dense image order validated with {channel_order} colors")
    for cam_info, dense_index, color_error in zip(
        cam_infos, dense_indices, color_errors
    ):
        print(
            f"  {cam_info.image_name}: dense_index={dense_index}, "
            f"color_MAE={color_error:.4f}"
        )

    confidence = None
    for confidence_path in confidence_paths:
        if os.path.exists(confidence_path):
            confidence = reshape_confidence(
                np.load(confidence_path, mmap_mode="r"), dense_points.shape
            )
            print(f"Using dense confidence from {confidence_path}")
            break

    image_depth_scale_map = {}
    metric_dense_normals = np.empty(dense_points.shape, dtype=np.float32)
    dense_height, dense_width = dense_points.shape[1:3]

    for cam_info, dense_index in zip(cam_infos, dense_indices):
        if cam_info.depth_map is None or cam_info.world_normal_map is None:
            raise FileNotFoundError(
                f"Missing Metric3D depth/normal prior for {cam_info.image_name}; "
                f"run estimate_depth_normal_priors.py -s {path} first"
            )

        dense_confidence = None if confidence is None else confidence[dense_index]
        scale, sample_count, relative_mad = estimate_depth_scale(
            dense_points[dense_index],
            cam_info.depth_map,
            cam_info.R,
            cam_info.T,
            confidence=dense_confidence,
        )
        image_depth_scale_map[cam_info.image_id] = {
            "estimated_scale": scale,
            "estimated_scale_variance": relative_mad,
        }
        metric_dense_normals[dense_index] = cv2.resize(
            cam_info.world_normal_map,
            (dense_width, dense_height),
            interpolation=cv2.INTER_LINEAR,
        )
        print(
            f"  {cam_info.image_name}: depth_scale={scale:.6g}, "
            f"samples={sample_count}, relative_MAD={relative_mad:.4f}"
        )

    source_normals, source_invalid = sanitize_normals(source_pcd.normals)
    metric_normals_flat = metric_dense_normals.reshape(-1, 3)
    source_metric_normals = (
        metric_normals_flat
        if source_indices is None
        else metric_normals_flat[np.asarray(source_indices, dtype=np.int64)]
    )
    source_prior_normals, prior_invalid = sanitize_normals(
        source_metric_normals, fallback_normals=source_normals
    )
    if source_invalid or prior_invalid:
        print(
            f"[WARN] Invalid normals: source={source_invalid}, "
            f"Metric3D={prior_invalid}; safe fallbacks were used"
        )

    num_generated_points = min(75000, len(cam_infos) * 1000)
    generated_xyz, generated_rgb, generated_normals = gaussian_generator(
        cam_infos, cam_intrinsics, num_generated_points, image_depth_scale_map
    )
    generated_normals, generated_invalid = sanitize_normals(generated_normals)
    if generated_invalid:
        print(f"[WARN] Replaced {generated_invalid} invalid generated normals")

    xyz = np.vstack((np.asarray(source_pcd.points), generated_xyz))
    rgb = np.vstack((np.asarray(source_pcd.colors), generated_rgb / 255.0))
    normals = np.vstack((source_prior_normals, generated_normals))
    print(
        f"Trackless dense initialization: {len(source_pcd.points)} source + "
        f"{len(generated_xyz)} Metric3D-generated = {len(xyz)} points"
    )
    return BasicPointCloud(points=xyz, colors=rgb, normals=normals)

def readColmapSceneInfo(path, images, eval, initialization_prior, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    cam_infos_image_id_map = {}
    for i, cam_info in enumerate(cam_infos):
        cam_infos_image_id_map[cam_info.image_id] = i

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if initialization_prior:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        if os.path.exists(txt_path):
            xyz, rgb, _, tracks = read_points3D_text_with_tracks(txt_path)
        elif os.path.exists(bin_path):
            print(f"points3D.txt not found at {txt_path}; reading tracks from points3D.bin instead.")
            xyz, rgb, _, tracks = read_points3D_binary_with_tracks(bin_path)
        else:
            dense_points_path = os.path.join(path, "sparse/0/points3D_all.npy")
            if not os.path.exists(dense_points_path):
                raise FileNotFoundError(
                    "Monocular initialization requires points3D.txt, points3D.bin, "
                    "or a trackless dense points3D_all.npy export"
                )
            pcd = readTracklessDensePrior(
                path, cam_infos, cam_intrinsics, ply_path, initialization_prior
            )
            return SceneInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             ply_path=ply_path)

        # normals = []
        # for track in tracks:
        #     print("track", track)
        #     track_normals = []
        #     for image_id, point_idx in track:
        #         cam_info = cam_infos[image_id - 1]
        #         x, y = cam_extrinsics[image_id].xys[point_idx]
        #         normal_map = cam_info.world_normal_map
        #
        #         track_normal = estimate_subpixel_normal(normal_map, x, y)
        #         track_normals.append(track_normal)
        #
        #     track_normals = np.vstack(track_normals)
        #     mean_normal = np.mean(track_normals, axis=0)
        #     normalized_normal = mean_normal / np.linalg.norm(mean_normal)
        #     normals.append(normalized_normal)

        # Store list of monocular depth prior scaling factors for each image
        image_depth_scale_map = {}
        # Store (x,y) coords of all required points for each image so that remap/interpolation can be done over each image once at the end
        image_xy_map = defaultdict(list)
        for i, track in enumerate(tracks):
            world_coord = xyz[i, :]
            for image_id, point_idx in track:
                x, y = cam_extrinsics[image_id].xys[point_idx]
                image_xy_map[image_id].append((x, y))

                cam_info = cam_infos[cam_infos_image_id_map[image_id]]

                # World to camera transform
                camera_coord = cam_info.R.T @ world_coord + cam_info.T

                fx, fy, cx, cy = cam_intrinsics[cam_info.uid].params
                u = camera_coord[0] / camera_coord[2] * fx + cx
                v = camera_coord[1] / camera_coord[2] * fy + cy

                is_valid = (u >= 0) & (u < cam_info.width) & (v >= 0) & (v < cam_info.height)
                if not is_valid:
                    continue

                # Take floor of pixel coordinate (approximation for simplicity)
                prior_depth = cam_info.depth_map[int(v), int(u)]

                scale = camera_coord[2] / prior_depth
                if image_id in image_depth_scale_map:
                    image_depth_scale_map[image_id]["scales"].append(scale)
                else:
                    image_depth_scale_map[image_id] = {"scales": [scale]}

        # Perform remapping or interpolation
        remapped_normals_per_image = {}
        for image_id, xy_list in image_xy_map.items():
            xy_array = np.array(xy_list, dtype=np.float32)  # shape (N, 2)
            map_x = xy_array[:, 0].reshape(-1, 1)
            map_y = xy_array[:, 1].reshape(-1, 1)

            normal_map = cam_infos[cam_infos_image_id_map[image_id]].world_normal_map  # shape (H, W, 3)

            # remap 3-channel normal map for all x, y at once
            remapped_normals = cv2.remap(
                normal_map, map_x, map_y,
                interpolation=cv2.INTER_LINEAR
            )  # shape (N, 3)

            remapped_normals_per_image[image_id] = remapped_normals

        # Take mean of all normals belonging to a 3D point
        image_counter = defaultdict(int)
        normals = []
        for track in tracks:
            track_normals = []
            for image_id, point_idx in track:
                track_normal = remapped_normals_per_image[image_id][image_counter[image_id], 0]
                track_normal = track_normal / np.linalg.norm(track_normal)
                track_normals.append(track_normal)
                image_counter[image_id] += 1

            track_normals = np.vstack(track_normals)
            mean_normal = np.mean(track_normals, axis=0)
            normalized_normal = mean_normal / np.linalg.norm(mean_normal)
            normals.append(normalized_normal)
        # np.save("./normals_tmp_efficient.npy", np.array(normals))

        normals = np.array(normals)

        print("Normals before:",  normals.shape)

        if initialization_prior == "monocular_depth" or initialization_prior == "monocular_depth_normal":
            for image_id, values in image_depth_scale_map.items():
                image_depth_scale_map[image_id]["estimated_scale"] = np.median(np.array(values["scales"]))
                image_depth_scale_map[image_id]["estimated_scale_variance"] = np.array(values["scales"]).std()

            # num_generated_points = int(xyz.shape[0] * 1)  # 100%, TODO: Make this configurable
            num_generated_points = min(75000, len(cam_infos) * 1000)
            print("Generated points:", num_generated_points)
            generated_xyz, generated_rgb, generated_normals = gaussian_generator(cam_infos, cam_intrinsics, num_generated_points, image_depth_scale_map)

            xyz = np.vstack((xyz, generated_xyz))
            rgb = np.vstack((rgb, generated_rgb))
            normals = np.vstack((normals, generated_normals))

        print("Normals after:", normals.shape)

        storePly(ply_path, xyz, rgb, normals)

        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_id=-1,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
