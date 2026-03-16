import os
from typing import List, Optional, Tuple

import numpy as np

from .common import (
    branch_key,
    c2w_in_view_space,
    load_metadata,
    selected_frame_indices,
    session_file,
    world_to_view,
)


def _origin_shift(w2c_all) -> np.ndarray:
    first = c2w_in_view_space(w2c_all[0])
    return first[:3, 3].copy()


def _sample_flat_indices(
    valid_indices: np.ndarray, budget: Optional[int], rng: np.random.Generator
) -> np.ndarray:
    if budget is None or budget <= 0 or valid_indices.size <= budget:
        return valid_indices
    keep = rng.choice(valid_indices.size, size=int(budget), replace=False)
    return valid_indices[keep]


def _depth_points_from_flat(depth, intri, w2c, flat_indices):
    h, w = depth.shape
    ys = flat_indices // w
    xs = flat_indices % w
    z = depth.reshape(-1)[flat_indices].astype(np.float64)
    fx = float(intri[0, 0])
    fy = float(intri[1, 1])
    cx = float(intri[0, 2])
    cy = float(intri[1, 2])
    x = (xs.astype(np.float64) - cx) * z / max(fx, 1e-12)
    y = (ys.astype(np.float64) - cy) * z / max(fy, 1e-12)
    pts_cam = np.stack([x, y, z], axis=1)
    R = w2c[:3, :3].astype(np.float64)
    t = w2c[:3, 3].astype(np.float64)
    return (R.T @ (pts_cam.T - t[:, None])).T.astype(np.float32, copy=False)


def _camera_points_to_world(points, w2c):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = w2c[:3, :3].astype(np.float64)
    t = w2c[:3, 3].astype(np.float64)
    return (R.T @ (pts.T - t[:, None])).T.astype(np.float32, copy=False)


def collect_points(
    session_dir: str,
    branch: str,
    display_mode: str,
    frame_index: int,
    mask_sky: bool,
    max_points: Optional[int],
    seed: int = 0,
):
    branch = branch_key(branch)
    meta = load_metadata(session_dir)
    frame_ids = selected_frame_indices(meta["num_frames"], frame_index, display_mode)
    if not frame_ids:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.zeros(3, dtype=np.float64),
        )

    images = np.load(session_file(session_dir, "images.npy"), mmap_mode="r")
    w2c = np.load(session_file(session_dir, "w2c.npy"), mmap_mode="r")
    origin_shift = _origin_shift(w2c)
    sky = None
    if mask_sky and os.path.exists(session_file(session_dir, "sky_masks.npy")):
        sky = np.load(session_file(session_dir, "sky_masks.npy"), mmap_mode="r")

    if branch == "point_head":
        point_head = np.load(session_file(session_dir, "point_head.npy"), mmap_mode="r")
        source = point_head
        depth = None
        intri = None
    else:
        source = None
        depth = np.load(session_file(session_dir, "depth.npy"), mmap_mode="r")
        intri = np.load(session_file(session_dir, "intri.npy"), mmap_mode="r")

    per_frame_budget = None
    if max_points is not None and max_points > 0:
        per_frame_budget = max(int(max_points) // max(len(frame_ids), 1), 1)

    rng = np.random.default_rng(seed)
    points = []
    colors = []
    for idx in frame_ids:
        rgb_flat = images[idx].reshape(-1, 3)
        if branch == "point_head":
            pts_map = source[idx]
            valid = np.isfinite(pts_map).all(axis=-1).reshape(-1)
            if sky is not None:
                valid &= sky[idx].reshape(-1) > 0
            flat = np.flatnonzero(valid)
            if flat.size == 0:
                continue
            flat = _sample_flat_indices(flat, per_frame_budget, rng)
            pts_cam = pts_map.reshape(-1, 3)[flat]
            pts_world = _camera_points_to_world(pts_cam, w2c[idx])
        else:
            depth_i = depth[idx]
            valid = (np.isfinite(depth_i) & (depth_i > 0)).reshape(-1)
            if sky is not None:
                valid &= sky[idx].reshape(-1) > 0
            flat = np.flatnonzero(valid)
            if flat.size == 0:
                continue
            flat = _sample_flat_indices(flat, per_frame_budget, rng)
            pts_world = _depth_points_from_flat(depth_i, intri[idx], w2c[idx], flat)

        pts_view = world_to_view(pts_world) - origin_shift[None]
        points.append(pts_view.astype(np.float32, copy=False))
        colors.append(rgb_flat[flat].astype(np.uint8, copy=False))

    if not points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            origin_shift,
        )
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0), origin_shift


def _frustum_corners_camera(intri, image_hw, depth_scale):
    h, w = image_hw
    fx = float(intri[0, 0])
    fy = float(intri[1, 1])
    cx = float(intri[0, 2])
    cy = float(intri[1, 2])
    corners = np.array(
        [
            [
                (0.0 - cx) * depth_scale / max(fx, 1e-12),
                (0.0 - cy) * depth_scale / max(fy, 1e-12),
                depth_scale,
            ],
            [
                ((w - 1.0) - cx) * depth_scale / max(fx, 1e-12),
                (0.0 - cy) * depth_scale / max(fy, 1e-12),
                depth_scale,
            ],
            [
                ((w - 1.0) - cx) * depth_scale / max(fx, 1e-12),
                ((h - 1.0) - cy) * depth_scale / max(fy, 1e-12),
                depth_scale,
            ],
            [
                (0.0 - cx) * depth_scale / max(fx, 1e-12),
                ((h - 1.0) - cy) * depth_scale / max(fy, 1e-12),
                depth_scale,
            ],
        ],
        dtype=np.float64,
    )
    return corners


def camera_geometry(
    session_dir: str,
    display_mode: str,
    frame_index: int,
    camera_scale_ratio: float,
    points_hint=None,
):
    meta = load_metadata(session_dir)
    frame_ids = selected_frame_indices(meta["num_frames"], frame_index, display_mode)
    w2c = np.load(session_file(session_dir, "w2c.npy"), mmap_mode="r")
    intri = np.load(session_file(session_dir, "intri.npy"), mmap_mode="r")
    origin_shift = _origin_shift(w2c)

    center_points = np.array(
        [c2w_in_view_space(w2c[idx], origin_shift)[:3, 3] for idx in frame_ids],
        dtype=np.float64,
    )
    center_extent = 1.0
    if len(center_points) > 1:
        center_extent = float(
            np.linalg.norm(center_points.max(axis=0) - center_points.min(axis=0))
        )

    point_extent = 0.0
    if points_hint is not None and len(points_hint) > 0:
        lo = np.percentile(points_hint, 5, axis=0)
        hi = np.percentile(points_hint, 95, axis=0)
        point_extent = float(np.linalg.norm(hi - lo))

    extent = max(center_extent, point_extent, 1.0)
    depth_scale = extent * float(camera_scale_ratio)

    centers = []
    frustums = []
    for idx in frame_ids:
        c2w_view = c2w_in_view_space(w2c[idx], origin_shift)
        center = c2w_view[:3, 3]
        corners_cam = _frustum_corners_camera(
            intri[idx], (meta["height"], meta["width"]), depth_scale
        )
        corners_world = (c2w_view[:3, :3] @ corners_cam.T).T + center[None]
        centers.append(center)
        frustums.append((center, corners_world))
    return np.asarray(centers, dtype=np.float64), frustums, origin_shift
