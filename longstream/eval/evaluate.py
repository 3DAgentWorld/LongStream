import json
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from longstream.data import LongStreamDataLoader
from longstream.eval.io import (
    frame_stems,
    read_depth,
    read_opencv_camera_yml,
    read_pointcloud_xyz,
    read_pred_w2c_txt,
)
from longstream.eval.metrics import ate_rmse, chamfer_and_f1, transform_points
from longstream.utils.sky_mask import sky_mask_filename


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sequence_output_dir(output_root, seq_name):
    return os.path.join(output_root, seq_name)


def _sequence_metrics_path(output_root, seq_name):
    return os.path.join(output_root, "metrics", f"{seq_name}.json")


def _sequence_plot_path(output_root, seq_name):
    return os.path.join(output_root, "plots", f"{seq_name}_traj_3d.png")


def _world_xyz_to_plot_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=-1)


def _set_equal_3d_axes(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-6))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _load_gt_pose_data(seq_info):
    if seq_info.camera is not None:
        cam_dir = os.path.join(seq_info.scene_root, "cameras", seq_info.camera)
        extri_path = os.path.join(cam_dir, "extri.yml")
        intri_path = os.path.join(cam_dir, "intri.yml")
        if os.path.exists(extri_path):
            extri, intri, image_sizes = read_opencv_camera_yml(extri_path, intri_path)
            return extri, intri, image_sizes

    extri_path = os.path.join(seq_info.scene_root, "extri.yml")
    intri_path = os.path.join(seq_info.scene_root, "intri.yml")
    if not os.path.exists(extri_path):
        return None, None, None
    extri, intri, image_sizes = read_opencv_camera_yml(extri_path, intri_path)
    return extri, intri, image_sizes


def _resolve_gt_depth_root(seq_info):
    if seq_info.camera is not None:
        camera_depth_root = os.path.join(seq_info.scene_root, "depths", seq_info.camera)
        if os.path.isdir(camera_depth_root):
            return camera_depth_root
    depth_root = os.path.join(seq_info.scene_root, "depths")
    if os.path.isdir(depth_root):
        return depth_root
    return None


def _resolve_gt_depth_path(seq_info, depth_root, image_path, stem):
    rel_path = os.path.relpath(image_path, seq_info.image_dir)
    rel_stem = os.path.splitext(rel_path)[0]
    file_stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(depth_root, f"{stem}.exr"),
        os.path.join(depth_root, rel_stem + ".exr"),
        os.path.join(depth_root, stem, f"{file_stem}.exr"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _resize_long_edge(arr, long_edge_size, interpolation):
    h, w = arr.shape[:2]
    scale = float(long_edge_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _prepare_map_for_eval(
    arr, size, crop, patch_size, target_shape, interpolation, square_ok=False
):
    h0, w0 = arr.shape[:2]
    long_edge = round(size * max(w0 / h0, h0 / w0)) if size == 224 else size
    arr = _resize_long_edge(arr, long_edge, interpolation)

    h, w = arr.shape[:2]
    cx, cy = w // 2, h // 2

    if size == 224:
        half = min(cx, cy)
        target_w = 2 * half
        target_h = 2 * half
        if crop:
            arr = arr[cy - half : cy + half, cx - half : cx + half]
        else:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=interpolation)
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        target_w = 2 * halfw
        target_h = 2 * halfh
        if crop:
            arr = arr[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=interpolation)

    if arr.shape[:2] != tuple(target_shape):
        arr = cv2.resize(
            arr, (target_shape[1], target_shape[0]), interpolation=interpolation
        )
    return arr


def _sky_mask_path(seq_dir, image_path):
    return os.path.join(seq_dir, "sky_masks", sky_mask_filename(image_path))


def _sample_frame_points(points, max_points, rng):
    if max_points is None or len(points) <= max_points:
        return points
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep]


def _depth_to_world_points(depth, intri, extri, valid_mask):
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float64)
    fx = float(intri[0, 0])
    fy = float(intri[1, 1])
    cx = float(intri[0, 2])
    cy = float(intri[1, 2])

    x = (xs.astype(np.float64) - cx) * z / max(fx, 1e-12)
    y = (ys.astype(np.float64) - cy) * z / max(fy, 1e-12)
    pts_cam = np.stack([x, y, z], axis=1)

    R = extri[:3, :3]
    t = extri[:3, 3]
    pts_world = (R.T @ (pts_cam.T - t[:, None])).T
    return pts_world.astype(np.float32, copy=False)


def _load_gt_pointcloud(seq_info, seq_dir, gt_extri, gt_intri, eval_cfg):
    if not gt_extri or not gt_intri:
        return None

    gt_dir = _resolve_gt_depth_root(seq_info)
    if gt_dir is None:
        return None

    eval_max_points = int(eval_cfg.get("point_eval_max_points", 100000))
    oversample_factor = int(eval_cfg.get("point_eval_oversample_factor", 4))
    per_frame_budget = max(
        (eval_max_points * oversample_factor) // max(len(seq_info.image_paths), 1), 1
    )
    rng = np.random.default_rng(0)
    chunks = []

    for image_path, stem in zip(
        seq_info.image_paths, frame_stems(seq_info.image_paths)
    ):
        depth_path = _resolve_gt_depth_path(seq_info, gt_dir, image_path, stem)
        if depth_path is None or stem not in gt_extri or stem not in gt_intri:
            continue

        depth = read_depth(depth_path)
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            continue

        sky_path = _sky_mask_path(seq_dir, image_path)
        if os.path.exists(sky_path):
            sky_mask = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
            if sky_mask is not None:
                if sky_mask.shape[:2] != depth.shape[:2]:
                    sky_mask = cv2.resize(
                        sky_mask,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                valid &= sky_mask > 0
        if not np.any(valid):
            continue

        pts_world = _depth_to_world_points(depth, gt_intri[stem], gt_extri[stem], valid)
        if len(pts_world) == 0:
            continue
        chunks.append(_sample_frame_points(pts_world, per_frame_budget, rng))

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _evaluate_pointclouds(seq_info, seq_dir, eval_cfg, pose_align, gt_cloud):
    if pose_align is None or gt_cloud is None:
        return None

    scale, R, t = pose_align
    point_paths = {
        "point_head": [
            os.path.join(seq_dir, "points", "point_head_full.npy"),
            os.path.join(seq_dir, "points", "point_head_full.npz"),
            os.path.join(seq_dir, "points", "point_head_full.ply"),
        ],
        "dpt_unproj": [
            os.path.join(seq_dir, "points", "dpt_unproj_full.npy"),
            os.path.join(seq_dir, "points", "dpt_unproj_full.npz"),
            os.path.join(seq_dir, "points", "dpt_unproj_full.ply"),
        ],
    }
    threshold = float(eval_cfg.get("point_f1_threshold", 0.25))
    max_points = int(eval_cfg.get("point_eval_max_points", 100000))
    voxel_size = eval_cfg.get("point_eval_voxel_size", None)
    voxel_size = None if voxel_size in (None, "", "null") else float(voxel_size)

    metrics_by_branch = {}
    for branch, candidates in point_paths.items():
        path = next(
            (candidate for candidate in candidates if os.path.exists(candidate)), None
        )
        if path is None:
            continue
        pred_cloud = read_pointcloud_xyz(path)
        pred_cloud = transform_points(pred_cloud, scale, R, t)
        metrics = chamfer_and_f1(
            pred_cloud,
            gt_cloud,
            threshold=threshold,
            max_points=max_points,
            voxel_size=voxel_size,
            seed=0 if branch == "point_head" else 1,
        )
        if metrics is not None:
            metrics_by_branch[branch] = metrics
    return metrics_by_branch or None


def _evaluate_video_dpt(seq_info, seq_dir, eval_cfg, data_cfg):
    pred_dir = os.path.join(seq_dir, "depth", "dpt")
    gt_dir = _resolve_gt_depth_root(seq_info)
    if not os.path.isdir(pred_dir) or gt_dir is None:
        return None

    size = int(data_cfg.get("size", 518))
    crop = bool(data_cfg.get("crop", False))
    patch_size = int(data_cfg.get("patch_size", 14))
    rel_delta_threshold = float(eval_cfg.get("depth_rel_delta_threshold", 1.25))

    abs_rel_sum = 0.0
    rel_delta_hits = 0
    valid_pixels = 0
    evaluated_frames = 0

    stems = frame_stems(seq_info.image_paths)
    for frame_id, stem in enumerate(stems):
        pred_path = os.path.join(pred_dir, f"frame_{frame_id:06d}.npy")
        gt_path = _resolve_gt_depth_path(
            seq_info, gt_dir, seq_info.image_paths[frame_id], stem
        )
        if not os.path.exists(pred_path) or gt_path is None:
            continue

        pred = np.load(pred_path).astype(np.float32)
        gt = read_depth(gt_path)
        gt = _prepare_map_for_eval(
            gt,
            size=size,
            crop=crop,
            patch_size=patch_size,
            target_shape=pred.shape,
            interpolation=cv2.INTER_NEAREST,
        )

        valid = np.isfinite(gt) & (gt > 0)
        if not np.any(valid):
            continue

        sky_mask_path = _sky_mask_path(seq_dir, seq_info.image_paths[frame_id])
        if os.path.exists(sky_mask_path):
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
            if sky_mask is not None:
                sky_mask = _prepare_map_for_eval(
                    sky_mask,
                    size=size,
                    crop=crop,
                    patch_size=patch_size,
                    target_shape=pred.shape,
                    interpolation=cv2.INTER_NEAREST,
                )
                valid &= sky_mask > 0

        valid &= np.isfinite(pred)
        if not np.any(valid):
            continue

        pred_valid = pred[valid].astype(np.float64)
        gt_valid = gt[valid].astype(np.float64)
        pred_safe = np.clip(pred_valid, 1e-6, None)
        gt_safe = np.clip(gt_valid, 1e-6, None)

        abs_rel_sum += np.sum(np.abs(pred_valid - gt_valid) / gt_safe)
        rel_ratio = np.maximum(gt_safe / pred_safe, pred_safe / gt_safe)
        rel_delta_hits += int(np.sum(rel_ratio < rel_delta_threshold))
        valid_pixels += int(gt_valid.size)
        evaluated_frames += 1

    if valid_pixels == 0:
        return None

    return {
        "abs_rel": float(abs_rel_sum / valid_pixels),
        "rel_delta": float(rel_delta_hits / valid_pixels),
        "rel_delta_threshold": rel_delta_threshold,
        "num_valid_pixels": int(valid_pixels),
        "num_frames": int(evaluated_frames),
    }


def _extract_pose_pairs(seq_info, pred_pose_path, gt_extri):
    frame_ids, pred_w2c = read_pred_w2c_txt(pred_pose_path)
    if not pred_w2c:
        return None

    stems = frame_stems(seq_info.image_paths)
    pred_xyz = []
    gt_xyz = []

    for frame_id, pred_mat in zip(frame_ids, pred_w2c):
        if frame_id < 0 or frame_id >= len(stems):
            continue
        stem = stems[frame_id]
        if stem not in gt_extri:
            continue
        pred_c2w = np.linalg.inv(pred_mat)
        gt_c2w = np.linalg.inv(gt_extri[stem])
        pred_xyz.append(pred_c2w[:3, 3])
        gt_xyz.append(gt_c2w[:3, 3])

    if len(pred_xyz) < 3:
        return None
    return np.asarray(pred_xyz, dtype=np.float64), np.asarray(gt_xyz, dtype=np.float64)


def _save_traj_plot_3d(path, pred_xyz, gt_xyz):
    _ensure_dir(os.path.dirname(path))
    pred_plot = _world_xyz_to_plot_xyz(pred_xyz)
    gt_plot = _world_xyz_to_plot_xyz(gt_xyz)
    origin = gt_plot[:1]
    pred_plot = pred_plot - origin
    gt_plot = gt_plot - origin
    all_plot = np.concatenate([pred_plot, gt_plot], axis=0)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_plot[:, 0],
        gt_plot[:, 1],
        gt_plot[:, 2],
        label="gt",
        linewidth=2.0,
        color="#1f77b4",
    )
    ax.plot(
        pred_plot[:, 0],
        pred_plot[:, 1],
        pred_plot[:, 2],
        label="pred",
        linewidth=2.0,
        color="#d62728",
    )
    _set_equal_3d_axes(ax, all_plot)
    ax.view_init(elev=24, azim=-118)
    ax.set_xlabel("x_right")
    ax.set_ylabel("z_forward")
    ax.set_zlabel("y_up")
    ax.legend(loc="best")
    ax.set_title("Trajectory 3D (Sim3-aligned view)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_sequence(seq_info, output_root, eval_cfg, data_cfg):
    seq_dir = _sequence_output_dir(output_root, seq_info.name)
    result = {
        "sequence": seq_info.name,
        "output_dir": seq_dir,
        "has_gt": False,
        "has_gt_pose": False,
        "has_gt_depth": False,
    }

    gt_extri, gt_intri, _ = _load_gt_pose_data(seq_info)
    pose_align = None
    if gt_extri:
        result["has_gt"] = True
        result["has_gt_pose"] = True

        pred_pose_path = os.path.join(seq_dir, "poses", "abs_pose.txt")
        pairs = _extract_pose_pairs(seq_info, pred_pose_path, gt_extri)
        if pairs is not None:
            pred_xyz, gt_xyz = pairs
            pose_metrics = ate_rmse(
                pred_xyz, gt_xyz, align_scale=bool(eval_cfg.get("align_scale", True))
            )
            sim3_scale = float(pose_metrics.get("sim3_scale", 1.0))
            pred_xyz_aligned = transform_points(
                pred_xyz,
                sim3_scale,
                np.asarray(pose_metrics["sim3_rotation"], dtype=np.float64),
                np.asarray(pose_metrics["sim3_translation"], dtype=np.float64),
            )
            pose_align = (
                sim3_scale,
                np.asarray(pose_metrics["sim3_rotation"], dtype=np.float64),
                np.asarray(pose_metrics["sim3_translation"], dtype=np.float64),
            )
            plot_path = _sequence_plot_path(output_root, seq_info.name)
            _save_traj_plot_3d(plot_path, pred_xyz_aligned, gt_xyz)
            pose_metrics.pop("sim3_scale", None)
            pose_metrics["traj_3d_plot"] = plot_path
            result["pose"] = pose_metrics

    video_dpt_metrics = _evaluate_video_dpt(seq_info, seq_dir, eval_cfg, data_cfg)
    if video_dpt_metrics is not None:
        result["has_gt"] = True
        result["has_gt_depth"] = True
        result["video_dpt"] = video_dpt_metrics

    gt_cloud = _load_gt_pointcloud(seq_info, seq_dir, gt_extri, gt_intri, eval_cfg)
    pointcloud_metrics = _evaluate_pointclouds(
        seq_info, seq_dir, eval_cfg, pose_align, gt_cloud
    )
    if pointcloud_metrics is not None:
        result["has_gt"] = True
        result["has_gt_depth"] = True
        result["pointcloud"] = pointcloud_metrics

    if not result["has_gt"]:
        result["skipped"] = "missing_gt"

    return result


def _mean_metric(sequence_results, group_name, metric_name):
    values = []
    for item in sequence_results:
        group = item
        for key in group_name.split("."):
            if not isinstance(group, dict):
                group = None
                break
            group = group.get(key)
        if not isinstance(group, dict):
            continue
        if metric_name in group:
            values.append(float(group[metric_name]))
    if not values:
        return None
    return float(np.mean(values))


def evaluate_predictions_cfg(cfg):
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["format"] = "generalizable"
    output_cfg = cfg.get("output", {})
    eval_cfg = cfg.get("evaluation", {})
    output_root = output_cfg.get("root", "outputs")
    _ensure_dir(output_root)

    loader = LongStreamDataLoader(data_cfg)
    sequence_results = []
    for seq_info in loader.iter_sequence_infos():
        print(f"[longstream] eval {seq_info.name}: start", flush=True)
        metrics = evaluate_sequence(seq_info, output_root, eval_cfg, data_cfg)
        sequence_results.append(metrics)
        metrics_path = _sequence_metrics_path(output_root, seq_info.name)
        _ensure_dir(os.path.dirname(metrics_path))
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[longstream] eval {seq_info.name}: wrote {metrics_path}", flush=True)

    summary = {
        "num_sequences": len(sequence_results),
        "num_sequences_with_gt": sum(1 for x in sequence_results if x.get("has_gt")),
        "num_sequences_with_pose_gt": sum(
            1 for x in sequence_results if x.get("has_gt_pose")
        ),
        "num_sequences_with_depth_gt": sum(
            1 for x in sequence_results if x.get("has_gt_depth")
        ),
        "ate_mean": _mean_metric(sequence_results, "pose", "ate_mean"),
        "ate_rmse_mean": _mean_metric(sequence_results, "pose", "ate_rmse"),
        "video_dpt_abs_rel_mean": _mean_metric(
            sequence_results, "video_dpt", "abs_rel"
        ),
        "video_dpt_rel_delta_mean": _mean_metric(
            sequence_results, "video_dpt", "rel_delta"
        ),
        "point_head_cd_mean": _mean_metric(
            sequence_results, "pointcloud.point_head", "cd"
        ),
        "point_head_f1_mean": _mean_metric(
            sequence_results, "pointcloud.point_head", "f1"
        ),
        "dpt_unproj_cd_mean": _mean_metric(
            sequence_results, "pointcloud.dpt_unproj", "cd"
        ),
        "dpt_unproj_f1_mean": _mean_metric(
            sequence_results, "pointcloud.dpt_unproj", "f1"
        ),
        "sequences": sequence_results,
    }

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[longstream] eval: wrote {summary_path}", flush=True)
    return summary
