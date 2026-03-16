import numpy as np
from scipy.spatial import cKDTree


def similarity_align(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Expected Nx3 source and target point sets")
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var = np.mean(np.sum(src_centered ** 2, axis=1))
        scale = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    else:
        scale = 1.0
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def transform_points(points, scale, R, t):
    return (scale * (R @ points.T)).T + t[None]


def ate_rmse(pred_xyz, gt_xyz, align_scale=True):
    scale, R, t = similarity_align(pred_xyz, gt_xyz, with_scale=align_scale)
    pred_aligned = transform_points(pred_xyz, scale, R, t)
    err = np.linalg.norm(pred_aligned - gt_xyz, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean": float(np.mean(err)),
        "ate_median": float(np.median(err)),
        "num_pose_pairs": int(len(err)),
        "align_scale": bool(align_scale),
        "sim3_scale": float(scale),
        "sim3_rotation": R.tolist(),
        "sim3_translation": t.tolist(),
    }


def _voxel_downsample(points, voxel_size):
    if voxel_size is None:
        return points
    voxel_size = float(voxel_size)
    if voxel_size <= 0 or len(points) == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(coords, axis=0, return_index=True)
    keep.sort()
    return points[keep]


def _sample_points(points, max_points, seed):
    if max_points is None or len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=int(max_points), replace=False)
    return points[keep]


def prepare_pointcloud(points, max_points=None, voxel_size=None, seed=0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    points = _voxel_downsample(points, voxel_size)
    points = _sample_points(points, max_points, seed)
    return points


def chamfer_and_f1(
    pred_points, gt_points, threshold=0.25, max_points=None, voxel_size=None, seed=0
):
    pred = prepare_pointcloud(
        pred_points, max_points=max_points, voxel_size=voxel_size, seed=seed
    )
    gt = prepare_pointcloud(
        gt_points, max_points=max_points, voxel_size=voxel_size, seed=seed + 1
    )
    if len(pred) == 0 or len(gt) == 0:
        return None

    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    dist_pred_to_gt, _ = gt_tree.query(pred, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt, k=1)

    acc = float(np.mean(dist_pred_to_gt))
    comp = float(np.mean(dist_gt_to_pred))
    precision = float(np.mean(dist_pred_to_gt < threshold))
    recall = float(np.mean(dist_gt_to_pred < threshold))
    denom = precision + recall
    f1 = 0.0 if denom <= 0 else float(2.0 * precision * recall / denom)
    return {
        "cd": float(acc + comp),
        "acc": acc,
        "comp": comp,
        "f1": f1,
        "f1_threshold": float(threshold),
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
    }
