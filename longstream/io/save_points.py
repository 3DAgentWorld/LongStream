import os
import numpy as np


def _maybe_downsample(points, colors=None, max_points=None, seed=0):
    pts = np.asarray(points).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(-1, 3)
    if max_points is None or pts.shape[0] <= int(max_points):
        return pts, cols
    rng = np.random.default_rng(seed)
    keep = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]
    return pts, cols


def save_pointcloud(path, points, colors=None, max_points=None, seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts, cols = _maybe_downsample(
        points, colors=colors, max_points=max_points, seed=seed
    )
    pts = pts.astype(np.float32, copy=False)
    if colors is not None:
        if cols.max() <= 1.0:
            cols = (cols * 255.0).astype(np.uint8)
        else:
            cols = cols.astype(np.uint8)
        has_color = True
    else:
        cols = None
        has_color = False

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n".encode("ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if has_color:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        if has_color:
            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
            vertex_data = np.empty(pts.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = pts[:, 0]
            vertex_data["y"] = pts[:, 1]
            vertex_data["z"] = pts[:, 2]
            vertex_data["red"] = cols[:, 0]
            vertex_data["green"] = cols[:, 1]
            vertex_data["blue"] = cols[:, 2]
            vertex_data.tofile(f)
        else:
            vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            vertex_data = np.empty(pts.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = pts[:, 0]
            vertex_data["y"] = pts[:, 1]
            vertex_data["z"] = pts[:, 2]
            vertex_data.tofile(f)
