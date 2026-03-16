import json
import os
from typing import List

import numpy as np

BRANCH_OPTIONS = [
    "Point Head + Pose",
    "Depth Projection + Pose",
]
BRANCH_TO_KEY = {
    "Point Head + Pose": "point_head",
    "Depth Projection + Pose": "depth_projection",
}
DISPLAY_MODE_OPTIONS = [
    "Current Frame",
    "Accumulate to Frame",
    "All Frames",
]


def branch_key(label: str) -> str:
    return BRANCH_TO_KEY.get(label, "point_head")


def session_file(session_dir: str, name: str) -> str:
    return os.path.join(session_dir, name)


def load_metadata(session_dir: str) -> dict:
    with open(session_file(session_dir, "metadata.json"), "r") as f:
        return json.load(f)


def selected_frame_indices(
    num_frames: int, frame_index: int, display_mode: str
) -> List[int]:
    if num_frames <= 0:
        return []
    frame_index = int(np.clip(frame_index, 0, num_frames - 1))
    if display_mode == "Current Frame":
        return [frame_index]
    if display_mode == "Accumulate to Frame":
        return list(range(frame_index + 1))
    return list(range(num_frames))


def as_4x4(w2c):
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.shape == (4, 4):
        return w2c
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = w2c
    return out


_VIEW_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def world_to_view(points):
    points = np.asarray(points, dtype=np.float64)
    return points @ _VIEW_ROT.T


def camera_center_from_w2c(w2c):
    c2w = np.linalg.inv(as_4x4(w2c))
    return c2w[:3, 3]


def c2w_in_view_space(w2c, origin_shift=None):
    c2w = np.linalg.inv(as_4x4(w2c))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _VIEW_ROT @ c2w[:3, :3]
    out[:3, 3] = world_to_view(c2w[:3, 3][None])[0]
    if origin_shift is not None:
        out[:3, 3] -= np.asarray(origin_shift, dtype=np.float64)
    return out
