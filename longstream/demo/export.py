import os

import numpy as np

from .geometry import camera_geometry, collect_points

_CAMERA_COLORS = np.array(
    [
        [239, 68, 68, 255],
        [14, 165, 233, 255],
        [34, 197, 94, 255],
        [245, 158, 11, 255],
    ],
    dtype=np.uint8,
)


def _camera_mesh(center, corners, color):
    import trimesh

    vertices = np.vstack([center[None], corners]).astype(np.float32)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 2, 3],
            [1, 3, 4],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(color[None], (faces.shape[0], 1))
    return mesh


def export_glb(
    session_dir: str,
    branch: str,
    display_mode: str,
    frame_index: int,
    mask_sky: bool,
    show_cameras: bool,
    camera_scale: float,
    max_points: int,
) -> str:
    import trimesh

    points, colors, _ = collect_points(
        session_dir=session_dir,
        branch=branch,
        display_mode=display_mode,
        frame_index=frame_index,
        mask_sky=mask_sky,
        max_points=max_points,
        seed=13,
    )
    if len(points) == 0:
        raise ValueError("No valid points to export")

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))

    if show_cameras:
        _, frustums, _ = camera_geometry(
            session_dir=session_dir,
            display_mode=display_mode,
            frame_index=frame_index,
            camera_scale_ratio=camera_scale,
            points_hint=points,
        )
        for idx, (center, corners) in enumerate(frustums):
            scene.add_geometry(
                _camera_mesh(center, corners, _CAMERA_COLORS[idx % len(_CAMERA_COLORS)])
            )

    export_dir = os.path.join(session_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    branch_slug = branch.lower().replace(" + ", "_").replace(" ", "_")
    mode_slug = display_mode.replace(" ", "_").lower()
    filename = f"{branch_slug}_{mode_slug}_{frame_index:04d}_sky{int(mask_sky)}_cam{int(show_cameras)}.glb"
    path = os.path.join(export_dir, filename)
    scene.export(path)
    return path
