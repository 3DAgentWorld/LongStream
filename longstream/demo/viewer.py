import numpy as np
import plotly.graph_objects as go

from longstream.demo.backend import load_frame_previews

from .common import load_metadata
from .geometry import camera_geometry, collect_points


def _empty_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data"),
    )
    return fig


def _camera_lines(frustums):
    xs, ys, zs = [], [], []
    for center, corners in frustums:
        order = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for a, b in order:
            xs.extend([corners[a, 0], corners[b, 0], None])
            ys.extend([corners[a, 1], corners[b, 1], None])
            zs.extend([corners[a, 2], corners[b, 2], None])
        for corner in corners:
            xs.extend([center[0], corner[0], None])
            ys.extend([center[1], corner[1], None])
            zs.extend([center[2], corner[2], None])
    return xs, ys, zs


def build_interactive_figure(
    session_dir: str,
    branch: str,
    display_mode: str,
    frame_index: int,
    point_size: float,
    opacity: float,
    preview_max_points: int,
    show_cameras: bool,
    camera_scale: float,
    mask_sky: bool,
):
    meta = load_metadata(session_dir)
    points, colors, _ = collect_points(
        session_dir=session_dir,
        branch=branch,
        display_mode=display_mode,
        frame_index=frame_index,
        mask_sky=mask_sky,
        max_points=preview_max_points,
        seed=frame_index,
    )
    if len(points) == 0:
        return _empty_figure("No valid points for the current selection")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=float(point_size),
                color=[f"rgb({r},{g},{b})" for r, g, b in colors],
                opacity=float(opacity),
            ),
            hoverinfo="skip",
            name="points",
        )
    )

    if show_cameras:
        centers, frustums, _ = camera_geometry(
            session_dir=session_dir,
            display_mode=display_mode,
            frame_index=frame_index,
            camera_scale_ratio=camera_scale,
            points_hint=points,
        )
        if len(centers) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    z=centers[:, 2],
                    mode="lines",
                    line=dict(color="#16a34a", width=2),
                    name="trajectory",
                    hoverinfo="skip",
                )
            )
            xs, ys, zs = _camera_lines(frustums)
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="#22c55e", width=1.5),
                    name="cameras",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data",
            xaxis_title="x_right",
            yaxis_title="z_forward",
            zaxis_title="y_up",
            bgcolor="#f8fafc",
            camera=dict(
                up=dict(x=0.0, y=0.0, z=1.0),
                eye=dict(x=-1.0, y=-1.8, z=0.9),
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def build_frame_outputs(session_dir: str, frame_index: int):
    rgb, depth, label = load_frame_previews(session_dir, frame_index)
    return rgb, depth, label
