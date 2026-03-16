import os

import gradio as gr

from longstream.demo import BRANCH_OPTIONS, create_demo_session, load_metadata
from longstream.demo.backend import load_frame_previews
from longstream.demo.export import export_glb
from longstream.demo.viewer import build_interactive_figure


DEFAULT_KEYFRAME_STRIDE = 8
DEFAULT_REFRESH = 3
DEFAULT_WINDOW_SIZE = 48
DEFAULT_CHECKPOINT = os.getenv("LONGSTREAM_CHECKPOINT", "checkpoints/50_longstream.pt")


def _run_stable_demo(
    image_dir,
    uploaded_files,
    uploaded_video,
    checkpoint,
    device,
    mode,
    streaming_mode,
    refresh,
    window_size,
    compute_sky,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    if not image_dir and not uploaded_files and not uploaded_video:
        raise gr.Error("Provide an image folder, upload images, or upload a video.")
    session_dir = create_demo_session(
        image_dir=image_dir or "",
        uploaded_files=uploaded_files,
        uploaded_video=uploaded_video,
        checkpoint=checkpoint,
        device=device,
        mode=mode,
        streaming_mode=streaming_mode,
        keyframe_stride=DEFAULT_KEYFRAME_STRIDE,
        refresh=int(refresh),
        window_size=int(window_size),
        compute_sky=bool(compute_sky),
    )
    fig = build_interactive_figure(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )
    glb_path = export_glb(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )
    rgb, depth, frame_label = load_frame_previews(session_dir, 0)
    meta = load_metadata(session_dir)
    slider = gr.update(
        minimum=0,
        maximum=max(meta["num_frames"] - 1, 0),
        value=0,
        step=1,
        interactive=meta["num_frames"] > 1,
    )
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    status = f"Ready: {meta['num_frames']} frames | branch={branch_label}{sky_msg}"
    return (
        fig,
        glb_path,
        session_dir,
        rgb,
        depth,
        frame_label,
        slider,
        status,
    )


def _update_stable_scene(
    session_dir,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    if not session_dir or not os.path.isdir(session_dir):
        return None, None, "Run reconstruction first."
    fig = build_interactive_figure(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )
    glb_path = export_glb(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )
    meta = load_metadata(session_dir)
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    return fig, glb_path, f"Updated preview: {branch_label}{sky_msg}"


def _update_frame_preview(session_dir, frame_index):
    if not session_dir or not os.path.isdir(session_dir):
        return None, None, ""
    rgb, depth, label = load_frame_previews(session_dir, int(frame_index))
    return rgb, depth, label


def main():
    with gr.Blocks(title="LongStream Demo") as demo:
        session_dir = gr.Textbox(visible=False)

        gr.Markdown("# LongStream Demo")

        with gr.Row():
            image_dir = gr.Textbox(
                label="Image Folder", placeholder="/path/to/sequence"
            )
            uploaded_files = gr.File(
                label="Upload Images", file_count="multiple", file_types=["image"]
            )
            uploaded_video = gr.File(
                label="Upload Video", file_count="single", file_types=["video"]
            )

        with gr.Row():
            checkpoint = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
            device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda")

        with gr.Accordion("Inference", open=False):
            with gr.Row():
                mode = gr.Dropdown(
                    label="Mode",
                    choices=["streaming_refresh", "batch_refresh"],
                    value="batch_refresh",
                )
                streaming_mode = gr.Dropdown(
                    label="Streaming Mode", choices=["causal", "window"], value="causal"
                )
            with gr.Row():
                refresh = gr.Slider(
                    label="Refresh", minimum=2, maximum=9, step=1, value=DEFAULT_REFRESH
                )
                window_size = gr.Slider(
                    label="Window Size",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=DEFAULT_WINDOW_SIZE,
                )
                compute_sky = gr.Checkbox(label="Compute Sky Masks", value=True)

        with gr.Accordion("GLB Settings", open=True):
            with gr.Row():
                branch_label = gr.Dropdown(
                    label="Point Cloud Branch",
                    choices=BRANCH_OPTIONS,
                    value="Point Head + Pose",
                )
                show_cameras = gr.Checkbox(label="Show Cameras", value=True)
                mask_sky = gr.Checkbox(label="Mask Sky", value=True)
            with gr.Row():
                point_size = gr.Slider(
                    label="Point Size",
                    minimum=0.05,
                    maximum=2.0,
                    step=0.05,
                    value=0.3,
                )
                opacity = gr.Slider(
                    label="Opacity",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.75,
                )
                preview_max_points = gr.Slider(
                    label="Preview Max Points",
                    minimum=5000,
                    maximum=1000000,
                    step=10000,
                    value=100000,
                )
            with gr.Row():
                camera_scale = gr.Slider(
                    label="Camera Scale",
                    minimum=0.001,
                    maximum=0.05,
                    step=0.001,
                    value=0.01,
                )
                glb_max_points = gr.Slider(
                    label="GLB Max Points",
                    minimum=20000,
                    maximum=1000000,
                    step=10000,
                    value=400000,
                )

        run_btn = gr.Button("Run Stable Demo", variant="primary")
        status = gr.Markdown("Provide input images, then run reconstruction.")

        plot = gr.Plot(label="Scene Preview")

        glb_file = gr.File(label="Download GLB")

        with gr.Row():
            frame_slider = gr.Slider(
                label="Preview Frame",
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                interactive=False,
            )
            frame_label = gr.Textbox(label="Frame")
        with gr.Row():
            rgb_preview = gr.Image(label="RGB", type="numpy")
            depth_preview = gr.Image(label="Depth Plasma", type="numpy")

        run_btn.click(
            _run_stable_demo,
            inputs=[
                image_dir,
                uploaded_files,
                uploaded_video,
                checkpoint,
                device,
                mode,
                streaming_mode,
                refresh,
                window_size,
                compute_sky,
                branch_label,
                show_cameras,
                mask_sky,
                camera_scale,
                point_size,
                opacity,
                preview_max_points,
                glb_max_points,
            ],
            outputs=[
                plot,
                glb_file,
                session_dir,
                rgb_preview,
                depth_preview,
                frame_label,
                frame_slider,
                status,
            ],
        )

        for component in [
            branch_label,
            show_cameras,
            mask_sky,
            camera_scale,
            point_size,
            opacity,
            preview_max_points,
            glb_max_points,
        ]:
            component.change(
                _update_stable_scene,
                inputs=[
                    session_dir,
                    branch_label,
                    show_cameras,
                    mask_sky,
                    camera_scale,
                    point_size,
                    opacity,
                    preview_max_points,
                    glb_max_points,
                ],
                outputs=[plot, glb_file, status],
            )

        frame_slider.change(
            _update_frame_preview,
            inputs=[session_dir, frame_slider],
            outputs=[rgb_preview, depth_preview, frame_label],
        )

    demo.launch()


if __name__ == "__main__":
    main()
