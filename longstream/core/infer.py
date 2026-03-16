import argparse
import os
import yaml
import cv2
import numpy as np
import torch
from PIL import Image

from longstream.core.model import LongStreamModel
from longstream.data.dataloader import LongStreamDataLoader
from longstream.streaming.keyframe_selector import KeyframeSelector
from longstream.streaming.refresh import run_batch_refresh, run_streaming_refresh
from longstream.utils.vendor.models.components.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from longstream.utils.camera import compose_abs_from_rel
from longstream.utils.depth import colorize_depth, unproject_depth_to_points
from longstream.utils.sky_mask import compute_sky_mask
from longstream.io.save_points import save_pointcloud
from longstream.io.save_poses_txt import save_w2c_txt, save_intri_txt, save_rel_pose_txt
from longstream.io.save_images import save_image_sequence, save_video


def _to_uint8_rgb(images):
    imgs = images.detach().cpu().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255.0).astype(np.uint8)
    return imgs


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _apply_sky_mask(depth, mask):
    if mask is None:
        return depth
    m = (mask > 0).astype(np.float32)
    return depth * m


def _camera_points_to_world(points, extri):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(extri[:3, :3], dtype=np.float64)
    t = np.asarray(extri[:3, 3], dtype=np.float64)
    world = (R.T @ (pts.T - t[:, None])).T
    return world.astype(np.float32, copy=False)


def _mask_points_and_colors(points, colors, mask):
    pts = points.reshape(-1, 3)
    cols = None if colors is None else colors.reshape(-1, 3)
    if mask is None:
        return pts, cols
    valid = mask.reshape(-1) > 0
    pts = pts[valid]
    if cols is not None:
        cols = cols[valid]
    return pts, cols


def _resize_long_edge(arr, long_edge_size, interpolation):
    h, w = arr.shape[:2]
    scale = float(long_edge_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _prepare_mask_for_model(
    mask, size, crop, patch_size, target_shape, square_ok=False
):
    if mask is None:
        return None
    long_edge = (
        round(size * max(mask.shape[1] / mask.shape[0], mask.shape[0] / mask.shape[1]))
        if size == 224
        else size
    )
    mask = _resize_long_edge(mask, long_edge, cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        target_w = 2 * half
        target_h = 2 * half
        if crop:
            mask = mask[cy - half : cy + half, cx - half : cx + half]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        target_w = 2 * halfw
        target_h = 2 * halfh
        if crop:
            mask = mask[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

    if mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask


def _save_full_pointcloud(path, point_chunks, color_chunks, max_points=None, seed=0):
    if not point_chunks:
        return
    points = np.concatenate(point_chunks, axis=0)
    colors = None
    if color_chunks and len(color_chunks) == len(point_chunks):
        colors = np.concatenate(color_chunks, axis=0)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    np.save(os.path.splitext(path)[0] + ".npy", points.astype(np.float32, copy=False))
    save_pointcloud(path, points, colors=colors, max_points=None, seed=seed)


def run_inference_cfg(cfg: dict):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    output_cfg = cfg.get("output", {})

    print(f"[longstream] device={device}", flush=True)
    model = LongStreamModel(model_cfg).to(device)
    model.eval()
    print("[longstream] model ready", flush=True)

    loader = LongStreamDataLoader(data_cfg)

    keyframe_stride = int(infer_cfg.get("keyframe_stride", 8))
    keyframe_mode = infer_cfg.get("keyframe_mode", "fixed")
    refresh = int(
        infer_cfg.get("refresh", int(infer_cfg.get("keyframes_per_batch", 3)) + 1)
    )
    if refresh < 2:
        raise ValueError(
            "refresh must be >= 2 because it counts both keyframe endpoints"
        )
    mode = infer_cfg.get("mode", "streaming_refresh")
    if mode == "streaming":
        mode = "streaming_refresh"
    streaming_mode = infer_cfg.get("streaming_mode", "causal")
    window_size = int(infer_cfg.get("window_size", 5))

    selector = KeyframeSelector(
        min_interval=keyframe_stride,
        max_interval=keyframe_stride,
        force_first=True,
        mode="random" if keyframe_mode == "random" else "fixed",
    )

    out_root = output_cfg.get("root", "outputs")
    _ensure_dir(out_root)
    save_videos = bool(output_cfg.get("save_videos", True))
    save_points = bool(output_cfg.get("save_points", True))
    save_frame_points = bool(output_cfg.get("save_frame_points", True))
    save_depth = bool(output_cfg.get("save_depth", True))
    save_images = bool(output_cfg.get("save_images", True))
    mask_sky = bool(output_cfg.get("mask_sky", True))
    max_full_pointcloud_points = output_cfg.get("max_full_pointcloud_points", None)
    if max_full_pointcloud_points is not None:
        max_full_pointcloud_points = int(max_full_pointcloud_points)
    max_frame_pointcloud_points = output_cfg.get("max_frame_pointcloud_points", None)
    if max_frame_pointcloud_points is not None:
        max_frame_pointcloud_points = int(max_frame_pointcloud_points)
    skyseg_path = output_cfg.get(
        "skyseg_path",
        os.path.join(os.path.dirname(__file__), "..", "..", "skyseg.onnx"),
    )

    with torch.no_grad():
        for seq in loader:
            images = seq.images
            B, S, C, H, W = images.shape
            print(
                f"[longstream] sequence {seq.name}: inference start ({S} frames)",
                flush=True,
            )

            is_keyframe, keyframe_indices = selector.select_keyframes(
                S, B, images.device
            )

            rel_pose_cfg = infer_cfg.get("rel_pose_head_cfg", {"num_iterations": 4})

            if mode == "batch_refresh":
                outputs = run_batch_refresh(
                    model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    streaming_mode,
                    keyframe_stride,
                    refresh,
                    rel_pose_cfg,
                )
            elif mode == "streaming_refresh":
                outputs = run_streaming_refresh(
                    model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    streaming_mode,
                    window_size,
                    refresh,
                    rel_pose_cfg,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")
            print(f"[longstream] sequence {seq.name}: inference done", flush=True)
            if device_type == "cuda":
                torch.cuda.empty_cache()

            seq_dir = os.path.join(out_root, seq.name)
            _ensure_dir(seq_dir)

            frame_ids = list(range(S))
            rgb = _to_uint8_rgb(images[0].permute(0, 2, 3, 1))

            if "rel_pose_enc" in outputs:
                rel_pose_enc = outputs["rel_pose_enc"][0]
                abs_pose_enc = compose_abs_from_rel(rel_pose_enc, keyframe_indices[0])
                extri, intri = pose_encoding_to_extri_intri(
                    abs_pose_enc[None], image_size_hw=(H, W)
                )
                extri_np = extri[0].detach().cpu().numpy()
                intri_np = intri[0].detach().cpu().numpy()

                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose.txt"), extri_np, frame_ids
                )
                save_intri_txt(os.path.join(pose_dir, "intri.txt"), intri_np, frame_ids)
                save_rel_pose_txt(
                    os.path.join(pose_dir, "rel_pose.txt"), rel_pose_enc, frame_ids
                )
            elif "pose_enc" in outputs:
                pose_enc = outputs["pose_enc"][0]
                extri, intri = pose_encoding_to_extri_intri(
                    pose_enc[None], image_size_hw=(H, W)
                )
                extri_np = extri[0].detach().cpu().numpy()
                intri_np = intri[0].detach().cpu().numpy()

                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose.txt"), extri_np, frame_ids
                )
                save_intri_txt(os.path.join(pose_dir, "intri.txt"), intri_np, frame_ids)

            if save_images:
                print(f"[longstream] sequence {seq.name}: saving rgb", flush=True)
                rgb_dir = os.path.join(seq_dir, "images", "rgb")
                save_image_sequence(rgb_dir, list(rgb))
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "images", "rgb.mp4"),
                        os.path.join(rgb_dir, "frame_*.png"),
                    )

            sky_masks = None
            if mask_sky:
                raw_sky_masks = compute_sky_mask(
                    seq.image_paths, skyseg_path, os.path.join(seq_dir, "sky_masks")
                )
                if raw_sky_masks is not None:
                    sky_masks = [
                        _prepare_mask_for_model(
                            mask,
                            size=int(data_cfg.get("size", 518)),
                            crop=bool(data_cfg.get("crop", False)),
                            patch_size=int(data_cfg.get("patch_size", 14)),
                            target_shape=(H, W),
                        )
                        for mask in raw_sky_masks
                    ]

            if save_depth and "depth" in outputs:
                print(f"[longstream] sequence {seq.name}: saving depth", flush=True)
                depth = outputs["depth"][0, :, :, :, 0].detach().cpu().numpy()
                depth_dir = os.path.join(seq_dir, "depth", "dpt")
                _ensure_dir(depth_dir)
                color_dir = os.path.join(seq_dir, "depth", "dpt_plasma")
                _ensure_dir(color_dir)

                color_frames = []
                for i in range(S):
                    d = depth[i]
                    if sky_masks is not None and sky_masks[i] is not None:
                        d = _apply_sky_mask(d, sky_masks[i])
                    np.save(os.path.join(depth_dir, f"frame_{i:06d}.npy"), d)
                    colored = colorize_depth(d, cmap="plasma")
                    Image.fromarray(colored).save(
                        os.path.join(color_dir, f"frame_{i:06d}.png")
                    )
                    color_frames.append(colored)
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "depth", "dpt_plasma.mp4"),
                        os.path.join(color_dir, "frame_*.png"),
                    )

            if save_points:
                print(
                    f"[longstream] sequence {seq.name}: saving point clouds", flush=True
                )
                if "world_points" in outputs:
                    if "rel_pose_enc" in outputs:
                        abs_pose_enc = compose_abs_from_rel(
                            outputs["rel_pose_enc"][0], keyframe_indices[0]
                        )
                        extri, intri = pose_encoding_to_extri_intri(
                            abs_pose_enc[None], image_size_hw=(H, W)
                        )
                    else:
                        extri, intri = pose_encoding_to_extri_intri(
                            outputs["pose_enc"][0][None], image_size_hw=(H, W)
                        )
                    extri = extri[0]
                    intri = intri[0]

                    pts_dir = os.path.join(seq_dir, "points", "point_head")
                    _ensure_dir(pts_dir)
                    pts = outputs["world_points"][0].detach().cpu().numpy()
                    full_pts = []
                    full_cols = []
                    for i in range(S):
                        pts_world = _camera_points_to_world(
                            pts[i], extri[i].detach().cpu().numpy()
                        )
                        pts_world = pts_world.reshape(pts[i].shape)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            None if sky_masks is None else sky_masks[i],
                        )
                        if save_frame_points:
                            save_pointcloud(
                                os.path.join(pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    _save_full_pointcloud(
                        os.path.join(seq_dir, "points", "point_head_full.ply"),
                        full_pts,
                        full_cols,
                        max_points=max_full_pointcloud_points,
                        seed=0,
                    )

                if "depth" in outputs and (
                    "rel_pose_enc" in outputs or "pose_enc" in outputs
                ):
                    depth = outputs["depth"][0, :, :, :, 0]
                    if "rel_pose_enc" in outputs:
                        abs_pose_enc = compose_abs_from_rel(
                            outputs["rel_pose_enc"][0], keyframe_indices[0]
                        )
                        extri, intri = pose_encoding_to_extri_intri(
                            abs_pose_enc[None], image_size_hw=(H, W)
                        )
                    else:
                        extri, intri = pose_encoding_to_extri_intri(
                            outputs["pose_enc"][0][None], image_size_hw=(H, W)
                        )

                    extri = extri[0]
                    intri = intri[0]
                    dpt_pts_dir = os.path.join(seq_dir, "points", "dpt_unproj")
                    _ensure_dir(dpt_pts_dir)
                    full_pts = []
                    full_cols = []

                    for i in range(S):
                        d = depth[i]
                        pts_cam = unproject_depth_to_points(d[None], intri[i : i + 1])[
                            0
                        ]
                        R = extri[i, :3, :3]
                        t = extri[i, :3, 3]
                        pts_world = (
                            R.t() @ (pts_cam.reshape(-1, 3).t() - t[:, None])
                        ).t()
                        pts_world = pts_world.cpu().numpy().reshape(-1, 3)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            None if sky_masks is None else sky_masks[i],
                        )
                        if save_frame_points:
                            save_pointcloud(
                                os.path.join(dpt_pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    _save_full_pointcloud(
                        os.path.join(seq_dir, "points", "dpt_unproj_full.ply"),
                        full_pts,
                        full_cols,
                        max_points=max_full_pointcloud_points,
                        seed=1,
                    )
            del outputs
            if device_type == "cuda":
                torch.cuda.empty_cache()


def run_inference(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_inference_cfg(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
