import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from longstream.core.cli import default_config_path
from longstream.core.model import LongStreamModel
from longstream.streaming.keyframe_selector import KeyframeSelector
from longstream.streaming.refresh import run_batch_refresh, run_streaming_refresh
from longstream.utils.camera import compose_abs_from_rel
from longstream.utils.depth import colorize_depth
from longstream.utils.hub import resolve_checkpoint_path
from longstream.utils.sky_mask import compute_sky_mask
from longstream.utils.vendor.dust3r.utils.image import load_images_for_eval
from longstream.utils.vendor.models.components.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

from .common import load_metadata, session_file

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_MODEL_CACHE = {}


def _resolve_file_path(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and "name" in item:
        return item["name"]
    if hasattr(item, "name"):
        return item.name
    return str(item)


def _natural_sort_key(path: str):
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    parts = re.split(r"(\d+)", stem)
    key = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key, name.lower()


def _sorted_image_paths(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(_IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    return sorted(files, key=_natural_sort_key)


def _session_root() -> str:
    root = os.path.join(tempfile.gettempdir(), "longstream_demo_sessions")
    os.makedirs(root, exist_ok=True)
    return root


def _new_session_dir() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return tempfile.mkdtemp(prefix=f"longstream_{stamp}_", dir=_session_root())


def _copy_uploaded_images(uploaded_files: Iterable, session_dir: str) -> List[str]:
    input_dir = os.path.join(session_dir, "input_images")
    os.makedirs(input_dir, exist_ok=True)
    copied = []
    sources = sorted(
        (_resolve_file_path(x) for x in uploaded_files if x),
        key=_natural_sort_key,
    )
    for src in sources:
        if not src or not os.path.isfile(src):
            continue
        dst = os.path.join(input_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _extract_uploaded_video(uploaded_video, session_dir: str) -> List[str]:
    src = _resolve_file_path(uploaded_video)
    if not src:
        return []
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    input_dir = os.path.join(session_dir, "input_images")
    os.makedirs(input_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"unable to open video: {src}")

    image_paths = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dst = os.path.join(input_dir, f"{frame_id:06d}.png")
        if not cv2.imwrite(dst, frame):
            cap.release()
            raise ValueError(f"failed to write extracted frame: {dst}")
        image_paths.append(dst)
        frame_id += 1
    cap.release()

    if not image_paths:
        raise ValueError(f"no frames extracted from video: {src}")
    return image_paths


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
    h0, w0 = mask.shape[:2]
    long_edge = round(size * max(w0 / h0, h0 / w0)) if size == 224 else size
    mask = _resize_long_edge(mask, long_edge, cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        if crop:
            mask = mask[cy - half : cy + half, cx - half : cx + half]
        else:
            mask = cv2.resize(
                mask, (2 * half, 2 * half), interpolation=cv2.INTER_NEAREST
            )
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        if crop:
            mask = mask[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            mask = cv2.resize(
                mask, (2 * halfw, 2 * halfh), interpolation=cv2.INTER_NEAREST
            )

    if mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask.astype(np.uint8, copy=False)


def _load_base_config(config_path: Optional[str] = None) -> dict:
    path = config_path or default_config_path()
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_demo_checkpoint(checkpoint: str) -> str:
    local_candidates = []
    for candidate in [checkpoint, os.getenv("LONGSTREAM_CHECKPOINT", "")]:
        if isinstance(candidate, str) and candidate:
            local_candidates.append(candidate)

    for candidate in local_candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    hf_cfg = {
        "repo_id": os.getenv("LONGSTREAM_HF_REPO", "NicolasCC/LongStream"),
        "filename": os.getenv("LONGSTREAM_HF_FILE", "50_longstream.pt"),
        "revision": os.getenv("LONGSTREAM_HF_REVISION"),
        "local_dir": os.getenv("LONGSTREAM_HF_LOCAL_DIR", "checkpoints"),
    }
    resolved = resolve_checkpoint_path(None, hf_cfg)
    if resolved and os.path.exists(resolved):
        return os.path.abspath(resolved)

    if hf_cfg["repo_id"] and hf_cfg["filename"]:
        raise FileNotFoundError(
            "checkpoint not found locally and Hugging Face resolution failed: "
            f"repo_id={hf_cfg['repo_id']} filename={hf_cfg['filename']}"
        )

    searched = ", ".join(local_candidates) if local_candidates else "<none>"
    raise FileNotFoundError(
        "checkpoint not found. "
        f"searched local paths: {searched}. "
        "You can also set LONGSTREAM_HF_REPO and LONGSTREAM_HF_FILE."
    )


def _model_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _cache_key(checkpoint: str, device: str, model_cfg: dict) -> Tuple[str, str, str]:
    rel_cfg = json.dumps(model_cfg.get("longstream_cfg", {}), sort_keys=True)
    return checkpoint, device, rel_cfg


def get_or_load_model(checkpoint: str, device: str, model_cfg: dict) -> LongStreamModel:
    device = _model_device(device)
    cfg = json.loads(json.dumps(model_cfg))
    cfg["checkpoint"] = checkpoint
    key = _cache_key(checkpoint, device, cfg)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = LongStreamModel(cfg).to(device)
        model.eval()
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model
    return model


def _load_images(
    image_paths: List[str], size: int, crop: bool, patch_size: int
) -> torch.Tensor:
    views = load_images_for_eval(
        image_paths, size=size, verbose=False, crop=crop, patch_size=patch_size
    )
    imgs = torch.cat([view["img"] for view in views], dim=0)
    images = (imgs.unsqueeze(0) + 1.0) / 2.0
    return images


def _select_keyframes(images: torch.Tensor, keyframe_stride: int, keyframe_mode: str):
    selector = KeyframeSelector(
        min_interval=keyframe_stride,
        max_interval=keyframe_stride,
        force_first=True,
        mode="random" if keyframe_mode == "random" else "fixed",
    )
    return selector.select_keyframes(images.shape[1], images.shape[0], images.device)


def _run_model(images: torch.Tensor, model: LongStreamModel, infer_cfg: dict):
    keyframe_stride = int(infer_cfg.get("keyframe_stride", 8))
    keyframe_mode = infer_cfg.get("keyframe_mode", "fixed")
    refresh = int(infer_cfg.get("refresh", 4))
    mode = infer_cfg.get("mode", "streaming_refresh")
    streaming_mode = infer_cfg.get("streaming_mode", "causal")
    window_size = int(infer_cfg.get("window_size", 48))
    rel_pose_cfg = infer_cfg.get("rel_pose_head_cfg", {"num_iterations": 4})

    is_keyframe, keyframe_indices = _select_keyframes(
        images, keyframe_stride, keyframe_mode
    )
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
        raise ValueError(f"Unsupported demo inference mode: {mode}")
    return outputs, keyframe_indices


def _compute_pose_outputs(
    outputs: dict, keyframe_indices: torch.Tensor, image_hw: Tuple[int, int]
):
    if "rel_pose_enc" in outputs:
        rel_pose_enc = outputs["rel_pose_enc"][0]
        abs_pose_enc = compose_abs_from_rel(rel_pose_enc, keyframe_indices[0])
        extri, intri = pose_encoding_to_extri_intri(
            abs_pose_enc[None], image_size_hw=image_hw
        )
        return (
            rel_pose_enc.detach().cpu().numpy(),
            extri[0].detach().cpu().numpy(),
            intri[0].detach().cpu().numpy(),
        )
    if "pose_enc" in outputs:
        pose_enc = outputs["pose_enc"][0]
        extri, intri = pose_encoding_to_extri_intri(
            pose_enc[None], image_size_hw=image_hw
        )
        return None, extri[0].detach().cpu().numpy(), intri[0].detach().cpu().numpy()
    raise RuntimeError("Model outputs contain neither rel_pose_enc nor pose_enc")


def _compute_sky_masks(
    image_paths: List[str],
    target_shape: Tuple[int, int],
    data_cfg: dict,
    skyseg_path: str,
    session_dir: str,
):
    raw_masks = compute_sky_mask(
        image_paths, skyseg_path, os.path.join(session_dir, "sky_masks_raw")
    )
    if raw_masks is None:
        return None
    masks = []
    for mask in raw_masks:
        masks.append(
            _prepare_mask_for_model(
                mask,
                size=int(data_cfg.get("size", 518)),
                crop=bool(data_cfg.get("crop", False)),
                patch_size=int(data_cfg.get("patch_size", 14)),
                target_shape=target_shape,
            )
        )
    return np.stack(masks, axis=0)


def create_demo_session(
    image_dir: str,
    uploaded_files,
    uploaded_video,
    checkpoint: str,
    device: str,
    mode: str,
    streaming_mode: str,
    keyframe_stride: int,
    refresh: int,
    window_size: int,
    compute_sky: bool,
    config_path: Optional[str] = None,
) -> str:
    checkpoint = _resolve_demo_checkpoint(checkpoint)

    session_dir = _new_session_dir()
    base_cfg = _load_base_config(config_path)
    data_cfg = dict(base_cfg.get("data", {}))
    model_cfg = dict(base_cfg.get("model", {}))
    infer_cfg = dict(base_cfg.get("inference", {}))

    if image_dir:
        image_dir = os.path.abspath(image_dir)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        image_paths = _sorted_image_paths(image_dir)
        input_root = image_dir
    elif uploaded_video:
        image_paths = _extract_uploaded_video(uploaded_video, session_dir)
        input_root = _resolve_file_path(uploaded_video)
    else:
        image_paths = _copy_uploaded_images(uploaded_files or [], session_dir)
        input_root = os.path.dirname(image_paths[0]) if image_paths else ""

    if not image_paths:
        raise ValueError("No input images found")

    data_cfg["size"] = int(data_cfg.get("size", 518))
    data_cfg["crop"] = bool(data_cfg.get("crop", False))
    data_cfg["patch_size"] = int(data_cfg.get("patch_size", 14))

    device = _model_device(device)
    model = get_or_load_model(checkpoint, device, model_cfg)

    images = _load_images(
        image_paths, data_cfg["size"], data_cfg["crop"], data_cfg["patch_size"]
    )
    infer_cfg.update(
        {
            "mode": mode,
            "streaming_mode": streaming_mode,
            "keyframe_stride": int(keyframe_stride),
            "refresh": int(refresh),
            "window_size": int(window_size),
        }
    )

    with torch.no_grad():
        outputs, keyframe_indices = _run_model(images, model, infer_cfg)
        h, w = images.shape[-2:]
        rel_pose_enc, extri, intri = _compute_pose_outputs(
            outputs, keyframe_indices, (h, w)
        )
        point_head = (
            outputs["world_points"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        depth = (
            outputs["depth"][0, :, :, :, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    if device == "cuda":
        torch.cuda.empty_cache()

    images_uint8 = np.clip(
        images[0].permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
    ).astype(np.uint8)
    sky_masks = None
    if compute_sky:
        skyseg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "skyseg.onnx"
        )
        sky_masks = _compute_sky_masks(
            image_paths, (h, w), data_cfg, skyseg_path, session_dir
        )

    np.save(session_file(session_dir, "images.npy"), images_uint8)
    np.save(session_file(session_dir, "depth.npy"), depth)
    np.save(session_file(session_dir, "point_head.npy"), point_head)
    np.save(session_file(session_dir, "w2c.npy"), extri)
    np.save(session_file(session_dir, "intri.npy"), intri)
    if rel_pose_enc is not None:
        np.save(
            session_file(session_dir, "rel_pose_enc.npy"),
            rel_pose_enc.astype(np.float32, copy=False),
        )
    if sky_masks is not None:
        np.save(
            session_file(session_dir, "sky_masks.npy"),
            sky_masks.astype(np.uint8, copy=False),
        )

    sky_removed_ratio = None
    if sky_masks is not None:
        sky_removed_ratio = float(1.0 - (sky_masks > 0).mean())

    metadata = {
        "session_dir": session_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint": os.path.abspath(checkpoint),
        "device": device,
        "mode": mode,
        "streaming_mode": streaming_mode,
        "keyframe_stride": int(keyframe_stride),
        "refresh": int(refresh),
        "window_size": int(window_size),
        "num_frames": int(images_uint8.shape[0]),
        "height": int(images_uint8.shape[1]),
        "width": int(images_uint8.shape[2]),
        "input_root": input_root,
        "image_paths": image_paths,
        "has_sky_masks": bool(sky_masks is not None),
        "sky_removed_ratio": sky_removed_ratio,
    }
    with open(session_file(session_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del outputs
    return session_dir


def load_frame_previews(session_dir: str, frame_index: int):
    meta = load_metadata(session_dir)
    frame_index = int(np.clip(frame_index, 0, meta["num_frames"] - 1))
    images = np.load(session_file(session_dir, "images.npy"), mmap_mode="r")
    depth = np.load(session_file(session_dir, "depth.npy"), mmap_mode="r")
    rgb = np.array(images[frame_index])
    depth_color = colorize_depth(np.array(depth[frame_index]), cmap="plasma")
    label = f"Frame {frame_index + 1}/{meta['num_frames']}"
    return rgb, depth_color, label
