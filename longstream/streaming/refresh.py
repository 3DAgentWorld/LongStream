import torch
from typing import Dict, Any, List

from longstream.streaming.stream_session import StreamSession

_SEQUENCE_OUTPUT_KEYS = {
    "pose_enc",
    "rel_pose_enc",
    "world_points",
    "world_points_conf",
    "depth",
    "depth_conf",
}
_SCALAR_OUTPUT_KEYS = {
    "predicted_scale_factor",
    "global_scale",
}


def _refresh_intervals(refresh: int) -> int:
    refresh = int(refresh)
    if refresh < 2:
        raise ValueError("refresh must be >= 2")
    return refresh - 1


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _move_scalar_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _append_batch_output(
    stitched_tensors: Dict[str, List[torch.Tensor]],
    stitched_scalars: Dict[str, Any],
    output: Dict[str, Any],
    actual_frames: int,
    slice_start: int,
) -> None:
    for key in _SEQUENCE_OUTPUT_KEYS:
        value = output.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2 or value.shape[1] != actual_frames:
            continue
        stitched_tensors.setdefault(key, []).append(
            value[:, slice_start:].detach().cpu()
        )

    for key in _SCALAR_OUTPUT_KEYS:
        if key in output:
            stitched_scalars[key] = _move_scalar_to_cpu(output[key])


def _finalize_stitched_batches(
    stitched_tensors: Dict[str, List[torch.Tensor]],
    stitched_scalars: Dict[str, Any],
) -> Dict[str, Any]:
    stitched_output: Dict[str, Any] = {}
    for key, chunks in stitched_tensors.items():
        if not chunks:
            continue
        stitched_output[key] = (
            chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        )
    stitched_output.update(stitched_scalars)
    return stitched_output


def run_batch_refresh(
    model,
    images,
    is_keyframe,
    keyframe_indices,
    mode: str,
    keyframe_stride: int,
    refresh: int,
    rel_pose_cfg,
):
    B, S = images.shape[:2]
    device = _model_device(model)
    refresh_intervals = _refresh_intervals(refresh)
    frames_per_batch = refresh_intervals * keyframe_stride + 1
    step_frames = refresh_intervals * keyframe_stride

    stitched_tensors: Dict[str, List[torch.Tensor]] = {}
    stitched_scalars: Dict[str, Any] = {}
    num_batches = (S + step_frames - 1) // step_frames
    for batch_idx in range(num_batches):
        start_frame = batch_idx * step_frames
        end_frame = min(start_frame + frames_per_batch, S)
        batch_images = images[:, start_frame:end_frame].to(device, non_blocking=True)
        batch_is_keyframe = (
            is_keyframe[:, start_frame:end_frame].clone()
            if is_keyframe is not None
            else None
        )
        batch_keyframe_indices = (
            keyframe_indices[:, start_frame:end_frame].clone()
            if keyframe_indices is not None
            else None
        )

        if batch_idx > 0 and batch_is_keyframe is not None:
            batch_is_keyframe[:, 0] = True
            if batch_keyframe_indices is not None:
                batch_keyframe_indices[:, 0] = start_frame

        if batch_keyframe_indices is not None:
            batch_keyframe_indices = batch_keyframe_indices - start_frame
            batch_keyframe_indices = torch.clamp(
                batch_keyframe_indices, 0, end_frame - start_frame - 1
            )

        batch_rel_pose_inputs = None
        if rel_pose_cfg is not None and batch_is_keyframe is not None:
            batch_is_keyframe = batch_is_keyframe.to(device, non_blocking=True)
            if batch_keyframe_indices is not None:
                batch_keyframe_indices = batch_keyframe_indices.to(
                    device, non_blocking=True
                )
            batch_rel_pose_inputs = {
                "is_keyframe": batch_is_keyframe,
                "keyframe_indices": batch_keyframe_indices,
                "num_iterations": rel_pose_cfg.get("num_iterations", 4),
            }
        elif batch_is_keyframe is not None:
            batch_is_keyframe = batch_is_keyframe.to(device, non_blocking=True)

        batch_output = model(
            images=batch_images,
            mode=mode,
            rel_pose_inputs=batch_rel_pose_inputs,
            is_keyframe=batch_is_keyframe,
        )

        _append_batch_output(
            stitched_tensors,
            stitched_scalars,
            batch_output,
            actual_frames=end_frame - start_frame,
            slice_start=0 if batch_idx == 0 else 1,
        )
        del batch_output
        del batch_images
        del batch_is_keyframe
        del batch_keyframe_indices

    return _finalize_stitched_batches(stitched_tensors, stitched_scalars)


def run_streaming_refresh(
    model,
    images,
    is_keyframe,
    keyframe_indices,
    mode: str,
    window_size: int,
    refresh: int,
    rel_pose_cfg,
):
    B, S = images.shape[:2]
    device = _model_device(model)
    refresh_intervals = _refresh_intervals(refresh)
    session = StreamSession(model, mode=mode, window_size=window_size)
    keyframe_count = 0
    segment_start = 0
    for s in range(S):
        frame_images = images[:, s : s + 1].to(device, non_blocking=True)
        is_keyframe_s = (
            is_keyframe[:, s : s + 1].to(device, non_blocking=True)
            if is_keyframe is not None
            else None
        )
        if keyframe_indices is not None:
            keyframe_indices_s = keyframe_indices[:, s : s + 1].clone() - segment_start
            keyframe_indices_s = torch.clamp(keyframe_indices_s, min=0)
            keyframe_indices_s = keyframe_indices_s.to(device, non_blocking=True)
        else:
            keyframe_indices_s = None
        session.forward_stream(
            frame_images,
            is_keyframe=is_keyframe_s,
            keyframe_indices=keyframe_indices_s,
            record=True,
        )
        if is_keyframe_s is None or not bool(is_keyframe_s.item()) or s <= 0:
            del frame_images
            if is_keyframe_s is not None:
                del is_keyframe_s
            if keyframe_indices_s is not None:
                del keyframe_indices_s
            continue
        keyframe_count += 1
        if keyframe_count % refresh_intervals == 0:
            session.clear_cache_only()
            segment_start = s
            if keyframe_indices_s is not None:
                keyframe_indices_self = torch.zeros_like(keyframe_indices_s)
            else:
                keyframe_indices_self = None
            session.forward_stream(
                frame_images,
                is_keyframe=is_keyframe_s,
                keyframe_indices=keyframe_indices_self,
                record=False,
            )
        del frame_images
        if is_keyframe_s is not None:
            del is_keyframe_s
        if keyframe_indices_s is not None:
            del keyframe_indices_s
    return session.get_all_predictions()
