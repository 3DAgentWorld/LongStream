import os
import torch
from typing import Dict, Any

from longstream.models.longstream import LongStream
from longstream.utils.hub import resolve_checkpoint_path


class LongStreamModel(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any] | None):
        super().__init__()
        cfg = cfg or {}

        ckpt_path = resolve_checkpoint_path(
            cfg.get("checkpoint", None), cfg.get("hf", None)
        )

        stream_cfg = dict(cfg.get("longstream_cfg", {}) or {})
        rel_pose_cfg = stream_cfg.pop(
            "rel_pose_head_cfg", cfg.get("rel_pose_head_cfg", None)
        )
        use_rel_pose_head = bool(stream_cfg.pop("use_rel_pose_head", False))
        if use_rel_pose_head and rel_pose_cfg is not None:
            stream_cfg["rel_pose_head_cfg"] = rel_pose_cfg
        self.longstream = LongStream(**stream_cfg)

        if ckpt_path:
            self.load_checkpoint(ckpt_path, strict=bool(cfg.get("strict_load", True)))

    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            raise TypeError("Unsupported checkpoint format")

        if state:
            first_key = next(iter(state.keys()))
            if first_key.startswith("sampler.longstream."):
                state = {k.replace("sampler.", "", 1): v for k, v in state.items()}

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            msg = f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            if strict:
                raise RuntimeError(msg)
            print(msg)

    def forward(self, *args, **kwargs):
        return self.longstream(*args, **kwargs)

    @property
    def aggregator(self):
        return self.longstream.aggregator

    @property
    def camera_head(self):
        return getattr(self.longstream, "camera_head", None)

    @property
    def rel_pose_head(self):
        return getattr(self.longstream, "rel_pose_head", None)
