from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from longstream.utils.vendor.dust3r.utils.misc import freeze_all_params
from longstream.utils.vendor.models.components.aggregator.streamaggregator import (
    STreamAggregator,
)
from longstream.utils.vendor.models.components.heads.camera_head import (
    CameraHead,
    RelPoseHead,
)
from longstream.utils.vendor.models.components.heads.dpt_head import DPTHead


class LongStream(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        freeze="none",
        rel_pose_head_cfg=None,
        use_role_embedding=True,
        enable_scale_token=False,
        scale_token_config=None,
        disable_keyframe_distinction=False,
        enable_camera_head=True,
        use_segment_mask=False,
        use_3d_rope=False,
        rope_freq=100,
        window_size=5000,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.enable_scale_token = enable_scale_token
        self.enable_camera_head = enable_camera_head
        self.window_size = window_size

        self.aggregator = STreamAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_role_embedding=use_role_embedding,
            disable_keyframe_distinction=disable_keyframe_distinction,
            use_segment_mask=use_segment_mask,
            use_3d_rope=use_3d_rope,
            rope_freq=rope_freq,
            window_size=window_size,
        )

        if self.enable_camera_head:
            self.camera_head = CameraHead(dim_in=2 * embed_dim, window_size=window_size)
        else:
            self.camera_head = None
        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
        )
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )

        self.rel_pose_head = None
        self.reinit_camera_head_when_rel_enabled = False

        if rel_pose_head_cfg is not None:
            enable = rel_pose_head_cfg.get("enabled", True)
            if enable:

                head_cfg = {
                    "dim_in": 2 * embed_dim,
                    "trunk_depth": rel_pose_head_cfg.get("trunk_depth", 4),
                    "pose_mode": rel_pose_head_cfg.get("pose_mode", "SE3"),
                    "num_heads": rel_pose_head_cfg.get("num_heads", 16),
                    "mlp_ratio": rel_pose_head_cfg.get("mlp_ratio", 4),
                    "init_values": rel_pose_head_cfg.get("init_values", 0.01),
                    "trans_act": rel_pose_head_cfg.get("trans_act", "linear"),
                    "quat_act": rel_pose_head_cfg.get("quat_act", "linear"),
                    "fl_act": rel_pose_head_cfg.get("fl_act", "relu"),
                    "use_global_scale": rel_pose_head_cfg.get(
                        "use_global_scale", False
                    ),
                    "use_pair_cross_attn": rel_pose_head_cfg.get(
                        "use_pair_cross_attn", False
                    ),
                    "detach_reference": rel_pose_head_cfg.get(
                        "detach_reference", False
                    ),
                    "xattn_temperature": rel_pose_head_cfg.get(
                        "xattn_temperature", 1.0
                    ),
                    "use_precat": rel_pose_head_cfg.get("use_precat", False),
                    "use_kf_role_embed": rel_pose_head_cfg.get(
                        "use_kf_role_embed", True
                    ),
                    "kf_role_embed_init_std": rel_pose_head_cfg.get(
                        "kf_role_embed_init_std", 0.02
                    ),
                    "window_size": window_size,
                }
                self.rel_pose_head = RelPoseHead(**head_cfg)

                self.reinit_camera_head_when_rel_enabled = rel_pose_head_cfg.get(
                    "reinit_camera_head", False
                )

                if self.reinit_camera_head_when_rel_enabled:
                    pass

        if self.enable_scale_token:
            self._init_scale_components(scale_token_config or {})

        self.set_freeze(freeze)

    def reinitialize_camera_head(self):
        """
        Reinitialize camera_head with fresh weights.

        This is useful when:
        1. Loading a pretrained checkpoint that has camera_head weights
        2. But we want to train camera_head from scratch with new settings (e.g., quaternion normalization)

        This method should be called AFTER checkpoint loading.
        """

        old_camera_head = self.camera_head
        dim_in = old_camera_head.token_norm.normalized_shape[0]

        self.camera_head = CameraHead(dim_in=dim_in)

        device = next(old_camera_head.parameters()).device
        self.camera_head = self.camera_head.to(device)

    def _init_scale_components(self, config):
        self.scale_token = nn.Parameter(torch.zeros(self.embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        self.scale_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        for m in self.scale_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        import math

        nn.init.constant_(self.scale_head[-1].bias, math.log(30.0))

    def set_freeze(self, freeze):
        self.freeze = freeze

        to_be_frozen = {
            "none": [],
            "encoder": [self.aggregator.patch_embed],
        }
        freeze_all_params(to_be_frozen[freeze])

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        aggregator_kv_cache_list: Optional[List[List[torch.Tensor]]] = None,
        camera_head_kv_cache_list: Optional[List[List[List[torch.Tensor]]]] = None,
        rel_pose_inputs: Optional[Dict] = None,
        is_keyframe: Optional[torch.Tensor] = None,
    ):

        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]

        additional_tokens = None
        if self.enable_scale_token:

            scale_token_base = self.scale_token.unsqueeze(0).repeat(batch_size, 1)
            additional_tokens = scale_token_base.unsqueeze(-1)

        keyframe_indices = None
        if rel_pose_inputs is not None and "keyframe_indices" in rel_pose_inputs:
            keyframe_indices = rel_pose_inputs["keyframe_indices"]

        if aggregator_kv_cache_list is not None:
            (
                aggregated_tokens_list,
                patch_start_idx,
                aggregator_kv_cache_list,
                _,
            ) = self.aggregator(
                images,
                mode=mode,
                kv_cache_list=aggregator_kv_cache_list,
                is_keyframe=is_keyframe,
                keyframe_indices=keyframe_indices,
                additional_tokens=additional_tokens,
                reorder_keyframes_first=False,
            )
        else:
            aggregated_tokens_list, patch_start_idx, _ = self.aggregator(
                images,
                mode=mode,
                is_keyframe=is_keyframe,
                keyframe_indices=keyframe_indices,
                additional_tokens=additional_tokens,
                reorder_keyframes_first=False,
            )

        predictions = {}

        predicted_scale_factor = None
        if self.enable_scale_token and additional_tokens is not None:

            if len(aggregated_tokens_list) > 0:
                last_layer_features = aggregated_tokens_list[-1]

                scale_token_idx = patch_start_idx - 1
                scale_token_output_features = last_layer_features[
                    :, :, scale_token_idx, :
                ]

                scale_token_output_features = scale_token_output_features.mean(dim=1)

                scale_logits = self.scale_head(scale_token_output_features).squeeze(-1)

                predicted_scale_factor = torch.exp(scale_logits)

                predictions["predicted_scale_factor"] = predicted_scale_factor
                predictions["scale_token_features"] = scale_token_output_features

        if self.enable_camera_head and self.camera_head is not None:
            if camera_head_kv_cache_list is not None:
                pose_enc_list, camera_head_kv_cache_list = self.camera_head(
                    aggregated_tokens_list,
                    mode=mode,
                    kv_cache_list=camera_head_kv_cache_list,
                )
            else:
                pose_enc_list = self.camera_head(aggregated_tokens_list, mode=mode)

            final_pose_enc = pose_enc_list[-1]
            if self.enable_scale_token and predicted_scale_factor is not None:
                scale = predicted_scale_factor.view(-1, 1, 1)

                scaled_t = final_pose_enc[..., :3] * scale
                scaled_pose_enc = torch.cat([scaled_t, final_pose_enc[..., 3:]], dim=-1)
                predictions["pose_enc"] = scaled_pose_enc
            else:
                predictions["pose_enc"] = final_pose_enc

            if self.training:

                if self.enable_scale_token and predicted_scale_factor is not None:
                    scale = predicted_scale_factor.view(-1, 1, 1)
                    scaled_pose_enc_list = []
                    for pose_enc in pose_enc_list:

                        scaled_t = pose_enc[..., :3] * scale
                        scaled_pose_enc = torch.cat(
                            [scaled_t, pose_enc[..., 3:]], dim=-1
                        )
                        scaled_pose_enc_list.append(scaled_pose_enc)
                    predictions["pose_enc_list"] = scaled_pose_enc_list
                else:
                    predictions["pose_enc_list"] = pose_enc_list

        if self.rel_pose_head is not None and rel_pose_inputs is not None:

            rel_kwargs = dict(
                aggregated_tokens_list=aggregated_tokens_list,
                keyframe_indices=rel_pose_inputs.get("keyframe_indices"),
                is_keyframe=rel_pose_inputs.get("is_keyframe", is_keyframe),
                num_iterations=rel_pose_inputs.get("num_iterations", 4),
                mode=mode,
                kv_cache_list=rel_pose_inputs.get("kv_cache_list"),
            )

            rel_kwargs = {k: v for k, v in rel_kwargs.items() if v is not None}

            rel_result = self.rel_pose_head(**rel_kwargs)

            if isinstance(rel_result, dict):

                pose_enc = rel_result["pose_enc"]
                if pose_enc.dtype != torch.float32:
                    pose_enc = pose_enc.float()

                if self.enable_scale_token and predicted_scale_factor is not None:
                    scale = predicted_scale_factor.view(-1, 1, 1)

                    scaled_t = pose_enc[..., :3] * scale
                    scaled_rel_pose_enc = torch.cat(
                        [scaled_t, pose_enc[..., 3:]], dim=-1
                    )
                    predictions["rel_pose_enc"] = scaled_rel_pose_enc

                    if "pose_enc_list" in rel_result:
                        scaled_pose_enc_list = []
                        for iter_pose in rel_result["pose_enc_list"]:
                            scaled_t = iter_pose[..., :3] * scale
                            scaled_iter_pose = torch.cat(
                                [scaled_t, iter_pose[..., 3:]], dim=-1
                            )
                            scaled_pose_enc_list.append(scaled_iter_pose)
                        predictions["rel_pose_enc_list"] = scaled_pose_enc_list
                else:
                    predictions["rel_pose_enc"] = pose_enc

                    if "pose_enc_list" in rel_result:
                        predictions["rel_pose_enc_list"] = rel_result["pose_enc_list"]

                predictions["is_keyframe"] = rel_result.get("is_keyframe")
                predictions["keyframe_indices"] = rel_result.get("keyframe_indices")

                if "global_scale" in rel_result:
                    predictions["global_scale"] = rel_result["global_scale"]

            if "kv_cache_list" in rel_result:
                predictions["rel_pose_kv_cache_list"] = rel_result["kv_cache_list"]

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

            if self.enable_scale_token and predicted_scale_factor is not None:
                scale = predicted_scale_factor.view(-1, 1, 1, 1, 1)
                predictions["world_points"] = pts3d * scale
            else:
                predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

            if self.enable_scale_token and predicted_scale_factor is not None:
                scale = predicted_scale_factor.view(-1, 1, 1, 1, 1)
                predictions["depth"] = depth * scale
            else:
                predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if aggregator_kv_cache_list is not None:
            predictions["aggregator_kv_cache_list"] = aggregator_kv_cache_list

        if camera_head_kv_cache_list is not None:
            predictions["camera_head_kv_cache_list"] = camera_head_kv_cache_list

        if not self.training:
            predictions["images"] = images

        return predictions
