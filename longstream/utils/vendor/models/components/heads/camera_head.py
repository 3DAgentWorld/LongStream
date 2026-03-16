from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

from ..layers import Mlp
from ..layers.block import Block
from .head_act import activate_pose


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        window_size: int = 5,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.window_size = window_size

        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True)
        )

        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def _create_attn_mask(
        self, S: int, mode: str, dtype: torch.dtype, device: torch.device
    ) -> Optional[torch.Tensor]:
        N = S
        mask = torch.zeros((N, N), dtype=dtype, device=device)

        if mode == "causal":
            for i in range(S):
                curr_view_start = i
                curr_view_end = i + 1
                mask[curr_view_start:curr_view_end, curr_view_end:] = float("-inf")
        elif mode == "window":
            for i in range(S):
                curr_view_start = i
                curr_view_end = i + 1
                mask[curr_view_start:curr_view_end, 1:] = float("-inf")
                start_view = max(1, i - self.window_size + 1)
                mask[curr_view_start:curr_view_end, start_view : (i + 1)] = 0
        elif mode == "full":
            mask = None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        return mask

    def forward(
        self,
        aggregated_tokens_list: list,
        num_iterations: int = 4,
        mode: str = "causal",
        kv_cache_list: Optional[List[List[List[torch.Tensor]]]] = None,
    ) -> Union[list, Tuple[list, List[List[List[torch.Tensor]]]]]:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for
                each iterations and each attention layer of the camera head

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """

        tokens = aggregated_tokens_list[-1]

        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        B, S, C = pose_tokens.shape
        attn_mask = None
        if kv_cache_list is None:
            attn_mask = self._create_attn_mask(
                S, mode, pose_tokens.dtype, pose_tokens.device
            )

        pred_pose_enc_list = self.trunk_fn(
            pose_tokens, num_iterations, attn_mask, kv_cache_list
        )
        return pred_pose_enc_list

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        num_iterations: int,
        attn_mask: Optional[torch.Tensor],
        kv_cache_list: Optional[List[List[List[torch.Tensor]]]] = None,
    ) -> Union[list, Tuple[list, List[List[List[torch.Tensor]]]]]:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for iter in range(num_iterations):

            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:

                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(
                3, dim=-1
            )

            adaln_output = self.adaln_norm(pose_tokens)
            modulated_output = modulate(adaln_output, shift_msa, scale_msa)
            gated_output = gate_msa * modulated_output
            pose_tokens_modulated = gated_output + pose_tokens

            for i in range(self.trunk_depth):
                if kv_cache_list is not None:
                    pose_tokens_modulated, kv_cache_list[iter][i] = self.trunk[i](
                        pose_tokens_modulated,
                        attn_mask=attn_mask,
                        kv_cache=kv_cache_list[iter][i],
                    )
                else:
                    pose_tokens_modulated = self.trunk[i](
                        pose_tokens_modulated, attn_mask=attn_mask
                    )

            trunk_norm_output = self.trunk_norm(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(trunk_norm_output)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        if kv_cache_list is not None:
            return pred_pose_enc_list, kv_cache_list
        else:
            return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """

    return x * (1 + scale) + shift


class RelPoseHead(nn.Module):
    """
    Enhanced Relative Pose Head for dynamic keyframe-based pose prediction.

    Key features:
    1. True relative pose prediction (not incremental from fixed anchor)
    2. Dynamic keyframe switching support
    3. SE(3) and Sim(3) pose modes
    4. Role-aware processing for keyframes vs non-keyframes
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_mode: str = "SE3",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        use_global_scale: bool = False,
        use_pair_cross_attn: bool = False,
        detach_reference: bool = False,
        xattn_temperature: float = 1.0,
        use_precat: bool = False,
        use_kf_role_embed: bool = True,
        kf_role_embed_init_std: float = 0.02,
        window_size: int = 50000,
    ):
        super().__init__()

        self.pose_mode = pose_mode
        self.use_global_scale = use_global_scale and (pose_mode == "Sim3")
        self.use_pair_cross_attn = use_pair_cross_attn
        self.detach_reference = detach_reference
        self.xattn_temperature = xattn_temperature
        self.use_precat = use_precat
        self.use_kf_role_embed = use_kf_role_embed
        self.kf_role_embed_init_std = kf_role_embed_init_std

        self.target_dim = 9

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.window_size = 50000

        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True)
        )

        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

        if self.use_global_scale:
            self.global_scale = nn.Parameter(torch.ones(1))

        if self.use_pair_cross_attn:
            self.xattn_q = nn.Linear(dim_in, dim_in, bias=False)
            self.xattn_k = nn.Linear(dim_in, dim_in, bias=False)
            self.xattn_v = nn.Linear(dim_in, dim_in, bias=False)
            self.xattn_out = nn.Linear(dim_in, dim_in, bias=False)

        if self.use_precat:
            self.precat_proj = nn.Linear(dim_in * 2, dim_in, bias=True)

        if self.use_kf_role_embed:
            self.kf_role_embed = nn.Parameter(torch.randn(1, 1, dim_in))
            nn.init.normal_(self.kf_role_embed, std=self.kf_role_embed_init_std)
        else:
            self.kf_role_embed = None

    def _create_attn_mask(
        self, S: int, mode: str, dtype: torch.dtype, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Create attention mask for the given mode."""
        N = S

        if mode == "causal":
            mask = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                mask[i, i + 1 :] = float("-inf")
            return mask
        elif mode == "window":
            mask = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                mask[i, :] = float("-inf")
                start = max(0, i - self.window_size + 1)
                mask[i, start : i + 1] = 0
            return mask
        elif mode == "full":
            return None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

    def forward(
        self,
        aggregated_tokens_list: list,
        keyframe_indices: torch.Tensor,
        is_keyframe: torch.Tensor,
        num_iterations: int = 4,
        mode: str = "causal",
        kv_cache_list: Optional[List[List[List[torch.Tensor]]]] = None,
        compute_switch_poses: bool = False,
    ):
        """
        Forward pass for relative pose prediction.

        Args:
            aggregated_tokens_list: List of aggregated tokens from the network
            keyframe_indices: Indices of reference keyframes for each frame [B, S]
            is_keyframe: Boolean mask indicating keyframes [B, S]
            num_iterations: Number of iterative refinement steps
            mode: Attention mode ("causal", "window", or "full")
            kv_cache_list: Optional KV cache for streaming

        Returns:
            dict containing:
                - pose_enc: Predicted relative poses [B, S, 9]
                - is_keyframe: Keyframe mask [B, S]
                - keyframe_indices: Reference keyframe indices [B, S]
                - global_scale: Global scale for Sim(3) mode (if applicable)
        """
        mode = "causal"

        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        B, S, C = pose_tokens.shape

        if kv_cache_list is not None and S == 1:

            if not hasattr(self, "_keyframe_tokens_cache"):

                self._keyframe_tokens_cache = {}
                self._current_frame_id = 0

                self._frame_info = []

            curr_is_kf = is_keyframe[0, 0].item() if is_keyframe is not None else True
            curr_ref_idx = (
                keyframe_indices[0, 0].item()
                if keyframe_indices is not None
                else self._current_frame_id
            )
            self._frame_info.append((curr_is_kf, curr_ref_idx))

            if curr_is_kf:

                self._keyframe_tokens_cache[
                    self._current_frame_id
                ] = pose_tokens.squeeze(1)

            self._current_frame_id += 1

        ref_tokens = None
        if keyframe_indices is not None:
            if kv_cache_list is not None and S == 1:

                ref_frame_id = keyframe_indices[0, 0].item()

                if ref_frame_id in self._keyframe_tokens_cache:

                    ref_tokens = self._keyframe_tokens_cache[ref_frame_id].unsqueeze(1)
                else:

                    ref_tokens = pose_tokens

                if self.detach_reference:
                    ref_tokens = ref_tokens.detach()
            else:

                total_frames = pose_tokens.shape[1]
                ref_idx = (
                    keyframe_indices.clamp(0, total_frames - 1)
                    .unsqueeze(-1)
                    .expand(-1, -1, C)
                )
                ref_tokens = torch.gather(pose_tokens, dim=1, index=ref_idx)
                if self.detach_reference:
                    ref_tokens = ref_tokens.detach()

            if (
                self.use_kf_role_embed
                and ref_tokens is not None
                and self.kf_role_embed is not None
            ):

                current_indices = (
                    torch.arange(S, device=keyframe_indices.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )

                is_self_ref = current_indices == keyframe_indices

                add_kf_embed_mask = (~is_self_ref).unsqueeze(-1).float()

                ref_tokens = ref_tokens + add_kf_embed_mask * self.kf_role_embed.expand(
                    B, S, -1
                )

        if self.use_pair_cross_attn and (ref_tokens is not None):
            q = self.xattn_q(pose_tokens)
            k = self.xattn_k(ref_tokens)
            v = self.xattn_v(ref_tokens)

            scale = (q * k).sum(dim=-1, keepdim=True) / (C ** 0.5)
            gate = torch.sigmoid(scale / self.xattn_temperature)

            pair_info = self.xattn_out(gate * v)
            pose_tokens = pose_tokens + pair_info

        if self.use_precat and (ref_tokens is not None):
            pose_tokens = self.precat_proj(torch.cat([pose_tokens, ref_tokens], dim=-1))

        attn_mask = None
        if keyframe_indices is not None:

            ref = keyframe_indices
            B = ref.shape[0]

            j_indices = (
                torch.arange(S, device=pose_tokens.device)
                .view(1, 1, S)
                .expand(B, S, -1)
            )
            is_ref_frame = j_indices == ref[:, :, None]
            same_ref = ref[:, :, None] == ref[:, None, :]
            can_attend_nonkf = is_ref_frame | same_ref

            idx = torch.arange(S, device=pose_tokens.device)

            if mode == "causal":

                causal_mask = idx[None, :, None] >= idx[None, None, :]

                prev_kf_mask = torch.zeros(
                    B, S, S, dtype=torch.bool, device=pose_tokens.device
                )
                for b in range(B):
                    kf_positions = [i for i in range(S) if is_keyframe[b, i]]
                    for kf_idx, kf_pos in enumerate(kf_positions):
                        if kf_idx == 0:

                            prev_kf_mask[b, kf_pos, : kf_pos + 1] = True
                        else:

                            prev_kf_pos = kf_positions[kf_idx - 1]
                            prev_kf_mask[b, kf_pos, prev_kf_pos : kf_pos + 1] = True

                can_attend_kf = causal_mask.expand(B, -1, -1) & prev_kf_mask
                mode_constraint = causal_mask
            elif mode == "window":

                causal_mask = idx[None, :, None] >= idx[None, None, :]
                window_mask = (
                    idx[None, :, None] - idx[None, None, :]
                ) < self.window_size
                window_causal = causal_mask & window_mask
                can_attend_kf = window_causal.expand(B, -1, -1)
                mode_constraint = window_causal
            elif mode == "full":

                can_attend_kf = torch.ones(
                    B, S, S, dtype=torch.bool, device=pose_tokens.device
                )
                mode_constraint = torch.ones(
                    1, S, S, dtype=torch.bool, device=pose_tokens.device
                )
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")

            is_kf_expanded = is_keyframe[:, :, None].expand(-1, -1, S)
            can_attend = torch.where(is_kf_expanded, can_attend_kf, can_attend_nonkf)

            mask_bool = can_attend & mode_constraint

            zero = torch.zeros(1, dtype=pose_tokens.dtype, device=pose_tokens.device)
            neg_inf = torch.full(
                (1,), float("-inf"), dtype=pose_tokens.dtype, device=pose_tokens.device
            )
            attn_mask = torch.where(mask_bool, zero, neg_inf)[:, None, :, :]

            if kv_cache_list is not None and len(kv_cache_list) > 0 and S == 1:

                k_cache = kv_cache_list[0][0][0]
                if k_cache is not None:
                    cache_len = k_cache.shape[2]

                    curr_idx = len(self._frame_info) - 1
                    curr_is_kf, curr_ref_idx = self._frame_info[curr_idx]

                    cache_mask_vals = []
                    visible_frames = []
                    for i in range(max(0, curr_idx - cache_len), curr_idx):
                        cache_is_kf, cache_ref_idx = self._frame_info[i]

                        if curr_is_kf:

                            prev_kf_idx = None
                            for j in range(curr_idx - 1, -1, -1):
                                if self._frame_info[j][0]:
                                    prev_kf_idx = j
                                    break

                            if prev_kf_idx is not None:

                                can_see = i >= prev_kf_idx
                            else:

                                can_see = True
                        else:

                            is_ref = i == curr_ref_idx
                            same_ref = cache_ref_idx == curr_ref_idx
                            can_see = is_ref or same_ref

                        mask_val = zero if can_see else neg_inf
                        cache_mask_vals.append(mask_val)
                        if can_see:
                            visible_frames.append(i)

                    if len(cache_mask_vals) > 0:
                        cache_mask = torch.stack(cache_mask_vals, dim=0).view(
                            1, 1, 1, len(cache_mask_vals)
                        )
                        cache_mask = cache_mask.expand(B, 1, S, len(cache_mask_vals))
                        attn_mask = torch.cat([cache_mask, attn_mask], dim=-1)

        else:

            attn_mask = self._create_attn_mask(
                S, mode, pose_tokens.dtype, pose_tokens.device
            )

        pred_pose_enc_list = []
        pred_pose_enc = None

        for iter_idx in range(num_iterations):

            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:

                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(
                3, dim=-1
            )

            pose_tokens_modulated = gate_msa * modulate(
                self.adaln_norm(pose_tokens), shift_msa, scale_msa
            )
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for i in range(self.trunk_depth):
                if (
                    kv_cache_list is not None
                    and iter_idx < len(kv_cache_list)
                    and i < len(kv_cache_list[iter_idx])
                ):
                    pose_tokens_modulated, kv_cache_list[iter_idx][i] = self.trunk[i](
                        pose_tokens_modulated,
                        attn_mask=attn_mask,
                        kv_cache=kv_cache_list[iter_idx][i],
                    )
                else:
                    pose_tokens_modulated = self.trunk[i](
                        pose_tokens_modulated, attn_mask=attn_mask
                    )

            trunk_norm_output = self.trunk_norm(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(trunk_norm_output)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )

            pred_pose_enc_list.append(activated_pose)

        final_pose_enc = pred_pose_enc_list[-1]
        if final_pose_enc.dtype != torch.float32:
            final_pose_enc = final_pose_enc.float()

        result = {
            "pose_enc": final_pose_enc,
            "is_keyframe": is_keyframe,
            "keyframe_indices": keyframe_indices,
        }

        if self.training and len(pred_pose_enc_list) > 0:
            result["pose_enc_list"] = pred_pose_enc_list

        if compute_switch_poses:
            switch_poses = self._compute_switch_poses(
                pred_pose_enc_list[-1], keyframe_indices, is_keyframe
            )
            result["switch_poses"] = switch_poses

        if self.use_global_scale:
            result["global_scale"] = self.global_scale.expand(B, 1)

        if kv_cache_list is not None:
            result["kv_cache_list"] = kv_cache_list

        return result

    def _compute_switch_poses(self, poses, keyframe_indices, is_keyframe):
        """
        Compute T_{k'←k} for keyframe switches.

        Returns a dictionary mapping (k, k') pairs to the relative transformation.
        """
        B, S, _ = poses.shape
        switch_poses = {}

        for b in range(B):
            prev_kf_idx = None
            for s in range(S):
                if is_keyframe[b, s]:
                    if prev_kf_idx is not None:

                        key = (b, prev_kf_idx, s)
                        switch_poses[key] = poses[b, s].clone()
                    prev_kf_idx = s

        return switch_poses
