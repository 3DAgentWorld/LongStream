import logging
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
from torch.utils.checkpoint import checkpoint

from ..layers import PatchEmbed
from ..layers.block import Block
from ..layers.rope import (
    RotaryPositionEmbedding2D,
    RotaryPositionEmbedding3D,
    PositionGetter,
)
from ..layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class STreamAggregator(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_role_embedding=True,
        disable_keyframe_distinction=False,
        keyframe_stride=8,
        use_segment_mask=False,
        use_3d_rope=False,
        window_size=5000,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim
        )

        if rope_freq > 0:
            if use_3d_rope:
                self.rope = RotaryPositionEmbedding3D(frequency=rope_freq)
            else:
                self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        else:
            self.rope = None

        self.position_getter = PositionGetter() if self.rope is not None else None
        self.use_3d_rope = use_3d_rope

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size
        self.use_role_embedding = use_role_embedding
        self.disable_keyframe_distinction = disable_keyframe_distinction
        self.num_register_tokens = num_register_tokens
        self.use_segment_mask = use_segment_mask
        self.window_size = window_size

        self.camera_token_norm = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_token_norm = nn.Parameter(
            torch.randn(1, num_register_tokens, embed_dim)
        )
        nn.init.normal_(self.camera_token_norm, std=1e-6)
        nn.init.normal_(self.register_token_norm, std=1e-6)

        if not disable_keyframe_distinction:
            self.camera_token_key = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.register_token_key = nn.Parameter(
                torch.randn(1, num_register_tokens, embed_dim)
            )
            nn.init.normal_(self.camera_token_key, std=1e-6)
            nn.init.normal_(self.register_token_key, std=1e-6)
        else:

            self.camera_token_key = None
            self.register_token_key = None

        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        if self.use_role_embedding:
            self.role_embed_key = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.role_embed_norm = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.normal_(self.role_embed_key, std=0.02)
            nn.init.normal_(self.role_embed_norm, std=0.02)

        self.patch_start_idx = 1 + num_register_tokens

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def _create_attn_mask(
        self,
        S: int,
        P: int,
        mode: str,
        dtype: torch.dtype,
        device: torch.device,
        reorder_indices: Optional[torch.Tensor] = None,
        is_keyframe: Optional[torch.Tensor] = None,
        keyframe_indices: Optional[torch.Tensor] = None,
    ):
        """
        Create attention mask based on mode and optionally adjust for reordering.

        Args:
            S: Sequence length
            P: Tokens per frame
            mode: "causal", "window", or "full"
            dtype: Data type
            device: Device
            reorder_indices: Optional reordering indices [B*S] for keyframe-first ordering
            is_keyframe: Optional keyframe mask [B, S]
            keyframe_indices: Optional reference keyframe indices [B, S]

        Returns:
            Attention mask [N, N] where N = S * P (or [B, 1, N, N] if segment-aware)
        """
        N = S * P

        if mode == "full":
            return None

        if (
            self.use_segment_mask
            and is_keyframe is not None
            and keyframe_indices is not None
        ):
            B = is_keyframe.shape[0]

            should_print = False
            if should_print:
                print(f"\n[Aggregator Segment Mask DEBUG]")
                print(f"  use_segment_mask={self.use_segment_mask}")
                print(f"  Mode: {mode}")
                print(f"  Sequence length: {S}, Tokens per frame: {P}")
                print(f"  is_keyframe[0]: {is_keyframe[0].tolist()}")
                print(f"  keyframe_indices[0]: {keyframe_indices[0].tolist()}")

            ref = keyframe_indices

            idx = torch.arange(S, device=device)
            j_indices = torch.arange(S, device=device).view(1, 1, S).expand(B, S, -1)

            is_ref_frame = j_indices == ref[:, :, None]
            same_ref = ref[:, :, None] == ref[:, None, :]
            can_attend_nonkf = is_ref_frame | same_ref

            if mode == "causal":
                causal_mask = idx[None, :, None] >= idx[None, None, :]

                prev_kf_mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
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
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")

            is_kf_expanded = is_keyframe[:, :, None].expand(-1, -1, S)
            can_attend = torch.where(is_kf_expanded, can_attend_kf, can_attend_nonkf)

            mask_bool = can_attend & mode_constraint

            mask_bool_expanded = torch.zeros(B, N, N, dtype=torch.bool, device=device)
            for i in range(S):
                for j in range(S):
                    mask_bool_expanded[
                        :, i * P : (i + 1) * P, j * P : (j + 1) * P
                    ] = mask_bool[:, i : i + 1, j : j + 1]

            zero = torch.zeros(1, dtype=dtype, device=device)
            neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)
            segment_mask = torch.where(mask_bool_expanded, zero, neg_inf).unsqueeze(1)

            if should_print:

                kf_positions = [i for i in range(S) if is_keyframe[0, i]]
                if len(kf_positions) >= 2:

                    kf_pos = kf_positions[1]
                    visible_frames = []
                    for j in range(S):
                        if mask_bool[0, kf_pos, j]:
                            visible_frames.append(j)
                    print(
                        f"  Frame {kf_pos} (keyframe) can attend to frames: {visible_frames}"
                    )

                    if kf_pos + 1 < S:
                        post_pos = kf_pos + 1
                        visible_frames = []
                        for j in range(S):
                            if mask_bool[0, post_pos, j]:
                                visible_frames.append(j)
                        print(
                            f"  Frame {post_pos} (post-switch) can attend to frames: {visible_frames}"
                        )
                        print(f"  ✅ Segment mask is working correctly!")

            return segment_mask

        if mode == "causal":
            mask_original = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P

                mask_original[curr_view_start:curr_view_end, curr_view_end:] = float(
                    "-inf"
                )
        elif mode == "window":
            mask_original = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P

                mask_original[curr_view_start:curr_view_end, P:] = float("-inf")

                start_view = max(1, i - self.window_size + 1)
                mask_original[
                    curr_view_start:curr_view_end, start_view * P : (i + 1) * P
                ] = 0
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        if reorder_indices is not None:

            mask_reordered = torch.zeros_like(mask_original)

            for new_i in range(S):
                for new_j in range(S):
                    orig_i = reorder_indices[new_i].item()
                    orig_j = reorder_indices[new_j].item()

                    mask_reordered[
                        new_i * P : (new_i + 1) * P, new_j * P : (new_j + 1) * P
                    ] = mask_original[
                        orig_i * P : (orig_i + 1) * P, orig_j * P : (orig_j + 1) * P
                    ]

            return mask_reordered

        return mask_original

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        kv_cache_list: Optional[List[List[torch.Tensor]]] = None,
        is_keyframe: Optional[torch.Tensor] = None,
        keyframe_indices: Optional[torch.Tensor] = None,
        additional_tokens: Optional[torch.Tensor] = None,
        reorder_keyframes_first: bool = False,
    ) -> Union[
        Tuple[List[torch.Tensor], int],
        Tuple[List[torch.Tensor], int, List[List[torch.Tensor]]],
    ]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator
            is_keyframe (torch.Tensor): Boolean tensor indicating keyframes [B, S]
            keyframe_indices (torch.Tensor): Reference keyframe indices for each frame [B, S]
            additional_tokens (torch.Tensor): Additional tokens to insert (e.g., scale token) [B, C, T]
            reorder_keyframes_first (bool): If True, reorder tokens so keyframes come first

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std

        images = images.view(B * S, C_in, H, W)

        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if is_keyframe is not None:

            camera_token, register_token = self._select_role_tokens(B, S, is_keyframe)
        else:

            is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None
            camera_token = slice_expand_and_flatten(
                self.camera_token, B, S, is_anchor_exist=is_anchor_exist
            )
            register_token = slice_expand_and_flatten(
                self.register_token, B, S, is_anchor_exist=is_anchor_exist
            )

        if additional_tokens is not None:

            T = additional_tokens.shape[-1]
            additional_tokens_expanded = (
                additional_tokens.unsqueeze(1).repeat(1, S, 1, 1).view(B * S, T, C)
            )

            tokens = torch.cat(
                [
                    camera_token,
                    register_token,
                    additional_tokens_expanded,
                    patch_tokens,
                ],
                dim=1,
            )
            patch_start_idx_with_additional = self.patch_start_idx + T

        else:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
            patch_start_idx_with_additional = self.patch_start_idx

        if (
            self.use_role_embedding
            and not self.disable_keyframe_distinction
            and is_keyframe is not None
        ):
            P_patch = patch_tokens.shape[1]
            tokens = self._apply_role_embedding(tokens, B, S, P_patch, is_keyframe)

        pos = None
        if self.rope is not None and self.position_getter is not None:
            if self.use_3d_rope:

                pos = self.position_getter.get_3d_positions(
                    B,
                    S,
                    H // self.patch_size,
                    W // self.patch_size,
                    device=images.device,
                )
            else:

                pos = self.position_getter(
                    B * S,
                    H // self.patch_size,
                    W // self.patch_size,
                    device=images.device,
                )

        if patch_start_idx_with_additional > 0 and pos is not None:

            pos = pos + 1
            pos_dim = 3 if self.use_3d_rope else 2
            pos_special = (
                torch.zeros(B * S, patch_start_idx_with_additional, pos_dim)
                .to(images.device)
                .to(pos.dtype)
            )

            if self.use_3d_rope:

                temporal_indices = torch.arange(
                    S, device=images.device, dtype=pos.dtype
                )
                temporal_indices = temporal_indices.repeat_interleave(B).view(
                    B * S, 1, 1
                )
                temporal_indices = temporal_indices.expand(
                    -1, patch_start_idx_with_additional, -1
                )
                pos_special[:, :, 2:3] = temporal_indices

            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        reorder_indices = None
        restore_indices = None
        frame_reorder_map = None
        if is_keyframe is not None and reorder_keyframes_first:
            (
                tokens,
                pos,
                reorder_indices,
                restore_indices,
            ) = self._reorder_keyframes_first(tokens, pos, B, S, P, is_keyframe)

            if B > 0 and reorder_indices is not None:
                frame_reorder_map = torch.zeros(
                    S, dtype=torch.long, device=tokens.device
                )
                for new_s in range(S):
                    orig_frame_in_batch = reorder_indices[new_s].item() % S
                    frame_reorder_map[new_s] = orig_frame_in_batch

        attn_mask = None
        if kv_cache_list is None:
            attn_mask = self._create_attn_mask(
                S,
                P,
                mode,
                tokens.dtype,
                tokens.device,
                reorder_indices=frame_reorder_map,
                is_keyframe=is_keyframe,
                keyframe_indices=keyframe_indices,
            )

        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_idx in range(self.aa_block_num):
            frame_intermediates = None
            global_intermediates = None

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    (
                        tokens,
                        frame_idx,
                        frame_intermediates,
                    ) = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if kv_cache_list is not None:
                        kv_cache = kv_cache_list[global_idx]
                        (
                            tokens,
                            global_idx,
                            global_intermediates,
                            kv_cache,
                        ) = self._process_global_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            attn_mask=attn_mask,
                            kv_cache=kv_cache,
                        )
                        kv_cache_list[global_idx - 1] = kv_cache
                    else:
                        (
                            tokens,
                            global_idx,
                            global_intermediates,
                        ) = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if frame_intermediates is not None and global_intermediates is not None:
                for i in range(len(frame_intermediates)):

                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

        if kv_cache_list is not None:
            return (
                output_list,
                patch_start_idx_with_additional,
                kv_cache_list,
                restore_indices,
            )
        else:
            return output_list, patch_start_idx_with_additional, restore_indices

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos: Optional[torch.Tensor] = None
    ):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """

        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None:

            pos_dim = pos.shape[-1]
            expected_shape = (B * S, P, pos_dim)
            if pos.shape != expected_shape:
                pos = pos.view(B, S, P, pos_dim).view(B * S, P, pos_dim)

        intermediates = []

        for _ in range(self.aa_block_size):
            tokens = checkpoint(
                self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False
            )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[torch.Tensor]] = None,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None:

            pos_dim = pos.shape[-1]
            expected_shape = (B, S * P, pos_dim)
            if pos.shape != expected_shape:
                pos = pos.view(B, S, P, pos_dim).view(B, S * P, pos_dim)

        intermediates = []

        for _ in range(self.aa_block_size):
            if kv_cache is not None:
                result = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    attn_mask,
                    kv_cache,
                    use_reentrant=False,
                )
                tokens, kv_cache = result
            else:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    attn_mask,
                    use_reentrant=False,
                )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if kv_cache is not None:
            return tokens, global_idx, intermediates, kv_cache

        return tokens, global_idx, intermediates

    def _select_role_tokens(self, B, S, is_keyframe):
        """
        Select camera and register tokens based on keyframe mask.

        When disable_keyframe_distinction=True, all frames use the same tokens (camera_token_norm/register_token_norm).
        When disable_keyframe_distinction=False, keyframes use _key tokens and normal frames use _norm tokens.

        Args:
            B: Batch size
            S: Sequence length
            is_keyframe: Boolean tensor [B, S] indicating keyframes

        Returns:
            camera_token: Selected camera tokens [B*S, 1, C]
            register_token: Selected register tokens [B*S, num_register_tokens, C]
        """

        device = is_keyframe.device
        is_keyframe = is_keyframe.bool()

        if self.disable_keyframe_distinction:

            camera_token = self.camera_token_norm.expand(B * S, -1, -1)
            register_token = self.register_token_norm.expand(B * S, -1, -1)
            return camera_token, register_token

        if self.camera_token_key is None or self.register_token_key is None:
            raise RuntimeError(
                "camera_token_key and register_token_key are not initialized. "
                "This happens when disable_keyframe_distinction=True but is_keyframe distinction is requested. "
                "Please set disable_keyframe_distinction=False in the configuration."
            )

        camera_tokens = []
        register_tokens = []

        for b in range(B):
            for s in range(S):
                if is_keyframe[b, s]:

                    camera_tokens.append(self.camera_token_key)
                    register_tokens.append(self.register_token_key)
                else:

                    camera_tokens.append(self.camera_token_norm)
                    register_tokens.append(self.register_token_norm)

        camera_token = torch.cat(camera_tokens, dim=0)
        register_token = torch.cat(register_tokens, dim=0)

        return camera_token, register_token

    def _apply_role_embedding(self, tokens, B, S, P_patch, is_keyframe):
        """
        Apply role embeddings to all tokens (including patches) for attention bias.

        🔥 使用 FP32 进行 role embedding 计算，避免数值不稳定和 NaN

        Args:
            tokens: Combined tokens [B*S, total_tokens, C]
            B: Batch size
            S: Sequence length
            P_patch: Number of patch tokens per image
            is_keyframe: Boolean tensor [B, S] indicating keyframes

        Returns:
            tokens_with_role: Tokens with role embeddings added [B*S, total_tokens, C]
        """

        device = tokens.device
        is_keyframe = is_keyframe.bool()
        _, total_tokens, C = tokens.shape

        role_embeds = []

        for b in range(B):
            for s in range(S):
                if is_keyframe[b, s]:

                    role_embed = self.role_embed_key.expand(1, total_tokens, -1)
                else:

                    role_embed = self.role_embed_norm.expand(1, total_tokens, -1)
                role_embeds.append(role_embed)

        role_embedding = torch.cat(role_embeds, dim=0)

        tokens_with_role = tokens + 0.1 * role_embedding

        return tokens_with_role

    def _reorder_keyframes_first(self, tokens, pos, B, S, P, is_keyframe):
        """
        Reorder tokens so that keyframe tokens come first, followed by normal frame tokens.

        Args:
            tokens: Combined tokens [B*S, P, C]
            pos: Position embeddings [B*S, P, 2] or [B*S, P, 3] or None
                 (2D for spatial-only RoPE, 3D for spatial+temporal RoPE)
            B: Batch size
            S: Sequence length
            P: Number of tokens per frame
            is_keyframe: Boolean tensor [B, S] indicating keyframes

        Returns:
            reordered_tokens: Tokens with keyframes first [B*S, P, C]
            reordered_pos: Position embeddings reordered [B*S, P, 2/3] or None
            reorder_indices: Indices used for reordering [B*S]
            restore_indices: Indices to restore original order [B*S]
        """
        device = tokens.device
        is_keyframe = is_keyframe.bool()

        reorder_indices = []
        for b in range(B):
            keyframe_indices = []
            normal_indices = []
            for s in range(S):
                idx = b * S + s
                if is_keyframe[b, s]:
                    keyframe_indices.append(idx)
                else:
                    normal_indices.append(idx)

            reorder_indices.extend(keyframe_indices + normal_indices)

        reorder_indices = torch.tensor(reorder_indices, device=device, dtype=torch.long)

        restore_indices = torch.zeros_like(reorder_indices)
        restore_indices[reorder_indices] = torch.arange(
            B * S, device=device, dtype=torch.long
        )

        reordered_tokens = tokens[reorder_indices]

        reordered_pos = None
        if pos is not None:
            reordered_pos = pos[reorder_indices]

        return reordered_tokens, reordered_pos, reorder_indices, restore_indices


def slice_expand_and_flatten(token_tensor, B, S, is_anchor_exist=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    if is_anchor_exist:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])

    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])

    combined = torch.cat([query, others], dim=1)

    combined = combined.view(B * S, *combined.shape[2:])
    return combined
