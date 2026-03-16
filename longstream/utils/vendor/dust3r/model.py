import os
from copy import deepcopy

import huggingface_hub
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from longstream.utils.vendor.croco.models.croco import CroCoNet
from longstream.utils.vendor.croco.models.blocks import Block
from longstream.utils.vendor.croco.models.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
)
from packaging import version

from longstream.utils.vendor.dust3r.patch_embed import get_patch_embed

from .heads import head_factory
from .utils.misc import (
    fill_default_args,
    freeze_all_params,
    interleave,
    is_symmetrized,
    transpose_to_landscape,
)
import torch.autograd.profiler as profiler

inf = float("inf")

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"
), "Outdated huggingface_hub version, please reinstall requirements.txt"


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(
            output_mode,
            head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(
                pretrained_model_name_or_path, **kw
            )

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def load_state_dict(self, ckpt, **kw):

        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        self.downstream_head1 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        self.downstream_head2 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )

        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):

        x, pos = self.patch_embed(image, true_shape=true_shape)

        assert self.enc_pos_embed is None

        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]

        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )

        if is_symmetrized(view1, view2):

            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1, img2, shape1, shape2
            )

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]

        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):

            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)

            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)

            final_output.append((f1, f2))

        del final_output[1]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape

        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(self, view1, view2):

        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(
            view1, view2
        )

        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2["pts3d_in_other_view"] = res2.pop("pts3d")
        return res1, res2


class FlashDUSt3R(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """Two siamese encoders, followed by a single large decoder.
    The goal is to output 3d points directly, processing multiple views.
    """

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        decoder_pos_embed_type="sinusoidal",
        attn_implementation="pytorch_naive",
        random_image_idx_embedding=False,
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.random_image_idx_embedding = random_image_idx_embedding
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        croco_kwargs["attn_implementation"] = attn_implementation
        super().__init__(**croco_kwargs)

        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(self.dec_embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        del self.dec_blocks
        torch.cuda.empty_cache()

        self.decoder_pos_embed_type = decoder_pos_embed_type
        self.multiview_dec_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dec_embed_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    norm_layer=nn.LayerNorm,
                    attn_implementation=attn_implementation,
                )
                for _ in range(12)
            ]
        )
        self.set_downstream_head(
            output_mode,
            head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            return super(FlashDUSt3R, cls).from_pretrained(
                pretrained_model_name_or_path, **kw
            )

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def set_freeze(self, freeze):
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
            "sandwich": [
                self.mask_token,
                self.patch_embed,
                self.enc_blocks,
                self.downstream_head,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        self.downstream_head = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )

        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):

        x, pos = self.patch_embed(image, true_shape=true_shape)

        assert self.enc_pos_embed is None

        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos

    def _encode_images(self, views):
        B = views[0]["img"].shape[0]
        encoded_feats, positions, shapes = [], [], []

        for view in views:
            img = view["img"]
            true_shape = view.get(
                "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
            )
            feat, pos = self._encode_image(img, true_shape)
            encoded_feats.append(feat)
            positions.append(pos)
            shapes.append(true_shape)

        return encoded_feats, positions, shapes

    def _generate_per_rank_generator(self):

        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        per_rank_seed = per_forward_pass_seed + world_rank

        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(
        self, encoded_feats, batch_size, num_views, max_image_idx, device
    ):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """

        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        image_ids[:, 0] = 0

        per_rank_generator = self._generate_per_rank_generator()

        for b in range(batch_size):

            random_ids = (
                torch.randperm(max_image_idx, generator=per_rank_generator)[
                    : num_views - 1
                ]
                + 1
            )
            image_ids[b, 1:] = random_ids

        image_ids = image_ids.to(device)

        image_pos_list = []

        for i in range(num_views):

            num_patches = encoded_feats[i].shape[1]

            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]

            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(
                1, num_patches, 1
            )

            image_pos_list.append(image_pos_for_view)

        image_pos = torch.cat(image_pos_list, dim=1)

        return image_pos

    def _decoder(self, encoded_feats, positions, image_ids):
        x = torch.cat(encoded_feats, dim=1)
        pos = torch.cat(positions, dim=1)

        final_output = [x]

        x = self.decoder_embed(x)

        if self.random_image_idx_embedding:

            image_pos = self._get_random_image_pos(
                encoded_feats=encoded_feats,
                batch_size=encoded_feats[0].shape[0],
                num_views=len(encoded_feats),
                max_image_idx=self.image_idx_emb.shape[0] - 1,
                device=x.device,
            )
        else:

            num_images = (torch.max(image_ids) + 1).cpu().item()
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]

        x += image_pos

        for blk in self.multiview_dec_blocks:
            x = blk(x, pos)
            final_output.append(x)

        x = self.dec_norm(x)
        final_output[-1] = x
        return final_output

    def forward(self, views):
        """
        Args:
            views (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched

        Returns:
            list[dict]: a list of results for each view
        """

        encoded_feats, positions, shapes = self._encode_images(views)

        num_images = len(views)
        B, _, _ = encoded_feats[0].shape

        different_resolution_across_views = not all(
            encoded_feats[0].shape[1] == encoded_feat.shape[1]
            for encoded_feat in encoded_feats
        )

        image_ids = []

        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]

            image_ids.extend([i] * num_patches)

        image_ids = (
            torch.tensor(image_ids * B).reshape(B, -1).to(encoded_feats[0].device)
        )

        dec_output = self._decoder(encoded_feats, positions, image_ids)

        final_results = [{} for _ in range(num_images)]

        with profiler.record_function("head: gathered outputs"):

            gathered_outputs_list = []
            if different_resolution_across_views:
                for img_id in range(num_images):
                    gathered_outputs_per_view = []
                    for layer_output in dec_output:
                        B, P, D = layer_output.shape
                        mask = image_ids == img_id
                        gathered_output = layer_output[mask].view(B, -1, D)
                        gathered_outputs_per_view.append(gathered_output)
                    gathered_outputs_list.append(gathered_outputs_per_view)
            else:
                for layer_output in dec_output:
                    B, P, D = layer_output.shape
                    gathered_outputs_per_view = []
                    for img_id in range(num_images):
                        mask = image_ids == img_id
                        gathered_output = layer_output[mask].view(B, -1, D)
                        gathered_outputs_per_view.append(gathered_output)
                    gathered_outputs_list.append(
                        torch.cat(gathered_outputs_per_view, dim=0)
                    )

        with profiler.record_function("head: forward pass"):
            if different_resolution_across_views:

                final_results = [{} for _ in range(num_images)]
                for img_id in range(num_images):
                    img_result = self.head(
                        gathered_outputs_list[img_id], shapes[img_id]
                    )

                    for key in img_result.keys():
                        if key == "pts3d":
                            final_results[img_id]["pts3d_in_other_view"] = img_result[
                                key
                            ]
                        else:
                            final_results[img_id][key] = img_result[key]
            else:

                concatenated_shapes = torch.cat(shapes, dim=0)

                result = self.head(gathered_outputs_list, concatenated_shapes)

                final_results = [{} for _ in range(num_images)]

                for key in result.keys():
                    for img_id in range(num_images):
                        img_result = result[key][img_id * B : (img_id + 1) * B]
                        if key == "pts3d":
                            final_results[img_id]["pts3d_in_other_view"] = img_result
                        else:
                            final_results[img_id][key] = img_result

        return final_results
