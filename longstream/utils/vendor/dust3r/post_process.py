import numpy as np
import torch

from longstream.utils.vendor.dust3r.utils.geometry import xy_grid


def estimate_focal_knowing_depth(
    pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)
    pts3d = pts3d.flatten(1, 2)

    if focal_mode == "median":
        with torch.no_grad():

            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":

        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        for iter in range(10):

            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)

            w = dis.clip(min=1e-8).reciprocal()

            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)

    return focal


def estimate_focal_knowing_depth_and_confidence_mask(
    pts3d, pp, conf_mask, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    This function considers only points where conf_mask is True.
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    pixels = xy_grid(W, H, device=pts3d.device).view(1, H, W, 2) - pp.view(-1, 1, 1, 2)

    conf_mask = conf_mask.view(B, H, W)
    valid_indices = conf_mask

    pts3d_valid = pts3d[valid_indices]
    pixels_valid = pixels[valid_indices]

    if pts3d_valid.numel() == 0:

        focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
        return torch.tensor([focal_base])

    if focal_mode == "median":
        with torch.no_grad():

            u, v = pixels_valid.unbind(dim=-1)
            x, y, z = pts3d_valid.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            f_votes = torch.cat((fx_votes.view(-1), fy_votes.view(-1)), dim=-1)
            focal = torch.nanmedian(f_votes).unsqueeze(0)

    elif focal_mode == "weiszfeld":

        xy_over_z = (pts3d_valid[..., :2] / pts3d_valid[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )

        dot_xy_px = (xy_over_z * pixels_valid).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean() / dot_xy_xy.mean()

        for _ in range(100):

            dis = (pixels_valid - focal * xy_over_z).norm(dim=-1)
            w = dis.clip(min=1e-8).reciprocal()

            focal = (w * dot_xy_px).sum() / (w * dot_xy_xy).sum()
        focal = focal.unsqueeze(0)
    else:
        raise ValueError(f"bad focal_mode={focal_mode}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    return focal
