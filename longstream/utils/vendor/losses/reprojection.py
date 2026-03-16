from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def project_points(
    points: torch.Tensor, K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = points[..., 2].clamp(min=1e-6)
    x = points[..., 0] / z
    y = points[..., 1] / z

    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)

    u = fx * x + cx
    v = fy * y + cy
    return torch.stack((u, v), dim=-1), z


def sample_features(features: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear sample feature maps at pixel-space coordinates."""
    B, C, H, W = features.shape
    u, v = uv[..., 0], uv[..., 1]
    grid_u = 2.0 * (u / (W - 1)) - 1.0
    grid_v = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack((grid_u, grid_v), dim=-1).view(B, -1, 1, 2)
    sampled = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def feature_reprojection_energy(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    depth1: torch.Tensor,
    T1_to_2: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    n_samples: int = 1024,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute a photometric/feature reprojection energy using randomly sampled pixels.
    """
    B, C, H, W = feat1.shape
    device = feat1.device

    ys = torch.randint(0, H, (B, n_samples), device=device)
    xs = torch.randint(0, W, (B, n_samples), device=device)
    flat_idx = ys * W + xs

    feat1_flat = feat1.view(B, C, -1).permute(0, 2, 1)
    f1 = torch.gather(feat1_flat, 1, flat_idx.unsqueeze(-1).expand(-1, -1, C))

    depth_flat = depth1.view(B, -1)
    d1 = torch.gather(depth_flat, 1, flat_idx)

    u = xs.float()
    v = ys.float()
    fx = K1[:, 0, 0].unsqueeze(-1)
    fy = K1[:, 1, 1].unsqueeze(-1)
    cx = K1[:, 0, 2].unsqueeze(-1)
    cy = K1[:, 1, 2].unsqueeze(-1)

    X = torch.stack(((u - cx) / fx * d1, (v - cy) / fy * d1, d1), dim=-1)

    R = T1_to_2[:, :3, :3]
    t = T1_to_2[:, :3, 3]
    X_transformed = (R @ X.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)

    uv2, z2 = project_points(X_transformed, K2)
    f2 = sample_features(feat2, uv2)

    diff = (f1 - f2).pow(2).sum(dim=-1)

    valid = z2 > 1e-6
    valid = valid & (uv2[..., 0] >= 0.0) & (uv2[..., 0] <= W - 1)
    valid = valid & (uv2[..., 1] >= 0.0) & (uv2[..., 1] <= H - 1)
    if mask is not None:
        valid = valid & mask.bool()

    valid_f = valid.float()
    energy = torch.sqrt(diff + 1e-4) * valid_f
    denom = valid_f.sum(dim=-1).clamp(min=1.0)
    energy = energy.sum(dim=-1) / denom
    return energy.mean()


def energy_drop_loss(
    energy_before: torch.Tensor, energy_after: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    return torch.clamp(energy_after - energy_before + margin, min=0.0)
