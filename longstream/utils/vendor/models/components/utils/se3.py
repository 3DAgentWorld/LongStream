from typing import Tuple

import torch


def _batch_eye(
    batch_shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    eye = torch.eye(3, dtype=dtype, device=device)
    view_shape = (1,) * len(batch_shape) + (3, 3)
    return eye.view(view_shape).expand(batch_shape + (3, 3)).clone()


def _skew(w: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = w.unbind(dim=-1)
    zeros = torch.zeros_like(wx)
    return torch.stack(
        (
            torch.stack((zeros, -wz, wy), dim=-1),
            torch.stack((wz, zeros, -wx), dim=-1),
            torch.stack((-wy, wx, zeros), dim=-1),
        ),
        dim=-2,
    )


def _taylor_A(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta * theta
    return torch.where(
        theta2 > 1e-12,
        torch.sin(theta) / theta,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
    )


def _taylor_B(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta * theta
    return torch.where(
        theta2 > 1e-12,
        (1.0 - torch.cos(theta)) / theta2,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
    )


def _taylor_C(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta * theta
    return torch.where(
        theta2 > 1e-12,
        (theta - torch.sin(theta)) / (theta2 * theta),
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
    )


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    v = xi[..., :3]
    w = xi[..., 3:]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    batch_shape = xi.shape[:-1]

    theta_safe = torch.where(theta > 1e-9, theta, torch.ones_like(theta))
    w_hat = torch.where(theta > 1e-9, w / theta_safe, torch.zeros_like(w))
    W = _skew(w_hat)

    A = _taylor_A(theta)[..., None]
    B = _taylor_B(theta)[..., None]
    C = _taylor_C(theta)[..., None]

    eye = _batch_eye(batch_shape, dtype=xi.dtype, device=xi.device)
    W2 = W @ W
    R = eye + A * W + B * W2
    V = eye + B * W + C * W2
    t = (V @ v.unsqueeze(-1)).squeeze(-1)

    T = torch.zeros(*batch_shape, 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def _rotation_log(R: torch.Tensor) -> torch.Tensor:
    from longstream.utils.vendor.dust3r.utils.camera import matrix_to_quaternion

    q = matrix_to_quaternion(R)
    qw = q[..., 0]
    q_xyz = q[..., 1:]
    sin_half = torch.linalg.norm(q_xyz, dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(sin_half.squeeze(-1), qw.clamp(min=1e-9)).unsqueeze(-1)

    mask = sin_half > 1e-7
    axis = torch.zeros_like(q_xyz)
    axis[mask] = q_xyz[mask] / sin_half[mask]

    w = torch.zeros_like(q_xyz)
    w[mask] = axis[mask] * theta[mask]
    w[~mask] = 2.0 * q_xyz[~mask]
    return w


def se3_log(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = _rotation_log(R)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    batch_shape = w.shape[:-1]

    theta_safe = torch.where(theta > 1e-9, theta, torch.ones_like(theta))
    w_hat = torch.where(theta > 1e-9, w / theta_safe, torch.zeros_like(w))
    W = _skew(w_hat)

    B = _taylor_B(theta)[..., None]
    C = _taylor_C(theta)[..., None]
    eye = _batch_eye(batch_shape, dtype=T.dtype, device=T.device)
    V = eye + B * W + C * (W @ W)

    v = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat((v, w), dim=-1)


def compose(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    return T1 @ T2


def inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def identity(batch_shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
    eye4 = torch.eye(4, dtype=dtype or torch.float32, device=device)
    view_shape = (1,) * len(batch_shape) + (4, 4)
    return eye4.view(view_shape).expand(batch_shape + (4, 4)).clone()
