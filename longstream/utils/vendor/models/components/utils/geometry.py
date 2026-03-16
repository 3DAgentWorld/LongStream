import os
import torch
import numpy as np


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    point_mask = depth_map > eps

    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def normalize_depth_using_non_zero_pixels(depth, return_norm_factor=False):
    """
    Normalize the depth by the average depth of non-zero depth pixels.
    Compatible with MapAnything's implementation.

    Args:
        depth (torch.Tensor): Depth tensor of size [B, H, W, 1].
        return_norm_factor (bool): Whether to return the normalization factor.

    Returns:
        normalized_depth (torch.Tensor): Normalized depth tensor.
        norm_factor (torch.Tensor): Norm factor tensor of size B (if return_norm_factor=True).
    """
    assert depth.ndim == 4 and depth.shape[3] == 1

    valid_depth_mask = depth > 0
    valid_sum = torch.sum(depth * valid_depth_mask, dim=(1, 2, 3))
    valid_count = torch.sum(valid_depth_mask, dim=(1, 2, 3))

    norm_factor = valid_sum / (valid_count + 1e-8)
    while norm_factor.ndim < depth.ndim:
        norm_factor.unsqueeze_(-1)

    norm_factor = norm_factor.clip(min=1e-8)
    normalized_depth = depth / norm_factor

    output = (
        (normalized_depth, norm_factor.squeeze(-1).squeeze(-1).squeeze(-1))
        if return_norm_factor
        else normalized_depth
    )

    return output


def normalize_pose_translations(pose_translations, return_norm_factor=False):
    """
    Normalize the pose translations by the average norm of the non-zero pose translations.
    Compatible with MapAnything's implementation.

    Args:
        pose_translations (torch.Tensor): Pose translations tensor of size [B, V, 3].
            B is the batch size, V is the number of views.
        return_norm_factor (bool): Whether to return the normalization factor.

    Returns:
        normalized_pose_translations (torch.Tensor): Normalized pose translations tensor of size [B, V, 3].
        norm_factor (torch.Tensor): Norm factor tensor of size B (if return_norm_factor=True).
    """
    assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3

    pose_translations_dis = pose_translations.norm(dim=-1)
    non_zero_pose_translations_dis = pose_translations_dis > 0

    sum_of_all_views_pose_translations = pose_translations_dis.sum(dim=1)
    count_of_all_views_with_non_zero_pose_translations = (
        non_zero_pose_translations_dis.sum(dim=1)
    )
    norm_factor = sum_of_all_views_pose_translations / (
        count_of_all_views_with_non_zero_pose_translations + 1e-8
    )

    norm_factor = norm_factor.clip(min=1e-8)
    normalized_pose_translations = pose_translations / norm_factor.unsqueeze(
        -1
    ).unsqueeze(-1)

    output = (
        (normalized_pose_translations, norm_factor)
        if return_norm_factor
        else normalized_pose_translations
    )

    return output


def apply_log_to_norm(input_data):
    """
    Normalize the input data and apply a logarithmic transformation based on the normalization factor.
    Compatible with MapAnything's implementation.

    Args:
        input_data (torch.Tensor): The input tensor to be normalized and transformed.

    Returns:
        torch.Tensor: The transformed tensor after normalization and logarithmic scaling.
    """
    org_d = input_data.norm(dim=-1, keepdim=True)
    input_data = input_data / org_d.clip(min=1e-8)
    input_data = input_data * torch.log1p(org_d)
    return input_data


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """

    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:

        R_transposed = np.transpose(R, (0, 2, 1))

        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
