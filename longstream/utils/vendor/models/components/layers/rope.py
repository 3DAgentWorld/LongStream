import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D/3D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations. It supports both
    2D (spatial only) and 3D (spatial + temporal) position generation.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return (
            cached_positions.view(1, height * width, 2)
            .expand(batch_size, -1, -1)
            .clone()
        )

    def get_3d_positions(
        self,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generates 3D positions (spatial + temporal) for a batch of frame sequences.

        Args:
            batch_size: Number of samples in the batch (B).
            seq_len: Number of frames in the sequence (S).
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size*seq_len, height*width, 3) containing y, x, t coordinates.
            The temporal coordinate (t) is the frame index, shared by all patches in the same frame.
        """

        spatial_positions = self(1, height, width, device)

        temporal_indices = torch.arange(seq_len, device=device)

        batch_seq_size = batch_size * seq_len

        spatial_expanded = spatial_positions.expand(batch_seq_size, -1, -1)

        num_patches = height * width
        temporal_column = temporal_indices.repeat(batch_size)
        temporal_column = temporal_column.view(batch_seq_size, 1, 1)
        temporal_column = temporal_column.expand(-1, num_patches, -1)

        positions_3d = torch.cat([spatial_expanded, temporal_column], dim=-1)

        return positions_3d


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:

            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)

            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """

        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """

        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert (
            positions.ndim == 3 and positions.shape[-1] == 2
        ), "Positions must have shape (batch_size, n_tokens, 2)"

        feature_dim = tokens.size(-1) // 2

        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )

        return torch.cat((vertical_features, horizontal_features), dim=-1)


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding implementation.

    This module extends 2D RoPE to handle 3D positions (spatial + temporal).
    It applies rotary position embeddings based on y, x, and t coordinates,
    splitting the feature dimension into three parts.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 3D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:

            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)

            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """

        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 3D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 6 (for even distribution).
            positions: Position tensor of shape (batch_size, n_tokens, 3) containing
                      the y, x, and t coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 3D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """

        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert (
            positions.ndim == 3 and positions.shape[-1] == 3
        ), "Positions must have shape (batch_size, n_tokens, 3)"

        total_dim = tokens.size(-1)
        dim_per_axis = total_dim // 3

        if dim_per_axis % 2 != 0:
            dim_per_axis = dim_per_axis - 1

        y_dim = dim_per_axis
        x_dim = dim_per_axis
        t_dim = total_dim - y_dim - x_dim

        if t_dim % 2 != 0:

            x_dim = x_dim - 1
            t_dim = total_dim - y_dim - x_dim

        y_features = tokens[..., :y_dim]
        x_features = tokens[..., y_dim : y_dim + x_dim]
        t_features = tokens[..., y_dim + x_dim :]

        max_position = int(positions.max()) + 1

        cos_comp_y, sin_comp_y = self._compute_frequency_components(
            y_dim, max_position, tokens.device, tokens.dtype
        )
        y_features = self._apply_1d_rope(
            y_features, positions[..., 0], cos_comp_y, sin_comp_y
        )

        cos_comp_x, sin_comp_x = self._compute_frequency_components(
            x_dim, max_position, tokens.device, tokens.dtype
        )
        x_features = self._apply_1d_rope(
            x_features, positions[..., 1], cos_comp_x, sin_comp_x
        )

        cos_comp_t, sin_comp_t = self._compute_frequency_components(
            t_dim, max_position, tokens.device, tokens.dtype
        )
        t_features = self._apply_1d_rope(
            t_features, positions[..., 2], cos_comp_t, sin_comp_t
        )

        return torch.cat((y_features, x_features, t_features), dim=-1)
