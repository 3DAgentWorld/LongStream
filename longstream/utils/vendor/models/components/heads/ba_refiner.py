import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, nhead: int = 8, dropout: float = 0.0, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class MeanBARefiner(nn.Module):
    """Windowed BA refiner that predicts pose (SE3) and log-depth residuals."""

    def __init__(
        self,
        dim_in: Optional[int] = None,
        dim_hidden: int = 512,
        depth: int = 3,
        nhead: int = 8,
        depth_mode: str = "grid",
        hw: Optional[Tuple[int, int]] = None,
        rank: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth_mode not in {"grid", "lowrank"}:
            raise ValueError(f"Unsupported depth_mode: {depth_mode}")
        if depth_mode == "lowrank" and rank is None:
            raise ValueError("rank must be provided when depth_mode='lowrank'")

        self.depth_mode = depth_mode
        self.default_hw = hw
        self.rank = rank
        self.dim_hidden = dim_hidden

        if dim_in is None:
            self.input_proj = nn.LazyLinear(dim_hidden)
        else:
            self.input_proj = nn.Linear(dim_in, dim_hidden)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim_hidden, nhead=nhead, dropout=dropout, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim_hidden)

        self.pose_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, 6),
        )
        self.pose_gate = nn.Parameter(torch.zeros(1))

        self.depth_hidden = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
        )
        if depth_mode == "lowrank":
            self.depth_proj: Optional[nn.Linear] = nn.Linear(dim_hidden, rank)
        else:
            self.depth_proj = (
                nn.Linear(dim_hidden, hw[0] * hw[1]) if hw is not None else None
            )
        self.depth_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        cam_tokens: torch.Tensor,
        frame_summaries: torch.Tensor,
        pose0_rel_6d: torch.Tensor,
        depth0_log_low: torch.Tensor,
        extras: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = cam_tokens.shape
        flat_depth = depth0_log_low.reshape(B, S, -1)

        features = [cam_tokens, frame_summaries, pose0_rel_6d, flat_depth]
        if extras:
            for value in extras.values():
                if value is None:
                    continue
                if value.dim() == 2:
                    value = value.unsqueeze(1)
                if value.shape[0] != B:
                    if value.shape[0] == 1:
                        value = value.expand(B, *value.shape[1:])
                    else:
                        raise ValueError("Extras must broadcast along batch dimension")
                if value.dim() == 4:
                    value = value.reshape(B, S, -1)
                features.append(value)

        fused = torch.cat(features, dim=-1)
        hidden = self.input_proj(fused)
        for blk in self.blocks:
            hidden = blk(hidden)
        hidden = self.output_norm(hidden)

        dpose = self.pose_head(hidden)
        dpose = torch.tanh(dpose) * torch.sigmoid(self.pose_gate)

        depth_feat = self.depth_hidden(hidden)
        if self.depth_mode == "grid":
            depth_dim = flat_depth.shape[-1]
            if self.depth_proj is None or self.depth_proj.out_features != depth_dim:
                self.depth_proj = nn.Linear(self.dim_hidden, depth_dim)
            self.depth_proj = self.depth_proj.to(
                device=depth_feat.device, dtype=depth_feat.dtype
            )
            ddepth = self.depth_proj(depth_feat)
            h, w = depth0_log_low.shape[-2:]
            ddepth = ddepth.view(B, S, h, w)
        else:
            self.depth_proj = self.depth_proj.to(
                device=depth_feat.device, dtype=depth_feat.dtype
            )
            ddepth = self.depth_proj(depth_feat)

        ddepth = torch.tanh(ddepth) * torch.sigmoid(self.depth_gate)
        return dpose, ddepth
