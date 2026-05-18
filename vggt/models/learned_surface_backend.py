from __future__ import annotations

import torch
from torch import nn


class LearnedSurfaceBackend(nn.Module):
    """Small identity-initialized local surface residual backend.

    This backend is intentionally separate from the high-res crop branch. It is
    meant for local continuous-surface experiments where point/depth/normal
    residuals are conditioned on geometry features and then masked before any
    reinsertion into the dense VGGT point map.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 96, max_delta: float = 1.5e-3) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Linear(hidden_dim, 7)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.65)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
        raw = self.delta(self.net(features))
        apply = torch.sigmoid(raw[..., 6:7])
        return {
            "delta_point": torch.tanh(raw[..., :3]) * self.max_delta * apply,
            "delta_normal": torch.tanh(raw[..., 3:6]) * 0.12 * apply,
            "apply": apply,
        }
