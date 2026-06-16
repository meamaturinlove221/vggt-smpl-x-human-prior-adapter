from __future__ import annotations

import torch
from torch import nn


class HumanHandTokenResidualDecoder(nn.Module):
    """Camera-aware VGGT-token hand decoder used by V11 research routes.

    This module intentionally contains no success logic. Training/evaluation
    scripts must prove real-token margins and wrist/finger visual gates before
    any output can be promoted.
    """

    def __init__(self, token_dim: int = 2048, hidden_dim: int = 256, hand_tokens: int = 32, surface_points: int = 512) -> None:
        super().__init__()
        self.left_hand_tokens = nn.Parameter(torch.randn(hand_tokens, hidden_dim) * 0.02)
        self.right_hand_tokens = nn.Parameter(torch.randn(hand_tokens, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.camera_proj = nn.Linear(16, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.surface_points = int(surface_points)
        self.surface_head = nn.Linear(hidden_dim, surface_points * 3)
        self.visibility_head = nn.Linear(hidden_dim, surface_points)
        self.confidence_head = nn.Linear(hidden_dim, surface_points)
        self.wrist_bridge_head = nn.Linear(hidden_dim, 64 * 3)

    def _decode_side(self, hand_seed: torch.Tensor, tokens: torch.Tensor, camera_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = tokens.shape[0]
        seed = hand_seed.unsqueeze(0).expand(batch, -1, -1) + camera_embed.unsqueeze(1)
        attended, _ = self.cross_attention(seed, tokens, tokens)
        latent = self.refine(attended).mean(dim=1)
        surface = self.surface_head(latent).view(batch, self.surface_points, 3)
        wrist_bridge = self.wrist_bridge_head(latent).view(batch, 64, 3)
        return {
            "surface": surface,
            "wrist_bridge": wrist_bridge,
            "visibility": torch.sigmoid(self.visibility_head(latent)),
            "confidence": torch.sigmoid(self.confidence_head(latent)),
        }

    def forward(self, vggt_roi_tokens: torch.Tensor, camera_tokens: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if vggt_roi_tokens.ndim != 3:
            raise ValueError("vggt_roi_tokens must have shape [batch, token, channel]")
        if camera_tokens.ndim != 2:
            raise ValueError("camera_tokens must have shape [batch, 16]")
        tokens = self.token_proj(vggt_roi_tokens)
        camera_embed = self.camera_proj(camera_tokens)
        return {
            "left": self._decode_side(self.left_hand_tokens, tokens, camera_embed),
            "right": self._decode_side(self.right_hand_tokens, tokens, camera_embed),
        }
