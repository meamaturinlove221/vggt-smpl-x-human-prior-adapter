from __future__ import annotations

import torch
from torch import nn


class HumanHairStrandGaussian(nn.Module):
    """Trainable scalp-root strand Gaussian parameterization for V11 hair routes.

    It provides root proposal, strand direction, scale, opacity, visibility, and
    confidence heads. Downstream scripts must still prove topology/visual gates;
    this module never implies a pass by itself.
    """

    def __init__(self, token_dim: int = 2048, hidden_dim: int = 192, root_count: int = 256, segments: int = 6) -> None:
        super().__init__()
        self.root_count = int(root_count)
        self.segments = int(segments)
        self.root_seed = nn.Parameter(torch.randn(root_count, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.scaffold_proj = nn.Linear(3, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=6, batch_first=True)
        self.refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.root_offset = nn.Linear(hidden_dim, 3)
        self.direction = nn.Linear(hidden_dim, segments * 3)
        self.scale = nn.Linear(hidden_dim, segments * 3)
        self.opacity = nn.Linear(hidden_dim, segments)
        self.visibility = nn.Linear(hidden_dim, segments)
        self.confidence = nn.Linear(hidden_dim, segments)
        self.head_shell_leakage = nn.Linear(hidden_dim, 1)

    def forward(self, vggt_head_tokens: torch.Tensor, scalp_roots: torch.Tensor) -> dict[str, torch.Tensor]:
        if vggt_head_tokens.ndim != 3:
            raise ValueError("vggt_head_tokens must have shape [batch, token, channel]")
        if scalp_roots.ndim != 3 or scalp_roots.shape[-1] != 3:
            raise ValueError("scalp_roots must have shape [batch, root, 3]")
        batch = vggt_head_tokens.shape[0]
        tokens = self.token_proj(vggt_head_tokens)
        roots = scalp_roots[:, : self.root_count]
        if roots.shape[1] < self.root_count:
            pad = roots[:, -1:].expand(batch, self.root_count - roots.shape[1], 3)
            roots = torch.cat([roots, pad], dim=1)
        queries = self.root_seed.unsqueeze(0).expand(batch, -1, -1) + self.scaffold_proj(roots)
        attended, _ = self.cross_attention(queries, tokens, tokens)
        latent = self.refine(attended)
        strand_dirs = self.direction(latent).view(batch, self.root_count, self.segments, 3)
        scales = torch.nn.functional.softplus(self.scale(latent)).view(batch, self.root_count, self.segments, 3)
        opacities = torch.sigmoid(self.opacity(latent))
        visibility = torch.sigmoid(self.visibility(latent))
        confidence = torch.sigmoid(self.confidence(latent))
        root_positions = roots + self.root_offset(latent)
        cumulative = torch.cumsum(strand_dirs, dim=2)
        strand_points = root_positions.unsqueeze(2) + cumulative
        return {
            "root_positions": root_positions,
            "strand_points": strand_points,
            "scale": scales,
            "opacity": opacities,
            "visibility": visibility,
            "confidence": confidence,
            "head_shell_leakage": torch.sigmoid(self.head_shell_leakage(latent)).squeeze(-1),
        }
