from __future__ import annotations

import torch
from torch import nn

from models.v310_sparseconv_vggt_token_adapter import SparseConvVGGTTokenAdapter


class SparseConvLatentToVGGTInput(nn.Module):
    """Project sparse latent patch features to VGGT's sparse_prior_tokens API.

    This wrapper is intentionally independent from any concrete SparseConv
    backend. A real SparseConv route can project or splat its latent field to a
    per-view patch grid, then call this module to obtain [B, V, N_patch, C_vggt]
    tokens that are accepted by `VGGT.forward(..., sparse_prior_tokens=...)`.
    """

    def __init__(self, latent_dim: int, vggt_token_dim: int, mode: str = "gated_add") -> None:
        super().__init__()
        self.adapter = SparseConvVGGTTokenAdapter(latent_dim, vggt_token_dim, mode=mode)

    def forward(self, image_tokens: torch.Tensor, sparse_latent_patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.adapter(image_tokens, sparse_latent_patch_tokens)

