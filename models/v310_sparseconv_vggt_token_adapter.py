from __future__ import annotations

import torch
from torch import nn


class SparseConvVGGTTokenAdapter(nn.Module):
    """Shape-aligned adapter for projected SparseConv features.

    The module is intentionally small: it accepts a per-view patch tensor from a
    sparse latent field projection and returns token-shaped features that can be
    injected into a frozen VGGT token stream by gated add, FiLM, or prefix mode.
    The controller uses this module for the V310 diagnostic branch; production
    VGGT integration still has to wire the returned tensors into the real
    backbone.
    """

    def __init__(self, in_dim: int, token_dim: int, mode: str = "gated_add", prefix_tokens: int = 8) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.token_dim = int(token_dim)
        self.mode = mode
        self.prefix_tokens = int(prefix_tokens)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.film = nn.Linear(self.token_dim, self.token_dim * 2)
        self.prefix_pool = nn.AdaptiveAvgPool1d(self.prefix_tokens)

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, image_tokens: torch.Tensor, sparse_patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return injected tokens and diagnostics.

        Args:
            image_tokens: [B, V, N, C]
            sparse_patch_tokens: [B, V, N, F]
        """
        if image_tokens.ndim != 4 or sparse_patch_tokens.ndim != 4:
            raise ValueError("image_tokens and sparse_patch_tokens must both be [B,V,N,C/F]")
        if image_tokens.shape[:3] != sparse_patch_tokens.shape[:3]:
            raise ValueError("Token grids are not shape aligned")
        prior = self.proj(sparse_patch_tokens)
        gate = self.gate
        if self.mode == "gated_add":
            out = image_tokens + gate * prior
            return {"tokens": out, "prior_tokens": prior, "gate": gate}
        if self.mode == "film":
            gamma, beta = self.film(prior).chunk(2, dim=-1)
            out = image_tokens * (1.0 + gate * torch.tanh(gamma)) + gate * beta
            return {"tokens": out, "prior_tokens": prior, "gate": gate, "gamma": gamma, "beta": beta}
        if self.mode == "prefix":
            b, v, n, c = prior.shape
            pooled = self.prefix_pool(prior.reshape(b * v, n, c).transpose(1, 2)).transpose(1, 2)
            prefix = pooled.reshape(b, v, self.prefix_tokens, c)
            return {"tokens": image_tokens, "prefix_tokens": prefix, "prior_tokens": prior, "gate": gate}
        raise ValueError(f"Unsupported adapter mode: {self.mode}")

