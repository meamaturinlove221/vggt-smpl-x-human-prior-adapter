from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AdapterShape:
    batch: int
    views: int
    tokens: int
    channels: int


class SMPLXFeatureEncoder(nn.Module):
    """Encode dense SMPL-X/human prior maps into VGGT-aligned patch tokens.

    This module is deliberately standalone: callers can feed its output into the
    existing VGGT human-prior adapters or use it for adapter smoke tests without
    changing the default VGGT forward path.
    """

    def __init__(
        self,
        in_chans: int,
        token_dim: int = 1024,
        patch_size: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        blocks: list[nn.Module] = []
        width = int(hidden_dim)
        blocks.append(nn.Conv2d(self.in_chans, width, kernel_size=3, padding=1))
        blocks.append(nn.GELU())
        for _ in range(max(0, int(num_layers) - 1)):
            blocks.append(nn.Conv2d(width, width, kernel_size=3, padding=1))
            blocks.append(nn.GELU())
        blocks.append(nn.Conv2d(width, self.token_dim, kernel_size=self.patch_size, stride=self.patch_size))
        self.net = nn.Sequential(*blocks)

    def forward(self, prior_maps: torch.Tensor) -> torch.Tensor:
        """Return patch tokens with shape [B, S, N_patch, C]."""
        if prior_maps.ndim != 5:
            raise ValueError("prior_maps must have shape [B, S, C, H, W]")
        bsz, views, chans, height, width = prior_maps.shape
        if chans != self.in_chans:
            raise ValueError(f"expected {self.in_chans} prior channels, got {chans}")
        x = prior_maps.reshape(bsz * views, chans, height, width)
        tokens = self.net(x).flatten(2).transpose(1, 2).contiguous()
        return tokens.view(bsz, views, tokens.shape[1], self.token_dim)

    def output_shape(self, prior_maps: torch.Tensor) -> AdapterShape:
        tokens = self.forward(prior_maps)
        return AdapterShape(tokens.shape[0], tokens.shape[1], tokens.shape[2], tokens.shape[3])


class GatedTokenInjection(nn.Module):
    """Small collection of shape-aligned token injection modes."""

    def __init__(self, dim: int, mode: str = "add") -> None:
        super().__init__()
        if mode not in {"add", "film"}:
            raise ValueError("mode must be 'add' or 'film'")
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(dim))
        if mode == "film":
            self.beta = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape != prior_tokens.shape:
            raise ValueError(f"token shape mismatch: {tuple(tokens.shape)} vs {tuple(prior_tokens.shape)}")
        prior = self.norm(prior_tokens.to(dtype=tokens.dtype, device=tokens.device))
        gate = self.gamma.to(dtype=tokens.dtype, device=tokens.device)
        if self.mode == "add":
            return tokens + self.delta(prior) * gate
        scale = torch.tanh(self.delta(prior)) * gate
        shift = self.beta(prior) * gate
        return tokens * (1.0 + scale) + shift


class LoRALinear(nn.Module):
    """Non-invasive LoRA wrapper for linear projections used in smoke/search."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_total_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def lora_target_names(module: nn.Module, substrings: Iterable[str] = ("attn.qkv", "attn.proj", "mlp")) -> list[str]:
    targets: list[str] = []
    needles = tuple(str(s) for s in substrings)
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and any(needle in name for needle in needles):
            targets.append(name)
    return targets


class HumanResidualFieldHead(nn.Module):
    """Dense human-region residual field head conditioned on prior feature maps."""

    def __init__(
        self,
        in_chans: int,
        hidden_dim: int = 64,
        max_delta_point: float = 1.5e-3,
        max_delta_depth: float = 1.5e-3,
        max_delta_normal: float = 0.10,
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.max_delta_point = float(max_delta_point)
        self.max_delta_depth = float(max_delta_depth)
        self.max_delta_normal = float(max_delta_normal)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_chans, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 8, kernel_size=1),
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        final = self.net[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, prior_maps: torch.Tensor, human_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if prior_maps.ndim != 5:
            raise ValueError("prior_maps must have shape [B, S, C, H, W]")
        bsz, views, chans, height, width = prior_maps.shape
        if chans != self.in_chans:
            raise ValueError(f"expected {self.in_chans} prior channels, got {chans}")
        raw = self.net(prior_maps.reshape(bsz * views, chans, height, width))
        raw = raw.view(bsz, views, 8, height, width).permute(0, 1, 3, 4, 2).contiguous()
        gate = torch.sigmoid(raw[..., 7:8] + self.gate_bias)
        if human_mask is not None:
            if human_mask.ndim == 4:
                human_mask = human_mask[..., None]
            gate = gate * human_mask.to(dtype=gate.dtype, device=gate.device)
        return {
            "delta_point": torch.tanh(raw[..., 0:3]) * self.max_delta_point * gate,
            "delta_depth": torch.tanh(raw[..., 3:4]) * self.max_delta_depth * gate,
            "delta_normal": torch.tanh(raw[..., 4:7]) * self.max_delta_normal * gate,
            "apply_gate": gate,
        }


def normalize_prior_maps(prior_maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = prior_maps.mean(dim=(-2, -1), keepdim=True)
    std = prior_maps.std(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    return (prior_maps - mean) / std
