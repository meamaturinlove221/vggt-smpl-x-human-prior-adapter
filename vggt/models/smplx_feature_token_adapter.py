from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


AdapterMode = Literal["tokens", "add", "film", "prefix", "cross_attn"]


@dataclass(frozen=True)
class FeatureTokenShape:
    batch: int
    views: int
    patches: int
    channels: int


def _group_norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SMPLXFeatureTokenAdapter(nn.Module):
    """Patchify SMPL-X feature images and smoke-test VGGT token fusion paths."""

    def __init__(
        self,
        in_chans: int,
        c_vggt: int = 1024,
        patch_size: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mode: AdapterMode = "add",
        prefix_tokens: int = 4,
        num_heads: int = 4,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if int(in_chans) <= 0:
            raise ValueError("in_chans must be positive")
        if int(c_vggt) <= 0:
            raise ValueError("c_vggt must be positive")
        if int(patch_size) <= 0:
            raise ValueError("patch_size must be positive")
        if mode not in {"tokens", "add", "film", "prefix", "cross_attn"}:
            raise ValueError("mode must be one of: tokens, add, film, prefix, cross_attn")
        heads = max(1, int(num_heads))
        if int(c_vggt) % heads != 0:
            raise ValueError("c_vggt must be divisible by num_heads")

        self.in_chans = int(in_chans)
        self.c_vggt = int(c_vggt)
        self.patch_size = int(patch_size)
        self.mode = mode
        self.prefix_tokens = int(max(0, prefix_tokens))

        width = int(max(16, hidden_dim))
        layers: list[nn.Module] = [
            nn.Conv2d(self.in_chans, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(width), width),
            nn.GELU(),
        ]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_norm_groups(width), width),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(width, self.c_vggt, kernel_size=self.patch_size, stride=self.patch_size))
        self.patchifier = nn.Sequential(*layers)

        self.prior_norm = nn.LayerNorm(self.c_vggt)
        self.token_norm = nn.LayerNorm(self.c_vggt)
        self.add_proj = nn.Linear(self.c_vggt, self.c_vggt)
        self.film_proj = nn.Linear(self.c_vggt, self.c_vggt * 2)
        self.cross_attn = nn.MultiheadAttention(
            self.c_vggt,
            num_heads=heads,
            batch_first=True,
        )
        self.cross_out = nn.Linear(self.c_vggt, self.c_vggt)
        if self.prefix_tokens > 0:
            self.prefix_proj = nn.Linear(self.c_vggt, self.prefix_tokens * self.c_vggt)
        else:
            self.prefix_proj = None

        self.add_gamma = nn.Parameter(torch.full((self.c_vggt,), float(gate_init)))
        self.film_gamma = nn.Parameter(torch.full((self.c_vggt,), float(gate_init)))
        self.cross_gamma = nn.Parameter(torch.full((self.c_vggt,), float(gate_init)))
        self.prefix_gamma = nn.Parameter(torch.full((self.c_vggt,), float(gate_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.patchifier:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.patchifier[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.xavier_uniform_(final.weight, gain=0.2)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        nn.init.xavier_uniform_(self.add_proj.weight, gain=0.5)
        nn.init.zeros_(self.add_proj.bias)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        nn.init.zeros_(self.cross_out.weight)
        nn.init.zeros_(self.cross_out.bias)
        if self.prefix_proj is not None:
            nn.init.zeros_(self.prefix_proj.weight)
            nn.init.zeros_(self.prefix_proj.bias)

    def _validate_feature_images(self, feature_images: torch.Tensor) -> tuple[int, int, int, int, int]:
        if not torch.is_tensor(feature_images):
            raise TypeError("feature_images must be a torch.Tensor")
        if feature_images.ndim != 5:
            raise ValueError(f"feature_images must have shape [B, V, C, H, W], got {tuple(feature_images.shape)}")
        if not torch.is_floating_point(feature_images):
            raise TypeError("feature_images must be a floating point tensor")
        batch, views, channels, height, width = feature_images.shape
        if channels != self.in_chans:
            raise ValueError(f"expected {self.in_chans} SMPL-X feature channels, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"feature image H/W must be divisible by patch_size={self.patch_size}, got {(height, width)}"
            )
        return batch, views, channels, height, width

    def patchify(self, feature_images: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = self._validate_feature_images(feature_images)
        x = feature_images.reshape(batch * views, channels, height, width)
        tokens = self.patchifier(x).flatten(2).transpose(1, 2).contiguous()
        return tokens.view(batch, views, tokens.shape[1], self.c_vggt)

    def output_shape(self, feature_images: torch.Tensor) -> FeatureTokenShape:
        tokens = self.patchify(feature_images)
        return FeatureTokenShape(tokens.shape[0], tokens.shape[1], tokens.shape[2], tokens.shape[3])

    def _flatten_vggt_tokens(
        self,
        vggt_tokens: torch.Tensor,
        *,
        batch: int,
        views: int,
    ) -> tuple[torch.Tensor, bool]:
        if vggt_tokens.ndim == 4:
            if vggt_tokens.shape[0] != batch or vggt_tokens.shape[1] != views or vggt_tokens.shape[-1] != self.c_vggt:
                raise ValueError(
                    "vggt_tokens must have shape [B, V, N, C_vggt] matching feature tokens, "
                    f"got {tuple(vggt_tokens.shape)}"
                )
            return vggt_tokens.reshape(batch * views, vggt_tokens.shape[2], self.c_vggt), True
        if vggt_tokens.ndim == 3:
            if vggt_tokens.shape[0] != batch * views or vggt_tokens.shape[-1] != self.c_vggt:
                raise ValueError(
                    "flattened vggt_tokens must have shape [B*V, N, C_vggt], "
                    f"got {tuple(vggt_tokens.shape)}"
                )
            return vggt_tokens, False
        raise ValueError("vggt_tokens must have shape [B, V, N, C] or [B*V, N, C]")

    def _restore_vggt_tokens(self, tokens: torch.Tensor, *, batch: int, views: int, restore_4d: bool) -> torch.Tensor:
        if restore_4d:
            return tokens.view(batch, views, tokens.shape[1], tokens.shape[2])
        return tokens

    def _patch_slice(self, tokens: torch.Tensor, num_patch_tokens: int, patch_start_idx: int) -> slice:
        start = int(patch_start_idx)
        if start < 0:
            raise ValueError("patch_start_idx must be non-negative")
        end = start + int(num_patch_tokens)
        if tokens.shape[1] < end:
            raise ValueError(
                "vggt_tokens does not contain enough patch tokens for the SMPL-X tokens: "
                f"needed end index {end}, got token length {tokens.shape[1]}"
            )
        return slice(start, end)

    def adapt_tokens(
        self,
        vggt_tokens: torch.Tensor,
        smplx_patch_tokens: torch.Tensor,
        *,
        mode: AdapterMode | None = None,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        if smplx_patch_tokens.ndim != 4 or smplx_patch_tokens.shape[-1] != self.c_vggt:
            raise ValueError(
                "smplx_patch_tokens must have shape [B, V, N_patch, C_vggt], "
                f"got {tuple(smplx_patch_tokens.shape)}"
            )
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in {"add", "film", "prefix", "cross_attn"}:
            raise ValueError("adapt_tokens mode must be one of: add, film, prefix, cross_attn")

        batch, views, num_patch_tokens, _ = smplx_patch_tokens.shape
        tokens, restore_4d = self._flatten_vggt_tokens(vggt_tokens, batch=batch, views=views)
        prior = smplx_patch_tokens.reshape(batch * views, num_patch_tokens, self.c_vggt).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )

        if selected_mode == "prefix":
            if self.prefix_proj is None:
                raise ValueError("prefix mode requires prefix_tokens > 0")
            summary = prior.mean(dim=1)
            prefix = self.prefix_proj(self.prior_norm(summary)).view(batch * views, self.prefix_tokens, self.c_vggt)
            prefix = prefix * self.prefix_gamma.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            start = int(patch_start_idx)
            if start < 0 or start > tokens.shape[1]:
                raise ValueError("patch_start_idx must lie within vggt_tokens for prefix insertion")
            tokens = torch.cat([tokens[:, :start], prefix, tokens[:, start:]], dim=1)
            return self._restore_vggt_tokens(tokens, batch=batch, views=views, restore_4d=restore_4d)

        patch_slice = self._patch_slice(tokens, num_patch_tokens, int(patch_start_idx))
        token_patch = tokens[:, patch_slice]
        if selected_mode == "add":
            delta = self.add_proj(self.prior_norm(prior))
            gamma = self.add_gamma.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            updated_patch = token_patch + delta * gamma
        elif selected_mode == "film":
            scale_shift = self.film_proj(self.prior_norm(prior))
            scale, shift = scale_shift.chunk(2, dim=-1)
            gamma = self.film_gamma.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            updated_patch = token_patch * (1.0 + torch.tanh(scale) * gamma) + shift * gamma
        else:
            query = self.token_norm(token_patch)
            context = self.prior_norm(prior)
            delta, _ = self.cross_attn(query, context, context, need_weights=False)
            delta = self.cross_out(delta)
            gamma = self.cross_gamma.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            updated_patch = token_patch + delta * gamma

        tokens = tokens.clone()
        tokens[:, patch_slice] = updated_patch
        return self._restore_vggt_tokens(tokens, batch=batch, views=views, restore_4d=restore_4d)

    def forward(
        self,
        feature_images: torch.Tensor,
        vggt_tokens: torch.Tensor | None = None,
        *,
        mode: AdapterMode | None = None,
        patch_start_idx: int = 0,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | str]:
        smplx_tokens = self.patchify(feature_images)
        selected_mode = self.mode if mode is None else mode
        if selected_mode == "tokens" or vggt_tokens is None:
            if return_dict:
                return {"smplx_patch_tokens": smplx_tokens}
            return smplx_tokens
        fused = self.adapt_tokens(
            vggt_tokens,
            smplx_tokens,
            mode=selected_mode,
            patch_start_idx=patch_start_idx,
        )
        if return_dict:
            return {
                "tokens": fused,
                "smplx_patch_tokens": smplx_tokens,
                "mode": selected_mode,
            }
        return fused
