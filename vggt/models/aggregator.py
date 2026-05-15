# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class TokenPriorAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, tokens: torch.Tensor, prior_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if prior_tokens is None:
            return tokens
        prior_tokens = prior_tokens.to(device=tokens.device, dtype=tokens.dtype)
        prior_delta = self.proj(self.norm(prior_tokens))
        if prior_delta.dtype != tokens.dtype:
            prior_delta = prior_delta.to(dtype=tokens.dtype)
        gamma = self.gamma
        if gamma.dtype != tokens.dtype or gamma.device != tokens.device:
            gamma = gamma.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + prior_delta * gamma


class SummaryTokenPriorAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.summary_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=max(1, num_heads), batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, tokens: torch.Tensor, summary_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if summary_tokens is None or summary_tokens.numel() == 0:
            return tokens

        summary_tokens = summary_tokens.to(device=tokens.device, dtype=tokens.dtype)
        query = self.token_norm(tokens)
        context = self.summary_norm(summary_tokens)
        prior_delta, _ = self.cross_attn(query, context, context, need_weights=False)
        prior_delta = self.out_proj(prior_delta)
        if prior_delta.dtype != tokens.dtype:
            prior_delta = prior_delta.to(dtype=tokens.dtype)
        gamma = self.gamma
        if gamma.dtype != tokens.dtype or gamma.device != tokens.device:
            gamma = gamma.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + prior_delta * gamma


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_human_prior_fusion=True,
        human_prior_in_chans=11,
        human_prior_hidden_dim=128,
        human_prior_scales=(1, 2, 4),
        enable_human_prior_summary=True,
        human_prior_summary_in_dim=8,
        human_prior_summary_num_heads=4,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.enable_human_prior_fusion = bool(enable_human_prior_fusion)
        self.human_prior_in_chans = int(human_prior_in_chans)
        self.human_prior_hidden_dim = int(max(16, human_prior_hidden_dim))
        if human_prior_scales is None:
            human_prior_scales = (1,)
        self.human_prior_scales = tuple(
            int(scale)
            for scale in human_prior_scales
            if int(scale) >= 1
        ) or (1,)
        self.enable_human_prior_summary = bool(enable_human_prior_summary)
        self.human_prior_summary_in_dim = int(max(1, human_prior_summary_in_dim))
        self.human_prior_summary_num_heads = int(max(1, human_prior_summary_num_heads))
        # Compatibility aliases used by older tooling/reports.  Keep them in
        # sync with the canonical names so audits do not read a missing
        # attribute as "no prior".
        self.human_prior_channels = self.human_prior_in_chans
        self.human_prior_summary_channels = self.human_prior_summary_in_dim

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        if self.enable_human_prior_fusion:
            self.human_prior_patch_embeds = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.human_prior_in_chans, self.human_prior_hidden_dim, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(self.human_prior_hidden_dim, self.human_prior_hidden_dim, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(self.human_prior_hidden_dim, embed_dim, kernel_size=patch_size, stride=patch_size),
                    )
                    for _ in self.human_prior_scales
                ]
            )
            self.input_prior_scale_logits = nn.Parameter(torch.zeros(len(self.human_prior_scales), dtype=torch.float32))
            self.frame_prior_scale_logits = nn.Parameter(
                torch.zeros(depth, len(self.human_prior_scales), dtype=torch.float32)
            )
            self.global_prior_scale_logits = nn.Parameter(
                torch.zeros(depth, len(self.human_prior_scales), dtype=torch.float32)
            )
            self.input_prior_adapter = TokenPriorAdapter(embed_dim)
            self.frame_prior_adapters = nn.ModuleList([TokenPriorAdapter(embed_dim) for _ in range(depth)])
            self.global_prior_adapters = nn.ModuleList([TokenPriorAdapter(embed_dim) for _ in range(depth)])
        else:
            self.human_prior_patch_embeds = nn.ModuleList()
            self.input_prior_scale_logits = None
            self.frame_prior_scale_logits = None
            self.global_prior_scale_logits = None
            self.input_prior_adapter = None
            self.frame_prior_adapters = nn.ModuleList()
            self.global_prior_adapters = nn.ModuleList()

        if self.enable_human_prior_summary:
            self.human_prior_summary_proj = nn.Sequential(
                nn.LayerNorm(self.human_prior_summary_in_dim),
                nn.Linear(self.human_prior_summary_in_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.global_summary_adapters = nn.ModuleList(
                [SummaryTokenPriorAdapter(embed_dim, num_heads=self.human_prior_summary_num_heads) for _ in range(depth)]
            )
        else:
            self.human_prior_summary_proj = None
            self.global_summary_adapters = nn.ModuleList()

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        human_prior_feature_maps: Optional[torch.Tensor] = None,
        human_prior_summary_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        prior_tokens_frame_by_scale = self._build_human_prior_tokens(
            human_prior_feature_maps,
            B=B,
            S=S,
            H=H,
            W=W,
            device=tokens.device,
            token_dtype=tokens.dtype,
        )
        input_prior_tokens = self._combine_multiscale_prior_tokens(
            prior_tokens_frame_by_scale,
            self.input_prior_scale_logits,
        )
        if input_prior_tokens is not None and self.input_prior_adapter is not None:
            tokens = self.input_prior_adapter(tokens, input_prior_tokens)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        prior_tokens_global_by_scale = None
        if prior_tokens_frame_by_scale is not None:
            prior_tokens_global_by_scale = [
                prior_tokens.view(B, S, P, C).view(B, S * P, C)
                for prior_tokens in prior_tokens_frame_by_scale
            ]
        prior_summary_tokens = self._build_human_prior_summary_tokens(
            human_prior_summary_tokens,
            B=B,
            device=tokens.device,
            token_dtype=tokens.dtype,
        )

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, prior_tokens_by_scale=prior_tokens_frame_by_scale
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_idx,
                        pos=pos,
                        prior_tokens_by_scale=prior_tokens_global_by_scale,
                        summary_tokens=prior_summary_tokens,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _build_human_prior_tokens(
        self,
        human_prior_feature_maps: Optional[torch.Tensor],
        *,
        B: int,
        S: int,
        H: int,
        W: int,
        device: torch.device,
        token_dtype: torch.dtype,
    ) -> Optional[List[torch.Tensor]]:
        if (
            not self.enable_human_prior_fusion
            or len(self.human_prior_patch_embeds) == 0
            or human_prior_feature_maps is None
        ):
            return None

        prior = human_prior_feature_maps
        if prior.dim() == 4:
            prior = prior.unsqueeze(2)
        if prior.dim() != 5:
            raise ValueError(
                "human_prior_feature_maps must have shape [B, S, C, H, W] or [B, S, H, W], "
                f"got {tuple(prior.shape)}"
            )
        if prior.shape[0] != B or prior.shape[1] != S:
            raise ValueError(
                "human_prior_feature_maps batch/sequence dims must match images: "
                f"{tuple(prior.shape[:2])} vs {(B, S)}"
            )
        if prior.shape[-2] != H or prior.shape[-1] != W:
            raise ValueError(
                "human_prior_feature_maps spatial size must match images after preprocessing: "
                f"{tuple(prior.shape[-2:])} vs {(H, W)}"
            )

        prior = prior.to(device=device, dtype=torch.float32).view(B * S, prior.shape[2], H, W)
        prior = self._match_human_prior_channels(prior)
        tokens_per_scale = []
        for scale, patch_embed in zip(self.human_prior_scales, self.human_prior_patch_embeds):
            scaled_prior = self._build_single_scale_human_prior(prior, scale)
            prior_patch_tokens = patch_embed(scaled_prior)
            prior_patch_tokens = prior_patch_tokens.flatten(2).transpose(1, 2)
            if prior_patch_tokens.dtype != token_dtype:
                prior_patch_tokens = prior_patch_tokens.to(dtype=token_dtype)
            zeros = prior_patch_tokens.new_zeros(B * S, self.patch_start_idx, prior_patch_tokens.shape[-1])
            tokens_per_scale.append(torch.cat([zeros, prior_patch_tokens], dim=1))
        return tokens_per_scale

    def _match_human_prior_channels(self, prior: torch.Tensor) -> torch.Tensor:
        current_chans = int(prior.shape[1])
        if current_chans == self.human_prior_in_chans:
            return prior
        if current_chans > self.human_prior_in_chans:
            return prior[:, : self.human_prior_in_chans]
        pad = prior.new_zeros(
            prior.shape[0],
            self.human_prior_in_chans - current_chans,
            prior.shape[2],
            prior.shape[3],
        )
        return torch.cat([prior, pad], dim=1)

    def _build_single_scale_human_prior(self, prior: torch.Tensor, scale: int) -> torch.Tensor:
        scale = int(max(1, scale))
        if scale == 1:
            return prior
        pooled = F.avg_pool2d(prior, kernel_size=scale, stride=scale, ceil_mode=False, count_include_pad=False)
        if pooled.shape[-2:] != prior.shape[-2:]:
            pooled = F.interpolate(pooled, size=prior.shape[-2:], mode="bilinear", align_corners=False)
        return pooled

    def _combine_multiscale_prior_tokens(
        self,
        prior_tokens_by_scale: Optional[List[torch.Tensor]],
        scale_logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if prior_tokens_by_scale is None or len(prior_tokens_by_scale) == 0:
            return None
        if len(prior_tokens_by_scale) == 1 or scale_logits is None:
            return prior_tokens_by_scale[0]

        reference = prior_tokens_by_scale[0]
        weights = torch.softmax(scale_logits.to(device=reference.device, dtype=torch.float32), dim=-1)
        combined = reference.new_zeros(reference.shape)
        for scale_idx, prior_tokens in enumerate(prior_tokens_by_scale):
            combined = combined + prior_tokens * weights[scale_idx].to(device=reference.device, dtype=reference.dtype)
        return combined

    def _build_human_prior_summary_tokens(
        self,
        human_prior_summary_tokens: Optional[torch.Tensor],
        *,
        B: int,
        device: torch.device,
        token_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if (
            not self.enable_human_prior_summary
            or self.human_prior_summary_proj is None
            or human_prior_summary_tokens is None
        ):
            return None

        summary = human_prior_summary_tokens
        if summary.dim() == 2:
            summary = summary.unsqueeze(0)
        if summary.dim() != 3:
            raise ValueError(
                "human_prior_summary_tokens must have shape [B, T, F] or [T, F], "
                f"got {tuple(summary.shape)}"
            )
        if summary.shape[0] != B:
            raise ValueError(
                "human_prior_summary_tokens batch dim must match images: "
                f"{summary.shape[0]} vs {B}"
            )
        summary = summary.to(device=device, dtype=torch.float32)
        if summary.shape[-1] > self.human_prior_summary_in_dim:
            summary = summary[..., : self.human_prior_summary_in_dim]
        elif summary.shape[-1] < self.human_prior_summary_in_dim:
            pad = summary.new_zeros(summary.shape[0], summary.shape[1], self.human_prior_summary_in_dim - summary.shape[-1])
            summary = torch.cat([summary, pad], dim=-1)
        projected = self.human_prior_summary_proj(summary)
        if projected.dtype != token_dtype:
            projected = projected.to(dtype=token_dtype)
        return projected

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, prior_tokens_by_scale=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            prior_tokens = self._combine_multiscale_prior_tokens(
                prior_tokens_by_scale,
                None if self.frame_prior_scale_logits is None else self.frame_prior_scale_logits[frame_idx],
            )
            if prior_tokens is not None:
                tokens = self.frame_prior_adapters[frame_idx](tokens, prior_tokens)
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, prior_tokens_by_scale=None, summary_tokens=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            prior_tokens = self._combine_multiscale_prior_tokens(
                prior_tokens_by_scale,
                None if self.global_prior_scale_logits is None else self.global_prior_scale_logits[global_idx],
            )
            if summary_tokens is not None and len(self.global_summary_adapters) > 0:
                tokens = self.global_summary_adapters[global_idx](tokens, summary_tokens)
            if prior_tokens is not None:
                tokens = self.global_prior_adapters[global_idx](tokens, prior_tokens)
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
