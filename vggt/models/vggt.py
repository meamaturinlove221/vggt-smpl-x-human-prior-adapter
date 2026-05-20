# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.highres_crop_geometry import HighResCropGeometryBranch


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        patch_embed="dinov2_vitl14_reg",
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_normal=False,
        enable_track=True,
        enable_human_prior_fusion=True,
        human_prior_in_chans=17,
        human_prior_hidden_dim=128,
        human_prior_scales=(1, 2, 4),
        enable_human_prior_summary=True,
        human_prior_summary_in_dim=12,
        human_prior_summary_num_heads=4,
        # Backward-compatible aliases used by earlier SMPL-X/VGGT configs and
        # checkpoint payloads.  The canonical names are the *_in_chans /
        # *_in_dim arguments above, but silently ignoring these older keys was
        # enough to make prior-enabled runs fall back to a truncated/default
        # prior path.
        human_prior_channels=None,
        human_prior_summary_channels=None,
        human_prior_multi_scale_factors=None,
        human_prior_gate_init=None,
        human_prior_enable_input_fusion=None,
        human_prior_enable_frame_fusion=None,
        human_prior_enable_global_fusion=None,
        human_prior_enable_summary_fusion=None,
        enable_highres_crop_geometry=False,
        highres_crop_feature_dim=32,
        highres_crop_hidden_dim=128,
    ):
        super().__init__()

        if human_prior_channels is not None:
            human_prior_in_chans = human_prior_channels
        if human_prior_summary_channels is not None:
            human_prior_summary_in_dim = human_prior_summary_channels
        if human_prior_multi_scale_factors is not None:
            human_prior_scales = human_prior_multi_scale_factors
        if human_prior_enable_summary_fusion is not None:
            enable_human_prior_summary = bool(human_prior_enable_summary_fusion)
        if any(flag is not None for flag in (
            human_prior_enable_input_fusion,
            human_prior_enable_frame_fusion,
            human_prior_enable_global_fusion,
        )):
            enable_human_prior_fusion = bool(
                (human_prior_enable_input_fusion if human_prior_enable_input_fusion is not None else True)
                or (human_prior_enable_frame_fusion if human_prior_enable_frame_fusion is not None else True)
                or (human_prior_enable_global_fusion if human_prior_enable_global_fusion is not None else True)
            )

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            patch_embed=patch_embed,
            embed_dim=embed_dim,
            enable_human_prior_fusion=enable_human_prior_fusion,
            human_prior_in_chans=human_prior_in_chans,
            human_prior_hidden_dim=human_prior_hidden_dim,
            human_prior_scales=human_prior_scales,
            enable_human_prior_summary=enable_human_prior_summary,
            human_prior_summary_in_dim=human_prior_summary_in_dim,
            human_prior_summary_num_heads=human_prior_summary_num_heads,
        )
        if human_prior_gate_init is not None:
            self._init_human_prior_gates(float(human_prior_gate_init))

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.normal_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="norm", conf_activation="expp1") if enable_normal else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.highres_crop_geometry = (
            HighResCropGeometryBranch(feature_dim=highres_crop_feature_dim, hidden_dim=highres_crop_hidden_dim)
            if enable_highres_crop_geometry
            else None
        )

    def _init_human_prior_gates(self, value: float) -> None:
        if self.aggregator.input_prior_adapter is not None:
            nn.init.constant_(self.aggregator.input_prior_adapter.gamma, value)
        for adapter in self.aggregator.frame_prior_adapters:
            nn.init.constant_(adapter.gamma, value)
        for adapter in self.aggregator.global_prior_adapters:
            nn.init.constant_(adapter.gamma, value)
        for adapter in self.aggregator.global_summary_adapters:
            nn.init.constant_(adapter.gamma, value)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        human_prior_feature_maps: torch.Tensor = None,
        human_prior_summary_tokens: torch.Tensor = None,
        prior_maps: torch.Tensor = None,
        prior_summary_tokens: torch.Tensor = None,
        sparse_prior_tokens: torch.Tensor = None,
        highres_crop_features: torch.Tensor = None,
        highres_crop_indices: torch.Tensor = None,
        highres_crop_weights: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - normal (torch.Tensor): Predicted surface normal maps with shape [B, S, H, W, 3]
                - normal_conf (torch.Tensor): Confidence scores for normal predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        if human_prior_feature_maps is None and prior_maps is not None:
            human_prior_feature_maps = prior_maps
        if human_prior_summary_tokens is None and prior_summary_tokens is not None:
            human_prior_summary_tokens = prior_summary_tokens

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            if human_prior_feature_maps is not None and len(human_prior_feature_maps.shape) in (3, 4):
                human_prior_feature_maps = human_prior_feature_maps.unsqueeze(0)
            if human_prior_summary_tokens is not None and len(human_prior_summary_tokens.shape) == 2:
                human_prior_summary_tokens = human_prior_summary_tokens.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            human_prior_feature_maps=human_prior_feature_maps,
            human_prior_summary_tokens=human_prior_summary_tokens,
            sparse_prior_tokens=sparse_prior_tokens,
        )

        predictions = {}

        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.normal_head is not None:
                normal, normal_conf = self.normal_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["normal"] = normal
                predictions["normal_conf"] = normal_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if (
                self.highres_crop_geometry is not None
                and highres_crop_features is not None
                and highres_crop_indices is not None
                and "world_points" in predictions
            ):
                crop_updates = self.highres_crop_geometry(
                    world_points=predictions["world_points"],
                    depth=predictions.get("depth"),
                    normal=predictions.get("normal"),
                    crop_features=highres_crop_features,
                    crop_indices=highres_crop_indices,
                    crop_weights=highres_crop_weights,
                )
                predictions["world_points"] = crop_updates["world_points"]
                predictions["crop_delta_point"] = crop_updates["crop_delta_point"]
                predictions["crop_apply_gate"] = crop_updates["crop_apply_gate"]
                predictions["crop_uncertainty"] = crop_updates["crop_uncertainty"]
                if "depth" in crop_updates:
                    predictions["depth"] = crop_updates["depth"]
                    predictions["crop_delta_depth"] = crop_updates["crop_delta_depth"]
                if "normal" in crop_updates:
                    predictions["normal"] = crop_updates["normal"]
                    predictions["crop_delta_normal"] = crop_updates["crop_delta_normal"]

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
