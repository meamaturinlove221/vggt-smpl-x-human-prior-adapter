# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor, isfinite

HYDRA_META_KEYS = {"_target_", "_recursive_", "_convert_", "_partial_"}
LOSS_WEIGHT_SCHEDULE_META_KEYS = {
    "weight_stage2_start",
    "weight_stage2_end",
    "weight_stage2_value",
}
REGRESSION_LOSS_EXTRA_ALLOWED_KEYS = {
    "anchor_conditioned_disagreement_cameras",
    "anchor_conditioned_disagreement_conf_scale",
    "anchor_conditioned_disagreement_reg_scale",
    "anchor_conditioned_disagreement_train_only",
}


def _loss_call_kwargs(loss_cfg):
    if loss_cfg is None:
        return {}
    return {
        key: value
        for key, value in loss_cfg.items()
        if key not in HYDRA_META_KEYS
        and key != "weight"
        and key not in LOSS_WEIGHT_SCHEDULE_META_KEYS
    }


def _collect_regression_loss_extra_kwargs(kwargs):
    filtered_kwargs = {}
    unsupported_keys = []
    for key, value in kwargs.items():
        if key in HYDRA_META_KEYS or key == "weight":
            continue
        if key in REGRESSION_LOSS_EXTRA_ALLOWED_KEYS:
            filtered_kwargs[key] = value
            continue
        unsupported_keys.append(key)

    if unsupported_keys:
        unsupported_keys = ", ".join(sorted(unsupported_keys))
        raise TypeError(
            "compute_depth_loss received unsupported regression kwargs: "
            f"{unsupported_keys}"
        )

    return filtered_kwargs


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, unproject_geometry=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.unproject_geometry = unproject_geometry

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(
                predictions, batch, **_loss_call_kwargs(self.camera)
            )
            camera_weight = resolve_two_stage_scalar(
                self.camera["weight"],
                batch,
                start=self.camera.get("weight_stage2_start"),
                end=self.camera.get("weight_stage2_end"),
                stage2_value=self.camera.get("weight_stage2_value"),
            )
            camera_loss = camera_loss_dict["loss_camera"] * camera_weight
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(
                predictions, batch, **_loss_call_kwargs(self.depth)
            )
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_weight = resolve_two_stage_scalar(
                self.depth["weight"],
                batch,
                start=self.depth.get("weight_stage2_start"),
                end=self.depth.get("weight_stage2_end"),
                stage2_value=self.depth.get("weight_stage2_value"),
            )
            depth_loss = depth_loss * depth_weight
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(
                predictions, batch, **_loss_call_kwargs(self.point)
            )
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_weight = resolve_two_stage_scalar(
                self.point["weight"],
                batch,
                start=self.point.get("weight_stage2_start"),
                end=self.point.get("weight_stage2_end"),
                stage2_value=self.point.get("weight_stage2_value"),
            )
            point_loss = point_loss * point_weight
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Geometry-chain auxiliary loss built from predicted depth + predicted camera.
        if self.unproject_geometry is not None and "depth" in predictions and "pose_enc" in predictions:
            unproject_geometry_loss_dict = compute_unproject_geometry_loss(
                predictions, batch, **_loss_call_kwargs(self.unproject_geometry)
            )
            unproject_geometry_weight = resolve_scheduled_loss_weight(
                self.unproject_geometry["weight"],
                self.unproject_geometry,
                batch,
            )
            unproject_geometry_weight = resolve_two_stage_scalar(
                unproject_geometry_weight,
                batch,
                start=self.unproject_geometry.get("weight_stage2_start"),
                end=self.unproject_geometry.get("weight_stage2_end"),
                stage2_value=self.unproject_geometry.get("weight_stage2_value"),
            )
            unproject_geometry_loss = (
                unproject_geometry_loss_dict["loss_unproject_geometry"] * unproject_geometry_weight
            )
            total_loss = total_loss + unproject_geometry_loss
            loss_dict.update(unproject_geometry_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
        
        loss_dict["objective"] = total_loss

        return loss_dict


def resolve_scheduled_loss_weight(base_weight, loss_cfg, batch):
    """
    Optionally applies a linear warmup schedule to an auxiliary loss weight.

    This is intentionally config-driven and fully backward compatible:
    if no warmup keys are provided, the original constant weight is returned.
    """
    warmup_end = loss_cfg.get("warmup_end", None)
    if warmup_end is None:
        return base_weight

    progress = batch.get("train_progress", None)
    if progress is None:
        return base_weight

    warmup_start = float(loss_cfg.get("warmup_start", 0.0))
    warmup_end = float(warmup_end)
    init_factor = float(loss_cfg.get("warmup_init_factor", 0.0))

    if warmup_end <= warmup_start:
        return base_weight

    progress = float(progress)
    if progress <= warmup_start:
        factor = init_factor
    elif progress >= warmup_end:
        factor = 1.0
    else:
        alpha = (progress - warmup_start) / (warmup_end - warmup_start)
        factor = init_factor + (1.0 - init_factor) * alpha

    return float(base_weight) * factor


def resolve_two_stage_alpha(batch, start=None, end=None):
    progress = None if batch is None else batch.get("train_progress", None)
    if progress is None:
        return None

    start = 0.5 if start is None else float(start)
    end = start if end is None else float(end)
    if end < start:
        end = start

    progress = float(progress)
    if progress <= start:
        return 0.0
    if end <= start or progress >= end:
        return 1.0
    return (progress - start) / (end - start)


def resolve_two_stage_scalar(base_value, batch, *, start=None, end=None, stage2_value=None):
    if stage2_value is None:
        return float(base_value)
    alpha = resolve_two_stage_alpha(batch, start=start, end=end)
    if alpha is None or alpha <= 0.0:
        return float(base_value)
    if alpha >= 1.0:
        return float(stage2_value)
    return float(base_value) + alpha * (float(stage2_value) - float(base_value))


def resolve_two_stage_label_scalars(base_values, batch, *, stage2_values=None, start=None, end=None):
    base_values = {
        str(label).strip(): float(scale)
        for label, scale in dict(base_values or {}).items()
        if str(label).strip()
    }
    stage2_values = {
        str(label).strip(): float(scale)
        for label, scale in dict(stage2_values or {}).items()
        if str(label).strip()
    }
    if not stage2_values:
        return base_values

    alpha = resolve_two_stage_alpha(batch, start=start, end=end)
    if alpha is None or alpha <= 0.0:
        return base_values
    if alpha >= 1.0:
        merged = dict(base_values)
        merged.update(stage2_values)
        return merged

    resolved = dict(base_values)
    for label in sorted(set(base_values) | set(stage2_values)):
        base_scale = float(base_values.get(label, 1.0))
        stage2_scale = float(stage2_values.get(label, base_scale))
        resolved[label] = base_scale + alpha * (stage2_scale - base_scale)
    return resolved


def _assemble_camera_component_dict(loss_T, loss_R, loss_FL):
    return {
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_FL": loss_FL,
    }


def _resolve_loss_fl_isolation_scale(loss_fl_isolation_scale):
    loss_fl_isolation_scale = float(loss_fl_isolation_scale)
    if not isfinite(loss_fl_isolation_scale):
        raise ValueError("loss_fl_isolation_scale must be finite.")
    return loss_fl_isolation_scale


def _resolve_loss_t_isolation_scale(loss_t_isolation_scale):
    loss_t_isolation_scale = float(loss_t_isolation_scale)
    if not isfinite(loss_t_isolation_scale):
        raise ValueError("loss_t_isolation_scale must be finite.")
    return loss_t_isolation_scale


def _compute_total_camera_loss(
    camera_component_dict,
    *,
    weight_trans,
    weight_rot,
    weight_focal,
    loss_t_isolation_scale=1.0,
    loss_fl_isolation_scale=1.0,
):
    isolated_t_scale = _resolve_loss_t_isolation_scale(loss_t_isolation_scale)
    isolated_fl_scale = _resolve_loss_fl_isolation_scale(loss_fl_isolation_scale)
    return (
        camera_component_dict["loss_T"] * isolated_t_scale * weight_trans
        + camera_component_dict["loss_R"] * weight_rot
        + camera_component_dict["loss_FL"] * isolated_fl_scale * weight_focal
    )


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    sample_manifest_applied_scale=1.0,
    sample_manifest_applied_trans_scale=1.0,
    sample_manifest_applied_rot_scale=1.0,
    sample_manifest_applied_focal_scale=1.0,
    sample_manifest_label_focal_scales=None,
    sample_manifest_schedule_start=None,
    sample_manifest_schedule_end=None,
    sample_manifest_applied_scale_stage2=None,
    sample_manifest_applied_trans_scale_stage2=None,
    sample_manifest_applied_rot_scale_stage2=None,
    sample_manifest_applied_focal_scale_stage2=None,
    sample_manifest_label_focal_scales_stage2=None,
    sample_manifest_applied_train_only=True,
    loss_t_isolation_scale=1.0,
    loss_fl_isolation_scale=1.0,
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames that actually have enough supervised points.
    # This matters for source-only raw views, which intentionally carry
    # GT cameras but zero geometry supervision.
    valid_frame_mask = point_masks.sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    sample_manifest_scale = float(sample_manifest_applied_scale)
    sample_manifest_trans_scale = float(sample_manifest_applied_trans_scale)
    sample_manifest_rot_scale = float(sample_manifest_applied_rot_scale)
    sample_manifest_focal_scale = float(sample_manifest_applied_focal_scale)
    sample_manifest_label_focal_scales = sample_manifest_label_focal_scales or {}
    sample_manifest_label_focal_scales = {
        str(label).strip(): float(scale)
        for label, scale in dict(sample_manifest_label_focal_scales).items()
        if str(label).strip()
    }
    sample_manifest_scale = resolve_two_stage_scalar(
        sample_manifest_scale,
        batch_data,
        start=sample_manifest_schedule_start,
        end=sample_manifest_schedule_end,
        stage2_value=sample_manifest_applied_scale_stage2,
    )
    sample_manifest_trans_scale = resolve_two_stage_scalar(
        sample_manifest_trans_scale,
        batch_data,
        start=sample_manifest_schedule_start,
        end=sample_manifest_schedule_end,
        stage2_value=sample_manifest_applied_trans_scale_stage2,
    )
    sample_manifest_rot_scale = resolve_two_stage_scalar(
        sample_manifest_rot_scale,
        batch_data,
        start=sample_manifest_schedule_start,
        end=sample_manifest_schedule_end,
        stage2_value=sample_manifest_applied_rot_scale_stage2,
    )
    sample_manifest_focal_scale = resolve_two_stage_scalar(
        sample_manifest_focal_scale,
        batch_data,
        start=sample_manifest_schedule_start,
        end=sample_manifest_schedule_end,
        stage2_value=sample_manifest_applied_focal_scale_stage2,
    )
    sample_manifest_label_focal_scales = resolve_two_stage_label_scalars(
        sample_manifest_label_focal_scales,
        batch_data,
        stage2_values=sample_manifest_label_focal_scales_stage2,
        start=sample_manifest_schedule_start,
        end=sample_manifest_schedule_end,
    )
    sample_manifest_flags = _normalize_batch_manifest_applied(
        batch_data.get("selection_sample_manifest_applied"),
        valid_frame_mask.shape[0],
    )
    sample_manifest_labels = _normalize_batch_manifest_labels(
        batch_data.get("selection_sample_manifest_label"),
        valid_frame_mask.shape[0],
    )
    apply_manifest_scale = any(
        scale != 1.0
        for scale in (
            sample_manifest_scale,
            sample_manifest_trans_scale,
            sample_manifest_rot_scale,
            sample_manifest_focal_scale,
        )
    )
    if not apply_manifest_scale:
        apply_manifest_scale = any(
            float(scale) != 1.0 for scale in sample_manifest_label_focal_scales.values()
        )
    if apply_manifest_scale and bool(sample_manifest_applied_train_only):
        phase = str(batch_data.get("_loss_phase", "")).lower()
        apply_manifest_scale = phase == "train"

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            pred_pose_valid = pred_pose_stage[valid_frame_mask].clone()
            gt_pose_valid = gt_pose_encoding[valid_frame_mask].clone()
            sample_weights_T = None
            sample_weights_R = None
            sample_weights_FL = None
            if apply_manifest_scale:
                per_sample_scale_T = torch.ones(
                    (valid_frame_mask.shape[0],),
                    device=pred_pose_stage.device,
                    dtype=pred_pose_stage.dtype,
                )
                per_sample_scale_R = per_sample_scale_T.clone()
                per_sample_scale_FL = per_sample_scale_T.clone()
                for batch_idx, manifest_applied in enumerate(sample_manifest_flags):
                    manifest_label = sample_manifest_labels[batch_idx]
                    label_focal_scale = sample_manifest_label_focal_scales.get(manifest_label, 1.0)
                    per_sample_scale_FL[batch_idx] = float(label_focal_scale)
                    if manifest_applied:
                        per_sample_scale_T[batch_idx] = sample_manifest_scale * sample_manifest_trans_scale
                        per_sample_scale_R[batch_idx] = sample_manifest_scale * sample_manifest_rot_scale
                        per_sample_scale_FL[batch_idx] = (
                            sample_manifest_scale * sample_manifest_focal_scale * per_sample_scale_FL[batch_idx]
                        )
                sample_weights_T = per_sample_scale_T.unsqueeze(-1).expand_as(valid_frame_mask)[valid_frame_mask]
                sample_weights_R = per_sample_scale_R.unsqueeze(-1).expand_as(valid_frame_mask)[valid_frame_mask]
                sample_weights_FL = per_sample_scale_FL.unsqueeze(-1).expand_as(valid_frame_mask)[valid_frame_mask]
            if sample_weights_T is None:
                loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                    pred_pose_valid,
                    gt_pose_valid,
                    loss_type=loss_type
                )
            elif loss_type == "l1":
                loss_T_values = (pred_pose_valid[..., :3] - gt_pose_valid[..., :3]).abs().clamp(max=100)
                loss_R_values = (pred_pose_valid[..., 3:7] - gt_pose_valid[..., 3:7]).abs()
                loss_FL_values = (pred_pose_valid[..., 7:] - gt_pose_valid[..., 7:]).abs()
                loss_T_stage = _weighted_camera_component_mean(loss_T_values, sample_weights_T)
                loss_R_stage = _weighted_camera_component_mean(loss_R_values, sample_weights_R)
                loss_FL_stage = _weighted_camera_component_mean(loss_FL_values, sample_weights_FL)
            elif loss_type == "l2":
                loss_T_values = (pred_pose_valid[..., :3] - gt_pose_valid[..., :3]).norm(dim=-1, keepdim=True)
                loss_R_values = (pred_pose_valid[..., 3:7] - gt_pose_valid[..., 3:7]).norm(dim=-1)
                loss_FL_values = (pred_pose_valid[..., 7:] - gt_pose_valid[..., 7:]).norm(dim=-1)
                loss_T_stage = _weighted_camera_component_mean(loss_T_values, sample_weights_T)
                loss_R_stage = _weighted_camera_component_mean(loss_R_values, sample_weights_R)
                loss_FL_stage = _weighted_camera_component_mean(loss_FL_values, sample_weights_FL)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        stage_component_dict = _assemble_camera_component_dict(
            loss_T_stage,
            loss_R_stage,
            loss_FL_stage,
        )
        # Accumulate weighted losses across stages
        total_loss_T += stage_component_dict["loss_T"] * stage_weight
        total_loss_R += stage_component_dict["loss_R"] * stage_weight
        total_loss_FL += stage_component_dict["loss_FL"] * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    camera_component_dict = _assemble_camera_component_dict(
        avg_loss_T,
        avg_loss_R,
        avg_loss_FL,
    )

    # Compute total weighted camera loss
    total_camera_loss = _compute_total_camera_loss(
        camera_component_dict,
        weight_trans=weight_trans,
        weight_rot=weight_rot,
        weight_focal=weight_focal,
        loss_t_isolation_scale=loss_t_isolation_scale,
        loss_fl_isolation_scale=loss_fl_isolation_scale,
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": camera_component_dict["loss_T"],
        "loss_R": camera_component_dict["loss_R"],
        "loss_FL": camera_component_dict["loss_FL"]
    }


def unproject_depth_and_pose_to_world_points(
    pred_depth,
    pred_pose_enc,
    image_size_hw,
    pose_encoding_type="absT_quaR_FoV",
):
    """Differentiably reconstruct world points from the predicted depth + camera branch."""
    if pred_depth.shape[-1] != 1:
        raise ValueError(f"Expected pred_depth to have a singleton channel dim, got {pred_depth.shape}")

    depth = pred_depth[..., 0]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pred_pose_enc,
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=True,
    )

    bb, ss, hh, ww = depth.shape
    device = depth.device
    dtype = depth.dtype

    ys = torch.arange(hh, device=device, dtype=dtype)
    xs = torch.arange(ww, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.view(1, 1, hh, ww)
    grid_y = grid_y.view(1, 1, hh, ww)

    fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    cam_x = (grid_x - cx) * depth / fx
    cam_y = (grid_y - cy) * depth / fy
    cam_z = depth
    cam_points = torch.stack((cam_x, cam_y, cam_z), dim=-1)

    world_to_cam_R = extrinsics[..., :3, :3]
    world_to_cam_t = extrinsics[..., :3, 3]
    cam_to_world_R = world_to_cam_R.transpose(-1, -2)
    cam_to_world_t = -torch.matmul(cam_to_world_R, world_to_cam_t.unsqueeze(-1)).squeeze(-1)

    world_points = torch.einsum("bsij,bshwj->bshwi", cam_to_world_R, cam_points)
    world_points = world_points + cam_to_world_t.unsqueeze(-2).unsqueeze(-2)
    return world_points


def _images_to_rgb01(images):
    if images is None:
        return None
    if images.ndim != 5:
        raise ValueError(f"Expected batch['images'] to have 5 dims, got {images.shape}")
    if images.shape[2] in (1, 3):
        rgb = images.permute(0, 1, 3, 4, 2).to(torch.float32)
    elif images.shape[-1] in (1, 3):
        rgb = images.to(torch.float32)
    else:
        raise ValueError(f"Unable to infer image channel axis from {images.shape}")

    max_value = float(rgb.detach().amax().item()) if rgb.numel() else 1.0
    min_value = float(rgb.detach().amin().item()) if rgb.numel() else 0.0
    if max_value > 1.5:
        rgb = rgb / 255.0
    elif min_value < 0.0:
        rgb = (rgb + 1.0) / 2.0
    return rgb.clamp(0.0, 1.0)


def _build_bottom_band_mask_like(mask_hw: torch.Tensor, bottom_band_ratio: float) -> torch.Tensor:
    bottom_band_ratio = float(max(0.0, min(0.95, bottom_band_ratio)))
    if bottom_band_ratio <= 0.0:
        return torch.zeros_like(mask_hw, dtype=torch.bool)
    hh = int(mask_hw.shape[-2])
    cutoff = int(hh * (1.0 - bottom_band_ratio))
    bottom = torch.zeros_like(mask_hw, dtype=torch.bool)
    if cutoff < hh:
        bottom[..., cutoff:, :] = True
    return bottom


def _profile_peak_surrogate(coords01, weights, *, bin_count=32, sigma=0.045):
    if coords01.numel() == 0 or weights.numel() == 0:
        return weights.new_zeros(())
    centers = torch.linspace(0.0, 1.0, steps=int(bin_count), device=coords01.device, dtype=coords01.dtype)
    sigma = max(float(sigma), 1e-4)
    basis = torch.exp(-((coords01.unsqueeze(-1) - centers.view(1, -1)) ** 2) / (2.0 * sigma * sigma))
    profile = (basis * weights.unsqueeze(-1)).sum(dim=0)
    profile = profile / profile.sum().clamp_min(1e-8)
    return 1.0 - (profile * profile).sum()


def _compute_anchor_projection_alignment_losses(
    pred_world_points,
    predictions,
    batch,
    *,
    image_size_hw,
    pose_encoding_type,
    eps,
    support_pull_to_fg_weight,
    support_pull_bottom_band_extra_scale,
    bg_black_weight,
    bg_black_bottom_band_extra_scale,
    fg_support_spread_weight,
    fg_profile_peak_surrogate_weight,
    profile_peak_bin_count,
    profile_peak_sigma,
    alignment_bottom_band_ratio,
    exclude_anchor_source_view=True,
):
    if (
        support_pull_to_fg_weight <= 0.0
        and bg_black_weight <= 0.0
        and fg_support_spread_weight <= 0.0
        and fg_profile_peak_surrogate_weight <= 0.0
    ):
        zero = (0.0 * pred_world_points).mean()
        return zero, zero, zero, zero

    foreground_masks = batch.get("foreground_masks", None)
    images = batch.get("images", None)
    pred_depth_conf = predictions.get("depth_conf", None)
    if foreground_masks is None or images is None:
        zero = (0.0 * pred_world_points).mean()
        return zero, zero, zero, zero

    anchor_view_indices = _normalize_batch_anchor_view_indices(
        batch.get("selection_anchor_view_index"),
        pred_world_points.shape[0],
    )
    if not anchor_view_indices:
        zero = (0.0 * pred_world_points).mean()
        return zero, zero, zero, zero

    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=True,
    )
    images01 = _images_to_rgb01(images)
    bb, ss, hh, ww, _ = pred_world_points.shape

    support_total = pred_world_points.new_zeros(())
    support_denom = pred_world_points.new_zeros(())
    black_total = pred_world_points.new_zeros(())
    black_denom = pred_world_points.new_zeros(())
    spread_total = pred_world_points.new_zeros(())
    spread_count = pred_world_points.new_zeros(())
    peak_total = pred_world_points.new_zeros(())
    peak_count = pred_world_points.new_zeros(())

    for batch_idx, anchor_idx in enumerate(anchor_view_indices):
        if anchor_idx is None or not (0 <= int(anchor_idx) < ss):
            continue
        anchor_idx = int(anchor_idx)
        anchor_fg = foreground_masks[batch_idx, anchor_idx].to(device=pred_world_points.device, dtype=torch.float32)
        anchor_fg_input = anchor_fg.unsqueeze(0).unsqueeze(0)
        bottom_band = _build_bottom_band_mask_like(anchor_fg > 0.5, alignment_bottom_band_ratio).to(torch.float32)
        bottom_band_input = bottom_band.unsqueeze(0).unsqueeze(0)

        anchor_extrinsic = pred_extrinsics[batch_idx, anchor_idx]
        anchor_intrinsic = pred_intrinsics[batch_idx, anchor_idx]
        anchor_R = anchor_extrinsic[:3, :3]
        anchor_t = anchor_extrinsic[:3, 3]
        fx = anchor_intrinsic[0, 0]
        fy = anchor_intrinsic[1, 1]
        cx = anchor_intrinsic[0, 2]
        cy = anchor_intrinsic[1, 2]
        sample_inside_x = []
        sample_inside_y = []
        sample_inside_w = []

        for view_idx in range(ss):
            if exclude_anchor_source_view and view_idx == anchor_idx:
                continue

            world_points = pred_world_points[batch_idx, view_idx]
            if not torch.isfinite(world_points).any():
                continue

            cam_points = torch.einsum("ij,hwj->hwi", anchor_R, world_points)
            cam_points = cam_points + anchor_t.view(1, 1, 3)
            z = cam_points[..., 2]
            valid = torch.isfinite(cam_points).all(dim=-1) & (z > eps)
            if not bool(valid.any().item()):
                continue

            u = (fx * cam_points[..., 0] / z.clamp_min(eps)) + cx
            v = (fy * cam_points[..., 1] / z.clamp_min(eps)) + cy
            x_norm = (u / max(float(ww - 1), 1.0)) * 2.0 - 1.0
            y_norm = (v / max(float(hh - 1), 1.0)) * 2.0 - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

            fg_sample = F.grid_sample(
                anchor_fg_input,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0).squeeze(0)
            bottom_sample = F.grid_sample(
                bottom_band_input,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0).squeeze(0)

            outside_fg = (1.0 - fg_sample).clamp(0.0, 1.0)
            if pred_depth_conf is not None:
                source_support = check_and_fix_inf_nan(pred_depth_conf[batch_idx, view_idx], "anchor_projection_depth_conf").to(dtype=pred_world_points.dtype).clamp_min(0.0)
                support_weight = torch.where(valid, source_support, torch.zeros_like(source_support))
            else:
                support_weight = valid.to(dtype=pred_world_points.dtype)
            if not bool(support_weight.any().item()):
                continue

            inside_fg_weight = support_weight * fg_sample
            if bool((inside_fg_weight > 0).any().item()):
                x01 = ((x_norm + 1.0) * 0.5).clamp(0.0, 1.0)
                y01 = ((y_norm + 1.0) * 0.5).clamp(0.0, 1.0)
                sample_inside_x.append(x01.reshape(-1))
                sample_inside_y.append(y01.reshape(-1))
                sample_inside_w.append(inside_fg_weight.reshape(-1))

            if support_pull_to_fg_weight > 0.0:
                outside_penalty = outside_fg * (
                    1.0 + bottom_sample * max(float(support_pull_bottom_band_extra_scale) - 1.0, 0.0)
                )
                support_total = support_total + (outside_penalty * support_weight).sum()
                support_denom = support_denom + support_weight.sum()

            if bg_black_weight > 0.0:
                source_rgb = images01[batch_idx, view_idx]
                source_intensity = source_rgb.mean(dim=-1)
                black_penalty = source_intensity * outside_fg * (
                    1.0 + bottom_sample * max(float(bg_black_bottom_band_extra_scale) - 1.0, 0.0)
                )
                black_total = black_total + (black_penalty * support_weight).sum()
                black_denom = black_denom + support_weight.sum()

        if sample_inside_w:
            inside_x = torch.cat(sample_inside_x, dim=0)
            inside_y = torch.cat(sample_inside_y, dim=0)
            inside_w = torch.cat(sample_inside_w, dim=0).clamp_min(0.0)
            valid_inside = inside_w > 0.0
            inside_x = inside_x[valid_inside]
            inside_y = inside_y[valid_inside]
            inside_w = inside_w[valid_inside]
            if inside_w.numel() > 0 and float(inside_w.sum().detach().item()) > 0.0:
                prob = inside_w / inside_w.sum().clamp_min(1e-8)
                center_x = (prob * inside_x).sum()
                center_y = (prob * inside_y).sum()
                spread_loss = (prob * ((inside_x - center_x) ** 2 + (inside_y - center_y) ** 2)).sum()
                peak_loss = (
                    _profile_peak_surrogate(inside_x, prob, bin_count=int(profile_peak_bin_count), sigma=float(profile_peak_sigma))
                    + _profile_peak_surrogate(inside_y, prob, bin_count=int(profile_peak_bin_count), sigma=float(profile_peak_sigma))
                )
                spread_total = spread_total + spread_loss
                spread_count = spread_count + 1.0
                peak_total = peak_total + peak_loss
                peak_count = peak_count + 1.0

    zero = (0.0 * pred_world_points).mean()
    support_loss = support_total / support_denom.clamp_min(eps) if float(support_denom.detach().item()) > 0.0 else zero
    black_loss = black_total / black_denom.clamp_min(eps) if float(black_denom.detach().item()) > 0.0 else zero
    spread_loss = spread_total / spread_count.clamp_min(1.0) if float(spread_count.detach().item()) > 0.0 else zero
    peak_loss = peak_total / peak_count.clamp_min(1.0) if float(peak_count.detach().item()) > 0.0 else zero
    return support_loss, black_loss, spread_loss, peak_loss


def compute_unproject_geometry_loss(
    predictions,
    batch,
    loss_type="l2",
    valid_range=0.98,
    min_valid_points=100,
    eps=1e-8,
    pose_encoding_type="absT_quaR_FoV",
    use_depth_conf_gate=False,
    detach_depth_conf=True,
    depth_conf_power=1.0,
    depth_conf_threshold=0.0,
    use_foreground_region_mask=False,
    foreground_erode_px=0,
    foreground_drop_bottom_ratio=0.0,
    human_prior_mask_key=None,
    human_prior_feature_map_key=None,
    human_prior_mask_floor=0.35,
    human_prior_scale=1.0,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
    human_prior_conf_floor_weight=0.0,
    human_prior_conf_floor=0.35,
    human_prior_depth_presence_weight=0.0,
    human_prior_pseudo_mask_key=None,
    human_prior_pseudo_world_key=None,
    human_prior_pseudo_weight=0.0,
    support_pull_to_fg_weight=0.0,
    support_pull_bottom_band_extra_scale=2.0,
    bg_black_weight=0.0,
    bg_black_bottom_band_extra_scale=3.0,
    fg_support_spread_weight=0.0,
    fg_profile_peak_surrogate_weight=0.0,
    profile_peak_bin_count=32,
    profile_peak_sigma=0.045,
    alignment_bottom_band_ratio=0.2,
    exclude_anchor_source_view=True,
    **kwargs,
):
    """
    Minimal geometry-chain loss:
    predicted depth + predicted camera -> differentiable unprojection -> world-point regression.
    """
    del kwargs

    image_hw = batch["images"].shape[-2:]
    pred_world_points = unproject_depth_and_pose_to_world_points(
        predictions["depth"],
        predictions["pose_enc"],
        image_size_hw=image_hw,
        pose_encoding_type=pose_encoding_type,
    )
    gt_world_points = check_and_fix_inf_nan(batch["world_points"], "gt_world_points")
    pred_depth_mask = predictions["depth"][..., 0] > eps
    gt_mask = batch["point_masks"].clone()
    valid_mask = gt_mask & pred_depth_mask & torch.isfinite(pred_world_points).all(dim=-1)
    conf_weights = None
    zero = (0.0 * predictions["depth"]).mean()
    region_mask = None
    human_prior_pseudo_mask, _, human_prior_pseudo_world_points = _build_human_prior_pseudo_supervision_components(
        batch,
        gt_mask,
        human_prior_pseudo_mask_key=human_prior_pseudo_mask_key,
        human_prior_pseudo_world_key=human_prior_pseudo_world_key,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    if human_prior_pseudo_mask is not None:
        human_prior_pseudo_mask = human_prior_pseudo_mask & ~gt_mask

    if use_foreground_region_mask:
        foreground_masks = batch.get("foreground_masks", None)
        if foreground_masks is None:
            raise ValueError("use_foreground_region_mask=True requires batch['foreground_masks'].")
        region_mask = foreground_masks.to(device=valid_mask.device, dtype=torch.bool)
        region_mask = build_reliable_foreground_region_mask(
            region_mask,
            erode_px=int(foreground_erode_px),
            drop_bottom_ratio=float(foreground_drop_bottom_ratio),
        )
        valid_mask = valid_mask & region_mask
        if human_prior_pseudo_mask is not None:
            human_prior_pseudo_mask = human_prior_pseudo_mask & region_mask

    if use_depth_conf_gate:
        pred_depth_conf = predictions.get("depth_conf", None)
        if pred_depth_conf is None:
            raise ValueError("use_depth_conf_gate=True requires predictions['depth_conf'].")
        if detach_depth_conf:
            pred_depth_conf = pred_depth_conf.detach()
        pred_depth_conf = check_and_fix_inf_nan(pred_depth_conf, "pred_depth_conf_gate")
        valid_mask = valid_mask & torch.isfinite(pred_depth_conf)
        if depth_conf_threshold > 0:
            valid_mask = valid_mask & (pred_depth_conf >= depth_conf_threshold)

    has_gt_supervision = int(valid_mask.sum().item()) >= int(min_valid_points)
    has_pseudo_supervision = human_prior_pseudo_mask is not None and bool(human_prior_pseudo_mask.any().item())
    if not has_gt_supervision and not has_pseudo_supervision:
        return {
            "loss_unproject_geometry": zero,
            "loss_unproject_geometry_base": zero,
            "loss_support_pull_to_fg": zero,
            "loss_bg_black": zero,
            "loss_fg_support_spread": zero,
            "loss_fg_profile_peak_surrogate": zero,
            "loss_human_prior_conf_floor": zero,
            "loss_human_prior_depth_presence": zero,
            "loss_human_prior_pseudo_unproject": zero,
        }

    human_prior_mask, human_prior_feature_map = _build_human_prior_region_components(
        batch,
        valid_mask,
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    human_prior_scale_map = _build_human_prior_scale_map(
        batch,
        predictions["depth"][..., 0],
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_mask_floor=human_prior_mask_floor,
        human_prior_scale=human_prior_scale,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    human_prior_target_mask = _build_human_prior_target_mask(
        human_prior_mask,
        human_prior_feature_map,
    )

    loss = zero
    if has_gt_supervision:
        diff = pred_world_points[valid_mask] - gt_world_points[valid_mask]
        if use_depth_conf_gate:
            conf_weights = pred_depth_conf[valid_mask].clamp_min(eps).pow(depth_conf_power)
        if loss_type == "l1":
            loss_values = diff.abs().mean(dim=-1)
        elif loss_type == "l2":
            loss_values = diff.norm(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type for unproject geometry loss: {loss_type}")

        loss_values = check_and_fix_inf_nan(loss_values, "loss_unproject_geometry")
        human_prior_loss_scale = None
        if human_prior_scale_map is not None:
            human_prior_loss_scale = check_and_fix_inf_nan(
                human_prior_scale_map[valid_mask],
                "loss_unproject_geometry_human_prior_scale",
            ).clamp_min(1e-6)
        if valid_range > 0:
            quantile_mask = build_quantile_mask(loss_values, valid_range)
            if quantile_mask is not None:
                loss_values = loss_values[quantile_mask]
                if conf_weights is not None:
                    conf_weights = conf_weights[quantile_mask]
                if human_prior_loss_scale is not None:
                    human_prior_loss_scale = human_prior_loss_scale[quantile_mask]

        if human_prior_loss_scale is not None:
            loss_values = loss_values * human_prior_loss_scale

        if loss_values.numel() == 0:
            loss = zero
        elif conf_weights is not None:
            conf_weights = check_and_fix_inf_nan(conf_weights, "loss_unproject_geometry_conf_weights")
            loss = (loss_values * conf_weights).sum() / conf_weights.sum().clamp_min(eps)
        else:
            loss = loss_values.mean()

    human_prior_pseudo_unproject_loss = zero
    if (
        float(human_prior_pseudo_weight) > 0.0
        and human_prior_pseudo_mask is not None
        and human_prior_pseudo_world_points is not None
    ):
        pseudo_valid_mask = human_prior_pseudo_mask & pred_depth_mask & torch.isfinite(pred_world_points).all(dim=-1)
        if use_depth_conf_gate:
            pseudo_valid_mask = pseudo_valid_mask & torch.isfinite(pred_depth_conf)
        if bool(pseudo_valid_mask.any().item()):
            pseudo_diff = pred_world_points[pseudo_valid_mask] - human_prior_pseudo_world_points[pseudo_valid_mask]
            if loss_type == "l1":
                pseudo_loss_values = pseudo_diff.abs().mean(dim=-1)
            elif loss_type == "l2":
                pseudo_loss_values = pseudo_diff.norm(dim=-1)
            else:
                raise ValueError(f"Unknown loss_type for unproject geometry loss: {loss_type}")

            pseudo_loss_values = check_and_fix_inf_nan(
                pseudo_loss_values,
                "loss_human_prior_pseudo_unproject",
            )
            if human_prior_scale_map is not None:
                pseudo_scale = check_and_fix_inf_nan(
                    human_prior_scale_map[pseudo_valid_mask],
                    "loss_human_prior_pseudo_unproject_scale",
                ).clamp_min(1e-6)
                pseudo_loss_values = pseudo_loss_values * pseudo_scale
            if valid_range > 0:
                pseudo_quantile_mask = build_quantile_mask(pseudo_loss_values, valid_range)
                if pseudo_quantile_mask is not None:
                    pseudo_loss_values = pseudo_loss_values[pseudo_quantile_mask]
            if pseudo_loss_values.numel() > 0:
                human_prior_pseudo_unproject_loss = pseudo_loss_values.mean()

    human_prior_supervision_mask = gt_mask
    if human_prior_pseudo_mask is not None:
        human_prior_supervision_mask = human_prior_supervision_mask | human_prior_pseudo_mask

    human_prior_conf_floor_loss = zero
    if float(human_prior_conf_floor_weight) > 0.0:
        pred_depth_conf = predictions.get("depth_conf", None)
        if pred_depth_conf is None:
            raise ValueError(
                "human_prior_conf_floor_weight > 0 requires predictions['depth_conf']."
            )
        if human_prior_target_mask is not None:
            conf_target_mask = human_prior_target_mask & human_prior_supervision_mask & torch.isfinite(pred_depth_conf)
            if bool(conf_target_mask.any().item()):
                pred_depth_conf = check_and_fix_inf_nan(
                    pred_depth_conf,
                    "human_prior_conf_floor_depth_conf",
                ).clamp(0.0, 1.0)
                human_prior_conf_floor_loss = F.relu(
                    float(human_prior_conf_floor) - pred_depth_conf[conf_target_mask]
                ).mean()

    human_prior_depth_presence_loss = zero
    if float(human_prior_depth_presence_weight) > 0.0 and human_prior_target_mask is not None:
        presence_target_mask = human_prior_target_mask & human_prior_supervision_mask
        if bool(presence_target_mask.any().item()):
            human_prior_depth_presence_loss = (
                ~pred_depth_mask[presence_target_mask]
            ).to(dtype=predictions["depth"].dtype).mean()

    support_pull_loss, bg_black_loss, fg_spread_loss, fg_peak_loss = _compute_anchor_projection_alignment_losses(
        pred_world_points,
        predictions,
        batch,
        image_size_hw=image_hw,
        pose_encoding_type=pose_encoding_type,
        eps=eps,
        support_pull_to_fg_weight=float(support_pull_to_fg_weight),
        support_pull_bottom_band_extra_scale=float(support_pull_bottom_band_extra_scale),
        bg_black_weight=float(bg_black_weight),
        bg_black_bottom_band_extra_scale=float(bg_black_bottom_band_extra_scale),
        fg_support_spread_weight=float(fg_support_spread_weight),
        fg_profile_peak_surrogate_weight=float(fg_profile_peak_surrogate_weight),
        profile_peak_bin_count=int(profile_peak_bin_count),
        profile_peak_sigma=float(profile_peak_sigma),
        alignment_bottom_band_ratio=float(alignment_bottom_band_ratio),
        exclude_anchor_source_view=bool(exclude_anchor_source_view),
    )
    total_loss = loss
    if float(support_pull_to_fg_weight) > 0.0:
        total_loss = total_loss + float(support_pull_to_fg_weight) * support_pull_loss
    if float(bg_black_weight) > 0.0:
        total_loss = total_loss + float(bg_black_weight) * bg_black_loss
    if float(fg_support_spread_weight) > 0.0:
        total_loss = total_loss + float(fg_support_spread_weight) * fg_spread_loss
    if float(fg_profile_peak_surrogate_weight) > 0.0:
        total_loss = total_loss + float(fg_profile_peak_surrogate_weight) * fg_peak_loss
    if float(human_prior_conf_floor_weight) > 0.0:
        total_loss = total_loss + float(human_prior_conf_floor_weight) * human_prior_conf_floor_loss
    if float(human_prior_depth_presence_weight) > 0.0:
        total_loss = total_loss + float(human_prior_depth_presence_weight) * human_prior_depth_presence_loss
    if float(human_prior_pseudo_weight) > 0.0:
        total_loss = total_loss + float(human_prior_pseudo_weight) * human_prior_pseudo_unproject_loss

    return {
        "loss_unproject_geometry": total_loss,
        "loss_unproject_geometry_base": loss,
        "loss_support_pull_to_fg": support_pull_loss,
        "loss_bg_black": bg_black_loss,
        "loss_fg_support_spread": fg_spread_loss,
        "loss_fg_profile_peak_surrogate": fg_peak_loss,
        "loss_human_prior_conf_floor": human_prior_conf_floor_loss,
        "loss_human_prior_depth_presence": human_prior_depth_presence_loss,
        "loss_human_prior_pseudo_unproject": human_prior_pseudo_unproject_loss,
    }


def build_reliable_foreground_region_mask(mask, erode_px=0, drop_bottom_ratio=0.0, edge_band_px=0):
    """
    Build a stricter target-side region mask from the raw foreground mask.

    This supports a minimal "reliable region" treatment without changing the
    baseline camera/depth losses:
    - optional binary erosion to suppress uncertain silhouette edges
    - optional selection of a foreground boundary ring
    - optional removal of a bottom image band to avoid floor/ground contact zones
    """
    region_mask = mask.to(dtype=torch.bool)
    bb, ss, hh, ww = region_mask.shape

    erode_px = int(max(0, erode_px))
    edge_band_px = int(max(0, edge_band_px))
    if erode_px > 0 and edge_band_px > 0:
        raise ValueError("erode_px and edge_band_px are mutually exclusive region selectors.")

    if edge_band_px > 0:
        kernel = 2 * edge_band_px + 1
        flat = region_mask.reshape(bb * ss, 1, hh, ww)
        inv = (~flat).float()
        eroded = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=edge_band_px) == 0
        edge_mask = flat & ~eroded
        region_mask = edge_mask.reshape(bb, ss, hh, ww)
    elif erode_px > 0:
        kernel = 2 * erode_px + 1
        flat = region_mask.reshape(bb * ss, 1, hh, ww)
        inv = (~flat).float()
        # Binary erosion via dilation of the inverse mask.
        eroded = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=erode_px) == 0
        region_mask = eroded.reshape(bb, ss, hh, ww)

    drop_bottom_ratio = float(max(0.0, min(0.95, drop_bottom_ratio)))
    if drop_bottom_ratio > 0.0:
        cutoff = int(hh * (1.0 - drop_bottom_ratio))
        if cutoff < hh:
            region_mask[..., cutoff:, :] = False

    return region_mask

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def _weighted_camera_component_mean(values, sample_weights=None):
    values = check_and_fix_inf_nan(values, "camera_component_values")
    if sample_weights is None:
        return values.mean()
    sample_weights = check_and_fix_inf_nan(sample_weights, "camera_component_sample_weights")
    sample_weights = sample_weights.to(device=values.device, dtype=values.dtype)
    if values.ndim > 1:
        sample_weights = sample_weights.view(-1, *([1] * (values.ndim - 1)))
    weighted_values = values * sample_weights
    weight_sum = sample_weights.sum()
    if values.ndim > 1:
        feature_count = 1
        for dim in values.shape[1:]:
            feature_count *= int(dim)
        weight_sum = weight_sum * float(feature_count)
    if float(weight_sum.detach().item()) <= 0.0:
        return weighted_values.sum() * 0.0
    return weighted_values.sum() / weight_sum


def _normalize_batch_manifest_applied(selection_sample_manifest_applied, batch_size):
    if batch_size <= 0:
        return [False] * batch_size
    if selection_sample_manifest_applied is None:
        return [False] * batch_size
    if isinstance(selection_sample_manifest_applied, torch.Tensor):
        if selection_sample_manifest_applied.ndim == 0:
            values = [selection_sample_manifest_applied.item()]
        else:
            values = selection_sample_manifest_applied.detach().cpu().reshape(-1).tolist()
    elif isinstance(selection_sample_manifest_applied, (list, tuple)):
        values = list(selection_sample_manifest_applied)
    else:
        values = [selection_sample_manifest_applied]
    values = [bool(value) for value in values]
    if len(values) == batch_size:
        return values
    if len(values) == 1:
        return values * batch_size
    if len(values) < batch_size:
        return values + ([False] * (batch_size - len(values)))
    return values[:batch_size]


def _normalize_batch_manifest_labels(selection_sample_manifest_label, batch_size):
    if batch_size <= 0:
        return [""] * batch_size
    if selection_sample_manifest_label is None:
        return [""] * batch_size
    if isinstance(selection_sample_manifest_label, torch.Tensor):
        values = [str(value) for value in selection_sample_manifest_label.detach().cpu().reshape(-1).tolist()]
    elif isinstance(selection_sample_manifest_label, (list, tuple)):
        values = [str(value or "") for value in selection_sample_manifest_label]
    else:
        values = [str(selection_sample_manifest_label or "")]
    values = [str(value).strip() for value in values]
    if len(values) == batch_size:
        return values
    if len(values) == 1:
        return values * batch_size
    if len(values) < batch_size:
        return values + ([""] * (batch_size - len(values)))
    return values[:batch_size]


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(
    predictions,
    batch,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    conf_loss_aggregation="pixel_mean",
    respect_conf_mask_in_grad_conf=False,
    anchor_conditioned_conf_target_cameras=None,
    anchor_conditioned_conf_target_scale=1.0,
    anchor_conditioned_conf_target_train_only=True,
    anchor_conditioned_conf_target_foreground_erode_px=0,
    anchor_conditioned_conf_target_foreground_edge_band_px=0,
    anchor_conditioned_conf_target_foreground_bottom_ratio=0.0,
    anchor_conditioned_conf_target_quality_min=None,
    anchor_conditioned_conf_target_quality_max=None,
    anchor_conditioned_conf_target_quality_interp="none",
    anchor_conditioned_conf_target_quality_low=None,
    anchor_conditioned_conf_target_quality_high=None,
    anchor_conditioned_conf_mask_cameras=None,
    anchor_conditioned_conf_mask_train_only=True,
    anchor_conditioned_conf_mask_foreground_erode_px=0,
    anchor_conditioned_conf_mask_foreground_edge_band_px=0,
    anchor_conditioned_conf_mask_foreground_bottom_ratio=0.0,
    anchor_conditioned_conf_mask_quality_min=None,
    anchor_conditioned_conf_mask_quality_max=None,
    anchor_conditioned_reg_target_cameras=None,
    anchor_conditioned_reg_target_scale=1.0,
    anchor_conditioned_reg_target_train_only=True,
    anchor_conditioned_reg_target_foreground_erode_px=0,
    anchor_conditioned_reg_target_foreground_edge_band_px=0,
    anchor_conditioned_reg_target_foreground_bottom_ratio=0.0,
    anchor_conditioned_reg_target_quality_min=None,
    anchor_conditioned_reg_target_quality_max=None,
    anchor_conditioned_reg_target_quality_interp="none",
    anchor_conditioned_reg_target_quality_low=None,
    anchor_conditioned_reg_target_quality_high=None,
    anchor_conditioned_reg_target_anchor_view_only=False,
    anchor_conditioned_reg_target_depth_conf_min=None,
    anchor_conditioned_reg_target_depth_conf_max=None,
    anchor_conditioned_reg_target_depth_conf_interp="none",
    anchor_conditioned_reg_target_depth_conf_low=None,
    anchor_conditioned_reg_target_depth_conf_high=None,
    anchor_conditioned_conf_target_depth_conf_min=None,
    anchor_conditioned_conf_target_depth_conf_max=None,
    anchor_conditioned_conf_target_depth_conf_interp="none",
    anchor_conditioned_conf_target_depth_conf_low=None,
    anchor_conditioned_conf_target_depth_conf_high=None,
    anchor_conditioned_disagreement_cameras=None,
    anchor_conditioned_disagreement_conf_scale=1.0,
    anchor_conditioned_disagreement_reg_scale=1.0,
    anchor_conditioned_disagreement_train_only=True,
    anchor_conditioned_unproject_consistency_cameras=None,
    anchor_conditioned_unproject_consistency_conf_scale=1.0,
    anchor_conditioned_unproject_consistency_reg_scale=1.0,
    anchor_conditioned_unproject_consistency_train_only=True,
    human_prior_mask_key=None,
    human_prior_feature_map_key=None,
    human_prior_mask_floor=0.35,
    human_prior_reg_scale=1.0,
    human_prior_conf_scale=1.0,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
    human_prior_pseudo_depth_key=None,
    human_prior_pseudo_mask_key=None,
    human_prior_pseudo_weight=0.0,
    **kwargs,
):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
        conf_loss_aggregation: How to aggregate confidence loss across valid pixels/views
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask
    conf_depth_mask = batch.get('conf_depth_point_masks', gt_depth_mask).clone()
    conf_depth_mask = conf_depth_mask & gt_depth_mask
    anchor_conditioned_conf_mask_drop_map = _build_anchor_conditioned_conf_mask_drop_map(
        batch,
        conf_depth_mask,
        anchor_conditioned_conf_mask_cameras=anchor_conditioned_conf_mask_cameras,
        anchor_conditioned_conf_mask_train_only=anchor_conditioned_conf_mask_train_only,
        anchor_conditioned_conf_mask_foreground_erode_px=anchor_conditioned_conf_mask_foreground_erode_px,
        anchor_conditioned_conf_mask_foreground_edge_band_px=anchor_conditioned_conf_mask_foreground_edge_band_px,
        anchor_conditioned_conf_mask_foreground_bottom_ratio=anchor_conditioned_conf_mask_foreground_bottom_ratio,
        anchor_conditioned_conf_mask_quality_min=anchor_conditioned_conf_mask_quality_min,
        anchor_conditioned_conf_mask_quality_max=anchor_conditioned_conf_mask_quality_max,
    )
    if anchor_conditioned_conf_mask_drop_map is not None:
        # Route the targeted pixels out of conf-depth supervision while leaving
        # the underlying gt_depth mask unchanged for the regression branch.
        conf_depth_mask = conf_depth_mask & ~anchor_conditioned_conf_mask_drop_map

    human_prior_pseudo_mask, human_prior_pseudo_depth, _ = _build_human_prior_pseudo_supervision_components(
        batch,
        gt_depth_mask,
        human_prior_pseudo_mask_key=human_prior_pseudo_mask_key,
        human_prior_pseudo_depth_key=human_prior_pseudo_depth_key,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    if human_prior_pseudo_mask is not None:
        human_prior_pseudo_mask = human_prior_pseudo_mask & ~gt_depth_mask

    if gt_depth_mask.sum() < 100 and (
        human_prior_pseudo_mask is None or not bool(human_prior_pseudo_mask.any().item())
    ):
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {
            f"loss_conf_depth": dummy_loss,
            f"loss_reg_depth": dummy_loss,
            f"loss_grad_depth": dummy_loss,
            f"loss_human_prior_pseudo_depth": dummy_loss,
        }
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    regression_loss_extra_kwargs = _collect_regression_loss_extra_kwargs(kwargs)
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth,
        gt_depth,
        gt_depth_mask,
        conf=pred_depth_conf,
        conf_mask=conf_depth_mask,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
        conf_loss_aggregation=conf_loss_aggregation,
        respect_conf_mask_in_grad_conf=respect_conf_mask_in_grad_conf,
        batch=batch,
        anchor_conditioned_reg_target_cameras=anchor_conditioned_reg_target_cameras,
        anchor_conditioned_reg_target_scale=anchor_conditioned_reg_target_scale,
        anchor_conditioned_reg_target_train_only=anchor_conditioned_reg_target_train_only,
        anchor_conditioned_reg_target_foreground_erode_px=anchor_conditioned_reg_target_foreground_erode_px,
        anchor_conditioned_reg_target_foreground_edge_band_px=anchor_conditioned_reg_target_foreground_edge_band_px,
        anchor_conditioned_reg_target_foreground_bottom_ratio=anchor_conditioned_reg_target_foreground_bottom_ratio,
        anchor_conditioned_reg_target_quality_min=anchor_conditioned_reg_target_quality_min,
        anchor_conditioned_reg_target_quality_max=anchor_conditioned_reg_target_quality_max,
        anchor_conditioned_reg_target_quality_interp=anchor_conditioned_reg_target_quality_interp,
        anchor_conditioned_reg_target_quality_low=anchor_conditioned_reg_target_quality_low,
        anchor_conditioned_reg_target_quality_high=anchor_conditioned_reg_target_quality_high,
        anchor_conditioned_reg_target_anchor_view_only=anchor_conditioned_reg_target_anchor_view_only,
        anchor_conditioned_reg_target_depth_conf_min=anchor_conditioned_reg_target_depth_conf_min,
        anchor_conditioned_reg_target_depth_conf_max=anchor_conditioned_reg_target_depth_conf_max,
        anchor_conditioned_reg_target_depth_conf_interp=anchor_conditioned_reg_target_depth_conf_interp,
        anchor_conditioned_reg_target_depth_conf_low=anchor_conditioned_reg_target_depth_conf_low,
        anchor_conditioned_reg_target_depth_conf_high=anchor_conditioned_reg_target_depth_conf_high,
        anchor_conditioned_conf_target_cameras=anchor_conditioned_conf_target_cameras,
        anchor_conditioned_conf_target_scale=anchor_conditioned_conf_target_scale,
        anchor_conditioned_conf_target_train_only=anchor_conditioned_conf_target_train_only,
        anchor_conditioned_conf_target_foreground_erode_px=anchor_conditioned_conf_target_foreground_erode_px,
        anchor_conditioned_conf_target_foreground_edge_band_px=anchor_conditioned_conf_target_foreground_edge_band_px,
        anchor_conditioned_conf_target_foreground_bottom_ratio=anchor_conditioned_conf_target_foreground_bottom_ratio,
        anchor_conditioned_conf_target_quality_min=anchor_conditioned_conf_target_quality_min,
        anchor_conditioned_conf_target_quality_max=anchor_conditioned_conf_target_quality_max,
        anchor_conditioned_conf_target_quality_interp=anchor_conditioned_conf_target_quality_interp,
        anchor_conditioned_conf_target_quality_low=anchor_conditioned_conf_target_quality_low,
        anchor_conditioned_conf_target_quality_high=anchor_conditioned_conf_target_quality_high,
        anchor_conditioned_conf_target_anchor_view_only=False,
        anchor_conditioned_conf_target_depth_conf_min=anchor_conditioned_conf_target_depth_conf_min,
        anchor_conditioned_conf_target_depth_conf_max=anchor_conditioned_conf_target_depth_conf_max,
        anchor_conditioned_conf_target_depth_conf_interp=anchor_conditioned_conf_target_depth_conf_interp,
        anchor_conditioned_conf_target_depth_conf_low=anchor_conditioned_conf_target_depth_conf_low,
        anchor_conditioned_conf_target_depth_conf_high=anchor_conditioned_conf_target_depth_conf_high,
        anchor_conditioned_disagreement_cameras=anchor_conditioned_disagreement_cameras,
        anchor_conditioned_disagreement_conf_scale=anchor_conditioned_disagreement_conf_scale,
        anchor_conditioned_disagreement_reg_scale=anchor_conditioned_disagreement_reg_scale,
        anchor_conditioned_disagreement_train_only=anchor_conditioned_disagreement_train_only,
        anchor_conditioned_unproject_consistency_cameras=anchor_conditioned_unproject_consistency_cameras,
        anchor_conditioned_unproject_consistency_conf_scale=anchor_conditioned_unproject_consistency_conf_scale,
        anchor_conditioned_unproject_consistency_reg_scale=anchor_conditioned_unproject_consistency_reg_scale,
        anchor_conditioned_unproject_consistency_train_only=anchor_conditioned_unproject_consistency_train_only,
        anchor_conditioned_unproject_consistency_pred_pose_enc=predictions.get("pose_enc", None),
        anchor_conditioned_unproject_consistency_image_size_hw=batch["images"].shape[-2:] if "images" in batch else None,
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_mask_floor=human_prior_mask_floor,
        human_prior_reg_scale=human_prior_reg_scale,
        human_prior_conf_scale=human_prior_conf_scale,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
        **regression_loss_extra_kwargs,
    )

    human_prior_pseudo_depth_loss = (0.0 * pred_depth).mean()
    if (
        float(human_prior_pseudo_weight) > 0.0
        and human_prior_pseudo_mask is not None
        and human_prior_pseudo_depth is not None
        and bool(human_prior_pseudo_mask.any().item())
    ):
        pseudo_depth_loss_values = (
            pred_depth[..., 0][human_prior_pseudo_mask] - human_prior_pseudo_depth[human_prior_pseudo_mask]
        ).abs()
        if pseudo_depth_loss_values.numel() > 0:
            if valid_range > 0:
                pseudo_depth_loss_values = filter_by_quantile(pseudo_depth_loss_values, valid_range)
            pseudo_depth_loss_values = check_and_fix_inf_nan(
                pseudo_depth_loss_values,
                "loss_human_prior_pseudo_depth",
            )
            if pseudo_depth_loss_values.numel() > 0:
                human_prior_pseudo_depth_loss = pseudo_depth_loss_values.mean()

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg + float(human_prior_pseudo_weight) * human_prior_pseudo_depth_loss,
        f"loss_grad_depth": loss_grad,
        f"loss_human_prior_pseudo_depth": human_prior_pseudo_depth_loss,
    }

    return loss_dict


def _normalize_batch_anchor_cameras(selection_anchor_camera, batch_size):
    if batch_size <= 0:
        return []
    if selection_anchor_camera is None:
        return [None] * batch_size
    if isinstance(selection_anchor_camera, str):
        values = [selection_anchor_camera]
    elif isinstance(selection_anchor_camera, (list, tuple)):
        values = list(selection_anchor_camera)
    else:
        values = [selection_anchor_camera]
    values = [None if value is None else str(value) for value in values]
    if len(values) == batch_size:
        return values
    if len(values) == 1:
        return values * batch_size
    if len(values) < batch_size:
        return values + ([None] * (batch_size - len(values)))
    return values[:batch_size]


def _normalize_batch_anchor_quality_scores(selection_anchor_quality_score, batch_size):
    if batch_size <= 0:
        return []
    if selection_anchor_quality_score is None:
        return [None] * batch_size
    if isinstance(selection_anchor_quality_score, torch.Tensor):
        if selection_anchor_quality_score.ndim == 0:
            values = [selection_anchor_quality_score.item()]
        else:
            values = selection_anchor_quality_score.detach().cpu().reshape(-1).tolist()
    elif isinstance(selection_anchor_quality_score, (list, tuple)):
        values = list(selection_anchor_quality_score)
    else:
        values = [selection_anchor_quality_score]

    normalized = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        value = float(value)
        normalized.append(value if isfinite(value) else None)

    if len(normalized) == batch_size:
        return normalized
    if len(normalized) == 1:
        return normalized * batch_size
    if len(normalized) < batch_size:
        return normalized + ([None] * (batch_size - len(normalized)))
    return normalized[:batch_size]


def _normalize_batch_anchor_view_indices(selection_anchor_view_index, batch_size):
    if batch_size <= 0:
        return []
    if selection_anchor_view_index is None:
        return [None] * batch_size
    if isinstance(selection_anchor_view_index, torch.Tensor):
        if selection_anchor_view_index.ndim == 0:
            values = [selection_anchor_view_index.item()]
        else:
            values = selection_anchor_view_index.detach().cpu().reshape(-1).tolist()
    elif isinstance(selection_anchor_view_index, (list, tuple)):
        values = list(selection_anchor_view_index)
    else:
        values = [selection_anchor_view_index]

    normalized = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        try:
            idx = int(value)
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        normalized.append(idx if idx >= 0 else None)

    if len(normalized) == batch_size:
        return normalized
    if len(normalized) == 1:
        return normalized * batch_size
    if len(normalized) < batch_size:
        return normalized + ([None] * (batch_size - len(normalized)))
    return normalized[:batch_size]


def _build_anchor_view_only_mask(batch, reference_mask):
    if batch is None or reference_mask is None:
        return None
    anchor_view_indices = _normalize_batch_anchor_view_indices(
        batch.get("selection_anchor_view_index"),
        reference_mask.shape[0],
    )
    if not anchor_view_indices:
        return None
    anchor_view_mask = torch.zeros_like(reference_mask, dtype=torch.bool)
    view_count = int(reference_mask.shape[1])
    for batch_idx, view_idx in enumerate(anchor_view_indices):
        if view_idx is None:
            continue
        if 0 <= int(view_idx) < view_count:
            anchor_view_mask[batch_idx, int(view_idx)] = True
    if not bool(anchor_view_mask.any().item()):
        return None
    return anchor_view_mask


def _should_apply_human_prior(batch, human_prior_train_only):
    if not bool(human_prior_train_only):
        return True
    phase = str(batch.get("_loss_phase", "")).lower()
    return phase == "train"


def _build_human_prior_region_components(
    batch,
    reference_mask,
    *,
    human_prior_mask_key=None,
    human_prior_feature_map_key=None,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
):
    if batch is None or reference_mask is None:
        return None, None
    if not human_prior_mask_key and not human_prior_feature_map_key:
        return None, None

    if not _should_apply_human_prior(batch, human_prior_train_only):
        return None, None

    prior_mask = None
    if human_prior_mask_key:
        prior_mask = batch.get(human_prior_mask_key, None)
        if prior_mask is None:
            raise ValueError(
                f"human prior mask key '{human_prior_mask_key}' requires batch['{human_prior_mask_key}']."
            )
        prior_mask = prior_mask.to(device=reference_mask.device, dtype=torch.bool)
        if tuple(prior_mask.shape) != tuple(reference_mask.shape):
            raise ValueError(
                f"batch['{human_prior_mask_key}'] shape {tuple(prior_mask.shape)} does not match "
                f"reference shape {tuple(reference_mask.shape)}."
            )

    prior_feature_map = None
    if human_prior_feature_map_key:
        prior_feature_map = batch.get(human_prior_feature_map_key, None)
        if prior_feature_map is None:
            raise ValueError(
                "human prior feature map key "
                f"'{human_prior_feature_map_key}' requires batch['{human_prior_feature_map_key}']."
            )
        prior_feature_map = prior_feature_map.to(device=reference_mask.device, dtype=torch.float32)
        if tuple(prior_feature_map.shape) != tuple(reference_mask.shape):
            raise ValueError(
                f"batch['{human_prior_feature_map_key}'] shape {tuple(prior_feature_map.shape)} does not match "
                f"reference shape {tuple(reference_mask.shape)}."
            )
        prior_feature_map = check_and_fix_inf_nan(
            prior_feature_map,
            f"{human_prior_feature_map_key}_human_prior_feature_map",
        ).clamp_min(0.0)
        max_per_view = prior_feature_map.reshape(prior_feature_map.shape[0], prior_feature_map.shape[1], -1).amax(
            dim=-1,
            keepdim=True,
        ).unsqueeze(-1)
        prior_feature_map = torch.where(
            max_per_view > 0.0,
            prior_feature_map / max_per_view.clamp_min(1e-6),
            torch.zeros_like(prior_feature_map),
        )

    if human_prior_anchor_view_only:
        anchor_view_mask = _build_anchor_view_only_mask(batch, reference_mask)
        if anchor_view_mask is None:
            return None, None
        prior_mask = anchor_view_mask if prior_mask is None else (prior_mask & anchor_view_mask)
        if prior_feature_map is not None:
            prior_feature_map = prior_feature_map * anchor_view_mask.to(dtype=prior_feature_map.dtype)

    if prior_mask is not None and not bool(prior_mask.any().item()) and prior_feature_map is None:
        return None, None
    if prior_feature_map is not None and float(prior_feature_map.detach().amax().item()) <= 0.0 and prior_mask is None:
        return None, None
    return prior_mask, prior_feature_map


def _build_human_prior_pseudo_supervision_components(
    batch,
    reference_mask,
    *,
    human_prior_pseudo_mask_key=None,
    human_prior_pseudo_depth_key=None,
    human_prior_pseudo_world_key=None,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
):
    if batch is None or reference_mask is None:
        return None, None, None
    if not human_prior_pseudo_mask_key and not human_prior_pseudo_depth_key and not human_prior_pseudo_world_key:
        return None, None, None
    if not _should_apply_human_prior(batch, human_prior_train_only):
        return None, None, None

    pseudo_mask = None
    if human_prior_pseudo_mask_key:
        pseudo_mask = batch.get(human_prior_pseudo_mask_key, None)
        if pseudo_mask is None:
            raise ValueError(
                "human prior pseudo mask key "
                f"'{human_prior_pseudo_mask_key}' requires batch['{human_prior_pseudo_mask_key}']."
            )
        pseudo_mask = pseudo_mask.to(device=reference_mask.device, dtype=torch.bool)
        if tuple(pseudo_mask.shape) != tuple(reference_mask.shape):
            raise ValueError(
                f"batch['{human_prior_pseudo_mask_key}'] shape {tuple(pseudo_mask.shape)} does not match "
                f"reference shape {tuple(reference_mask.shape)}."
            )

    pseudo_depth = None
    if human_prior_pseudo_depth_key:
        pseudo_depth = batch.get(human_prior_pseudo_depth_key, None)
        if pseudo_depth is None:
            raise ValueError(
                "human prior pseudo depth key "
                f"'{human_prior_pseudo_depth_key}' requires batch['{human_prior_pseudo_depth_key}']."
            )
        pseudo_depth = pseudo_depth.to(device=reference_mask.device, dtype=torch.float32)
        if tuple(pseudo_depth.shape) != tuple(reference_mask.shape):
            raise ValueError(
                f"batch['{human_prior_pseudo_depth_key}'] shape {tuple(pseudo_depth.shape)} does not match "
                f"reference shape {tuple(reference_mask.shape)}."
            )
        pseudo_depth = check_and_fix_inf_nan(
            pseudo_depth,
            f"{human_prior_pseudo_depth_key}_human_prior_pseudo_depth",
        )

    pseudo_world = None
    if human_prior_pseudo_world_key:
        pseudo_world = batch.get(human_prior_pseudo_world_key, None)
        if pseudo_world is None:
            raise ValueError(
                "human prior pseudo world key "
                f"'{human_prior_pseudo_world_key}' requires batch['{human_prior_pseudo_world_key}']."
            )
        pseudo_world = pseudo_world.to(device=reference_mask.device, dtype=torch.float32)
        expected_shape = tuple(reference_mask.shape) + (3,)
        if tuple(pseudo_world.shape) != expected_shape:
            raise ValueError(
                f"batch['{human_prior_pseudo_world_key}'] shape {tuple(pseudo_world.shape)} does not match "
                f"reference shape {expected_shape}."
            )
        pseudo_world = check_and_fix_inf_nan(
            pseudo_world,
            f"{human_prior_pseudo_world_key}_human_prior_pseudo_world",
            hard_max=None,
        )

    if human_prior_anchor_view_only:
        anchor_view_mask = _build_anchor_view_only_mask(batch, reference_mask)
        if anchor_view_mask is None:
            return None, None, None
        pseudo_mask = anchor_view_mask if pseudo_mask is None else (pseudo_mask & anchor_view_mask)
        if pseudo_depth is not None:
            pseudo_depth = torch.where(anchor_view_mask, pseudo_depth, torch.zeros_like(pseudo_depth))
        if pseudo_world is not None:
            pseudo_world = torch.where(
                anchor_view_mask.unsqueeze(-1),
                pseudo_world,
                torch.zeros_like(pseudo_world),
            )

    resolved_mask = pseudo_mask
    if pseudo_depth is not None:
        depth_valid_mask = torch.isfinite(pseudo_depth) & (pseudo_depth > 0.0)
        resolved_mask = depth_valid_mask if resolved_mask is None else (resolved_mask & depth_valid_mask)
    if pseudo_world is not None:
        world_valid_mask = torch.isfinite(pseudo_world).all(dim=-1)
        resolved_mask = world_valid_mask if resolved_mask is None else (resolved_mask & world_valid_mask)

    if resolved_mask is None or not bool(resolved_mask.any().item()):
        return None, None, None

    if pseudo_depth is not None:
        pseudo_depth = torch.where(resolved_mask, pseudo_depth, torch.zeros_like(pseudo_depth))
    if pseudo_world is not None:
        pseudo_world = torch.where(
            resolved_mask.unsqueeze(-1),
            pseudo_world,
            torch.zeros_like(pseudo_world),
        )
    return resolved_mask, pseudo_depth, pseudo_world


def _build_human_prior_weight_map(
    reference_map,
    *,
    prior_mask=None,
    prior_feature_map=None,
    human_prior_mask_floor=0.35,
):
    if prior_mask is None and prior_feature_map is None:
        return None

    mask_floor = float(max(0.0, min(1.0, human_prior_mask_floor)))
    weight_map = torch.zeros_like(reference_map, dtype=reference_map.dtype)
    if prior_feature_map is not None:
        weight_map = torch.maximum(
            weight_map,
            prior_feature_map.to(device=reference_map.device, dtype=reference_map.dtype),
        )
    if prior_mask is not None and mask_floor > 0.0:
        weight_map = torch.maximum(
            weight_map,
            prior_mask.to(device=reference_map.device, dtype=reference_map.dtype) * mask_floor,
        )
    return weight_map.clamp(0.0, 1.0)


def _build_human_prior_scale_map(
    batch,
    reference_map,
    *,
    human_prior_mask_key=None,
    human_prior_feature_map_key=None,
    human_prior_mask_floor=0.35,
    human_prior_scale=1.0,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
):
    scale_value = float(human_prior_scale)
    if scale_value == 1.0:
        return None
    prior_mask, prior_feature_map = _build_human_prior_region_components(
        batch,
        reference_map,
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    prior_weight_map = _build_human_prior_weight_map(
        reference_map,
        prior_mask=prior_mask,
        prior_feature_map=prior_feature_map,
        human_prior_mask_floor=human_prior_mask_floor,
    )
    if prior_weight_map is None:
        return None
    return 1.0 + prior_weight_map * (scale_value - 1.0)


def _build_human_prior_target_mask(
    prior_mask,
    prior_feature_map,
    *,
    feature_threshold=1e-6,
):
    target_mask = prior_mask
    if prior_feature_map is not None:
        feature_mask = prior_feature_map > float(feature_threshold)
        target_mask = feature_mask if target_mask is None else (target_mask | feature_mask)
    return target_mask


def _build_anchor_conditioned_disagreement_scale_maps(
    batch,
    reg_map,
    conf,
    conf_mask,
    *,
    anchor_conditioned_disagreement_cameras=None,
    anchor_conditioned_disagreement_conf_scale=1.0,
    anchor_conditioned_disagreement_reg_scale=1.0,
    anchor_conditioned_disagreement_train_only=True,
):
    if batch is None or reg_map is None or conf is None or conf_mask is None:
        return None, None
    if anchor_conditioned_disagreement_cameras is None:
        return None, None
    conf_scale = float(anchor_conditioned_disagreement_conf_scale)
    reg_scale = float(anchor_conditioned_disagreement_reg_scale)
    if conf_scale == 1.0 and reg_scale == 1.0:
        return None, None

    if anchor_conditioned_disagreement_train_only:
        phase = str(batch.get("_loss_phase", "")).lower()
        if phase != "train":
            return None, None

    if isinstance(anchor_conditioned_disagreement_cameras, str):
        target_cameras = {anchor_conditioned_disagreement_cameras}
    else:
        target_cameras = {
            str(camera_name)
            for camera_name in anchor_conditioned_disagreement_cameras
            if camera_name is not None
        }
    if not target_cameras:
        return None, None

    batch_anchor_cameras = _normalize_batch_anchor_cameras(
        batch.get("selection_anchor_camera"),
        reg_map.shape[0],
    )
    per_sample_target = torch.zeros(
        (reg_map.shape[0], 1, 1, 1),
        device=reg_map.device,
        dtype=torch.bool,
    )
    for batch_idx, camera_name in enumerate(batch_anchor_cameras):
        if camera_name in target_cameras:
            per_sample_target[batch_idx] = True
    if not bool(per_sample_target.any().item()):
        return None, None

    anchor_view_mask = _build_anchor_view_only_mask(batch, conf_mask)
    if anchor_view_mask is None:
        return None, None

    target_mask = conf_mask & anchor_view_mask & per_sample_target.expand_as(conf_mask)
    if not bool(target_mask.any().item()):
        return None, None

    detached_reg = check_and_fix_inf_nan(reg_map.detach(), "anchor_conditioned_disagreement_reg_map")
    detached_conf = check_and_fix_inf_nan(conf.detach(), "anchor_conditioned_disagreement_conf").clamp(0.0, 1.0)
    masked_reg = torch.where(target_mask, detached_reg, torch.zeros_like(detached_reg))
    per_sample_max = masked_reg.reshape(reg_map.shape[0], -1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    reg_norm = (detached_reg / per_sample_max).clamp(0.0, 1.0)
    disagreement = torch.abs(reg_norm - (1.0 - detached_conf).clamp(0.0, 1.0))
    disagreement = check_and_fix_inf_nan(disagreement, "anchor_conditioned_disagreement_map")

    ones = torch.ones_like(reg_map)
    conf_scale_map = None
    reg_scale_map = None
    if conf_scale != 1.0:
        conf_scale_map = torch.where(
            target_mask,
            1.0 + disagreement * (conf_scale - 1.0),
            ones,
        )
    if reg_scale != 1.0:
        reg_scale_map = torch.where(
            target_mask,
            1.0 + disagreement * (reg_scale - 1.0),
            ones,
        )
    return conf_scale_map, reg_scale_map


def _build_anchor_conditioned_unproject_consistency_scale_maps(
    batch,
    pred_depth,
    pred_pose_enc,
    reg_mask,
    conf_mask,
    *,
    image_size_hw=None,
    anchor_conditioned_unproject_consistency_cameras=None,
    anchor_conditioned_unproject_consistency_conf_scale=1.0,
    anchor_conditioned_unproject_consistency_reg_scale=1.0,
    anchor_conditioned_unproject_consistency_train_only=True,
    pose_encoding_type="absT_quaR_FoV",
):
    if batch is None or pred_depth is None or reg_mask is None or conf_mask is None:
        return None, None
    if anchor_conditioned_unproject_consistency_cameras is None:
        return None, None
    conf_scale = float(anchor_conditioned_unproject_consistency_conf_scale)
    reg_scale = float(anchor_conditioned_unproject_consistency_reg_scale)
    if conf_scale == 1.0 and reg_scale == 1.0:
        return None, None

    if anchor_conditioned_unproject_consistency_train_only:
        phase = str(batch.get("_loss_phase", "")).lower()
        if phase != "train":
            return None, None

    if isinstance(anchor_conditioned_unproject_consistency_cameras, str):
        target_cameras = {anchor_conditioned_unproject_consistency_cameras}
    else:
        target_cameras = {
            str(camera_name)
            for camera_name in anchor_conditioned_unproject_consistency_cameras
            if camera_name is not None
        }
    if not target_cameras:
        return None, None

    batch_anchor_cameras = _normalize_batch_anchor_cameras(
        batch.get("selection_anchor_camera"),
        reg_mask.shape[0],
    )
    per_sample_target = torch.zeros(
        (reg_mask.shape[0], 1, 1, 1),
        device=reg_mask.device,
        dtype=torch.bool,
    )
    for batch_idx, camera_name in enumerate(batch_anchor_cameras):
        if camera_name in target_cameras:
            per_sample_target[batch_idx] = True
    if not bool(per_sample_target.any().item()):
        return None, None

    anchor_view_mask = _build_anchor_view_only_mask(batch, conf_mask)
    if anchor_view_mask is None:
        return None, None

    if pred_pose_enc is None:
        raise ValueError(
            "anchor_conditioned_unproject_consistency_* requires predictions['pose_enc']."
        )
    if image_size_hw is None:
        images = batch.get("images", None)
        if images is None:
            raise ValueError(
                "anchor_conditioned_unproject_consistency_* requires batch['images'] or "
                "an explicit image_size_hw."
            )
        image_size_hw = images.shape[-2:]
    if "world_points" not in batch:
        raise ValueError(
            "anchor_conditioned_unproject_consistency_* requires batch['world_points']."
        )

    pred_world_points = unproject_depth_and_pose_to_world_points(
        pred_depth,
        pred_pose_enc,
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
    )
    finite_pred_world = torch.isfinite(pred_world_points).all(dim=-1)
    pred_world_points = check_and_fix_inf_nan(
        pred_world_points,
        "anchor_conditioned_unproject_consistency_pred_world_points",
    )
    gt_world_points = check_and_fix_inf_nan(
        batch["world_points"],
        "anchor_conditioned_unproject_consistency_gt_world_points",
    )

    target_mask = (
        conf_mask
        & reg_mask
        & anchor_view_mask
        & per_sample_target.expand_as(conf_mask)
        & finite_pred_world
        & (pred_depth[..., 0] > 0)
    )
    if not bool(target_mask.any().item()):
        return None, None

    residual = torch.norm(pred_world_points - gt_world_points, dim=-1)
    residual = check_and_fix_inf_nan(
        residual,
        "anchor_conditioned_unproject_consistency_residual",
    ).detach()
    masked_residual = torch.where(target_mask, residual, torch.zeros_like(residual))
    per_sample_max = (
        masked_residual.reshape(pred_depth.shape[0], -1)
        .amax(dim=1)
        .clamp_min(1e-6)
        .view(-1, 1, 1, 1)
    )
    normalized_residual = (residual / per_sample_max).clamp(0.0, 1.0)

    ones = torch.ones_like(normalized_residual)
    conf_scale_map = None
    reg_scale_map = None
    if conf_scale != 1.0:
        conf_scale_map = torch.where(
            target_mask,
            1.0 + normalized_residual * (conf_scale - 1.0),
            ones,
        )
    if reg_scale != 1.0:
        reg_scale_map = torch.where(
            target_mask,
            1.0 + normalized_residual * (reg_scale - 1.0),
            ones,
        )
    return conf_scale_map, reg_scale_map


def _build_anchor_conditioned_conf_scale_map(
    batch,
    conf_reg_map,
    anchor_conditioned_conf_target_cameras=None,
    anchor_conditioned_conf_target_scale=1.0,
    anchor_conditioned_conf_target_train_only=True,
    anchor_conditioned_conf_target_foreground_erode_px=0,
    anchor_conditioned_conf_target_foreground_edge_band_px=0,
    anchor_conditioned_conf_target_foreground_bottom_ratio=0.0,
    anchor_conditioned_conf_target_quality_min=None,
    anchor_conditioned_conf_target_quality_max=None,
    anchor_conditioned_conf_target_quality_interp="none",
    anchor_conditioned_conf_target_quality_low=None,
    anchor_conditioned_conf_target_quality_high=None,
    anchor_conditioned_conf_target_anchor_view_only=False,
    anchor_conditioned_conf_target_depth_conf_min=None,
    anchor_conditioned_conf_target_depth_conf_max=None,
    anchor_conditioned_conf_target_depth_conf_interp="none",
    anchor_conditioned_conf_target_depth_conf_low=None,
    anchor_conditioned_conf_target_depth_conf_high=None,
    anchor_conditioned_disagreement_cameras=None,
    anchor_conditioned_disagreement_conf_scale=1.0,
    anchor_conditioned_disagreement_reg_scale=1.0,
    anchor_conditioned_disagreement_train_only=True,
):
    if batch is None:
        return None
    if anchor_conditioned_conf_target_cameras is None:
        return None
    if isinstance(anchor_conditioned_conf_target_cameras, str):
        target_cameras = {anchor_conditioned_conf_target_cameras}
    else:
        target_cameras = {
            str(camera_name)
            for camera_name in anchor_conditioned_conf_target_cameras
            if camera_name is not None
        }
    if not target_cameras:
        return None

    scale_value = float(anchor_conditioned_conf_target_scale)
    if scale_value == 1.0:
        return None

    if anchor_conditioned_conf_target_train_only:
        phase = str(batch.get("_loss_phase", "")).lower()
        if phase != "train":
            return None

    batch_anchor_cameras = _normalize_batch_anchor_cameras(
        batch.get("selection_anchor_camera"),
        conf_reg_map.shape[0],
    )
    batch_anchor_quality_scores = _normalize_batch_anchor_quality_scores(
        batch.get("selection_anchor_quality_score"),
        conf_reg_map.shape[0],
    )
    per_sample_scale = torch.ones(
        (conf_reg_map.shape[0], 1, 1, 1),
        device=conf_reg_map.device,
        dtype=conf_reg_map.dtype,
    )
    quality_min = None if anchor_conditioned_conf_target_quality_min is None else float(anchor_conditioned_conf_target_quality_min)
    quality_max = None if anchor_conditioned_conf_target_quality_max is None else float(anchor_conditioned_conf_target_quality_max)
    quality_interp = str(anchor_conditioned_conf_target_quality_interp or "none").lower()
    quality_low = None if anchor_conditioned_conf_target_quality_low is None else float(anchor_conditioned_conf_target_quality_low)
    quality_high = None if anchor_conditioned_conf_target_quality_high is None else float(anchor_conditioned_conf_target_quality_high)
    if quality_interp not in {"none", "linear", "quadratic", "smoothstep"}:
        raise ValueError(
            "anchor_conditioned_conf_target_quality_interp must be one of "
            "'none', 'linear', 'quadratic', or 'smoothstep'."
        )
    if quality_interp != "none" and (quality_min is not None or quality_max is not None):
        raise ValueError(
            "anchor_conditioned_conf_target_quality_min/max are mutually exclusive with "
            "anchor_conditioned_conf_target_quality_interp != 'none'."
        )
    if quality_interp in {"linear", "quadratic", "smoothstep"}:
        if quality_low is None or quality_high is None:
            raise ValueError(
                "anchor_conditioned_conf_target_quality_interp in "
                "{'linear','quadratic','smoothstep'} requires "
                "anchor_conditioned_conf_target_quality_low/high."
            )
        if quality_high <= quality_low:
            raise ValueError(
                "anchor_conditioned_conf_target_quality_high must be greater than "
                "anchor_conditioned_conf_target_quality_low."
            )
    for batch_idx, camera_name in enumerate(batch_anchor_cameras):
        if camera_name not in target_cameras:
            continue
        quality_score = batch_anchor_quality_scores[batch_idx]
        if quality_interp in {"linear", "quadratic", "smoothstep"}:
            if quality_score is None:
                continue
            raw_quality_weight = (float(quality_score) - quality_low) / (quality_high - quality_low)
            raw_quality_weight = float(max(0.0, min(1.0, raw_quality_weight)))
            if raw_quality_weight <= 0.0:
                continue
            if quality_interp == "quadratic":
                quality_weight = raw_quality_weight * raw_quality_weight
            elif quality_interp == "smoothstep":
                quality_weight = raw_quality_weight * raw_quality_weight * (3.0 - 2.0 * raw_quality_weight)
            else:
                quality_weight = raw_quality_weight
            per_sample_scale[batch_idx] = 1.0 + quality_weight * (scale_value - 1.0)
            continue
        if quality_min is not None and (quality_score is None or quality_score < quality_min):
            continue
        if quality_max is not None and (quality_score is None or quality_score > quality_max):
            continue
        per_sample_scale[batch_idx] = scale_value
    scale_map = per_sample_scale.expand(-1, conf_reg_map.shape[1], conf_reg_map.shape[2], conf_reg_map.shape[3])

    target_mask = None

    if anchor_conditioned_conf_target_anchor_view_only:
        anchor_view_mask = _build_anchor_view_only_mask(batch, scale_map)
        if anchor_view_mask is None:
            return None
        target_mask = anchor_view_mask

    erode_px = int(max(0, anchor_conditioned_conf_target_foreground_erode_px))
    edge_band_px = int(max(0, anchor_conditioned_conf_target_foreground_edge_band_px))
    bottom_ratio = float(max(0.0, min(0.95, anchor_conditioned_conf_target_foreground_bottom_ratio)))
    if erode_px > 0 or edge_band_px > 0 or bottom_ratio > 0.0:
        foreground_masks = batch.get("foreground_masks", None)
        if foreground_masks is None:
            raise ValueError(
                "anchor_conditioned_conf_target_foreground_erode_px / foreground_edge_band_px / "
                "foreground_bottom_ratio "
                "requires batch['foreground_masks']."
            )
        foreground_masks = foreground_masks.to(device=conf_reg_map.device, dtype=torch.bool)
        target_region_mask = build_reliable_foreground_region_mask(
            foreground_masks,
            erode_px=erode_px,
            edge_band_px=edge_band_px,
            drop_bottom_ratio=bottom_ratio,
        )
        target_mask = target_region_mask if target_mask is None else (target_mask & target_region_mask)

    depth_conf_min = None if anchor_conditioned_conf_target_depth_conf_min is None else float(anchor_conditioned_conf_target_depth_conf_min)
    depth_conf_max = None if anchor_conditioned_conf_target_depth_conf_max is None else float(anchor_conditioned_conf_target_depth_conf_max)
    depth_conf_interp = str(anchor_conditioned_conf_target_depth_conf_interp or "none").lower()
    depth_conf_low = None if anchor_conditioned_conf_target_depth_conf_low is None else float(anchor_conditioned_conf_target_depth_conf_low)
    depth_conf_high = None if anchor_conditioned_conf_target_depth_conf_high is None else float(anchor_conditioned_conf_target_depth_conf_high)
    if depth_conf_interp not in {"none", "linear", "quadratic", "smoothstep"}:
        raise ValueError(
            "anchor_conditioned_conf_target_depth_conf_interp must be one of "
            "'none', 'linear', 'quadratic', or 'smoothstep'."
        )
    if depth_conf_interp != "none" and (depth_conf_min is not None or depth_conf_max is not None):
        raise ValueError(
            "anchor_conditioned_conf_target_depth_conf_min/max are mutually exclusive with "
            "anchor_conditioned_conf_target_depth_conf_interp != 'none'."
        )
    if depth_conf_interp in {"linear", "quadratic", "smoothstep"}:
        if depth_conf_low is None or depth_conf_high is None:
            raise ValueError(
                "anchor_conditioned_conf_target_depth_conf_interp in "
                "{'linear','quadratic','smoothstep'} requires "
                "anchor_conditioned_conf_target_depth_conf_low/high."
            )
        if depth_conf_high <= depth_conf_low:
            raise ValueError(
                "anchor_conditioned_conf_target_depth_conf_high must be greater than "
                "anchor_conditioned_conf_target_depth_conf_low."
            )

    depth_conf_maps = None
    if (
        depth_conf_min is not None
        or depth_conf_max is not None
        or depth_conf_interp in {"linear", "quadratic", "smoothstep"}
    ):
        depth_conf_maps = batch.get("depth_conf_maps", None)
        if depth_conf_maps is None:
            raise ValueError(
                "anchor_conditioned_conf_target_depth_conf_* requires batch['depth_conf_maps']."
            )
        depth_conf_maps = depth_conf_maps.to(device=conf_reg_map.device, dtype=conf_reg_map.dtype)
    if depth_conf_interp in {"linear", "quadratic", "smoothstep"}:
        valid_depth_conf = torch.isfinite(depth_conf_maps)
        depth_conf_position = torch.zeros_like(depth_conf_maps)
        depth_conf_position[valid_depth_conf] = (
            (depth_conf_maps[valid_depth_conf] - depth_conf_low) / (depth_conf_high - depth_conf_low)
        ).clamp(0.0, 1.0)
        depth_conf_effect = 1.0 - depth_conf_position
        if depth_conf_interp == "quadratic":
            depth_conf_effect = depth_conf_effect * depth_conf_effect
        elif depth_conf_interp == "smoothstep":
            depth_conf_effect = depth_conf_effect * depth_conf_effect * (3.0 - 2.0 * depth_conf_effect)
        scale_map = torch.where(
            valid_depth_conf,
            1.0 + depth_conf_effect * (scale_map - 1.0),
            torch.ones_like(scale_map),
        )
    if depth_conf_min is not None or depth_conf_max is not None:
        depth_conf_mask = torch.isfinite(depth_conf_maps)
        if depth_conf_min is not None:
            depth_conf_mask = depth_conf_mask & (depth_conf_maps >= depth_conf_min)
        if depth_conf_max is not None:
            depth_conf_mask = depth_conf_mask & (depth_conf_maps <= depth_conf_max)
        target_mask = depth_conf_mask if target_mask is None else (target_mask & depth_conf_mask)

    if target_mask is None:
        return scale_map
    return torch.where(target_mask, scale_map, torch.ones_like(scale_map))


def _build_anchor_conditioned_conf_mask_drop_map(
    batch,
    reference_mask,
    anchor_conditioned_conf_mask_cameras=None,
    anchor_conditioned_conf_mask_train_only=True,
    anchor_conditioned_conf_mask_foreground_erode_px=0,
    anchor_conditioned_conf_mask_foreground_edge_band_px=0,
    anchor_conditioned_conf_mask_foreground_bottom_ratio=0.0,
    anchor_conditioned_conf_mask_quality_min=None,
    anchor_conditioned_conf_mask_quality_max=None,
):
    if batch is None:
        return None
    if anchor_conditioned_conf_mask_cameras is None:
        return None
    if isinstance(anchor_conditioned_conf_mask_cameras, str):
        target_cameras = {anchor_conditioned_conf_mask_cameras}
    else:
        target_cameras = {
            str(camera_name)
            for camera_name in anchor_conditioned_conf_mask_cameras
            if camera_name is not None
        }
    if not target_cameras:
        return None

    if anchor_conditioned_conf_mask_train_only:
        phase = str(batch.get("_loss_phase", "")).lower()
        if phase != "train":
            return None

    batch_anchor_cameras = _normalize_batch_anchor_cameras(
        batch.get("selection_anchor_camera"),
        reference_mask.shape[0],
    )
    batch_anchor_quality_scores = _normalize_batch_anchor_quality_scores(
        batch.get("selection_anchor_quality_score"),
        reference_mask.shape[0],
    )
    quality_min = None if anchor_conditioned_conf_mask_quality_min is None else float(anchor_conditioned_conf_mask_quality_min)
    quality_max = None if anchor_conditioned_conf_mask_quality_max is None else float(anchor_conditioned_conf_mask_quality_max)
    per_sample_drop = torch.zeros(
        (reference_mask.shape[0], 1, 1, 1),
        device=reference_mask.device,
        dtype=torch.bool,
    )
    for batch_idx, camera_name in enumerate(batch_anchor_cameras):
        if camera_name not in target_cameras:
            continue
        quality_score = batch_anchor_quality_scores[batch_idx]
        if quality_min is not None and (quality_score is None or quality_score < quality_min):
            continue
        if quality_max is not None and (quality_score is None or quality_score > quality_max):
            continue
        per_sample_drop[batch_idx] = True

    if not bool(per_sample_drop.any().item()):
        return None

    drop_map = per_sample_drop.expand_as(reference_mask)
    erode_px = int(max(0, anchor_conditioned_conf_mask_foreground_erode_px))
    edge_band_px = int(max(0, anchor_conditioned_conf_mask_foreground_edge_band_px))
    bottom_ratio = float(max(0.0, min(0.95, anchor_conditioned_conf_mask_foreground_bottom_ratio)))
    if erode_px <= 0 and edge_band_px <= 0 and bottom_ratio <= 0.0:
        return drop_map & reference_mask

    foreground_masks = batch.get("foreground_masks", None)
    if foreground_masks is None:
        raise ValueError(
            "anchor_conditioned_conf_mask_foreground_erode_px / foreground_edge_band_px / "
            "foreground_bottom_ratio "
            "requires batch['foreground_masks']."
        )
    foreground_masks = foreground_masks.to(device=reference_mask.device, dtype=torch.bool)
    target_region_mask = build_reliable_foreground_region_mask(
        foreground_masks,
        erode_px=erode_px,
        edge_band_px=edge_band_px,
        drop_bottom_ratio=bottom_ratio,
    )
    return drop_map & target_region_mask & reference_mask


def regression_loss(
    pred,
    gt,
    mask,
    conf=None,
    conf_mask=None,
    gradient_loss_fn=None,
    gamma=1.0,
    alpha=0.2,
    valid_range=-1,
    conf_loss_aggregation="pixel_mean",
    respect_conf_mask_in_grad_conf=False,
    batch=None,
    anchor_conditioned_reg_target_cameras=None,
    anchor_conditioned_reg_target_scale=1.0,
    anchor_conditioned_reg_target_train_only=True,
    anchor_conditioned_reg_target_foreground_erode_px=0,
    anchor_conditioned_reg_target_foreground_edge_band_px=0,
    anchor_conditioned_reg_target_foreground_bottom_ratio=0.0,
    anchor_conditioned_reg_target_quality_min=None,
    anchor_conditioned_reg_target_quality_max=None,
    anchor_conditioned_reg_target_quality_interp="none",
    anchor_conditioned_reg_target_quality_low=None,
    anchor_conditioned_reg_target_quality_high=None,
    anchor_conditioned_reg_target_anchor_view_only=False,
    anchor_conditioned_reg_target_depth_conf_min=None,
    anchor_conditioned_reg_target_depth_conf_max=None,
    anchor_conditioned_reg_target_depth_conf_interp="none",
    anchor_conditioned_reg_target_depth_conf_low=None,
    anchor_conditioned_reg_target_depth_conf_high=None,
    anchor_conditioned_conf_target_cameras=None,
    anchor_conditioned_conf_target_scale=1.0,
    anchor_conditioned_conf_target_train_only=True,
    anchor_conditioned_conf_target_foreground_erode_px=0,
    anchor_conditioned_conf_target_foreground_edge_band_px=0,
    anchor_conditioned_conf_target_foreground_bottom_ratio=0.0,
    anchor_conditioned_conf_target_quality_min=None,
    anchor_conditioned_conf_target_quality_max=None,
    anchor_conditioned_conf_target_quality_interp="none",
    anchor_conditioned_conf_target_quality_low=None,
    anchor_conditioned_conf_target_quality_high=None,
    anchor_conditioned_conf_target_anchor_view_only=False,
    anchor_conditioned_conf_target_depth_conf_min=None,
    anchor_conditioned_conf_target_depth_conf_max=None,
    anchor_conditioned_conf_target_depth_conf_interp="none",
    anchor_conditioned_conf_target_depth_conf_low=None,
    anchor_conditioned_conf_target_depth_conf_high=None,
    anchor_conditioned_disagreement_cameras=None,
    anchor_conditioned_disagreement_conf_scale=1.0,
    anchor_conditioned_disagreement_reg_scale=1.0,
    anchor_conditioned_disagreement_train_only=True,
    anchor_conditioned_unproject_consistency_cameras=None,
    anchor_conditioned_unproject_consistency_conf_scale=1.0,
    anchor_conditioned_unproject_consistency_reg_scale=1.0,
    anchor_conditioned_unproject_consistency_train_only=True,
    anchor_conditioned_unproject_consistency_pred_pose_enc=None,
    anchor_conditioned_unproject_consistency_image_size_hw=None,
    anchor_conditioned_unproject_consistency_pose_encoding_type="absT_quaR_FoV",
    human_prior_mask_key=None,
    human_prior_feature_map_key=None,
    human_prior_mask_floor=0.35,
    human_prior_reg_scale=1.0,
    human_prior_conf_scale=1.0,
    human_prior_train_only=True,
    human_prior_anchor_view_only=False,
):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
        conf_loss_aggregation: How to aggregate the confidence loss across valid pixels/views

    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    if conf_mask is None:
        conf_mask = mask
    conf_mask = conf_mask & mask

    reg_map = torch.norm(gt - pred, dim=-1)
    reg_map = check_and_fix_inf_nan(reg_map, "loss_reg_map")

    reg_target_scale_map = _build_anchor_conditioned_conf_scale_map(
        batch,
        reg_map,
        anchor_conditioned_conf_target_cameras=anchor_conditioned_reg_target_cameras,
        anchor_conditioned_conf_target_scale=anchor_conditioned_reg_target_scale,
        anchor_conditioned_conf_target_train_only=anchor_conditioned_reg_target_train_only,
        anchor_conditioned_conf_target_foreground_erode_px=anchor_conditioned_reg_target_foreground_erode_px,
        anchor_conditioned_conf_target_foreground_edge_band_px=anchor_conditioned_reg_target_foreground_edge_band_px,
        anchor_conditioned_conf_target_foreground_bottom_ratio=anchor_conditioned_reg_target_foreground_bottom_ratio,
        anchor_conditioned_conf_target_quality_min=anchor_conditioned_reg_target_quality_min,
        anchor_conditioned_conf_target_quality_max=anchor_conditioned_reg_target_quality_max,
        anchor_conditioned_conf_target_quality_interp=anchor_conditioned_reg_target_quality_interp,
        anchor_conditioned_conf_target_quality_low=anchor_conditioned_reg_target_quality_low,
        anchor_conditioned_conf_target_quality_high=anchor_conditioned_reg_target_quality_high,
        anchor_conditioned_conf_target_anchor_view_only=anchor_conditioned_reg_target_anchor_view_only,
        anchor_conditioned_conf_target_depth_conf_min=anchor_conditioned_reg_target_depth_conf_min,
        anchor_conditioned_conf_target_depth_conf_max=anchor_conditioned_reg_target_depth_conf_max,
        anchor_conditioned_conf_target_depth_conf_interp=anchor_conditioned_reg_target_depth_conf_interp,
        anchor_conditioned_conf_target_depth_conf_low=anchor_conditioned_reg_target_depth_conf_low,
        anchor_conditioned_conf_target_depth_conf_high=anchor_conditioned_reg_target_depth_conf_high,
    )
    reg_target_map = reg_map if reg_target_scale_map is None else reg_map * reg_target_scale_map

    conf_reg_scale_map = _build_anchor_conditioned_conf_scale_map(
        batch,
        reg_target_map,
        anchor_conditioned_conf_target_cameras=anchor_conditioned_conf_target_cameras,
        anchor_conditioned_conf_target_scale=anchor_conditioned_conf_target_scale,
        anchor_conditioned_conf_target_train_only=anchor_conditioned_conf_target_train_only,
        anchor_conditioned_conf_target_foreground_erode_px=anchor_conditioned_conf_target_foreground_erode_px,
        anchor_conditioned_conf_target_foreground_edge_band_px=anchor_conditioned_conf_target_foreground_edge_band_px,
        anchor_conditioned_conf_target_foreground_bottom_ratio=anchor_conditioned_conf_target_foreground_bottom_ratio,
        anchor_conditioned_conf_target_quality_min=anchor_conditioned_conf_target_quality_min,
        anchor_conditioned_conf_target_quality_max=anchor_conditioned_conf_target_quality_max,
        anchor_conditioned_conf_target_quality_interp=anchor_conditioned_conf_target_quality_interp,
        anchor_conditioned_conf_target_quality_low=anchor_conditioned_conf_target_quality_low,
        anchor_conditioned_conf_target_quality_high=anchor_conditioned_conf_target_quality_high,
        anchor_conditioned_conf_target_anchor_view_only=anchor_conditioned_conf_target_anchor_view_only,
        anchor_conditioned_conf_target_depth_conf_min=anchor_conditioned_conf_target_depth_conf_min,
        anchor_conditioned_conf_target_depth_conf_max=anchor_conditioned_conf_target_depth_conf_max,
        anchor_conditioned_conf_target_depth_conf_interp=anchor_conditioned_conf_target_depth_conf_interp,
        anchor_conditioned_conf_target_depth_conf_low=anchor_conditioned_conf_target_depth_conf_low,
        anchor_conditioned_conf_target_depth_conf_high=anchor_conditioned_conf_target_depth_conf_high,
    )
    conf_reg_map = reg_target_map if conf_reg_scale_map is None else reg_target_map * conf_reg_scale_map
    disagreement_conf_scale_map, disagreement_reg_scale_map = _build_anchor_conditioned_disagreement_scale_maps(
        batch,
        reg_map,
        conf,
        conf_mask,
        anchor_conditioned_disagreement_cameras=anchor_conditioned_disagreement_cameras,
        anchor_conditioned_disagreement_conf_scale=anchor_conditioned_disagreement_conf_scale,
        anchor_conditioned_disagreement_reg_scale=anchor_conditioned_disagreement_reg_scale,
        anchor_conditioned_disagreement_train_only=anchor_conditioned_disagreement_train_only,
    )
    (
        unproject_consistency_conf_scale_map,
        unproject_consistency_reg_scale_map,
    ) = _build_anchor_conditioned_unproject_consistency_scale_maps(
        batch,
        pred,
        anchor_conditioned_unproject_consistency_pred_pose_enc,
        mask,
        conf_mask,
        image_size_hw=anchor_conditioned_unproject_consistency_image_size_hw,
        anchor_conditioned_unproject_consistency_cameras=anchor_conditioned_unproject_consistency_cameras,
        anchor_conditioned_unproject_consistency_conf_scale=anchor_conditioned_unproject_consistency_conf_scale,
        anchor_conditioned_unproject_consistency_reg_scale=anchor_conditioned_unproject_consistency_reg_scale,
        anchor_conditioned_unproject_consistency_train_only=anchor_conditioned_unproject_consistency_train_only,
        pose_encoding_type=anchor_conditioned_unproject_consistency_pose_encoding_type,
    )
    if disagreement_reg_scale_map is not None:
        reg_target_map = reg_target_map * disagreement_reg_scale_map
    if disagreement_conf_scale_map is not None:
        conf_reg_map = conf_reg_map * disagreement_conf_scale_map
    if unproject_consistency_reg_scale_map is not None:
        reg_target_map = reg_target_map * unproject_consistency_reg_scale_map
    if unproject_consistency_conf_scale_map is not None:
        conf_reg_map = conf_reg_map * unproject_consistency_conf_scale_map

    human_prior_reg_scale_map = _build_human_prior_scale_map(
        batch,
        reg_target_map,
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_mask_floor=human_prior_mask_floor,
        human_prior_scale=human_prior_reg_scale,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    human_prior_conf_scale_map = _build_human_prior_scale_map(
        batch,
        conf_reg_map,
        human_prior_mask_key=human_prior_mask_key,
        human_prior_feature_map_key=human_prior_feature_map_key,
        human_prior_mask_floor=human_prior_mask_floor,
        human_prior_scale=human_prior_conf_scale,
        human_prior_train_only=human_prior_train_only,
        human_prior_anchor_view_only=human_prior_anchor_view_only,
    )
    if human_prior_reg_scale_map is not None:
        reg_target_map = reg_target_map * human_prior_reg_scale_map
    if human_prior_conf_scale_map is not None:
        conf_reg_map = conf_reg_map * human_prior_conf_scale_map

    # Compute L2 distance between predicted and ground truth points
    loss_reg = reg_target_map[mask]
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones.
    if conf_loss_aggregation == "pixel_mean":
        conf_reg = conf_reg_map[conf_mask]
        conf_reg = check_and_fix_inf_nan(conf_reg, "conf_reg")
        loss_conf_values = gamma * conf_reg * conf[conf_mask] - alpha * torch.log(conf[conf_mask])
        loss_conf_values = check_and_fix_inf_nan(loss_conf_values, "loss_conf")
        loss_conf = None
    elif conf_loss_aggregation == "active_view_mean":
        per_view_conf_losses = []
        for batch_idx in range(bb):
            for view_idx in range(ss):
                view_conf_mask = conf_mask[batch_idx, view_idx]
                if int(view_conf_mask.sum().item()) <= 0:
                    continue

                view_reg = conf_reg_map[batch_idx, view_idx][view_conf_mask]
                view_reg = check_and_fix_inf_nan(view_reg, "conf_reg_view")
                view_conf = conf[batch_idx, view_idx][view_conf_mask]
                view_loss = gamma * view_reg * view_conf - alpha * torch.log(view_conf)
                view_loss = check_and_fix_inf_nan(view_loss, "loss_conf_view")
                if valid_range > 0:
                    view_loss = filter_by_quantile(view_loss, valid_range)
                per_view_conf_losses.append(view_loss.mean())

        if per_view_conf_losses:
            loss_conf = torch.stack(per_view_conf_losses).mean()
            loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf_depth")
        else:
            loss_conf = (0.0 * pred).mean()
        loss_conf_values = None
    else:
        raise ValueError(
            "conf_loss_aggregation must be one of 'pixel_mean' or 'active_view_mean'."
        )
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if gradient_loss_fn is not None and "conf" in gradient_loss_fn:
        if conf is None:
            raise ValueError("gradient_loss_fn requests confidence weighting but conf is None.")
        if respect_conf_mask_in_grad_conf:
            # Pixels outside conf_mask should keep plain gradient supervision
            # without confidence weighting or confidence regularization.
            grad_conf = torch.where(conf_mask, conf, torch.ones_like(conf))
        else:
            grad_conf = conf
        to_feed_conf = grad_conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf_values is not None:
        if loss_conf_values.numel() > 0:
            # Filter out outliers using quantile-based thresholding
            if valid_range > 0:
                loss_conf_values = filter_by_quantile(loss_conf_values, valid_range)

            loss_conf_values = check_and_fix_inf_nan(loss_conf_values, "loss_conf_depth")
            loss_conf = loss_conf_values.mean()
        else:
            loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        mask_x_f = mask_x.to(dtype=grad_x.dtype)
        mask_y_f = mask_y.to(dtype=grad_y.dtype)

        # Keep the confidence regularizer inside the valid gradient support only.
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x) * mask_x_f
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y) * mask_y_f

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def build_quantile_mask(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Build a quantile mask used to remove extreme outliers.
    """
    if loss_tensor.numel() <= min_elements:
        return None

    sampled_tensor = loss_tensor
    if loss_tensor.numel() > 100000000:
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        sampled_tensor = loss_tensor.view(-1)[indices]

    sampled_tensor = sampled_tensor.clamp(max=hard_max)
    quantile_thresh = torch_quantile(sampled_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    clamped_tensor = loss_tensor.clamp(max=hard_max)
    quantile_mask = clamped_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return quantile_mask
    return None


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    quantile_mask = build_quantile_mask(
        loss_tensor,
        valid_range,
        min_elements=min_elements,
        hard_max=hard_max,
    )
    if quantile_mask is not None:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''
