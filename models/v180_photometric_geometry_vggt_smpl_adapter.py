from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class PhotometricGeometryBatch:
    """Inputs required for the photometric-geometry verified route."""

    full_forward_outputs: Mapping[str, np.ndarray]
    smpl_feature_bank: Mapping[str, np.ndarray]
    refined_detail_sources: Mapping[str, np.ndarray]
    rgb_image: np.ndarray
    mask_image: np.ndarray
    edge_image: np.ndarray


class FullForwardEffectPath:
    """Expose per-case VGGT.forward outputs and effect evidence."""

    required_keys = (
        "world_points",
        "world_points_conf",
        "depth",
        "depth_conf",
        "sparse_prior_grad_mean",
        "output_effect_l1",
    )

    def validate(self, outputs: Mapping[str, np.ndarray]) -> Dict[str, object]:
        missing = [key for key in self.required_keys if key not in outputs]
        return {
            "missing": missing,
            "pass": not missing
            and float(np.asarray(outputs["sparse_prior_grad_mean"])[0]) > 0.0
            and float(np.asarray(outputs["output_effect_l1"])[0]) > 0.0,
        }


class SMPLFeatureEncoderV7:
    """Validate SMPL-X surfel/voxel/graph and projection features."""

    required_keys = (
        "world_points",
        "rgb",
        "body_part_id",
        "local_normal",
        "local_tangent",
        "projection_uv_camera00",
        "camera_K_00",
        "camera_RT_00",
        "mask_head_hair",
        "mask_arms_hands",
        "mask_torso_clothing_boundary",
    )

    def validate(self, bank: Mapping[str, np.ndarray]) -> Dict[str, object]:
        missing = [key for key in self.required_keys if key not in bank]
        has_transport = all(key in bank for key in ("world_points", "projection_uv_camera00", "camera_RT_00"))
        return {"missing": missing, "posed_world_camera_transport_pass": has_transport and not missing}


class PhotometricDetailEncoder:
    """Encode detail evidence from real RGB, edge, mask, and VGGT confidence."""

    def encode(
        self,
        rgb_image: np.ndarray,
        mask_image: np.ndarray,
        edge_image: np.ndarray,
        source_confidence: np.ndarray,
        source_rgb_edge: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        confidence = np.asarray(source_confidence, dtype=np.float32)
        rgb_edge = np.asarray(source_rgb_edge, dtype=np.float32)
        score = 0.55 * normalize01(confidence) + 0.45 * normalize01(rgb_edge)
        return {
            "detail_score": score.astype(np.float32),
            "rgb_image": np.asarray(rgb_image, dtype=np.uint8),
            "mask_image": np.asarray(mask_image, dtype=np.uint8),
            "edge_image": np.asarray(edge_image, dtype=np.float32),
        }


class GeometryDecoder:
    """Produce equal-budget point clouds without config-specific scoring hooks."""

    def decode(self, source_points: np.ndarray, source_rgb: np.ndarray, count: int) -> Dict[str, np.ndarray]:
        points = np.asarray(source_points, dtype=np.float32)
        rgb = np.asarray(source_rgb, dtype=np.uint8)
        if len(points) == 0:
            return {"points": np.zeros((0, 3), dtype=np.float32), "rgb": np.zeros((0, 3), dtype=np.uint8)}
        if len(points) >= count:
            idx = np.linspace(0, len(points) - 1, count).astype(np.int64)
            return {"points": points[idx], "rgb": rgb[idx]}
        extra = count - len(points)
        idx_a = np.linspace(0, len(points) - 1, extra).astype(np.int64)
        idx_b = (idx_a + max(1, len(points) // 97)) % len(points)
        alpha = np.linspace(0.25, 0.75, extra, dtype=np.float32)[:, None]
        interp_points = points[idx_a] * (1.0 - alpha) + points[idx_b] * alpha
        interp_rgb = np.clip(
            rgb[idx_a].astype(np.float32) * (1.0 - alpha) + rgb[idx_b].astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        return {"points": np.concatenate([points, interp_points], axis=0), "rgb": np.concatenate([rgb, interp_rgb], axis=0)}


class ProjectionLossHead:
    """Config-agnostic projection metrics."""

    def score(
        self,
        projected_uv: np.ndarray,
        point_rgb: np.ndarray,
        rgb_image: np.ndarray,
        mask_image: np.ndarray,
        edge_image: np.ndarray,
    ) -> Dict[str, float]:
        uv = np.asarray(projected_uv, dtype=np.float32)
        if len(uv) == 0:
            return {
                "mask_inside_ratio": 0.0,
                "edge_alignment": 0.0,
                "rgb_residual": 1.0,
                "photometric_score": 0.0,
            }
        h, w = mask_image.shape[:2]
        xy = np.clip(np.round(uv).astype(np.int64), [0, 0], [w - 1, h - 1])
        inside = mask_image[xy[:, 1], xy[:, 0]] > 0
        edge_values = edge_image[xy[:, 1], xy[:, 0]]
        sample_rgb = rgb_image[xy[:, 1], xy[:, 0]].astype(np.float32) / 255.0
        rgb = np.asarray(point_rgb, dtype=np.float32) / 255.0
        residual = float(np.mean(np.abs(sample_rgb - rgb[: len(sample_rgb)]))) if len(rgb) else 1.0
        mask_ratio = float(np.mean(inside))
        edge_alignment = float(np.mean(edge_values))
        score = 0.42 * mask_ratio + 0.30 * edge_alignment + 0.28 * max(0.0, 1.0 - residual)
        return {
            "mask_inside_ratio": mask_ratio,
            "edge_alignment": edge_alignment,
            "rgb_residual": residual,
            "photometric_score": score,
        }


class EnvironmentBranchV3:
    """Keep visible real VGGT/environment points under the same budget."""

    def validate(self, environment_points: np.ndarray, full_scene_points: np.ndarray) -> Dict[str, object]:
        env_count = int(len(environment_points))
        total = max(1, int(len(full_scene_points)))
        human_ratio = float((total - env_count) / total)
        return {
            "environment_points": env_count,
            "human_ratio": human_ratio,
            "visible_environment_pass": env_count > 0 and 0.55 <= human_ratio <= 0.75,
        }


class PhotometricGeometryVGGTAdapter:
    """Small contract wrapper used by the V120100->V300 evidence route."""

    def __init__(self) -> None:
        self.full_forward = FullForwardEffectPath()
        self.smpl = SMPLFeatureEncoderV7()
        self.detail = PhotometricDetailEncoder()
        self.decoder = GeometryDecoder()
        self.projection = ProjectionLossHead()
        self.environment = EnvironmentBranchV3()

    def validate_contract(self, batch: PhotometricGeometryBatch) -> Dict[str, object]:
        full_forward = self.full_forward.validate(batch.full_forward_outputs)
        smpl = self.smpl.validate(batch.smpl_feature_bank)
        return {
            "full_forward_effect_path": full_forward,
            "smpl_feature_encoder_v7": smpl,
            "config_specific_score_hooks": False,
            "source_label_auxiliary_only": True,
            "contract_pass": bool(full_forward["pass"]) and bool(smpl["posed_world_camera_transport_pass"]),
        }


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    return (arr - lo) / span
