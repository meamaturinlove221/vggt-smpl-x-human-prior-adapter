from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_VERTEX_FEATURE_NAMES = (
    "density",
    "world_x",
    "world_y",
    "world_z",
    "cam_x",
    "cam_y",
    "cam_z",
    "radius",
)

DEFAULT_SURFACE_FEATURE_NAMES = (
    "density",
    "visibility",
    "body_local_x",
    "body_local_y",
    "body_local_z",
    "cam_x",
    "cam_y",
    "cam_z",
    "uv_x",
    "uv_y",
    "radius",
    "vertex_id_sin",
    "vertex_id_cos",
    "body_part_embed_x",
    "body_part_embed_y",
    "skinning_embed_x",
    "skinning_embed_y",
)

DEFAULT_SUMMARY_FEATURE_NAMES = (
    "center_x",
    "center_y",
    "center_z",
    "spread_x",
    "spread_y",
    "spread_z",
    "radius_mean",
    "occupancy_ratio",
    "body_part_embed_x",
    "body_part_embed_y",
    "skinning_embed_x",
    "skinning_embed_y",
)

DEFAULT_SUMMARY_BIN_NAMES = tuple(
    f"{height}_{lateral}_{depth}"
    for height in ("lower", "mid", "upper")
    for lateral in ("left", "right")
    for depth in ("back", "front")
)

_BODY_PART_ANCHOR_POINTS = np.asarray(
    [
        [0.00, 0.86, 0.06],   # head
        [0.00, 0.28, 0.02],   # torso
        [0.00, -0.06, 0.00],  # pelvis
        [-0.50, 0.26, 0.04],  # left arm
        [0.50, 0.26, 0.04],   # right arm
        [-0.88, 0.10, 0.08],  # left hand
        [0.88, 0.10, 0.08],   # right hand
        [-0.18, -0.54, 0.04], # left leg
        [0.18, -0.54, 0.04],  # right leg
        [-0.20, -0.96, 0.14], # left foot
        [0.20, -0.96, 0.14],  # right foot
    ],
    dtype=np.float32,
)
_BODY_PART_EMBEDDINGS = np.stack(
    [
        np.cos(np.linspace(0.0, 2.0 * np.pi, num=len(_BODY_PART_ANCHOR_POINTS), endpoint=False)),
        np.sin(np.linspace(0.0, 2.0 * np.pi, num=len(_BODY_PART_ANCHOR_POINTS), endpoint=False)),
    ],
    axis=1,
).astype(np.float32)
_BODY_PART_TO_CHAIN = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],  # head -> core
        [1.0, 0.0, 0.0, 0.0],  # torso -> core
        [1.0, 0.0, 0.0, 0.0],  # pelvis -> core
        [0.0, 1.0, 0.0, 0.0],  # left arm
        [0.0, 0.0, 1.0, 0.0],  # right arm
        [0.0, 1.0, 0.0, 1.0],  # left hand -> left + distal
        [0.0, 0.0, 1.0, 1.0],  # right hand -> right + distal
        [0.0, 1.0, 0.0, 0.0],  # left leg
        [0.0, 0.0, 1.0, 0.0],  # right leg
        [0.0, 1.0, 0.0, 1.0],  # left foot -> left + distal
        [0.0, 0.0, 1.0, 1.0],  # right foot -> right + distal
    ],
    dtype=np.float32,
)
_SKINNING_CHAIN_EMBEDDINGS = np.asarray(
    [
        [0.00, 1.00],   # core
        [-1.0, 0.00],   # left chain
        [1.00, 0.00],   # right chain
        [0.00, -1.0],   # distal chain
    ],
    dtype=np.float32,
)


def _morph_kernel(radius: int):
    radius = int(max(0, radius))
    if radius <= 0:
        return None
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def binary_dilate_mask(mask, radius: int):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def binary_erode_mask(mask, radius: int):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.erode(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def binary_close_mask(mask, radius: int):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def build_edge_band_mask(mask, radius: int):
    radius = int(max(0, radius))
    if radius <= 0:
        return np.zeros_like(np.asarray(mask, dtype=bool), dtype=bool)
    dilated = binary_dilate_mask(mask, radius)
    eroded = binary_erode_mask(mask, radius)
    return dilated & ~eroded


def _safe_body_scale(points_xyz: np.ndarray, eps: float = 1e-6) -> float:
    if points_xyz.size == 0:
        return 1.0
    centered = points_xyz - points_xyz.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(centered, axis=1)
    finite = np.isfinite(radial)
    if not np.any(finite):
        return 1.0
    scale = float(np.percentile(radial[finite], 90))
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.mean(radial[finite]))
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
    return max(scale, eps)


def _softmax_numpy(logits: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=axis, keepdims=True), eps, None)


def build_vertex_identity_condition_features(
    body_local_vertices,
    *,
    clip_value: float = 6.0,
    eps: float = 1e-6,
) -> dict[str, np.ndarray | float]:
    local_rel, body_scale = normalize_body_local_vertices(body_local_vertices, clip_value=clip_value, eps=eps)
    if local_rel.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return {
            "local_rel": local_rel,
            "body_scale": float(body_scale),
            "vertex_id_embedding": empty,
            "body_part_embedding": empty,
            "skinning_embedding": empty,
        }

    vertex_count = int(local_rel.shape[0])
    denom = float(max(vertex_count - 1, 1))
    phase = (2.0 * np.pi * np.arange(vertex_count, dtype=np.float32)) / denom
    vertex_id_embedding = np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)

    dist_sq = np.sum(
        (local_rel[:, None, :] - _BODY_PART_ANCHOR_POINTS[None, :, :]) ** 2,
        axis=-1,
    ).astype(np.float32)
    body_part_weights = _softmax_numpy(-(dist_sq / 0.18), axis=1, eps=eps)
    body_part_embedding = (body_part_weights @ _BODY_PART_EMBEDDINGS).astype(np.float32)

    chain_weights = body_part_weights @ _BODY_PART_TO_CHAIN
    chain_weights = chain_weights / np.clip(chain_weights.sum(axis=1, keepdims=True), eps, None)
    skinning_embedding = (chain_weights @ _SKINNING_CHAIN_EMBEDDINGS).astype(np.float32)

    return {
        "local_rel": local_rel.astype(np.float32),
        "body_scale": float(body_scale),
        "vertex_id_embedding": vertex_id_embedding,
        "body_part_embedding": body_part_embedding,
        "skinning_embedding": skinning_embedding,
    }


def build_vertex_geometry_features(
    vertices_world,
    cam_points=None,
    *,
    clip_value: float = 6.0,
    eps: float = 1e-6,
) -> np.ndarray:
    world_points = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0:
        return np.zeros((0, len(DEFAULT_VERTEX_FEATURE_NAMES) - 1), dtype=np.float32)

    if cam_points is None:
        cam_points = world_points
    cam_points = np.asarray(cam_points, dtype=np.float32).reshape(-1, 3)

    center = world_points.mean(axis=0, keepdims=True)
    scale = _safe_body_scale(world_points, eps=eps)
    world_rel = np.clip((world_points - center) / scale, -clip_value, clip_value)
    cam_rel = np.clip(cam_points / scale, -clip_value, clip_value)
    radius = np.clip(np.linalg.norm(world_rel, axis=1, keepdims=True), 0.0, clip_value)
    return np.concatenate([world_rel, cam_rel, radius], axis=1).astype(np.float32)


def axis_angle_to_rotation_matrix(axis_angle) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float32).reshape(3, 1))
    return np.asarray(rotation, dtype=np.float32)


def build_body_local_vertices(
    vertices_world,
    *,
    root_orient=None,
    translation=None,
    scale: float = 1.0,
) -> np.ndarray:
    vertices = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    if vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    local = vertices.astype(np.float32, copy=True)
    safe_scale = float(scale) if np.isfinite(scale) and abs(float(scale)) > 1e-6 else 1.0
    if safe_scale != 1.0:
        local = local / safe_scale

    if translation is not None:
        local = local - np.asarray(translation, dtype=np.float32).reshape(1, 3)

    if root_orient is not None:
        rotation = axis_angle_to_rotation_matrix(root_orient)
        local = local @ rotation

    return local.astype(np.float32)


def build_body_local_vertices_from_pose_params(vertices_world, pose_params: dict[str, Any] | None) -> np.ndarray:
    if not pose_params:
        return build_body_local_vertices(vertices_world)

    if "Rh" in pose_params:
        root_orient = np.asarray(pose_params["Rh"], dtype=np.float32).reshape(-1, 3)[0]
        translation = np.asarray(pose_params.get("Th", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1, 3)[0]
        scale = 1.0
    elif "fullpose" in pose_params:
        root_orient = np.asarray(pose_params["fullpose"], dtype=np.float32).reshape(-1, 3)[0]
        translation = np.asarray(
            pose_params.get("transl", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)[0]
        scale = float(pose_params.get("scale", 1.0) or 1.0)
    else:
        root_orient = None
        translation = None
        scale = 1.0

    return build_body_local_vertices(
        vertices_world,
        root_orient=root_orient,
        translation=translation,
        scale=scale,
    )


def apply_pose_aligned_prior_noise(
    vertices_world,
    body_local_vertices,
    *,
    rotation_deg_std: float = 0.0,
    translation_scale_std: float = 0.0,
    scale_std: float = 0.0,
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    world_points = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    body_local = np.asarray(body_local_vertices, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0 or body_local.size == 0 or len(world_points) != len(body_local):
        return world_points.astype(np.float32), body_local.astype(np.float32)

    rotation_deg_std = float(max(0.0, rotation_deg_std))
    translation_scale_std = float(max(0.0, translation_scale_std))
    scale_std = float(max(0.0, scale_std))
    if rotation_deg_std <= 0.0 and translation_scale_std <= 0.0 and scale_std <= 0.0:
        return world_points.astype(np.float32, copy=True), body_local.astype(np.float32, copy=True)

    if rng is None:
        rng = np.random.default_rng()

    rotvec = np.deg2rad(rng.normal(loc=0.0, scale=rotation_deg_std, size=3).astype(np.float32))
    rotation = axis_angle_to_rotation_matrix(rotvec)

    body_scale = _safe_body_scale(body_local, eps=eps)
    translation = rng.normal(
        loc=0.0,
        scale=body_scale * translation_scale_std,
        size=3,
    ).astype(np.float32)
    scale_jitter = float(np.clip(1.0 + rng.normal(loc=0.0, scale=scale_std), 0.85, 1.15))

    world_center = world_points.mean(axis=0, keepdims=True).astype(np.float32)
    noisy_world_points = ((world_points - world_center) * scale_jitter) @ rotation.T + world_center + translation.reshape(1, 3)
    noisy_body_local = (body_local * scale_jitter) @ rotation.T
    return noisy_world_points.astype(np.float32), noisy_body_local.astype(np.float32)


def normalize_body_local_vertices(
    body_local_vertices,
    *,
    clip_value: float = 6.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    vertices = np.asarray(body_local_vertices, dtype=np.float32).reshape(-1, 3)
    if vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), 1.0

    center = vertices.mean(axis=0, keepdims=True)
    scale = _safe_body_scale(vertices, eps=eps)
    local_rel = np.clip((vertices - center) / scale, -clip_value, clip_value).astype(np.float32)
    return local_rel, float(scale)


def build_human_summary_tokens(
    body_local_vertices,
    *,
    clip_value: float = 6.0,
    eps: float = 1e-6,
) -> np.ndarray:
    aux = build_vertex_identity_condition_features(body_local_vertices, clip_value=clip_value, eps=eps)
    local_rel = np.asarray(aux["local_rel"], dtype=np.float32)
    if local_rel.size == 0:
        return np.zeros((len(DEFAULT_SUMMARY_BIN_NAMES), len(DEFAULT_SUMMARY_FEATURE_NAMES)), dtype=np.float32)

    body_part_embedding = np.asarray(aux["body_part_embedding"], dtype=np.float32)
    skinning_embedding = np.asarray(aux["skinning_embedding"], dtype=np.float32)
    y = local_rel[:, 1]
    y_lo, y_hi = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0])
    radius = np.linalg.norm(local_rel, axis=1).astype(np.float32)
    tokens = []
    for height_name in ("lower", "mid", "upper"):
        if height_name == "lower":
            height_mask = y <= y_lo
        elif height_name == "mid":
            height_mask = (y > y_lo) & (y <= y_hi)
        else:
            height_mask = y > y_hi

        for lateral_name in ("left", "right"):
            lateral_mask = local_rel[:, 0] <= 0.0 if lateral_name == "left" else local_rel[:, 0] > 0.0
            for depth_name in ("back", "front"):
                depth_mask = local_rel[:, 2] <= 0.0 if depth_name == "back" else local_rel[:, 2] > 0.0
                mask = height_mask & lateral_mask & depth_mask
                if not np.any(mask):
                    tokens.append(np.zeros(len(DEFAULT_SUMMARY_FEATURE_NAMES), dtype=np.float32))
                    continue

                selected = local_rel[mask]
                token = np.concatenate(
                    [
                        selected.mean(axis=0),
                        selected.std(axis=0),
                        np.asarray(
                            [
                                float(radius[mask].mean()),
                                float(mask.mean()),
                            ],
                            dtype=np.float32,
                        ),
                        body_part_embedding[mask].mean(axis=0),
                        skinning_embedding[mask].mean(axis=0),
                    ],
                    axis=0,
                )
                tokens.append(token.astype(np.float32))

    return np.stack(tokens, axis=0).astype(np.float32)


def build_pose_aligned_surface_feature_maps(
    vertices_world,
    body_local_vertices,
    extrinsic,
    intrinsic,
    target_hw,
    *,
    point_radius_px: int = 2,
    close_px: int = 4,
    gaussian_sigma: float = 2.5,
    clip_value: float = 6.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hh, ww = int(target_hw[0]), int(target_hw[1])
    empty_mask = np.zeros((hh, ww), dtype=bool)
    empty_density = np.zeros((hh, ww), dtype=np.float32)
    empty_visibility = np.zeros((hh, ww), dtype=np.float32)
    empty_surface = np.zeros((len(DEFAULT_SURFACE_FEATURE_NAMES), hh, ww), dtype=np.float32)

    world_points = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    body_local = np.asarray(body_local_vertices, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0 or body_local.size == 0 or len(world_points) != len(body_local):
        return empty_mask, empty_density, empty_surface, empty_visibility

    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32).reshape(1, 3)
    cam_points = world_points @ rotation.T + translation

    finite_mask = (
        np.isfinite(world_points).all(axis=1)
        & np.isfinite(body_local).all(axis=1)
        & np.isfinite(cam_points).all(axis=1)
    )
    world_points = world_points[finite_mask]
    body_local = body_local[finite_mask]
    cam_points = cam_points[finite_mask]
    if world_points.size == 0:
        return empty_mask, empty_density, empty_surface, empty_visibility

    valid_depth = cam_points[:, 2] > float(eps)
    world_points = world_points[valid_depth]
    body_local = body_local[valid_depth]
    cam_points = cam_points[valid_depth]
    if world_points.size == 0:
        return empty_mask, empty_density, empty_surface, empty_visibility

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    depth = np.clip(cam_points[:, 2], float(eps), None)
    u = fx * cam_points[:, 0] / depth + cx
    v = fy * cam_points[:, 1] / depth + cy
    in_view = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u <= float(max(ww - 1, 0)))
        & (v >= 0.0)
        & (v <= float(max(hh - 1, 0)))
    )
    if not np.any(in_view):
        return empty_mask, empty_density, empty_surface, empty_visibility

    world_points = world_points[in_view]
    body_local = body_local[in_view]
    cam_points = cam_points[in_view]
    depth = depth[in_view].astype(np.float32)
    xx = np.rint(u[in_view]).astype(np.int32)
    yy = np.rint(v[in_view]).astype(np.int32)

    aux = build_vertex_identity_condition_features(body_local, clip_value=clip_value, eps=eps)
    local_rel = np.asarray(aux["local_rel"], dtype=np.float32)
    body_scale = float(aux["body_scale"])
    vertex_id_embedding = np.asarray(aux["vertex_id_embedding"], dtype=np.float32)
    body_part_embedding = np.asarray(aux["body_part_embedding"], dtype=np.float32)
    skinning_embedding = np.asarray(aux["skinning_embedding"], dtype=np.float32)
    cam_rel = np.clip(cam_points / max(body_scale, eps), -clip_value, clip_value).astype(np.float32)
    uv = np.stack(
        [
            (2.0 * u[in_view] / float(max(ww - 1, 1))) - 1.0,
            (2.0 * v[in_view] / float(max(hh - 1, 1))) - 1.0,
        ],
        axis=1,
    ).astype(np.float32)
    radius = np.clip(np.linalg.norm(local_rel, axis=1, keepdims=True), 0.0, clip_value).astype(np.float32)
    per_vertex_features = np.concatenate(
        [
            local_rel,
            cam_rel,
            uv,
            radius,
            vertex_id_embedding,
            body_part_embedding,
            skinning_embedding,
        ],
        axis=1,
    ).astype(np.float32)

    visible_pixel_owner = {}
    for point_idx in np.argsort(depth, kind="stable"):
        x = int(xx[point_idx])
        y = int(yy[point_idx])
        depth_value = float(depth[point_idx])
        key = (y, x)
        prev = visible_pixel_owner.get(key)
        if prev is not None and depth_value >= prev[0]:
            continue
        visible_pixel_owner[key] = (depth_value, point_idx)

    if not visible_pixel_owner:
        return empty_mask, empty_density, empty_surface, empty_visibility

    visible_indices = np.asarray([item[1] for item in visible_pixel_owner.values()], dtype=np.int64)
    vis_x = xx[visible_indices]
    vis_y = yy[visible_indices]
    visible_occ = np.zeros((hh, ww), dtype=np.float32)
    visible_mask = np.zeros((hh, ww), dtype=bool)
    visible_mask[vis_y, vis_x] = True

    feature_sum = np.zeros((per_vertex_features.shape[1], hh, ww), dtype=np.float32)
    np.add.at(visible_occ, (vis_y, vis_x), 1.0)
    for channel_idx in range(per_vertex_features.shape[1]):
        np.add.at(feature_sum[channel_idx], (vis_y, vis_x), per_vertex_features[visible_indices, channel_idx])

    if gaussian_sigma > 0.0 and np.any(visible_occ > 0.0):
        density_blur = cv2.GaussianBlur(visible_occ, (0, 0), sigmaX=float(gaussian_sigma), sigmaY=float(gaussian_sigma))
        feature_sum_blur = np.stack(
            [
                cv2.GaussianBlur(channel, (0, 0), sigmaX=float(gaussian_sigma), sigmaY=float(gaussian_sigma))
                for channel in feature_sum
            ],
            axis=0,
        )
    else:
        density_blur = visible_occ
        feature_sum_blur = feature_sum

    surface_mask = binary_dilate_mask(visible_mask, point_radius_px)
    surface_mask = binary_close_mask(surface_mask, close_px)

    density_max = float(density_blur.max()) if density_blur.size > 0 else 0.0
    density_map = density_blur / density_max if density_max > 0.0 else density_blur
    density_map = np.maximum(density_map.astype(np.float32), surface_mask.astype(np.float32))
    visibility_map = np.maximum(visible_mask.astype(np.float32), surface_mask.astype(np.float32))

    feature_maps = np.zeros((len(DEFAULT_SURFACE_FEATURE_NAMES), hh, ww), dtype=np.float32)
    feature_maps[0] = density_map.astype(np.float32)
    feature_maps[1] = visibility_map.astype(np.float32)
    safe_density = np.where(density_blur > 1e-6, density_blur, 1.0).astype(np.float32)
    for channel_idx in range(feature_sum_blur.shape[0]):
        channel = feature_sum_blur[channel_idx] / safe_density
        channel = np.where(surface_mask, channel, 0.0).astype(np.float32)
        feature_maps[channel_idx + 2] = channel

    return (
        surface_mask.astype(bool),
        density_map.astype(np.float32),
        feature_maps.astype(np.float32),
        visibility_map.astype(np.float32),
    )


def project_vertices_to_feature_maps(
    vertices_world,
    extrinsic,
    intrinsic,
    target_hw,
    *,
    point_radius_px: int = 2,
    close_px: int = 4,
    gaussian_sigma: float = 2.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hh, ww = int(target_hw[0]), int(target_hw[1])
    empty_mask = np.zeros((hh, ww), dtype=bool)
    empty_density = np.zeros((hh, ww), dtype=np.float32)
    empty_vertex_features = np.zeros((len(DEFAULT_VERTEX_FEATURE_NAMES), hh, ww), dtype=np.float32)
    empty_depth_map = np.zeros((hh, ww), dtype=np.float32)
    empty_world_points = np.zeros((hh, ww, 3), dtype=np.float32)
    if vertices_world is None:
        return (
            empty_mask,
            empty_density,
            empty_vertex_features,
            empty_mask,
            empty_depth_map,
            empty_world_points,
        )

    world_points = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0:
        return (
            empty_mask,
            empty_density,
            empty_vertex_features,
            empty_mask,
            empty_depth_map,
            empty_world_points,
        )

    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32).reshape(1, 3)
    cam_points = world_points @ rotation.T + translation
    finite_mask = np.isfinite(cam_points).all(axis=1) & np.isfinite(world_points).all(axis=1)
    cam_points = cam_points[finite_mask]
    world_points = world_points[finite_mask]
    if cam_points.size == 0:
        return (
            empty_mask,
            empty_density,
            empty_vertex_features,
            empty_mask,
            empty_depth_map,
            empty_world_points,
        )

    valid_depth = cam_points[:, 2] > float(eps)
    cam_points = cam_points[valid_depth]
    world_points = world_points[valid_depth]
    if cam_points.size == 0:
        return (
            empty_mask,
            empty_density,
            empty_vertex_features,
            empty_mask,
            empty_depth_map,
            empty_world_points,
        )

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = np.clip(cam_points[:, 2], float(eps), None)
    u = fx * cam_points[:, 0] / z + cx
    v = fy * cam_points[:, 1] / z + cy
    in_view = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u <= float(max(ww - 1, 0)))
        & (v >= 0.0)
        & (v <= float(max(hh - 1, 0)))
    )
    if not np.any(in_view):
        return (
            empty_mask,
            empty_density,
            empty_vertex_features,
            empty_mask,
            empty_depth_map,
            empty_world_points,
        )

    xx = np.rint(u[in_view]).astype(np.int32)
    yy = np.rint(v[in_view]).astype(np.int32)
    cam_points = cam_points[in_view]
    world_points = world_points[in_view]
    depth_values = z[in_view].astype(np.float32)
    vertex_features = build_vertex_geometry_features(world_points, cam_points, eps=eps)

    raw_density = np.zeros((hh, ww), dtype=np.float32)
    feature_sum = np.zeros((vertex_features.shape[1], hh, ww), dtype=np.float32)
    np.add.at(raw_density, (yy, xx), 1.0)
    for channel_idx in range(vertex_features.shape[1]):
        np.add.at(feature_sum[channel_idx], (yy, xx), vertex_features[:, channel_idx])

    if gaussian_sigma > 0.0 and np.any(raw_density > 0.0):
        density_blur = cv2.GaussianBlur(raw_density, (0, 0), sigmaX=float(gaussian_sigma), sigmaY=float(gaussian_sigma))
        feature_sum_blur = np.stack(
            [
                cv2.GaussianBlur(channel, (0, 0), sigmaX=float(gaussian_sigma), sigmaY=float(gaussian_sigma))
                for channel in feature_sum
            ],
            axis=0,
        )
    else:
        density_blur = raw_density
        feature_sum_blur = feature_sum

    prior_mask = raw_density > 0.0
    prior_mask = binary_dilate_mask(prior_mask, point_radius_px)
    prior_mask = binary_close_mask(prior_mask, close_px)

    density_max = float(density_blur.max()) if density_blur.size > 0 else 0.0
    if density_max > 0.0:
        density_map = density_blur / density_max
    else:
        density_map = density_blur
    density_map = np.maximum(density_map.astype(np.float32), prior_mask.astype(np.float32))

    feature_maps = np.zeros((len(DEFAULT_VERTEX_FEATURE_NAMES), hh, ww), dtype=np.float32)
    feature_maps[0] = density_map.astype(np.float32)
    safe_density = np.where(density_blur > 1e-6, density_blur, 1.0).astype(np.float32)
    for channel_idx in range(feature_sum_blur.shape[0]):
        channel = feature_sum_blur[channel_idx] / safe_density
        channel = np.where(prior_mask, channel, 0.0).astype(np.float32)
        feature_maps[channel_idx + 1] = channel

    sparse_depth_map = np.full((hh, ww), np.inf, dtype=np.float32)
    sparse_world_points = np.zeros((hh, ww, 3), dtype=np.float32)
    sparse_mask = np.zeros((hh, ww), dtype=bool)
    for point_idx in np.argsort(depth_values, kind="stable"):
        x = int(xx[point_idx])
        y = int(yy[point_idx])
        depth_value = float(depth_values[point_idx])
        if sparse_mask[y, x] and depth_value >= float(sparse_depth_map[y, x]):
            continue
        sparse_mask[y, x] = True
        sparse_depth_map[y, x] = depth_value
        sparse_world_points[y, x] = world_points[point_idx]

    sparse_depth_map = np.where(sparse_mask, sparse_depth_map, 0.0).astype(np.float32)
    return (
        prior_mask.astype(bool),
        density_map.astype(np.float32),
        feature_maps.astype(np.float32),
        sparse_mask.astype(bool),
        sparse_depth_map,
        sparse_world_points.astype(np.float32),
    )


def preprocess_feature_map(
    feature_map,
    *,
    mode: str = "pad",
    target_size: int = 518,
    interpolation: str = "bilinear",
    pad_value: float = 0.0,
) -> np.ndarray:
    if mode not in ("crop", "pad"):
        raise ValueError("mode must be either 'crop' or 'pad'")

    array = np.asarray(feature_map)
    squeeze_channel = array.ndim == 2
    if squeeze_channel:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected feature_map with 2 or 3 dims, got shape {array.shape}")

    channels, height, width = array.shape
    target_size = int(target_size)
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)
    align_corners = interpolation in {"linear", "bilinear", "bicubic", "trilinear"}
    tensor = F.interpolate(
        tensor,
        size=(int(new_height), int(new_width)),
        mode=interpolation,
        align_corners=False if align_corners else None,
    )

    if mode == "crop":
        if int(new_height) > target_size:
            start_y = (int(new_height) - target_size) // 2
            tensor = tensor[:, :, start_y : start_y + target_size, :]
    else:
        canvas = torch.full(
            (1, channels, target_size, target_size),
            float(pad_value),
            dtype=tensor.dtype,
        )
        pad_top = (target_size - int(new_height)) // 2
        pad_left = (target_size - int(new_width)) // 2
        canvas[:, :, pad_top : pad_top + int(new_height), pad_left : pad_left + int(new_width)] = tensor
        tensor = canvas

    output = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    if squeeze_channel:
        return output[0]
    return output


def load_4k4d_smplx_frame(annotations_smc: str | Path, frame_id: int) -> dict[str, Any]:
    import h5py

    frame_idx = int(frame_id)
    with h5py.File(str(annotations_smc), "r") as handle:
        smplx_group = handle["SMPLx"]
        payload = {
            "betas": np.asarray(smplx_group["betas"][frame_idx], dtype=np.float32),
            "expression": np.asarray(smplx_group["expression"][frame_idx], dtype=np.float32),
            "fullpose": np.asarray(smplx_group["fullpose"][frame_idx], dtype=np.float32),
            "transl": np.asarray(smplx_group["transl"][frame_idx], dtype=np.float32),
            "scale": float(smplx_group["scale"][()]),
        }
    return payload


def load_zju_smpl_params(params_path: str | Path) -> dict[str, Any]:
    payload = np.load(str(params_path), allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected SMPL params dict in {params_path}, got {type(payload)!r}")
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload.items()
    }


@lru_cache(maxsize=4)
def _load_smplx_model(model_root: str, gender: str = "neutral"):
    import smplx

    resolved_path = Path(model_root).expanduser().resolve()
    if (resolved_path / "SMPLX_NEUTRAL.npz").is_file():
        resolved_path = resolved_path.parent
    resolved_root = str(resolved_path)
    model = smplx.create(
        resolved_root,
        model_type="smplx",
        gender=str(gender).lower(),
        use_pca=False,
        ext="npz",
        num_expression_coeffs=10,
    )
    model.eval()
    return model


def build_4k4d_smplx_vertices(
    model_root: str | Path,
    smplx_params: dict[str, Any],
    *,
    gender: str = "neutral",
    device: str = "cpu",
) -> np.ndarray:
    model = _load_smplx_model(str(model_root), gender=gender).to(device)
    fullpose = np.asarray(smplx_params["fullpose"], dtype=np.float32)
    output = model(
        betas=torch.tensor(np.asarray(smplx_params["betas"], dtype=np.float32)[None], dtype=torch.float32, device=device),
        expression=torch.tensor(
            np.asarray(smplx_params["expression"], dtype=np.float32)[None], dtype=torch.float32, device=device
        ),
        global_orient=torch.tensor(fullpose[None, 0], dtype=torch.float32, device=device),
        body_pose=torch.tensor(fullpose[None, 1:22].reshape(1, -1), dtype=torch.float32, device=device),
        jaw_pose=torch.tensor(fullpose[None, 22], dtype=torch.float32, device=device),
        leye_pose=torch.tensor(fullpose[None, 23], dtype=torch.float32, device=device),
        reye_pose=torch.tensor(fullpose[None, 24], dtype=torch.float32, device=device),
        left_hand_pose=torch.tensor(fullpose[None, 25:40].reshape(1, -1), dtype=torch.float32, device=device),
        right_hand_pose=torch.tensor(fullpose[None, 40:55].reshape(1, -1), dtype=torch.float32, device=device),
        transl=torch.tensor(np.asarray(smplx_params["transl"], dtype=np.float32)[None], dtype=torch.float32, device=device),
    )
    vertices_world = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    scale = float(smplx_params.get("scale", 1.0) or 1.0)
    if scale != 1.0:
        vertices_world = vertices_world * scale
    return vertices_world


def world_to_camera_extrinsic_from_4k4d(rt_matrix) -> np.ndarray:
    rt = np.asarray(rt_matrix, dtype=np.float32)
    if rt.shape == (4, 4):
        world_to_camera = np.linalg.inv(rt)[:3]
    elif rt.shape == (3, 4):
        full = np.eye(4, dtype=np.float32)
        full[:3] = rt
        world_to_camera = np.linalg.inv(full)[:3]
    else:
        raise ValueError(f"Unsupported RT shape: {rt.shape}")
    return world_to_camera.astype(np.float32)
