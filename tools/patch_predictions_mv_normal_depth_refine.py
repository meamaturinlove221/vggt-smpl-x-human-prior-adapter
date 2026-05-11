from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import (  # noqa: E402
    face_box_from_mask,
    head_box_from_mask,
    normal_to_rgb,
    point_map_to_normal_numpy,
    points_world_to_camera,
    preprocess_mask_image,
    preprocess_rgb_image,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch VGGT predictions with a self-consensus normal/depth/point residual gate. "
            "This is a post-inference normal-line probe, not a training pass."
        )
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-kind", choices=("head", "face", "face_core", "head_face", "full"), default="head_face")
    parser.add_argument("--view-indices", default="all")
    parser.add_argument("--source-views", default="all")
    parser.add_argument("--teacher-source", choices=("self_consensus",), default="self_consensus")
    parser.add_argument("--agreement-angle-deg", type=float, default=12.0)
    parser.add_argument("--max-depth-delta", type=float, default=0.025)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--normal-weight", type=float, default=0.35)
    parser.add_argument("--depth-point-weight", type=float, default=0.20)
    parser.add_argument("--xview-depth-weight", type=float, default=0.25)
    parser.add_argument("--smooth-weight", type=float, default=0.03)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--source-depth-tolerance", type=float, default=0.035)
    parser.add_argument("--mask-erode-pixels", type=int, default=1)
    parser.add_argument("--min-consensus-pixels", type=int, default=64)
    parser.add_argument("--copy-sidecar-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_indices(text: str | None, count: int, *, exclude: int | None = None) -> list[int]:
    if text is None or text.strip().lower() == "all":
        values = list(range(count))
    else:
        values = []
        for piece in text.split(","):
            piece = piece.strip()
            if not piece:
                continue
            idx = int(piece)
            if idx < 0:
                idx = count + idx
            if idx < 0 or idx >= count:
                raise IndexError(f"view index {piece} outside [0,{count})")
            values.append(idx)
    if exclude is not None:
        values = [idx for idx in values if idx != exclude]
    return sorted(set(values))


def _read_manifest(scene_dir: Path) -> dict[str, Any] | None:
    for name in ("scene_manifest.json", "scene_manifest"):
        path = scene_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _resolve_scene_path(scene_dir: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = scene_dir / path
    if candidate.exists():
        return candidate
    return path


def _sorted_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _scene_paths(scene_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest = _read_manifest(scene_dir)
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    if manifest:
        for record in manifest.get("exported_views", []):
            image_raw = record.get("image_path") or record.get("rgb_path") or record.get("image")
            mask_raw = record.get("mask_path") or record.get("mask")
            if image_raw:
                image_paths.append(_resolve_scene_path(scene_dir, image_raw))
            if mask_raw:
                mask_paths.append(_resolve_scene_path(scene_dir, mask_raw))
    if not image_paths:
        image_paths = _sorted_files(scene_dir / "images")
    if not mask_paths:
        mask_paths = _sorted_files(scene_dir / "masks")
    if not image_paths:
        raise FileNotFoundError(f"No images found in {scene_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {scene_dir}")
    return image_paths, mask_paths


def _load_scene(scene_dir: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if height != width:
        raise ValueError("This probe currently expects square VGGT inputs")
    image_paths, mask_paths = _scene_paths(scene_dir)
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}")
    images = [preprocess_rgb_image(path, height) for path in image_paths]
    masks = [preprocess_mask_image(path, height) for path in mask_paths]
    return np.stack(images), np.stack(masks).astype(bool), [path.name for path in image_paths]


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(pixels))):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        output = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return output


def _box_mask(mask: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=bool)
    if box is None:
        return output
    x0, y0, x1, y1 = [int(value) for value in box]
    output[y0:y1, x0:x1] = True
    return output & mask


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if roi_kind == "full":
        return mask
    if roi_kind == "head":
        return _box_mask(mask, head_box_from_mask(mask))
    if roi_kind == "face":
        return _box_mask(mask, face_box_from_mask(mask))
    if roi_kind == "head_face":
        return _box_mask(mask, head_box_from_mask(mask)) | _box_mask(mask, face_box_from_mask(mask))
    if roi_kind == "face_core":
        face = face_box_from_mask(mask)
        if face is None:
            return np.zeros(mask.shape, dtype=bool)
        x0, y0, x1, y1 = face
        width = x1 - x0
        height = y1 - y0
        core = (
            x0 + int(round(width * 0.15)),
            y0 + int(round(height * 0.12)),
            x1 - int(round(width * 0.15)),
            y1 - int(round(height * 0.10)),
        )
        return _box_mask(mask, core)
    raise ValueError(roi_kind)


def _normalize(vectors: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (norms > eps)
    output = np.zeros_like(values, dtype=np.float32)
    output[valid] = values[valid] / norms[valid, None]
    return output, valid


def _depth_to_camera_points_np(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return np.stack(((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), axis=-1).astype(np.float32)


def _unproject_depth_to_world(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    worlds = []
    for view_idx in range(depth.shape[0]):
        cam = _depth_to_camera_points_np(depth[view_idx], intrinsics[view_idx])
        rotation = extrinsics[view_idx, :, :3].astype(np.float32)
        translation = extrinsics[view_idx, :, 3].astype(np.float32)
        world = np.einsum("hwc,cr->hwr", cam - translation, rotation)
        worlds.append(world.astype(np.float32))
    return np.stack(worlds, axis=0)


def _build_self_consensus(
    pred_normal: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_points: np.ndarray,
    extrinsic: np.ndarray,
    valid_mask: np.ndarray,
    roi: np.ndarray,
    agreement_cos: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred, pred_valid = _normalize(pred_normal)
    depth_points = _depth_to_camera_points_np(depth, intrinsic)
    depth_normal, depth_valid = point_map_to_normal_numpy(depth_points, valid_mask & np.isfinite(depth) & (depth > 0))
    depth_normal, depth_vec_valid = _normalize(depth_normal)
    depth_valid &= depth_vec_valid
    point_camera = points_world_to_camera(world_points, extrinsic)
    point_normal, point_valid = point_map_to_normal_numpy(point_camera, valid_mask & np.isfinite(point_camera).all(axis=-1))
    point_normal, point_vec_valid = _normalize(point_normal)
    point_valid &= point_vec_valid
    base_valid = roi & pred_valid & depth_valid & point_valid
    pred_depth = np.sum(pred * depth_normal, axis=-1)
    pred_point = np.sum(pred * point_normal, axis=-1)
    depth_point = np.sum(depth_normal * point_normal, axis=-1)
    consensus = base_valid & (np.abs(pred_depth) >= agreement_cos) & (np.abs(pred_point) >= agreement_cos) & (np.abs(depth_point) >= agreement_cos)
    aligned_depth = depth_normal.copy()
    aligned_point = point_normal.copy()
    aligned_depth[pred_depth < 0] *= -1.0
    aligned_point[pred_point < 0] *= -1.0
    teacher, teacher_valid = _normalize(pred + aligned_depth + aligned_point)
    teacher[~consensus] = 0.0
    summary = {
        "roi_pixels": int(roi.sum()),
        "base_valid_pixels": int(base_valid.sum()),
        "consensus_pixels": int(consensus.sum()),
        "consensus_ratio_of_roi": float(consensus.sum() / max(int(roi.sum()), 1)),
        "pred_depth_abs_cos_median": float(np.median(np.abs(pred_depth[base_valid]))) if np.any(base_valid) else None,
        "pred_point_abs_cos_median": float(np.median(np.abs(pred_point[base_valid]))) if np.any(base_valid) else None,
        "depth_point_abs_cos_median": float(np.median(np.abs(depth_point[base_valid]))) if np.any(base_valid) else None,
    }
    return teacher.astype(np.float32), consensus & teacher_valid, summary


def _crop_bounds(mask: np.ndarray, pad: int = 4) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return None
    height, width = mask.shape
    return (
        max(0, int(xx.min()) - pad),
        max(0, int(yy.min()) - pad),
        min(width, int(xx.max()) + 1 + pad),
        min(height, int(yy.max()) + 1 + pad),
    )


def _depth_to_camera_points_torch(depth: torch.Tensor, intrinsic: np.ndarray, x0: int, y0: int) -> torch.Tensor:
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(y0, y0 + height, dtype=depth.dtype, device=depth.device),
        torch.arange(x0, x0 + width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return torch.stack(((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), dim=-1)


def _point_normals_torch(points: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded_mask = F.pad(mask.bool()[None], (1, 1, 1, 1), mode="constant", value=False)[0]
    padded_pts = F.pad(points.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="constant", value=0.0)[0].permute(1, 2, 0)
    center = padded_pts[1:-1, 1:-1]
    up = padded_pts[:-2, 1:-1]
    left = padded_pts[1:-1, :-2]
    down = padded_pts[2:, 1:-1]
    right = padded_pts[1:-1, 2:]
    up_dir = up - center
    left_dir = left - center
    down_dir = down - center
    right_dir = right - center
    normals = torch.stack(
        (
            torch.cross(up_dir, left_dir, dim=-1),
            torch.cross(left_dir, down_dir, dim=-1),
            torch.cross(down_dir, right_dir, dim=-1),
            torch.cross(right_dir, up_dir, dim=-1),
        ),
        dim=0,
    )
    valids = torch.stack(
        (
            padded_mask[:-2, 1:-1] & padded_mask[1:-1, 1:-1] & padded_mask[1:-1, :-2],
            padded_mask[1:-1, :-2] & padded_mask[1:-1, 1:-1] & padded_mask[2:, 1:-1],
            padded_mask[2:, 1:-1] & padded_mask[1:-1, 1:-1] & padded_mask[1:-1, 2:],
            padded_mask[1:-1, 2:] & padded_mask[1:-1, 1:-1] & padded_mask[:-2, 1:-1],
        ),
        dim=0,
    )
    weights = valids.float()[..., None]
    collapsed = (F.normalize(normals, p=2, dim=-1, eps=1e-6) * weights).sum(dim=0) / torch.clamp(weights.sum(dim=0), min=1e-6)
    collapsed = F.normalize(collapsed, p=2, dim=-1, eps=1e-6)
    valid = weights.sum(dim=0)[..., 0] > 0
    return collapsed, valid


def _world_to_camera_torch(points_world: torch.Tensor, extrinsic: np.ndarray) -> torch.Tensor:
    rotation = torch.as_tensor(extrinsic[:, :3], dtype=points_world.dtype, device=points_world.device)
    translation = torch.as_tensor(extrinsic[:, 3], dtype=points_world.dtype, device=points_world.device)
    return torch.einsum("hwc,rc->hwr", points_world, rotation) + translation


def _camera_to_world_torch(points_cam: torch.Tensor, extrinsic: np.ndarray) -> torch.Tensor:
    rotation = torch.as_tensor(extrinsic[:, :3], dtype=points_cam.dtype, device=points_cam.device)
    translation = torch.as_tensor(extrinsic[:, 3], dtype=points_cam.dtype, device=points_cam.device)
    return torch.einsum("hwc,cr->hwr", points_cam - translation, rotation)


def _xview_depth_loss(
    target_cam_points: torch.Tensor,
    target_apply: torch.Tensor,
    source_views: list[int],
    source_depths: torch.Tensor,
    source_masks: torch.Tensor,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    target_view: int,
    tolerance: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if not source_views:
        return target_cam_points.new_tensor(0.0)
    height, width = source_depths.shape[-2:]
    points_world = _camera_to_world_torch(target_cam_points, extrinsics[target_view])
    for source_view in source_views:
        src_cam = _world_to_camera_torch(points_world, extrinsics[source_view])
        z = src_cam[..., 2]
        fx = float(intrinsics[source_view, 0, 0])
        fy = float(intrinsics[source_view, 1, 1])
        cx = float(intrinsics[source_view, 0, 2])
        cy = float(intrinsics[source_view, 1, 2])
        u = fx * src_cam[..., 0] / torch.clamp(z, min=1e-6) + cx
        v = fy * src_cam[..., 1] / torch.clamp(z, min=1e-6) + cy
        grid_x = u / max(width - 1, 1) * 2.0 - 1.0
        grid_y = v / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)[None]
        sampled_depth = F.grid_sample(
            source_depths[source_view][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        sampled_mask = F.grid_sample(
            source_masks[source_view].float()[None, None],
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0] > 0.5
        valid = target_apply & sampled_mask & torch.isfinite(z) & torch.isfinite(sampled_depth) & (z > 0.05) & (sampled_depth > 0.05)
        valid = valid & ((z.detach() - sampled_depth.detach()).abs() <= float(tolerance))
        if torch.any(valid):
            losses.append(F.smooth_l1_loss(z[valid], sampled_depth[valid]))
    if not losses:
        return target_cam_points.new_tensor(0.0)
    return torch.stack(losses).mean()


def _optimize_view_depth(
    view_idx: int,
    depth: np.ndarray,
    teacher: np.ndarray,
    consensus: np.ndarray,
    roi: np.ndarray,
    world_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    all_depths: np.ndarray,
    all_masks: np.ndarray,
    source_views: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    box = _crop_bounds(consensus | roi, pad=4)
    if box is None or int(consensus.sum()) < int(args.min_consensus_pixels):
        return depth.copy(), {"status": "skipped", "reason": "too_few_consensus_pixels", "consensus_pixels": int(consensus.sum())}
    x0, y0, x1, y1 = box
    depth_crop = depth[y0:y1, x0:x1].astype(np.float32)
    teacher_crop = teacher[y0:y1, x0:x1].astype(np.float32)
    consensus_crop = consensus[y0:y1, x0:x1].astype(bool)
    roi_crop = roi[y0:y1, x0:x1].astype(bool)
    world_crop = world_points[y0:y1, x0:x1].astype(np.float32)
    depth_t = torch.from_numpy(depth_crop)
    param = torch.zeros_like(depth_t, requires_grad=True)
    teacher_t = torch.from_numpy(teacher_crop)
    consensus_t = torch.from_numpy(consensus_crop)
    roi_t = torch.from_numpy(roi_crop)
    world_t = torch.from_numpy(world_crop)
    all_depths_t = torch.from_numpy(all_depths.astype(np.float32))
    all_masks_t = torch.from_numpy(all_masks.astype(bool))
    optimizer = torch.optim.Adam([param], lr=float(args.lr))
    records = []
    for step in range(max(1, int(args.iters))):
        optimizer.zero_grad(set_to_none=True)
        delta = torch.tanh(param) * float(args.max_depth_delta)
        refined = torch.clamp(depth_t + delta, min=1e-4)
        apply_mask = consensus_t
        cam_points = _depth_to_camera_points_torch(refined, intrinsics[view_idx], x0, y0)
        depth_normals, normal_valid = _point_normals_torch(cam_points, roi_t & torch.isfinite(refined) & (refined > 0))
        normal_apply = apply_mask & normal_valid
        if torch.any(normal_apply):
            dot = torch.sum(depth_normals[normal_apply] * teacher_t[normal_apply], dim=-1).abs().clamp(0.0, 1.0)
            normal_loss = (1.0 - dot).mean()
        else:
            normal_loss = refined.new_tensor(0.0)
        point_camera = _world_to_camera_torch(world_t, extrinsics[view_idx])
        if torch.any(apply_mask):
            depth_point_loss = F.smooth_l1_loss(cam_points[apply_mask], point_camera[apply_mask])
            anchor_loss = torch.mean((delta[apply_mask] / max(float(args.max_depth_delta), 1e-6)) ** 2)
        else:
            depth_point_loss = refined.new_tensor(0.0)
            anchor_loss = refined.new_tensor(0.0)
        if torch.any(roi_t[:, 1:] & roi_t[:, :-1]):
            smooth_x = (delta[:, 1:] - delta[:, :-1]).abs()[roi_t[:, 1:] & roi_t[:, :-1]].mean()
        else:
            smooth_x = refined.new_tensor(0.0)
        if torch.any(roi_t[1:, :] & roi_t[:-1, :]):
            smooth_y = (delta[1:, :] - delta[:-1, :]).abs()[roi_t[1:, :] & roi_t[:-1, :]].mean()
        else:
            smooth_y = refined.new_tensor(0.0)
        xview_loss = _xview_depth_loss(
            cam_points,
            apply_mask,
            source_views,
            all_depths_t,
            all_masks_t,
            intrinsics,
            extrinsics,
            view_idx,
            float(args.source_depth_tolerance),
        )
        loss = (
            float(args.anchor_weight) * anchor_loss
            + float(args.normal_weight) * normal_loss
            + float(args.depth_point_weight) * depth_point_loss
            + float(args.xview_depth_weight) * xview_loss
            + float(args.smooth_weight) * (smooth_x + smooth_y)
        )
        loss.backward()
        optimizer.step()
        if step in {0, max(0, int(args.iters) // 2), max(0, int(args.iters) - 1)}:
            records.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "anchor": float(anchor_loss.detach().cpu()),
                    "normal": float(normal_loss.detach().cpu()),
                    "depth_point": float(depth_point_loss.detach().cpu()),
                    "xview_depth": float(xview_loss.detach().cpu()),
                    "smooth": float((smooth_x + smooth_y).detach().cpu()),
                }
            )
    with torch.no_grad():
        delta = torch.tanh(param) * float(args.max_depth_delta)
        refined_crop = torch.clamp(depth_t + delta, min=1e-4).cpu().numpy().astype(np.float32)
        delta_np = delta.cpu().numpy().astype(np.float32)
    updated = depth.copy()
    apply_np = consensus_crop
    updated_crop = updated[y0:y1, x0:x1]
    updated_crop[apply_np] = refined_crop[apply_np]
    updated[y0:y1, x0:x1] = updated_crop
    applied_delta = delta_np[apply_np]
    summary = {
        "status": "optimized",
        "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "roi_pixels": int(roi.sum()),
        "consensus_pixels": int(consensus.sum()),
        "applied_pixels": int(apply_np.sum()),
        "delta_abs_median": float(np.median(np.abs(applied_delta))) if applied_delta.size else 0.0,
        "delta_abs_p95": float(np.percentile(np.abs(applied_delta), 95)) if applied_delta.size else 0.0,
        "loss_records": records,
    }
    return updated.astype(np.float32), summary


def _save_preview(
    out_path: Path,
    rgb: np.ndarray,
    roi: np.ndarray,
    consensus: np.ndarray,
    teacher: np.ndarray,
    before_depth: np.ndarray,
    after_depth: np.ndarray,
    title: str,
) -> None:
    def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        base = rgb.copy().astype(np.float32)
        base[mask] = 0.55 * base[mask] + 0.45 * np.asarray(color, dtype=np.float32)
        return np.clip(base, 0, 255).astype(np.uint8)

    def depth_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = depth[mask & np.isfinite(depth)]
        if values.size:
            lo, hi = np.percentile(values, [2, 98])
            normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        else:
            normalized = np.zeros(depth.shape, dtype=np.float32)
        output = np.stack([normalized, normalized, normalized], axis=-1)
        output[~mask] = 1.0
        return (output * 255).astype(np.uint8)

    delta = after_depth - before_depth
    delta_norm = np.clip(delta / max(float(np.max(np.abs(delta[roi])) if np.any(roi) else 1e-6), 1e-6), -1, 1)
    delta_rgb = np.zeros((*delta.shape, 3), dtype=np.float32)
    delta_rgb[..., 0] = np.clip(delta_norm, 0, 1)
    delta_rgb[..., 2] = np.clip(-delta_norm, 0, 1)
    delta_rgb[..., 1] = 1 - np.abs(delta_norm)
    delta_rgb[~roi] = 1.0
    tiles = [
        ("RGB", rgb),
        ("ROI", overlay(roi, (255, 0, 0))),
        ("self-consensus", overlay(consensus, (0, 80, 255))),
        ("teacher normal", normal_to_rgb(teacher, consensus)),
        ("before depth", depth_rgb(before_depth, roi)),
        ("delta depth", (np.clip(delta_rgb, 0, 1) * 255).astype(np.uint8)),
    ]
    height, width = rgb.shape[:2]
    canvas = Image.new("RGB", (width * 3, height * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, (label, tile) in enumerate(tiles):
        x = (idx % 3) * width
        y = 44 + (idx // 3) * height
        canvas.paste(Image.fromarray(tile), (x, y))
        draw.rectangle((x, y, x + 180, y + 22), fill=(255, 255, 255))
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    for child in src_dir.iterdir():
        if child.is_file() and child.name != "predictions.npz":
            target = dst_dir / child.name
            if not target.exists():
                shutil.copy2(child, target)


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_npz)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(predictions_path, allow_pickle=False) as data:
        payload = {key: np.array(data[key]) for key in data.files}
    required = {"depth", "normal", "world_points", "intrinsic", "extrinsic"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Missing required prediction keys: {missing}")
    depth = np.asarray(payload["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    normals = np.asarray(payload["normal"], dtype=np.float32)
    pred_normals, pred_valid = _normalize(normals)
    intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
    extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
    world_points = np.asarray(payload["world_points"], dtype=np.float32)
    views, height, width = depth.shape
    images, masks, names = _load_scene(scene_dir, height, width)
    if images.shape[0] != views:
        raise ValueError(f"scene views {images.shape[0]} != prediction views {views}")
    target_views = _parse_indices(args.view_indices, views)
    agreement_cos = math.cos(math.radians(float(args.agreement_angle_deg)))
    refined_depth = depth.copy()
    teacher_stack = np.zeros_like(pred_normals, dtype=np.float32)
    consensus_stack = np.zeros((views, height, width), dtype=bool)
    roi_stack = np.zeros((views, height, width), dtype=bool)
    records: list[dict[str, Any]] = []
    for view_idx in target_views:
        valid_mask = _erode_mask(masks[view_idx], int(args.mask_erode_pixels)) & pred_valid[view_idx]
        roi = _roi_mask(valid_mask, args.roi_kind)
        teacher, consensus, consensus_summary = _build_self_consensus(
            pred_normals[view_idx],
            depth[view_idx],
            intrinsics[view_idx],
            world_points[view_idx],
            extrinsics[view_idx],
            valid_mask,
            roi,
            agreement_cos,
        )
        source_views = _parse_indices(args.source_views, views, exclude=view_idx)
        updated_depth, opt_summary = _optimize_view_depth(
            view_idx,
            depth[view_idx],
            teacher,
            consensus,
            roi,
            world_points[view_idx],
            intrinsics,
            extrinsics,
            depth,
            masks,
            source_views,
            args,
        )
        refined_depth[view_idx] = updated_depth
        teacher_stack[view_idx] = teacher
        consensus_stack[view_idx] = consensus
        roi_stack[view_idx] = roi
        record = {
            "view": int(view_idx),
            "image": names[view_idx],
            "source_views": [int(idx) for idx in source_views],
            "roi_kind": args.roi_kind,
            "agreement_angle_deg": float(args.agreement_angle_deg),
            "self_consensus": consensus_summary,
            "optimization": opt_summary,
        }
        records.append(record)
        _save_preview(
            output_dir / f"view{view_idx:02d}_mv_normal_depth_preview.png",
            images[view_idx],
            roi,
            consensus,
            teacher,
            depth[view_idx],
            updated_depth,
            f"r9 self-consensus view {view_idx}; consensus={int(consensus.sum())}",
        )
    payload["depth"] = refined_depth[..., None].astype(np.float32)
    payload["world_points"] = _unproject_depth_to_world(refined_depth, intrinsics, extrinsics).astype(np.float32)
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    np.savez_compressed(
        output_dir / "self_consensus_teacher_normals.npz",
        teacher_normal=teacher_stack.astype(np.float32),
        consensus_mask=consensus_stack.astype(bool),
        roi_mask=roi_stack.astype(bool),
    )
    if args.copy_sidecar_files:
        _copy_sidecars(predictions_path.parent, output_dir)
    summary = {
        "predictions_npz": str(predictions_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "views": int(views),
        "view_indices": [int(idx) for idx in target_views],
        "roi_kind": args.roi_kind,
        "teacher_source": args.teacher_source,
        "agreement_angle_deg": float(args.agreement_angle_deg),
        "max_depth_delta": float(args.max_depth_delta),
        "weights": {
            "anchor": float(args.anchor_weight),
            "normal": float(args.normal_weight),
            "depth_point": float(args.depth_point_weight),
            "xview_depth": float(args.xview_depth_weight),
            "smooth": float(args.smooth_weight),
        },
        "records": records,
        "truthful_gate": (
            "This tool only creates a candidate. It is pass only if same-protocol Open3D face/head/full visuals "
            "and ROI metrics improve without ghost shell, holes, or full-body breakage."
        ),
    }
    (output_dir / "mv_normal_depth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
