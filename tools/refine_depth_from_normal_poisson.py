"""Refine predicted depth with the predicted normal field.

This is a post-inference normal-line probe.  It keeps the VGGT prediction as the
data term, solves a small ROI-local log-depth Poisson problem from camera-space
normal gradients, then recomputes depth-unprojected world points.  The tool is
intentionally conservative: it writes a new predictions.npz and never mutates
the source prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import lsqr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask


def preprocess_mask(mask_path: Path, target_size: int) -> np.ndarray:
    image = Image.open(mask_path).convert("L")
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = array
    return canvas > 0


def load_mask_stack(scene_dir: Path, target_size: int) -> np.ndarray | None:
    mask_dir = scene_dir / "masks"
    if not mask_dir.is_dir():
        return None
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    if not mask_paths:
        return None
    return np.stack([preprocess_mask(path, target_size) for path in mask_paths], axis=0)


def mask_to_roi(mask: np.ndarray, roi: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if roi == "full":
        return mask
    box = head_box_from_mask(mask) if roi == "head" else face_box_from_mask(mask)
    output = np.zeros_like(mask, dtype=bool)
    if box is None:
        return output
    x0, y0, x1, y1 = box
    height, width = mask.shape
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x1 > x0 and y1 > y0:
        output[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return output


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_to_world(depth_map: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    if depth_map.ndim == 4:
        depth_values = depth_map[..., 0]
    else:
        depth_values = depth_map
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics.astype(np.float32))
    world_points = []
    for frame_idx in range(depth_values.shape[0]):
        depth = depth_values[frame_idx].astype(np.float32)
        intrinsic = intrinsics[frame_idx].astype(np.float32)
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        fx = max(float(intrinsic[0, 0]), 1e-6)
        fy = max(float(intrinsic[1, 1]), 1e-6)
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        cam_points = np.stack([x_cam, y_cam, depth], axis=-1)
        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world_points.append((cam_points @ rotation.T + translation).astype(np.float32))
    return np.stack(world_points, axis=0)


def depth_to_camera_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return np.stack(
        [
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        ],
        axis=-1,
    ).astype(np.float32)


def central_depth_normals(depth: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = depth_to_camera_points(depth, intrinsic)
    dx = points[:, 2:, :] - points[:, :-2, :]
    dy = points[2:, :, :] - points[:-2, :, :]
    dx_center = dx[1:-1, :, :]
    dy_center = dy[:, 1:-1, :]
    normals = np.cross(dx_center, dy_center)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(normals).all(axis=-1) & (norm[..., 0] > 1e-8)
    normals = normals / np.maximum(norm, 1e-8)
    full = np.zeros((*depth.shape, 3), dtype=np.float32)
    full[1:-1, 1:-1] = normals.astype(np.float32)
    full_valid = np.zeros(depth.shape, dtype=bool)
    full_valid[1:-1, 1:-1] = valid
    return full, full_valid


def unoriented_angle_stats(pred_normal: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    depth_normals, normal_valid = central_depth_normals(depth, intrinsic)
    pred = pred_normal.astype(np.float32)
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    pred = pred / np.maximum(pred_norm, 1e-8)
    valid = mask & normal_valid & np.isfinite(pred).all(axis=-1) & (pred_norm[..., 0] > 0.5)
    if int(valid.sum()) < 10:
        return {"pixels": int(valid.sum())}
    dot = np.sum(pred[valid] * depth_normals[valid], axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return {
        "pixels": int(valid.sum()),
        "mean_angle_deg": float(np.mean(angle)),
        "median_angle_deg": float(np.median(angle)),
        "p90_angle_deg": float(np.percentile(angle, 90)),
    }


@dataclass
class SolveSummary:
    pixels: int
    x_edges: int
    y_edges: int
    lsqr_iterations: int
    lsqr_residual_norm: float
    mean_abs_rel_delta: float
    p95_abs_rel_delta: float
    max_abs_rel_delta: float


def refine_single_depth(
    depth: np.ndarray,
    normal: np.ndarray,
    intrinsic: np.ndarray,
    roi_mask: np.ndarray,
    *,
    fidelity_weight: float,
    max_log_delta: float,
    slope_clip: float,
    normal_xy_sign: float,
    normal_z_sign: float,
    min_depth: float,
    max_iter: int,
) -> tuple[np.ndarray, SolveSummary | None]:
    depth = depth.astype(np.float32)
    normal = normal.astype(np.float32)
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    ray_x = (xs - cx) / fx
    ray_y = (ys - cy) / fy

    signed_normal = normal.copy()
    signed_normal[..., 0] *= float(normal_xy_sign)
    signed_normal[..., 1] *= float(normal_xy_sign)
    signed_normal[..., 2] *= float(normal_z_sign)
    normal_norm = np.linalg.norm(signed_normal, axis=-1, keepdims=True)
    signed_normal = signed_normal / np.maximum(normal_norm, 1e-8)
    denom = signed_normal[..., 0] * ray_x + signed_normal[..., 1] * ray_y + signed_normal[..., 2]

    valid = (
        roi_mask.astype(bool)
        & np.isfinite(depth)
        & (depth > float(min_depth))
        & np.isfinite(signed_normal).all(axis=-1)
        & (normal_norm[..., 0] > 0.5)
        & np.isfinite(denom)
        & (np.abs(denom) > 1e-4)
    )
    valid_indices = np.flatnonzero(valid.reshape(-1))
    pixel_count = int(valid_indices.size)
    if pixel_count < 100:
        return depth.copy(), None

    local_index = -np.ones(height * width, dtype=np.int64)
    local_index[valid_indices] = np.arange(pixel_count, dtype=np.int64)
    index_map = local_index.reshape(height, width)

    grad_x = -signed_normal[..., 0] / (fx * denom)
    grad_y = -signed_normal[..., 1] / (fy * denom)
    grad_x = np.clip(grad_x, -float(slope_clip), float(slope_clip))
    grad_y = np.clip(grad_y, -float(slope_clip), float(slope_clip))

    right = valid[:, :-1] & valid[:, 1:]
    down = valid[:-1, :] & valid[1:, :]
    left_ids = index_map[:, :-1][right]
    right_ids = index_map[:, 1:][right]
    up_ids = index_map[:-1, :][down]
    down_ids = index_map[1:, :][down]
    target_x = 0.5 * (grad_x[:, :-1][right] + grad_x[:, 1:][right])
    target_y = 0.5 * (grad_y[:-1, :][down] + grad_y[1:, :][down])

    row_count = int(left_ids.size + up_ids.size + pixel_count)
    row_ids: list[np.ndarray] = []
    col_ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rhs = np.empty(row_count, dtype=np.float32)
    cursor = 0

    if left_ids.size:
        rows = np.arange(cursor, cursor + left_ids.size, dtype=np.int64)
        row_ids.extend([rows, rows])
        col_ids.extend([left_ids, right_ids])
        values.extend([-np.ones_like(left_ids, dtype=np.float32), np.ones_like(right_ids, dtype=np.float32)])
        rhs[cursor : cursor + left_ids.size] = target_x.astype(np.float32)
        cursor += left_ids.size

    if up_ids.size:
        rows = np.arange(cursor, cursor + up_ids.size, dtype=np.int64)
        row_ids.extend([rows, rows])
        col_ids.extend([up_ids, down_ids])
        values.extend([-np.ones_like(up_ids, dtype=np.float32), np.ones_like(down_ids, dtype=np.float32)])
        rhs[cursor : cursor + up_ids.size] = target_y.astype(np.float32)
        cursor += up_ids.size

    fidelity = np.sqrt(float(fidelity_weight)).astype(np.float32) if hasattr(np.sqrt(float(fidelity_weight)), "astype") else np.float32(np.sqrt(float(fidelity_weight)))
    rows = np.arange(cursor, cursor + pixel_count, dtype=np.int64)
    row_ids.append(rows)
    col_ids.append(np.arange(pixel_count, dtype=np.int64))
    values.append(np.full(pixel_count, fidelity, dtype=np.float32))
    log_depth = np.log(np.maximum(depth.reshape(-1)[valid_indices], float(min_depth))).astype(np.float32)
    rhs[cursor : cursor + pixel_count] = fidelity * log_depth

    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(row_ids), np.concatenate(col_ids))),
        shape=(row_count, pixel_count),
    ).tocsr()
    solved = lsqr(matrix, rhs, atol=1e-5, btol=1e-5, iter_lim=int(max_iter), show=False)
    refined_log = solved[0].astype(np.float32)
    delta = np.clip(refined_log - log_depth, -float(max_log_delta), float(max_log_delta))
    refined_values = np.exp(log_depth + delta).astype(np.float32)

    refined = depth.copy()
    flat_refined = refined.reshape(-1)
    flat_refined[valid_indices] = refined_values
    rel_delta = np.abs(refined_values - depth.reshape(-1)[valid_indices]) / np.maximum(depth.reshape(-1)[valid_indices], float(min_depth))
    summary = SolveSummary(
        pixels=pixel_count,
        x_edges=int(left_ids.size),
        y_edges=int(up_ids.size),
        lsqr_iterations=int(solved[2]),
        lsqr_residual_norm=float(solved[3]),
        mean_abs_rel_delta=float(np.mean(rel_delta)),
        p95_abs_rel_delta=float(np.percentile(rel_delta, 95)),
        max_abs_rel_delta=float(np.max(rel_delta)),
    )
    return refined, summary


def parse_view_indices(value: str | None, view_count: int) -> set[int]:
    if not value:
        return set(range(view_count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(part))
    return {idx for idx in selected if 0 <= idx < view_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Input predictions.npz")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing masks/")
    parser.add_argument("--output-predictions", required=True, help="Output patched predictions.npz")
    parser.add_argument("--roi", choices=["face", "head", "full"], default="head")
    parser.add_argument("--fidelity-weight", type=float, default=0.2)
    parser.add_argument("--max-log-delta", type=float, default=0.03)
    parser.add_argument("--slope-clip", type=float, default=0.02)
    parser.add_argument("--normal-xy-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--normal-z-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--view-indices", default=None, help="Comma-separated indices or ranges, e.g. 0,2,4-5")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    prediction_path = Path(args.predictions).resolve()
    scene_dir = Path(args.scene_dir).resolve()
    output_path = Path(args.output_predictions).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(prediction_path) as source:
        data = {key: source[key] for key in source.files}

    required = {"depth", "normal", "intrinsic", "extrinsic"}
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"Missing required prediction keys: {missing}")

    depth = data["depth"].astype(np.float32).copy()
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"Expected depth shape [V,H,W,1], got {depth.shape}")
    normals = data["normal"].astype(np.float32)
    intrinsics = data["intrinsic"].astype(np.float32)
    view_count, height, width, _ = depth.shape
    if normals.shape[:3] != (view_count, height, width):
        raise ValueError(f"Normal shape {normals.shape} does not match depth {depth.shape}")

    masks = load_mask_stack(scene_dir, height)
    if masks is None:
        masks = np.isfinite(depth[..., 0]) & (depth[..., 0] > float(args.min_depth))
    if masks.shape[0] != view_count:
        raise ValueError(f"Mask count {masks.shape[0]} does not match prediction view count {view_count}")
    roi_masks = np.stack([mask_to_roi(mask, args.roi) for mask in masks], axis=0)

    selected_views = parse_view_indices(args.view_indices, view_count)
    refined_depth = depth.copy()
    summaries: list[dict[str, object]] = []
    for view_idx in range(view_count):
        roi_mask = roi_masks[view_idx]
        before = unoriented_angle_stats(normals[view_idx], depth[view_idx, ..., 0], intrinsics[view_idx], roi_mask)
        if view_idx in selected_views:
            updated, solve_summary = refine_single_depth(
                depth[view_idx, ..., 0],
                normals[view_idx],
                intrinsics[view_idx],
                roi_mask,
                fidelity_weight=float(args.fidelity_weight),
                max_log_delta=float(args.max_log_delta),
                slope_clip=float(args.slope_clip),
                normal_xy_sign=float(args.normal_xy_sign),
                normal_z_sign=float(args.normal_z_sign),
                min_depth=float(args.min_depth),
                max_iter=int(args.max_iter),
            )
            refined_depth[view_idx, ..., 0] = updated
        else:
            solve_summary = None
        after = unoriented_angle_stats(normals[view_idx], refined_depth[view_idx, ..., 0], intrinsics[view_idx], roi_mask)
        summaries.append(
            {
                "view_idx": int(view_idx),
                "selected": bool(view_idx in selected_views),
                "roi_pixels": int(roi_mask.sum()),
                "before": before,
                "after": after,
                "solve": solve_summary.__dict__ if solve_summary is not None else None,
            }
        )

    data["depth"] = refined_depth.astype(np.float32)
    data["world_points"] = unproject_depth_to_world(refined_depth, data["extrinsic"], data["intrinsic"]).astype(np.float32)
    np.savez_compressed(output_path, **data)

    summary = {
        "input_predictions": str(prediction_path),
        "output_predictions": str(output_path),
        "scene_dir": str(scene_dir),
        "roi": args.roi,
        "fidelity_weight": float(args.fidelity_weight),
        "max_log_delta": float(args.max_log_delta),
        "slope_clip": float(args.slope_clip),
        "normal_xy_sign": float(args.normal_xy_sign),
        "normal_z_sign": float(args.normal_z_sign),
        "selected_views": sorted(selected_views),
        "views": summaries,
    }
    summary_path = Path(args.summary_json).resolve() if args.summary_json else output_path.with_name("normal_depth_refine_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_predictions": str(output_path), "summary_json": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
