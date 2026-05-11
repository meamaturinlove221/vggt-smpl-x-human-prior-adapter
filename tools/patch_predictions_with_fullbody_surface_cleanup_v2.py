from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from render_open3d_pointcloud import mask_to_2d_roi  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative local full-body/hand candidate by using SMPL-X "
            "only as a low-confidence depth-hole hint, with connected body masks, "
            "camera-ray clamping, and fragment cleanup. Head/face protected pixels "
            "are never written."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--prior-maps", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--confidence-boost", type=float, default=128.0)
    parser.add_argument("--normal-confidence-boost", type=float, default=0.08)
    parser.add_argument("--max-depth-delta", type=float, default=0.055)
    parser.add_argument("--reject-prior-depth-delta", type=float, default=0.18)
    parser.add_argument("--local-depth-margin", type=float, default=0.075)
    parser.add_argument("--local-window", type=int, default=31)
    parser.add_argument("--min-local-anchors", type=float, default=12.0)
    parser.add_argument("--max-anchor-distance-body", type=float, default=28.0)
    parser.add_argument("--max-anchor-distance-hand", type=float, default=46.0)
    parser.add_argument("--anchor-min-pixels", type=int, default=512)
    parser.add_argument("--min-fill-component-pixels", type=int, default=12)
    parser.add_argument("--min-hand-component-pixels", type=int, default=18)
    parser.add_argument("--max-hand-components", type=int, default=4)
    parser.add_argument("--demote-depth-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-outlier-margin", type=float, default=0.24)
    parser.add_argument("--repair-hands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-body", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def closed_form_inverse_se3(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    out = np.tile(np.eye(4, dtype=np.float32), (extrinsic.shape[0], 1, 1))
    out[:, :3, :3] = rotation_t
    out[:, :3, 3] = -np.einsum("vij,vj->vi", rotation_t, translation)
    return out


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3].astype(np.float32)
    translation = extrinsic[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    cam_to_world = closed_form_inverse_se3(extrinsic.astype(np.float32))
    rotation = cam_to_world[:, :3, :3]
    translation = cam_to_world[:, :3, 3]
    return np.einsum("vij,vhwj->vhwi", rotation, points_cam.astype(np.float32)) + translation[:, None, None, :]


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    covariance = (dst0.T @ src0) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    rotation = u @ d @ vt
    variance = float(np.sum(src0 * src0) / src.shape[0])
    scale = float(np.trace(np.diag(singular_values) @ d) / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    return (scale * (rotation @ points.T).T + translation).astype(np.float32)


def robust_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    keep = np.ones(src.shape[0], dtype=bool)
    for _ in range(5):
        transform = umeyama_similarity(src[keep], dst[keep])
        residual = np.linalg.norm(apply_similarity(src, transform) - dst, axis=1)
        active = residual[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = max(median + 3.0 * mad + 1e-6, float(np.percentile(active, 80.0)))
        new_keep = residual <= threshold
        if new_keep.sum() < 128 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    transform = umeyama_similarity(src[keep], dst[keep])
    residual = np.linalg.norm(apply_similarity(src, transform) - dst, axis=1)
    return transform, keep, residual


def resize_bool(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 127


def resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_scene_stacks(scene_dir: Path, view_count: int, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    size = (int(hw[1]), int(hw[0]))
    manifest_path = scene_dir / "scene_manifest.json"
    views: list[dict[str, Any]]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = list(manifest["exported_views"])
        if len(views) != view_count:
            raise ValueError(f"scene view count {len(views)} != predictions view count {view_count}")
        image_paths = [Path(view["image_path"]) for view in views]
        mask_paths = [Path(view["mask_path"]) for view in views]
    else:
        image_paths = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file())
        mask_paths = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file())
        if len(image_paths) != view_count or len(mask_paths) != view_count:
            raise ValueError("scene images/masks do not match prediction view count")
        views = [{"camera_id": str(idx), "image_path": str(img), "mask_path": str(mask)} for idx, (img, mask) in enumerate(zip(image_paths, mask_paths))]

    rgbs = [resize_rgb(path, size) for path in image_paths]
    masks = [resize_bool(path, size) for path in mask_paths]
    return np.stack(rgbs, axis=0), np.stack(masks, axis=0), views


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def connected_body_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    cleaned = largest_component(cleaned)
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return largest_component(cleaned)


def protect_head_face_hair(mask: np.ndarray) -> np.ndarray:
    protected = np.zeros_like(mask, dtype=bool)
    head = head_box_from_mask(mask)
    face = face_box_from_mask(mask)
    if head is not None:
        x0, y0, x1, y1 = [int(v) for v in head]
        x0 = max(0, x0 - 8)
        y0 = max(0, y0 - 8)
        x1 = min(mask.shape[1], x1 + 8)
        y1 = min(mask.shape[0], y1 + 8)
        protected[y0:y1, x0:x1] = True
        h = max(1, y1 - y0)
        protected[
            max(0, y0 - 4) : min(mask.shape[0], y0 + int(round(0.58 * h))),
            max(0, x0 - 8) : min(mask.shape[1], x1 + 8),
        ] = True
    if face is not None:
        x0, y0, x1, y1 = [int(v) for v in face]
        protected[max(0, y0 - 4) : min(mask.shape[0], y1 + 4), max(0, x0 - 4) : min(mask.shape[1], x1 + 4)] = True
    return protected & mask


def keep_large_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_pixels):
            out |= labels == label
    return out


def distance_to_true(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    src = (~mask.astype(bool)).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)


def local_depth_mean(depth: np.ndarray, support: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(3, int(window) | 1)
    valid = support & np.isfinite(depth) & (depth > 0.0)
    valid_f = valid.astype(np.float32)
    depth_f = np.where(valid, depth, 0.0).astype(np.float32)
    sums = cv2.boxFilter(depth_f, -1, (window, window), normalize=False, borderType=cv2.BORDER_CONSTANT)
    counts = cv2.boxFilter(valid_f, -1, (window, window), normalize=False, borderType=cv2.BORDER_CONSTANT)
    mean = sums / np.maximum(counts, 1.0)
    return mean.astype(np.float32), counts.astype(np.float32)


def camera_rays_for_pixels(ys: np.ndarray, xs: np.ndarray, z: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x_cam = (xs.astype(np.float32) - cx) * z.astype(np.float32) / max(fx, 1e-6)
    y_cam = (ys.astype(np.float32) - cy) * z.astype(np.float32) / max(fy, 1e-6)
    return np.stack([x_cam, y_cam, z.astype(np.float32)], axis=-1)


def demote_small_hand_fragments(
    *,
    conf: np.ndarray,
    depth_conf: np.ndarray,
    normal_conf: np.ndarray,
    hand_roi: np.ndarray,
    threshold: float,
    min_pixels: int,
    max_components: int,
) -> int:
    kept = hand_roi & np.isfinite(conf) & (conf >= float(threshold))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(kept.astype(np.uint8), connectivity=8)
    components: list[tuple[int, int]] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_pixels):
            components.append((area, label))
    components.sort(reverse=True)
    allowed = np.zeros_like(kept, dtype=bool)
    for _, label in components[: int(max_components)]:
        allowed |= labels == label
    demote = kept & ~allowed
    if not bool(demote.any()):
        return 0
    conf[demote] = np.minimum(conf[demote], 1e-3)
    depth_conf[demote] = np.minimum(depth_conf[demote], 1e-3)
    normal_conf[demote] = np.minimum(normal_conf[demote], 1e-4)
    return int(demote.sum())


def make_overlay(
    path: Path,
    rgb: np.ndarray,
    fill_mask: np.ndarray,
    demote_mask: np.ndarray,
    protected: np.ndarray,
    connected: np.ndarray,
    label: str,
) -> None:
    arr = rgb.astype(np.float32).copy()
    not_connected = (~connected) & (protected | fill_mask | demote_mask)
    arr[fill_mask] = arr[fill_mask] * 0.42 + np.asarray([0, 220, 80], dtype=np.float32) * 0.58
    arr[demote_mask] = arr[demote_mask] * 0.40 + np.asarray([255, 40, 40], dtype=np.float32) * 0.60
    edge = cv2.dilate(protected.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~protected
    arr[edge] = np.asarray([255, 230, 0], dtype=np.float32)
    arr[not_connected] = arr[not_connected] * 0.40 + np.asarray([50, 50, 255], dtype=np.float32) * 0.60
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), label[:180], fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    prior = load_npz(args.prior_maps)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth2 = depth[..., 0].copy() if depth.ndim == 4 else depth.copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)

    prior_maps = np.asarray(prior["prior_maps"], dtype=np.float32)
    if prior_maps.shape[0] != world_points.shape[0] or prior_maps.shape[2:] != world_points.shape[1:3]:
        raise ValueError(f"prior shape {prior_maps.shape} incompatible with world_points {world_points.shape}")
    prior_cam = np.moveaxis(prior_maps[:, 23:26], 1, -1).astype(np.float32)
    prior_visible = prior_maps[:, 29] > 0.5

    view_count, height, width, _ = world_points.shape
    rgbs, masks, views = load_scene_stacks(args.scene_dir.resolve(), view_count, (height, width))
    cam_points = world_to_camera(world_points, extrinsic)

    filled_mask_all = np.zeros(masks.shape, dtype=bool)
    demoted_mask_all = np.zeros(masks.shape, dtype=bool)
    protected_all = np.zeros(masks.shape, dtype=bool)
    connected_all = np.zeros(masks.shape, dtype=bool)
    per_view: list[dict[str, Any]] = []

    for view_idx in range(view_count):
        raw_mask = masks[view_idx]
        connected = connected_body_mask(raw_mask)
        protected = protect_head_face_hair(connected)
        protected_all[view_idx] = protected
        connected_all[view_idx] = connected

        body_roi = connected & ~protected
        hand_roi = mask_to_2d_roi(connected, "hands", rgb=rgbs[view_idx]) & ~protected
        fill_roi = np.zeros_like(raw_mask, dtype=bool)
        if args.repair_body:
            fill_roi |= body_roi
        if args.repair_hands:
            fill_roi |= hand_roi

        finite_cam = np.isfinite(cam_points[view_idx]).all(axis=-1)
        finite_depth = np.isfinite(depth2[view_idx]) & (depth2[view_idx] > 0.0)
        support = connected & finite_cam & finite_depth & np.isfinite(world_conf[view_idx]) & (world_conf[view_idx] > 0.0)
        threshold = float(np.percentile(world_conf[view_idx][support], float(args.conf_percentile))) if support.any() else float("nan")
        high = support & (world_conf[view_idx] >= threshold)
        low = support & (world_conf[view_idx] < threshold)

        off_component = raw_mask & ~connected & ~protected
        if off_component.any():
            world_conf[view_idx][off_component] = np.minimum(world_conf[view_idx][off_component], 1e-3)
            depth_conf[view_idx][off_component] = np.minimum(depth_conf[view_idx][off_component], 1e-3)
            normal_conf[view_idx][off_component] = np.minimum(normal_conf[view_idx][off_component], 1e-4)
            demoted_mask_all[view_idx] |= off_component

        anchor = high & body_roi & prior_visible[view_idx]
        if int(anchor.sum()) < int(args.anchor_min_pixels):
            anchor = high & (body_roi | hand_roi) & prior_visible[view_idx]

        fill_target = low & fill_roi & prior_visible[view_idx]
        fill_target = keep_large_components(fill_target, int(args.min_fill_component_pixels))

        anchor_dist = distance_to_true(anchor)
        near_body = anchor_dist <= float(args.max_anchor_distance_body)
        near_hand = anchor_dist <= float(args.max_anchor_distance_hand)
        fill_target &= ((body_roi & near_body) | (hand_roi & near_hand))

        local_mean, local_count = local_depth_mean(depth2[view_idx], high & fill_roi, int(args.local_window))
        local_ok = local_count >= float(args.min_local_anchors)
        fill_target &= local_ok

        if int(anchor.sum()) < int(args.anchor_min_pixels) or int(fill_target.sum()) == 0:
            demoted_hands = demote_small_hand_fragments(
                conf=world_conf[view_idx],
                depth_conf=depth_conf[view_idx],
                normal_conf=normal_conf[view_idx],
                hand_roi=hand_roi,
                threshold=threshold,
                min_pixels=int(args.min_hand_component_pixels),
                max_components=int(args.max_hand_components),
            )
            if demoted_hands:
                demoted_mask_all[view_idx] |= hand_roi
            make_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_fullbody_surface_cleanup_v2.png",
                rgbs[view_idx],
                np.zeros_like(raw_mask, dtype=bool),
                demoted_mask_all[view_idx],
                protected,
                connected,
                f"skip fill={int(fill_target.sum())} anchor={int(anchor.sum())} hand_demote={demoted_hands}",
            )
            per_view.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": str(views[view_idx].get("camera_id")),
                    "skipped": True,
                    "reason": "too_few_anchor_or_fill_pixels",
                    "conf_threshold": threshold,
                    "connected_pixels": int(connected.sum()),
                    "protected_pixels": int(protected.sum()),
                    "anchor_pixels": int(anchor.sum()),
                    "fill_pixels_requested": int(fill_target.sum()),
                    "off_component_demoted_pixels": int(off_component.sum()),
                    "hand_fragment_demoted_pixels": int(demoted_hands),
                }
            )
            continue

        src = prior_cam[view_idx][anchor]
        dst = cam_points[view_idx][anchor]
        sample = min(24000, src.shape[0])
        if src.shape[0] > sample:
            rng = np.random.default_rng(20260429 + view_idx)
            idx = rng.choice(src.shape[0], size=sample, replace=False)
            src_fit = src[idx]
            dst_fit = dst[idx]
        else:
            src_fit = src
            dst_fit = dst
        transform, inliers, residual = robust_similarity(src_fit, dst_fit)

        yy, xx = np.nonzero(fill_target)
        proposed_cam = apply_similarity(prior_cam[view_idx][yy, xx], transform)
        current_z = depth2[view_idx, yy, xx].astype(np.float32)
        prior_z = proposed_cam[:, 2].astype(np.float32)
        raw_dz = prior_z - current_z
        keep = np.isfinite(raw_dz) & (np.abs(raw_dz) <= float(args.reject_prior_depth_delta))
        clipped_dz = np.clip(raw_dz, -float(args.max_depth_delta), float(args.max_depth_delta))
        new_z = current_z + clipped_dz
        local_z = local_mean[yy, xx]
        lo = local_z - float(args.local_depth_margin)
        hi = local_z + float(args.local_depth_margin)
        unclamped_z = new_z.copy()
        new_z = np.minimum(np.maximum(new_z, lo), hi)
        keep &= np.isfinite(new_z) & (new_z > 1e-4)

        yy = yy[keep]
        xx = xx[keep]
        if yy.size:
            new_cam = camera_rays_for_pixels(yy, xx, new_z[keep], intrinsic[view_idx])
            cam_points[view_idx, yy, xx] = new_cam
            depth2[view_idx, yy, xx] = new_z[keep]
            world_conf[view_idx, yy, xx] = np.maximum(world_conf[view_idx, yy, xx], float(args.confidence_boost))
            depth_conf[view_idx, yy, xx] = np.maximum(depth_conf[view_idx, yy, xx], float(args.confidence_boost))
            normal_conf[view_idx, yy, xx] = np.maximum(normal_conf[view_idx, yy, xx], float(args.normal_confidence_boost))
            fill_keep = np.zeros_like(raw_mask, dtype=bool)
            fill_keep[yy, xx] = True
            filled_mask_all[view_idx] = fill_keep
        else:
            fill_keep = np.zeros_like(raw_mask, dtype=bool)

        depth_outlier_demoted = 0
        if args.demote_depth_outliers:
            post_mean, post_count = local_depth_mean(depth2[view_idx], high & fill_roi, int(args.local_window))
            suspect = (
                fill_roi
                & ~protected
                & support
                & (post_count >= float(args.min_local_anchors))
                & (np.abs(depth2[view_idx] - post_mean) > float(args.depth_outlier_margin))
                & ~fill_keep
            )
            if suspect.any():
                world_conf[view_idx][suspect] = np.minimum(world_conf[view_idx][suspect], 1e-3)
                depth_conf[view_idx][suspect] = np.minimum(depth_conf[view_idx][suspect], 1e-3)
                normal_conf[view_idx][suspect] = np.minimum(normal_conf[view_idx][suspect], 1e-4)
                demoted_mask_all[view_idx] |= suspect
                depth_outlier_demoted = int(suspect.sum())

        hand_fragment_demoted = demote_small_hand_fragments(
            conf=world_conf[view_idx],
            depth_conf=depth_conf[view_idx],
            normal_conf=normal_conf[view_idx],
            hand_roi=hand_roi,
            threshold=threshold,
            min_pixels=int(args.min_hand_component_pixels),
            max_components=int(args.max_hand_components),
        )
        if hand_fragment_demoted:
            demoted_mask_all[view_idx] |= hand_roi

        make_overlay(
            output_dir / "overlays" / f"view_{view_idx:02d}_fullbody_surface_cleanup_v2.png",
            rgbs[view_idx],
            fill_keep,
            demoted_mask_all[view_idx],
            protected,
            connected,
            f"fill={int(fill_keep.sum())} body={int((fill_keep & body_roi).sum())} hand={int((fill_keep & hand_roi).sum())}",
        )

        requested = int(fill_target.sum())
        raw_dz_valid = raw_dz[np.isfinite(raw_dz)]
        written_dz = (new_z[keep] - current_z[keep]) if yy.size else np.asarray([], dtype=np.float32)
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": str(views[view_idx].get("camera_id")),
                "skipped": False,
                "conf_threshold": threshold,
                "connected_pixels": int(connected.sum()),
                "protected_pixels": int(protected.sum()),
                "anchor_pixels": int(anchor.sum()),
                "fit_inliers": int(inliers.sum()),
                "fit_residual_median": float(np.median(residual)),
                "fit_residual_p90": float(np.percentile(residual, 90.0)),
                "fill_pixels_requested": requested,
                "fill_pixels_written": int(fill_keep.sum()),
                "fill_pixels_rejected": int(requested - fill_keep.sum()),
                "body_pixels_written": int((fill_keep & body_roi).sum()),
                "hand_pixels_written": int((fill_keep & hand_roi).sum()),
                "off_component_demoted_pixels": int(off_component.sum()),
                "depth_outlier_demoted_pixels": depth_outlier_demoted,
                "hand_fragment_demoted_pixels": int(hand_fragment_demoted),
                "raw_prior_depth_delta_p50": float(np.percentile(raw_dz_valid, 50.0)) if raw_dz_valid.size else None,
                "raw_prior_depth_delta_p90_abs": float(np.percentile(np.abs(raw_dz_valid), 90.0)) if raw_dz_valid.size else None,
                "written_depth_delta_p50": float(np.percentile(written_dz, 50.0)) if written_dz.size else None,
                "written_depth_delta_p90_abs": float(np.percentile(np.abs(written_dz), 90.0)) if written_dz.size else None,
                "local_depth_clamped_pixels": int(np.count_nonzero(np.abs(unclamped_z[keep] - new_z[keep]) > 1e-6)) if yy.size else 0,
            }
        )

    world_points = camera_to_world(cam_points, extrinsic)
    out = dict(base)
    out["world_points"] = world_points.astype(base["world_points"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth"] = depth2[..., None].astype(base["depth"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["normal"] = normal.astype(base["normal"].dtype, copy=False)
    out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)

    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)

    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "prior_maps": str(args.prior_maps.resolve()),
        "output_predictions": str(output_path),
        "conf_percentile": float(args.conf_percentile),
        "confidence_boost": float(args.confidence_boost),
        "max_depth_delta": float(args.max_depth_delta),
        "reject_prior_depth_delta": float(args.reject_prior_depth_delta),
        "local_depth_margin": float(args.local_depth_margin),
        "filled_pixels": int(filled_mask_all.sum()),
        "demoted_pixels": int(demoted_mask_all.sum()),
        "protected_pixels": int(protected_all.sum()),
        "connected_pixels": int(connected_all.sum()),
        "per_view": per_view,
        "truthful_status": (
            "local_fullbody_hand_candidate_v2_depth_axis_prior_hint_only_head_face_protected_no_cloud"
        ),
    }
    (output_dir / "fullbody_surface_cleanup_v2_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
