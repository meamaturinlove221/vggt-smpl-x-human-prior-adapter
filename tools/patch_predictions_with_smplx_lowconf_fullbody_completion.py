from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

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
            "Use the matching SMPL-X prior map to complete only low-confidence "
            "full-body/hand regions while protecting head, face, and hairline."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--prior-maps", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--confidence-boost", type=float, default=180.0)
    parser.add_argument("--max-cam-delta", type=float, default=0.22)
    parser.add_argument("--anchor-min-pixels", type=int, default=256)
    parser.add_argument("--min-fill-pixels", type=int, default=32)
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


def load_scene_stacks(scene_dir: Path, view_count: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads((scene_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    views = manifest["exported_views"]
    if len(views) != view_count:
        raise ValueError(f"scene view count {len(views)} != predictions view count {view_count}")
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for view in views:
        rgb = Image.open(view["image_path"]).convert("RGB")
        mask = Image.open(view["mask_path"]).convert("L")
        if rgb.size != (518, 518):
            rgb = rgb.resize((518, 518), Image.Resampling.BILINEAR)
        if mask.size != (518, 518):
            mask = mask.resize((518, 518), Image.Resampling.NEAREST)
        rgbs.append(np.asarray(rgb, dtype=np.uint8))
        masks.append(np.asarray(mask, dtype=np.uint8) > 127)
    return np.stack(rgbs, axis=0), np.stack(masks, axis=0), views


def protect_head_face_hair(mask: np.ndarray) -> np.ndarray:
    protected = np.zeros_like(mask, dtype=bool)
    head = head_box_from_mask(mask)
    face = face_box_from_mask(mask)
    for box in (head, face):
        if box is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        pad = 8 if box == head else 4
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(mask.shape[1], x1 + pad)
        y1 = min(mask.shape[0], y1 + pad)
        protected[y0:y1, x0:x1] = True
    if head is not None:
        x0, y0, x1, y1 = [int(v) for v in head]
        h = max(1, y1 - y0)
        protected[max(0, y0 - 4) : min(mask.shape[0], y0 + int(round(0.55 * h))), max(0, x0 - 8) : min(mask.shape[1], x1 + 8)] = True
    return protected & mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask & False
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def make_overlay(path: Path, rgb: np.ndarray, fill_mask: np.ndarray, protect_mask: np.ndarray, label: str) -> None:
    arr = rgb.astype(np.float32).copy()
    arr[fill_mask] = arr[fill_mask] * 0.45 + np.asarray([40, 100, 255], dtype=np.float32) * 0.55
    edge = cv2.dilate(protect_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~protect_mask
    arr[edge] = np.asarray([255, 230, 0], dtype=np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(out)
    draw.text((8, 8), label, fill=(0, 255, 0))
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

    prior_maps = np.asarray(prior["prior_maps"], dtype=np.float32)
    if prior_maps.shape[0] != world_points.shape[0] or prior_maps.shape[2:] != world_points.shape[1:3]:
        raise ValueError(f"prior shape {prior_maps.shape} incompatible with world_points {world_points.shape}")
    prior_cam_norm = np.moveaxis(prior_maps[:, 23:26], 1, -1).astype(np.float32)
    prior_normals = np.moveaxis(prior_maps[:, 26:29], 1, -1).astype(np.float32)
    prior_visible = prior_maps[:, 29] > 0.5

    view_count = int(world_points.shape[0])
    rgbs, masks, views = load_scene_stacks(args.scene_dir.resolve(), view_count)
    cam_points = world_to_camera(world_points, extrinsic)
    filled_mask = np.zeros(masks.shape, dtype=bool)
    per_view: list[dict[str, Any]] = []

    for view_idx in range(view_count):
        mask = masks[view_idx]
        finite = np.isfinite(cam_points[view_idx]).all(axis=-1)
        support = mask & prior_visible[view_idx] & finite & np.isfinite(world_conf[view_idx]) & (world_conf[view_idx] > 0)
        threshold = float(np.percentile(world_conf[view_idx][support], float(args.conf_percentile))) if support.any() else float("nan")
        high = support & (world_conf[view_idx] >= threshold)
        low = support & (world_conf[view_idx] < threshold)

        protected = protect_head_face_hair(mask)
        hand_roi = mask_to_2d_roi(mask, "hands", rgb=rgbs[view_idx])
        body_roi = mask & ~protected
        fill_roi = np.zeros_like(mask, dtype=bool)
        if args.repair_body:
            fill_roi |= body_roi
        if args.repair_hands:
            fill_roi |= hand_roi
        fill_target = low & fill_roi & ~protected

        anchor = high & body_roi & ~protected
        if int(anchor.sum()) < int(args.anchor_min_pixels):
            # Let strong hand/body pixels help alignment when the torso is sparse.
            anchor = high & (body_roi | hand_roi) & ~protected
        if int(anchor.sum()) < int(args.anchor_min_pixels) or int(fill_target.sum()) < int(args.min_fill_pixels):
            make_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_smplx_lowconf_completion.png",
                rgbs[view_idx],
                fill_target,
                protected,
                f"skip fill={int(fill_target.sum())} anchor={int(anchor.sum())}",
            )
            per_view.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": str(views[view_idx].get("camera_id")),
                    "skipped": True,
                    "reason": "too_few_anchor_or_fill_pixels",
                    "conf_threshold": threshold,
                    "anchor_pixels": int(anchor.sum()),
                    "fill_pixels": int(fill_target.sum()),
                }
            )
            continue

        src = prior_cam_norm[view_idx][anchor]
        dst = cam_points[view_idx][anchor]
        sample = min(20000, src.shape[0])
        if src.shape[0] > sample:
            rng = np.random.default_rng(20260429 + view_idx)
            idx = rng.choice(src.shape[0], size=sample, replace=False)
            src_fit = src[idx]
            dst_fit = dst[idx]
        else:
            src_fit = src
            dst_fit = dst
        transform, inliers, residual = robust_similarity(src_fit, dst_fit)

        filled_cam = apply_similarity(prior_cam_norm[view_idx][fill_target], transform)
        current_cam = cam_points[view_idx][fill_target]
        delta = filled_cam - current_cam
        delta_norm = np.linalg.norm(delta, axis=1)
        keep = delta_norm <= float(args.max_cam_delta)
        if not keep.any():
            make_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_smplx_lowconf_completion.png",
                rgbs[view_idx],
                fill_target & False,
                protected,
                f"skip delta rejected fill={int(fill_target.sum())}",
            )
            per_view.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": str(views[view_idx].get("camera_id")),
                    "skipped": True,
                    "reason": "all_delta_rejected",
                    "conf_threshold": threshold,
                    "anchor_pixels": int(anchor.sum()),
                    "fill_pixels": int(fill_target.sum()),
                    "delta_p50": float(np.percentile(delta_norm, 50.0)),
                    "delta_p90": float(np.percentile(delta_norm, 90.0)),
                }
            )
            continue

        yy, xx = np.nonzero(fill_target)
        yy = yy[keep]
        xx = xx[keep]
        fill_keep = np.zeros_like(mask, dtype=bool)
        fill_keep[yy, xx] = True
        cam_points[view_idx, yy, xx] = filled_cam[keep]
        depth2[view_idx, yy, xx] = np.maximum(1e-4, filled_cam[keep, 2])
        world_conf[view_idx, yy, xx] = np.maximum(world_conf[view_idx, yy, xx], float(args.confidence_boost))
        depth_conf[view_idx, yy, xx] = np.maximum(depth_conf[view_idx, yy, xx], float(args.confidence_boost))
        normal[view_idx, yy, xx] = prior_normals[view_idx, yy, xx]
        normal_conf[view_idx, yy, xx] = np.maximum(normal_conf[view_idx, yy, xx], 1.0)
        filled_mask[view_idx, yy, xx] = True

        make_overlay(
            output_dir / "overlays" / f"view_{view_idx:02d}_smplx_lowconf_completion.png",
            rgbs[view_idx],
            fill_keep,
            protected,
            f"fill={int(fill_keep.sum())} anchor={int(anchor.sum())}",
        )
        hand_filled = int((fill_keep & hand_roi).sum())
        body_filled = int((fill_keep & body_roi).sum())
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": str(views[view_idx].get("camera_id")),
                "skipped": False,
                "conf_threshold": threshold,
                "anchor_pixels": int(anchor.sum()),
                "fit_inliers": int(inliers.sum()),
                "fit_residual_median": float(np.median(residual)),
                "fit_residual_p90": float(np.percentile(residual, 90.0)),
                "fill_pixels_requested": int(fill_target.sum()),
                "fill_pixels_written": int(fill_keep.sum()),
                "hand_pixels_written": hand_filled,
                "body_pixels_written": body_filled,
                "delta_p50": float(np.percentile(delta_norm, 50.0)),
                "delta_p90": float(np.percentile(delta_norm, 90.0)),
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
        "max_cam_delta": float(args.max_cam_delta),
        "filled_pixels": int(filled_mask.sum()),
        "per_view": per_view,
        "truthful_status": "local_smplx_lowconf_fullbody_hand_completion_diagnostic_not_training_result",
    }
    (output_dir / "smplx_lowconf_completion_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
