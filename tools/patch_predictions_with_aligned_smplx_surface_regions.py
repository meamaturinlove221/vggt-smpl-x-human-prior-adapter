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

from normal_line_multiview_eval import build_roi_masks  # noqa: E402
from render_open3d_pointcloud import mask_to_2d_roi  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch VGGT predictions with a per-view aligned SMPL-X posed surface. "
            "This is a local diagnostic for continuous body/head/hand scaffolding: "
            "SMPL-X points are first aligned to the current VGGT camera-space point "
            "map with a robust similarity transform, then blended only in selected "
            "regions."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-summary", default="", type=Path)
    parser.add_argument("--prior-maps", default="", type=Path)
    parser.add_argument(
        "--regions",
        default="face,head",
        help="Comma-separated subset of face,head,body,hands,full.",
    )
    parser.add_argument("--view-indices", default="")
    parser.add_argument("--alpha-face", type=float, default=0.65)
    parser.add_argument("--alpha-head", type=float, default=0.45)
    parser.add_argument("--alpha-body", type=float, default=0.35)
    parser.add_argument("--alpha-hands", type=float, default=0.70)
    parser.add_argument("--conf-boost", type=float, default=160.0)
    parser.add_argument("--normal-conf-boost", type=float, default=1.0)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-cam-delta", type=float, default=0.22)
    parser.add_argument("--anchor-min-pixels", type=int, default=512)
    parser.add_argument("--sample-anchors", type=int, default=25000)
    parser.add_argument("--protect-face", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--protect-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
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


def parse_regions(spec: str) -> set[str]:
    allowed = {"face", "head", "body", "hands", "full"}
    regions = {item.strip().lower() for item in str(spec).split(",") if item.strip()}
    unknown = sorted(regions - allowed)
    if unknown:
        raise ValueError(f"Unsupported regions: {unknown}; allowed={sorted(allowed)}")
    if not regions:
        raise ValueError("--regions selected nothing")
    return regions


def parse_view_indices(spec: str, view_count: int) -> np.ndarray:
    selected = np.zeros((view_count,), dtype=bool)
    if not str(spec).strip():
        selected[:] = True
        return selected
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= view_count:
            raise ValueError(f"view index {index} outside [0, {view_count})")
        selected[index] = True
    if not selected.any():
        raise ValueError("--view-indices did not select any views")
    return selected


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


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


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
    variance = float(np.sum(src0 * src0) / max(src.shape[0], 1))
    scale = float(np.trace(np.diag(singular_values) @ d) / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    return (scale * (rotation @ points.T).T + translation).astype(np.float32)


def rotate_normals(normals: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    _, rotation, _ = transform
    return normalize_vectors((rotation @ normals.T).T.astype(np.float32))


def robust_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray]:
    keep = np.ones(src.shape[0], dtype=bool)
    for _ in range(5):
        transform = umeyama_similarity(src[keep], dst[keep])
        residual = np.linalg.norm(apply_similarity(src, transform) - dst, axis=1)
        active = residual[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = max(median + 3.0 * mad + 1e-6, float(np.percentile(active, 80.0)))
        new_keep = residual <= threshold
        if int(new_keep.sum()) < 128 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    transform = umeyama_similarity(src[keep], dst[keep])
    residual = np.linalg.norm(apply_similarity(src, transform) - dst, axis=1)
    return transform, residual


def load_scene_stacks(scene_dir: Path, view_count: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads((scene_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    views = list(manifest["exported_views"])
    if len(views) != view_count:
        raise ValueError(f"scene view count {len(views)} != prediction view count {view_count}")
    height, width = size
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for view in views:
        rgb = Image.open(view["image_path"]).convert("RGB")
        mask = Image.open(view["mask_path"]).convert("L")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        rgbs.append(np.asarray(rgb, dtype=np.uint8))
        masks.append(np.asarray(mask, dtype=np.uint8) > 127)
    return np.stack(rgbs, axis=0), np.stack(masks, axis=0), views


def protected_mask(mask: np.ndarray, protect_face: bool, protect_head: bool) -> np.ndarray:
    protected = np.zeros_like(mask, dtype=bool)
    if protect_head:
        box = head_box_from_mask(mask)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            protected[max(0, y0 - 8) : min(mask.shape[0], y1 + 8), max(0, x0 - 8) : min(mask.shape[1], x1 + 8)] = True
    if protect_face:
        box = face_box_from_mask(mask)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            protected[max(0, y0 - 6) : min(mask.shape[0], y1 + 6), max(0, x0 - 6) : min(mask.shape[1], x1 + 6)] = True
    return protected & mask


def region_alpha(mask: np.ndarray, rgb: np.ndarray, regions: set[str], args: argparse.Namespace) -> np.ndarray:
    rois = build_roi_masks(mask.astype(bool))
    alpha = np.zeros_like(mask, dtype=np.float32)
    if "full" in regions:
        alpha = np.maximum(alpha, mask.astype(np.float32) * max(float(args.alpha_body), float(args.alpha_head)))
    if "body" in regions:
        alpha = np.maximum(alpha, (mask & ~rois["head"]).astype(np.float32) * float(args.alpha_body))
    if "head" in regions:
        alpha = np.maximum(alpha, rois["head"].astype(np.float32) * float(args.alpha_head))
    if "face" in regions:
        alpha = np.maximum(alpha, rois["face"].astype(np.float32) * float(args.alpha_face))
    if "hands" in regions:
        hands = mask_to_2d_roi(mask.astype(bool), "hands", rgb=rgb)
        alpha = np.maximum(alpha, hands.astype(np.float32) * float(args.alpha_hands))
    alpha[protected_mask(mask, bool(args.protect_face), bool(args.protect_head))] = 0.0
    return alpha


def save_overlay(path: Path, rgb: np.ndarray, patch: np.ndarray, label: str) -> None:
    arr = rgb.astype(np.float32).copy()
    arr[patch] = arr[patch] * 0.38 + np.asarray([40, 120, 255], dtype=np.float32) * 0.62
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(out)
    draw.text((8, 8), label, fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    output_npz = args.output_npz.resolve()
    output_summary = args.output_summary.resolve() if str(args.output_summary) else output_npz.with_suffix(".json")
    prior_path = args.prior_maps if str(args.prior_maps) else args.scene_dir / "prior_maps.npz"
    prior = load_npz(prior_path)

    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth2 = depth[..., 0].copy() if depth.ndim == 4 else depth.copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)

    prior_maps = np.asarray(prior["prior_maps"], dtype=np.float32)
    channels = [str(item) for item in prior["prior_channels"]]
    idx = {name: i for i, name in enumerate(channels)}
    required = [
        "smplx_posed_cam_x",
        "smplx_posed_cam_y",
        "smplx_posed_cam_z",
        "smplx_cam_nx",
        "smplx_cam_ny",
        "smplx_cam_nz",
        "smplx_visible_mask",
    ]
    missing = [name for name in required if name not in idx]
    if missing:
        raise KeyError(f"Missing prior channels: {missing}")
    if prior_maps.shape[0] != world_points.shape[0] or prior_maps.shape[2:] != world_points.shape[1:3]:
        raise ValueError(f"prior shape {prior_maps.shape} incompatible with world_points {world_points.shape}")

    view_count, height, width, _ = world_points.shape
    selected_views = parse_view_indices(str(args.view_indices), view_count)
    regions = parse_regions(str(args.regions))
    rgbs, masks, views = load_scene_stacks(args.scene_dir.resolve(), view_count, (height, width))

    smplx_cam = np.stack(
        [
            prior_maps[:, idx["smplx_posed_cam_x"]],
            prior_maps[:, idx["smplx_posed_cam_y"]],
            prior_maps[:, idx["smplx_posed_cam_z"]],
        ],
        axis=-1,
    ).astype(np.float32)
    smplx_normals = np.stack(
        [
            prior_maps[:, idx["smplx_cam_nx"]],
            prior_maps[:, idx["smplx_cam_ny"]],
            prior_maps[:, idx["smplx_cam_nz"]],
        ],
        axis=-1,
    ).astype(np.float32)
    prior_mask = np.asarray(prior.get("prior_mask", np.ones(world_points.shape[:3], dtype=bool)), dtype=bool)
    prior_visible = prior_mask & (prior_maps[:, idx["smplx_visible_mask"]] > 0.5)
    cam_points = world_to_camera(world_points, extrinsic)

    patch_mask_total = np.zeros(world_points.shape[:3], dtype=bool)
    per_view: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260429)

    for view_idx in range(view_count):
        if not bool(selected_views[view_idx]):
            per_view.append({"view_index": int(view_idx), "camera_id": str(views[view_idx].get("camera_id")), "skipped": True, "reason": "view_not_selected"})
            continue
        mask = masks[view_idx]
        finite_base = np.isfinite(cam_points[view_idx]).all(axis=-1)
        finite_prior = np.isfinite(smplx_cam[view_idx]).all(axis=-1) & (smplx_cam[view_idx, ..., 2] > 0.0)
        support = mask & prior_visible[view_idx] & finite_base & finite_prior & np.isfinite(world_conf[view_idx]) & (world_conf[view_idx] > 0.0)
        if int(support.sum()) < int(args.anchor_min_pixels):
            per_view.append({"view_index": int(view_idx), "camera_id": str(views[view_idx].get("camera_id")), "skipped": True, "reason": "too_few_support", "support_pixels": int(support.sum())})
            continue
        threshold = float(np.percentile(world_conf[view_idx][support], float(args.conf_percentile)))
        anchor = support & (world_conf[view_idx] >= threshold)
        if int(anchor.sum()) < int(args.anchor_min_pixels):
            anchor = support
        if int(anchor.sum()) > int(args.sample_anchors):
            flat = np.flatnonzero(anchor.reshape(-1))
            chosen = rng.choice(flat, size=int(args.sample_anchors), replace=False)
            sampled = np.zeros(anchor.size, dtype=bool)
            sampled[chosen] = True
            anchor = sampled.reshape(anchor.shape)

        src = smplx_cam[view_idx][anchor]
        dst = cam_points[view_idx][anchor]
        transform, residual = robust_similarity(src, dst)
        alpha = region_alpha(mask, rgbs[view_idx], regions, args)
        target = (alpha > 0.0) & support
        if not target.any():
            per_view.append({"view_index": int(view_idx), "camera_id": str(views[view_idx].get("camera_id")), "skipped": True, "reason": "empty_target", "anchor_pixels": int(anchor.sum())})
            continue

        smplx_aligned = apply_similarity(smplx_cam[view_idx][target], transform)
        current = cam_points[view_idx][target]
        delta = smplx_aligned - current
        delta_norm = np.linalg.norm(delta, axis=1)
        keep = delta_norm <= float(args.max_cam_delta)
        yy, xx = np.nonzero(target)
        yy = yy[keep]
        xx = xx[keep]
        if yy.size == 0:
            per_view.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": str(views[view_idx].get("camera_id")),
                    "skipped": True,
                    "reason": "all_delta_rejected",
                    "anchor_pixels": int(anchor.sum()),
                    "target_pixels": int(target.sum()),
                    "delta_p50": float(np.percentile(delta_norm, 50.0)),
                    "delta_p90": float(np.percentile(delta_norm, 90.0)),
                }
            )
            continue

        blended_alpha = alpha[yy, xx, None].astype(np.float32)
        cam_points[view_idx, yy, xx] = cam_points[view_idx, yy, xx] * (1.0 - blended_alpha) + smplx_aligned[keep] * blended_alpha
        depth2[view_idx, yy, xx] = np.maximum(1e-4, cam_points[view_idx, yy, xx, 2])
        world_conf[view_idx, yy, xx] = np.maximum(world_conf[view_idx, yy, xx], float(args.conf_boost))
        depth_conf[view_idx, yy, xx] = np.maximum(depth_conf[view_idx, yy, xx], float(args.conf_boost))
        aligned_normals = -rotate_normals(smplx_normals[view_idx][target][keep], transform)
        current_normals = normal[view_idx, yy, xx]
        normal[view_idx, yy, xx] = normalize_vectors(current_normals * (1.0 - blended_alpha) + aligned_normals * blended_alpha)
        normal_conf[view_idx, yy, xx] = np.maximum(normal_conf[view_idx, yy, xx], float(args.normal_conf_boost))
        patch_mask_total[view_idx, yy, xx] = True

        if bool(args.write_debug):
            save_overlay(
                output_npz.parent / "overlays" / f"view_{view_idx:02d}_aligned_smplx_regions.png",
                rgbs[view_idx],
                patch_mask_total[view_idx],
                f"patch={int(yy.size)} anchor={int(anchor.sum())}",
            )
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": str(views[view_idx].get("camera_id")),
                "skipped": False,
                "support_pixels": int(support.sum()),
                "anchor_pixels": int(anchor.sum()),
                "target_pixels": int(target.sum()),
                "patch_pixels": int(yy.size),
                "fit_residual_median": float(np.median(residual)),
                "fit_residual_p90": float(np.percentile(residual, 90.0)),
                "delta_p50": float(np.percentile(delta_norm, 50.0)),
                "delta_p90": float(np.percentile(delta_norm, 90.0)),
                "similarity_scale": float(transform[0]),
            }
        )

    world_points = camera_to_world(cam_points, extrinsic)
    out: dict[str, Any] = {key: np.asarray(value) for key, value in base.items()}
    out["world_points"] = world_points.astype(np.float32)
    out["world_points_conf"] = world_conf.astype(np.float32)
    out["depth"] = depth2[..., None].astype(np.float32) if depth.ndim == 4 else depth2.astype(np.float32)
    out["depth_conf"] = depth_conf.astype(np.float32)
    out["normal"] = normalize_vectors(normal).astype(np.float32)
    out["normal_conf"] = normal_conf.astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "prior_maps": str(prior_path.resolve()),
        "output_npz": str(output_npz),
        "regions": sorted(regions),
        "view_indices": [int(i) for i in np.flatnonzero(selected_views)],
        "alpha_face": float(args.alpha_face),
        "alpha_head": float(args.alpha_head),
        "alpha_body": float(args.alpha_body),
        "alpha_hands": float(args.alpha_hands),
        "conf_boost": float(args.conf_boost),
        "max_cam_delta": float(args.max_cam_delta),
        "patch_pixels": int(patch_mask_total.sum()),
        "patch_pixels_by_view": [int(patch_mask_total[i].sum()) for i in range(view_count)],
        "per_view": per_view,
        "truthful_status": "local_aligned_smplx_surface_diagnostic_not_final_pass",
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
