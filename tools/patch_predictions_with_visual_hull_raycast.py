from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from patch_predictions_with_image_face_relief import update_predicted_normals  # noqa: E402
from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace human-mask pixels with a shared sparse-view visual-hull "
            "raycast surface. This is a local diagnostic for whether mask/camera "
            "geometry can remove shell holes without decoupling world_points and depth."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-summary", default="", type=Path)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--mask-dilate", type=int, default=1)
    parser.add_argument("--bbox-margin-frac", type=float, default=0.08)
    parser.add_argument("--min-inside-views", type=int, default=5)
    parser.add_argument("--min-projected-views", type=int, default=5)
    parser.add_argument("--ray-chunk", type=int, default=24576)
    parser.add_argument("--voxel-chunk", type=int, default=262144)
    parser.add_argument("--conf-mode", choices=("preserve", "max"), default="preserve")
    parser.add_argument("--conf-boost", type=float, default=220.0)
    parser.add_argument("--replace-mask-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-debug", action="store_true")
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


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return (points_cam - translation[None, :]) @ rotation


def camera_center_world(extrinsic: np.ndarray) -> np.ndarray:
    origin_cam = np.zeros((1, 3), dtype=np.float32)
    return camera_to_world(origin_cam, extrinsic)[0]


def project_world(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    u = fx * cam[:, 0] / np.clip(z, 1e-6, None) + cx
    v = fy * cam[:, 1] / np.clip(z, 1e-6, None) + cy
    return u, v, z


def load_masks(scene_dir: Path, view_count: int, hw: tuple[int, int], dilate: int) -> tuple[np.ndarray, list[np.ndarray]]:
    masks: list[np.ndarray] = []
    rgbs: list[np.ndarray] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, hw)
        mask = scene.mask.astype(bool)
        if dilate > 0:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=int(dilate)).astype(bool)
        masks.append(mask)
        rgbs.append(scene.rgb)
    return np.stack(masks, axis=0), rgbs


def bbox_from_base(base_points: np.ndarray, masks: np.ndarray, margin_frac: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(base_points).all(axis=-1)
    support = finite & masks
    points = base_points[support]
    if points.shape[0] < 128:
        points = base_points[finite]
    if points.shape[0] < 128:
        raise RuntimeError("not enough finite points to build visual-hull bbox")
    lo = np.percentile(points, 0.5, axis=0).astype(np.float32)
    hi = np.percentile(points, 99.5, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 1e-4)
    margin = span * float(margin_frac)
    return (lo - margin).astype(np.float32), (hi + margin).astype(np.float32)


def build_visual_hull(
    *,
    bmin: np.ndarray,
    bmax: np.ndarray,
    masks: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    resolution: int,
    min_inside_views: int,
    min_projected_views: int,
    chunk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    res = int(resolution)
    axes = [np.linspace(float(bmin[i]), float(bmax[i]), res, dtype=np.float32) for i in range(3)]
    total = res**3
    occupancy = np.zeros(total, dtype=bool)
    height, width = masks.shape[1:]
    view_count = masks.shape[0]

    for start in range(0, total, int(chunk)):
        stop = min(total, start + int(chunk))
        idx = np.arange(start, stop, dtype=np.int64)
        ix = idx // (res * res)
        rem = idx - ix * res * res
        iy = rem // res
        iz = rem - iy * res
        points = np.stack((axes[0][ix], axes[1][iy], axes[2][iz]), axis=-1)
        inside_count = np.zeros(points.shape[0], dtype=np.int16)
        projected_count = np.zeros(points.shape[0], dtype=np.int16)
        for view_idx in range(view_count):
            u, v, z = project_world(points, extrinsic[view_idx], intrinsic[view_idx])
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            valid = (z > 1e-4) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            projected_count += valid.astype(np.int16)
            if np.any(valid):
                inside = np.zeros(points.shape[0], dtype=bool)
                valid_idx = np.nonzero(valid)[0]
                inside[valid_idx] = masks[view_idx, vi[valid_idx], ui[valid_idx]]
                inside_count += inside.astype(np.int16)
        keep = (projected_count >= int(min_projected_views)) & (inside_count >= int(min_inside_views))
        occupancy[start:stop] = keep

    occupancy3 = occupancy.reshape(res, res, res)
    stats = {
        "resolution": res,
        "voxels_total": int(total),
        "voxels_occupied": int(occupancy3.sum()),
        "occupied_ratio": float(occupancy3.mean()),
        "min_inside_views": int(min_inside_views),
        "min_projected_views": int(min_projected_views),
        "bbox_min": bmin.tolist(),
        "bbox_max": bmax.tolist(),
    }
    return occupancy3, stats


def rays_for_view(
    mask: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return yy, xx, empty, empty
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    dirs_cam = np.stack(
        (
            (xx.astype(np.float32) - cx) / max(abs(fx), 1e-6),
            (yy.astype(np.float32) - cy) / max(abs(fy), 1e-6),
            np.ones_like(xx, dtype=np.float32),
        ),
        axis=-1,
    )
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    dirs_world = dirs_cam @ rotation
    dirs_world /= np.clip(np.linalg.norm(dirs_world, axis=-1, keepdims=True), 1e-6, None)
    origin = camera_center_world(extrinsic)
    origins = np.broadcast_to(origin[None, :], dirs_world.shape).astype(np.float32)
    return yy, xx, origins, dirs_world.astype(np.float32)


def ray_box_intersection(
    origins: np.ndarray,
    dirs: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv = 1.0 / np.where(np.abs(dirs) < 1e-8, np.sign(dirs) * 1e-8 + (dirs == 0) * 1e-8, dirs)
    t0 = (bmin[None, :] - origins) * inv
    t1 = (bmax[None, :] - origins) * inv
    tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
    valid = tmax > np.maximum(tmin, 0.0)
    return np.maximum(tmin, 0.0).astype(np.float32), tmax.astype(np.float32), valid


def sample_occupancy(points: np.ndarray, occupancy: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    res = occupancy.shape[0]
    scaled = (points - bmin[None, :]) / np.clip((bmax - bmin)[None, :], 1e-8, None)
    idx = np.floor(scaled * float(res - 1)).astype(np.int32)
    valid = np.all((idx >= 0) & (idx < res), axis=1)
    out = np.zeros(points.shape[0], dtype=bool)
    if np.any(valid):
        ii = idx[valid]
        out[valid] = occupancy[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def raycast_view(
    mask: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    occupancy: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
    steps: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    yy, xx, origins, dirs = rays_for_view(mask, extrinsic, intrinsic)
    hit_mask = np.zeros(mask.shape, dtype=bool)
    hit_world = np.zeros((*mask.shape, 3), dtype=np.float32)
    hit_depth = np.zeros(mask.shape, dtype=np.float32)
    if yy.size == 0:
        return hit_mask, hit_world, hit_depth, {"mask_pixels": 0, "hit_pixels": 0, "hit_ratio": 0.0}

    for start in range(0, yy.size, int(chunk)):
        stop = min(yy.size, start + int(chunk))
        y_chunk = yy[start:stop]
        x_chunk = xx[start:stop]
        o = origins[start:stop]
        d = dirs[start:stop]
        near, far, valid_ray = ray_box_intersection(o, d, bmin, bmax)
        hit = np.zeros(stop - start, dtype=bool)
        first = np.zeros((stop - start, 3), dtype=np.float32)
        if np.any(valid_ray):
            active = valid_ray.copy()
            for step_idx in range(int(steps)):
                alpha = (float(step_idx) + 0.5) / float(max(steps, 1))
                t = near + (far - near) * alpha
                pts = o + d * t[:, None]
                occ = active & sample_occupancy(pts, occupancy, bmin, bmax)
                new_hit = occ & ~hit
                if np.any(new_hit):
                    first[new_hit] = pts[new_hit]
                    hit[new_hit] = True
                    active[new_hit] = False
                if not np.any(active):
                    break
        if np.any(hit):
            ys = y_chunk[hit]
            xs = x_chunk[hit]
            hit_world[ys, xs] = first[hit]
            cam = points_world_to_camera(hit_world, extrinsic)
            hit_depth[ys, xs] = np.maximum(1e-4, cam[ys, xs, 2])
            hit_mask[ys, xs] = True

    stats = {
        "mask_pixels": int(mask.sum()),
        "ray_pixels": int(yy.size),
        "hit_pixels": int(hit_mask.sum()),
        "hit_ratio": float(hit_mask.sum() / max(yy.size, 1)),
    }
    return hit_mask, hit_world, hit_depth, stats


def save_debug(path: Path, rgb: np.ndarray, mask: np.ndarray, hit: np.ndarray) -> None:
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * 0.72 + np.asarray([40, 90, 255], dtype=np.float32) * 0.28
    out[hit] = out[hit] * 0.42 + np.asarray([0, 220, 80], dtype=np.float32) * 0.58
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    with np.load(args.base_predictions, allow_pickle=False) as payload:
        base = {key: np.asarray(payload[key]) for key in payload.files}

    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    masks, rgbs = load_masks(args.scene_dir, view_count, (height, width), int(args.mask_dilate))
    bmin, bmax = bbox_from_base(world_points, masks, float(args.bbox_margin_frac))
    occupancy, hull_stats = build_visual_hull(
        bmin=bmin,
        bmax=bmax,
        masks=masks,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(args.resolution),
        min_inside_views=int(args.min_inside_views),
        min_projected_views=int(args.min_projected_views),
        chunk=int(args.voxel_chunk),
    )
    if int(hull_stats["voxels_occupied"]) <= 0:
        raise RuntimeError(f"visual hull is empty: {hull_stats}")

    depth2 = depth[..., 0].copy() if depth.ndim == 4 and depth.shape[-1] == 1 else depth.copy()
    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    per_view: dict[str, Any] = {}
    for view_idx in range(view_count):
        hit, hit_world, hit_depth, stats = raycast_view(
            masks[view_idx],
            extrinsic[view_idx],
            intrinsic[view_idx],
            occupancy,
            bmin,
            bmax,
            int(args.steps),
            int(args.ray_chunk),
        )
        replace = hit & masks[view_idx] if bool(args.replace_mask_only) else hit
        world_points[view_idx][replace] = hit_world[replace]
        depth2[view_idx][replace] = hit_depth[replace]
        if str(args.conf_mode) == "max":
            world_conf[view_idx][replace] = np.maximum(world_conf[view_idx][replace], float(args.conf_boost))
            depth_conf[view_idx][replace] = np.maximum(depth_conf[view_idx][replace], float(args.conf_boost))
            normal_conf[view_idx][replace] = np.maximum(normal_conf[view_idx][replace], 1.0)
        patch_mask[view_idx] = replace
        if bool(args.write_debug):
            save_debug(args.output_npz.parent / f"debug_view_{view_idx:02d}.png", rgbs[view_idx], masks[view_idx], replace)
        per_view[str(view_idx)] = {**stats, "replace_pixels": int(replace.sum())}

    normal, normal_update_count = update_predicted_normals(normal, world_points, extrinsic, patch_mask)
    out: dict[str, Any] = dict(base)
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth2[..., None].astype(np.float32) if depth.ndim == 4 and depth.shape[-1] == 1 else depth2.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    out["world_points_conf"] = world_conf.astype(np.float32)
    out["depth_conf"] = depth_conf.astype(np.float32)
    out["normal_conf"] = normal_conf.astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    depth_points = unproject_depth_map_to_point_map_numpy(out["depth"], extrinsic, intrinsic)
    agreement = np.linalg.norm(points_world_to_camera(world_points, extrinsic[:, :3, :]) - points_world_to_camera(depth_points, extrinsic[:, :3, :]), axis=-1)
    valid_agreement = patch_mask & np.isfinite(agreement)
    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_npz": str(args.output_npz.resolve()),
        "hull": hull_stats,
        "per_view": per_view,
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "conf_mode": str(args.conf_mode),
        "depth_world_l2_on_replaced": {
            "count": int(valid_agreement.sum()),
            "mean": float(np.mean(agreement[valid_agreement])) if np.any(valid_agreement) else 0.0,
            "p90": float(np.percentile(agreement[valid_agreement], 90.0)) if np.any(valid_agreement) else 0.0,
        },
        "truthful_status": "local_visual_hull_raycast_diagnostic_not_final_pass",
    }
    output_summary = args.output_summary if str(args.output_summary) and args.output_summary != Path(".") else args.output_npz.with_suffix(".json")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
