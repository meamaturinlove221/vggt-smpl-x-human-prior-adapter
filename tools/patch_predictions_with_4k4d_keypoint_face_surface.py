from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


# 4K4D annotation layout for seq 0012_11 frame 0:
# Keypoints_2D has 131 points, with a 68-point face block at [63, 131).
# Keypoints_3D has the corresponding face block shifted by +13: [76, 144).
FACE_2D = np.arange(63, 131, dtype=np.int64)
FACE_3D = np.arange(76, 144, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch predictions.npz with a local 4K4D keypoint-derived face surface. "
            "This is a diagnostic local geometry bridge: it uses 4K4D 2D/3D "
            "keypoints to create a dense face surface in VGGT world coordinates."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-views", default="all", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--patch-views", default="all", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--min-conf", type=float, default=5.0)
    parser.add_argument("--confidence-boost", type=float, default=220.0)
    parser.add_argument("--edge-shrink", type=float, default=0.0, help="Unused reserve knob; kept for report provenance.")
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


def parse_view_set(spec: str, view_count: int) -> set[int]:
    text = str(spec).strip().lower()
    if not text or text == "all":
        return set(range(view_count))
    out: set[int] = set()
    for part in text.split(","):
        if not part.strip():
            continue
        idx = int(part)
        if idx < 0 or idx >= view_count:
            raise IndexError(f"view index {idx} outside [0,{view_count})")
        out.add(idx)
    return out


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError(f"bad similarity inputs: src={src.shape} dst={dst.shape}")
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
        if active.size < 8:
            break
        threshold = float(np.median(active) + 3.0 * np.median(np.abs(active - np.median(active)) + 1e-8))
        threshold = max(threshold, float(np.percentile(active, 80.0)))
        new_keep = residual <= threshold
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    transform = umeyama_similarity(src[keep], dst[keep])
    residual = np.linalg.norm(apply_similarity(src, transform) - dst, axis=1)
    return transform, keep, residual


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3].astype(np.float32)
    translation = extrinsic[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def keypoints_2d_for_view(ann: h5py.File, camera_id: str, view: dict[str, Any]) -> np.ndarray:
    keypoints = np.asarray(ann[f"Keypoints_2D/{camera_id}"][0], dtype=np.float32)
    height, width = view["source_image_size"]
    out = keypoints.copy()
    out[:, 0] = out[:, 0] / float(width) * 518.0
    out[:, 1] = out[:, 1] / float(height) * 518.0
    return out


def collect_fit_pairs(
    *,
    ann: h5py.File,
    manifest: dict[str, Any],
    keypoints3d: np.ndarray,
    world_points: np.ndarray,
    world_conf: np.ndarray,
    fit_views: set[int],
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for view_idx, view in enumerate(manifest["exported_views"]):
        if view_idx not in fit_views:
            continue
        camera_id = str(view["camera_id"])
        keypoints2d = keypoints_2d_for_view(ann, camera_id, view)
        mask = np.asarray(Image.open(view["mask_path"]).convert("L")) > 127
        for k2d_idx, k3d_idx in zip(FACE_2D, FACE_3D):
            u_f, v_f, score = keypoints2d[int(k2d_idx)]
            u = int(round(float(u_f)))
            v = int(round(float(v_f)))
            if score <= 0.0 or not (0 <= u < 518 and 0 <= v < 518):
                continue
            if not mask[v, u] or float(world_conf[view_idx, v, u]) < float(min_conf):
                continue
            point = world_points[view_idx, v, u]
            if not np.isfinite(point).all():
                continue
            src.append(keypoints3d[int(k3d_idx)])
            dst.append(point.astype(np.float32))
            rows.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": camera_id,
                    "k2d_index": int(k2d_idx),
                    "k3d_index": int(k3d_idx),
                    "u": int(u),
                    "v": int(v),
                    "score": float(score),
                    "world_conf": float(world_conf[view_idx, v, u]),
                }
            )
    if len(src) < 8:
        raise RuntimeError(f"too few face keypoint fit pairs: {len(src)}")
    return np.asarray(src, dtype=np.float32), np.asarray(dst, dtype=np.float32), rows


def rasterize_keypoint_surface(
    uv: np.ndarray,
    points3d: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(points3d).all(axis=1)
    uv = uv[valid].astype(np.float32)
    points3d = points3d[valid].astype(np.float32)
    if uv.shape[0] < 8:
        return np.zeros((*support.shape, 3), dtype=np.float32), np.zeros(support.shape, dtype=bool)

    triangulation = Delaunay(uv)
    x0 = max(0, int(np.floor(float(np.min(uv[:, 0])))) - 2)
    x1 = min(support.shape[1] - 1, int(np.ceil(float(np.max(uv[:, 0])))) + 2)
    y0 = max(0, int(np.floor(float(np.min(uv[:, 1])))) - 2)
    y1 = min(support.shape[0] - 1, int(np.ceil(float(np.max(uv[:, 1])))) + 2)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((*support.shape, 3), dtype=np.float32), np.zeros(support.shape, dtype=bool)

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    query = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    simplex = triangulation.find_simplex(query)
    inside = simplex >= 0
    if not inside.any():
        return np.zeros((*support.shape, 3), dtype=np.float32), np.zeros(support.shape, dtype=bool)

    transform = triangulation.transform[simplex[inside], :2]
    delta = query[inside] - triangulation.transform[simplex[inside], 2]
    bary12 = np.einsum("nij,nj->ni", transform, delta)
    bary = np.c_[bary12, 1.0 - bary12.sum(axis=1)]
    vertices = triangulation.simplices[simplex[inside]]
    interp = np.einsum("ni,nij->nj", bary, points3d[vertices])

    surface = np.zeros((*support.shape, 3), dtype=np.float32)
    mask = np.zeros(support.shape, dtype=bool)
    q_inside = query[inside].astype(np.int32)
    surface[q_inside[:, 1], q_inside[:, 0]] = interp.astype(np.float32)
    mask[q_inside[:, 1], q_inside[:, 0]] = True
    mask &= support
    return surface, mask


def save_overlay(
    path: Path,
    image_path: str,
    keypoints_uv: np.ndarray,
    patched_mask: np.ndarray,
    view_meta: dict[str, Any],
) -> None:
    image = Image.open(image_path).convert("RGB")
    if image.size != (518, 518):
        image = image.resize((518, 518), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    arr[patched_mask] = arr[patched_mask] * 0.55 + np.asarray([50, 110, 255], dtype=np.float32) * 0.45
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for idx, (u, v) in enumerate(keypoints_uv):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        draw.ellipse((float(u) - 1.5, float(v) - 1.5, float(u) + 1.5, float(v) + 1.5), fill=(255, 230, 0))
        if idx % 12 == 0:
            draw.text((float(u) + 2.0, float(v) + 2.0), str(idx), fill=(255, 255, 0))
    draw.text((8, 8), f"patched={int(patched_mask.sum())} cam={view_meta.get('camera_id')}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    scene_dir = args.scene_dir.resolve()
    manifest = json.loads((scene_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    annotations_smc = Path(manifest["annotations_smc"])

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

    view_count = int(world_points.shape[0])
    fit_views = parse_view_set(str(args.fit_views), view_count)
    patch_views = parse_view_set(str(args.patch_views), view_count)

    patched_mask = np.zeros(depth2.shape, dtype=bool)
    per_view: list[dict[str, Any]] = []
    with h5py.File(str(annotations_smc), "r") as ann:
        keypoints3d = np.asarray(ann["Keypoints_3D/keypoints3d"][int(manifest["frame_id"]), :, :3], dtype=np.float32)
        src, dst, fit_rows = collect_fit_pairs(
            ann=ann,
            manifest=manifest,
            keypoints3d=keypoints3d,
            world_points=world_points,
            world_conf=world_conf,
            fit_views=fit_views,
            min_conf=float(args.min_conf),
        )
        transform, inlier_mask, residual = robust_similarity(src, dst)
        transformed_face = apply_similarity(keypoints3d[FACE_3D], transform)

        for view_idx, view in enumerate(manifest["exported_views"]):
            camera_id = str(view["camera_id"])
            keypoints2d = keypoints_2d_for_view(ann, camera_id, view)
            face_uv = keypoints2d[FACE_2D, :2]
            support = np.asarray(Image.open(view["mask_path"]).convert("L")) > 127
            if view_idx not in patch_views:
                per_view.append({"view_index": int(view_idx), "camera_id": camera_id, "skipped": True})
                continue
            surface, surface_mask = rasterize_keypoint_surface(face_uv, transformed_face, support)
            if surface_mask.any():
                world_points[view_idx, surface_mask] = surface[surface_mask]
                cam = (
                    np.einsum("ij,hwj->hwi", extrinsic[view_idx, :3, :3], world_points[view_idx])
                    + extrinsic[view_idx, :3, 3][None, None, :]
                )
                z = cam[..., 2]
                valid_z = surface_mask & np.isfinite(z) & (z > 1e-4)
                depth2[view_idx, valid_z] = z[valid_z]
                world_conf[view_idx, valid_z] = np.maximum(world_conf[view_idx, valid_z], float(args.confidence_boost))
                depth_conf[view_idx, valid_z] = np.maximum(depth_conf[view_idx, valid_z], float(args.confidence_boost))
                normal_conf[view_idx, valid_z] = np.maximum(normal_conf[view_idx, valid_z], 1.0)
                patched_mask[view_idx, valid_z] = True
                per_view.append(
                    {
                        "view_index": int(view_idx),
                        "camera_id": camera_id,
                        "patched_pixels": int(valid_z.sum()),
                        "face_uv_bbox": [
                            float(np.nanmin(face_uv[:, 0])),
                            float(np.nanmin(face_uv[:, 1])),
                            float(np.nanmax(face_uv[:, 0])),
                            float(np.nanmax(face_uv[:, 1])),
                        ],
                    }
                )
            else:
                per_view.append({"view_index": int(view_idx), "camera_id": camera_id, "patched_pixels": 0})
            save_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_keypoint_face_surface.png",
                str(view["image_path"]),
                face_uv,
                patched_mask[view_idx],
                view,
            )

    cam_points = world_to_camera(world_points, extrinsic)
    for view_idx in range(view_count):
        if not patched_mask[view_idx].any():
            continue
        normal_map, valid = point_map_to_normal_numpy(cam_points[view_idx], patched_mask[view_idx])
        use = valid & patched_mask[view_idx]
        if use.any():
            old = normal[view_idx]
            dot = np.sum(old[use] * normal_map[use], axis=-1)
            if dot.size and float(np.nanmean(dot)) < 0.0:
                normal_map = -normal_map
            normal[view_idx, use] = normal_map[use]

    out = dict(base)
    out["world_points"] = world_points.astype(base["world_points"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth"] = depth2[..., None].astype(base["depth"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["normal"] = normal.astype(base["normal"].dtype, copy=False)
    out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)
    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)

    transformed_bbox = {
        "min": np.min(transformed_face, axis=0),
        "max": np.max(transformed_face, axis=0),
        "extent": np.max(transformed_face, axis=0) - np.min(transformed_face, axis=0),
    }
    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(scene_dir),
        "annotations_smc": str(annotations_smc),
        "output_predictions": str(output_path),
        "fit_views": sorted(int(v) for v in fit_views),
        "patch_views": sorted(int(v) for v in patch_views),
        "fit_pairs": int(src.shape[0]),
        "fit_inliers": int(inlier_mask.sum()),
        "fit_residual_median": float(np.median(residual)),
        "fit_residual_p90": float(np.percentile(residual, 90.0)),
        "similarity_scale": float(transform[0]),
        "face_2d_indices": [int(FACE_2D[0]), int(FACE_2D[-1])],
        "face_3d_indices": [int(FACE_3D[0]), int(FACE_3D[-1])],
        "transformed_face_bbox": json_ready(transformed_bbox),
        "patched_pixels": int(patched_mask.sum()),
        "per_view": per_view,
        "fit_rows_sample": fit_rows[:12],
        "truthful_status": "local_4k4d_keypoint_face_surface_diagnostic_not_training_result",
    }
    (output_dir / "keypoint_face_surface_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
