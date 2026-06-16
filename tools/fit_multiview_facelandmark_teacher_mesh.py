from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_mesh_raycast_training_case import _rays_for_pixels, _world_to_cam  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, preprocess_mask_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Triangulate a single shared 3D face mesh from multi-view MediaPipe landmarks. "
            "This is a teacher construction diagnostic; the output mesh must pass "
            "audit_headface_teacher_surface.py before any training."
        )
    )
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--face-landmarker-task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-views", default="all")
    parser.add_argument("--min-face-detection-confidence", type=float, default=0.02)
    parser.add_argument("--head-crop-pad", type=int, default=50)
    parser.add_argument("--min-landmark-views", type=int, default=4)
    parser.add_argument("--max-reproj-error-px", type=float, default=18.0)
    parser.add_argument("--max-robust-iters", type=int, default=4)
    parser.add_argument("--min-landmarks", type=int, default=160)
    parser.add_argument("--coarse-mesh", default="", help="Optional aligned coarse head mesh to merge with the face patch.")
    parser.add_argument("--coarse-face-replace-radius", type=float, default=0.035)
    parser.add_argument("--smooth-iterations", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
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


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "scene_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_scene_path(scene_dir: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = scene_dir / path
    if candidate.exists():
        return candidate
    return path


def parse_views(spec: str, count: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all" or not text:
        return list(range(count))
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= count:
            raise IndexError(f"view index {idx} outside [0,{count})")
        out.append(idx)
    return sorted(set(out))


def clamp_box(box: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    return x0, y0, x1, y1


def detect_landmarks(
    *,
    detector: vision.FaceLandmarker,
    image_path: Path,
    mask_path: Path,
    target_size: int,
    pad: int,
) -> tuple[np.ndarray | None, dict[str, Any], Image.Image]:
    image = Image.open(image_path).convert("RGB")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    mask = preprocess_mask_image(mask_path, target_size=target_size).astype(bool)
    head_box = head_box_from_mask(mask)
    meta: dict[str, Any] = {"detected": False, "head_box": None}
    if head_box is None:
        meta["reason"] = "no_head_box"
        return None, meta, image
    x0, y0, x1, y1 = head_box
    x0, y0, x1, y1 = clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), target_size, target_size)
    if x1 <= x0 + 16 or y1 <= y0 + 16:
        meta.update({"reason": "tiny_head_box", "head_box": [x0, y0, x1, y1]})
        return None, meta, image
    crop = image.crop((x0, y0, x1, y1)).resize((512, 512), Image.Resampling.BICUBIC)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop)))
    if not result.face_landmarks:
        meta.update({"reason": "no_facemesh", "head_box": [x0, y0, x1, y1]})
        return None, meta, image
    coords = []
    for lm in result.face_landmarks[0]:
        coords.append([x0 + float(lm.x) * (x1 - x0), y0 + float(lm.y) * (y1 - y0), float(lm.z)])
    coords_np = np.asarray(coords, dtype=np.float32)
    xi = np.clip(np.rint(coords_np[:, 0]).astype(np.int32), 0, target_size - 1)
    yi = np.clip(np.rint(coords_np[:, 1]).astype(np.int32), 0, target_size - 1)
    inside = mask[yi, xi]
    meta.update(
        {
            "detected": True,
            "head_box": [x0, y0, x1, y1],
            "landmarks": int(coords_np.shape[0]),
            "inside_mask": int(inside.sum()),
            "inside_mask_ratio": float(inside.mean()) if inside.size else 0.0,
        }
    )
    return coords_np, meta, image


def triangulate_rays(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    eye = np.eye(3, dtype=np.float64)
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins.astype(np.float64), directions.astype(np.float64)):
        d = direction / max(float(np.linalg.norm(direction)), 1e-9)
        proj = eye - np.outer(d, d)
        a += proj
        b += proj @ origin
    return np.linalg.solve(a + 1e-8 * eye, b).astype(np.float32)


def project_point(point: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[float, float, float]:
    cam = _world_to_cam(point.reshape(1, 3).astype(np.float32), extrinsic)[0]
    z = float(cam[2])
    u = float(intrinsic[0, 0] * cam[0] / max(z, 1e-6) + intrinsic[0, 2])
    v = float(intrinsic[1, 1] * cam[1] / max(z, 1e-6) + intrinsic[1, 2])
    return u, v, z


def robust_triangulate(
    observations: list[dict[str, Any]],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    min_views: int,
    max_reproj_error_px: float,
    max_iters: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if len(observations) < int(min_views):
        return None, {"status": "too_few_observations", "observations": len(observations)}
    active = np.ones(len(observations), dtype=bool)
    point: np.ndarray | None = None
    errors = np.full(len(observations), np.inf, dtype=np.float32)
    for _ in range(max(1, int(max_iters))):
        active_obs = [obs for keep, obs in zip(active, observations) if keep]
        if len(active_obs) < int(min_views):
            break
        xs = np.asarray([obs["x"] for obs in active_obs], dtype=np.float32)
        ys = np.asarray([obs["y"] for obs in active_obs], dtype=np.float32)
        view_ids = [int(obs["view_index"]) for obs in active_obs]
        rays = []
        for x, y, view_idx in zip(xs, ys, view_ids):
            rays.append(_rays_for_pixels(np.asarray([x], dtype=np.float32), np.asarray([y], dtype=np.float32), intrinsic[view_idx], extrinsic[view_idx])[0])
        rays_np = np.asarray(rays, dtype=np.float32)
        try:
            point = triangulate_rays(rays_np[:, :3], rays_np[:, 3:])
        except np.linalg.LinAlgError:
            return None, {"status": "singular", "observations": len(observations)}
        for obs_idx, obs in enumerate(observations):
            u, v, z = project_point(point, intrinsic[int(obs["view_index"])], extrinsic[int(obs["view_index"])])
            if z <= 0.05 or not np.isfinite([u, v, z]).all():
                errors[obs_idx] = np.inf
            else:
                errors[obs_idx] = float(np.hypot(u - float(obs["x"]), v - float(obs["y"])))
        threshold = max(float(max_reproj_error_px), float(np.nanmedian(errors[active]) * 2.5))
        new_active = errors <= threshold
        if int(new_active.sum()) < int(min_views):
            break
        if np.array_equal(new_active, active):
            active = new_active
            break
        active = new_active
    if point is None or int(active.sum()) < int(min_views):
        return None, {
            "status": "failed_after_robust",
            "observations": len(observations),
            "active": int(active.sum()),
            "median_error": float(np.nanmedian(errors[np.isfinite(errors)])) if np.isfinite(errors).any() else None,
        }
    final_errors = errors[active]
    if not final_errors.size or float(np.median(final_errors)) > float(max_reproj_error_px):
        return None, {
            "status": "high_reprojection_error",
            "observations": len(observations),
            "active": int(active.sum()),
            "median_error": float(np.median(final_errors)) if final_errors.size else None,
        }
    return point.astype(np.float32), {
        "status": "ok",
        "observations": len(observations),
        "active": int(active.sum()),
        "median_error": float(np.median(final_errors)),
        "p90_error": float(np.percentile(final_errors, 90)),
        "max_error": float(np.max(final_errors)),
        "active_views": [int(observations[idx]["view_index"]) for idx in np.nonzero(active)[0]],
    }


def mediapipe_triangles(valid: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        a = int(a)
        b = int(b)
        if a >= valid.shape[0] or b >= valid.shape[0] or not valid[a] or not valid[b]:
            continue
        if a > b:
            a, b = b, a
        edges.add((a, b))
    neighbors: dict[int, set[int]] = {}
    for a, b in edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    triangles = set()
    for a, b in edges:
        for c in neighbors.get(a, set()).intersection(neighbors.get(b, set())):
            tri = tuple(sorted((a, b, c)))
            if len(set(tri)) == 3:
                triangles.add(tri)
    return np.asarray(sorted(triangles), dtype=np.int32)


def save_detection_overlay(path: Path, image: Image.Image, coords: np.ndarray | None, meta: dict[str, Any]) -> None:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if meta.get("head_box") is not None:
        draw.rectangle(tuple(meta["head_box"]), outline=(255, 255, 0), width=2)
    if coords is not None:
        for x, y, _ in coords[::4]:
            r = 1
            draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r), fill=(0, 255, 60))
    draw.text((8, 8), f"detected={meta.get('detected')} inside={meta.get('inside_mask', 0)}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def write_mesh(points: np.ndarray, triangles: np.ndarray, output_path: Path, *, coarse_mesh: Path | None, replace_radius: float, smooth_iterations: int) -> dict[str, Any]:
    import open3d as o3d

    valid_vertex_ids = np.unique(triangles.reshape(-1))
    id_map = {int(old): idx for idx, old in enumerate(valid_vertex_ids.tolist())}
    compact_points = points[valid_vertex_ids].astype(np.float64)
    compact_triangles = np.asarray([[id_map[int(v)] for v in tri] for tri in triangles], dtype=np.int32)
    face_mesh = o3d.geometry.TriangleMesh()
    face_mesh.vertices = o3d.utility.Vector3dVector(compact_points)
    face_mesh.triangles = o3d.utility.Vector3iVector(compact_triangles)
    face_mesh.remove_degenerate_triangles()
    face_mesh.remove_duplicated_triangles()
    face_mesh.remove_duplicated_vertices()
    face_mesh.remove_non_manifold_edges()
    face_mesh.compute_vertex_normals()

    merged = face_mesh
    coarse_removed = 0
    if coarse_mesh is not None and coarse_mesh.is_file():
        coarse = o3d.io.read_triangle_mesh(str(coarse_mesh))
        if len(coarse.triangles):
            coarse.compute_vertex_normals()
            if replace_radius > 0 and len(face_mesh.vertices):
                face_pcd = o3d.geometry.PointCloud()
                face_pcd.points = face_mesh.vertices
                tree = o3d.geometry.KDTreeFlann(face_pcd)
                remove = []
                radius = float(replace_radius)
                for vertex in np.asarray(coarse.vertices):
                    count, _, _ = tree.search_radius_vector_3d(vertex, radius)
                    remove.append(count > 0)
                remove_np = np.asarray(remove, dtype=bool)
                coarse_removed = int(remove_np.sum())
                coarse.remove_vertices_by_mask(remove_np)
                coarse.remove_degenerate_triangles()
                coarse.remove_duplicated_triangles()
                coarse.remove_duplicated_vertices()
                coarse.remove_non_manifold_edges()
            merged = coarse + face_mesh
            merged.remove_degenerate_triangles()
            merged.remove_duplicated_triangles()
            merged.remove_duplicated_vertices()
            merged.remove_non_manifold_edges()
            merged.compute_vertex_normals()
    if smooth_iterations > 0 and len(merged.triangles):
        merged = merged.filter_smooth_simple(number_of_iterations=int(smooth_iterations))
        merged.compute_vertex_normals()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), merged, write_ascii=False, compressed=False)
    o3d.io.write_triangle_mesh(str(output_path.with_name("facelandmark_face_patch_only.ply")), face_mesh, write_ascii=False, compressed=False)
    return {
        "face_vertices": int(len(face_mesh.vertices)),
        "face_triangles": int(len(face_mesh.triangles)),
        "merged_vertices": int(len(merged.vertices)),
        "merged_triangles": int(len(merged.triangles)),
        "coarse_mesh": str(coarse_mesh) if coarse_mesh is not None else "",
        "coarse_vertices_removed_near_face": int(coarse_removed),
    }


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(scene_dir)
    with np.load(predictions_path, allow_pickle=False) as payload:
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float32)
    views = intrinsic.shape[0]
    view_indices = parse_views(str(args.target_views), views)
    target_size = 518

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(Path(args.face_landmarker_task).resolve())),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=float(args.min_face_detection_confidence),
        min_face_presence_confidence=float(args.min_face_detection_confidence),
    )
    observations: list[list[dict[str, Any]]] = [[] for _ in range(478)]
    detections: list[dict[str, Any]] = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for view_idx in view_indices:
            view = manifest["exported_views"][view_idx]
            image_path = resolve_scene_path(scene_dir, view["image_path"])
            mask_path = resolve_scene_path(scene_dir, view["mask_path"])
            coords, meta, image = detect_landmarks(
                detector=detector,
                image_path=image_path,
                mask_path=mask_path,
                target_size=target_size,
                pad=int(args.head_crop_pad),
            )
            meta.update({"view_index": int(view_idx), "camera_id": str(view.get("camera_id"))})
            detections.append(meta)
            save_detection_overlay(output_dir / "detections" / f"view{view_idx:02d}_{view.get('camera_id')}.png", image, coords, meta)
            if coords is None:
                continue
            for lm_idx, (x, y, z_rel) in enumerate(coords):
                if lm_idx >= len(observations):
                    continue
                if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < target_size and 0 <= y < target_size):
                    continue
                observations[lm_idx].append(
                    {
                        "view_index": int(view_idx),
                        "camera_id": str(view.get("camera_id")),
                        "x": float(x),
                        "y": float(y),
                        "relative_z": float(z_rel),
                    }
                )

    points = np.full((478, 3), np.nan, dtype=np.float32)
    valid = np.zeros(478, dtype=bool)
    landmark_rows = []
    for lm_idx, obs in enumerate(observations):
        point, row = robust_triangulate(
            obs,
            intrinsic,
            extrinsic,
            min_views=int(args.min_landmark_views),
            max_reproj_error_px=float(args.max_reproj_error_px),
            max_iters=int(args.max_robust_iters),
        )
        row["landmark_index"] = int(lm_idx)
        if point is not None:
            points[lm_idx] = point
            valid[lm_idx] = True
        landmark_rows.append(row)
    if int(valid.sum()) < int(args.min_landmarks):
        raise RuntimeError(f"Only {int(valid.sum())} landmarks triangulated; need {args.min_landmarks}.")
    triangles = mediapipe_triangles(valid)
    if triangles.size == 0:
        raise RuntimeError("No valid MediaPipe triangles after triangulation.")
    mesh_path = output_dir / "facelandmark_shared_surface.ply"
    mesh_summary = write_mesh(
        points,
        triangles,
        mesh_path,
        coarse_mesh=Path(args.coarse_mesh).expanduser().resolve() if str(args.coarse_mesh).strip() else None,
        replace_radius=float(args.coarse_face_replace_radius),
        smooth_iterations=int(args.smooth_iterations),
    )
    ok_rows = [row for row in landmark_rows if row.get("status") == "ok"]
    median_errors = [float(row["median_error"]) for row in ok_rows if row.get("median_error") is not None]
    summary = {
        "task": "fit_multiview_facelandmark_teacher_mesh",
        "truthful_status": "teacher_candidate_requires_strict_gate",
        "scene_dir": str(scene_dir),
        "predictions_npz": str(predictions_path),
        "face_landmarker_task": str(Path(args.face_landmarker_task).resolve()),
        "view_indices": view_indices,
        "detections": detections,
        "detected_views": int(sum(bool(row.get("detected")) for row in detections)),
        "triangulated_landmarks": int(valid.sum()),
        "valid_triangles": int(triangles.shape[0]),
        "reprojection_median_error_percentiles": [float(v) for v in np.percentile(median_errors, [0, 25, 50, 75, 90, 95, 100])] if median_errors else [],
        "mesh": mesh_summary,
        "outputs": {
            "mesh": str(mesh_path.resolve()),
            "face_patch_only": str(mesh_path.with_name("facelandmark_face_patch_only.ply").resolve()),
        },
        "next_required_gate": (
            "Render and audit this shared face mesh. It is not a teacher pass unless it passes "
            "the strict numeric gate and explicit Open3D visual review after projection to the original 6-view protocol."
        ),
    }
    (output_dir / "facelandmark_teacher_summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
