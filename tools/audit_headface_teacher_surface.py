from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_mesh_raycast_training_case import _rays_for_pixels, _roi_mask, _world_to_cam  # noqa: E402
from render_open3d_pointcloud import _save_open3d_renders, _save_projection_fallback  # noqa: E402
from vggt.utils.normal_refiner import preprocess_mask_image  # noqa: E402


ROI_KINDS = ("face_core", "face", "head", "head_face", "shoulder", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified gate for head/face teacher surfaces. It audits whether a "
            "teacher mesh, point cloud, or dense NPZ target can be projected "
            "back into the original sparse-view headshoulder protocol as a "
            "continuous, depth-compatible face/head surface."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher-mesh", default="", help="Aligned triangle mesh path.")
    source.add_argument("--teacher-pointcloud", default="", help="Aligned point cloud PLY path.")
    source.add_argument("--teacher-npz", default="", help="Dense teacher target NPZ with world_points/depths.")
    parser.add_argument("--predictions-npz", required=True, help="Reference predictions containing camera/depth.")
    parser.add_argument("--scene-dir", required=True, help="Target original sparse-view scene directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-views", default="3", help="Comma separated view indices or 'all'.")
    parser.add_argument("--roi-kinds", default="face_core,head_face", help="Comma separated ROI kinds.")
    parser.add_argument("--depth-tolerance", type=float, default=0.06)
    parser.add_argument("--max-hole-ratio", type=float, default=0.35)
    parser.add_argument("--min-coverage", type=float, default=0.58)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.78)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--max-median-depth-residual", type=float, default=0.025)
    parser.add_argument("--max-p90-depth-residual", type=float, default=0.055)
    parser.add_argument("--min-hit-pixels-face-core", type=int, default=4500)
    parser.add_argument("--min-hit-pixels-head-face", type=int, default=11000)
    parser.add_argument("--min-hit-pixels-head", type=int, default=9000)
    parser.add_argument("--min-hit-pixels-face", type=int, default=5000)
    parser.add_argument("--min-hit-pixels-default", type=int, default=4500)
    parser.add_argument("--point-splat-radius", type=int, default=1)
    parser.add_argument("--point-max-source-points", type=int, default=1200000)
    parser.add_argument("--npz-world-key", default="world_points")
    parser.add_argument("--npz-mask-key", default="teacher_mask")
    parser.add_argument("--npz-depth-key", default="depths")
    parser.add_argument("--render-width", type=int, default=1200)
    parser.add_argument("--render-height", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--skip-open3d", action="store_true")
    return parser.parse_args()


def parse_indices(spec: str, count: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all":
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


def parse_rois(spec: str) -> list[str]:
    out: list[str] = []
    for item in str(spec).split(","):
        roi = item.strip()
        if not roi:
            continue
        if roi not in ROI_KINDS:
            raise ValueError(f"unknown ROI kind {roi!r}; choices={ROI_KINDS}")
        out.append(roi)
    if not out:
        raise ValueError("empty --roi-kinds")
    return out


def load_scene_image(scene_dir: Path, view_index: int, target_size: int) -> np.ndarray:
    paths = sorted((scene_dir / "images").glob("*"))
    if not paths:
        raise FileNotFoundError(scene_dir / "images")
    image = Image.open(paths[view_index]).convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def load_scene_mask(scene_dir: Path, view_index: int, target_size: int) -> np.ndarray:
    paths = sorted((scene_dir / "masks").glob("*"))
    if not paths:
        raise FileNotFoundError(scene_dir / "masks")
    return preprocess_mask_image(paths[view_index], target_size=target_size).astype(bool)


def connected_component_stats(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return {"components": 0, "largest_component_pixels": 0, "largest_component_ratio": 0.0}
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = 0
    components = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        components += 1
        count = 0
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        visited[sy, sx] = True
        while queue:
            y, x = queue.popleft()
            count += 1
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        largest = max(largest, count)
    return {
        "components": int(components),
        "largest_component_pixels": int(largest),
        "largest_component_ratio": float(largest / max(total, 1)),
    }


def percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99)}


def min_hit_pixels_for_roi(args: argparse.Namespace, roi: str) -> int:
    if roi == "face_core":
        return int(args.min_hit_pixels_face_core)
    if roi == "head_face":
        return int(args.min_hit_pixels_head_face)
    if roi == "head":
        return int(args.min_hit_pixels_head)
    if roi == "face":
        return int(args.min_hit_pixels_face)
    return int(args.min_hit_pixels_default)


def overlay_rgb(rgb: np.ndarray, roi: np.ndarray, raw: np.ndarray, depth_ok: np.ndarray) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    roi_only = roi & ~raw
    raw_only = raw & ~depth_ok
    out[roi_only] = 0.62 * out[roi_only] + np.array([30, 144, 255], dtype=np.float32) * 0.38
    out[raw_only] = 0.58 * out[raw_only] + np.array([255, 210, 0], dtype=np.float32) * 0.42
    out[depth_ok] = 0.48 * out[depth_ok] + np.array([0, 220, 70], dtype=np.float32) * 0.52
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "blue=missing ROI, yellow=teacher hit, green=depth-compatible", fill=(0, 0, 0))
    return image


def save_bool_mask(path: Path, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(canvas).save(path)


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return (points_cam - translation[None, :]) @ rotation


def unproject_depth(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    cam = np.stack(((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), axis=-1)
    flat = cam.reshape(-1, 3)
    return camera_to_world(flat, extrinsic).reshape(height, width, 3).astype(np.float32)


class TeacherSource:
    def query(
        self,
        *,
        view_index: int,
        roi: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        anchor_depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class MeshTeacher(TeacherSource):
    def __init__(self, path: Path) -> None:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise ValueError(f"teacher mesh has no triangles: {path}")
        mesh.compute_vertex_normals()
        self._o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def query(
        self,
        *,
        view_index: int,
        roi: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        anchor_depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rays = _rays_for_pixels(xs, ys, intrinsic, extrinsic)
        answers = self.scene.cast_rays(self._o3d.core.Tensor(rays, dtype=self._o3d.core.Dtype.Float32))
        t_hit = answers["t_hit"].numpy()
        raw_valid = np.isfinite(t_hit)
        origins = rays[:, :3]
        directions = rays[:, 3:]
        world_hit = origins + directions * np.where(raw_valid, t_hit, 0.0)[:, None]
        cam_hit = _world_to_cam(world_hit.astype(np.float32), extrinsic)
        hit_depth = cam_hit[:, 2]
        return raw_valid, hit_depth.astype(np.float32), world_hit.astype(np.float32)


class PointCloudTeacher(TeacherSource):
    def __init__(self, path: Path, *, max_points: int, splat_radius: int) -> None:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if points.size == 0:
            raise ValueError(f"teacher point cloud has no points: {path}")
        finite = np.isfinite(points).all(axis=-1)
        points = points[finite]
        colors = colors[finite] if colors.shape[0] == finite.shape[0] else np.zeros((points.shape[0], 3), dtype=np.float32)
        if points.shape[0] > int(max_points):
            rng = np.random.default_rng(20260501)
            keep = rng.choice(points.shape[0], size=int(max_points), replace=False)
            points = points[keep]
            colors = colors[keep]
        self.points = points
        self.colors = colors
        self.splat_radius = int(max(0, splat_radius))

    def query(
        self,
        *,
        view_index: int,
        roi: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        anchor_depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = _world_to_cam(self.points, extrinsic)
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=-1) & (z > 0.05)
        cam = cam[valid]
        source_points = self.points[valid]
        if cam.shape[0] == 0:
            return np.zeros(xs.shape, dtype=bool), np.zeros(xs.shape, dtype=np.float32), np.zeros((xs.size, 3), dtype=np.float32)
        u = np.round(cam[:, 0] * float(intrinsic[0, 0]) / np.maximum(cam[:, 2], 1e-6) + float(intrinsic[0, 2])).astype(np.int32)
        v = np.round(cam[:, 1] * float(intrinsic[1, 1]) / np.maximum(cam[:, 2], 1e-6) + float(intrinsic[1, 2])).astype(np.int32)
        height, width = roi.shape
        zbuffer = np.full((height, width), np.inf, dtype=np.float32)
        index_map = np.full((height, width), -1, dtype=np.int64)
        for offset_y in range(-self.splat_radius, self.splat_radius + 1):
            for offset_x in range(-self.splat_radius, self.splat_radius + 1):
                uu = u + offset_x
                vv = v + offset_y
                inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                if not np.any(inside):
                    continue
                order = np.argsort(cam[inside, 2])
                src_ids = np.nonzero(inside)[0][order]
                for src_idx in src_ids:
                    py = int(vv[src_idx])
                    px = int(uu[src_idx])
                    depth = float(cam[src_idx, 2])
                    if depth < zbuffer[py, px]:
                        zbuffer[py, px] = depth
                        index_map[py, px] = src_idx
        raw = index_map[ys, xs] >= 0
        hit_depth = np.where(raw, zbuffer[ys, xs], 0.0).astype(np.float32)
        world = np.zeros((xs.size, 3), dtype=np.float32)
        valid_ids = index_map[ys[raw], xs[raw]].astype(np.int64)
        world[raw] = source_points[valid_ids]
        return raw, hit_depth, world


class DenseNpzTeacher(TeacherSource):
    def __init__(self, path: Path, *, world_key: str, mask_key: str, depth_key: str, reference_intrinsic: np.ndarray, reference_extrinsic: np.ndarray) -> None:
        payload = np.load(path, allow_pickle=False)
        self.world_points = None
        self.depth = None
        self.mask = None
        if world_key in payload.files:
            self.world_points = np.asarray(payload[world_key], dtype=np.float32)
        elif "world_points" in payload.files:
            self.world_points = np.asarray(payload["world_points"], dtype=np.float32)
        if depth_key in payload.files:
            self.depth = np.asarray(payload[depth_key], dtype=np.float32)
        elif "depth" in payload.files:
            self.depth = np.asarray(payload["depth"], dtype=np.float32)
            if self.depth.ndim == 4 and self.depth.shape[-1] == 1:
                self.depth = self.depth[..., 0]
        elif "depths" in payload.files:
            self.depth = np.asarray(payload["depths"], dtype=np.float32)
        if mask_key in payload.files:
            self.mask = np.asarray(payload[mask_key], dtype=bool)
        elif "teacher_mask" in payload.files:
            self.mask = np.asarray(payload["teacher_mask"], dtype=bool)
        elif "roi_mask" in payload.files:
            self.mask = np.asarray(payload["roi_mask"], dtype=bool)
        if self.world_points is None and self.depth is None:
            raise ValueError(f"{path} has neither world points nor depth-like key")
        if self.world_points is None and self.depth is not None:
            worlds = []
            for view_idx in range(self.depth.shape[0]):
                worlds.append(unproject_depth(self.depth[view_idx], reference_intrinsic[view_idx], reference_extrinsic[view_idx]))
            self.world_points = np.stack(worlds, axis=0)
        if self.depth is None and self.world_points is not None:
            depths = []
            for view_idx in range(self.world_points.shape[0]):
                cam = _world_to_cam(self.world_points[view_idx].reshape(-1, 3), reference_extrinsic[view_idx])
                depths.append(cam[:, 2].reshape(self.world_points.shape[1:3]))
            self.depth = np.stack(depths, axis=0).astype(np.float32)
        if self.mask is None:
            self.mask = np.isfinite(self.world_points).all(axis=-1) & np.isfinite(self.depth) & (self.depth > 0.05)

    def query(
        self,
        *,
        view_index: int,
        roi: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        anchor_depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if view_index >= self.world_points.shape[0]:
            return np.zeros(xs.shape, dtype=bool), np.zeros(xs.shape, dtype=np.float32), np.zeros((xs.size, 3), dtype=np.float32)
        raw = self.mask[view_index, ys, xs] & np.isfinite(self.depth[view_index, ys, xs])
        hit_depth = np.where(raw, self.depth[view_index, ys, xs], 0.0).astype(np.float32)
        world = np.zeros((xs.size, 3), dtype=np.float32)
        world[raw] = self.world_points[view_index, ys[raw], xs[raw]]
        return raw, hit_depth, world


def build_teacher_source(args: argparse.Namespace, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[str, TeacherSource, str]:
    if args.teacher_mesh:
        path = Path(args.teacher_mesh).resolve()
        return "mesh", MeshTeacher(path), str(path)
    if args.teacher_pointcloud:
        path = Path(args.teacher_pointcloud).resolve()
        return "pointcloud", PointCloudTeacher(
            path,
            max_points=int(args.point_max_source_points),
            splat_radius=int(args.point_splat_radius),
        ), str(path)
    path = Path(args.teacher_npz).resolve()
    return "npz", DenseNpzTeacher(
        path,
        world_key=str(args.npz_world_key),
        mask_key=str(args.npz_mask_key),
        depth_key=str(args.npz_depth_key),
        reference_intrinsic=intrinsics,
        reference_extrinsic=extrinsics,
    ), str(path)


def audit_one(
    *,
    args: argparse.Namespace,
    source: TeacherSource,
    source_kind: str,
    scene_dir: Path,
    output_dir: Path,
    view_index: int,
    roi_kind: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    anchor_depth: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    roi = _roi_mask(mask, roi_kind)
    ys, xs = np.nonzero(roi)
    prefix = f"view{view_index:02d}_{roi_kind}"
    if xs.size == 0:
        summary = {
            "view_index": view_index,
            "roi_kind": roi_kind,
            "roi_pixels": 0,
            "gate": {"pass": False, "reason": "empty ROI"},
        }
        return summary, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    raw_valid, hit_depth, world_hit = source.query(
        view_index=view_index,
        roi=roi,
        xs=xs,
        ys=ys,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        anchor_depth=anchor_depth,
    )
    residual = np.abs(hit_depth - anchor_depth[ys, xs])
    depth_ok_flat = raw_valid & (hit_depth > 0.05) & np.isfinite(residual) & (residual <= float(args.depth_tolerance))
    raw_mask = np.zeros_like(roi, dtype=bool)
    depth_ok_mask = np.zeros_like(roi, dtype=bool)
    raw_mask[ys[raw_valid], xs[raw_valid]] = True
    depth_ok_mask[ys[depth_ok_flat], xs[depth_ok_flat]] = True
    missing_mask = roi & ~depth_ok_mask
    raw_components = connected_component_stats(raw_mask)
    ok_components = connected_component_stats(depth_ok_mask)
    roi_pixels = int(roi.sum())
    raw_hit_pixels = int(raw_mask.sum())
    ok_hit_pixels = int(depth_ok_mask.sum())
    residual_stats = percentiles(residual[depth_ok_flat])
    median_residual = residual_stats["p50"]
    p90_residual = residual_stats["p90"]
    min_hits = min_hit_pixels_for_roi(args, roi_kind)
    gate = {
        "hit_pixels": ok_hit_pixels >= min_hits,
        "coverage": ok_hit_pixels / max(roi_pixels, 1) >= float(args.min_coverage),
        "hole_ratio": (1.0 - ok_hit_pixels / max(roi_pixels, 1)) <= float(args.max_hole_ratio),
        "largest_component_ratio": ok_components["largest_component_ratio"] >= float(args.min_largest_component_ratio),
        "fragment_count": ok_components["components"] <= int(args.max_components),
        "median_depth_residual": median_residual is not None and float(median_residual) <= float(args.max_median_depth_residual),
        "p90_depth_residual": p90_residual is not None and float(p90_residual) <= float(args.max_p90_depth_residual),
    }
    gate["pass"] = bool(all(gate.values()))

    overlay_path = output_dir / f"{prefix}_overlay_rgb_depth_compat.png"
    coverage_path = output_dir / f"{prefix}_coverage_mask.png"
    missing_path = output_dir / f"{prefix}_missing_mask.png"
    overlay_rgb(rgb, roi=roi, raw=raw_mask, depth_ok=depth_ok_mask).save(overlay_path)
    save_bool_mask(coverage_path, depth_ok_mask, (0, 220, 70))
    save_bool_mask(missing_path, missing_mask, (30, 144, 255))

    points = world_hit[depth_ok_flat]
    colors = rgb[ys[depth_ok_flat], xs[depth_ok_flat]]
    summary = {
        "view_index": int(view_index),
        "roi_kind": roi_kind,
        "source_kind": source_kind,
        "roi_pixels": roi_pixels,
        "raw_visible": {
            "hit_pixels": raw_hit_pixels,
            "coverage": float(raw_hit_pixels / max(roi_pixels, 1)),
            "hole_ratio": float(1.0 - raw_hit_pixels / max(roi_pixels, 1)),
            "components": raw_components,
            "depth_residual": percentiles(residual[raw_valid]),
        },
        "depth_compatible": {
            "hit_pixels": ok_hit_pixels,
            "coverage": float(ok_hit_pixels / max(roi_pixels, 1)),
            "hole_ratio": float(1.0 - ok_hit_pixels / max(roi_pixels, 1)),
            "components": ok_components,
            "depth_residual": residual_stats,
        },
        "gate_thresholds": {
            "min_hit_pixels": min_hits,
            "min_coverage": float(args.min_coverage),
            "max_hole_ratio": float(args.max_hole_ratio),
            "min_largest_component_ratio": float(args.min_largest_component_ratio),
            "max_components": int(args.max_components),
            "max_median_depth_residual": float(args.max_median_depth_residual),
            "max_p90_depth_residual": float(args.max_p90_depth_residual),
        },
        "gate": gate,
        "artifacts": {
            "overlay_rgb_depth_compat": str(overlay_path),
            "coverage_mask": str(coverage_path),
            "missing_mask": str(missing_path),
        },
    }
    return summary, points.astype(np.float32), colors.astype(np.uint8)


def render_teacher_points(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    roi_kind: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        return {"pass": False, "reason": "no teacher points to render", "output_dir": str(output_dir)}
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    ply_path = output_dir / "teacher_depth_compatible_points.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)
    render_roi = "face" if roi_kind in {"face_core", "face"} else ("head" if roi_kind in {"head", "head_face"} else "full")
    try:
        screenshots = [] if args.skip_open3d else _save_open3d_renders(
            points=points,
            colors=colors,
            output_dir=output_dir,
            roi=render_roi,
            width=int(args.render_width),
            height=int(args.render_height),
            point_size=float(args.point_size),
            interactive=False,
        )
        backend = "open3d_visualizer"
    except Exception as exc:
        screenshots = _save_projection_fallback(
            points=points,
            colors=colors,
            output_dir=output_dir,
            roi=render_roi,
            width=int(args.render_width),
            height=int(args.render_height),
        )
        backend = f"projection_fallback_after_{type(exc).__name__}: {exc}"
    return {
        "pass": bool(ply_path.exists() and (args.skip_open3d or screenshots)),
        "output_dir": str(output_dir),
        "ply_path": str(ply_path),
        "screenshots": screenshots,
        "render_backend": backend,
        "points_rendered": int(points.shape[0]),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Head/Face Teacher Gate: {summary['source_kind']}",
        "",
        f"- Source: `{summary['source_path']}`",
        f"- Scene: `{summary['scene_dir']}`",
        f"- Reference predictions: `{summary['predictions_npz']}`",
        f"- Overall pass: `{summary['gate']['pass']}`",
        "",
        "| View | ROI | Coverage | Hole | LCC | Fragments | Median residual | P90 residual | Hit pixels | Pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["entries"]:
        comp = row["depth_compatible"]["components"]
        residual = row["depth_compatible"]["depth_residual"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["view_index"]),
                    str(row["roi_kind"]),
                    f"{float(row['depth_compatible']['coverage']):.4f}",
                    f"{float(row['depth_compatible']['hole_ratio']):.4f}",
                    f"{float(comp['largest_component_ratio']):.4f}",
                    str(comp["components"]),
                    "NA" if residual["p50"] is None else f"{float(residual['p50']):.5f}",
                    "NA" if residual["p90"] is None else f"{float(residual['p90']):.5f}",
                    str(row["depth_compatible"]["hit_pixels"]),
                    str(row["gate"]["pass"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Reasons", ""])
    failures = [row for row in summary["entries"] if not row["gate"]["pass"]]
    if failures:
        for row in failures:
            failed_checks = [key for key, ok in row["gate"].items() if key != "pass" and not ok]
            lines.append(f"- view {row['view_index']} / {row['roi_kind']}: {', '.join(failed_checks)}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Truthful Note",
            "",
            "A pass here only means the teacher is eligible for local smoke training. It is not a mentor-final VGGT result and it does not permit cloud upload by itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_npz).resolve()
    scene_dir = Path(args.scene_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(predictions_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
        depth = np.asarray(payload["depth"], dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
    view_count, height, width = depth.shape
    views = parse_indices(str(args.target_views), view_count)
    rois = parse_rois(str(args.roi_kinds))
    source_kind, teacher, source_path = build_teacher_source(args, intrinsics, extrinsics)

    entries: list[dict[str, Any]] = []
    points_by_roi: dict[str, list[np.ndarray]] = {roi: [] for roi in rois}
    colors_by_roi: dict[str, list[np.ndarray]] = {roi: [] for roi in rois}
    for view_index in views:
        rgb = load_scene_image(scene_dir, view_index=view_index, target_size=height)
        mask = load_scene_mask(scene_dir, view_index=view_index, target_size=height)
        for roi_kind in rois:
            entry, points, colors = audit_one(
                args=args,
                source=teacher,
                source_kind=source_kind,
                scene_dir=scene_dir,
                output_dir=output_dir,
                view_index=view_index,
                roi_kind=roi_kind,
                rgb=rgb,
                mask=mask,
                intrinsic=intrinsics[view_index],
                extrinsic=extrinsics[view_index],
                anchor_depth=depth[view_index],
            )
            entries.append(entry)
            if points.shape[0]:
                points_by_roi[roi_kind].append(points)
                colors_by_roi[roi_kind].append(colors)

    render_outputs: dict[str, Any] = {}
    for roi_kind in rois:
        points = np.concatenate(points_by_roi[roi_kind], axis=0) if points_by_roi[roi_kind] else np.zeros((0, 3), dtype=np.float32)
        colors = np.concatenate(colors_by_roi[roi_kind], axis=0) if colors_by_roi[roi_kind] else np.zeros((0, 3), dtype=np.uint8)
        render_outputs[roi_kind] = render_teacher_points(
            points=points,
            colors=colors,
            output_dir=output_dir / f"open3d_teacher_{roi_kind}",
            roi_kind=roi_kind,
            args=args,
        )

    overall_pass = bool(entries and all(row["gate"].get("pass") for row in entries))
    summary = {
        "source_kind": source_kind,
        "source_path": source_path,
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "target_views": views,
        "roi_kinds": rois,
        "depth_tolerance": float(args.depth_tolerance),
        "gate": {"pass": overall_pass},
        "entries": entries,
        "render_outputs": render_outputs,
        "truthful_note": (
            "Teacher gate only. PASS allows a local one-frame ROI overfit smoke; "
            "it is not a model result and never permits cloud upload by itself."
        ),
    }
    (output_dir / "teacher_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "teacher_gate_summary.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
