from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
OUT = ROOT / "output" / "mentor_report_v50r2" / "open3d_clear_human"
IMG = OUT / "images"
PLY = OUT / "ply"
REPORTS = ROOT / "reports"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_npz(name: str) -> dict[str, np.ndarray]:
    with np.load(PKG / name, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def robust_points(points: np.ndarray, mask: np.ndarray | None = None, stride: int = 1) -> np.ndarray:
    pts = np.asarray(points)
    if mask is not None:
        pts = pts[np.asarray(mask).astype(bool)]
    else:
        pts = pts.reshape(-1, pts.shape[-1])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return pts.reshape(0, 3)
    # Remove exact zero placeholders and extreme stray values.
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
    if len(pts) == 0:
        return pts.reshape(0, 3)
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[keep]
    if stride > 1:
        pts = pts[::stride]
    return pts.astype(np.float64, copy=False)


def colorize(pts: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.float64), (len(pts), 1))


def pcd_from_parts(parts: list[tuple[np.ndarray, tuple[float, float, float]]]) -> o3d.geometry.PointCloud:
    points = []
    colors = []
    for pts, c in parts:
        if len(pts):
            points.append(pts)
            colors.append(colorize(pts, c))
    pcd = o3d.geometry.PointCloud()
    if points:
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
    return pcd


def save_ply(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def camera_basis(pts: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = pts.mean(axis=0)
    if view == "front":
        eye_dir = np.array([0.0, -1.0, 0.35])
        up = np.array([0.0, 0.25, 1.0])
    elif view == "side":
        eye_dir = np.array([1.0, -0.15, 0.25])
        up = np.array([0.0, 0.2, 1.0])
    elif view == "top":
        eye_dir = np.array([0.05, -0.25, 1.0])
        up = np.array([0.0, 1.0, 0.2])
    else:
        eye_dir = np.array([0.8, -0.8, 0.55])
        up = np.array([0.0, 0.25, 1.0])
    eye_dir = eye_dir / np.linalg.norm(eye_dir)
    up = up / np.linalg.norm(up)
    right = np.cross(eye_dir, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, eye_dir)
    up = up / np.linalg.norm(up)
    return center, right, up


def render_projection(path: Path, pcd: o3d.geometry.PointCloud, title: str, view: str = "iso", size=(1500, 1200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) == 0:
        Image.new("RGB", size, "white").save(path)
        return
    center, right, up = camera_basis(pts, view)
    rel = pts - center
    x = rel @ right
    y = rel @ up
    depth = rel @ np.cross(right, up)
    # Focus on robust centered body crop.
    q = np.percentile(np.stack([x, y], axis=1), [1, 99], axis=0)
    xmin, ymin = q[0]
    xmax, ymax = q[1]
    padx = max((xmax - xmin) * 0.08, 1e-3)
    pady = max((ymax - ymin) * 0.08, 1e-3)
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady
    keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    x, y, depth, cols = x[keep], y[keep], depth[keep], cols[keep]
    order = np.argsort(depth)
    x, y, cols = x[order], y[order], cols[order]
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img, "RGBA")
    sx = (W - 140) / max(xmax - xmin, 1e-6)
    sy = (H - 160) / max(ymax - ymin, 1e-6)
    s = min(sx, sy)
    px = (x - (xmin + xmax) / 2.0) * s + W / 2
    py = H / 2 - (y - (ymin + ymax) / 2.0) * s
    # Point radius adapts to count; large enough for readable person silhouette.
    r = 1.8 if len(px) > 60000 else 2.4 if len(px) > 15000 else 3.4
    for xx, yy, cc in zip(px, py, cols):
        rgb = tuple(int(max(0, min(255, v * 255))) for v in cc)
        draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=rgb + (180,))
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 42)
        small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.rectangle((0, 0, W, 78), fill=(255, 255, 255, 230))
    draw.text((40, 18), title, fill=(20, 20, 20, 255), font=font)
    draw.text((40, H - 48), f"Open3D PLY source, centered crop, view={view}, points={len(pts):,}", fill=(60, 60, 60, 255), font=small)
    img.save(path)


def try_open3d_visualizer(path: Path, pcd: o3d.geometry.PointCloud, view: str, size=(1500, 1200)) -> bool:
    try:
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return False
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(window_name=f"v50r2_{view}", width=size[0], height=size[1], visible=False)
        if not ok:
            return False
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 2.0 if len(pts) > 60000 else 4.0
        ctr = vis.get_view_control()
        center = pts.mean(axis=0)
        ctr.set_lookat(center.tolist())
        if view == "front":
            front = [0.0, -1.0, 0.05]
            up = [0.0, 0.0, 1.0]
            zoom = 0.58
        elif view == "side":
            front = [1.0, -0.05, 0.05]
            up = [0.0, 0.0, 1.0]
            zoom = 0.58
        elif view == "top":
            front = [0.0, 0.0, 1.0]
            up = [0.0, 1.0, 0.0]
            zoom = 0.58
        else:
            front = [0.75, -0.75, 0.45]
            up = [0.0, 0.0, 1.0]
            zoom = 0.62
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        for _ in range(6):
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(str(path), do_render=True)
        vis.destroy_window()
        return path.exists() and path.stat().st_size > 0
    except Exception:
        try:
            vis.destroy_window()  # type: ignore[name-defined]
        except Exception:
            pass
        return False


def try_open3d_offscreen(path: Path, pcd: o3d.geometry.PointCloud, title: str, view: str) -> bool:
    # Windows headless Open3D rendering is often unavailable. Keep this optional.
    try:
        import open3d.visualization.rendering as rendering

        w, h = 1500, 1200
        renderer = rendering.OffscreenRenderer(w, h)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0
        renderer.scene.set_background([1, 1, 1, 1])
        renderer.scene.add_geometry("pcd", pcd, mat)
        pts = np.asarray(pcd.points)
        center = pts.mean(axis=0)
        extent = max(np.ptp(pts, axis=0).max(), 1e-3)
        if view == "front":
            eye = center + np.array([0, -2.4 * extent, 0.8 * extent])
            up = [0, 0, 1]
        elif view == "side":
            eye = center + np.array([2.4 * extent, -0.3 * extent, 0.7 * extent])
            up = [0, 0, 1]
        elif view == "top":
            eye = center + np.array([0.2 * extent, -0.2 * extent, 2.8 * extent])
            up = [0, 1, 0]
        else:
            eye = center + np.array([1.7 * extent, -1.7 * extent, 1.1 * extent])
            up = [0, 0, 1]
        renderer.setup_camera(35.0, center, eye, up)
        img = renderer.render_to_image()
        o3d.io.write_image(str(path), img)
        return True
    except Exception:
        return False


def export_case(name: str, title: str, parts: list[tuple[np.ndarray, tuple[float, float, float]]], views: list[str]) -> dict[str, Any]:
    pcd = pcd_from_parts(parts)
    pcd = pcd.voxel_down_sample(voxel_size=0.0025) if len(pcd.points) > 80000 else pcd
    ply_path = PLY / f"{name}.ply"
    save_ply(ply_path, pcd)
    out_images = []
    for view in views:
        img_path = IMG / f"{name}_{view}.png"
        ok = try_open3d_visualizer(img_path, pcd, view)
        if not ok:
            ok = try_open3d_offscreen(img_path, pcd, title, view)
        if not ok:
            render_projection(img_path, pcd, title, view=view)
        out_images.append(str(img_path.resolve()))
    return {
        "name": name,
        "title": title,
        "ply": str(ply_path.resolve()),
        "images": out_images,
        "point_count": int(len(pcd.points)),
        "bbox_min": np.asarray(pcd.get_min_bound()).tolist() if len(pcd.points) else [],
        "bbox_max": np.asarray(pcd.get_max_bound()).tolist() if len(pcd.points) else [],
    }


def mesh_to_pcd(vertices: np.ndarray, faces: np.ndarray, samples: int = 120000) -> np.ndarray:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=samples, init_factor=4)
    return np.asarray(pcd.points)


def export_mesh_case(name: str, title: str, vertices: np.ndarray, faces: np.ndarray, views: list[str]) -> dict[str, Any]:
    pts = mesh_to_pcd(vertices, faces, samples=110000)
    pcd = pcd_from_parts([(pts, (0.52, 0.68, 0.88))])
    ply_path = PLY / f"{name}.ply"
    save_ply(ply_path, pcd)
    out_images = []
    for view in views:
        img_path = IMG / f"{name}_{view}.png"
        ok = try_open3d_visualizer(img_path, pcd, view)
        if not ok:
            ok = try_open3d_offscreen(img_path, pcd, title, view)
        if not ok:
            render_projection(img_path, pcd, title, view=view)
        out_images.append(str(img_path.resolve()))
    return {
        "name": name,
        "title": title,
        "ply": str(ply_path.resolve()),
        "images": out_images,
        "point_count": int(len(pcd.points)),
        "bbox_min": np.asarray(pcd.get_min_bound()).tolist() if len(pcd.points) else [],
        "bbox_max": np.asarray(pcd.get_max_bound()).tolist() if len(pcd.points) else [],
        "source": "V15 SMPL-X native raster vertices/faces",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    PLY.mkdir(parents=True, exist_ok=True)
    cand = load_npz("candidate_files__candidate_points.npz")
    head = load_npz("candidate_files__head_face_patch.npz")
    hand = load_npz("candidate_files__hand_patch.npz")
    temp = load_npz("candidate_files__temporal_teacher.npz")
    v15 = np.load(ROOT / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz", allow_pickle=True)

    cp = cand["candidate_points_world"]
    hp = head["refined_points_world"]
    hm = head["refined_region_id_map"]
    fp = hp[hm > 0]
    body_mask = temp["target_frame_region_id_map"] == 1
    body_pts = robust_points(temp["target_frame_points"], body_mask, stride=1)
    head_pts = robust_points(hp, hm == 1, stride=1)
    face_pts = robust_points(hp, hm == 2, stride=1)
    hand_pts = hand["hand_points_world"]
    hand_ids = hand["hand_region_id_map"]
    left_pts = robust_points(hand_pts, hand_ids == 1, stride=1)
    right_pts = robust_points(hand_pts, hand_ids == 2, stride=1)
    temporal_pts = robust_points(temp["target_frame_points"], temp["target_frame_region_id_map"] > 0, stride=1)

    cases = []
    cases.append(export_mesh_case(
        "open3d_smplx_native_full_body_clear",
        "SMPL-X Native Full Body Prior",
        np.asarray(v15["vertices"]),
        np.asarray(v15["faces"]),
        ["front", "side", "iso", "top"],
    ))
    cases.append(export_case(
        "open3d_full_body_centered",
        "V50R2 Candidate Full Body Risk View",
        [
            (body_pts, (0.55, 0.65, 0.78)),
            (head_pts, (0.95, 0.55, 0.25)),
            (face_pts, (0.95, 0.28, 0.22)),
            (left_pts, (0.25, 0.62, 0.95)),
            (right_pts, (0.15, 0.35, 0.9)),
        ],
        ["front", "side", "iso"],
    ))
    cases.append(export_case(
        "open3d_head_face_centered",
        "V50R2 Candidate Head / Face Risk View",
        [(head_pts, (0.95, 0.62, 0.25)), (face_pts, (0.95, 0.18, 0.16))],
        ["front", "side", "iso"],
    ))
    cases.append(export_case(
        "open3d_hairline_centered",
        "V50R2 Candidate Hairline / Head Risk View",
        [(head_pts, (0.95, 0.62, 0.25)), (face_pts, (0.75, 0.30, 0.18))],
        ["front", "top", "iso"],
    ))
    cases.append(export_case(
        "open3d_left_hand_centered",
        "V50R2 Candidate Left Hand Risk View",
        [(left_pts, (0.20, 0.55, 0.95))],
        ["front", "side", "iso"],
    ))
    cases.append(export_case(
        "open3d_right_hand_centered",
        "V50R2 Candidate Right Hand Risk View",
        [(right_pts, (0.10, 0.25, 0.85))],
        ["front", "side", "iso"],
    ))
    cases.append(export_case(
        "open3d_temporal_centered",
        "V50R2 Temporal Support Centered",
        [(temporal_pts, (0.40, 0.68, 0.55))],
        ["front", "side", "iso"],
    ))
    cases.append(export_case(
        "open3d_60view_support_centered",
        "V50R2 60-View Support Centered",
        [(body_pts, (0.55, 0.65, 0.78)), (head_pts, (0.95, 0.55, 0.25)), (face_pts, (0.95, 0.18, 0.16)), (left_pts, (0.2, 0.55, 0.95)), (right_pts, (0.1, 0.25, 0.85))],
        ["front", "iso"],
    ))

    report = {
        "task": "v223_open3d_clear_human_screenshots",
        "created_utc": now(),
        "open3d_version": o3d.__version__,
        "output_dir": str(OUT.resolve()),
        "image_dir": str(IMG.resolve()),
        "ply_dir": str(PLY.resolve()),
        "cases": cases,
        "point_counts_before_downsample": {
            "body": int(len(body_pts)),
            "head": int(len(head_pts)),
            "face": int(len(face_pts)),
            "left_hand": int(len(left_pts)),
            "right_hand": int(len(right_pts)),
            "temporal": int(len(temporal_pts)),
        },
        "note": "Images are captured from Open3D PLY sources with Open3D Visualizer first. Projection fallback is used only if both Visualizer and offscreen rendering are unavailable.",
    }
    (REPORTS / "20260509_v50r2_open3d_clear_human_screenshots.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# V50R2 Open3D Clear Human Screenshot List", ""]
    for case in cases:
        lines.append(f"## {case['title']}")
        lines.append(f"- PLY: `{case['ply']}`")
        lines.append(f"- points: `{case['point_count']}`")
        for img in case["images"]:
            lines.append(f"- image: `{img}`")
        lines.append("")
    (REPORTS / "20260509_v50r2_open3d_clear_human_screenshots.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
