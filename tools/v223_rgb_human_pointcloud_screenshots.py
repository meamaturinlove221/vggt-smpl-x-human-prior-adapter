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
CASE = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
ROI = ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
OUT = ROOT / "output" / "mentor_report_v50r2" / "open3d_rgb_human_pointcloud"
IMG = OUT / "images"
PLY = OUT / "ply"
REPORTS = ROOT / "reports"
CANONICAL_V50R2_SOURCE_SCRIPT = ROOT / "tools" / "v223_v50r2_view_consistent_sources.py"


def _run_v50r2_view_consistent_replacement() -> int:
    import runpy

    runpy.run_path(str(CANONICAL_V50R2_SOURCE_SCRIPT), run_name="__main__")
    return 0
REPORT_JSON = REPORTS / "20260509_v50r2_open3d_rgb_human_pointcloud.json"
REPORT_MD = REPORTS / "20260509_v50r2_open3d_rgb_human_pointcloud.md"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def finite_nonzero(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = points[mask.astype(bool)]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    lo = np.percentile(pts, 0.2, axis=0)
    hi = np.percentile(pts, 99.8, axis=0)
    return pts[np.all((pts >= lo) & (pts <= hi), axis=1)].astype(np.float64, copy=False)


def points_and_colors(points: np.ndarray, images: np.ndarray, mask: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    pts = points[mask.astype(bool)]
    rgb = images[mask.astype(bool)]
    keep = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
    pts = pts[keep]
    rgb = rgb[keep]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    lo = np.percentile(pts, 0.2, axis=0)
    hi = np.percentile(pts, 99.8, axis=0)
    keep2 = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[keep2][::stride].astype(np.float64, copy=False)
    rgb = rgb[keep2][::stride].astype(np.float64) / 255.0
    return pts, rgb


def make_pcd(parts: list[tuple[np.ndarray, np.ndarray]]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pts = [p for p, _ in parts if len(p)]
    cols = [c for _, c in parts if len(c)]
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols, axis=0))
    return pcd


def crop_pcd_by_percentile(pcd: o3d.geometry.PointCloud, lo_p=0.5, hi_p=99.5) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) == 0:
        return pcd
    lo = np.percentile(pts, lo_p, axis=0)
    hi = np.percentile(pts, hi_p, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    out.colors = o3d.utility.Vector3dVector(cols[keep])
    return out


def save_ply(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def try_visualizer(path: Path, pcd: o3d.geometry.PointCloud, view: str, point_size: float = 2.2, size=(1500, 1100)) -> bool:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return False
    try:
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(window_name=f"rgb_{view}", width=size[0], height=size[1], visible=False)
        if not ok:
            return False
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = point_size
        ctr = vis.get_view_control()
        ctr.set_lookat(pts.mean(axis=0).tolist())
        if view == "front":
            front, up, zoom = [0.0, -1.0, 0.05], [0.0, 0.0, 1.0], 0.62
        elif view == "side":
            front, up, zoom = [1.0, -0.05, 0.03], [0.0, 0.0, 1.0], 0.62
        elif view == "top":
            front, up, zoom = [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.62
        else:
            front, up, zoom = [0.72, -0.72, 0.38], [0.0, 0.0, 1.0], 0.66
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        for _ in range(8):
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


def projection_fallback(path: Path, pcd: o3d.geometry.PointCloud, title: str, view: str, size=(1500, 1100)) -> None:
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img, "RGBA")
    if len(pts):
        center = pts.mean(axis=0)
        if view == "front":
            eye = np.array([0, -1, 0.05], dtype=float)
            up = np.array([0, 0, 1], dtype=float)
        elif view == "side":
            eye = np.array([1, -0.05, 0.03], dtype=float)
            up = np.array([0, 0, 1], dtype=float)
        else:
            eye = np.array([0.72, -0.72, 0.38], dtype=float)
            up = np.array([0, 0, 1], dtype=float)
        eye /= np.linalg.norm(eye)
        right = np.cross(eye, up)
        right /= max(np.linalg.norm(right), 1e-8)
        up = np.cross(right, eye)
        up /= max(np.linalg.norm(up), 1e-8)
        rel = pts - center
        x = rel @ right
        y = rel @ up
        z = rel @ eye
        q = np.percentile(np.stack([x, y], axis=1), [1, 99], axis=0)
        xmin, ymin = q[0]
        xmax, ymax = q[1]
        pad = 0.08
        xmin -= (xmax - xmin) * pad
        xmax += (xmax - xmin) * pad
        ymin -= (ymax - ymin) * pad
        ymax += (ymax - ymin) * pad
        keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x, y, z, cols = x[keep], y[keep], z[keep], cols[keep]
        order = np.argsort(z)
        x, y, cols = x[order], y[order], cols[order]
        scale = min((W - 160) / max(xmax - xmin, 1e-8), (H - 180) / max(ymax - ymin, 1e-8))
        px = (x - (xmin + xmax) / 2.0) * scale + W / 2
        py = H / 2 - (y - (ymin + ymax) / 2.0) * scale
        r = 1.8 if len(px) > 50000 else 2.5
        for xx, yy, c in zip(px, py, cols):
            rgb = tuple(int(np.clip(v * 255, 0, 255)) for v in c)
            draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=rgb + (210,))
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 30)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 24), title, fill=(0, 0, 0, 255), font=font)
    img.save(path)


def render_case(name: str, title: str, pcd: o3d.geometry.PointCloud, views: list[str], point_size: float = 2.2) -> dict[str, Any]:
    pcd = crop_pcd_by_percentile(pcd)
    ply_path = PLY / f"{name}.ply"
    save_ply(ply_path, pcd)
    images = []
    for view in views:
        img_path = IMG / f"{name}_{view}.png"
        if not try_visualizer(img_path, pcd, view, point_size=point_size):
            projection_fallback(img_path, pcd, title, view)
        images.append(str(img_path.resolve()))
    return {
        "name": name,
        "title": title,
        "ply": str(ply_path.resolve()),
        "images": images,
        "point_count": int(len(pcd.points)),
        "bbox_min": np.asarray(pcd.get_min_bound()).tolist() if len(pcd.points) else [],
        "bbox_max": np.asarray(pcd.get_max_bound()).tolist() if len(pcd.points) else [],
    }


def make_sheet(path: Path, items: list[tuple[str, Path]], cols: int = 4, thumb=(520, 380)) -> None:
    W, H = thumb
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * W, rows * (H + 44)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, img_path) in enumerate(items):
        x = (idx % cols) * W
        y = (idx // cols) * (H + 44)
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((W - 20, H - 20), Image.LANCZOS)
        sheet.paste(im, (x + (W - im.width) // 2, y + 8))
        draw.text((x + 10, y + H + 14), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    IMG.mkdir(parents=True, exist_ok=True)
    PLY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    inputs = load_npz(CASE / "inputs.npz")
    candidate = load_npz(PKG / "candidate_files__candidate_points.npz")
    head = load_npz(PKG / "candidate_files__head_face_patch.npz")
    hand = load_npz(PKG / "candidate_files__hand_patch.npz")
    roi = load_npz(ROI)

    images = inputs["images"]
    masks = inputs["point_masks"] | inputs["prior_mask"] | (inputs["soft_alpha"] > 0.15)
    points = candidate["candidate_points_world"]
    region = roi["roi_maps"]
    roi_names = [str(x) for x in roi["roi_names"]]
    roi_index = {name: i for i, name in enumerate(roi_names)}

    full_pts, full_cols = points_and_colors(points, images, masks, stride=1)
    full = make_pcd([(full_pts, full_cols)])

    target_mask = masks[0]
    target_pts, target_cols = points_and_colors(points[:1], images[:1], target_mask[None, ...], stride=1)
    target = make_pcd([(target_pts, target_cols)])

    head_mask = head["head_mask"].astype(bool) | (head["refined_region_id_map"] > 0)
    head_pts, head_cols = points_and_colors(head["refined_points_world"], images, head_mask, stride=1)
    head_pcd = make_pcd([(head_pts, head_cols)])

    face_mask = head["face_mask"].astype(bool) | (head["refined_region_id_map"] == 2)
    face_pts, face_cols = points_and_colors(head["refined_points_world"], images, face_mask, stride=1)
    face_pcd = make_pcd([(face_pts, face_cols)])

    left_mask = hand["hand_region_id_map"] == 1
    right_mask = hand["hand_region_id_map"] == 2
    left_pts, left_cols = points_and_colors(hand["hand_points_world"], images, left_mask, stride=1)
    right_pts, right_cols = points_and_colors(hand["hand_points_world"], images, right_mask, stride=1)
    left_pcd = make_pcd([(left_pts, left_cols)])
    right_pcd = make_pcd([(right_pts, right_cols)])

    # Head/shoulder crop keeps the same visual style as the old Kinect comparison board.
    hs_mask = masks.copy()
    if "head" in roi_index:
        hs_mask |= region[:, roi_index["head"]].astype(bool)
    if "face_front" in roi_index:
        hs_mask |= region[:, roi_index["face_front"]].astype(bool)
    # Use upper half of target human crop as a fallback if ROI labels are sparse.
    yy = np.arange(hs_mask.shape[1])[None, :, None]
    hs_mask &= yy < int(hs_mask.shape[1] * 0.70)
    hs_pts, hs_cols = points_and_colors(points, images, hs_mask, stride=1)
    hs_pcd = make_pcd([(hs_pts, hs_cols)])

    cases = [
        render_case("rgb_v50r2_full_body", "V50R2 RGB human point cloud full body", full, ["front", "side", "iso"], point_size=2.0),
        render_case("rgb_v50r2_target_view", "V50R2 RGB target-view point cloud", target, ["front", "iso"], point_size=2.4),
        render_case("rgb_v50r2_head_shoulders", "V50R2 RGB head/shoulders point cloud", hs_pcd, ["front", "side", "iso"], point_size=2.8),
        render_case("rgb_v50r2_head_face", "V50R2 RGB head/face patch point cloud", head_pcd, ["front", "side", "iso"], point_size=3.0),
        render_case("rgb_v50r2_face_close", "V50R2 RGB face-close point cloud", face_pcd, ["front", "iso"], point_size=3.2),
        render_case("rgb_v50r2_left_hand", "V50R2 RGB left hand point cloud", left_pcd, ["front", "iso"], point_size=3.2),
        render_case("rgb_v50r2_right_hand", "V50R2 RGB right hand point cloud", right_pcd, ["front", "iso"], point_size=3.2),
    ]

    sheet_items = [
        ("full front", IMG / "rgb_v50r2_full_body_front.png"),
        ("full iso", IMG / "rgb_v50r2_full_body_iso.png"),
        ("target view", IMG / "rgb_v50r2_target_view_front.png"),
        ("head shoulders", IMG / "rgb_v50r2_head_shoulders_front.png"),
        ("head face", IMG / "rgb_v50r2_head_face_front.png"),
        ("face close", IMG / "rgb_v50r2_face_close_front.png"),
        ("left hand", IMG / "rgb_v50r2_left_hand_front.png"),
        ("right hand", IMG / "rgb_v50r2_right_hand_front.png"),
    ]
    make_sheet(IMG / "rgb_v50r2_human_pointcloud_contact_sheet.png", sheet_items, cols=4)

    report = {
        "task": "v223_rgb_human_pointcloud_screenshots",
        "created_utc": now(),
        "open3d_version": o3d.__version__,
        "rgb_source": str((CASE / "inputs.npz").resolve()),
        "point_source": str((PKG / "candidate_files__candidate_points.npz").resolve()),
        "head_face_source": str((PKG / "candidate_files__head_face_patch.npz").resolve()),
        "hand_source": str((PKG / "candidate_files__hand_patch.npz").resolve()),
        "output_dir": str(OUT.resolve()),
        "image_dir": str(IMG.resolve()),
        "ply_dir": str(PLY.resolve()),
        "cases": cases,
        "contact_sheet": str((IMG / "rgb_v50r2_human_pointcloud_contact_sheet.png").resolve()),
        "point_counts": {
            "full": int(len(full.points)),
            "target": int(len(target.points)),
            "head_shoulders": int(len(hs_pcd.points)),
            "head_face": int(len(head_pcd.points)),
            "face": int(len(face_pcd.points)),
            "left_hand": int(len(left_pcd.points)),
            "right_hand": int(len(right_pcd.points)),
        },
        "note": "These are RGB-colored human point-cloud screenshots. Colors come from the aligned 6-view input images at the same 518x518 pixel coordinates as the candidate/world point maps.",
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# V50R2 RGB Human Point Cloud Image List", ""]
    lines.append(f"- contact sheet: `{report['contact_sheet']}`")
    lines.append("")
    for case in cases:
        lines.append(f"## {case['title']}")
        lines.append(f"- PLY: `{case['ply']}`")
        lines.append(f"- points: `{case['point_count']}`")
        for img in case["images"]:
            lines.append(f"- image: `{img}`")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
