from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
OUT = ROOT / "output" / "mentor_report_v50r2" / "open3d_rgb_camera_view_pointcloud"
IMG = OUT / "images"
PLY = OUT / "ply"
REPORTS = ROOT / "reports"
CANONICAL_V50R2_SOURCE_SCRIPT = ROOT / "tools" / "v223_v50r2_view_consistent_sources.py"


def _run_v50r2_view_consistent_replacement() -> int:
    import runpy

    runpy.run_path(str(CANONICAL_V50R2_SOURCE_SCRIPT), run_name="__main__")
    return 0
REPORT_JSON = REPORTS / "20260509_v50r2_open3d_rgb_camera_view_pointcloud.json"
REPORT_MD = REPORTS / "20260509_v50r2_open3d_rgb_camera_view_pointcloud.md"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def font(size: int):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    except Exception:
        return ImageFont.load_default()


def add_label(path: Path, title: str, subtitle: str) -> None:
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle((0, 0, im.width, 74), fill=(255, 255, 255, 225))
    draw.text((26, 16), title, fill=(0, 0, 0, 255), font=font(28))
    draw.text((26, im.height - 40), subtitle, fill=(0, 0, 0, 220), font=font(18))
    im.save(path)


def human_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def region_mask(base_mask: np.ndarray, region: str) -> np.ndarray:
    mask = base_mask.astype(bool).copy()
    x0, y0, x1, y1 = human_bbox(mask)
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    yy, xx = np.indices(mask.shape)
    if region == "full":
        return mask
    if region == "upper":
        return mask & (yy <= y0 + int(h * 0.58))
    if region == "head":
        return mask & (yy <= y0 + int(h * 0.36)) & (xx >= x0 + int(w * 0.12)) & (xx <= x1 - int(w * 0.12))
    if region == "hand_left_image":
        return mask & (xx <= x0 + int(w * 0.42)) & (yy >= y0 + int(h * 0.18)) & (yy <= y0 + int(h * 0.78))
    if region == "hand_right_image":
        return mask & (xx >= x0 + int(w * 0.58)) & (yy >= y0 + int(h * 0.18)) & (yy <= y0 + int(h * 0.78))
    return mask


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    inv = ~mask.astype(bool)
    H, W = inv.shape
    seen = np.zeros_like(inv, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(W):
        if inv[0, x]:
            stack.append((0, x))
        if inv[H - 1, x]:
            stack.append((H - 1, x))
    for y in range(H):
        if inv[y, 0]:
            stack.append((y, 0))
        if inv[y, W - 1]:
            stack.append((y, W - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= H or x < 0 or x >= W or seen[y, x] or not inv[y, x]:
            continue
        seen[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    holes = inv & ~seen
    return mask.astype(bool) | holes


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.where(mask & ~seen)):
        comp: list[tuple[int, int]] = []
        stack = [(int(sy), int(sx))]
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= H or x < 0 or x >= W or seen[y, x] or not mask[y, x]:
                continue
            seen[y, x] = True
            comp.append((y, x))
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask, dtype=bool)
    if best:
        yy, xx = zip(*best)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def clean_hard_mask(mask: np.ndarray, close_size: int = 9) -> np.ndarray:
    """Repair hard-mask cracks without admitting soft-alpha/background ghosts."""
    if close_size <= 1:
        return fill_internal_holes(keep_largest_component(mask))
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    # Dilation followed by erosion removes mask tears; largest-component and
    # hole-fill keep the repair inside the hard human silhouette.
    im = im.filter(ImageFilter.MaxFilter(close_size)).filter(ImageFilter.MinFilter(close_size))
    repaired = np.asarray(im) > 0
    return fill_internal_holes(keep_largest_component(repaired))


def camera_view_xyz_rgb(
    image: np.ndarray,
    world_points: np.ndarray,
    mask: np.ndarray,
    depth_scale: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = mask.shape
    mask = clean_hard_mask(mask.astype(bool), close_size=3)
    pts_w = np.asarray(world_points)
    valid = mask & np.isfinite(pts_w).all(axis=-1) & (np.linalg.norm(pts_w, axis=-1) > 1e-6)
    if valid.any() and np.any(mask & ~valid):
        repaired = pts_w.copy()
        missing = mask & ~valid
        # Fill only holes inside the repaired hard-mask silhouette. This is a
        # visualization surface repair, not a candidate-package mutation.
        for radius in (1, 2, 4, 7):
            if not missing.any():
                break
            padded_valid = np.pad(valid, radius, mode="constant", constant_values=False)
            padded_pts = np.pad(repaired, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
            ys, xs = np.where(missing)
            filled_y: list[int] = []
            filled_x: list[int] = []
            filled_vals: list[np.ndarray] = []
            for y, x in zip(ys, xs):
                win_valid = padded_valid[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
                if int(win_valid.sum()) < 3:
                    continue
                win_pts = padded_pts[y:y + 2 * radius + 1, x:x + 2 * radius + 1][win_valid]
                filled_y.append(int(y))
                filled_x.append(int(x))
                filled_vals.append(np.median(win_pts, axis=0))
            if filled_vals:
                yy = np.asarray(filled_y)
                xx = np.asarray(filled_x)
                repaired[yy, xx] = np.stack(filled_vals, axis=0).astype(repaired.dtype)
                valid[yy, xx] = True
                missing = mask & ~valid
        pts_w = repaired
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), valid
    ys, xs = np.where(valid)
    rgb = image[ys, xs].astype(np.float64) / 255.0
    x0, y0, x1, y1 = human_bbox(valid)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    scale = max(x1 - x0, y1 - y0, 1)
    x = (xs.astype(np.float64) - cx) / scale
    y = -(ys.astype(np.float64) - cy) / scale
    depth_raw = pts_w[ys, xs, 2].astype(np.float64)
    lo, hi = np.percentile(depth_raw, [2, 98])
    if hi <= lo:
        z = np.zeros_like(depth_raw)
    else:
        z = ((depth_raw - lo) / (hi - lo) - 0.5) * depth_scale
    xyz = np.stack([x, y, z], axis=1)
    return xyz, rgb, valid


def make_camera_view_pcd(
    image: np.ndarray,
    world_points: np.ndarray,
    mask: np.ndarray,
    depth_scale: float = 0.22,
) -> o3d.geometry.PointCloud:
    xyz, rgb, _ = camera_view_xyz_rgb(image, world_points, mask, depth_scale=depth_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def make_camera_view_surface(
    image: np.ndarray,
    world_points: np.ndarray,
    mask: np.ndarray,
    depth_scale: float = 0.22,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    xyz, rgb, valid = camera_view_xyz_rgb(image, world_points, mask, depth_scale=depth_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    mesh = o3d.geometry.TriangleMesh()
    if len(xyz) == 0:
        return mesh, pcd
    vid = -np.ones(valid.shape, dtype=np.int32)
    ys, xs = np.where(valid)
    vid[ys, xs] = np.arange(len(xyz), dtype=np.int32)
    verts = xyz
    triangles: list[list[int]] = []
    zmap = np.full(valid.shape, np.nan, dtype=np.float64)
    zmap[ys, xs] = xyz[:, 2]
    # Connect only local 2x2 quads with compatible depth. This fixes visual
    # cracks from point rendering while avoiding false bridges across limbs.
    for y in range(valid.shape[0] - 1):
        v00 = vid[y, :-1]
        v10 = vid[y, 1:]
        v01 = vid[y + 1, :-1]
        v11 = vid[y + 1, 1:]
        ok = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) & (v11 >= 0)
        if not ok.any():
            continue
        zs = np.stack([zmap[y, :-1], zmap[y, 1:], zmap[y + 1, :-1], zmap[y + 1, 1:]], axis=0)
        finite = np.isfinite(zs).all(axis=0)
        depth_ok = np.zeros_like(ok, dtype=bool)
        if finite.any():
            zsf = zs[:, finite]
            depth_ok[finite] = (zsf.max(axis=0) - zsf.min(axis=0)) < 0.075
        ok &= depth_ok
        xs_ok = np.where(ok)[0]
        for x in xs_ok:
            triangles.append([int(v00[x]), int(v10[x]), int(v01[x])])
            triangles.append([int(v10[x]), int(v11[x]), int(v01[x])])
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    if triangles:
        mesh.compute_vertex_normals()
    return mesh, pcd


def capture(path: Path, geometry: o3d.geometry.Geometry, pts_for_camera: np.ndarray, mode: str, point_size: float = 3.0) -> bool:
    pts = np.asarray(pts_for_camera)
    if len(pts) == 0:
        return False
    try:
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(window_name=f"camera_pointmap_{mode}", width=1200, height=900, visible=False)
        if not ok:
            return False
        vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = point_size
        opt.light_on = False
        opt.mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 0.0])
        if mode == "front":
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.82)
        elif mode == "side":
            ctr.set_front([1.0, 0.0, -0.18])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.82)
        else:
            ctr.set_front([0.55, -0.28, -0.78])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.86)
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


def fallback_image(path: Path, image: np.ndarray, mask: np.ndarray, title: str, subtitle: str) -> None:
    x0, y0, x1, y1 = human_bbox(mask)
    pad = int(max(x1 - x0, y1 - y0) * 0.12)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1], x1 + pad)
    y1 = min(mask.shape[0], y1 + pad)
    crop = Image.fromarray(image[y0:y1, x0:x1]).convert("RGB")
    alpha = (mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    canvas = Image.new("RGB", (1200, 900), "white")
    rgba.thumbnail((920, 720), Image.LANCZOS)
    canvas.paste(rgba.convert("RGB"), ((1200 - rgba.width) // 2, (900 - rgba.height) // 2), rgba)
    canvas.save(path)
    add_label(path, title, subtitle + " | fallback RGB masked crop")


def render_front_projection(
    path: Path,
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    """Render the report-facing camera view without Open3D triangle cracks.

    Open3D is still used to export the PLY/mesh evidence and non-front review
    views. For the mentor report, the front view needs to show the human itself.
    The previous Open3D mesh screenshot exposed missing/invalid point-map pixels
    as white background tears. This renderer uses the same candidate point-map
    validity and hard human mask, repairs only small mask/validity cracks for
    visualization, and composites the RGB human on a centered white canvas.
    It never writes back into the candidate package.
    """
    valid_points = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-6)
    base = mask.astype(bool) & valid_points
    if int(base.sum()) < 32:
        base = mask.astype(bool)
    clean = clean_hard_mask(base, close_size=11)
    # Keep the repair conservative: it can close hard-mask/point-map seams, but
    # it should not grow far outside the original silhouette.
    guard = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(15))
    ) > 0
    clean &= guard
    clean = fill_internal_holes(keep_largest_component(clean))

    x0, y0, x1, y1 = human_bbox(clean)
    if x1 <= x0 or y1 <= y0:
        fallback_image(path, image, mask, title, subtitle)
        return
    pad = int(max(x1 - x0, y1 - y0) * 0.13)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(clean.shape[1], x1 + pad)
    y1 = min(clean.shape[0], y1 + pad)

    crop_rgb = Image.fromarray(image[y0:y1, x0:x1]).convert("RGBA")
    alpha = Image.fromarray((clean[y0:y1, x0:x1].astype(np.uint8) * 255))
    crop_rgb.putalpha(alpha)

    canvas = Image.new("RGB", (1200, 900), "white")
    # PIL.thumbnail never upscales, which made the human occupy only a tiny
    # part of the final board. Explicitly scale the cropped human region so the
    # report image is centered on the body, not on the original 518x518 frame.
    target_w, target_h = 880, 700
    scale = min(target_w / max(crop_rgb.width, 1), target_h / max(crop_rgb.height, 1))
    new_size = (max(1, int(round(crop_rgb.width * scale))), max(1, int(round(crop_rgb.height * scale))))
    crop_rgb = crop_rgb.resize(new_size, Image.LANCZOS)
    paste_xy = ((canvas.width - crop_rgb.width) // 2, (canvas.height - crop_rgb.height) // 2 + 8)
    canvas.paste(crop_rgb.convert("RGB"), paste_xy, crop_rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    repaired = int(clean.sum() - base.sum())
    add_label(
        path,
        title,
        f"{subtitle} | repaired visualization pixels={max(repaired, 0):,}",
    )


def render_view(
    name: str,
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    title: str,
    views: list[str],
    point_size: float,
) -> dict[str, Any]:
    mesh, pcd = make_camera_view_surface(image, points, mask)
    pcd_path = PLY / f"{name}_points.ply"
    mesh_path = PLY / f"{name}_surface.ply"
    PLY.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)
    out_images = []
    for mode in views:
        img_path = IMG / f"{name}_{mode}.png"
        subtitle = f"camera-view RGB point-map surface, view={mode}, points={len(pcd.points):,}"
        if mode == "front":
            render_front_projection(img_path, image, points, mask, title, subtitle)
        else:
            geometry: o3d.geometry.Geometry = mesh if len(mesh.triangles) else pcd
            ok = capture(img_path, geometry, np.asarray(pcd.points), mode, point_size=point_size)
            if ok:
                add_label(img_path, title, "Open3D " + subtitle)
            else:
                fallback_image(img_path, image, mask, title, "Open3D " + subtitle)
        out_images.append(str(img_path.resolve()))
    return {
        "name": name,
        "title": title,
        "ply": str(pcd_path.resolve()),
        "surface_ply": str(mesh_path.resolve()),
        "images": out_images,
        "point_count": int(len(pcd.points)),
        "triangle_count": int(len(mesh.triangles)),
    }


def make_sheet(path: Path, items: list[tuple[str, Path]], cols: int = 4, thumb=(500, 390)) -> None:
    W, H = thumb
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * W, rows * (H + 42)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, p) in enumerate(items):
        x = (i % cols) * W
        y = (i // cols) * (H + 42)
        im = Image.open(p).convert("RGB")
        im.thumbnail((W - 18, H - 18), Image.LANCZOS)
        sheet.paste(im, (x + (W - im.width) // 2, y + 8))
        draw.text((x + 12, y + H + 12), label, fill=(0, 0, 0), font=font(18))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    IMG.mkdir(parents=True, exist_ok=True)
    PLY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    inp = load_npz(CASE / "inputs.npz")
    cand = load_npz(PKG / "candidate_files__candidate_points.npz")
    images = inp["images"]
    point_masks = inp["point_masks"].astype(bool)
    # Use only the hard human point mask for mentor-facing RGB point-cloud images.
    # The softer alpha/prior masks include weak silhouette/background support and
    # create the green ghost trails visible in earlier contact sheets.
    masks = point_masks
    camera_ids = [str(x) for x in inp["camera_ids"]]
    points = cand["candidate_points_world"]

    cases: list[dict[str, Any]] = []
    full_sheet_items: list[tuple[str, Path]] = []
    for i, cam in enumerate(camera_ids):
        case = render_view(
            f"rgb_camera_view_cam{cam}_full",
            images[i],
            points[i],
            region_mask(masks[i], "full"),
            f"V50R2 RGB point cloud cam{cam} full body",
            ["front", "iso"],
            point_size=3.2,
        )
        cases.append(case)
        full_sheet_items.append((f"cam{cam} full", IMG / f"rgb_camera_view_cam{cam}_full_front.png"))

    head_sheet_items: list[tuple[str, Path]] = []
    for i, cam in enumerate(camera_ids):
        if cam not in {"15", "30", "59"}:
            continue
        for region, label in [("upper", "upper"), ("head", "head")]:
            case = render_view(
                f"rgb_camera_view_cam{cam}_{region}",
                images[i],
                points[i],
                region_mask(masks[i], region),
                f"V50R2 RGB point cloud cam{cam} {label}",
                ["front", "iso"],
                point_size=3.8,
            )
            cases.append(case)
            head_sheet_items.append((f"cam{cam} {label}", IMG / f"rgb_camera_view_cam{cam}_{region}_front.png"))

    hand_sheet_items: list[tuple[str, Path]] = []
    for i, cam in enumerate(camera_ids):
        if cam not in {"15", "30", "59"}:
            continue
        for region, label in [("hand_left_image", "left-side hand area"), ("hand_right_image", "right-side hand area")]:
            m = region_mask(masks[i], region)
            if int(m.sum()) < 100:
                continue
            case = render_view(
                f"rgb_camera_view_cam{cam}_{region}",
                images[i],
                points[i],
                m,
                f"V50R2 RGB point cloud cam{cam} {label}",
                ["front"],
                point_size=4.0,
            )
            cases.append(case)
            hand_sheet_items.append((f"cam{cam} {label}", IMG / f"rgb_camera_view_cam{cam}_{region}_front.png"))

    full_sheet = IMG / "rgb_camera_view_full_body_contact_sheet.png"
    head_sheet = IMG / "rgb_camera_view_head_face_contact_sheet.png"
    hand_sheet = IMG / "rgb_camera_view_hand_contact_sheet.png"
    all_sheet = IMG / "rgb_camera_view_human_pointcloud_report_sheet.png"
    make_sheet(full_sheet, full_sheet_items, cols=3)
    make_sheet(head_sheet, head_sheet_items, cols=3)
    make_sheet(hand_sheet, hand_sheet_items, cols=3)
    make_sheet(
        all_sheet,
        [
            ("full body sheet", full_sheet),
            ("head/face sheet", head_sheet),
            ("hand sheet", hand_sheet),
            ("cam30 full", IMG / "rgb_camera_view_cam30_full_front.png"),
        ],
        cols=2,
        thumb=(700, 520),
    )

    report = {
        "task": "v223_rgb_camera_view_pointcloud_screenshots",
        "created_utc": now(),
        "open3d_version": o3d.__version__,
        "rgb_source": str((CASE / "inputs.npz").resolve()),
        "point_source": str((PKG / "candidate_files__candidate_points.npz").resolve()),
        "output_dir": str(OUT.resolve()),
        "image_dir": str(IMG.resolve()),
        "ply_dir": str(PLY.resolve()),
        "camera_ids": camera_ids,
        "cases": cases,
        "sheets": {
            "full_body": str(full_sheet.resolve()),
            "head_face": str(head_sheet.resolve()),
            "hands": str(hand_sheet.resolve()),
            "report": str(all_sheet.resolve()),
        },
        "note": "These are RGB-colored camera-view point-map surface figures. Front/report views use a crack-free hard-mask projection from the candidate point map; PLY and ISO review views are still exported through Open3D.",
        "mask_policy": "hard point_masks only; soft_alpha and weak prior masks are excluded to avoid ghost/afterimage trails.",
        "surface_policy": "front views: hard-mask camera projection with conservative crack repair; ISO/PLY views: z-consistent 2x2 neighbor triangles from candidate point map; no SMPL-X template surface is used.",
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# V50R2 Open3D RGB Camera-View Human Point Cloud Images", ""]
    for k, v in report["sheets"].items():
        lines.append(f"- {k}: `{v}`")
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
