from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf_v223_repaired"
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
OUT = ROOT / "output" / "mentor_report_v50r2" / "v223_view_consistent_pointcloud"
PLY_DIR = OUT / "ply"
IMG_DIR = OUT / "images"
REPORTS = ROOT / "reports"


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


def preprocess_pad_image(path: Path, target_size: int = 518, is_mask: bool = False) -> np.ndarray:
    mode = "L" if is_mask else "RGB"
    img = Image.open(path).convert(mode)
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    interp = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    img = img.resize((int(new_width), int(new_height)), interp)
    canvas_color = 0 if is_mask else (255, 255, 255)
    canvas = Image.new(mode, (target_size, target_size), canvas_color)
    left = (target_size - int(new_width)) // 2
    top = (target_size - int(new_height)) // 2
    canvas.paste(img, (left, top))
    arr = np.asarray(canvas)
    if is_mask:
        return arr > 127
    return arr.astype(np.uint8)


def finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-8)


def despill_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16).copy()
    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    spill = (g > 38) & (g > r + 5) & (g > b + 5) & (r < 190) & (b < 190)
    if spill.any():
        arr[spill, 1] = np.minimum(arr[spill, 1], np.maximum(r[spill], b[spill]) + 2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header + "\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.where(mask & ~seen)):
        stack = [(int(sy), int(sx))]
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w or seen[y, x] or not mask[y, x]:
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


def fill_holes(mask: np.ndarray) -> np.ndarray:
    inv = ~mask.astype(bool)
    h, w = inv.shape
    seen = np.zeros_like(inv, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if inv[0, x]:
            stack.append((0, x))
        if inv[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if inv[y, 0]:
            stack.append((y, 0))
        if inv[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or seen[y, x] or not inv[y, x]:
            continue
        seen[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return mask.astype(bool) | (inv & ~seen)


def clean_mask(mask: np.ndarray, close_size: int = 5) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    if close_size > 1:
        im = im.filter(ImageFilter.MaxFilter(close_size)).filter(ImageFilter.MinFilter(close_size))
    out = np.asarray(im) > 0
    return fill_holes(keep_largest_component(out))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def crop_region_mask(mask: np.ndarray, region: str) -> np.ndarray:
    x0, y0, x1, y1 = bbox(mask)
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    yy, xx = np.indices(mask.shape)
    base = mask.astype(bool)
    if region == "full":
        return base
    if region == "head_face":
        return base & (yy >= y0) & (yy <= y0 + int(0.38 * h)) & (xx >= x0 + int(0.13 * w)) & (xx <= x1 - int(0.13 * w))
    if region == "upper":
        return base & (yy >= y0) & (yy <= y0 + int(0.62 * h))
    if region == "left_hand_img":
        return base & (xx <= x0 + int(0.46 * w)) & (yy >= y0 + int(0.16 * h)) & (yy <= y0 + int(0.82 * h))
    if region == "right_hand_img":
        return base & (xx >= x0 + int(0.54 * w)) & (yy >= y0 + int(0.16 * h)) & (yy <= y0 + int(0.82 * h))
    return base


def export_view_ply(
    path: Path,
    points_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    valid = mask.astype(bool) & finite_points(points_map)
    yy, xx = np.where(valid)
    if max_points and len(xx) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(xx), max_points, replace=False))
        yy = yy[keep]
        xx = xx[keep]
    pts = points_map[yy, xx].astype(np.float32)
    rgb = despill_rgb(image[yy, xx])
    write_ply(path, pts, rgb)
    return {
        "path": str(path.resolve()),
        "points": int(len(pts)),
        "bbox_min": np.min(pts, axis=0).tolist() if len(pts) else None,
        "bbox_max": np.max(pts, axis=0).tolist() if len(pts) else None,
    }


def image_depth_visual_points(
    points_map: np.ndarray,
    mask: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an upright visualization coordinate frame from candidate point-map depth.

    The raw candidate points are still exported separately. This frame is only for
    human-readable MeshLab/Open3D review, where image x/y keep the person upright
    and the candidate point-map depth supplies the z relief.
    """
    if len(xx) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    raw = points_map[yy, xx].astype(np.float64)
    depth = raw[:, 2].copy()
    if (not np.isfinite(depth).all()) or float(np.nanpercentile(depth, 98) - np.nanpercentile(depth, 2)) < 1e-6:
        depth = np.linalg.norm(raw, axis=1)

    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((len(xx),), dtype=bool)

    d2, d50, d98 = np.percentile(depth[finite], [2, 50, 98])
    span = max(float(d98 - d2), 1e-6)
    keep = finite & (depth >= d2 - 0.20 * span) & (depth <= d98 + 0.20 * span)
    if int(keep.sum()) < 32:
        keep = finite

    x0, y0, x1, y1 = bbox(mask)
    cx = (x0 + x1 - 1) * 0.5
    cy = (y0 + y1 - 1) * 0.5
    scale = max(float(x1 - x0), float(y1 - y0), 1.0)
    x = (xx.astype(np.float64) - cx) / scale
    y = -(yy.astype(np.float64) - cy) / scale
    z = (depth - d50) / span * 0.42
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts[keep], keep


def export_visualization_ply(
    path: Path,
    points_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    valid = mask.astype(bool) & finite_points(points_map)
    yy, xx = np.where(valid)
    pts, keep = image_depth_visual_points(points_map, mask, yy, xx)
    yy = yy[keep]
    xx = xx[keep]
    if max_points and len(xx) > max_points:
        rng = np.random.default_rng(seed)
        choice = np.sort(rng.choice(len(xx), max_points, replace=False))
        yy = yy[choice]
        xx = xx[choice]
        pts = pts[choice]
    rgb = despill_rgb(image[yy, xx])
    write_ply(path, pts, rgb)
    return {
        "path": str(path.resolve()),
        "points": int(len(pts)),
        "coordinate_frame": "image_xy_plus_candidate_depth_z_for_visual_review",
        "bbox_min": np.min(pts, axis=0).tolist() if len(pts) else None,
        "bbox_max": np.max(pts, axis=0).tolist() if len(pts) else None,
    }


def draw_point_cloud_panel(
    image: np.ndarray,
    points_map: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str,
    *,
    yaw_deg: float = -20.0,
    pitch_deg: float = 8.0,
    max_points: int = 14000,
    seed: int = 0,
    size: tuple[int, int] = (640, 520),
    point_radius: int = 1,
) -> Image.Image:
    valid = mask.astype(bool) & finite_points(points_map)
    yy, xx = np.where(valid)
    if len(xx) == 0:
        return Image.new("RGB", size, "white")
    if len(xx) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xx), max_points, replace=False)
        yy = yy[keep]
        xx = xx[keep]
    pts, keep = image_depth_visual_points(points_map, mask, yy, xx)
    yy = yy[keep]
    xx = xx[keep]
    if len(xx) == 0:
        return Image.new("RGB", size, "white")
    rgb = despill_rgb(image[yy, xx])

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    ry = np.array(
        [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]],
        dtype=np.float64,
    )
    p = pts @ (ry @ rx).T
    depth = p[:, 2]
    order = np.argsort(depth)
    p = p[order]
    rgb = rgb[order]
    depth = depth[order]
    w, h = size
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, w, 58), fill=(255, 255, 255, 235))
    draw.text((16, 12), title, fill=(0, 0, 0, 255), font=font(24))
    draw.text((16, h - 34), subtitle, fill=(0, 0, 0, 220), font=font(14))

    x = p[:, 0]
    y = -p[:, 1]
    x_span = max(float(np.percentile(x, 99) - np.percentile(x, 1)), 1e-6)
    y_span = max(float(np.percentile(y, 99) - np.percentile(y, 1)), 1e-6)
    s = min((w * 0.72) / x_span, (h * 0.72) / y_span)
    u = (x - np.median(x)) * s + w * 0.52
    v = (y - np.median(y)) * s + h * 0.52
    z_norm = (depth - np.percentile(depth, 2)) / max(float(np.percentile(depth, 98) - np.percentile(depth, 2)), 1e-6)
    shade = np.clip(0.72 + 0.38 * z_norm, 0.48, 1.10)
    cols = np.clip(rgb.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)
    for px, py, col in zip(u, v, cols):
        if -4 <= px < w + 4 and -4 <= py < h + 4:
            draw.ellipse(
                (px - point_radius, py - point_radius, px + point_radius, py + point_radius),
                fill=(int(col[0]), int(col[1]), int(col[2]), 230),
            )
    return canvas


def make_sheet(path: Path, panels: list[tuple[str, Image.Image]], cols: int = 3, cell: tuple[int, int] = (640, 560)) -> None:
    cw, ch = cell
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * (ch + 42)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(panels):
        x = (idx % cols) * cw
        y = (idx // cols) * (ch + 42)
        thumb = img.copy()
        thumb.thumbnail((cw - 18, ch - 18), Image.LANCZOS)
        sheet.paste(thumb, (x + (cw - thumb.width) // 2, y + 8))
        draw.text((x + 14, y + ch + 10), label, fill=(0, 0, 0), font=font(20))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def mark_old_invalid() -> list[dict[str, str]]:
    invalids: list[dict[str, str]] = []
    old_root = ROOT / "output" / "mentor_report_v50r2"
    invalid_dir = old_root / "invalid_v15_v16_misaligned_exports"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "pointcloud_sources",
        "open3d_rgb_camera_view_pointcloud",
        "kinect_style_3d_pointcloud",
        "reference_angle_pointcloud",
        "rgb_dot_pointcloud",
    ]:
        path = old_root / rel
        if not path.exists() or path.resolve() == OUT.resolve():
            continue
        marker = path / "INVALID_SOURCE_PROTOCOL_DO_NOT_SEND.txt"
        marker.write_text(
            "Invalid for mentor-facing V50R2 point-cloud evidence: this directory was generated with V15/V16 6-view case images/camera ids, while V50R2 candidate points are V42 frame0000 views 00,01,06,11,16,21. Regenerate from tools/v223_v50r2_view_consistent_sources.py.\n",
            encoding="utf-8",
        )
        invalids.append({"path": str(path.resolve()), "reason": "V15/V16 source protocol mismatch"})
    return invalids


def main() -> int:
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    scene_manifest = json.loads((SCENE / "scene_manifest.json").read_text(encoding="utf-8"))
    views = scene_manifest["exported_views"]
    camera_ids = [str(v["camera_id"]) for v in views[:6]]
    image_paths = [Path(v["image_path"]) for v in views[:6]]
    mask_paths = [Path(v["mask_path"]) for v in views[:6]]

    cand = load_npz(PKG / "candidate_files__candidate_points.npz")["candidate_points_world"]
    v42 = load_npz(PKG / "v42_prior_enabled_payload__research_points_world.npz")["frame0000"]
    hand = load_npz(PKG / "candidate_files__hand_patch.npz")
    head = load_npz(PKG / "candidate_files__head_face_patch.npz")
    normals = load_npz(PKG / "candidate_files__candidate_normals.npz")["candidate_normals_geometric"]
    mapping_errors = []
    for idx in range(cand.shape[0]):
        diff = np.abs(cand[idx] - v42[idx])
        mapping_errors.append({"view_index": idx, "max_abs": float(diff.max()), "mean_abs": float(diff.mean())})

    images = [preprocess_pad_image(p, 518, False) for p in image_paths]
    masks = [clean_mask(preprocess_pad_image(p, 518, True), close_size=5) for p in mask_paths]

    records: dict[str, Any] = {
        "task": "v223_v50r2_view_consistent_sources",
        "created_utc": now(),
        "status": "DONE_PASS",
        "source_policy": "V50R2 candidate points are V42 frame0000 first six views. RGB/mask are rebuilt from the same 12views_tmf scene order using VGGT pad preprocessing.",
        "forbidden_policy": "No V15/V16 images, masks, camera ids, or targets are used for V50R2 mentor-facing point clouds.",
        "scene": str(SCENE.resolve()),
        "candidate_package": str(PKG.resolve()),
        "camera_ids": camera_ids,
        "view_mapping": mapping_errors,
        "ply": {},
        "images": {},
        "old_invalid_exports": mark_old_invalid(),
    }

    full_panels: list[tuple[str, Image.Image]] = []
    upper_panels: list[tuple[str, Image.Image]] = []
    head_panels: list[tuple[str, Image.Image]] = []
    hand_panels: list[tuple[str, Image.Image]] = []

    for idx, cam in enumerate(camera_ids):
        base_mask = masks[idx] & finite_points(cand[idx])
        full = crop_region_mask(base_mask, "full")
        upper = crop_region_mask(base_mask, "upper")
        head_face = (head["head_mask"][idx].astype(bool) | head["face_mask"][idx].astype(bool)) & base_mask
        if int(head_face.sum()) < 128:
            head_face = crop_region_mask(base_mask, "head_face")
        left_hand = (hand["hand_region_id_map"][idx] == 1) & base_mask
        right_hand = (hand["hand_region_id_map"][idx] == 2) & base_mask
        if int(left_hand.sum()) < 64:
            left_hand = crop_region_mask(base_mask, "left_hand_img")
        if int(right_hand.sum()) < 64:
            right_hand = crop_region_mask(base_mask, "right_hand_img")

        view_name = f"view{idx:02d}_cam{cam}"
        records["ply"][f"{view_name}_full"] = export_view_ply(
            PLY_DIR / f"v50r2_{view_name}_full_v42_consistent.ply",
            cand[idx],
            images[idx],
            full,
            max_points=None,
            seed=1000 + idx,
        )
        records["ply"][f"{view_name}_full_visual"] = export_visualization_ply(
            PLY_DIR / f"v50r2_{view_name}_full_v42_consistent_visual_upright.ply",
            cand[idx],
            images[idx],
            full,
            max_points=None,
            seed=1100 + idx,
        )
        records["ply"][f"{view_name}_head_face"] = export_view_ply(
            PLY_DIR / f"v50r2_{view_name}_head_face_v42_consistent.ply",
            cand[idx],
            images[idx],
            head_face,
            max_points=None,
            seed=2000 + idx,
        )
        records["ply"][f"{view_name}_head_face_visual"] = export_visualization_ply(
            PLY_DIR / f"v50r2_{view_name}_head_face_v42_consistent_visual_upright.ply",
            cand[idx],
            images[idx],
            head_face,
            max_points=None,
            seed=2100 + idx,
        )
        records["ply"][f"{view_name}_left_hand"] = export_view_ply(
            PLY_DIR / f"v50r2_{view_name}_left_hand_v42_consistent.ply",
            cand[idx],
            images[idx],
            left_hand,
            max_points=None,
            seed=3000 + idx,
        )
        records["ply"][f"{view_name}_left_hand_visual"] = export_visualization_ply(
            PLY_DIR / f"v50r2_{view_name}_left_hand_v42_consistent_visual_upright.ply",
            cand[idx],
            images[idx],
            left_hand,
            max_points=None,
            seed=3100 + idx,
        )
        records["ply"][f"{view_name}_right_hand"] = export_view_ply(
            PLY_DIR / f"v50r2_{view_name}_right_hand_v42_consistent.ply",
            cand[idx],
            images[idx],
            right_hand,
            max_points=None,
            seed=4000 + idx,
        )
        records["ply"][f"{view_name}_right_hand_visual"] = export_visualization_ply(
            PLY_DIR / f"v50r2_{view_name}_right_hand_v42_consistent_visual_upright.ply",
            cand[idx],
            images[idx],
            right_hand,
            max_points=None,
            seed=4100 + idx,
        )

        yaw = -15.0 if idx % 2 == 0 else 15.0
        full_panels.append(
            (
                f"cam{cam} full",
                draw_point_cloud_panel(
                    images[idx],
                    cand[idx],
                    full,
                    f"V50R2 point cloud cam{cam} full",
                    f"V42-consistent view {idx}, points={records['ply'][f'{view_name}_full']['points']:,}",
                    yaw_deg=yaw,
                    pitch_deg=8,
                    max_points=13000,
                    seed=5000 + idx,
                    point_radius=1,
                ),
            )
        )
        upper_panels.append(
            (
                f"cam{cam} upper",
                draw_point_cloud_panel(
                    images[idx],
                    cand[idx],
                    upper,
                    f"V50R2 point cloud cam{cam} upper body",
                    "RGB-colored candidate point map, oblique render",
                    yaw_deg=yaw,
                    pitch_deg=9,
                    max_points=9000,
                    seed=6000 + idx,
                    point_radius=1,
                ),
            )
        )
        head_panels.append(
            (
                f"cam{cam} head/face",
                draw_point_cloud_panel(
                    images[idx],
                    cand[idx],
                    head_face,
                    f"V50R2 point cloud cam{cam} head/face",
                    f"ROI points={records['ply'][f'{view_name}_head_face']['points']:,}",
                    yaw_deg=yaw,
                    pitch_deg=9,
                    max_points=5500,
                    seed=7000 + idx,
                    point_radius=1,
                ),
            )
        )
        hand_panels.append(
            (
                f"cam{cam} L hand",
                draw_point_cloud_panel(
                    images[idx],
                    cand[idx],
                    left_hand,
                    f"V50R2 point cloud cam{cam} left hand",
                    f"ROI points={records['ply'][f'{view_name}_left_hand']['points']:,}",
                    yaw_deg=yaw,
                    pitch_deg=10,
                    max_points=2500,
                    seed=8000 + idx,
                    point_radius=1,
                ),
            )
        )
        hand_panels.append(
            (
                f"cam{cam} R hand",
                draw_point_cloud_panel(
                    images[idx],
                    cand[idx],
                    right_hand,
                    f"V50R2 point cloud cam{cam} right hand",
                    f"ROI points={records['ply'][f'{view_name}_right_hand']['points']:,}",
                    yaw_deg=yaw,
                    pitch_deg=10,
                    max_points=2500,
                    seed=9000 + idx,
                    point_radius=1,
                ),
            )
        )

    sheets = {
        "full_body": IMG_DIR / "V223_V50R2_full_body_pointcloud_v42_consistent.png",
        "upper_body": IMG_DIR / "V223_V50R2_upper_body_pointcloud_v42_consistent.png",
        "head_face": IMG_DIR / "V223_V50R2_head_face_pointcloud_v42_consistent.png",
        "hands": IMG_DIR / "V223_V50R2_hands_pointcloud_v42_consistent.png",
    }
    make_sheet(sheets["full_body"], full_panels, cols=3)
    make_sheet(sheets["upper_body"], upper_panels, cols=3)
    make_sheet(sheets["head_face"], head_panels, cols=3)
    make_sheet(sheets["hands"], hand_panels, cols=3)
    records["images"] = {k: str(v.resolve()) for k, v in sheets.items()}

    mentor_img = ROOT / "output" / "mentor_report_v50r2" / "images"
    mentor_img.mkdir(parents=True, exist_ok=True)
    copied = {}
    for idx, key in enumerate(("full_body", "upper_body", "head_face", "hands"), start=1):
        dst = mentor_img / f"{idx:02d}_V223_{key}_pointcloud_v42_consistent.png"
        shutil.copy2(sheets[key], dst)
        copied[key] = str(dst.resolve())
    records["mentor_report_images"] = copied

    json_path = REPORTS / "20260509_v50r2_view_consistent_pointcloud_sources.json"
    md_path = REPORTS / "20260509_v50r2_view_consistent_pointcloud_sources.md"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# V50R2 View-Consistent Point Cloud Sources",
        "",
        "This replaces the previous V15/V16-misaligned point-cloud exports.",
        "",
        "## Source Decision",
        "",
        "- V50R2 candidate points are exactly `v42_prior_enabled_payload__research_points_world.npz/frame0000[:6]`.",
        f"- Correct view order: `{', '.join(camera_ids)}`.",
        "- The old cam15/cam30/cam45/cam59 V50R2 figures were invalid because they used V15 6-view images and masks.",
        "",
        "## Mentor Images",
    ]
    for key, path in copied.items():
        lines.append(f"- {key}: `{path}`")
    lines += ["", "## Point Cloud PLYs"]
    for key, row in records["ply"].items():
        lines.append(f"- {key}: `{row['path']}` ({row['points']} points)")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
