from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from v223_v50r2_view_consistent_sources import (
    bbox,
    clean_mask,
    crop_region_mask,
    despill_rgb,
    finite_points,
    font,
    image_depth_visual_points,
    preprocess_pad_image,
    write_ply,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf_v223_repaired"
OUT = ROOT / "output" / "mentor_report_v50r2" / "vertical_vggt_baseline_comparison"
IMG_DIR = OUT / "images"
PLY_DIR = OUT / "ply"
REPORTS = ROOT / "reports"
MD_REPORT = REPORTS / "20260509_v50r2_vggt_vertical_baseline_comparison.md"
JSON_REPORT = REPORTS / "20260509_v50r2_vggt_vertical_baseline_comparison.json"
CSV_REPORT = REPORTS / "20260509_v50r2_vggt_vertical_baseline_metrics.csv"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    point_path: Path
    point_key: str
    method_type: str
    confidence_path: Path | None = None
    confidence_key: str | None = None
    normal_path: Path | None = None
    normal_key: str | None = None
    head_patch_path: Path | None = None
    hand_patch_path: Path | None = None
    notes: str = ""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def points_from_npz(path: Path, key: str) -> np.ndarray:
    data = load_npz(path)
    arr = data[key]
    if arr.ndim == 4 and arr.shape[0] >= 6:
        return arr[:6].astype(np.float32)
    raise ValueError(f"{path}::{key} expected shape (views,H,W,3), got {arr.shape}")


def confidence_from_npz(path: Path | None, key: str | None) -> np.ndarray | None:
    if path is None or key is None or not path.exists():
        return None
    data = load_npz(path)
    if key not in data:
        return None
    arr = data[key]
    if arr.ndim >= 3 and arr.shape[0] >= 6:
        return arr[:6].astype(np.float32)
    return None


def normal_available(path: Path | None, key: str | None) -> bool:
    return bool(path is not None and key is not None and path.exists() and key in load_npz(path))


def load_scene() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    manifest = json.loads((SCENE / "scene_manifest.json").read_text(encoding="utf-8"))
    views = manifest["exported_views"][:6]
    camera_ids = [str(v["camera_id"]) for v in views]
    images = [preprocess_pad_image(Path(v["image_path"]), 518, False) for v in views]
    masks = [clean_mask(preprocess_pad_image(Path(v["mask_path"]), 518, True), close_size=5) for v in views]
    return camera_ids, images, masks


def region_masks(base: np.ndarray, *, head_patch: dict[str, np.ndarray] | None, hand_patch: dict[str, np.ndarray] | None, view: int) -> dict[str, np.ndarray]:
    masks = {
        "full": crop_region_mask(base, "full"),
        "upper": crop_region_mask(base, "upper"),
        "head_face": crop_region_mask(base, "head_face"),
        "left_hand": crop_region_mask(base, "left_hand_img"),
        "right_hand": crop_region_mask(base, "right_hand_img"),
    }
    if head_patch is not None:
        hp = (head_patch.get("head_mask", np.zeros_like(base))[view].astype(bool) |
              head_patch.get("face_mask", np.zeros_like(base))[view].astype(bool)) & base
        if int(hp.sum()) >= 128:
            masks["head_face"] = hp
    if hand_patch is not None:
        region = hand_patch.get("hand_region_id_map")
        if region is not None:
            left = (region[view] == 1) & base
            right = (region[view] == 2) & base
            if int(left.sum()) >= 64:
                masks["left_hand"] = left
            if int(right.sum()) >= 64:
                masks["right_hand"] = right
    return masks


def point_neighbor_delta(points: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    valid = valid.astype(bool) & finite_points(points)
    if int(valid.sum()) < 16:
        return float("nan"), float("nan")
    right = valid[:, 1:] & valid[:, :-1]
    down = valid[1:, :] & valid[:-1, :]
    vals = []
    if right.any():
        d = np.linalg.norm(points[:, 1:][right] - points[:, :-1][right], axis=-1)
        vals.append(d)
    if down.any():
        d = np.linalg.norm(points[1:, :][down] - points[:-1, :][down], axis=-1)
        vals.append(d)
    if not vals:
        return float("nan"), float("nan")
    all_d = np.concatenate(vals)
    return float(np.nanmedian(all_d)), float(np.nanpercentile(all_d, 95))


def metric_row(
    method: MethodSpec,
    view_idx: int,
    cam: str,
    region: str,
    points: np.ndarray,
    mask: np.ndarray,
    baseline_points: np.ndarray | None,
    conf: np.ndarray | None,
    has_normals: bool,
) -> dict[str, Any]:
    valid = mask.astype(bool) & finite_points(points)
    yy, xx = np.where(valid)
    pts = points[yy, xx].astype(np.float64)
    region_px = int(mask.sum())
    valid_px = int(len(pts))
    coverage = float(valid_px / max(region_px, 1))
    if len(pts):
        p2, p50, p98 = np.percentile(pts, [2, 50, 98], axis=0)
        span = p98 - p2
        z_relief = float(np.nanpercentile(pts[:, 2], 95) - np.nanpercentile(pts[:, 2], 5))
        xyz_span_norm = float(np.linalg.norm(span))
    else:
        span = np.full(3, np.nan)
        z_relief = float("nan")
        xyz_span_norm = float("nan")
    med_step, p95_step = point_neighbor_delta(points, valid)
    l2_from_baseline = float("nan")
    if baseline_points is not None:
        bvalid = valid & finite_points(baseline_points)
        by, bx = np.where(bvalid)
        if len(bx):
            l2_from_baseline = float(np.linalg.norm(points[by, bx] - baseline_points[by, bx], axis=-1).mean())
    conf_mean = float("nan")
    if conf is not None and conf.shape[:2] == mask.shape and valid_px:
        conf_mean = float(np.nanmean(conf[valid]))
    return {
        "method": method.key,
        "label": method.label,
        "method_type": method.method_type,
        "view_index": view_idx,
        "camera_id": cam,
        "region": region,
        "region_pixels": region_px,
        "valid_points": valid_px,
        "coverage": coverage,
        "xyz_span_x": float(span[0]),
        "xyz_span_y": float(span[1]),
        "xyz_span_z": float(span[2]),
        "xyz_span_norm": xyz_span_norm,
        "z_relief_p95_p5": z_relief,
        "neighbor_delta_median": med_step,
        "neighbor_delta_p95": p95_step,
        "mean_l2_from_v25_baseline": l2_from_baseline,
        "confidence_mean": conf_mean,
        "normal_available": has_normals,
        "notes": method.notes,
    }


def palette_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float64)
    lo, hi = np.nanpercentile(d, [2, 98]) if d.size else (0.0, 1.0)
    t = np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    r = np.clip(255 * (1.2 * t - 0.1), 0, 255)
    g = np.clip(255 * (1.0 - np.abs(t - 0.52) * 1.7), 0, 255)
    b = np.clip(255 * (1.1 - 1.2 * t), 0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def draw_panel(
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str,
    *,
    color_mode: str,
    yaw_deg: float,
    pitch_deg: float = 9.0,
    max_points: int = 9000,
    seed: int = 0,
    size: tuple[int, int] = (440, 390),
    point_radius: int = 1,
) -> Image.Image:
    valid = mask.astype(bool) & finite_points(points)
    yy, xx = np.where(valid)
    if len(xx) == 0:
        canvas = Image.new("RGB", size, "white")
        ImageDraw.Draw(canvas).text((12, 12), title + "\nEMPTY", fill=(0, 0, 0), font=font(18))
        return canvas
    if len(xx) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(xx), max_points, replace=False))
        yy = yy[keep]
        xx = xx[keep]
    vis_pts, keep = image_depth_visual_points(points, mask, yy, xx)
    yy = yy[keep]
    xx = xx[keep]
    if len(xx) == 0:
        canvas = Image.new("RGB", size, "white")
        ImageDraw.Draw(canvas).text((12, 12), title + "\nEMPTY_AFTER_DEPTH_FILTER", fill=(0, 0, 0), font=font(18))
        return canvas
    if color_mode == "depth":
        raw_depth = points[yy, xx, 2]
        colors = palette_depth(raw_depth)
    else:
        colors = despill_rgb(image[yy, xx])

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
    p = vis_pts @ (ry @ rx).T
    order = np.argsort(p[:, 2])
    p = p[order]
    colors = colors[order]

    w, h = size
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((12, 10), title, fill=(0, 0, 0, 255), font=font(17))
    draw.text((12, h - 26), subtitle, fill=(0, 0, 0, 210), font=font(12))

    x = p[:, 0]
    y = -p[:, 1]
    x_span = max(float(np.percentile(x, 99) - np.percentile(x, 1)), 1e-6)
    y_span = max(float(np.percentile(y, 99) - np.percentile(y, 1)), 1e-6)
    scale = min((w * 0.70) / x_span, (h * 0.70) / y_span)
    u = (x - np.median(x)) * scale + w * 0.52
    v = (y - np.median(y)) * scale + h * 0.52
    z = p[:, 2]
    shade = (z - np.percentile(z, 2)) / max(float(np.percentile(z, 98) - np.percentile(z, 2)), 1e-6)
    shade = np.clip(0.62 + 0.45 * shade, 0.45, 1.10)
    shaded = np.clip(colors.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)
    for px, py, col in zip(u, v, shaded):
        if -4 <= px < w + 4 and -4 <= py < h + 4:
            draw.ellipse((px - point_radius, py - point_radius, px + point_radius, py + point_radius), fill=(int(col[0]), int(col[1]), int(col[2]), 225))
    return canvas


def make_grid(path: Path, rows: list[tuple[str, list[Image.Image]]], col_labels: list[str]) -> None:
    cell_w, cell_h = 440, 390
    left_w = 210
    top_h = 58
    img = Image.new("RGB", (left_w + len(col_labels) * cell_w, top_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(img)
    for ci, label in enumerate(col_labels):
        draw.text((left_w + ci * cell_w + 12, 18), label, fill=(0, 0, 0), font=font(19))
    for ri, (row_label, panels) in enumerate(rows):
        y = top_h + ri * cell_h
        draw.rectangle((0, y, left_w, y + cell_h), fill=(250, 250, 250))
        for j, line in enumerate(row_label.split("\n")):
            draw.text((14, y + 22 + j * 24), line, fill=(0, 0, 0), font=font(18 if j == 0 else 14))
        for ci, panel in enumerate(panels):
            img.paste(panel, (left_w + ci * cell_w, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def export_region_ply(path: Path, points: np.ndarray, image: np.ndarray, mask: np.ndarray) -> int:
    valid = mask.astype(bool) & finite_points(points)
    yy, xx = np.where(valid)
    write_ply(path, points[yy, xx].astype(np.float32), despill_rgb(image[yy, xx]))
    return int(len(xx))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["method"], r["region"]) for r in rows})
    numeric = [
        "region_pixels",
        "valid_points",
        "coverage",
        "xyz_span_norm",
        "z_relief_p95_p5",
        "neighbor_delta_median",
        "neighbor_delta_p95",
        "mean_l2_from_v25_baseline",
        "confidence_mean",
    ]
    for method, region in keys:
        sub = [r for r in rows if r["method"] == method and r["region"] == region]
        rec = {"method": method, "label": sub[0]["label"], "method_type": sub[0]["method_type"], "region": region, "views": len(sub)}
        for n in numeric:
            vals = np.array([r[n] for r in sub], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rec[f"{n}_mean"] = float(vals.mean()) if len(vals) else None
        rec["normal_available"] = bool(sub[0]["normal_available"])
        rec["notes"] = sub[0]["notes"]
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def method_to_json(method: MethodSpec) -> dict[str, Any]:
    return {
        "key": method.key,
        "label": method.label,
        "point_path": str(method.point_path),
        "point_key": method.point_key,
        "method_type": method.method_type,
        "confidence_path": str(method.confidence_path) if method.confidence_path else None,
        "confidence_key": method.confidence_key,
        "normal_path": str(method.normal_path) if method.normal_path else None,
        "normal_key": method.normal_key,
        "head_patch_path": str(method.head_patch_path) if method.head_patch_path else None,
        "hand_patch_path": str(method.hand_patch_path) if method.hand_patch_path else None,
        "notes": method.notes,
    }


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    methods = [
        MethodSpec(
            key="v25_base_vggt",
            label="V25 base VGGT",
            point_path=ROOT / "output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_points_world.npz",
            point_key="frame0000",
            confidence_path=ROOT / "output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_confidence.npz",
            confidence_key="frame0000_world_points_conf",
            method_type="vggt_baseline",
            notes="base VGGT research prediction, human_prior_channels=0",
        ),
        MethodSpec(
            key="v42_prior_enabled",
            label="V42 prior-enabled VGGT",
            point_path=ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_points_world.npz",
            point_key="frame0000",
            confidence_path=ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_confidence.npz",
            confidence_key="frame0000_world_points_conf",
            normal_path=ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_normals_geometric.npz",
            normal_key="frame0000",
            method_type="vggt_prior_enabled",
            notes="prior-enabled prediction; this is the main point-map source used by V50R2",
        ),
        MethodSpec(
            key="v50r2_candidate",
            label="V50R2 candidate package",
            point_path=ROOT / "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__candidate_points.npz",
            point_key="candidate_points_world",
            confidence_path=ROOT / "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/v42_prior_enabled_payload__research_confidence.npz",
            confidence_key="frame0000_world_points_conf",
            normal_path=ROOT / "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__candidate_normals.npz",
            normal_key="candidate_normals_geometric",
            head_patch_path=ROOT / "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__head_face_patch.npz",
            hand_patch_path=ROOT / "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__hand_patch.npz",
            method_type="vggt_candidate_package",
            notes="strict candidate package; main point map is V42 frame0000 first six views plus packaged head/hand/normal evidence",
        ),
        MethodSpec(
            key="v16_smplx_prior_only",
            label="V16 SMPL-X prior only",
            point_path=ROOT / "output/surface_research_cloud_preflight/V16_smplx_native_prior_case_restored_dir/targets.npz",
            point_key="prior_points",
            confidence_path=None,
            confidence_key=None,
            normal_path=ROOT / "output/surface_research_cloud_preflight/V16_smplx_native_prior_case_restored_dir/targets.npz",
            normal_key="prior_normals",
            method_type="prior_reference_not_vggt_output",
            notes="SMPL-X prior-only reference, not a VGGT prediction and not a candidate pass",
        ),
    ]

    camera_ids, images, masks = load_scene()
    point_maps = {m.key: points_from_npz(m.point_path, m.point_key) for m in methods}
    conf_maps = {m.key: confidence_from_npz(m.confidence_path, m.confidence_key) for m in methods}
    head_patches = {m.key: load_npz(m.head_patch_path) if m.head_patch_path and m.head_patch_path.exists() else None for m in methods}
    hand_patches = {m.key: load_npz(m.hand_patch_path) if m.hand_patch_path and m.hand_patch_path.exists() else None for m in methods}
    normal_flags = {m.key: normal_available(m.normal_path, m.normal_key) for m in methods}
    baseline = point_maps["v25_base_vggt"]

    rows: list[dict[str, Any]] = []
    ply_records: dict[str, Any] = {}
    region_order = ["full", "head_face", "left_hand", "right_hand"]
    selected_views = [0, 2, 4]
    view_labels = [f"cam{camera_ids[i]}" for i in selected_views]

    for method in methods:
        pts = point_maps[method.key]
        conf = conf_maps[method.key]
        for vi, cam in enumerate(camera_ids):
            base_mask = masks[vi] & finite_points(pts[vi])
            regions = region_masks(base_mask, head_patch=head_patches[method.key], hand_patch=hand_patches[method.key], view=vi)
            for region, rmask in regions.items():
                rows.append(
                    metric_row(
                        method,
                        vi,
                        cam,
                        region,
                        pts[vi],
                        rmask,
                        baseline[vi] if method.key != "v25_base_vggt" and method.method_type != "prior_reference_not_vggt_output" else None,
                        conf[vi] if conf is not None else None,
                        normal_flags[method.key],
                    )
                )
                if region in region_order and vi in selected_views:
                    ply_path = PLY_DIR / f"{method.key}_cam{cam}_{region}.ply"
                    n = export_region_ply(ply_path, pts[vi], images[vi], rmask)
                    ply_records[f"{method.key}_cam{cam}_{region}"] = {"path": str(ply_path.resolve()), "points": n}

    agg = aggregate(rows)
    write_csv(CSV_REPORT, rows)

    images_out: dict[str, str] = {}
    for region in region_order:
        for color_mode in ["rgb", "depth"]:
            grid_rows = []
            for method in methods:
                panels = []
                pts = point_maps[method.key]
                for vi in selected_views:
                    base_mask = masks[vi] & finite_points(pts[vi])
                    regions = region_masks(base_mask, head_patch=head_patches[method.key], hand_patch=hand_patches[method.key], view=vi)
                    rmask = regions[region]
                    npts = int((rmask & finite_points(pts[vi])).sum())
                    panels.append(
                        draw_panel(
                            images[vi],
                            pts[vi],
                            rmask,
                            f"{method.label} / cam{camera_ids[vi]}",
                            f"{region}, points={npts:,}, {color_mode}",
                            color_mode=color_mode,
                            yaw_deg=-24 if vi % 2 == 0 else 18,
                            seed=10000 + vi,
                            max_points=9000 if region == "full" else 4500,
                            point_radius=1,
                        )
                    )
                grid_rows.append((method.label + "\n" + method.method_type, panels))
            out_path = IMG_DIR / f"v50r2_vertical_{region}_{color_mode}_comparison.png"
            make_grid(out_path, grid_rows, view_labels)
            images_out[f"{region}_{color_mode}"] = str(out_path.resolve())

    # Copy the highest-signal sheets into the mentor report image directory.
    mentor_img = ROOT / "output/mentor_report_v50r2/images"
    mentor_img.mkdir(parents=True, exist_ok=True)
    copied = {}
    copy_order = [
        ("09_vertical_full_body_rgb", "full_rgb"),
        ("10_vertical_head_face_rgb", "head_face_rgb"),
        ("11_vertical_hands_rgb", "hands_rgb"),
        ("12_vertical_full_body_depth", "full_depth"),
        ("13_vertical_head_face_depth", "head_face_depth"),
        ("14_vertical_hands_depth", "hands_depth"),
    ]
    # Build a combined hands sheet from left/right by stacking existing grids.
    hands_rgb = IMG_DIR / "v50r2_vertical_hands_rgb_comparison.png"
    hands_depth = IMG_DIR / "v50r2_vertical_hands_depth_comparison.png"
    for color_mode in ["rgb", "depth"]:
        left = Image.open(IMG_DIR / f"v50r2_vertical_left_hand_{color_mode}_comparison.png").convert("RGB")
        right = Image.open(IMG_DIR / f"v50r2_vertical_right_hand_{color_mode}_comparison.png").convert("RGB")
        combo = Image.new("RGB", (max(left.width, right.width), left.height + right.height + 20), "white")
        combo.paste(left, (0, 0))
        combo.paste(right, (0, left.height + 20))
        (hands_rgb if color_mode == "rgb" else hands_depth).parent.mkdir(parents=True, exist_ok=True)
        combo.save(hands_rgb if color_mode == "rgb" else hands_depth)
        images_out[f"hands_{color_mode}"] = str((hands_rgb if color_mode == "rgb" else hands_depth).resolve())

    source_map = {
        "full_rgb": IMG_DIR / "v50r2_vertical_full_rgb_comparison.png",
        "head_face_rgb": IMG_DIR / "v50r2_vertical_head_face_rgb_comparison.png",
        "hands_rgb": hands_rgb,
        "full_depth": IMG_DIR / "v50r2_vertical_full_depth_comparison.png",
        "head_face_depth": IMG_DIR / "v50r2_vertical_head_face_depth_comparison.png",
        "hands_depth": hands_depth,
    }
    for stem, key in copy_order:
        src = source_map[key]
        dst = mentor_img / f"{stem}.png"
        shutil.copy2(src, dst)
        copied[key] = str(dst.resolve())

    # Numeric high-level deltas against baseline.
    agg_by_key = {(r["method"], r["region"]): r for r in agg}
    summary_table = []
    for method in methods:
        for region in ["full", "head_face", "left_hand", "right_hand"]:
            r = agg_by_key.get((method.key, region), {})
            summary_table.append(
                {
                    "method": method.key,
                    "label": method.label,
                    "region": region,
                    "valid_points_mean": r.get("valid_points_mean"),
                    "coverage_mean": r.get("coverage_mean"),
                    "z_relief_p95_p5_mean": r.get("z_relief_p95_p5_mean"),
                    "neighbor_delta_p95_mean": r.get("neighbor_delta_p95_mean"),
                    "mean_l2_from_v25_baseline_mean": r.get("mean_l2_from_v25_baseline_mean"),
                    "normal_available": r.get("normal_available"),
                    "method_type": method.method_type,
                }
            )

    same_point_delta = float(np.max(np.abs(point_maps["v50r2_candidate"] - point_maps["v42_prior_enabled"][:6])))
    v42_baseline_delta = float(np.mean(np.linalg.norm(point_maps["v42_prior_enabled"][:6] - baseline, axis=-1)))
    report = {
        "task": "v50r2_vggt_vertical_baseline_comparison",
        "created_utc": now(),
        "status": "DONE_PASS",
        "scene": str(SCENE.resolve()),
        "camera_ids": camera_ids,
        "methods": [method_to_json(m) for m in methods],
        "images": images_out,
        "mentor_images": copied,
        "ply": ply_records,
        "metrics_csv": str(CSV_REPORT.resolve()),
        "aggregate_metrics": agg,
        "summary_table": summary_table,
        "key_findings": {
            "v42_vs_v25_mean_l2_on_all_pixels": v42_baseline_delta,
            "v50r2_candidate_points_vs_v42_max_abs": same_point_delta,
            "interpretation": "V50R2 candidate main point map is identical to V42 prior-enabled first-six-view frame0000 point map; V50R2 adds packaged head/hand/normal/visual-gate evidence rather than a new full-body point-map coordinate field.",
        },
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return "NA"
        try:
            if not np.isfinite(float(v)):
                return "NA"
            return f"{float(v):.6g}"
        except Exception:
            return str(v)

    lines = [
        "# V50R2 与 VGGT baseline 的纵向比较",
        "",
        "本轮比较只使用同一 case、同一 `frame0000`、同一 V42/V50R2 六视角协议：`" + ", ".join(camera_ids) + "`。这样可以避免再次出现 view / mask / camera id 混用导致的假对比。",
        "",
        "## 比较对象",
        "",
        "- `V25 base VGGT`：base VGGT research prediction，作为本轮 VGGT baseline。",
        "- `V42 prior-enabled VGGT`：接入 SMPL-X prior / HumanPriorAdapter 后的 prior-enabled prediction。",
        "- `V50R2 candidate package`：当前 strict candidate package。注意它的主 point map 与 V42 frame0000 前六视角完全一致，额外价值在 head/face patch、hand patch、normal evidence、visual gate 和 package/registry 交付。",
        "- `V16 SMPL-X prior only`：SMPL-X native prior-only reference，不是 VGGT 输出，不参与 baseline 胜负结论。",
        "",
        "## 关键数值结论",
        "",
        f"- V42 prior-enabled 与 V25 baseline 的全像素 mean L2 差异：`{v42_baseline_delta:.8f}`。",
        f"- V50R2 candidate 主点图与 V42 prior-enabled 前六视角 max abs 差异：`{same_point_delta:.8f}`。",
        "- 因此，如果只看主 point map 坐标，V50R2 不应被写成相对 V42 又发生了一次大幅几何提升；V50R2 的提升是 candidate package、normal/region evidence、visual gate 和正式解锁层面的闭环。",
        "",
        "## 区域均值表",
        "",
        "| method | region | valid_points_mean | coverage_mean | z_relief | neighbor_delta_p95 | L2 vs V25 | normal |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_table:
        lines.append(
            f"| {r['label']} | {r['region']} | {fmt(r['valid_points_mean'])} | {fmt(r['coverage_mean'])} | {fmt(r['z_relief_p95_p5_mean'])} | {fmt(r['neighbor_delta_p95_mean'])} | {fmt(r['mean_l2_from_v25_baseline_mean'])} | {r['normal_available']} |"
        )
    lines += [
        "",
        "## 点云图合集",
        "",
        "### Full body RGB-colored point cloud",
        f"![full rgb]({copied['full_rgb']})",
        "",
        "### Head / face RGB-colored point cloud",
        f"![head face rgb]({copied['head_face_rgb']})",
        "",
        "### Hands RGB-colored point cloud",
        f"![hands rgb]({copied['hands_rgb']})",
        "",
        "### Geometry/depth-colored checks",
        f"![full depth]({copied['full_depth']})",
        "",
        f"![head face depth]({copied['head_face_depth']})",
        "",
        f"![hands depth]({copied['hands_depth']})",
        "",
        "## 给导师的表述建议",
        "",
        "这一组结果说明：相比 base VGGT，prior-enabled 路线确实让输出进入了可审查、可打包、带 normal/region evidence 的 candidate 闭环；但从主 point map 的几何坐标看，V42 相对 V25 的变化幅度并不大，V50R2 主点图又与 V42 相同。因此当前版本不能夸大成“细节几何已经明显大幅超过 baseline”。更稳的结论是：baseline 的全身轮廓已经能出来，SMPL-X prior-enabled 路线主要补齐了人体区域、normal/region consistency、head/hand candidate evidence 和 D-line 交付闭环；下一步真正的改进空间仍在 head/face/hairline 局部表面、右手，以及能直接改变 target-view point map 的局部几何优化。",
        "",
        "## 输出文件",
        "",
        f"- JSON: `{JSON_REPORT.resolve()}`",
        f"- CSV: `{CSV_REPORT.resolve()}`",
        f"- Image dir: `{IMG_DIR.resolve()}`",
        f"- PLY dir: `{PLY_DIR.resolve()}`",
    ]
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
