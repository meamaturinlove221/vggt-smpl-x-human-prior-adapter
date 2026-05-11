from __future__ import annotations

import argparse
import json
import sys
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

from normal_line_multiview_eval import (  # noqa: E402
    ROI_ORDER,
    build_roi_masks,
    depth_to_camera_points,
    load_prediction_bundle,
    load_scene_view,
    normalize_target_view,
    parse_entry_spec,
)
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why a candidate loses face/head ROI points against a baseline."
    )
    parser.add_argument("--baseline", required=True, help="name:predictions.npz:scene_dir entry")
    parser.add_argument("--candidate", required=True, help="name:predictions.npz:scene_dir entry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--fixed-threshold", type=float, default=38.5067)
    parser.add_argument("--target-view", type=int, default=0)
    parser.add_argument(
        "--normal-format",
        choices=("auto", "vector", "rgb01", "rgb255"),
        default="auto",
    )
    return parser.parse_args()


def percentile_keep(conf: np.ndarray, support: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    valid = support & np.isfinite(conf) & (conf > 0.0)
    if not np.any(valid):
        return valid.copy(), float("nan")
    threshold = float(np.percentile(conf[valid], percentile))
    return valid & (conf >= threshold), threshold


def fixed_keep(conf: np.ndarray, support: np.ndarray, threshold: float) -> np.ndarray:
    return support & np.isfinite(conf) & (conf > 0.0) & (conf >= float(threshold))


def load_scene_masks(scene_dir: Path, view_count: int, target_hw: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    masks: list[np.ndarray] = []
    paths: list[str] = []
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, target_hw)
        masks.append(scene.mask.astype(bool))
        paths.append(str(scene.mask_path))
    return np.stack(masks, axis=0), paths


def filtered_points(world_points: np.ndarray, world_conf: np.ndarray, masks: np.ndarray, percentile: float) -> tuple[np.ndarray, dict[str, Any]]:
    points = world_points.reshape(-1, 3)
    conf = world_conf.reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0.0) & masks.reshape(-1)
    if not np.any(valid):
        raise RuntimeError("No valid points after filtering")
    threshold = float(np.percentile(conf[valid], percentile))
    keep = valid & (conf >= threshold)
    if not np.any(keep):
        keep = valid
    return points[keep], {
        "valid_points_before_conf": int(valid.sum()),
        "conf_threshold": threshold,
        "points_after_conf": int(keep.sum()),
    }


def apply_3d_roi(points: np.ndarray, roi: str) -> dict[str, Any]:
    if roi == "full":
        return {"roi": roi, "points_after_roi": int(len(points))}
    if len(points) < 32:
        return {"roi": roi, "fallback": "too_few_points", "points_after_roi": int(len(points))}
    height_like = -points[:, 1]
    head_percentile = 78.0 if roi == "head" else 74.0
    head_cut = float(np.percentile(height_like, head_percentile))
    head_mask = height_like >= head_cut
    if int(head_mask.sum()) < 512:
        head_cut = float(np.percentile(height_like, 68.0))
        head_mask = height_like >= head_cut
    roi_mask = head_mask
    summary: dict[str, Any] = {
        "roi": roi,
        "vertical_axis": "-y_is_up",
        "head_cut_height_like": head_cut,
        "points_after_head_cut": int(head_mask.sum()),
    }
    if roi == "face":
        head_points = points[head_mask]
        if len(head_points) >= 256:
            x_lo, x_hi = np.percentile(head_points[:, 0], [20.0, 80.0])
            z_lo, z_hi = np.percentile(head_points[:, 2], [15.0, 85.0])
            head_height_like = -head_points[:, 1]
            height_lo = float(np.percentile(head_height_like, 25.0))
            face_mask = (
                head_mask
                & (points[:, 0] >= float(x_lo))
                & (points[:, 0] <= float(x_hi))
                & (points[:, 2] >= float(z_lo))
                & (points[:, 2] <= float(z_hi))
                & (height_like >= height_lo)
            )
            if int(face_mask.sum()) >= 128:
                roi_mask = face_mask
                summary.update(
                    {
                        "x_lo": float(x_lo),
                        "x_hi": float(x_hi),
                        "z_lo": float(z_lo),
                        "z_hi": float(z_hi),
                        "face_height_like_lo": height_lo,
                    }
                )
            else:
                summary["fallback"] = "face_mask_too_small"
        else:
            summary["fallback"] = "head_mask_too_small"
    summary["points_after_roi"] = int(roi_mask.sum())
    return summary


def summarize_values(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float32)[mask & np.isfinite(values)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.percentile(selected, 10.0)),
        "p40": float(np.percentile(selected, 40.0)),
        "p90": float(np.percentile(selected, 90.0)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
    }


def summarize_depth_point_delta(bundle: Any, masks: np.ndarray, roi_masks: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for roi_name in ROI_ORDER:
        deltas: list[np.ndarray] = []
        z_deltas: list[np.ndarray] = []
        for view_idx in range(bundle.world_points.shape[0]):
            world_points = bundle.world_points[view_idx]
            if bundle.extrinsic is None:
                point_camera = world_points
            else:
                point_camera = points_world_to_camera(world_points, bundle.extrinsic[view_idx])
            intrinsic = bundle.intrinsic[view_idx] if bundle.intrinsic is not None else np.eye(3, dtype=np.float32)
            depth_points = depth_to_camera_points(bundle.depth[view_idx], intrinsic)
            roi = roi_masks[view_idx][roi_name]
            valid = roi & masks[view_idx] & np.isfinite(point_camera).all(axis=-1) & np.isfinite(depth_points).all(axis=-1)
            if np.any(valid):
                diff = point_camera[valid] - depth_points[valid]
                deltas.append(np.linalg.norm(diff, axis=-1).astype(np.float32))
                z_deltas.append(np.abs(diff[:, 2]).astype(np.float32))
        if deltas:
            all_delta = np.concatenate(deltas)
            all_z = np.concatenate(z_deltas)
            result[roi_name] = {
                "l2_mean": float(np.mean(all_delta)),
                "l2_median": float(np.median(all_delta)),
                "l2_p90": float(np.percentile(all_delta, 90.0)),
                "z_abs_mean": float(np.mean(all_z)),
                "z_abs_median": float(np.median(all_z)),
                "count": int(all_delta.size),
            }
        else:
            result[roi_name] = {"count": 0}
    return result


def overlay_lost_new(rgb: np.ndarray, baseline_keep: np.ndarray, candidate_keep: np.ndarray, roi: np.ndarray, path: Path) -> None:
    base = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    overlay = np.array(base, dtype=np.float32)
    roi_only = roi & ~(baseline_keep | candidate_keep)
    kept_both = roi & baseline_keep & candidate_keep
    lost = roi & baseline_keep & ~candidate_keep
    new = roi & ~baseline_keep & candidate_keep
    overlay[roi_only] = overlay[roi_only] * 0.55 + np.array([150, 150, 150], dtype=np.float32) * 0.45
    overlay[kept_both] = overlay[kept_both] * 0.50 + np.array([80, 220, 80], dtype=np.float32) * 0.50
    overlay[lost] = overlay[lost] * 0.40 + np.array([255, 40, 40], dtype=np.float32) * 0.60
    overlay[new] = overlay[new] * 0.40 + np.array([40, 120, 255], dtype=np.float32) * 0.60
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 360, 64), fill=(255, 255, 255))
    draw.text((8, 8), "green=both red=lost blue=new gray=roi", fill=(0, 0, 0))
    draw.text((8, 32), f"lost={int(lost.sum())} new={int(new.sum())} both={int(kept_both.sum())}", fill=(0, 0, 0))
    image.save(path)


def draw_histograms(hist_data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    width, height = 1200, 720
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [
        ("world_points_conf", (margin, 60, width - margin, 250)),
        ("depth_conf", (margin, 280, width - margin, 470)),
        ("normal_conf", (margin, 500, width - margin, 690)),
    ]
    colors = {"baseline": (220, 40, 40), "candidate": (40, 100, 230)}
    for field, box in panels:
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(0, 0, 0))
        draw.text((x0, y0 - 24), field, fill=(0, 0, 0))
        values = [hist_data[name][field] for name in ("baseline", "candidate") if hist_data[name][field].size > 0]
        if not values:
            continue
        joined = np.concatenate(values)
        lo = float(np.percentile(joined, 1.0))
        hi = float(np.percentile(joined, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 50)
        max_count = 1
        hists: dict[str, np.ndarray] = {}
        for name in ("baseline", "candidate"):
            vals = hist_data[name][field]
            hist, _ = np.histogram(vals[np.isfinite(vals)], bins=bins)
            hists[name] = hist
            max_count = max(max_count, int(hist.max()))
        for name, hist in hists.items():
            color = colors[name]
            points = []
            for idx, count in enumerate(hist):
                x = x0 + int((idx / max(1, len(hist) - 1)) * (x1 - x0))
                y = y1 - int((float(count) / max_count) * (y1 - y0 - 8))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
        draw.text((x1 - 260, y0 + 8), "baseline", fill=colors["baseline"])
        draw.text((x1 - 160, y0 + 8), "candidate", fill=colors["candidate"])
        draw.text((x0 + 4, y1 + 2), f"{lo:.3f}", fill=(0, 0, 0))
        draw.text((x1 - 70, y1 + 2), f"{hi:.3f}", fill=(0, 0, 0))
    image.save(path)


def analyze_entry(spec_text: str, normal_format: str, masks: np.ndarray) -> tuple[Any, dict[str, Any], list[dict[str, np.ndarray]]]:
    spec = parse_entry_spec(spec_text)
    bundle = load_prediction_bundle(spec.predictions_npz, normal_format)
    roi_masks: list[dict[str, np.ndarray]] = []
    for view_idx in range(bundle.normal.shape[0]):
        roi_masks.append(build_roi_masks(masks[view_idx]))
    points, filter_summary = filtered_points(bundle.world_points, bundle.world_points_conf, masks, percentile=40.0)
    roi_summary = {roi_name: apply_3d_roi(points, roi_name) for roi_name in ROI_ORDER}
    return bundle, {"name": spec.name, "filter_summary": filter_summary, "roi_summary": roi_summary}, roi_masks


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_spec = parse_entry_spec(args.baseline)
    candidate_spec = parse_entry_spec(args.candidate)
    baseline_bundle = load_prediction_bundle(baseline_spec.predictions_npz, args.normal_format)
    target_hw = tuple(int(value) for value in baseline_bundle.normal.shape[1:3])
    masks, mask_paths = load_scene_masks(baseline_spec.scene_dir, baseline_bundle.normal.shape[0], target_hw)

    baseline_bundle, baseline_summary, baseline_rois = analyze_entry(args.baseline, args.normal_format, masks)
    candidate_bundle, candidate_summary, candidate_rois = analyze_entry(args.candidate, args.normal_format, masks)
    view_idx = normalize_target_view(int(args.target_view), baseline_bundle.normal.shape[0])
    scene_view = load_scene_view(baseline_spec.scene_dir, view_idx, target_hw)

    base_p40_keep, base_p40 = percentile_keep(baseline_bundle.world_points_conf, masks, float(args.conf_percentile))
    cand_p40_keep, cand_p40 = percentile_keep(candidate_bundle.world_points_conf, masks, float(args.conf_percentile))
    base_fixed_keep = fixed_keep(baseline_bundle.world_points_conf, masks, float(args.fixed_threshold))
    cand_fixed_keep = fixed_keep(candidate_bundle.world_points_conf, masks, float(args.fixed_threshold))

    summary: dict[str, Any] = {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "scene_dir": str(baseline_spec.scene_dir),
        "mask_paths": mask_paths,
        "conf_percentile": float(args.conf_percentile),
        "baseline_p40_threshold": base_p40,
        "candidate_p40_threshold": cand_p40,
        "fixed_threshold": float(args.fixed_threshold),
        "roi_2d": {},
        "depth_point_delta": {
            "baseline": summarize_depth_point_delta(baseline_bundle, masks, baseline_rois),
            "candidate": summarize_depth_point_delta(candidate_bundle, masks, candidate_rois),
        },
        "outputs": {},
    }

    hist_data = {
        "baseline": {"world_points_conf": [], "depth_conf": [], "normal_conf": []},
        "candidate": {"world_points_conf": [], "depth_conf": [], "normal_conf": []},
    }
    for roi_name in ROI_ORDER:
        base_roi_stats: dict[str, Any] = {}
        cand_roi_stats: dict[str, Any] = {}
        for label, bundle, p40_keep, fixed_keep_map, roi_list, stats_out in (
            ("baseline", baseline_bundle, base_p40_keep, base_fixed_keep, baseline_rois, base_roi_stats),
            ("candidate", candidate_bundle, cand_p40_keep, cand_fixed_keep, candidate_rois, cand_roi_stats),
        ):
            roi_all = np.stack([roi_list[idx][roi_name] for idx in range(len(roi_list))], axis=0)
            stats_out["roi_pixels"] = int(roi_all.sum())
            stats_out["p40_kept"] = int((p40_keep & roi_all).sum())
            stats_out["fixed_kept"] = int((fixed_keep_map & roi_all).sum())
            stats_out["world_points_conf"] = summarize_values(bundle.world_points_conf, roi_all)
            stats_out["depth_conf"] = summarize_values(bundle.depth_conf, roi_all)
            stats_out["normal_conf"] = summarize_values(bundle.normal_conf, roi_all)
            if roi_name == "face":
                for field in ("world_points_conf", "depth_conf", "normal_conf"):
                    hist_data[label][field].append(np.asarray(getattr(bundle, field)[roi_all], dtype=np.float32))
        summary["roi_2d"][roi_name] = {
            "baseline": base_roi_stats,
            "candidate": cand_roi_stats,
            "p40_lost": int((base_p40_keep & ~cand_p40_keep & np.stack([baseline_rois[idx][roi_name] for idx in range(len(baseline_rois))], axis=0)).sum()),
            "p40_new": int((~base_p40_keep & cand_p40_keep & np.stack([baseline_rois[idx][roi_name] for idx in range(len(baseline_rois))], axis=0)).sum()),
            "fixed_lost": int((base_fixed_keep & ~cand_fixed_keep & np.stack([baseline_rois[idx][roi_name] for idx in range(len(baseline_rois))], axis=0)).sum()),
            "fixed_new": int((~base_fixed_keep & cand_fixed_keep & np.stack([baseline_rois[idx][roi_name] for idx in range(len(baseline_rois))], axis=0)).sum()),
        }

    hist_arrays = {
        name: {field: np.concatenate(values) if values else np.asarray([], dtype=np.float32) for field, values in fields.items()}
        for name, fields in hist_data.items()
    }
    hist_path = output_dir / "face_confidence_histograms.png"
    draw_histograms(hist_arrays, hist_path)
    summary["outputs"]["face_confidence_histograms"] = str(hist_path)

    view_face = baseline_rois[view_idx]["face"]
    overlay_p40 = output_dir / f"view{view_idx:02d}_face_p40_kept_lost_new.png"
    overlay_fixed = output_dir / f"view{view_idx:02d}_face_fixed_kept_lost_new.png"
    overlay_lost_new(scene_view.rgb, base_p40_keep[view_idx], cand_p40_keep[view_idx], view_face, overlay_p40)
    overlay_lost_new(scene_view.rgb, base_fixed_keep[view_idx], cand_fixed_keep[view_idx], view_face, overlay_fixed)
    summary["outputs"]["face_p40_overlay"] = str(overlay_p40)
    summary["outputs"]["face_fixed_overlay"] = str(overlay_fixed)

    json_path = output_dir / "face_roi_failure_diagnosis.json"
    md_path = output_dir / "face_roi_failure_diagnosis.md"
    summary["outputs"]["json"] = str(json_path)
    summary["outputs"]["markdown"] = str(md_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Face ROI failure diagnosis\n\n")
        handle.write(f"Baseline: `{baseline_summary['name']}`\n\n")
        handle.write(f"Candidate: `{candidate_summary['name']}`\n\n")
        handle.write(f"Baseline p40 threshold: `{base_p40:.4f}`\n\n")
        handle.write(f"Candidate p40 threshold: `{cand_p40:.4f}`\n\n")
        handle.write("## Same-protocol 3D ROI\n\n")
        handle.write("| Entry | Full | Head | Face |\n|---|---:|---:|---:|\n")
        for label, item in (("baseline", baseline_summary), ("candidate", candidate_summary)):
            roi = item["roi_summary"]
            handle.write(
                f"| {label} | {roi['full']['points_after_roi']} | {roi['head']['points_after_roi']} | {roi['face']['points_after_roi']} |\n"
            )
        handle.write("\n## 2D ROI confidence gates\n\n")
        handle.write("| ROI | base p40 kept | cand p40 kept | p40 lost | p40 new | base fixed | cand fixed | fixed lost | fixed new |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for roi_name in ROI_ORDER:
            roi = summary["roi_2d"][roi_name]
            handle.write(
                f"| {roi_name} | {roi['baseline']['p40_kept']} | {roi['candidate']['p40_kept']} | "
                f"{roi['p40_lost']} | {roi['p40_new']} | {roi['baseline']['fixed_kept']} | "
                f"{roi['candidate']['fixed_kept']} | {roi['fixed_lost']} | {roi['fixed_new']} |\n"
            )
        handle.write("\n## Depth vs point camera-space delta\n\n")
        handle.write("| Entry | ROI | L2 mean | L2 median | L2 p90 | Z abs mean |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for label in ("baseline", "candidate"):
            for roi_name, stats in summary["depth_point_delta"][label].items():
                handle.write(
                    f"| {label} | {roi_name} | {stats.get('l2_mean', '')} | {stats.get('l2_median', '')} | "
                    f"{stats.get('l2_p90', '')} | {stats.get('z_abs_mean', '')} |\n"
                )
        handle.write("\n## Outputs\n\n")
        for key, value in summary["outputs"].items():
            handle.write(f"- `{key}`: `{value}`\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {overlay_p40}")
    print(f"Wrote {overlay_fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
