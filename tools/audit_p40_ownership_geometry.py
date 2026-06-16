from __future__ import annotations

import argparse
import csv
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

from audit_fullbody_hand_integrity import hand_risk_mask  # noqa: E402
from normal_line_multiview_eval import (  # noqa: E402
    build_roi_masks,
    depth_to_camera_points,
    load_prediction_bundle,
    load_scene_view,
    parse_entry_spec,
    point_map_to_normal_numpy,
)
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


ROI_ORDER = ("full", "head", "face", "hands")
POINT_SOURCES = ("world_points", "depth_unprojection")
GROUP_ORDER = ("both", "lost", "new", "candidate_only", "baseline_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether p40/fixed kept points are owned by coherent geometry or by high-confidence fragments. "
            "This is read-only and does not claim pass."
        )
    )
    parser.add_argument("--baseline", required=True, help="name:predictions.npz:scene_dir entry")
    parser.add_argument("--candidate", required=True, help="name:predictions.npz:scene_dir entry")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--fixed-threshold", type=float, default=38.5067)
    parser.add_argument("--normal-format", choices=("auto", "vector", "rgb01", "rgb255"), default="auto")
    parser.add_argument("--target-view", type=int, default=0)
    return parser.parse_args()


def largest_component_ratio(mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return None
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        largest = max(largest, size)
    return float(largest / max(total, 1))


def central_protrusion(points: np.ndarray) -> float | None:
    if points.shape[0] < 128:
        return None
    x_vals = points[:, 0]
    z_vals = points[:, 2]
    x20, x40, x60, x80 = np.percentile(x_vals, [20.0, 40.0, 60.0, 80.0])
    center = (x_vals >= x40) & (x_vals <= x60)
    sides = (x_vals <= x20) | (x_vals >= x80)
    if int(center.sum()) < 16 or int(sides.sum()) < 16:
        return None
    return float(np.median(z_vals[sides]) - np.median(z_vals[center]))


def pca_metrics(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] < 8:
        return {"pca_count": int(points.shape[0])}
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eig = np.maximum(np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1], 0.0)
    eig0 = max(float(eig[0]), 1e-12)
    return {
        "pca_count": int(points.shape[0]),
        "pca_eig0": float(eig[0]),
        "pca_eig1": float(eig[1]),
        "pca_eig2": float(eig[2]),
        "pca_thinness_sqrt_e2_e0": float(np.sqrt(float(eig[2]) / eig0)),
        "pca_planarity_e2_over_sum": float(float(eig[2]) / max(float(np.sum(eig)), 1e-12)),
    }


def summarize_points(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": int(points.shape[0]),
        "x_range_p05_p95": float(np.percentile(points[:, 0], 95.0) - np.percentile(points[:, 0], 5.0)),
        "y_range_p05_p95": float(np.percentile(points[:, 1], 95.0) - np.percentile(points[:, 1], 5.0)),
        "z_range_p05_p95": float(np.percentile(points[:, 2], 95.0) - np.percentile(points[:, 2], 5.0)),
        "z_median": float(np.median(points[:, 2])),
        "z_p10": float(np.percentile(points[:, 2], 10.0)),
        "z_p90": float(np.percentile(points[:, 2], 90.0)),
        "central_protrusion_z": central_protrusion(points),
    }
    out.update(pca_metrics(points))
    return out


def threshold_for(conf: np.ndarray, support: np.ndarray, gate: str, percentile: float, fixed_threshold: float) -> float:
    if gate == "fixed":
        return float(fixed_threshold)
    valid = support & np.isfinite(conf) & (conf > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.percentile(conf[valid], percentile))


def camera_points_for_source(bundle: Any, view_idx: int, source: str) -> np.ndarray:
    if source == "depth_unprojection":
        intrinsic = bundle.intrinsic[view_idx] if bundle.intrinsic is not None else np.eye(3, dtype=np.float32)
        return depth_to_camera_points(bundle.depth[view_idx], intrinsic)
    world = bundle.world_points[view_idx]
    if bundle.extrinsic is None:
        return world.astype(np.float32, copy=False)
    return points_world_to_camera(world, bundle.extrinsic[view_idx])


def confidence_for_source(bundle: Any, source: str) -> np.ndarray:
    return bundle.depth_conf if source == "depth_unprojection" else bundle.world_points_conf


def normal_for_source(bundle: Any, view_idx: int, source: str, support: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_camera = camera_points_for_source(bundle, view_idx, source)
    conf = confidence_for_source(bundle, source)[view_idx]
    valid = (
        support
        & keep
        & np.isfinite(conf)
        & (conf > 0.0)
        & np.isfinite(points_camera).all(axis=-1)
    )
    normal, surface_valid = point_map_to_normal_numpy(points_camera, valid)
    norm = np.linalg.norm(normal, axis=-1)
    vector_valid = np.isfinite(normal).all(axis=-1) & (norm > 1e-6)
    normal = normal / np.maximum(norm[..., None], 1e-6)
    return normal.astype(np.float32), surface_valid & vector_valid


def scene_rois(scene_dir: Path, view_idx: int, shape: tuple[int, int], include_hands: bool) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    scene = load_scene_view(scene_dir, view_idx, shape)
    support = scene.mask.astype(bool)
    rois = build_roi_masks(support)
    if include_hands:
        hand_mask, _ = hand_risk_mask(scene.rgb, support)
        rois["hands"] = hand_mask.astype(bool)
    return support, rois, scene.rgb.astype(np.uint8)


def group_masks(base_keep: np.ndarray, cand_keep: np.ndarray) -> dict[str, np.ndarray]:
    both = base_keep & cand_keep
    lost = base_keep & ~cand_keep
    new = ~base_keep & cand_keep
    return {
        "both": both,
        "lost": lost,
        "new": new,
        "candidate_only": cand_keep,
        "baseline_only": base_keep,
    }


def group_normal_delta(
    base_normal: np.ndarray,
    base_valid: np.ndarray,
    cand_normal: np.ndarray,
    cand_valid: np.ndarray,
    group: np.ndarray,
) -> dict[str, Any]:
    valid = group & base_valid & cand_valid
    if not np.any(valid):
        return {"normal_valid_pixels": 0}
    dot = np.sum(base_normal[valid] * cand_normal[valid], axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    abs_angle = np.degrees(np.arccos(np.abs(dot)))
    signed_angle = np.degrees(np.arccos(dot))
    return {
        "normal_valid_pixels": int(valid.sum()),
        "normal_abs_angle_mean_deg": float(np.mean(abs_angle)),
        "normal_abs_angle_p90_deg": float(np.percentile(abs_angle, 90.0)),
        "normal_signed_cos_mean": float(np.mean(dot)),
        "normal_signed_negative_frac": float(np.mean(dot < 0.0)),
        "normal_signed_angle_mean_deg": float(np.mean(signed_angle)),
    }


def overlay_ownership(rgb: np.ndarray, roi: np.ndarray, base_keep: np.ndarray, cand_keep: np.ndarray, path: Path) -> None:
    image = rgb.astype(np.float32).copy()
    both = roi & base_keep & cand_keep
    lost = roi & base_keep & ~cand_keep
    new = roi & ~base_keep & cand_keep
    roi_only = roi & ~(both | lost | new)
    image[roi_only] = image[roi_only] * 0.60 + np.array([150, 150, 150], dtype=np.float32) * 0.40
    image[both] = image[both] * 0.45 + np.array([70, 220, 70], dtype=np.float32) * 0.55
    image[lost] = image[lost] * 0.35 + np.array([255, 45, 45], dtype=np.float32) * 0.65
    image[new] = image[new] * 0.35 + np.array([40, 120, 255], dtype=np.float32) * 0.65
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, 450, 70), fill=(255, 255, 255))
    draw.text((8, 8), "green=both red=lost blue=new gray=roi-only", fill=(0, 0, 0))
    draw.text((8, 34), f"both={int(both.sum())} lost={int(lost.sum())} new={int(new.sum())}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def audit() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    base_spec = parse_entry_spec(args.baseline)
    cand_spec = parse_entry_spec(args.candidate)
    base = load_prediction_bundle(base_spec.predictions_npz, args.normal_format)
    cand = load_prediction_bundle(cand_spec.predictions_npz, args.normal_format)
    if base.world_points.shape[:3] != cand.world_points.shape[:3]:
        raise ValueError(f"Shape mismatch: {base.world_points.shape} vs {cand.world_points.shape}")
    shape = tuple(int(v) for v in base.world_points.shape[1:3])
    view_count = int(base.world_points.shape[0])

    rows: list[dict[str, Any]] = []
    overlay_paths: list[str] = []
    for view_idx in range(view_count):
        support, rois, rgb = scene_rois(base_spec.scene_dir, view_idx, shape, include_hands=True)
        for source in POINT_SOURCES:
            base_conf = confidence_for_source(base, source)[view_idx]
            cand_conf = confidence_for_source(cand, source)[view_idx]
            base_points = camera_points_for_source(base, view_idx, source)
            cand_points = camera_points_for_source(cand, view_idx, source)
            for gate in ("p40", "fixed"):
                base_thr = threshold_for(base_conf, support, gate, args.conf_percentile, args.fixed_threshold)
                cand_thr = threshold_for(cand_conf, support, gate, args.conf_percentile, args.fixed_threshold)
                base_keep = support & np.isfinite(base_conf) & (base_conf > 0.0) & (base_conf >= base_thr)
                cand_keep = support & np.isfinite(cand_conf) & (cand_conf > 0.0) & (cand_conf >= cand_thr)
                base_normal, base_normal_valid = normal_for_source(base, view_idx, source, support, base_keep)
                cand_normal, cand_normal_valid = normal_for_source(cand, view_idx, source, support, cand_keep)
                for roi_name in ROI_ORDER:
                    roi = rois.get(roi_name)
                    if roi is None:
                        continue
                    masks = group_masks(base_keep & roi, cand_keep & roi)
                    if view_idx == int(args.target_view) and roi_name in {"face", "head", "hands"}:
                        overlay_path = overlay_dir / f"view{view_idx:02d}_{roi_name}_{source}_{gate}_ownership.png"
                        overlay_ownership(rgb, roi, base_keep, cand_keep, overlay_path)
                        overlay_paths.append(str(overlay_path))
                    for group_name in GROUP_ORDER:
                        group = masks[group_name]
                        selected = group & np.isfinite(cand_points).all(axis=-1)
                        baseline_selected = group & np.isfinite(base_points).all(axis=-1)
                        metric_points = cand_points[selected] if group_name in {"new", "candidate_only", "both"} else base_points[baseline_selected]
                        point_summary = summarize_points(metric_points)
                        normal_summary = group_normal_delta(
                            base_normal,
                            base_normal_valid,
                            cand_normal,
                            cand_normal_valid,
                            group,
                        )
                        roi_pixels = int(roi.sum())
                        rows.append(
                            {
                                "view": int(view_idx),
                                "roi": roi_name,
                                "source": source,
                                "gate": gate,
                                "group": group_name,
                                "baseline_threshold": float(base_thr) if np.isfinite(base_thr) else None,
                                "candidate_threshold": float(cand_thr) if np.isfinite(cand_thr) else None,
                                "roi_pixels": roi_pixels,
                                "pixels": int(group.sum()),
                                "coverage": float(group.sum() / max(roi_pixels, 1)),
                                "largest_component_ratio_2d": largest_component_ratio(group),
                                **point_summary,
                                **normal_summary,
                            }
                        )

    summary: dict[str, Any] = {
        "task": "p40_ownership_geometry_audit",
        "truthful_status": "diagnostic_only_not_candidate_gate_not_mentor_pass",
        "baseline": {
            "name": base_spec.name,
            "predictions_npz": str(base_spec.predictions_npz),
            "scene_dir": str(base_spec.scene_dir),
        },
        "candidate": {
            "name": cand_spec.name,
            "predictions_npz": str(cand_spec.predictions_npz),
            "scene_dir": str(cand_spec.scene_dir),
        },
        "rows": rows,
        "overlay_paths": overlay_paths,
    }
    summary_path = out_dir / "p40_ownership_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "p40_ownership_geometry_rows.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(summary, out_dir)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


def mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def sum_int(values: list[Any]) -> int:
    return int(sum(int(v) for v in values if v is not None))


def write_markdown(summary: dict[str, Any], out_dir: Path) -> None:
    rows = list(summary["rows"])
    lines = [
        f"# P40 Ownership Geometry Audit: {summary['candidate']['name']}",
        "",
        "This is a read-only diagnostic. It explains whether p40/fixed kept pixels correspond to coherent geometry or high-confidence fragments. It is not a candidate pass.",
        "",
        f"- baseline: `{summary['baseline']['name']}`",
        f"- candidate: `{summary['candidate']['name']}`",
        "",
        "## Target-View Face/Head/Hair-Relevant Groups",
        "",
        "| ROI | Source | Gate | Group | Pixels | Coverage | LCC2D | Z range | Thinness | Central protrusion | Normal abs angle |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if int(row["view"]) != 0 or row["roi"] not in {"head", "face", "hands"}:
            continue
        if row["group"] not in {"both", "lost", "new", "candidate_only"}:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["roi"]),
                    str(row["source"]),
                    str(row["gate"]),
                    str(row["group"]),
                    str(row["pixels"]),
                    f"{float(row['coverage']):.4f}",
                    "" if row.get("largest_component_ratio_2d") is None else f"{float(row['largest_component_ratio_2d']):.4f}",
                    "" if row.get("z_range_p05_p95") is None else f"{float(row['z_range_p05_p95']):.4f}",
                    "" if row.get("pca_thinness_sqrt_e2_e0") is None else f"{float(row['pca_thinness_sqrt_e2_e0']):.4f}",
                    "" if row.get("central_protrusion_z") is None else f"{float(row['central_protrusion_z']):.4f}",
                    "" if row.get("normal_abs_angle_mean_deg") is None else f"{float(row['normal_abs_angle_mean_deg']):.4f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## All-View Aggregate", ""])
    lines.append("| ROI | Source | Gate | Group | Pixels | Mean coverage | Mean LCC2D | Mean z range | Mean thinness | Mean normal abs angle |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for roi in ROI_ORDER:
        for source in POINT_SOURCES:
            for gate in ("p40", "fixed"):
                for group in GROUP_ORDER:
                    subset = [row for row in rows if row["roi"] == roi and row["source"] == source and row["gate"] == gate and row["group"] == group]
                    if not subset:
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                roi,
                                source,
                                gate,
                                group,
                                str(sum_int([row.get("pixels") for row in subset])),
                                "" if mean([row.get("coverage") for row in subset]) is None else f"{mean([row.get('coverage') for row in subset]):.4f}",
                                "" if mean([row.get("largest_component_ratio_2d") for row in subset]) is None else f"{mean([row.get('largest_component_ratio_2d') for row in subset]):.4f}",
                                "" if mean([row.get("z_range_p05_p95") for row in subset]) is None else f"{mean([row.get('z_range_p05_p95') for row in subset]):.4f}",
                                "" if mean([row.get("pca_thinness_sqrt_e2_e0") for row in subset]) is None else f"{mean([row.get('pca_thinness_sqrt_e2_e0') for row in subset]):.4f}",
                                "" if mean([row.get("normal_abs_angle_mean_deg") for row in subset]) is None else f"{mean([row.get('normal_abs_angle_mean_deg') for row in subset]):.4f}",
                            ]
                        )
                        + " |"
                    )
    lines.extend(["", "## Overlays", ""])
    for path in summary.get("overlay_paths", []):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "Blue/new pixels are not automatically improvements. They must also be spatially connected, have plausible thickness/protrusion, and survive the full Open3D visual gate. Red/lost pixels reveal where a candidate removed previously kept support.",
            "",
        ]
    )
    (out_dir / "p40_ownership_geometry_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(audit())
