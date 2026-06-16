from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    parse_entry_spec,
)
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure simple full/head/face shape metrics beyond point counts."
    )
    parser.add_argument("--entry", action="append", required=True, help="name:predictions.npz:scene_dir entry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--fixed-threshold", type=float, default=38.5067)
    parser.add_argument("--normal-format", choices=("auto", "vector", "rgb01", "rgb255"), default="auto")
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
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
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


def camera_points_for_source(bundle: Any, view_idx: int, point_source: str) -> np.ndarray:
    if point_source == "depth_unprojection":
        intrinsic = bundle.intrinsic[view_idx] if bundle.intrinsic is not None else np.eye(3, dtype=np.float32)
        return depth_to_camera_points(bundle.depth[view_idx], intrinsic)
    world = bundle.world_points[view_idx]
    if bundle.extrinsic is None:
        return world.astype(np.float32, copy=False)
    return points_world_to_camera(world, bundle.extrinsic[view_idx])


def confidence_for_source(bundle: Any, point_source: str) -> np.ndarray:
    return bundle.depth_conf if point_source == "depth_unprojection" else bundle.world_points_conf


def threshold_for_source(conf: np.ndarray, support: np.ndarray, gate: str, percentile: float, fixed_threshold: float) -> float:
    if gate == "fixed":
        return float(fixed_threshold)
    valid = support & np.isfinite(conf) & (conf > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.percentile(conf[valid], percentile))


def pca_metrics(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] < 8:
        return {"pca_count": int(points.shape[0])}
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eig = np.linalg.eigvalsh(cov).astype(np.float64)
    eig = np.maximum(np.sort(eig)[::-1], 0.0)
    largest = max(float(eig[0]), 1e-12)
    return {
        "pca_count": int(points.shape[0]),
        "pca_eig0": float(eig[0]),
        "pca_eig1": float(eig[1]),
        "pca_eig2": float(eig[2]),
        "pca_thinness_sqrt_e2_e0": float(np.sqrt(float(eig[2]) / largest)),
        "pca_planarity_e2_over_sum": float(float(eig[2]) / max(float(np.sum(eig)), 1e-12)),
    }


def central_protrusion(points: np.ndarray) -> float | None:
    if points.shape[0] < 128:
        return None
    x = points[:, 0]
    z = points[:, 2]
    x20, x40, x60, x80 = np.percentile(x, [20.0, 40.0, 60.0, 80.0])
    center = (x >= x40) & (x <= x60)
    sides = (x <= x20) | (x >= x80)
    if int(center.sum()) < 16 or int(sides.sum()) < 16:
        return None
    # Positive means the face center is closer to the camera than the side bands.
    return float(np.median(z[sides]) - np.median(z[center]))


def summarize_points(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": int(points.shape[0]),
        "x_range": float(np.percentile(points[:, 0], 95.0) - np.percentile(points[:, 0], 5.0)),
        "y_range": float(np.percentile(points[:, 1], 95.0) - np.percentile(points[:, 1], 5.0)),
        "z_range": float(np.percentile(points[:, 2], 95.0) - np.percentile(points[:, 2], 5.0)),
        "z_median": float(np.median(points[:, 2])),
        "z_p10": float(np.percentile(points[:, 2], 10.0)),
        "z_p90": float(np.percentile(points[:, 2], 90.0)),
        "central_protrusion_z": central_protrusion(points),
    }
    result.update(pca_metrics(points))
    return result


def row_for(
    *,
    entry: str,
    view_idx: int,
    roi_name: str,
    point_source: str,
    gate: str,
    threshold: float,
    roi_mask: np.ndarray,
    keep_mask: np.ndarray,
    points_camera: np.ndarray,
) -> dict[str, Any]:
    selected_mask = roi_mask & keep_mask & np.isfinite(points_camera).all(axis=-1)
    selected_points = points_camera[selected_mask]
    metrics = summarize_points(selected_points)
    lcc = largest_component_ratio(selected_mask)
    roi_pixels = int(roi_mask.sum())
    selected_pixels = int(selected_mask.sum())
    return {
        "entry": entry,
        "view": view_idx,
        "roi": roi_name,
        "point_source": point_source,
        "gate": gate,
        "threshold": threshold,
        "roi_pixels": roi_pixels,
        "selected_pixels": selected_pixels,
        "coverage": float(selected_pixels / max(roi_pixels, 1)),
        "largest_component_ratio": lcc,
        **metrics,
    }


def audit_entry(entry_text: str, normal_format: str, percentile: float, fixed_threshold: float) -> list[dict[str, Any]]:
    spec = parse_entry_spec(entry_text)
    bundle = load_prediction_bundle(spec.predictions_npz, normal_format)
    target_hw = tuple(int(value) for value in bundle.normal.shape[1:3])
    rows: list[dict[str, Any]] = []
    for view_idx in range(bundle.normal.shape[0]):
        scene = load_scene_view(spec.scene_dir, view_idx, target_hw)
        support = scene.mask.astype(bool)
        roi_masks = build_roi_masks(support)
        for point_source in ("world_points", "depth_unprojection"):
            conf = confidence_for_source(bundle, point_source)[view_idx]
            points_camera = camera_points_for_source(bundle, view_idx, point_source)
            for gate in ("p40", "fixed"):
                threshold = threshold_for_source(conf, support, gate, percentile, fixed_threshold)
                keep = support & np.isfinite(conf) & (conf > 0.0) & (conf >= threshold)
                for roi_name in ROI_ORDER:
                    rows.append(
                        row_for(
                            entry=spec.name,
                            view_idx=view_idx,
                            roi_name=roi_name,
                            point_source=point_source,
                            gate=gate,
                            threshold=threshold,
                            roi_mask=roi_masks[roi_name],
                            keep_mask=keep,
                            points_camera=points_camera,
                        )
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "entry",
        "view",
        "roi",
        "point_source",
        "gate",
        "threshold",
        "roi_pixels",
        "selected_pixels",
        "coverage",
        "largest_component_ratio",
        "count",
        "x_range",
        "y_range",
        "z_range",
        "z_median",
        "z_p10",
        "z_p90",
        "central_protrusion_z",
        "pca_eig0",
        "pca_eig1",
        "pca_eig2",
        "pca_thinness_sqrt_e2_e0",
        "pca_planarity_e2_over_sum",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["entry"], row["roi"], row["point_source"], row["gate"])
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (entry, roi, source, gate), items in sorted(grouped.items()):
        summaries.append(
            {
                "entry": entry,
                "roi": roi,
                "point_source": source,
                "gate": gate,
                "mean_selected_pixels": mean_or_none([item.get("selected_pixels") for item in items]),
                "mean_coverage": mean_or_none([item.get("coverage") for item in items]),
                "mean_largest_component_ratio": mean_or_none([item.get("largest_component_ratio") for item in items]),
                "mean_z_range": mean_or_none([item.get("z_range") for item in items]),
                "mean_central_protrusion_z": mean_or_none([item.get("central_protrusion_z") for item in items]),
                "mean_pca_thinness": mean_or_none([item.get("pca_thinness_sqrt_e2_e0") for item in items]),
            }
        )
    return summaries


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Face/head shape metrics\n\n")
        handle.write(
            "| Entry | ROI | Source | Gate | Mean pixels | Coverage | LCC ratio | Z range | Center protrusion | PCA thinness |\n"
        )
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["entry"]),
                        str(row["roi"]),
                        str(row["point_source"]),
                        str(row["gate"]),
                        fmt(row["mean_selected_pixels"]),
                        fmt(row["mean_coverage"]),
                        fmt(row["mean_largest_component_ratio"]),
                        fmt(row["mean_z_range"]),
                        fmt(row["mean_central_protrusion_z"]),
                        fmt(row["mean_pca_thinness"]),
                    ]
                )
                + " |\n"
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry_text in args.entry:
        rows.extend(audit_entry(entry_text, args.normal_format, float(args.conf_percentile), float(args.fixed_threshold)))
    summaries = grouped_summary(rows)
    json_path = output_dir / "face_shape_metrics.json"
    csv_path = output_dir / "face_shape_metrics.csv"
    md_path = output_dir / "face_shape_metrics.md"
    json_path.write_text(
        json.dumps(
            {
                "conf_percentile": float(args.conf_percentile),
                "fixed_threshold": float(args.fixed_threshold),
                "rows": rows,
                "summary": summaries,
                "outputs": {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    write_markdown(md_path, summaries)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
