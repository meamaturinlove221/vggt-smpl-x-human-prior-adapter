from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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

from normal_line_multiview_eval import build_roi_masks, load_scene_view, parse_entry_spec  # noqa: E402
from render_open3d_pointcloud import resolve_point_source  # noqa: E402


ROI_ORDER = ("head", "face")
POINT_SOURCES = ("world_points", "depth_unprojection")
GATES = ("p40", "fixed")


@dataclass(frozen=True)
class FlatCloud:
    points: np.ndarray
    keep_flat: np.ndarray
    kept_indices: np.ndarray
    view_indices: np.ndarray
    y_indices: np.ndarray
    x_indices: np.ndarray
    roi2d_flat: dict[str, np.ndarray]
    filter_summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether 2D head/face ROI pixels that survive confidence "
            "filtering remain inside the fused 3D head/face ROI. This is a "
            "read-only diagnostic for shell/ROI-escape failures."
        )
    )
    parser.add_argument("--entry", action="append", required=True, help="name:predictions.npz:scene_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--point-sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--fixed-threshold", type=float, default=38.5067)
    parser.add_argument("--target-view", type=int, default=0)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_scene_roi_stack(scene_dir: Path, view_count: int, shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    masks: list[np.ndarray] = []
    rois: dict[str, list[np.ndarray]] = {"full": [], "head": [], "face": []}
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, shape)
        support = scene.mask.astype(bool)
        masks.append(support)
        view_rois = build_roi_masks(support)
        for roi in rois:
            rois[roi].append(view_rois[roi].astype(bool))
    return {roi: np.stack(items, axis=0) for roi, items in rois.items()}, np.stack(masks, axis=0)


def threshold_for(conf: np.ndarray, support: np.ndarray, gate: str, percentile: float, fixed_threshold: float) -> float:
    if gate == "fixed":
        return float(fixed_threshold)
    values = conf[support & np.isfinite(conf) & (conf > 0.0)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, float(percentile)))


def make_flat_cloud(
    *,
    points_map: np.ndarray,
    conf: np.ndarray,
    rois: dict[str, np.ndarray],
    gate: str,
    percentile: float,
    fixed_threshold: float,
) -> FlatCloud:
    finite = np.isfinite(points_map).all(axis=-1)
    support = rois["full"] & finite & np.isfinite(conf) & (conf > 0.0)
    threshold = threshold_for(conf, support, gate, percentile, fixed_threshold)
    keep = support & (conf >= threshold)
    flat_keep = keep.reshape(-1)
    kept_indices = np.flatnonzero(flat_keep)
    view_count, height, width = points_map.shape[:3]
    v, y, x = np.unravel_index(kept_indices, (view_count, height, width))
    return FlatCloud(
        points=points_map.reshape(-1, 3)[kept_indices].astype(np.float32, copy=False),
        keep_flat=flat_keep,
        kept_indices=kept_indices,
        view_indices=np.asarray(v, dtype=np.int32),
        y_indices=np.asarray(y, dtype=np.int32),
        x_indices=np.asarray(x, dtype=np.int32),
        roi2d_flat={roi: rois[roi].reshape(-1) for roi in rois},
        filter_summary={
            "gate": gate,
            "threshold": threshold,
            "valid_points_before_conf": int(support.sum()),
            "points_after_conf": int(flat_keep.sum()),
        },
    )


def fused_roi_membership(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] == 0:
        empty = np.zeros((0,), dtype=bool)
        return {"head": empty, "face": empty, "summary": {"points": 0}}
    height_like = -points[:, 1]
    head_cut = float(np.percentile(height_like, 78.0))
    head = height_like >= head_cut
    if int(head.sum()) < 512:
        head_cut = float(np.percentile(height_like, 68.0))
        head = height_like >= head_cut

    face = np.zeros_like(head, dtype=bool)
    summary: dict[str, Any] = {
        "points": int(points.shape[0]),
        "head_cut_height_like": head_cut,
        "head_count": int(head.sum()),
    }
    head_points = points[head]
    if len(head_points) >= 256:
        x_lo, x_hi = np.percentile(head_points[:, 0], [20.0, 80.0])
        z_lo, z_hi = np.percentile(head_points[:, 2], [15.0, 85.0])
        head_height_like = -head_points[:, 1]
        height_lo = float(np.percentile(head_height_like, 25.0))
        face = (
            head
            & (points[:, 0] >= float(x_lo))
            & (points[:, 0] <= float(x_hi))
            & (points[:, 2] >= float(z_lo))
            & (points[:, 2] <= float(z_hi))
            & (height_like >= height_lo)
        )
        summary.update(
            {
                "face_count": int(face.sum()),
                "face_x_lo": float(x_lo),
                "face_x_hi": float(x_hi),
                "face_z_lo": float(z_lo),
                "face_z_hi": float(z_hi),
                "face_height_like_lo": height_lo,
            }
        )
    else:
        summary["face_count"] = 0
        summary["fallback"] = "head_too_small"
    return {"head": head, "face": face, "summary": summary}


def summarize_escape(cloud: FlatCloud, membership: dict[str, Any], roi: str) -> dict[str, Any]:
    flat_roi = cloud.roi2d_flat[roi][cloud.kept_indices]
    kept_2d = int(flat_roi.sum())
    in_head = membership["head"] & flat_roi
    in_face = membership["face"] & flat_roi
    escaped_head = flat_roi & ~membership["head"]
    escaped_face = flat_roi & ~membership["face"]
    out: dict[str, Any] = {
        "roi": roi,
        "kept_2d_points": kept_2d,
        "in_fused_head_points": int(in_head.sum()),
        "in_fused_head_ratio": float(in_head.sum() / max(kept_2d, 1)),
        "in_fused_face_points": int(in_face.sum()),
        "in_fused_face_ratio": float(in_face.sum() / max(kept_2d, 1)),
        "escaped_head_points": int(escaped_head.sum()),
        "escaped_face_points": int(escaped_face.sum()),
    }
    if kept_2d > 0:
        points = cloud.points[flat_roi]
        escaped = cloud.points[escaped_face] if roi == "face" else cloud.points[escaped_head]
        out.update(
            {
                "points_x_p10": float(np.percentile(points[:, 0], 10.0)),
                "points_x_p90": float(np.percentile(points[:, 0], 90.0)),
                "points_y_p10": float(np.percentile(points[:, 1], 10.0)),
                "points_y_p90": float(np.percentile(points[:, 1], 90.0)),
                "points_z_p10": float(np.percentile(points[:, 2], 10.0)),
                "points_z_p90": float(np.percentile(points[:, 2], 90.0)),
            }
        )
        if escaped.shape[0] > 0:
            out.update(
                {
                    "escaped_x_p10": float(np.percentile(escaped[:, 0], 10.0)),
                    "escaped_x_p90": float(np.percentile(escaped[:, 0], 90.0)),
                    "escaped_y_p10": float(np.percentile(escaped[:, 1], 10.0)),
                    "escaped_y_p90": float(np.percentile(escaped[:, 1], 90.0)),
                    "escaped_z_p10": float(np.percentile(escaped[:, 2], 10.0)),
                    "escaped_z_p90": float(np.percentile(escaped[:, 2], 90.0)),
                }
            )
    return out


def overlay_view(
    *,
    scene_dir: Path,
    view_idx: int,
    shape: tuple[int, int],
    cloud: FlatCloud,
    membership: dict[str, Any],
    roi: str,
    path: Path,
) -> None:
    scene = load_scene_view(scene_dir, view_idx, shape)
    roi_mask = build_roi_masks(scene.mask.astype(bool))[roi]
    good = np.zeros(shape, dtype=bool)
    escaped = np.zeros(shape, dtype=bool)
    selected = (cloud.view_indices == int(view_idx)) & cloud.roi2d_flat[roi][cloud.kept_indices]
    if roi == "face":
        good_selected = selected & membership["face"]
        escaped_selected = selected & ~membership["face"]
    else:
        good_selected = selected & membership["head"]
        escaped_selected = selected & ~membership["head"]
    good[cloud.y_indices[good_selected], cloud.x_indices[good_selected]] = True
    escaped[cloud.y_indices[escaped_selected], cloud.x_indices[escaped_selected]] = True

    overlay = np.asarray(scene.rgb, dtype=np.float32).copy()
    roi_only = roi_mask & ~(good | escaped)
    overlay[roi_only] = overlay[roi_only] * 0.55 + np.array([160, 160, 160], dtype=np.float32) * 0.45
    overlay[good] = overlay[good] * 0.40 + np.array([60, 220, 80], dtype=np.float32) * 0.60
    overlay[escaped] = overlay[escaped] * 0.40 + np.array([250, 45, 45], dtype=np.float32) * 0.60
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 430, 70), fill=(255, 255, 255))
    draw.text((8, 8), f"{roi}: green=2D kept and in fused 3D ROI", fill=(0, 0, 0))
    draw.text((8, 34), f"red=2D kept but escaped fused 3D ROI; good={int(good.sum())} escaped={int(escaped.sum())}", fill=(0, 0, 0))
    image.save(path)


def audit_entry(
    *,
    entry_text: str,
    point_sources: list[str],
    percentile: float,
    fixed_threshold: float,
    target_view: int,
    output_dir: Path,
) -> dict[str, Any]:
    spec = parse_entry_spec(entry_text)
    data = np.load(spec.predictions_npz, allow_pickle=False)
    view_count = int(np.asarray(data["world_points"]).shape[0])
    shape = tuple(int(v) for v in np.asarray(data["world_points"]).shape[1:3])
    rois, masks = load_scene_roi_stack(spec.scene_dir, view_count, shape)
    target_view = max(0, min(int(target_view), view_count - 1))
    rows: list[dict[str, Any]] = []
    overlays: list[str] = []
    source_payload: dict[str, Any] = {}
    for source in point_sources:
        points_map, conf = resolve_point_source(data, source)
        source_payload[source] = {}
        for gate in GATES:
            cloud = make_flat_cloud(
                points_map=np.asarray(points_map, dtype=np.float32),
                conf=np.asarray(conf, dtype=np.float32),
                rois=rois,
                gate=gate,
                percentile=percentile,
                fixed_threshold=fixed_threshold,
            )
            membership = fused_roi_membership(cloud.points)
            gate_payload = {"filter": cloud.filter_summary, "fused_roi": membership["summary"], "roi_escape": {}}
            for roi in ROI_ORDER:
                row = {
                    "entry": spec.name,
                    "source": source,
                    "gate": gate,
                    **summarize_escape(cloud, membership, roi),
                }
                rows.append(row)
                gate_payload["roi_escape"][roi] = row
                overlay_path = output_dir / f"{spec.name}_{source}_{gate}_{roi}_view{target_view:02d}_escape.png"
                overlay_view(
                    scene_dir=spec.scene_dir,
                    view_idx=target_view,
                    shape=shape,
                    cloud=cloud,
                    membership=membership,
                    roi=roi,
                    path=overlay_path,
                )
                overlays.append(str(overlay_path))
            source_payload[source][gate] = gate_payload
    return {
        "name": spec.name,
        "predictions_npz": str(spec.predictions_npz),
        "scene_dir": str(spec.scene_dir),
        "rows": rows,
        "sources": source_payload,
        "overlays": overlays,
        "mask_pixels": {roi: int(mask.sum()) for roi, mask in rois.items()},
    }


def write_csv(path: Path, payloads: list[dict[str, Any]]) -> None:
    fields = [
        "entry",
        "source",
        "gate",
        "roi",
        "kept_2d_points",
        "in_fused_head_points",
        "in_fused_head_ratio",
        "in_fused_face_points",
        "in_fused_face_ratio",
        "escaped_head_points",
        "escaped_face_points",
        "points_x_p10",
        "points_x_p90",
        "points_y_p10",
        "points_y_p90",
        "points_z_p10",
        "points_z_p90",
        "escaped_x_p10",
        "escaped_x_p90",
        "escaped_y_p10",
        "escaped_y_p90",
        "escaped_z_p10",
        "escaped_z_p90",
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def write_markdown(path: Path, payloads: list[dict[str, Any]]) -> None:
    rows = [row for payload in payloads for row in payload["rows"]]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ROI 3D Escape Audit\n\n")
        handle.write(
            "This read-only diagnostic asks whether 2D head/face pixels that pass confidence filtering also land inside the fused 3D head/face ROI. It is not a candidate pass.\n\n"
        )
        handle.write(
            "| Entry | Source | Gate | ROI | 2D kept | In fused head | Head ratio | In fused face | Face ratio | Escaped face |\n"
        )
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["entry"]),
                        str(row["source"]),
                        str(row["gate"]),
                        str(row["roi"]),
                        str(row["kept_2d_points"]),
                        str(row["in_fused_head_points"]),
                        fmt(row["in_fused_head_ratio"]),
                        str(row["in_fused_face_points"]),
                        fmt(row["in_fused_face_ratio"]),
                        str(row["escaped_face_points"]),
                    ]
                )
                + " |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A candidate can increase 2D ROI confidence while still failing mentor geometry if those 2D points escape the fused 3D head/face selection or form shell-like sheets. Use this with Open3D visual review; do not use it to tune thresholds into a pseudo-pass.\n"
        )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_sources = parse_csv(args.point_sources)
    for source in point_sources:
        if source not in POINT_SOURCES:
            raise ValueError(f"Unsupported point source: {source}")
    payloads = [
        audit_entry(
            entry_text=entry,
            point_sources=point_sources,
            percentile=float(args.conf_percentile),
            fixed_threshold=float(args.fixed_threshold),
            target_view=int(args.target_view),
            output_dir=output_dir,
        )
        for entry in args.entry
    ]
    summary = {
        "conf_percentile": float(args.conf_percentile),
        "fixed_threshold": float(args.fixed_threshold),
        "target_view": int(args.target_view),
        "entries": payloads,
    }
    (output_dir / "roi_3d_escape_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "roi_3d_escape_rows.csv", payloads)
    write_markdown(output_dir / "roi_3d_escape_audit.md", payloads)
    print(json.dumps({"output_dir": str(output_dir), "report": str(output_dir / "roi_3d_escape_audit.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
