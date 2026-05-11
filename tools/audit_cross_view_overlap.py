from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_fullbody_hand_integrity import hand_risk_mask  # noqa: E402
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402


POINT_SOURCES = ("world_points", "depth_unprojection")
ROI_MODES = ("full", "head", "face", "hands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit cross-view point overlap in shared world space. This is a "
            "read-only diagnostic for shell/fusion failures."
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rois", default="full,head,face,hands")
    parser.add_argument("--point-sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--sample-per-view", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--near-threshold", type=float, default=0.035)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_scene_rois(scene_dir: Path, shape: tuple[int, int], include_hands: bool) -> dict[str, np.ndarray]:
    roi_stacks: dict[str, list[np.ndarray]] = {roi: [] for roi in ROI_MODES if roi != "hands" or include_hands}
    view_idx = 0
    while True:
        try:
            scene = load_scene_view(scene_dir, view_idx, shape)
        except IndexError:
            break
        rois = build_roi_masks(scene.mask)
        for roi in ("full", "head", "face"):
            if roi in roi_stacks:
                roi_stacks[roi].append(rois[roi].astype(bool))
        if include_hands:
            hand_mask, _ = hand_risk_mask(scene.rgb, scene.mask)
            roi_stacks["hands"].append(hand_mask.astype(bool))
        view_idx += 1
    if view_idx == 0:
        raise RuntimeError(f"No scene views loaded from {scene_dir}")
    return {roi: np.stack(masks, axis=0) for roi, masks in roi_stacks.items()}


def resolve_points(data: np.lib.npyio.NpzFile, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "world_points":
        return np.asarray(data["world_points"], dtype=np.float32), np.asarray(data["world_points_conf"], dtype=np.float32)
    if source == "depth_unprojection":
        points = unproject_depth_map_to_point_map_numpy(data["depth"], data["extrinsic"], data["intrinsic"])
        return np.asarray(points, dtype=np.float32), np.asarray(data["depth_conf"], dtype=np.float32)
    raise ValueError(source)


def sample_points_by_view(
    *,
    points: np.ndarray,
    conf: np.ndarray,
    roi_mask: np.ndarray,
    conf_percentile: float,
    sample_per_view: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    per_view: list[np.ndarray] = []
    for view_idx in range(points.shape[0]):
        view_points = points[view_idx]
        view_conf = conf[view_idx]
        valid = (
            roi_mask[view_idx].astype(bool)
            & np.isfinite(view_points).all(axis=-1)
            & np.isfinite(view_conf)
            & (view_conf > 0)
        )
        if not np.any(valid):
            per_view.append(np.zeros((0, 3), dtype=np.float32))
            continue
        threshold = float(np.percentile(view_conf[valid], conf_percentile))
        keep = valid & (view_conf >= threshold)
        flat = view_points[keep].reshape(-1, 3)
        if len(flat) > int(sample_per_view):
            indices = rng.choice(len(flat), size=int(sample_per_view), replace=False)
            flat = flat[indices]
        per_view.append(np.asarray(flat, dtype=np.float32))
    return per_view


def distance_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def cross_view_stats(per_view: list[np.ndarray], near_threshold: float) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    all_dists: list[np.ndarray] = []
    for i, points_i in enumerate(per_view):
        if len(points_i) == 0:
            continue
        for j, points_j in enumerate(per_view):
            if i == j or len(points_j) == 0:
                continue
            tree = cKDTree(points_j)
            dists, _ = tree.query(points_i, k=1, workers=-1)
            dists = np.asarray(dists, dtype=np.float32)
            all_dists.append(dists)
            stats = distance_stats(dists)
            stats.update(
                {
                    "src_view": int(i),
                    "dst_view": int(j),
                    "near_frac": float((dists <= float(near_threshold)).mean()) if len(dists) else 0.0,
                }
            )
            pair_rows.append(stats)
    if all_dists:
        merged = np.concatenate(all_dists, axis=0)
    else:
        merged = np.zeros((0,), dtype=np.float32)
    summary = distance_stats(merged)
    summary["near_threshold"] = float(near_threshold)
    summary["near_frac"] = float((merged <= float(near_threshold)).mean()) if len(merged) else 0.0
    summary["pairs"] = pair_rows
    summary["per_view_points"] = [int(len(points)) for points in per_view]
    return summary


def same_view_source_delta(
    source_a: list[np.ndarray],
    source_b: list[np.ndarray],
    near_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_dists: list[np.ndarray] = []
    for view_idx, (points_a, points_b) in enumerate(zip(source_a, source_b)):
        if len(points_a) == 0 or len(points_b) == 0:
            rows.append({"view": int(view_idx), "count": 0})
            continue
        tree = cKDTree(points_b)
        dists, _ = tree.query(points_a, k=1, workers=-1)
        dists = np.asarray(dists, dtype=np.float32)
        all_dists.append(dists)
        stats = distance_stats(dists)
        stats.update(
            {
                "view": int(view_idx),
                "near_frac": float((dists <= float(near_threshold)).mean()) if len(dists) else 0.0,
            }
        )
        rows.append(stats)
    merged = np.concatenate(all_dists, axis=0) if all_dists else np.zeros((0,), dtype=np.float32)
    summary = distance_stats(merged)
    summary["near_threshold"] = float(near_threshold)
    summary["near_frac"] = float((merged <= float(near_threshold)).mean()) if len(merged) else 0.0
    summary["per_view"] = rows
    return summary


def write_markdown(payload: dict[str, Any], output_dir: Path) -> None:
    lines = [
        f"# Cross-View Overlap Audit: {payload['name']}",
        "",
        "This is a read-only local diagnostic. It samples each input view in a ROI and measures nearest-neighbor distance to other views in predicted world space. It does not claim pass.",
        "",
        f"- confidence percentile: `{payload['conf_percentile']}`",
        f"- sample per view: `{payload['sample_per_view']}`",
        f"- near threshold: `{payload['near_threshold']}`",
        "",
        "## Cross-View Summary",
        "",
        "| ROI | Source | Per-view points | NN p50 | NN p90 | Near frac |",
        "|---|---|---|---:|---:|---:|",
    ]
    for roi, by_source in payload["results"].items():
        for source, result in by_source.items():
            if source == "world_vs_depth_same_view":
                continue
            points = ",".join(str(v) for v in result.get("per_view_points", []))
            lines.append(
                f"| {roi} | {source} | {points} | {result.get('p50', '')} | {result.get('p90', '')} | {result.get('near_frac', '')} |"
            )
    lines.extend(
        [
            "",
            "## Same-View World vs Depth-Unprojection",
            "",
            "| ROI | p50 | p90 | Near frac |",
            "|---|---:|---:|---:|",
        ]
    )
    for roi, by_source in payload["results"].items():
        result = by_source.get("world_vs_depth_same_view")
        if not isinstance(result, dict):
            continue
        lines.append(f"| {roi} | {result.get('p50', '')} | {result.get('p90', '')} | {result.get('near_frac', '')} |")
    lines.extend(
        [
            "",
            "## Interpretation Reminder",
            "",
            "Low cross-view overlap indicates separated visible sheets or camera/world disagreement. A mentor-final candidate must still pass Open3D visual review for full/head/face/hands; this audit only locates a failure mechanism.",
            "",
        ]
    )
    (output_dir / "cross_view_overlap_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rois = parse_csv(args.rois)
    point_sources = parse_csv(args.point_sources)
    for roi in rois:
        if roi not in ROI_MODES:
            raise ValueError(f"Unsupported ROI: {roi}")
    for source in point_sources:
        if source not in POINT_SOURCES:
            raise ValueError(f"Unsupported point source: {source}")

    data = np.load(args.predictions_npz, allow_pickle=False)
    height, width = int(data["world_points"].shape[1]), int(data["world_points"].shape[2])
    roi_masks = load_scene_rois(Path(args.scene_dir).resolve(), (height, width), include_hands=("hands" in rois))
    rng = np.random.default_rng(int(args.seed))

    sampled: dict[str, dict[str, list[np.ndarray]]] = {}
    results: dict[str, dict[str, Any]] = {}
    for roi in rois:
        sampled[roi] = {}
        results[roi] = {}
        roi_mask = roi_masks[roi]
        for source in point_sources:
            points, conf = resolve_points(data, source)
            per_view = sample_points_by_view(
                points=points,
                conf=conf,
                roi_mask=roi_mask,
                conf_percentile=float(args.conf_percentile),
                sample_per_view=int(args.sample_per_view),
                rng=rng,
            )
            sampled[roi][source] = per_view
            results[roi][source] = cross_view_stats(per_view, near_threshold=float(args.near_threshold))
        if "world_points" in sampled[roi] and "depth_unprojection" in sampled[roi]:
            results[roi]["world_vs_depth_same_view"] = same_view_source_delta(
                sampled[roi]["world_points"],
                sampled[roi]["depth_unprojection"],
                near_threshold=float(args.near_threshold),
            )

    payload = {
        "name": args.name,
        "predictions_npz": str(Path(args.predictions_npz).resolve()),
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "conf_percentile": float(args.conf_percentile),
        "sample_per_view": int(args.sample_per_view),
        "near_threshold": float(args.near_threshold),
        "results": results,
    }
    (output_dir / "cross_view_overlap_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "report": str(output_dir / "cross_view_overlap_audit.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
