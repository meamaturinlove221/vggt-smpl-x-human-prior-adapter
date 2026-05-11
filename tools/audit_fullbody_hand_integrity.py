from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit full-body bottom-line integrity and hand-risk regions for a "
            "VGGT prediction. This is a gate report: it does not patch predictions."
        )
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--point-source", choices=("world_points", "depth_unprojection"), default="world_points")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.018)
    parser.add_argument("--cluster-eps", type=float, default=0.055)
    parser.add_argument("--cluster-min-points", type=int, default=4)
    parser.add_argument("--vertical-bins", type=int, default=10)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.86)
    parser.add_argument("--min-vertical-bin-ratio", type=float, default=0.018)
    parser.add_argument("--min-hand-kept-ratio", type=float, default=0.25)
    parser.add_argument("--min-hand-components", type=int, default=1)
    return parser.parse_args()


def percentile_threshold(conf: np.ndarray, valid: np.ndarray, percentile: float) -> float:
    values = conf[valid & np.isfinite(conf) & (conf > 0.0)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, float(percentile)))


def connected_component_stats_2d(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total <= 0:
        return {"components": 0, "largest_component_pixels": 0, "largest_component_ratio": 0.0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return {"components": 0, "largest_component_pixels": 0, "largest_component_ratio": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(areas.max())
    return {
        "components": int(count - 1),
        "largest_component_pixels": largest,
        "largest_component_ratio": float(largest / max(total, 1)),
    }


def largest_components(mask: np.ndarray, min_pixels: int = 32, max_components: int = 4) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_pixels):
            components.append((area, label))
    components.sort(reverse=True)
    return [labels == label for _, label in components[: int(max_components)]]


def skin_mask(rgb: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin = (cr > 130) & (cr < 180) & (cb > 70) & (cb < 135) & (y > 40)
    skin &= body_mask.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    skin = cv2.dilate(skin.astype(np.uint8), kernel, iterations=1).astype(bool)
    return skin & body_mask.astype(bool)


def hand_risk_mask(rgb: np.ndarray, body_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rois = build_roi_masks(body_mask.astype(bool))
    head = rois["head"]
    face = rois["face"]
    candidates = skin_mask(rgb, body_mask) & ~head & ~face

    # Drop very low leg pixels for the headshoulder/fullbody mixed crop. This
    # keeps the mask focused on hands/forearms while still surfacing a warning
    # when bare limbs dominate the view.
    ys = np.nonzero(body_mask)[0]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        lower_cut = int(y0 + 0.86 * max(1, y1 - y0 + 1))
        candidates &= np.arange(body_mask.shape[0])[:, None] <= lower_cut

    comps = largest_components(candidates, min_pixels=24, max_components=4)
    out = np.zeros_like(candidates, dtype=bool)
    for comp in comps:
        out |= comp
    return out, {
        "raw_skin_extremity_pixels": int(candidates.sum()),
        "selected_components": len(comps),
        "selected_pixels": int(out.sum()),
        "component_pixels": [int(comp.sum()) for comp in comps],
    }


def save_overlay(path: Path, rgb: np.ndarray, body: np.ndarray, kept: np.ndarray, hand: np.ndarray) -> None:
    out = rgb.astype(np.float32).copy()
    body_only = body & ~kept
    out[body_only] = out[body_only] * 0.68 + np.array([40, 90, 255], dtype=np.float32) * 0.32
    out[kept] = out[kept] * 0.45 + np.array([0, 220, 80], dtype=np.float32) * 0.55
    out[hand] = out[hand] * 0.35 + np.array([255, 120, 20], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def point_cloud_cluster_stats(points: np.ndarray, voxel_size: float, eps: float, min_points: int) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {
            "points": 0,
            "downsampled_points": 0,
            "clusters": 0,
            "largest_cluster_points": 0,
            "largest_component_ratio": 0.0,
        }
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if float(voxel_size) > 0.0:
            cloud = cloud.voxel_down_sample(float(voxel_size))
        down_count = int(np.asarray(cloud.points).shape[0])
        if down_count == 0:
            return {
                "points": int(points.shape[0]),
                "downsampled_points": 0,
                "clusters": 0,
                "largest_cluster_points": 0,
                "largest_component_ratio": 0.0,
            }
        labels = np.asarray(
            cloud.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False),
            dtype=np.int64,
        )
        valid = labels >= 0
        if not np.any(valid):
            return {
                "points": int(points.shape[0]),
                "downsampled_points": down_count,
                "clusters": 0,
                "noise_points": int((~valid).sum()),
                "largest_cluster_points": 0,
                "largest_component_ratio": 0.0,
            }
        counts = np.bincount(labels[valid])
        largest = int(counts.max())
        return {
            "points": int(points.shape[0]),
            "downsampled_points": down_count,
            "clusters": int(len(counts)),
            "noise_points": int((~valid).sum()),
            "largest_cluster_points": largest,
            "largest_component_ratio": float(largest / max(down_count, 1)),
        }
    except Exception as exc:  # pragma: no cover - fallback for envs without Open3D.
        return {
            "points": int(points.shape[0]),
            "error": f"{type(exc).__name__}: {exc}",
            "largest_component_ratio": 0.0,
        }


def vertical_band_stats(points: np.ndarray, bins: int) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"bins": int(bins), "counts": [], "min_bin_ratio": 0.0, "empty_bins": int(bins)}
    height_like = -points[:, 1]
    lo, hi = np.percentile(height_like, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"bins": int(bins), "counts": [], "min_bin_ratio": 0.0, "empty_bins": int(bins)}
    hist, edges = np.histogram(np.clip(height_like, lo, hi), bins=int(bins), range=(float(lo), float(hi)))
    ratios = hist.astype(np.float64) / max(int(hist.sum()), 1)
    return {
        "bins": int(bins),
        "height_like_range_p01_p99": [float(lo), float(hi)],
        "counts": [int(v) for v in hist.tolist()],
        "ratios": [float(v) for v in ratios.tolist()],
        "min_bin_ratio": float(ratios.min()) if ratios.size else 0.0,
        "empty_bins": int((hist == 0).sum()),
    }


def load_scene_stacks(scene_dir: Path, view_count: int, hw: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, hw)
        rgbs.append(scene.rgb)
        masks.append(scene.mask.astype(bool))
    return rgbs, masks


def main() -> int:
    args = parse_args()
    pred_path = Path(args.predictions_npz)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pred_path, allow_pickle=False)
    if str(args.point_source) == "world_points":
        points_map = np.asarray(data["world_points"], dtype=np.float32)
        conf = np.asarray(data["world_points_conf"], dtype=np.float32)
    else:
        points_map = unproject_depth_map_to_point_map_numpy(
            np.asarray(data["depth"], dtype=np.float32),
            np.asarray(data["extrinsic"], dtype=np.float32),
            np.asarray(data["intrinsic"], dtype=np.float32),
        )
        conf = np.asarray(data["depth_conf"], dtype=np.float32)

    view_count, height, width, _ = points_map.shape
    rgbs, masks_list = load_scene_stacks(scene_dir, view_count, (height, width))
    masks = np.stack(masks_list, axis=0)
    finite = np.isfinite(points_map).all(axis=-1)
    support = masks & finite & np.isfinite(conf) & (conf > 0.0)
    threshold = (
        float(args.conf_threshold)
        if args.conf_threshold is not None
        else percentile_threshold(conf, support, float(args.conf_percentile))
    )
    kept = support & (conf >= threshold)
    points = points_map[kept]

    per_view: dict[str, Any] = {}
    hand_views_ok = 0
    for view_idx in range(view_count):
        body = masks[view_idx]
        kept_view = kept[view_idx] & body
        rois = build_roi_masks(body)
        hand_mask, hand_summary = hand_risk_mask(rgbs[view_idx], body)
        hand_support_pixels = int(hand_mask.sum())
        hand_kept_pixels = int((kept_view & hand_mask).sum())
        hand_ratio = float(hand_kept_pixels / max(hand_support_pixels, 1)) if hand_support_pixels else 0.0
        hand_ok = bool(hand_support_pixels > 0 and hand_ratio >= float(args.min_hand_kept_ratio))
        hand_views_ok += int(hand_ok)

        y0y1 = np.nonzero(body)[0]
        band_stats: list[dict[str, Any]] = []
        if y0y1.size:
            y0, y1 = int(y0y1.min()), int(y0y1.max())
            band_edges = np.linspace(y0, y1 + 1, int(args.vertical_bins) + 1).astype(int)
            for band_idx in range(int(args.vertical_bins)):
                y0b, y1b = int(band_edges[band_idx]), int(band_edges[band_idx + 1])
                band = body.copy()
                band[:y0b, :] = False
                band[y1b:, :] = False
                pixels = int(band.sum())
                kept_pixels = int((kept_view & band).sum())
                band_stats.append(
                    {
                        "band": band_idx,
                        "pixels": pixels,
                        "kept_pixels": kept_pixels,
                        "kept_ratio": float(kept_pixels / max(pixels, 1)) if pixels else 0.0,
                    }
                )

        save_overlay(
            output_dir / f"view_{view_idx:02d}_fullbody_hand_overlay.png",
            rgbs[view_idx],
            body,
            kept_view,
            hand_mask,
        )
        per_view[str(view_idx)] = {
            "body_pixels": int(body.sum()),
            "kept_body_pixels": int(kept_view.sum()),
            "body_kept_ratio": float(kept_view.sum() / max(body.sum(), 1)),
            "body_components": connected_component_stats_2d(kept_view),
            "head_kept_ratio": float((kept_view & rois["head"]).sum() / max(int(rois["head"].sum()), 1)),
            "face_kept_ratio": float((kept_view & rois["face"]).sum() / max(int(rois["face"].sum()), 1)),
            "vertical_bands": band_stats,
            "hand_risk": {
                **hand_summary,
                "kept_pixels": hand_kept_pixels,
                "kept_ratio": hand_ratio,
                "gate_ok": hand_ok,
            },
            "overlay": str(output_dir / f"view_{view_idx:02d}_fullbody_hand_overlay.png"),
        }

    cluster = point_cloud_cluster_stats(
        points,
        voxel_size=float(args.voxel_size),
        eps=float(args.cluster_eps),
        min_points=int(args.cluster_min_points),
    )
    vertical = vertical_band_stats(points, int(args.vertical_bins))
    full_gate = {
        "nonzero_points": bool(points.shape[0] > 0),
        "largest_component_ratio": bool(
            cluster.get("largest_component_ratio", 0.0) >= float(args.min_largest_component_ratio)
        ),
        "vertical_coverage": bool(vertical.get("empty_bins", int(args.vertical_bins)) == 0)
        and bool(vertical.get("min_bin_ratio", 0.0) >= float(args.min_vertical_bin_ratio)),
    }
    hand_gate = {
        "eligible_views_with_hand_candidates": int(
            sum(1 for row in per_view.values() if row["hand_risk"]["selected_pixels"] > 0)
        ),
        "views_passing_hand_kept_ratio": int(hand_views_ok),
        "pass": bool(hand_views_ok >= int(args.min_hand_components)),
    }
    full_gate["pass"] = bool(all(full_gate.values()))

    summary = {
        "predictions_npz": str(pred_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "point_source": str(args.point_source),
        "conf_threshold": threshold,
        "conf_threshold_source": "absolute" if args.conf_threshold is not None else "percentile",
        "conf_percentile": None if args.conf_threshold is not None else float(args.conf_percentile),
        "points_after_conf": int(points.shape[0]),
        "cluster": cluster,
        "vertical_3d": vertical,
        "full_body_gate": full_gate,
        "hand_gate": hand_gate,
        "per_view": per_view,
        "truthful_note": (
            "Full-body and hand-risk audit is a bottom-line screen. A pass here does not replace "
            "Open3D visual review of full/head/face/hands, and a face/head shell remains a failure."
        ),
    }
    (output_dir / "fullbody_hand_integrity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if full_gate["pass"] and hand_gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
