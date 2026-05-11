from __future__ import annotations

import argparse
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

from render_open3d_pointcloud import load_2d_roi_mask_stack, load_mask_stack  # noqa: E402
from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse VGGT predictions through SMPL-X canonical correspondences. "
            "SMPL-X is used only as a cross-view correspondence index, never as "
            "a face/hair/clothing geometry teacher."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roi", choices=("full", "head", "face", "head_face"), default="head")
    parser.add_argument("--canonical-bin-size", type=float, default=0.018)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--min-bin-count", type=int, default=3)
    parser.add_argument("--min-bin-views", type=int, default=2)
    parser.add_argument("--max-bin-spread", type=float, default=0.075)
    parser.add_argument("--max-apply-distance", type=float, default=0.090)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--visible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence-boost", type=float, default=0.0)
    parser.add_argument("--normal-confidence-boost", type=float, default=0.0)
    parser.add_argument("--normal-dilate", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def channel_index(channel_names: list[str], name: str) -> int:
    try:
        return channel_names.index(name)
    except ValueError as exc:
        raise KeyError(f"Missing prior channel {name!r}; available={channel_names}") from exc


def roi_stack(scene_dir: Path, target_size: int, roi: str) -> np.ndarray:
    if roi == "full":
        return load_mask_stack(scene_dir / "masks", target_size=target_size).astype(bool)
    if roi == "head_face":
        head = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="head").astype(bool)
        face = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="face").astype(bool)
        return head | face
    return load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=roi).astype(bool)


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return np.einsum("...j,ij->...i", points_world.astype(np.float32), rotation) + translation


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[:, dy : dy + out.shape[1], dx : dx + out.shape[2]]
        out = grown
    return out


def recompute_normals(
    base: dict[str, np.ndarray],
    world_points: np.ndarray,
    changed_mask: np.ndarray,
    *,
    boost: float,
    dilate: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsics = np.asarray(base["extrinsic"], dtype=np.float32)
    use_mask = dilate_mask(changed_mask, int(dilate))
    per_view: dict[str, Any] = {}
    for view_idx in range(world_points.shape[0]):
        cam = world_to_camera(world_points[view_idx], extrinsics[view_idx])
        finite = np.isfinite(cam).all(axis=-1)
        normal_map, valid = point_map_to_normal_numpy(cam, finite)
        use = use_mask[view_idx] & valid
        mean_dot = None
        flipped = False
        if use.any():
            dot = np.sum(normal[view_idx][use] * normal_map[use], axis=-1)
            mean_dot = float(np.nanmean(dot)) if dot.size else 0.0
            if mean_dot < 0.0:
                normal_map = -normal_map
                flipped = True
            normal[view_idx][use] = normal_map[use]
            if float(boost) > 0.0:
                normal_conf[view_idx][use] = np.maximum(normal_conf[view_idx][use], float(boost))
        per_view[str(view_idx)] = {
            "seed_pixels": int(changed_mask[view_idx].sum()),
            "candidate_pixels": int(use_mask[view_idx].sum()),
            "normal_replaced_pixels": int(use.sum()),
            "mean_dot_before_optional_flip": mean_dot,
            "flipped_to_match_previous_convention": bool(flipped),
        }
    return normal, normal_conf, {
        "enabled": True,
        "normal_replaced_pixels_total": int(sum(row["normal_replaced_pixels"] for row in per_view.values())),
        "per_view": per_view,
    }


def quantize_canonical(canonical: np.ndarray, bin_size: float) -> np.ndarray:
    safe_bin = max(float(bin_size), 1e-6)
    return np.floor(canonical.astype(np.float32) / safe_bin).astype(np.int32)


def build_consensus(
    *,
    canonical_q: np.ndarray,
    world_points: np.ndarray,
    build_mask: np.ndarray,
    views: np.ndarray,
    min_bin_count: int,
    min_bin_views: int,
    max_bin_spread: float,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[str, Any]]:
    flat_q = canonical_q.reshape(-1, 3)
    flat_points = world_points.reshape(-1, 3)
    flat_mask = build_mask.reshape(-1)
    flat_views = views.reshape(-1)
    selected = np.flatnonzero(flat_mask)
    consensus: dict[tuple[int, int, int], np.ndarray] = {}
    rejected = {"too_few_count": 0, "too_few_views": 0, "spread": 0}
    if selected.size == 0:
        return consensus, {"selected_pixels": 0, "accepted_bins": 0, "rejected": rejected}

    keys = flat_q[selected]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_selected = selected[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], sorted_selected.size]
    spreads: list[float] = []
    counts: list[int] = []
    view_counts: list[int] = []

    for start, end in zip(starts, ends):
        group_indices = sorted_selected[start:end]
        bin_id = int(sorted_inverse[start])
        if group_indices.size < int(min_bin_count):
            rejected["too_few_count"] += 1
            continue
        group_views = np.unique(flat_views[group_indices])
        if group_views.size < int(min_bin_views):
            rejected["too_few_views"] += 1
            continue
        pts = flat_points[group_indices].astype(np.float32)
        center = np.median(pts, axis=0)
        distances = np.linalg.norm(pts - center[None, :], axis=1)
        spread = float(np.percentile(distances, 75)) if distances.size else float("inf")
        if not np.isfinite(spread) or spread > float(max_bin_spread):
            rejected["spread"] += 1
            continue
        consensus[tuple(int(v) for v in unique[bin_id])] = center.astype(np.float32)
        spreads.append(spread)
        counts.append(int(group_indices.size))
        view_counts.append(int(group_views.size))

    summary = {
        "selected_pixels": int(selected.size),
        "unique_bins": int(unique.shape[0]),
        "accepted_bins": int(len(consensus)),
        "rejected": rejected,
        "accepted_count_percentiles": [float(v) for v in np.percentile(counts, [0, 25, 50, 75, 95, 100])] if counts else [],
        "accepted_view_count_percentiles": [float(v) for v in np.percentile(view_counts, [0, 25, 50, 75, 95, 100])] if view_counts else [],
        "accepted_spread_percentiles": [float(v) for v in np.percentile(spreads, [0, 25, 50, 75, 95, 100])] if spreads else [],
    }
    return consensus, summary


def apply_consensus(
    *,
    canonical_q: np.ndarray,
    world_points: np.ndarray,
    depth: np.ndarray,
    world_conf: np.ndarray,
    depth_conf: np.ndarray,
    apply_mask: np.ndarray,
    extrinsics: np.ndarray,
    consensus: dict[tuple[int, int, int], np.ndarray],
    max_apply_distance: float,
    alpha: float,
    confidence_boost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fused_world = world_points.copy()
    fused_depth = depth.copy()
    fused_world_conf = world_conf.copy()
    fused_depth_conf = depth_conf.copy()
    changed = np.zeros(world_points.shape[:3], dtype=bool)
    per_view: dict[str, Any] = {}

    flat_q = canonical_q.reshape(-1, 3)
    flat_points = world_points.reshape(-1, 3)
    flat_apply = apply_mask.reshape(-1)
    flat_changed = changed.reshape(-1)
    flat_fused = fused_world.reshape(-1, 3)
    selected = np.flatnonzero(flat_apply)
    distances_applied: list[float] = []

    for flat_idx in selected:
        key = tuple(int(v) for v in flat_q[flat_idx])
        target = consensus.get(key)
        if target is None:
            continue
        current = flat_points[flat_idx]
        distance = float(np.linalg.norm(current - target))
        if not np.isfinite(distance) or distance > float(max_apply_distance):
            continue
        flat_fused[flat_idx] = (current + float(alpha) * (target - current)).astype(np.float32)
        flat_changed[flat_idx] = True
        distances_applied.append(distance)

    for view_idx in range(world_points.shape[0]):
        use = changed[view_idx]
        if use.any():
            cam = world_to_camera(fused_world[view_idx], extrinsics[view_idx])
            positive = use & np.isfinite(cam[..., 2]) & (cam[..., 2] > 1e-6)
            fused_depth[view_idx, positive, 0] = cam[..., 2][positive]
            if float(confidence_boost) > 0.0:
                fused_world_conf[view_idx, use] = np.maximum(fused_world_conf[view_idx, use], float(confidence_boost))
                fused_depth_conf[view_idx, positive] = np.maximum(fused_depth_conf[view_idx, positive], float(confidence_boost))
        per_view[str(view_idx)] = {
            "apply_candidates": int(apply_mask[view_idx].sum()),
            "changed_pixels": int(use.sum()),
        }

    summary = {
        "changed_pixels_total": int(changed.sum()),
        "changed_pixels_per_view": [int(changed[idx].sum()) for idx in range(changed.shape[0])],
        "apply_distance_percentiles": [float(v) for v in np.percentile(distances_applied, [0, 25, 50, 75, 95, 100])]
        if distances_applied
        else [],
        "per_view": per_view,
    }
    return fused_world, fused_depth, fused_world_conf, fused_depth_conf, changed, summary


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_npz(args.predictions)
    prior_path = args.scene_dir / "prior_maps.npz"
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    priors = load_npz(prior_path)
    channel_names = [str(value) for value in priors["prior_channels"].tolist()]
    canonical_indices = [
        channel_index(channel_names, "smplx_canonical_x"),
        channel_index(channel_names, "smplx_canonical_y"),
        channel_index(channel_names, "smplx_canonical_z"),
    ]
    visible_index = channel_names.index("smplx_visible_mask") if "smplx_visible_mask" in channel_names else None

    world_points = np.asarray(base["world_points"], dtype=np.float32)
    depth = np.asarray(base["depth"], dtype=np.float32)
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32)
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32)
    extrinsics = np.asarray(base["extrinsic"], dtype=np.float32)
    views, height, width = world_conf.shape
    canonical = np.asarray(priors["prior_maps"][:, canonical_indices], dtype=np.float32).transpose(0, 2, 3, 1)
    if canonical.shape[:3] != (views, height, width):
        raise ValueError(f"Prior canonical shape {canonical.shape} does not match predictions {(views, height, width)}")
    prior_mask = np.asarray(priors["prior_mask"], dtype=bool)
    if prior_mask.shape != (views, height, width):
        raise ValueError(f"prior_mask shape {prior_mask.shape} does not match predictions {(views, height, width)}")
    visible = np.ones_like(prior_mask, dtype=bool)
    if bool(args.visible_only) and visible_index is not None:
        visible = np.asarray(priors["prior_maps"][:, visible_index], dtype=np.float32) > 0.5

    masks = load_mask_stack(args.scene_dir / "masks", target_size=height).astype(bool)
    roi = roi_stack(args.scene_dir, target_size=height, roi=str(args.roi)).astype(bool)
    finite = np.isfinite(world_points).all(axis=-1) & np.isfinite(world_conf) & np.isfinite(depth[..., 0])
    support = masks & roi & prior_mask & visible & finite
    conf_thresholds = np.zeros((views,), dtype=np.float32)
    high_conf = np.zeros_like(support, dtype=bool)
    for view_idx in range(views):
        valid = support[view_idx]
        if valid.any():
            conf_thresholds[view_idx] = float(np.percentile(world_conf[view_idx][valid], float(args.conf_percentile)))
            high_conf[view_idx] = valid & (world_conf[view_idx] >= conf_thresholds[view_idx])
        else:
            conf_thresholds[view_idx] = np.inf

    canonical_q = quantize_canonical(canonical, float(args.canonical_bin_size))
    view_ids = np.broadcast_to(np.arange(views, dtype=np.int32)[:, None, None], (views, height, width))
    consensus, consensus_summary = build_consensus(
        canonical_q=canonical_q,
        world_points=world_points,
        build_mask=high_conf,
        views=view_ids,
        min_bin_count=int(args.min_bin_count),
        min_bin_views=int(args.min_bin_views),
        max_bin_spread=float(args.max_bin_spread),
    )
    fused_world, fused_depth, fused_world_conf, fused_depth_conf, changed, apply_summary = apply_consensus(
        canonical_q=canonical_q,
        world_points=world_points,
        depth=depth,
        world_conf=world_conf,
        depth_conf=depth_conf,
        apply_mask=support,
        extrinsics=extrinsics,
        consensus=consensus,
        max_apply_distance=float(args.max_apply_distance),
        alpha=float(args.alpha),
        confidence_boost=float(args.confidence_boost),
    )

    out = dict(base)
    out["world_points"] = fused_world.astype(base["world_points"].dtype, copy=False)
    out["depth"] = fused_depth.astype(base["depth"].dtype, copy=False)
    out["world_points_conf"] = fused_world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth_conf"] = fused_depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["smplx_canonical_consensus_mask"] = changed.astype(np.uint8)
    normal_summary = {"enabled": False}
    if "normal" in base and "normal_conf" in base:
        normals, normal_conf, normal_summary = recompute_normals(
            base,
            fused_world,
            changed,
            boost=float(args.normal_confidence_boost),
            dilate=int(args.normal_dilate),
        )
        out["normal"] = normals.astype(base["normal"].dtype, copy=False)
        out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)

    output_path = args.output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)
    support_counts = {
        "mask": int(masks.sum()),
        "roi": int(roi.sum()),
        "prior_mask": int(prior_mask.sum()),
        "visible": int(visible.sum()),
        "support": int(support.sum()),
        "high_conf_support": int(high_conf.sum()),
    }
    summary = {
        "task": "fuse_smplx_canonical_consensus_predictions",
        "truthful_status": "local_vggt_self_consensus_not_teacher_not_pass",
        "predictions": str(args.predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_predictions": str(output_path.resolve()),
        "roi": str(args.roi),
        "canonical_bin_size": float(args.canonical_bin_size),
        "conf_percentile": float(args.conf_percentile),
        "visible_only": bool(args.visible_only),
        "min_bin_count": int(args.min_bin_count),
        "min_bin_views": int(args.min_bin_views),
        "max_bin_spread": float(args.max_bin_spread),
        "max_apply_distance": float(args.max_apply_distance),
        "alpha": float(args.alpha),
        "confidence_boost": float(args.confidence_boost),
        "support_counts": support_counts,
        "conf_thresholds": [float(v) if np.isfinite(v) else None for v in conf_thresholds.tolist()],
        "consensus": consensus_summary,
        "apply": apply_summary,
        "normal_recompute": normal_summary,
        "notes": [
            "SMPL-X canonical channels are used only as cross-view correspondence bins.",
            "The fused 3D target is the robust median of VGGT predicted world points, not SMPL-X geometry.",
            "This is a local diagnostic candidate and must pass the full strict Open3D gate before any claim.",
        ],
    }
    (args.output_dir / "smplx_canonical_consensus_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
