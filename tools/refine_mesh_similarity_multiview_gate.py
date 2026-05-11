from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_headface_teacher_surface import (  # noqa: E402
    ROI_KINDS,
    _roi_mask,
    connected_component_stats,
    load_scene_mask,
    min_hit_pixels_for_roi,
    parse_indices,
    parse_rois,
    percentiles,
)
from build_mesh_raycast_training_case import _rays_for_pixels, _world_to_cam  # noqa: E402
from refine_mesh_similarity_for_visible_surface import (  # noqa: E402
    _apply_similarity_to_mesh,
    _camera_euler_to_world_rotation,
    _parse_range,
    _select_pivot,
)


@dataclass(frozen=True)
class PixelSet:
    rays: np.ndarray
    depth_ys: np.ndarray
    depth_xs: np.ndarray
    mask_ys: np.ndarray
    mask_xs: np.ndarray
    mask_height: int
    mask_width: int
    roi_pixels: int
    stride: int


@dataclass(frozen=True)
class RoiViewCase:
    view_index: int
    roi_kind: str
    anchor_depth: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    full_pixels: PixelSet
    search_pixels: PixelSet
    thresholds: dict[str, float]


@dataclass(frozen=True)
class CandidateTransform:
    index: int
    dx: float
    dy: float
    dz: float
    scale: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    delta_world: np.ndarray
    rotation_world: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search bounded global similarity transforms for a mesh against the "
            "strict head/face teacher gate across multiple views and ROI kinds. "
            "This is a numeric gate helper only; run the strict audit separately "
            "for final visual-review sign-off."
        )
    )
    parser.add_argument("--mesh-path", required=True, help="Input aligned triangle mesh.")
    parser.add_argument("--predictions-npz", required=True, help="Reference predictions containing intrinsic/extrinsic/depth.")
    parser.add_argument("--scene-dir", required=True, help="Sparse-view scene directory with masks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-views", default="all", help="Comma separated view indices or 'all'.")
    parser.add_argument(
        "--roi-kinds",
        default="face_core,head_face,hairline",
        help=f"Comma separated ROI kinds. Choices: {','.join(ROI_KINDS)}",
    )
    parser.add_argument(
        "--transform-view-index",
        type=int,
        default=-1,
        help=(
            "Camera whose axes define dx/dy/dz and yaw/pitch/roll. "
            "Default -1 uses the first selected target view."
        ),
    )
    parser.add_argument("--depth-tolerance", type=float, default=0.06)
    parser.add_argument("--max-hole-ratio", type=float, default=0.35)
    parser.add_argument("--min-coverage", type=float, default=0.58)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.78)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--min-fragment-pixels", type=int, default=32)
    parser.add_argument("--max-median-depth-residual", type=float, default=0.025)
    parser.add_argument("--max-p90-depth-residual", type=float, default=0.055)
    parser.add_argument("--min-hit-pixels-face-core", type=int, default=4500)
    parser.add_argument("--min-hit-pixels-head-face", type=int, default=11000)
    parser.add_argument("--min-hit-pixels-head", type=int, default=9000)
    parser.add_argument("--min-hit-pixels-face", type=int, default=5000)
    parser.add_argument("--min-hit-pixels-hairline", type=int, default=3500)
    parser.add_argument("--min-hit-pixels-default", type=int, default=4500)
    parser.add_argument("--dx", default="-0.02,0.02,0.01", help="Camera-x translation min,max,step in meters.")
    parser.add_argument("--dy", default="-0.02,0.02,0.01", help="Camera-y translation min,max,step in meters.")
    parser.add_argument("--dz", default="-0.02,0.02,0.01", help="Camera-z translation min,max,step in meters.")
    parser.add_argument("--scale", default="0.99,1.01,0.01", help="Uniform scale min,max,step.")
    parser.add_argument("--yaw-deg", default="-2,2,1", help="Camera-y yaw min,max,step in degrees.")
    parser.add_argument("--pitch-deg", default="0,0,1", help="Camera-x pitch min,max,step in degrees.")
    parser.add_argument("--roll-deg", default="0,0,1", help="Camera-z roll min,max,step in degrees.")
    parser.add_argument(
        "--pivot",
        choices=("centroid", "bbox", "origin"),
        default="centroid",
        help="World-space pivot for scale and rotation.",
    )
    parser.add_argument(
        "--search-stride",
        type=int,
        default=4,
        help=(
            "Rank candidates on an every-Nth-pixel ROI grid. Full metrics are "
            "computed for the top candidates only."
        ),
    )
    parser.add_argument("--refine-top-k", type=int, default=30, help="Search-ranked candidates to re-evaluate fully.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top candidates to write in JSON/Markdown.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10000,
        help="Abort if the grid is larger than this bounded-search limit. Set higher deliberately if needed.",
    )
    parser.add_argument("--progress-every", type=int, default=250, help="Print search progress every N candidates.")
    return parser.parse_args()


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        intrinsic_key = "intrinsic" if "intrinsic" in payload.files else "intrinsics"
        extrinsic_key = "extrinsic" if "extrinsic" in payload.files else "extrinsics"
        intrinsics = np.asarray(payload[intrinsic_key], dtype=np.float32)
        extrinsics = np.asarray(payload[extrinsic_key], dtype=np.float32)
        depth = np.asarray(payload["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape [views,height,width] or [views,height,width,1], got {depth.shape}")
    if intrinsics.shape[0] != depth.shape[0] or extrinsics.shape[0] != depth.shape[0]:
        raise ValueError(
            f"Camera/depth view mismatch: intrinsic={intrinsics.shape}, extrinsic={extrinsics.shape}, depth={depth.shape}"
        )
    return intrinsics, extrinsics, depth


def _make_pixel_set(
    roi: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    stride: int,
) -> PixelSet:
    stride = int(max(1, stride))
    roi = np.asarray(roi, dtype=bool)
    if stride == 1:
        mask_ys, mask_xs = np.nonzero(roi)
        depth_ys = mask_ys.astype(np.int64, copy=False)
        depth_xs = mask_xs.astype(np.int64, copy=False)
        mask_height, mask_width = roi.shape
    else:
        small_roi = roi[::stride, ::stride]
        mask_ys, mask_xs = np.nonzero(small_roi)
        depth_ys = (mask_ys * stride).astype(np.int64, copy=False)
        depth_xs = (mask_xs * stride).astype(np.int64, copy=False)
        mask_height, mask_width = small_roi.shape
    rays = _rays_for_pixels(depth_xs, depth_ys, intrinsic, extrinsic) if depth_xs.size else np.zeros((0, 6), dtype=np.float32)
    return PixelSet(
        rays=rays,
        depth_ys=depth_ys,
        depth_xs=depth_xs,
        mask_ys=mask_ys.astype(np.int64, copy=False),
        mask_xs=mask_xs.astype(np.int64, copy=False),
        mask_height=int(mask_height),
        mask_width=int(mask_width),
        roi_pixels=int(mask_xs.size),
        stride=stride,
    )


def _thresholds_for_roi(args: argparse.Namespace, roi_kind: str) -> dict[str, float]:
    return {
        "min_hit_pixels": float(min_hit_pixels_for_roi(args, roi_kind)),
        "min_coverage": float(args.min_coverage),
        "max_hole_ratio": float(args.max_hole_ratio),
        "min_largest_component_ratio": float(args.min_largest_component_ratio),
        "max_components": float(args.max_components),
        "max_median_depth_residual": float(args.max_median_depth_residual),
        "max_p90_depth_residual": float(args.max_p90_depth_residual),
    }


def _thresholds_for_pixels(case: RoiViewCase, pixels: PixelSet) -> dict[str, float]:
    thresholds = dict(case.thresholds)
    full_pixels = max(case.full_pixels.roi_pixels, 1)
    thresholds["min_hit_pixels"] = float(thresholds["min_hit_pixels"] * pixels.roi_pixels / full_pixels)
    return thresholds


def _prepare_cases(
    *,
    args: argparse.Namespace,
    scene_dir: Path,
    views: list[int],
    rois: list[str],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth: np.ndarray,
) -> list[RoiViewCase]:
    cases: list[RoiViewCase] = []
    _, height, _ = depth.shape
    for view_index in views:
        target_mask = load_scene_mask(scene_dir, view_index=view_index, target_size=int(height))
        for roi_kind in rois:
            roi = _roi_mask(target_mask, roi_kind)
            full_pixels = _make_pixel_set(
                roi,
                intrinsics[view_index],
                extrinsics[view_index],
                stride=1,
            )
            search_pixels = _make_pixel_set(
                roi,
                intrinsics[view_index],
                extrinsics[view_index],
                stride=int(args.search_stride),
            )
            cases.append(
                RoiViewCase(
                    view_index=int(view_index),
                    roi_kind=str(roi_kind),
                    anchor_depth=depth[view_index],
                    intrinsic=intrinsics[view_index],
                    extrinsic=extrinsics[view_index],
                    full_pixels=full_pixels,
                    search_pixels=search_pixels,
                    thresholds=_thresholds_for_roi(args, roi_kind),
                )
            )
    return cases


def _margin_min(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0 if value >= threshold else -1.0
    return float(value / threshold - 1.0)


def _margin_max(value: float | None, threshold: float) -> float:
    if value is None:
        return -1.0
    denom = max(abs(float(threshold)), 1e-6)
    return float((float(threshold) - float(value)) / denom)


def _gate_and_margins(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[dict[str, bool], dict[str, float]]:
    depth_metrics = metrics["depth_compatible"]
    components = depth_metrics["components"]
    residual = depth_metrics["depth_residual"]
    hit_pixels = float(depth_metrics["hit_pixels"])
    coverage = float(depth_metrics["coverage"])
    hole_ratio = float(depth_metrics["hole_ratio"])
    largest_component_ratio = float(components["largest_component_ratio"])
    meaningful_components = float(components["meaningful_components"])
    median_residual = residual["p50"]
    p90_residual = residual["p90"]
    margins = {
        "hit_pixels": _margin_min(hit_pixels, float(thresholds["min_hit_pixels"])),
        "coverage": _margin_min(coverage, float(thresholds["min_coverage"])),
        "hole_ratio": _margin_max(hole_ratio, float(thresholds["max_hole_ratio"])),
        "largest_component_ratio": _margin_min(largest_component_ratio, float(thresholds["min_largest_component_ratio"])),
        "fragment_count": _margin_max(meaningful_components, float(thresholds["max_components"])),
        "median_depth_residual": _margin_max(median_residual, float(thresholds["max_median_depth_residual"])),
        "p90_depth_residual": _margin_max(p90_residual, float(thresholds["max_p90_depth_residual"])),
    }
    gate = {
        "hit_pixels": bool(hit_pixels >= float(thresholds["min_hit_pixels"])),
        "coverage": bool(coverage >= float(thresholds["min_coverage"])),
        "hole_ratio": bool(hole_ratio <= float(thresholds["max_hole_ratio"])),
        "largest_component_ratio": bool(largest_component_ratio >= float(thresholds["min_largest_component_ratio"])),
        "fragment_count": bool(meaningful_components <= float(thresholds["max_components"])),
        "median_depth_residual": bool(median_residual is not None and float(median_residual) <= float(thresholds["max_median_depth_residual"])),
        "p90_depth_residual": bool(p90_residual is not None and float(p90_residual) <= float(thresholds["max_p90_depth_residual"])),
    }
    gate["pass"] = bool(all(gate.values()))
    return gate, margins


def _empty_metrics(case: RoiViewCase, pixels: PixelSet) -> dict[str, Any]:
    empty_components = connected_component_stats(
        np.zeros((pixels.mask_height, pixels.mask_width), dtype=bool),
        min_component_pixels=1,
    )
    return {
        "view_index": int(case.view_index),
        "roi_kind": case.roi_kind,
        "roi_pixels": int(pixels.roi_pixels),
        "sample_stride": int(pixels.stride),
        "raw_visible": {
            "hit_pixels": 0,
            "coverage": 0.0,
            "hole_ratio": 1.0,
            "components": empty_components,
            "depth_residual": percentiles(np.zeros((0,), dtype=np.float32)),
        },
        "depth_compatible": {
            "hit_pixels": 0,
            "coverage": 0.0,
            "hole_ratio": 1.0,
            "components": empty_components,
            "depth_residual": percentiles(np.zeros((0,), dtype=np.float32)),
        },
    }


def _metrics_for_similarity(
    *,
    scene: Any,
    o3d: Any,
    case: RoiViewCase,
    pixels: PixelSet,
    delta_world: np.ndarray,
    rotation_world: np.ndarray,
    scale: float,
    pivot_world: np.ndarray,
    depth_tolerance: float,
    min_fragment_pixels: int,
) -> dict[str, Any]:
    if pixels.roi_pixels == 0:
        return _empty_metrics(case, pixels)
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")

    rays = pixels.rays.copy()
    origins = np.asarray(pixels.rays[:, :3], dtype=np.float32)
    directions = np.asarray(pixels.rays[:, 3:], dtype=np.float32)
    rotation_world = np.asarray(rotation_world, dtype=np.float32)
    delta_world = np.asarray(delta_world, dtype=np.float32)
    pivot_world = np.asarray(pivot_world, dtype=np.float32)

    inv_origins = pivot_world[None] + ((origins - pivot_world[None] - delta_world[None]) @ rotation_world) / float(scale)
    inv_directions = directions @ rotation_world
    inv_directions /= np.clip(np.linalg.norm(inv_directions, axis=-1, keepdims=True), 1e-6, None)
    rays[:, :3] = inv_origins.astype(np.float32)
    rays[:, 3:] = inv_directions.astype(np.float32)

    answers = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = answers["t_hit"].numpy()
    raw_valid = np.isfinite(t_hit)

    original_hit = inv_origins + inv_directions * np.where(raw_valid, t_hit, 0.0)[:, None]
    world_hit = pivot_world[None] + (float(scale) * (original_hit - pivot_world[None])) @ rotation_world.T + delta_world[None]
    cam_hit = _world_to_cam(world_hit.astype(np.float32), case.extrinsic)
    hit_depth = cam_hit[:, 2]
    anchor = case.anchor_depth[pixels.depth_ys, pixels.depth_xs]
    residual = np.abs(hit_depth - anchor)
    residual_finite = np.isfinite(residual)
    depth_ok = raw_valid & (hit_depth > 0.05) & residual_finite & (residual <= float(depth_tolerance))

    raw_mask = np.zeros((pixels.mask_height, pixels.mask_width), dtype=bool)
    depth_ok_mask = np.zeros_like(raw_mask, dtype=bool)
    raw_mask[pixels.mask_ys[raw_valid], pixels.mask_xs[raw_valid]] = True
    depth_ok_mask[pixels.mask_ys[depth_ok], pixels.mask_xs[depth_ok]] = True

    raw_components = connected_component_stats(raw_mask, min_component_pixels=int(min_fragment_pixels))
    ok_components = connected_component_stats(depth_ok_mask, min_component_pixels=int(min_fragment_pixels))
    raw_hit_pixels = int(raw_mask.sum())
    ok_hit_pixels = int(depth_ok_mask.sum())
    roi_pixels = int(max(pixels.roi_pixels, 1))
    return {
        "view_index": int(case.view_index),
        "roi_kind": case.roi_kind,
        "roi_pixels": int(pixels.roi_pixels),
        "sample_stride": int(pixels.stride),
        "raw_visible": {
            "hit_pixels": raw_hit_pixels,
            "coverage": float(raw_hit_pixels / roi_pixels),
            "hole_ratio": float(1.0 - raw_hit_pixels / roi_pixels),
            "components": raw_components,
            "depth_residual": percentiles(residual[raw_valid & residual_finite]),
        },
        "depth_compatible": {
            "hit_pixels": ok_hit_pixels,
            "coverage": float(ok_hit_pixels / roi_pixels),
            "hole_ratio": float(1.0 - ok_hit_pixels / roi_pixels),
            "components": ok_components,
            "depth_residual": percentiles(residual[depth_ok]),
        },
    }


def _entry_summary(
    *,
    case: RoiViewCase,
    pixels: PixelSet,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    gate, margins = _gate_and_margins(metrics, thresholds)
    failed_checks = [name for name, ok in gate.items() if name != "pass" and not ok]
    return {
        **metrics,
        "full_roi_pixels": int(case.full_pixels.roi_pixels),
        "gate_thresholds": {
            "min_hit_pixels": float(thresholds["min_hit_pixels"]),
            "min_hit_pixels_full": float(case.thresholds["min_hit_pixels"]),
            "min_coverage": float(thresholds["min_coverage"]),
            "max_hole_ratio": float(thresholds["max_hole_ratio"]),
            "min_largest_component_ratio": float(thresholds["min_largest_component_ratio"]),
            "max_components": int(thresholds["max_components"]),
            "min_fragment_pixels": int(metrics["depth_compatible"]["components"]["min_component_pixels"]),
            "max_median_depth_residual": float(thresholds["max_median_depth_residual"]),
            "max_p90_depth_residual": float(thresholds["max_p90_depth_residual"]),
        },
        "gate": gate,
        "margins": margins,
        "score": {
            "worst_margin": float(min(margins.values())),
            "mean_margin": float(np.mean(list(margins.values()))),
            "failed_checks": failed_checks,
            "sampled": bool(pixels.stride != 1),
        },
    }


def _finite_or_none(values: list[float | None], *, reducer: str) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(min(finite) if reducer == "min" else max(finite))


def _candidate_parameters(candidate: CandidateTransform, *, include_matrix: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidate_index": int(candidate.index),
        "delta_cam": [float(candidate.dx), float(candidate.dy), float(candidate.dz)],
        "delta_world": [float(v) for v in candidate.delta_world],
        "scale": float(candidate.scale),
        "yaw_deg": float(candidate.yaw_deg),
        "pitch_deg": float(candidate.pitch_deg),
        "roll_deg": float(candidate.roll_deg),
    }
    if include_matrix:
        out["rotation_world"] = [[float(v) for v in row] for row in candidate.rotation_world]
    return out


def _candidate_summary(
    *,
    candidate: CandidateTransform,
    phase: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    entry_worst = [float(entry["score"]["worst_margin"]) for entry in entries]
    all_margins = [float(value) for entry in entries for value in entry["margins"].values()]
    failed_entries = [entry for entry in entries if not entry["gate"]["pass"]]
    coverages = [float(entry["depth_compatible"]["coverage"]) for entry in entries]
    holes = [float(entry["depth_compatible"]["hole_ratio"]) for entry in entries]
    lccs = [float(entry["depth_compatible"]["components"]["largest_component_ratio"]) for entry in entries]
    medians = [entry["depth_compatible"]["depth_residual"]["p50"] for entry in entries]
    p90s = [entry["depth_compatible"]["depth_residual"]["p90"] for entry in entries]
    return {
        "phase": phase,
        "parameters": _candidate_parameters(candidate, include_matrix=False),
        "score": {
            "numeric_pass": bool(entries and not failed_entries),
            "failed_entries": int(len(failed_entries)),
            "passed_entries": int(len(entries) - len(failed_entries)),
            "worst_margin": float(min(entry_worst)) if entry_worst else -1.0,
            "mean_entry_worst_margin": float(np.mean(entry_worst)) if entry_worst else -1.0,
            "mean_all_margins": float(np.mean(all_margins)) if all_margins else -1.0,
            "min_coverage": float(min(coverages)) if coverages else 0.0,
            "max_hole_ratio": float(max(holes)) if holes else 1.0,
            "min_largest_component_ratio": float(min(lccs)) if lccs else 0.0,
            "max_median_depth_residual": _finite_or_none(medians, reducer="max"),
            "max_p90_depth_residual": _finite_or_none(p90s, reducer="max"),
        },
        "entries": entries,
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, int, float, float, float, float, float]:
    score = row["score"]
    params = row["parameters"]
    max_p90 = score["max_p90_depth_residual"]
    p90_sort = 1.0e9 if max_p90 is None else float(max_p90)
    delta_norm = float(np.linalg.norm(np.asarray(params["delta_cam"], dtype=np.float64)))
    return (
        float(score["worst_margin"]),
        -int(score["failed_entries"]),
        float(score["mean_all_margins"]),
        float(score["min_coverage"]),
        -p90_sort,
        -abs(float(params["scale"]) - 1.0),
        -delta_norm,
    )


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _evaluate_candidate(
    *,
    scene: Any,
    o3d: Any,
    candidate: CandidateTransform,
    cases: list[RoiViewCase],
    args: argparse.Namespace,
    phase: str,
) -> dict[str, Any]:
    use_search = phase == "search"
    entries: list[dict[str, Any]] = []
    for case in cases:
        pixels = case.search_pixels if use_search else case.full_pixels
        thresholds = _thresholds_for_pixels(case, pixels)
        metrics = _metrics_for_similarity(
            scene=scene,
            o3d=o3d,
            case=case,
            pixels=pixels,
            delta_world=candidate.delta_world,
            rotation_world=candidate.rotation_world,
            scale=float(candidate.scale),
            pivot_world=args._pivot_world,
            depth_tolerance=float(args.depth_tolerance),
            min_fragment_pixels=int(args.min_fragment_pixels),
        )
        entries.append(_entry_summary(case=case, pixels=pixels, metrics=metrics, thresholds=thresholds))
    row = _candidate_summary(candidate=candidate, phase=phase, entries=entries)
    row["_candidate"] = candidate
    return row


def _iter_candidates(
    *,
    args: argparse.Namespace,
    extrinsics: np.ndarray,
    transform_view_index: int,
) -> tuple[int, list[CandidateTransform]]:
    dx_values = _parse_range(args.dx)
    dy_values = _parse_range(args.dy)
    dz_values = _parse_range(args.dz)
    scale_values = _parse_range(args.scale)
    yaw_values = _parse_range(args.yaw_deg)
    pitch_values = _parse_range(args.pitch_deg)
    roll_values = _parse_range(args.roll_deg)
    total = (
        len(dx_values)
        * len(dy_values)
        * len(dz_values)
        * len(scale_values)
        * len(yaw_values)
        * len(pitch_values)
        * len(roll_values)
    )
    if total <= 0:
        raise RuntimeError("Empty search grid.")
    if total > int(args.max_candidates):
        raise RuntimeError(
            f"Search grid has {total} candidates, above --max-candidates={args.max_candidates}. "
            "Narrow the grid or raise the bound deliberately."
        )

    reference_extrinsic = extrinsics[transform_view_index]
    camera_rotation = np.asarray(reference_extrinsic[:3, :3], dtype=np.float32)
    candidates: list[CandidateTransform] = []
    index = 0
    for yaw_deg in yaw_values:
        for pitch_deg in pitch_values:
            for roll_deg in roll_values:
                rotation_world = _camera_euler_to_world_rotation(
                    float(yaw_deg),
                    float(pitch_deg),
                    float(roll_deg),
                    reference_extrinsic,
                )
                for scale in scale_values:
                    for dx in dx_values:
                        for dy in dy_values:
                            for dz in dz_values:
                                delta_cam = np.array([dx, dy, dz], dtype=np.float32)
                                delta_world = camera_rotation.T @ delta_cam
                                candidates.append(
                                    CandidateTransform(
                                        index=index,
                                        dx=float(dx),
                                        dy=float(dy),
                                        dz=float(dz),
                                        scale=float(scale),
                                        yaw_deg=float(yaw_deg),
                                        pitch_deg=float(pitch_deg),
                                        roll_deg=float(roll_deg),
                                        delta_world=delta_world.astype(np.float64),
                                        rotation_world=np.asarray(rotation_world, dtype=np.float64),
                                    )
                                )
                                index += 1
    return total, candidates


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.{digits}f}"

    lines = [
        "# Multiview Mesh Similarity Refinement",
        "",
        f"- Input mesh: `{summary['mesh_path']}`",
        f"- Best mesh: `{summary['refined_mesh_path']}`",
        f"- Views: `{summary['target_views']}`",
        f"- ROIs: `{summary['roi_kinds']}`",
        f"- Numeric pass: `{summary['gate']['numeric_pass']}`",
        f"- Worst margin: `{fmt(summary['best']['score']['worst_margin'], 5)}`",
        "",
        "## Top Full-Resolution Candidates",
        "",
        "| Rank | Pass | Worst margin | Failed entries | Min cov | Max hole | Min LCC | Max med | Max p90 | dx dy dz | scale | yaw pitch roll |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for rank, row in enumerate(summary["top_full_candidates"], start=1):
        score = row["score"]
        params = row["parameters"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(score["numeric_pass"]),
                    fmt(score["worst_margin"], 5),
                    str(score["failed_entries"]),
                    fmt(score["min_coverage"]),
                    fmt(score["max_hole_ratio"]),
                    fmt(score["min_largest_component_ratio"]),
                    fmt(score["max_median_depth_residual"], 5),
                    fmt(score["max_p90_depth_residual"], 5),
                    " ".join(fmt(v, 3) for v in params["delta_cam"]),
                    fmt(params["scale"], 4),
                    " ".join(fmt(params[key], 2) for key in ("yaw_deg", "pitch_deg", "roll_deg")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Best Candidate Failures", ""])
    failures = [entry for entry in summary["best"]["entries"] if not entry["gate"]["pass"]]
    if failures:
        for entry in failures:
            failed = ", ".join(entry["score"]["failed_checks"])
            depth = entry["depth_compatible"]
            residual = depth["depth_residual"]
            lines.append(
                "- "
                f"view {entry['view_index']} / {entry['roi_kind']}: {failed}; "
                f"coverage={fmt(depth['coverage'])}, hole={fmt(depth['hole_ratio'])}, "
                f"lcc={fmt(depth['components']['largest_component_ratio'])}, "
                f"median={fmt(residual['p50'], 5)}, p90={fmt(residual['p90'], 5)}, "
                f"hits={depth['hit_pixels']}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Note",
            "",
            "This helper only searches a global similarity transform using numeric gate metrics. It does not perform the strict audit's explicit visual review.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)

    intrinsics, extrinsics, depth = _load_predictions(predictions_path)
    view_count = int(depth.shape[0])
    views = parse_indices(str(args.target_views), view_count)
    rois = parse_rois(str(args.roi_kinds))
    if not views:
        raise ValueError("No target views selected.")
    transform_view_index = int(args.transform_view_index)
    if transform_view_index < 0:
        transform_view_index = int(views[0])
    if transform_view_index < 0 or transform_view_index >= view_count:
        raise IndexError(f"transform_view_index={transform_view_index} outside [0,{view_count})")

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    pivot_world = _select_pivot(mesh, str(args.pivot))
    args._pivot_world = pivot_world

    cases = _prepare_cases(
        args=args,
        scene_dir=scene_dir,
        views=views,
        rois=rois,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depth,
    )
    total_candidates, candidates = _iter_candidates(
        args=args,
        extrinsics=extrinsics,
        transform_view_index=transform_view_index,
    )

    search_results: list[dict[str, Any]] = []
    progress_every = int(max(0, args.progress_every))
    for offset, candidate in enumerate(candidates, start=1):
        row = _evaluate_candidate(
            scene=scene,
            o3d=o3d,
            candidate=candidate,
            cases=cases,
            args=args,
            phase="search",
        )
        search_results.append(row)
        if progress_every and (offset % progress_every == 0 or offset == total_candidates):
            print(f"[search] {offset}/{total_candidates}", file=sys.stderr)

    search_results.sort(key=_candidate_sort_key, reverse=True)
    refine_count = min(max(1, int(args.refine_top_k)), len(search_results))
    full_results: list[dict[str, Any]] = []
    for row in search_results[:refine_count]:
        full_results.append(
            _evaluate_candidate(
                scene=scene,
                o3d=o3d,
                candidate=row["_candidate"],
                cases=cases,
                args=args,
                phase="full",
            )
        )
    full_results.sort(key=_candidate_sort_key, reverse=True)
    best = full_results[0]
    best_candidate: CandidateTransform = best["_candidate"]

    refined_mesh = _apply_similarity_to_mesh(
        mesh,
        delta_world=best_candidate.delta_world,
        rotation_world=best_candidate.rotation_world,
        scale=float(best_candidate.scale),
        pivot_world=pivot_world,
    )
    refined_mesh_path = output_dir / "mesh_similarity_multiview_refined.ply"
    o3d.io.write_triangle_mesh(str(refined_mesh_path), refined_mesh)

    public_best = _public_row(best)
    public_best["parameters"] = _candidate_parameters(best_candidate, include_matrix=True)
    summary = {
        "mesh_path": str(mesh_path),
        "refined_mesh_path": str(refined_mesh_path),
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "target_views": [int(v) for v in views],
        "roi_kinds": rois,
        "transform_view_index": int(transform_view_index),
        "depth_tolerance": float(args.depth_tolerance),
        "pivot": {
            "kind": str(args.pivot),
            "world": [float(v) for v in pivot_world],
        },
        "search": {
            "total_candidates": int(total_candidates),
            "search_stride": int(args.search_stride),
            "refine_top_k": int(refine_count),
            "top_k": int(args.top_k),
            "max_candidates": int(args.max_candidates),
            "grid": {
                "dx": [float(v) for v in _parse_range(args.dx)],
                "dy": [float(v) for v in _parse_range(args.dy)],
                "dz": [float(v) for v in _parse_range(args.dz)],
                "scale": [float(v) for v in _parse_range(args.scale)],
                "yaw_deg": [float(v) for v in _parse_range(args.yaw_deg)],
                "pitch_deg": [float(v) for v in _parse_range(args.pitch_deg)],
                "roll_deg": [float(v) for v in _parse_range(args.roll_deg)],
            },
        },
        "case_count": int(len(cases)),
        "cases": [
            {
                "view_index": int(case.view_index),
                "roi_kind": case.roi_kind,
                "full_roi_pixels": int(case.full_pixels.roi_pixels),
                "search_roi_pixels": int(case.search_pixels.roi_pixels),
                "thresholds": case.thresholds,
            }
            for case in cases
        ],
        "gate": {
            "numeric_pass": bool(public_best["score"]["numeric_pass"]),
            "visual_review_pass": None,
            "pass": bool(public_best["score"]["numeric_pass"]),
            "note": "Visual review is not performed by this helper.",
        },
        "best": public_best,
        "top_full_candidates": [_public_row(row) for row in full_results[: int(args.top_k)]],
        "top_search_candidates": [_public_row(row) for row in search_results[: int(args.top_k)]],
        "truthful_note": (
            "This is a bounded similarity-transform search for numeric teacher-gate metrics. "
            "A numeric pass should still be checked with tools/audit_headface_teacher_surface.py "
            "for strict visual-review sign-off."
        ),
    }
    summary_path = output_dir / "mesh_similarity_multiview_summary.json"
    markdown_path = output_dir / "mesh_similarity_multiview_summary.md"
    summary["summary_json_path"] = str(summary_path)
    summary["summary_markdown_path"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["gate"]["numeric_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
