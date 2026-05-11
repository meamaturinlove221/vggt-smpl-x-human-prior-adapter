from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import b_gs0_smplx_anchored_free_gaussian_smoke as bgs0  # noqa: E402


DEFAULT_BGS1_DIR = REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend"
DEFAULT_FUS3D_FIELDS = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D0_v2_latent_grid_sdf_backend_smoke/"
    "b_fus3d0_v2_latent_grid_sdf_fields.npz"
)
DEFAULT_LATENT_GRID = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_GS2_fus3d_guided_residual_gaussian_layer"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
    "registry_write": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "backend_smoke": True,
    "uses_b_gs1_artifacts": True,
    "spawns_new_residual_gaussians_from_fus3d": True,
    "does_not_rescore_b_gs1_free_candidates": True,
    "no_cloud": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "writes_checkpoint": False,
    "not_teacher": True,
    "not_candidate": True,
}

RESIDUAL_COLORS = {
    "fus3d_silhouette_residual": np.asarray([245, 188, 48], dtype=np.uint8),
    "fus3d_uncertain_surface_residual": np.asarray([54, 185, 232], dtype=np.uint8),
    "fus3d_confidence_residual": np.asarray([128, 214, 98], dtype=np.uint8),
    "fus3d_random_residual_control": np.asarray([178, 178, 178], dtype=np.uint8),
    "fus3d_confidence_only_control": np.asarray([204, 116, 238], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-GS2 Fus3D-guided residual Gaussian layer smoke. "
            "This consumes the existing B-GS1 constrained baseline artifact plus "
            "B-Fus3D0-v2 latent-grid fields, spawns new residual Gaussians from "
            "Fus3D uncertainty/confidence/silhouette-gap proxies, and compares "
            "against a constrained baseline and residual controls. It never "
            "writes cloud/export/pass state."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=bgs0.DEFAULT_SCENE_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="all")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--bgs1-dir", type=Path, default=DEFAULT_BGS1_DIR)
    parser.add_argument("--fus3d-fields", type=Path, default=DEFAULT_FUS3D_FIELDS)
    parser.add_argument("--latent-grid", type=Path, default=DEFAULT_LATENT_GRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--occupancy-threshold", type=float, default=0.56)
    parser.add_argument("--sdf-uncertainty-width", type=float, default=0.20)
    parser.add_argument("--normal-residual-scale", type=float, default=1.75)
    parser.add_argument("--min-mask-support", type=int, default=2)
    parser.add_argument("--min-visible-views", type=int, default=3)
    parser.add_argument("--max-outside-frac", type=float, default=0.35)
    parser.add_argument("--min-distance-to-baseline", type=float, default=0.006)
    parser.add_argument("--max-residual-gaussians", type=int, default=1150)
    parser.add_argument("--min-residual-gaussians", type=int, default=128)
    parser.add_argument("--residual-scale", type=float, default=0.006)
    parser.add_argument("--overfill-budget", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return bgs0.json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    bgs0.write_json(path, payload)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def stat_row(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0, "finite": 0}
    if not np.issubdtype(values.dtype, np.number):
        return {"count": int(values.size)}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - float(lo)) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def load_ascii_gaussian_ply(path: Path, family_name: str) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        second = handle.readline().strip()
        if first != "ply" or second != "format ascii 1.0":
            raise ValueError(f"{resolved} is not an ascii PLY written by the B-GS smoke tools")
        vertex_count: int | None = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{resolved} ended before end_header")
            line = line.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{resolved} missing element vertex header")
        rows = []
        for _ in range(vertex_count):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{resolved} ended inside vertex rows")
            rows.append(raw.split())
    prop_to_idx = {name: idx for idx, name in enumerate(properties)}
    required = ("x", "y", "z", "red", "green", "blue")
    missing = [name for name in required if name not in prop_to_idx]
    if missing:
        raise KeyError(f"{resolved} missing required PLY properties: {missing}")
    points = np.zeros((vertex_count, 3), dtype=np.float32)
    colors = np.zeros((vertex_count, 3), dtype=np.uint8)
    scale = np.full((vertex_count,), 0.004, dtype=np.float32)
    opacity = np.full((vertex_count,), 0.70, dtype=np.float32)
    for row_idx, row in enumerate(rows):
        points[row_idx] = [
            float(row[prop_to_idx["x"]]),
            float(row[prop_to_idx["y"]]),
            float(row[prop_to_idx["z"]]),
        ]
        colors[row_idx] = [
            int(float(row[prop_to_idx["red"]])),
            int(float(row[prop_to_idx["green"]])),
            int(float(row[prop_to_idx["blue"]])),
        ]
        if "scale" in prop_to_idx:
            scale[row_idx] = float(row[prop_to_idx["scale"]])
        if "opacity" in prop_to_idx:
            opacity[row_idx] = float(row[prop_to_idx["opacity"]])
    return {
        "points": points,
        "normals": np.zeros_like(points, dtype=np.float32),
        "colors": colors,
        "family": np.asarray([family_name] * vertex_count),
        "anchor_index": np.arange(vertex_count, dtype=np.int64),
        "scale": scale.astype(np.float32),
        "opacity": opacity.astype(np.float32),
    }


def render_masks_for_scoring(
    gaussians: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    *,
    target_size: int,
    point_radius: int,
) -> list[np.ndarray]:
    points = np.asarray(gaussians["points"], dtype=np.float32)
    colors = np.asarray(gaussians["colors"], dtype=np.uint8)
    masks: list[np.ndarray] = []
    for view in views:
        camera_id = bgs0.normalize_camera_id(view["camera_id"])
        camera = cameras[camera_id]
        intrinsic = bgs0.align_intrinsics_for_loaded_scene_view(
            np.asarray(camera["intrinsic"], dtype=np.float32),
            view,
            target_size=target_size,
        )
        uv, depth = bgs0.project_points(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        mask, _, _ = bgs0.draw_points(target_size, target_size, uv, depth, colors, radius=point_radius)
        masks.append(mask)
    return masks


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 384) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.full((query.shape[0],), np.inf, dtype=np.float32)
    out = np.full((query.shape[0],), np.inf, dtype=np.float32)
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        d2 = ((query[start:stop, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        out[start:stop] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    return out


def fallback_colors(occupancy: np.ndarray, confidence: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    occupancy = np.clip(np.asarray(occupancy, dtype=np.float32), 0.0, 1.0)
    confidence = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    visibility = np.clip(np.asarray(visibility, dtype=np.float32), 0.0, 1.0)
    colors = np.zeros((occupancy.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip(72 + 140 * occupancy, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(68 + 150 * confidence, 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(82 + 135 * visibility, 0, 255).astype(np.uint8)
    return colors


def classify_families(
    boundary_like: np.ndarray,
    gap_support: np.ndarray,
    uncertainty: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    boundary_like = np.asarray(boundary_like, dtype=bool)
    gap_support = np.asarray(gap_support, dtype=np.int32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    family = np.asarray(["fus3d_confidence_residual"] * uncertainty.shape[0])
    family[(uncertainty >= 0.58) & (confidence >= 0.45)] = "fus3d_uncertain_surface_residual"
    family[boundary_like | (gap_support > 0)] = "fus3d_silhouette_residual"
    return family


def score_fus3d_candidates(
    fields: dict[str, np.ndarray],
    latent: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    baseline_masks: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = ("points", "real_sdf", "real_occupancy", "real_confidence", "real_visibility", "real_normal_residual")
    missing = [key for key in required if key not in fields]
    if missing:
        raise KeyError(f"Fus3D field NPZ missing required arrays: {missing}")
    points = np.asarray(fields["points"], dtype=np.float32)
    sdf = np.asarray(fields["real_sdf"], dtype=np.float32)
    occupancy = np.asarray(fields["real_occupancy"], dtype=np.float32)
    confidence = np.asarray(fields["real_confidence"], dtype=np.float32)
    visibility = np.asarray(fields["real_visibility"], dtype=np.float32)
    normal_residual = np.asarray(fields["real_normal_residual"], dtype=np.float32)
    if normal_residual.shape != points.shape:
        raise ValueError(f"real_normal_residual shape {normal_residual.shape} != points shape {points.shape}")

    residual_points = (points + normal_residual * float(args.normal_residual_scale)).astype(np.float32)
    n = residual_points.shape[0]
    boundary_like = np.zeros((n,), dtype=bool)
    latent_mask_count = np.zeros((n,), dtype=np.float32)
    latent_visible_count = np.zeros((n,), dtype=np.float32)
    if latent:
        latent_points = np.asarray(latent.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        compatible = latent_points.shape == points.shape and np.allclose(latent_points, points, atol=1e-5)
        if compatible:
            if "boundary_like" in latent:
                boundary_like = np.asarray(latent["boundary_like"], dtype=bool)
            if "mask_count" in latent:
                latent_mask_count = np.asarray(latent["mask_count"], dtype=np.float32)
            if "visible_count" in latent:
                latent_visible_count = np.asarray(latent["visible_count"], dtype=np.float32)

    projected_visible = np.zeros((n,), dtype=np.int32)
    mask_support = np.zeros((n,), dtype=np.int32)
    outside = np.zeros((n,), dtype=np.int32)
    gap_support = np.zeros((n,), dtype=np.int32)
    rgb_sum = np.zeros((n, 3), dtype=np.float64)
    rgb_count = np.zeros((n,), dtype=np.int32)
    for view, base_mask in zip(views, baseline_masks, strict=False):
        camera_id = bgs0.normalize_camera_id(view["camera_id"])
        camera = cameras[camera_id]
        intrinsic = bgs0.align_intrinsics_for_loaded_scene_view(
            np.asarray(camera["intrinsic"], dtype=np.float32),
            view,
            target_size=args.target_size,
        )
        uv, depth = bgs0.project_points(residual_points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        xi = np.rint(uv[:, 0]).astype(np.int64)
        yi = np.rint(uv[:, 1]).astype(np.int64)
        inside = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 1e-6)
            & (xi >= 0)
            & (xi < int(args.target_size))
            & (yi >= 0)
            & (yi < int(args.target_size))
        )
        projected_visible += inside.astype(np.int32)
        if not np.any(inside):
            continue
        target_mask = np.asarray(view["mask"], dtype=bool)
        target_rgb = np.asarray(view["rgb"], dtype=np.uint8)
        hit = np.zeros((n,), dtype=bool)
        hit[inside] = target_mask[yi[inside], xi[inside]]
        gap = np.zeros((n,), dtype=bool)
        gap[inside] = target_mask[yi[inside], xi[inside]] & ~np.asarray(base_mask, dtype=bool)[yi[inside], xi[inside]]
        mask_support += hit.astype(np.int32)
        outside += (inside & ~hit).astype(np.int32)
        gap_support += gap.astype(np.int32)
        hit_idx = np.flatnonzero(hit)
        if hit_idx.size:
            rgb_sum[hit_idx] += target_rgb[yi[hit_idx], xi[hit_idx]].astype(np.float64)
            rgb_count[hit_idx] += 1

    fallback = fallback_colors(occupancy, confidence, visibility)
    colors = fallback.copy()
    has_rgb = rgb_count > 0
    if np.any(has_rgb):
        colors[has_rgb] = np.clip(rgb_sum[has_rgb] / rgb_count[has_rgb, None], 0.0, 255.0).astype(np.uint8)

    support_frac = mask_support.astype(np.float32) / np.clip(projected_visible.astype(np.float32), 1.0, None)
    outside_frac = outside.astype(np.float32) / np.clip(projected_visible.astype(np.float32), 1.0, None)
    gap_frac = gap_support.astype(np.float32) / max(1, len(views))
    latent_mask_ratio = latent_mask_count / np.clip(latent_visible_count, 1.0, None)
    sdf_uncertainty = np.exp(-np.abs(sdf) / max(float(args.sdf_uncertainty_width), 1e-6)).astype(np.float32)
    occupancy_band = np.clip(
        1.0 - np.abs(occupancy - float(args.occupancy_threshold)) / max(float(args.sdf_uncertainty_width), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    uncertainty = np.clip(0.65 * sdf_uncertainty + 0.35 * occupancy_band, 0.0, 1.0).astype(np.float32)
    nearest = nearest_distances(residual_points, np.asarray(baseline["points"], dtype=np.float32))
    duplicate_penalty = np.clip((float(args.min_distance_to_baseline) - nearest) / max(float(args.min_distance_to_baseline), 1e-6), 0.0, 1.0)
    distance_bonus = np.clip(nearest / max(float(args.min_distance_to_baseline) * 6.0, 1e-6), 0.0, 1.0)
    silhouette_proxy = np.clip(
        0.45 * support_frac
        + 0.25 * latent_mask_ratio
        + 0.20 * boundary_like.astype(np.float32)
        + 0.45 * gap_frac,
        0.0,
        1.0,
    ).astype(np.float32)
    score = (
        1.15 * uncertainty
        + 0.80 * confidence
        + 0.45 * visibility
        + 1.20 * silhouette_proxy
        + 0.55 * gap_frac
        + 0.18 * distance_bonus
        - 1.45 * outside_frac
        - 0.45 * duplicate_penalty
    ).astype(np.float32)
    eligible = (
        (projected_visible >= int(args.min_visible_views))
        & (mask_support >= int(args.min_mask_support))
        & (outside_frac <= float(args.max_outside_frac))
        & (nearest >= float(args.min_distance_to_baseline) * 0.25)
        & np.isfinite(score)
    )
    fallback_used = False
    if int(np.count_nonzero(eligible)) < int(args.min_residual_gaussians):
        fallback_used = True
        eligible = (
            (projected_visible >= max(1, int(args.min_visible_views) // 2))
            & (mask_support > 0)
            & (outside_frac <= min(0.75, float(args.max_outside_frac) * 1.75))
            & np.isfinite(score)
        )

    family = classify_families(boundary_like, gap_support, uncertainty, confidence)
    candidates = {
        "points": residual_points,
        "normals": normal_residual,
        "colors": colors,
        "family": family,
        "anchor_index": np.arange(n, dtype=np.int64),
        "scale": np.full((n,), float(args.residual_scale), dtype=np.float32),
        "opacity": np.clip(0.36 + 0.42 * confidence + 0.18 * uncertainty, 0.20, 0.88).astype(np.float32),
        "score": score,
        "uncertainty": uncertainty,
        "confidence": confidence.astype(np.float32),
        "visibility": visibility.astype(np.float32),
        "sdf": sdf.astype(np.float32),
        "occupancy": occupancy.astype(np.float32),
        "mask_support": mask_support,
        "projected_visible": projected_visible,
        "outside": outside,
        "outside_frac": outside_frac,
        "gap_support": gap_support,
        "gap_frac": gap_frac,
        "silhouette_proxy": silhouette_proxy,
        "nearest_baseline_distance": nearest,
        "eligible": eligible,
    }
    diagnostics = {
        "candidate_count": int(n),
        "eligible_count": int(np.count_nonzero(eligible)),
        "eligibility_fallback_used": bool(fallback_used),
        "score_stats_all": stat_row(score),
        "score_stats_eligible": stat_row(score[eligible]),
        "uncertainty_stats": stat_row(uncertainty),
        "confidence_stats": stat_row(confidence),
        "silhouette_proxy_stats": stat_row(silhouette_proxy),
        "mask_support_stats": stat_row(mask_support),
        "gap_support_stats": stat_row(gap_support),
        "outside_frac_stats": stat_row(outside_frac),
        "nearest_baseline_distance_stats": stat_row(nearest),
    }
    return candidates, diagnostics


def take_gaussians(
    candidates: dict[str, np.ndarray],
    idx: np.ndarray,
    *,
    family_override: str | None = None,
    color_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    out = {
        "points": np.asarray(candidates["points"], dtype=np.float32)[idx],
        "normals": np.asarray(candidates["normals"], dtype=np.float32)[idx],
        "colors": np.asarray(candidates["colors"], dtype=np.uint8)[idx],
        "family": np.asarray(candidates["family"])[idx],
        "anchor_index": np.asarray(candidates["anchor_index"], dtype=np.int64)[idx],
        "scale": np.asarray(candidates["scale"], dtype=np.float32)[idx],
        "opacity": np.asarray(candidates["opacity"], dtype=np.float32)[idx],
    }
    if family_override is not None:
        out["family"] = np.asarray([family_override] * idx.size)
    if color_override is not None and idx.size:
        color = np.asarray(color_override, dtype=np.uint8).reshape(1, 3)
        out["colors"] = np.repeat(color, idx.size, axis=0)
    return out


def select_indices(candidates: dict[str, np.ndarray], args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    eligible = np.asarray(candidates["eligible"], dtype=bool)
    score = np.asarray(candidates["score"], dtype=np.float32)
    max_count = int(args.max_residual_gaussians)
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size == 0:
        return {
            "guided": np.asarray([], dtype=np.int64),
            "random": np.asarray([], dtype=np.int64),
            "confidence_only": np.asarray([], dtype=np.int64),
        }
    order = eligible_idx[np.argsort(score[eligible_idx])[::-1]]
    guided = order[:max_count]
    count = int(guided.size)
    random_idx = rng.choice(eligible_idx, size=count, replace=False) if count and eligible_idx.size >= count else eligible_idx.copy()
    confidence_score = (
        np.asarray(candidates["confidence"], dtype=np.float32)
        + 0.35 * np.asarray(candidates["visibility"], dtype=np.float32)
        + 0.15 * normalize01(np.asarray(candidates["mask_support"], dtype=np.float32))
        - 0.50 * np.asarray(candidates["outside_frac"], dtype=np.float32)
    )
    conf_order = eligible_idx[np.argsort(confidence_score[eligible_idx])[::-1]]
    confidence_only = conf_order[:count]
    return {"guided": guided, "random": random_idx, "confidence_only": confidence_only}


def summarize_selection(name: str, candidates: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    family = np.asarray(candidates["family"]).astype(str)
    rows: dict[str, Any] = {}
    for item in sorted(set(family[idx].tolist())) if idx.size else []:
        mask = family[idx] == item
        rows[item] = int(np.count_nonzero(mask))
    return {
        "name": name,
        "count": int(idx.size),
        "family_counts": rows,
        "score_stats": stat_row(np.asarray(candidates["score"])[idx]) if idx.size else {"count": 0},
        "uncertainty_mean": float(np.asarray(candidates["uncertainty"])[idx].mean()) if idx.size else 0.0,
        "confidence_mean": float(np.asarray(candidates["confidence"])[idx].mean()) if idx.size else 0.0,
        "silhouette_proxy_mean": float(np.asarray(candidates["silhouette_proxy"])[idx].mean()) if idx.size else 0.0,
        "gap_support_mean": float(np.asarray(candidates["gap_support"])[idx].mean()) if idx.size else 0.0,
        "outside_frac_mean": float(np.asarray(candidates["outside_frac"])[idx].mean()) if idx.size else 0.0,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    lines = [
        "# B-GS2 Fus3D-Guided Residual Gaussian Layer",
        "",
        "Status: `research_only_residual_gaussian_smoke_no_export`",
        "",
        "This is a local research smoke. It uses the B-GS1 constrained baseline PLY as the base layer,",
        "spawns new residual Gaussians from B-Fus3D0-v2 latent-grid fields, and compares guided",
        "residuals against constrained and residual controls. It is not a teacher, candidate, export,",
        "or strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['strict_facts']['formal_cloud_train_infer_export']}",
        "teacher/candidate/predictions/registry = blocked",
        "```",
        "",
        "## What Is New",
        "",
        "- The residual candidate pool is the Fus3D latent grid, shifted by `real_normal_residual`.",
        "- B-GS1 free candidates are not reused or rescored.",
        "- Selection combines Fus3D SDF uncertainty, confidence, visibility, latent-grid boundary flags, and silhouette gap support against the B-GS1 constrained render.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Selection",
        "",
        "```json",
        json.dumps(summary["selection"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
        "## Outputs",
        "",
        "```text",
        *summary["key_outputs"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    bgs0.ensure_safe_path(args.output_dir)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    bgs1_dir = args.bgs1_dir.expanduser().resolve()
    baseline_ply = bgs1_dir / "b_gs1_constrained_baseline.ply"
    bgs1_summary = load_optional_json(bgs1_dir / "b_gs1_summary.json")
    baseline = load_ascii_gaussian_ply(baseline_ply, "b_gs1_constrained_baseline")
    fus3d_fields = load_npz(args.fus3d_fields)
    latent_grid = load_npz(args.latent_grid) if args.latent_grid.expanduser().resolve().is_file() else {}
    views, cameras, camera_source = bgs0.load_views(
        args.scene_dir,
        args.dataset_root,
        args.subset_name,
        args.view_indices,
        args.target_size,
    )
    baseline_masks = render_masks_for_scoring(
        baseline,
        views,
        cameras,
        target_size=args.target_size,
        point_radius=args.point_radius,
    )
    candidates, candidate_diagnostics = score_fus3d_candidates(
        fus3d_fields,
        latent_grid,
        baseline,
        views,
        cameras,
        baseline_masks,
        args,
    )
    rng = np.random.default_rng(int(args.seed))
    selections = select_indices(candidates, args, rng)
    guided_residual = take_gaussians(candidates, selections["guided"])
    random_residual = take_gaussians(
        candidates,
        selections["random"],
        family_override="fus3d_random_residual_control",
        color_override=RESIDUAL_COLORS["fus3d_random_residual_control"],
    )
    confidence_residual = take_gaussians(
        candidates,
        selections["confidence_only"],
        family_override="fus3d_confidence_only_control",
        color_override=RESIDUAL_COLORS["fus3d_confidence_only_control"],
    )

    guided_combined = bgs0.merge_gaussians([baseline, guided_residual])
    random_combined = bgs0.merge_gaussians([baseline, random_residual])
    confidence_combined = bgs0.merge_gaussians([baseline, confidence_residual])

    bgs0.write_ply(output_dir / "b_gs2_fus3d_guided_residual_only.ply", guided_residual)
    bgs0.write_ply(output_dir / "b_gs2_fus3d_guided_combined.ply", guided_combined)
    bgs0.write_ply(output_dir / "b_gs2_random_residual_control_combined.ply", random_combined)
    bgs0.write_ply(output_dir / "b_gs2_confidence_only_control_combined.ply", confidence_combined)
    np.savez_compressed(
        output_dir / "b_gs2_fus3d_guided_residual_diagnostics.npz",
        selected_guided_idx=selections["guided"],
        selected_random_idx=selections["random"],
        selected_confidence_only_idx=selections["confidence_only"],
        score=np.asarray(candidates["score"], dtype=np.float32),
        uncertainty=np.asarray(candidates["uncertainty"], dtype=np.float32),
        confidence=np.asarray(candidates["confidence"], dtype=np.float32),
        visibility=np.asarray(candidates["visibility"], dtype=np.float32),
        sdf=np.asarray(candidates["sdf"], dtype=np.float32),
        occupancy=np.asarray(candidates["occupancy"], dtype=np.float32),
        mask_support=np.asarray(candidates["mask_support"], dtype=np.int32),
        projected_visible=np.asarray(candidates["projected_visible"], dtype=np.int32),
        outside=np.asarray(candidates["outside"], dtype=np.int32),
        gap_support=np.asarray(candidates["gap_support"], dtype=np.int32),
        silhouette_proxy=np.asarray(candidates["silhouette_proxy"], dtype=np.float32),
        nearest_baseline_distance=np.asarray(candidates["nearest_baseline_distance"], dtype=np.float32),
        eligible=np.asarray(candidates["eligible"], dtype=bool),
    )

    metrics = {
        "constrained_baseline": bgs0.render_gaussian_set(
            "constrained_baseline",
            baseline,
            views,
            cameras,
            target_size=args.target_size,
            point_radius=args.point_radius,
            output_dir=output_dir,
        ),
        "fus3d_guided_residual": bgs0.render_gaussian_set(
            "fus3d_guided_residual",
            guided_combined,
            views,
            cameras,
            target_size=args.target_size,
            point_radius=args.point_radius,
            output_dir=output_dir,
        ),
        "random_residual_control": bgs0.render_gaussian_set(
            "random_residual_control",
            random_combined,
            views,
            cameras,
            target_size=args.target_size,
            point_radius=args.point_radius,
            output_dir=output_dir,
        ),
        "confidence_only_control": bgs0.render_gaussian_set(
            "confidence_only_control",
            confidence_combined,
            views,
            cameras,
            target_size=args.target_size,
            point_radius=args.point_radius,
            output_dir=output_dir,
        ),
    }
    base = metrics["constrained_baseline"]
    guided = metrics["fus3d_guided_residual"]
    random_control = metrics["random_residual_control"]
    confidence_control = metrics["confidence_only_control"]
    comparison = {
        "guided_minus_constrained_iou": float(guided["mean_iou"] - base["mean_iou"]),
        "guided_minus_constrained_overfill": float(guided["mean_overfill_ratio"] - base["mean_overfill_ratio"]),
        "guided_minus_constrained_rgb_residual": float(guided["mean_rgb_residual"] - base["mean_rgb_residual"]),
        "guided_minus_random_iou": float(guided["mean_iou"] - random_control["mean_iou"]),
        "guided_minus_random_overfill": float(guided["mean_overfill_ratio"] - random_control["mean_overfill_ratio"]),
        "guided_minus_random_rgb_residual": float(guided["mean_rgb_residual"] - random_control["mean_rgb_residual"]),
        "guided_minus_confidence_only_iou": float(guided["mean_iou"] - confidence_control["mean_iou"]),
        "guided_minus_confidence_only_overfill": float(guided["mean_overfill_ratio"] - confidence_control["mean_overfill_ratio"]),
        "guided_minus_confidence_only_rgb_residual": float(guided["mean_rgb_residual"] - confidence_control["mean_rgb_residual"]),
        "metrics": metrics,
    }
    bounded_success = (
        comparison["guided_minus_constrained_iou"] >= 0.0
        and comparison["guided_minus_constrained_overfill"] <= float(args.overfill_budget)
        and comparison["guided_minus_constrained_rgb_residual"] <= 0.0
        and comparison["guided_minus_random_iou"] >= 0.0
    )
    if bounded_success:
        decision = (
            "B-GS2 Fus3D-guided residual layer executed and met the bounded local smoke criteria. "
            "It remains research-only and writes no pass/export state."
        )
    else:
        decision = (
            "B-GS2 Fus3D-guided residual layer executed and produced real residual Gaussian artifacts, "
            "but the bounded IoU/overfill/RGB/control criteria were not all met. Treat this as a "
            "research smoke result, not a candidate or pass."
        )

    selection_summary = {
        "candidate_diagnostics": candidate_diagnostics,
        "guided": summarize_selection("guided", candidates, selections["guided"]),
        "random_control": summarize_selection("random_control", candidates, selections["random"]),
        "confidence_only_control": summarize_selection("confidence_only_control", candidates, selections["confidence_only"]),
    }
    key_outputs = [
        str((output_dir / "b_gs2_fus3d_guided_residual_only.ply").resolve()),
        str((output_dir / "b_gs2_fus3d_guided_combined.ply").resolve()),
        str((output_dir / "b_gs2_random_residual_control_combined.ply").resolve()),
        str((output_dir / "b_gs2_confidence_only_control_combined.ply").resolve()),
        str((output_dir / "b_gs2_fus3d_guided_residual_diagnostics.npz").resolve()),
        str((output_dir / "b_gs2_fus3d_guided_residual_summary.json").resolve()),
        str((output_dir / "b_gs2_fus3d_guided_residual_report.md").resolve()),
    ]
    summary = {
        "task": "b_gs2_fus3d_guided_residual_gaussian_layer",
        "schema_version": "B_GS2_fus3d_guided_residual_gaussian_layer_v1",
        "status": "research_only_residual_gaussian_smoke_no_export",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.expanduser().resolve()),
            "bgs1_dir": str(bgs1_dir),
            "bgs1_constrained_baseline_ply": str(baseline_ply),
            "bgs1_summary_json": str(bgs1_dir / "b_gs1_summary.json"),
            "fus3d_fields": str(args.fus3d_fields.expanduser().resolve()),
            "latent_grid": str(args.latent_grid.expanduser().resolve()),
            "camera_source": camera_source,
            "view_indices": [int(view["view_index"]) for view in views],
            "target_size": int(args.target_size),
        },
        "bgs1_artifact_context": {
            "loaded_constrained_baseline_points": int(baseline["points"].shape[0]),
            "bgs1_status": bgs1_summary.get("status") if bgs1_summary else None,
            "bgs1_selected_free_count": bgs1_summary.get("selected_free_count") if bgs1_summary else None,
            "bgs1_success_local_bounded": bgs1_summary.get("success_local_bounded") if bgs1_summary else None,
        },
        "parameters": {
            "occupancy_threshold": float(args.occupancy_threshold),
            "sdf_uncertainty_width": float(args.sdf_uncertainty_width),
            "normal_residual_scale": float(args.normal_residual_scale),
            "min_mask_support": int(args.min_mask_support),
            "min_visible_views": int(args.min_visible_views),
            "max_outside_frac": float(args.max_outside_frac),
            "min_distance_to_baseline": float(args.min_distance_to_baseline),
            "max_residual_gaussians": int(args.max_residual_gaussians),
            "point_radius": int(args.point_radius),
        },
        "selection": selection_summary,
        "comparison": comparison,
        "success_local_bounded": bool(bounded_success),
        "decision": decision,
        "key_outputs": key_outputs,
        "pass": False,
        "teacher_export": "blocked",
        "candidate_export": "blocked",
        "predictions_export": "blocked",
        "registry_write": "blocked",
    }
    write_json(output_dir / "b_gs2_fus3d_guided_residual_summary.json", summary)
    write_json(output_dir / "b_gs2_fus3d_guided_residual_comparison.json", comparison)
    write_report(output_dir / "b_gs2_fus3d_guided_residual_report.md", summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "success_local_bounded": bounded_success,
                    "guided_residual_count": int(selections["guided"].size),
                    "guided_minus_constrained_iou": comparison["guided_minus_constrained_iou"],
                    "guided_minus_constrained_overfill": comparison["guided_minus_constrained_overfill"],
                    "guided_minus_constrained_rgb_residual": comparison["guided_minus_constrained_rgb_residual"],
                    "output_dir": str(output_dir),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
