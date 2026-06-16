from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rasterize_shared_surfel_predictions import (  # noqa: E402
    ROI_NAMES,
    aggregate_surfels,
    build_failure_analysis,
    build_roi_masks,
    channel_index,
    collect_observations,
    json_ready,
    parse_sources,
    quantize_canonical,
    rasterize_surfels,
    recompute_normals,
)
from render_open3d_pointcloud import load_rgb_stack  # noqa: E402


REQUIRED_PREDICTION_KEYS = (
    "pose_enc",
    "extrinsic",
    "intrinsic",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "normal",
    "normal_conf",
)

OPTIONAL_RENDERER_MODULES = ("pytorch3d", "nvdiffrast", "kaolin")
OPTIONAL_GEOMETRY_MODULES = ("open3d", "trimesh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only surface optimization upper-bound preflight. This is not "
            "VGGT training and not a cloud-unblocker: it tests whether dense "
            "same-frame observations can form a shared human surface that "
            "rasterizes back to the original sparse-view protocol."
        )
    )
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--source-scene-dir", required=True, type=Path)
    parser.add_argument("--target-predictions", required=True, type=Path)
    parser.add_argument("--target-scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--canonical-bin-size", type=float, default=0.012)
    parser.add_argument("--min-surfel-observations", type=int, default=8)
    parser.add_argument("--min-surfel-views", type=int, default=3)
    parser.add_argument("--max-surfel-spread", type=float, default=0.050)
    parser.add_argument("--max-raster-distance", type=float, default=0.080)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--visible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-dilate", type=int, default=1)
    parser.add_argument("--debug-view-indices", default="0,2,3")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_preflight() -> dict[str, Any]:
    optional = {
        name: module_available(name)
        for name in ("torch", "smplx", "cv2", *OPTIONAL_RENDERER_MODULES, *OPTIONAL_GEOMETRY_MODULES)
    }
    differentiable_renderer_available = any(optional.get(name, False) for name in OPTIONAL_RENDERER_MODULES)
    return {
        "optional_modules": optional,
        "differentiable_renderer_available": bool(differentiable_renderer_available),
        "true_gradient_surface_optimization_available": bool(differentiable_renderer_available),
        "preflight_note": (
            "No differentiable renderer is required for this diagnostic. "
            "If pytorch3d/nvdiffrast/kaolin are all absent, the run is a dense-observation "
            "upper-bound surfel preflight, not true gradient-based silhouette/photometric optimization."
        ),
    }


def load_checked_npz(path: Path, *, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in required_keys if key not in payload.files]
            if missing:
                raise KeyError(f"{path} missing required key(s): {missing}; available={payload.files}")
            return {key: np.asarray(payload[key]) for key in payload.files}
    except Exception as exc:  # noqa: BLE001 - convert corrupt npz into a clear preflight failure.
        raise RuntimeError(f"Could not load NPZ {path}: {type(exc).__name__}: {exc}") from exc


def load_prior(scene_dir: Path, *, view_count: int, height: int, width: int, visible_only: bool) -> dict[str, Any]:
    prior_path = scene_dir / "prior_maps.npz"
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    with np.load(prior_path, allow_pickle=False) as priors:
        required = ("prior_maps", "prior_mask", "prior_channels")
        missing = [key for key in required if key not in priors.files]
        if missing:
            raise KeyError(f"{prior_path} missing {missing}")
        prior_maps = np.asarray(priors["prior_maps"], dtype=np.float32)
        prior_mask = np.asarray(priors["prior_mask"], dtype=bool)
        channel_names = [str(value) for value in priors["prior_channels"].tolist()]
    if prior_maps.shape[0] != view_count or prior_maps.shape[2:] != (height, width):
        raise ValueError(
            f"{prior_path} prior_maps shape {prior_maps.shape} does not match "
            f"predictions {(view_count, height, width)}"
        )
    if prior_mask.shape != (view_count, height, width):
        raise ValueError(f"{prior_path} prior_mask shape {prior_mask.shape} does not match {(view_count, height, width)}")
    canonical_indices = [
        channel_index(channel_names, "smplx_canonical_x"),
        channel_index(channel_names, "smplx_canonical_y"),
        channel_index(channel_names, "smplx_canonical_z"),
    ]
    canonical = prior_maps[:, canonical_indices].transpose(0, 2, 3, 1)
    visible = np.ones_like(prior_mask, dtype=bool)
    visible_index = channel_names.index("smplx_visible_mask") if "smplx_visible_mask" in channel_names else None
    if bool(visible_only) and visible_index is not None:
        visible = prior_maps[:, visible_index] > 0.5
    return {
        "path": str(prior_path.resolve()),
        "prior_maps": prior_maps,
        "prior_mask": prior_mask,
        "channel_names": channel_names,
        "canonical": canonical.astype(np.float32),
        "visible": visible.astype(bool),
        "has_visible_mask": bool(visible_index is not None),
        "body_part_channels": [name for name in channel_names if name.startswith("smplx_body_part_emb_")],
    }


def parse_view_indices(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if 0 <= idx < view_count:
            out.append(idx)
    return sorted(set(out))


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(finite))
        lo = float(np.nanmin(finite))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (np.clip(values, lo, hi) - lo) / (hi - lo)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.asarray(np.round(255.0 * scaled), dtype=np.uint8)


def save_debug_maps(output_dir: Path, rasterized: dict[str, np.ndarray], *, view_indices: list[int]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    changed = np.asarray(rasterized["shared_surfel_changed_mask"], dtype=np.uint8)
    support = np.asarray(rasterized["shared_surfel_view_support"], dtype=np.float32)
    distance = np.asarray(rasterized["shared_surfel_raster_distance"], dtype=np.float32)
    for view_idx in view_indices:
        changed_path = output_dir / f"view{view_idx:02d}_changed_mask.png"
        support_path = output_dir / f"view{view_idx:02d}_view_support.png"
        distance_path = output_dir / f"view{view_idx:02d}_raster_distance.png"
        Image.fromarray(changed[view_idx] * 255).save(changed_path)
        Image.fromarray(normalize_to_uint8(support[view_idx])).save(support_path)
        Image.fromarray(normalize_to_uint8(distance[view_idx])).save(distance_path)
        written.extend([str(changed_path.resolve()), str(support_path.resolve()), str(distance_path.resolve())])
    return written


def write_surfel_ply(path: Path, surfels: dict[str, np.ndarray]) -> None:
    points = np.asarray(surfels.get("position", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    support = np.asarray(surfels.get("view_support", np.zeros((points.shape[0],), dtype=np.float32)), dtype=np.float32)
    obs = np.asarray(surfels.get("observation_count", np.zeros((points.shape[0],), dtype=np.float32)), dtype=np.float32)
    support_u8 = normalize_to_uint8(support)
    obs_u8 = normalize_to_uint8(obs)
    blue = np.full((points.shape[0],), 180, dtype=np.uint8)
    colors = np.stack([support_u8, obs_u8, blue], axis=1) if points.size else np.zeros((0, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_blocked_outputs(output_dir: Path, payload: dict[str, Any]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "surface_upperbound_summary.json"
    summary_path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# Human Surface Upper-Bound Preflight",
        "",
        "## Status",
        "",
        f"- Status: `{payload.get('truthful_status', 'blocked')}`",
        "- Cloud: blocked",
        "- Mentor pass: not claimed",
        "",
        "## Blocking Reason",
        "",
        str(payload.get("error", "unknown")),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
    return 2


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deps = dependency_preflight()
    base_summary: dict[str, Any] = {
        "task": "optimize_human_surface_upperbound",
        "truthful_status": "starting_local_only_not_cloud",
        "dependency_preflight": deps,
        "source_predictions": str(args.source_predictions.resolve()),
        "source_scene_dir": str(args.source_scene_dir.resolve()),
        "target_predictions": str(args.target_predictions.resolve()),
        "target_scene_dir": str(args.target_scene_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "notes": [
            "This is not VGGT training.",
            "SMPL-X prior maps are used only as canonical correspondence, visibility, and weak body-part structure.",
            "The output must still go through the existing strict candidate/teacher gate and explicit Open3D visual review.",
            "If Open3D remains shell-like, this preflight is negative and must not be tuned into a numeric-only pass.",
        ],
    }

    try:
        sources = parse_sources(args.sources)
        source = load_checked_npz(args.source_predictions, required_keys=REQUIRED_PREDICTION_KEYS)
        target = load_checked_npz(args.target_predictions, required_keys=REQUIRED_PREDICTION_KEYS)
        source_world = np.asarray(source["world_points"], dtype=np.float32)
        target_world = np.asarray(target["world_points"], dtype=np.float32)
        source_view_count, source_height, source_width, _ = source_world.shape
        target_view_count, target_height, target_width, _ = target_world.shape
        if source_height != target_height or source_width != target_width:
            raise ValueError(
                f"source resolution {(source_height, source_width)} != target resolution {(target_height, target_width)}"
            )
        source_prior = load_prior(
            args.source_scene_dir,
            view_count=source_view_count,
            height=source_height,
            width=source_width,
            visible_only=bool(args.visible_only),
        )
        target_prior = load_prior(
            args.target_scene_dir,
            view_count=target_view_count,
            height=target_height,
            width=target_width,
            visible_only=bool(args.visible_only),
        )
        source_rgb = load_rgb_stack(args.source_scene_dir / "images", target_size=source_height)
        target_rgb = load_rgb_stack(args.target_scene_dir / "images", target_size=target_height)
        source_roi_masks = build_roi_masks(args.source_scene_dir, source_height, source_rgb)
        target_roi_masks = build_roi_masks(args.target_scene_dir, target_height, target_rgb)

        source_support = source_roi_masks["full"] & source_prior["prior_mask"] & source_prior["visible"]
        target_support = target_roi_masks["full"] & target_prior["prior_mask"] & target_prior["visible"]
        source_canonical_q = quantize_canonical(source_prior["canonical"], float(args.canonical_bin_size))
        target_canonical_q = quantize_canonical(target_prior["canonical"], float(args.canonical_bin_size))

        observations, obs_summary = collect_observations(
            sources=sources,
            predictions=source,
            canonical_q=source_canonical_q,
            support_base=source_support,
            roi_masks=source_roi_masks,
            conf_percentile=float(args.conf_percentile),
        )
        key_to_index, surfels, surfel_summary = aggregate_surfels(
            observations,
            min_observations=int(args.min_surfel_observations),
            min_views=int(args.min_surfel_views),
            max_spread=float(args.max_surfel_spread),
        )
        rasterized, raster_summary = rasterize_surfels(
            predictions=target,
            canonical_q=target_canonical_q,
            support_base=target_support,
            roi_masks=target_roi_masks,
            key_to_index=key_to_index,
            surfels=surfels,
            max_distance=float(args.max_raster_distance),
            alpha=float(args.alpha),
            ray_consistent=True,
        )

        out_predictions = {key: np.asarray(value) for key, value in target.items()}
        out_predictions["world_points"] = rasterized["world_points"]
        out_predictions["depth"] = rasterized["depth"]
        out_predictions["upperbound_changed_mask"] = rasterized["shared_surfel_changed_mask"]
        out_predictions["upperbound_view_support"] = rasterized["shared_surfel_view_support"]
        out_predictions["upperbound_observation_count"] = rasterized["shared_surfel_observation_count"]
        out_predictions["upperbound_spread_p75"] = rasterized["shared_surfel_spread_p75"]
        out_predictions["upperbound_raster_distance"] = rasterized["shared_surfel_raster_distance"]
        normal_summary = {"enabled": False}
        if "normal" in out_predictions and "normal_conf" in out_predictions:
            normal, normal_conf, normal_summary = recompute_normals(
                target,
                out_predictions["world_points"],
                rasterized["shared_surfel_changed_mask"].astype(bool),
                dilate=int(args.normal_dilate),
            )
            out_predictions["normal"] = normal
            out_predictions["normal_conf"] = normal_conf

        output_predictions = args.output_dir / "predictions.npz"
        output_surfels = args.output_dir / "optimized_surface_surfels.npz"
        output_ply = args.output_dir / "optimized_surface_surfels.ply"
        np.savez_compressed(output_predictions, **out_predictions)
        np.savez_compressed(output_surfels, **surfels)
        write_surfel_ply(output_ply, surfels)
        debug_maps = save_debug_maps(
            args.output_dir / "debug_maps",
            rasterized,
            view_indices=parse_view_indices(args.debug_view_indices, target_view_count),
        )

        failure_analysis = build_failure_analysis(surfel_summary=surfel_summary, raster_summary=raster_summary)
        dense_to_sparse = bool(source_view_count > target_view_count)
        differentiable_ready = bool(deps["differentiable_renderer_available"])
        truthful_status = (
            "dense_observation_upperbound_preflight_not_true_differentiable_optimization"
            if dense_to_sparse and not differentiable_ready
            else "local_surface_upperbound_preflight_not_cloud"
        )
        summary = {
            **base_summary,
            "truthful_status": truthful_status,
            "output_predictions": str(output_predictions.resolve()),
            "output_surfels": str(output_surfels.resolve()),
            "output_surfel_ply": str(output_ply.resolve()),
            "debug_maps": debug_maps,
            "config": {
                "sources": sources,
                "conf_percentile": float(args.conf_percentile),
                "canonical_bin_size": float(args.canonical_bin_size),
                "min_surfel_observations": int(args.min_surfel_observations),
                "min_surfel_views": int(args.min_surfel_views),
                "max_surfel_spread": float(args.max_surfel_spread),
                "max_raster_distance": float(args.max_raster_distance),
                "alpha": float(args.alpha),
                "visible_only": bool(args.visible_only),
            },
            "asset_summary": {
                "source_view_count": int(source_view_count),
                "target_view_count": int(target_view_count),
                "dense_to_sparse": dense_to_sparse,
                "source_prior": {
                    "path": source_prior["path"],
                    "has_visible_mask": source_prior["has_visible_mask"],
                    "body_part_channels": len(source_prior["body_part_channels"]),
                },
                "target_prior": {
                    "path": target_prior["path"],
                    "has_visible_mask": target_prior["has_visible_mask"],
                    "body_part_channels": len(target_prior["body_part_channels"]),
                },
                "source_support_pixels": int(source_support.sum()),
                "target_support_pixels": int(target_support.sum()),
                "source_roi_pixels": {roi: int(mask.sum()) for roi, mask in source_roi_masks.items()},
                "target_roi_pixels": {roi: int(mask.sum()) for roi, mask in target_roi_masks.items()},
            },
            "observation_summary": obs_summary,
            "surfel_summary": surfel_summary,
            "rasterization_summary": raster_summary,
            "normal_recompute": normal_summary,
            "failure_analysis": failure_analysis,
            "strict_gate_required": {
                "candidate_gate": "tools/package_normal_candidate_gate.py",
                "teacher_gate": "tools/audit_headface_teacher_surface.py if used as teacher",
                "cloud_allowed": False,
                "reason": "local strict gate must pass and explicit Open3D visual review must show normal human geometry",
            },
        }
        (args.output_dir / "surface_upperbound_summary.json").write_text(
            json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_lines = [
            "# Human Surface Upper-Bound Preflight",
            "",
            "## Truthful Status",
            "",
            f"- Status: `{truthful_status}`",
            "- Mentor pass: `False` until strict gate and visual review pass.",
            "- Cloud: `blocked`.",
            f"- Source views: `{source_view_count}`",
            f"- Target views: `{target_view_count}`",
            f"- Differentiable renderer available: `{differentiable_ready}`",
            "",
            "## Outputs",
            "",
            f"- Predictions: `{output_predictions.resolve()}`",
            f"- Surfels: `{output_surfels.resolve()}`",
            f"- Surfel PLY: `{output_ply.resolve()}`",
            "",
            "## Surface Evidence",
            "",
            f"- Accepted surfels: `{surfel_summary.get('accepted_surfels', 0)}`",
            f"- Face surfels: `{surfel_summary.get('roi_surfel_counts', {}).get('face', 0)}`",
            f"- Hand surfels: `{surfel_summary.get('roi_surfel_counts', {}).get('hands', 0)}`",
            f"- Changed target pixels: `{raster_summary.get('changed_pixels_total', 0)}`",
            f"- Face changed fraction: `{raster_summary.get('roi_rasterization', {}).get('face', {}).get('changed_fraction', 0.0):.4f}`",
            f"- Head changed fraction: `{raster_summary.get('roi_rasterization', {}).get('head', {}).get('changed_fraction', 0.0):.4f}`",
            f"- Hands changed fraction: `{raster_summary.get('roi_rasterization', {}).get('hands', {}).get('changed_fraction', 0.0):.4f}`",
            "",
            "## Required Next Gate",
            "",
            "Run this output through `tools/package_normal_candidate_gate.py` with candidate-specific full-body input. "
            "If the Open3D sheet is still a shell/slab/fragmented-hands result, freeze this upper-bound preflight.",
            "",
        ]
        (args.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
        print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - this is a preflight script; produce a durable blocker report.
        blocked = {
            **base_summary,
            "truthful_status": "blocked_before_surface_upperbound_output",
            "error": f"{type(exc).__name__}: {exc}",
            "cloud_allowed": False,
            "mentor_pass": False,
        }
        return write_blocked_outputs(args.output_dir, blocked)


if __name__ == "__main__":
    raise SystemExit(main())
