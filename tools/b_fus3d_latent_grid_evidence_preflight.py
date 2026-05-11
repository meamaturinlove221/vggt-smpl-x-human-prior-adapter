from __future__ import annotations

import argparse
import json
import math
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

from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    load_connected_mesh,
    load_view_rgb_mask,
    parse_view_indices,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TOKEN_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_latent_grid_evidence_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "latent_grid_evidence_preflight_only": True,
    "not_decoder": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "uses_vggt_depth_point_normal_as_hard_teacher": False,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_checkpoint": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D latent-grid evidence preflight. It samples a "
            "small 3D grid around the connected human scaffold, projects grid "
            "points into raw RGB/masks and VGGT aggregator patch tokens, and "
            "reports whether a real learned SDF/surface backend has enough "
            "multi-view evidence to be worth implementing. It never trains, "
            "exports a teacher/candidate, writes predictions, writes strict pass "
            "state, or calls cloud."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--grid-resolution", type=int, default=18)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.08)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--min-visible-views", type=int, default=2)
    parser.add_argument("--view-indices", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def stat_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"count": int(arr.size), "finite": 0}
    data = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(data.size),
        "min": float(data.min()),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.quantile(data, 0.50)),
        "mean": float(data.mean()),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(data.max()),
    }


def load_tokens(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        patch_start_idx = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
        selected_view_indices = [int(v) for v in np.asarray(payload["selected_view_indices"]).reshape(-1)]
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        raise ValueError(f"Expected token cache shape [1,S,T,C], got {tokens.shape}")
    patch_tokens = int(tokens.shape[2] - patch_start_idx)
    grid = int(round(math.sqrt(max(patch_tokens, 1))))
    if grid * grid != patch_tokens:
        raise ValueError(f"Patch-token count {patch_tokens} is not a square grid")
    return {
        "path": str(resolved),
        "tokens": tokens,
        "patch_start_idx": patch_start_idx,
        "selected_view_indices": selected_view_indices,
        "patch_grid": grid,
        "feature_dim": int(tokens.shape[-1]),
    }


def load_scene_views(
    scene_dir: Path,
    dataset_root: Path | None,
    subset_name: str,
    selected_view_indices: list[int],
    target_size: int,
) -> tuple[list[dict[str, Any]], str]:
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    exported = manifest["exported_views"]
    if selected_view_indices:
        view_indices = parse_view_indices(",".join(str(v) for v in selected_view_indices), len(exported))
    else:
        view_indices = list(range(min(6, len(exported))))
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    rows: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = exported[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb, mask = load_view_rgb_mask(view, target_size)
        rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "intrinsic": intrinsic.astype(np.float32),
                "world_to_cam": world_to_cam.astype(np.float32),
                "rgb": rgb.astype(np.float32) / 255.0,
                "mask": mask.astype(bool),
            }
        )
    return rows, camera_source


def make_grid(vertices: np.ndarray, resolution: int, padding_ratio: float, max_points: int) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    pad = np.maximum(extent * float(padding_ratio), 1e-4)
    lo = vmin - pad
    hi = vmax + pad
    xs = np.linspace(float(lo[0]), float(hi[0]), int(resolution), dtype=np.float32)
    ys = np.linspace(float(lo[1]), float(hi[1]), int(resolution), dtype=np.float32)
    zs = np.linspace(float(lo[2]), float(hi[2]), int(resolution), dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if max_points > 0 and grid.shape[0] > max_points:
        keep = np.linspace(0, grid.shape[0] - 1, int(max_points)).round().astype(np.int64)
        grid = grid[keep]
    return grid, {
        "bbox_min": lo,
        "bbox_max": hi,
        "bbox_extent": hi - lo,
        "grid_resolution": int(resolution),
        "sampled_points": int(grid.shape[0]),
    }


def project_points(points: np.ndarray, view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = view["world_to_cam"][:3, :3]
    translation = view["world_to_cam"][:3, 3]
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (view["intrinsic"] @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32), cam.astype(np.float32)


def gather_view_evidence(points: np.ndarray, views: list[dict[str, Any]], tokens: dict[str, Any]) -> dict[str, np.ndarray]:
    n = points.shape[0]
    view_count = len(views)
    patch_grid = int(tokens["patch_grid"])
    patch_start_idx = int(tokens["patch_start_idx"])
    token_array = np.asarray(tokens["tokens"], dtype=np.float32)
    token_view_to_pos = {int(view_idx): idx for idx, view_idx in enumerate(tokens["selected_view_indices"])}
    visible = np.zeros((n, view_count), dtype=bool)
    mask_inside = np.zeros((n, view_count), dtype=bool)
    rgb = np.zeros((n, view_count, 3), dtype=np.float32)
    token_feat = np.zeros((n, view_count, token_array.shape[-1]), dtype=np.float32)
    token_available = np.zeros((n, view_count), dtype=bool)
    depth = np.full((n, view_count), np.nan, dtype=np.float32)
    for vpos, view in enumerate(views):
        uv, z, _cam = project_points(points, view)
        h, w = view["mask"].shape
        xi = np.rint(uv[:, 0]).astype(np.int64)
        yi = np.rint(uv[:, 1]).astype(np.int64)
        inside = np.isfinite(uv).all(axis=1) & (z > 1e-6) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        visible[:, vpos] = inside
        depth[inside, vpos] = z[inside]
        if np.any(inside):
            rgb[inside, vpos] = view["rgb"][yi[inside], xi[inside]]
            mask_inside[inside, vpos] = view["mask"][yi[inside], xi[inside]]
            if int(view["view_index"]) in token_view_to_pos:
                patch_x = np.floor(np.clip(uv[inside, 0], 0, w - 1) / max(w / patch_grid, 1e-6)).astype(np.int64)
                patch_y = np.floor(np.clip(uv[inside, 1], 0, h - 1) / max(h / patch_grid, 1e-6)).astype(np.int64)
                patch_x = np.clip(patch_x, 0, patch_grid - 1)
                patch_y = np.clip(patch_y, 0, patch_grid - 1)
                token_ids = patch_start_idx + patch_y * patch_grid + patch_x
                tpos = token_view_to_pos[int(view["view_index"])]
                inside_idx = np.flatnonzero(inside)
                token_feat[inside_idx, vpos] = token_array[0, tpos, token_ids]
                token_available[inside_idx, vpos] = True
    return {
        "visible": visible,
        "mask_inside": mask_inside,
        "rgb": rgb,
        "token_feat": token_feat,
        "token_available": token_available,
        "depth": depth,
    }


def cosine_pair_stats(features: np.ndarray, available: np.ndarray) -> np.ndarray:
    n, v, c = features.shape
    out = np.full((n,), np.nan, dtype=np.float32)
    norms = np.linalg.norm(features, axis=-1)
    for i in range(n):
        idx = np.flatnonzero(available[i] & (norms[i] > 1e-8))
        if idx.size < 2:
            continue
        vals: list[float] = []
        for a in range(idx.size):
            fa = features[i, idx[a]]
            na = float(norms[i, idx[a]])
            for b in range(a + 1, idx.size):
                fb = features[i, idx[b]]
                nb = float(norms[i, idx[b]])
                vals.append(float(np.dot(fa, fb) / max(na * nb, 1e-8)))
        if vals:
            out[i] = float(np.mean(vals))
    return out


def color_points_by_score(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(scores)
    norm = np.zeros(scores.shape, dtype=np.float32)
    if finite.any():
        lo, hi = np.percentile(scores[finite], [5, 95])
        if hi <= lo:
            hi = float(scores[finite].max())
            lo = float(scores[finite].min())
        if hi > lo:
            norm[finite] = np.clip((scores[finite] - lo) / (hi - lo), 0, 1)
    return np.stack([norm, 0.25 + 0.5 * (1.0 - np.abs(norm - 0.5) * 2.0), 1.0 - norm], axis=1)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    cols = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols, strict=False):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "# B-Fus3D15 Latent Grid Evidence Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only evidence cache for a possible learned 3D SDF/surface backend.",
        "It is not a decoder, teacher, candidate, or strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {STRICT_FACTS['strict_candidate_passes']}",
        f"strict_teacher_passes = {STRICT_FACTS['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {STRICT_FACTS['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Evidence Summary",
        "",
        "```json",
        json.dumps(json_ready(summary["evidence_summary"]), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(json_ready(decision), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_connected_mesh(args.template_payload)
    tokens = load_tokens(args.token_cache)
    selected = parse_view_indices(args.view_indices, 60) if args.view_indices.strip() else list(tokens["selected_view_indices"])
    views, camera_source = load_scene_views(args.scene_dir, args.dataset_root, args.subset_name, selected, args.target_size)
    points, grid_meta = make_grid(mesh["vertices"], args.grid_resolution, args.bbox_padding_ratio, args.max_points)
    ev = gather_view_evidence(points, views, tokens)

    visible_count = ev["visible"].sum(axis=1).astype(np.int32)
    mask_count = ev["mask_inside"].sum(axis=1).astype(np.int32)
    token_count = ev["token_available"].sum(axis=1).astype(np.int32)
    supported = mask_count >= int(args.min_visible_views)
    rgb_valid = ev["mask_inside"]
    rgb_var = np.full((points.shape[0],), np.nan, dtype=np.float32)
    rgb_range = np.full((points.shape[0],), np.nan, dtype=np.float32)
    for idx in range(points.shape[0]):
        rows = ev["rgb"][idx, rgb_valid[idx]]
        if rows.shape[0] >= 2:
            rgb_var[idx] = float(rows.var(axis=0).mean())
            rgb_range[idx] = float((rows.max(axis=0) - rows.min(axis=0)).mean())
    token_cos = cosine_pair_stats(ev["token_feat"], ev["token_available"])
    occupancy_ratio = mask_count.astype(np.float32) / np.maximum(visible_count.astype(np.float32), 1.0)

    strong = supported & np.isfinite(rgb_var) & (token_count >= int(args.min_visible_views))
    evidence_score = (
        occupancy_ratio
        - 0.35 * np.nan_to_num(rgb_var, nan=0.25)
        + 0.10 * np.nan_to_num(token_cos, nan=0.0)
        + 0.05 * np.clip(token_count / max(len(views), 1), 0, 1)
    ).astype(np.float32)
    boundary_like = (occupancy_ratio > 0.15) & (occupancy_ratio < 0.85) & (visible_count >= int(args.min_visible_views))

    arrays_path = output_dir / "b_fus3d_latent_grid_evidence_arrays.npz"
    np.savez_compressed(
        arrays_path,
        points=points.astype(np.float32),
        visible_count=visible_count,
        mask_count=mask_count,
        token_count=token_count,
        occupancy_ratio=occupancy_ratio.astype(np.float32),
        rgb_variance=rgb_var.astype(np.float32),
        rgb_range=rgb_range.astype(np.float32),
        token_cosine=token_cos.astype(np.float32),
        evidence_score=evidence_score.astype(np.float32),
        boundary_like=boundary_like.astype(np.bool_),
        selected_view_indices=np.asarray([v["view_index"] for v in views], dtype=np.int32),
    )

    write_ply(output_dir / "latent_grid_colored_by_occupancy.ply", points, color_points_by_score(occupancy_ratio))
    write_ply(output_dir / "latent_grid_colored_by_evidence_score.ply", points, color_points_by_score(evidence_score))
    write_ply(output_dir / "latent_grid_boundary_like_points.ply", points[boundary_like], color_points_by_score(evidence_score[boundary_like]))

    evidence_summary = {
        "point_count": int(points.shape[0]),
        "view_count": int(len(views)),
        "camera_source": camera_source,
        "grid_meta": grid_meta,
        "visible_count": stat_array(visible_count),
        "mask_count": stat_array(mask_count),
        "token_count": stat_array(token_count),
        "occupancy_ratio": stat_array(occupancy_ratio),
        "rgb_variance_supported": stat_array(rgb_var[supported]),
        "rgb_range_supported": stat_array(rgb_range[supported]),
        "token_cosine_supported": stat_array(token_cos[supported]),
        "evidence_score": stat_array(evidence_score),
        "supported_count": int(supported.sum()),
        "supported_ratio": float(supported.mean()) if supported.size else 0.0,
        "strong_count": int(strong.sum()),
        "strong_ratio": float(strong.mean()) if strong.size else 0.0,
        "boundary_like_count": int(boundary_like.sum()),
        "boundary_like_ratio": float(boundary_like.mean()) if boundary_like.size else 0.0,
    }
    decision = {
        "status": "research_latent_grid_evidence_no_pass",
        "sdf_backend_signal_present": bool(evidence_summary["supported_ratio"] >= 0.10 and evidence_summary["boundary_like_count"] > 0),
        "sufficient_for_training": False,
        "sufficient_for_teacher_or_candidate": False,
        "interpretation": (
            "This preflight checks whether raw masks/RGB and VGGT tokens can be "
            "queried by a 3D latent grid. It is an evidence-routing contract, "
            "not a learned SDF, not a mesh, and not a strict visual result."
        ),
        "next_allowed_action": (
            "If signal is present, implement one bounded learned 3D SDF/surface-token "
            "backend smoke with fixed resolution and Open3D precheck. Do not tune "
            "B14 sparse offsets, thresholds, hidden size, or steps."
        ),
        "blocked_actions": [
            "do_not_claim_surface_or_visual_pass",
            "do_not_export_teacher_or_candidate",
            "do_not_unblock_cloud",
            "do_not_train_from_this_evidence_cache_alone",
            "do_not_restart_B14_sparse_offset_loop",
        ],
    }
    summary = {
        **STRICT_FACTS,
        "status": "research_only_latent_grid_evidence_preflight_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir),
            "token_cache": str(args.token_cache),
            "template_payload": str(args.template_payload),
            "selected_views": [int(v["view_index"]) for v in views],
            "target_size": int(args.target_size),
            "grid_resolution": int(args.grid_resolution),
            "min_visible_views": int(args.min_visible_views),
        },
        "evidence_summary": evidence_summary,
        "decision": decision,
        "outputs": {
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_fus3d_latent_grid_evidence_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_latent_grid_evidence_summary.md"),
            "status_report": str(args.status_report),
            "occupancy_ply": str(output_dir / "latent_grid_colored_by_occupancy.ply"),
            "evidence_ply": str(output_dir / "latent_grid_colored_by_evidence_score.ply"),
            "boundary_ply": str(output_dir / "latent_grid_boundary_like_points.ply"),
        },
    }
    write_json(output_dir / "b_fus3d_latent_grid_evidence_summary.json", summary)
    write_report(output_dir / "b_fus3d_latent_grid_evidence_summary.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready(summary["decision"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
