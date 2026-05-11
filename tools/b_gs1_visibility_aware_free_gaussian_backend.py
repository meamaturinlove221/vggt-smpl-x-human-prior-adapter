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

import b_gs0_smplx_anchored_free_gaussian_smoke as bgs0  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend")
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_gs1_visibility_aware_gaussian_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "backend_smoke": True,
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
FAMILY_LIMITS = {
    "hairline_free": 350,
    "head_free": 160,
    "left_hand_free": 450,
    "right_hand_free": 450,
    "clothing_free": 900,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-GS1 visibility-aware anchored/free Gaussian backend smoke. It scores "
            "free Gaussian families with multi-view visibility, mask support, "
            "depth-order proxy, and anti-overfill, then writes new 3D artifacts and "
            "render comparisons. This is research-only and never writes pass/export state."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=bgs0.DEFAULT_SCENE_DIR)
    parser.add_argument("--template-payload", type=Path, default=bgs0.DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="all")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--constrained-stride", type=int, default=4)
    parser.add_argument("--free-per-anchor", type=int, default=2)
    parser.add_argument("--raw-free-cap", type=int, default=16000)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-selected-free", type=int, default=2300)
    parser.add_argument("--outside-penalty", type=float, default=2.0)
    parser.add_argument("--overfill-budget", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return bgs0.json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    bgs0.write_json(path, payload)


def family_cap(name: str) -> int:
    return int(FAMILY_LIMITS.get(str(name), 300))


def score_free_points(
    free: dict[str, np.ndarray],
    constrained: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    *,
    target_size: int,
    min_support: int,
    outside_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(free["points"], dtype=np.float32)
    constrained_points = np.asarray(constrained["points"], dtype=np.float32)
    n = points.shape[0]
    support = np.zeros((n,), dtype=np.int32)
    outside = np.zeros((n,), dtype=np.int32)
    visible = np.zeros((n,), dtype=np.int32)
    nearest_depth_margin = np.full((n,), np.inf, dtype=np.float32)
    for view in views:
        camera_id = bgs0.normalize_camera_id(view["camera_id"])
        camera = cameras[camera_id]
        intrinsic = bgs0.align_intrinsics_for_loaded_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size=target_size)
        uv, depth = bgs0.project_points(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        base_uv, base_depth = bgs0.project_points(constrained_points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        xi = np.rint(uv[:, 0]).astype(np.int64)
        yi = np.rint(uv[:, 1]).astype(np.int64)
        inside = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 1e-6)
            & (xi >= 0)
            & (xi < target_size)
            & (yi >= 0)
            & (yi < target_size)
        )
        visible += inside.astype(np.int32)
        target = np.asarray(view["mask"], dtype=bool)
        hit = np.zeros((n,), dtype=bool)
        if np.any(inside):
            hit[inside] = target[yi[inside], xi[inside]]
        support += hit.astype(np.int32)
        outside += (inside & ~hit).astype(np.int32)

        base_inside = np.isfinite(base_uv).all(axis=1) & np.isfinite(base_depth) & (base_depth > 1e-6)
        if np.any(inside) and np.any(base_inside):
            # Small, cheap depth-order proxy: compare with the nearest projected
            # constrained point in pixel space for this view.
            base_xy = np.rint(base_uv[base_inside]).astype(np.float32)
            base_z = base_depth[base_inside].astype(np.float32)
            idxs = np.flatnonzero(inside)
            query_xy = np.stack([xi[inside], yi[inside]], axis=1).astype(np.float32)
            chunk = 512
            for start in range(0, query_xy.shape[0], chunk):
                stop = min(start + chunk, query_xy.shape[0])
                d2 = ((query_xy[start:stop, None, :] - base_xy[None, :, :]) ** 2).sum(axis=2)
                nearest = np.argmin(d2, axis=1)
                margin = depth[idxs[start:stop]] - base_z[nearest]
                nearest_depth_margin[idxs[start:stop]] = np.minimum(nearest_depth_margin[idxs[start:stop]], margin.astype(np.float32))

    family = np.asarray(free["family"]).astype(str)
    anchor = np.asarray(free["anchor_index"], dtype=np.int64)
    distance = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
    # Prefer points that are visible, inside masks, and close to the surface
    # rather than raw offset clouds. Penalize outside-mask and back-floating.
    finite_margin = np.where(np.isfinite(nearest_depth_margin), nearest_depth_margin, 0.0)
    depth_penalty = np.maximum(finite_margin, 0.0)
    anchor_penalty = np.zeros_like(distance)
    if anchor.size == points.shape[0]:
        # Free points are already generated around anchors; this term suppresses
        # far excursions without requiring template vertices here.
        anchor_penalty = np.clip(distance / max(float(np.quantile(distance, 0.95)), 1e-6), 0.0, 2.0)
    score = support.astype(np.float32) - float(outside_penalty) * outside.astype(np.float32)
    score += 0.15 * visible.astype(np.float32)
    score -= 0.30 * depth_penalty.astype(np.float32)
    score -= 0.05 * anchor_penalty.astype(np.float32)
    eligible = support >= int(min_support)

    selected: list[int] = []
    family_rows: dict[str, Any] = {}
    for name in sorted(set(family.tolist())):
        mask = family == name
        candidates = np.flatnonzero(mask & eligible)
        order = candidates[np.argsort(score[candidates])[::-1]] if candidates.size else np.asarray([], dtype=np.int64)
        cap = family_cap(name)
        keep = order[:cap]
        selected.extend(keep.tolist())
        family_rows[name] = {
            "raw_count": int(mask.sum()),
            "eligible_count": int(candidates.size),
            "selected_count": int(keep.size),
            "support_mean_selected": float(support[keep].mean()) if keep.size else 0.0,
            "outside_mean_selected": float(outside[keep].mean()) if keep.size else 0.0,
            "score_mean_selected": float(score[keep].mean()) if keep.size else None,
        }
    selected_idx = np.asarray(sorted(set(selected)), dtype=np.int64)
    diagnostics = {
        "support": support,
        "outside": outside,
        "visible": visible,
        "score": score,
        "depth_margin": nearest_depth_margin,
        "selected_idx": selected_idx,
        "family_rows": family_rows,
    }
    return selected_idx, diagnostics


def take_gaussians(gaussians: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    out = {key: value[idx] for key, value in gaussians.items()}
    return out


def build_raw_free(template: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    chunks = [
        bgs0.make_free_family(
            template,
            "hairline_vertex_mask",
            "hairline_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.012, 0.022),
            tangent_offsets=(-0.004, 0.004),
            max_count=max(64, args.raw_free_cap // 6),
        ),
        bgs0.make_free_family(
            template,
            "head_vertex_mask",
            "head_free",
            stride=max(2, args.constrained_stride * 5),
            per_anchor=1,
            normal_offsets=(0.014,),
            tangent_offsets=(0.0,),
            max_count=max(64, args.raw_free_cap // 10),
        ),
        bgs0.make_free_family(
            template,
            "left_hand_vertex_mask",
            "left_hand_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.010, 0.018),
            tangent_offsets=(-0.003, 0.003),
            max_count=max(64, args.raw_free_cap // 8),
        ),
        bgs0.make_free_family(
            template,
            "right_hand_vertex_mask",
            "right_hand_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.010, 0.018),
            tangent_offsets=(-0.003, 0.003),
            max_count=max(64, args.raw_free_cap // 8),
        ),
        bgs0.make_free_family(
            template,
            "lower_clothing_vertex_mask",
            "clothing_free",
            stride=max(1, args.constrained_stride),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.010, 0.020),
            tangent_offsets=(-0.005, 0.005),
            max_count=max(64, args.raw_free_cap // 3),
        ),
    ]
    return bgs0.merge_gaussians(chunks, max_points=args.raw_free_cap)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-GS1 Visibility-Aware Free Gaussian Backend",
        "",
        "Status: `research_only_visibility_backend_no_export`",
        "",
        "This is a real backend smoke that writes new 3D Gaussian artifacts and",
        "compares constrained, raw-free, visibility-aware selected, and random-free",
        "controls. It is not a teacher, candidate, strict pass, or cloud unblock.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['strict_facts']['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(summary["comparison"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Family Selection",
        "",
        "```json",
        json.dumps(summary["family_selection"], indent=2, ensure_ascii=False, sort_keys=True),
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


def main() -> None:
    args = parse_args()
    bgs0.ensure_safe_path(args.output_dir)
    bgs0.ensure_safe_path(args.status_report)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = bgs0.load_template(args.template_payload)
    views, cameras, camera_source = bgs0.load_views(
        args.scene_dir,
        args.dataset_root,
        args.subset_name,
        args.view_indices,
        args.target_size,
    )
    constrained = bgs0.make_constrained_gaussians(template, stride=args.constrained_stride)
    raw_free = build_raw_free(template, args)
    selected_idx, diagnostics = score_free_points(
        raw_free,
        constrained,
        views,
        cameras,
        target_size=args.target_size,
        min_support=args.min_support,
        outside_penalty=args.outside_penalty,
    )
    selected_idx = selected_idx[: int(args.max_selected_free)]
    selected_free = take_gaussians(raw_free, selected_idx)
    rng = np.random.default_rng(13)
    random_idx = selected_idx.copy()
    if random_idx.size:
        eligible_count = min(raw_free["points"].shape[0], random_idx.size)
        random_idx = rng.choice(raw_free["points"].shape[0], size=eligible_count, replace=False)
    random_free = take_gaussians(raw_free, random_idx)

    va_combined = bgs0.merge_gaussians([constrained, selected_free])
    raw_combined = bgs0.merge_gaussians([constrained, raw_free], max_points=constrained["points"].shape[0] + int(args.max_selected_free))
    random_combined = bgs0.merge_gaussians([constrained, random_free])

    bgs0.write_ply(args.output_dir / "b_gs1_constrained_baseline.ply", constrained)
    bgs0.write_ply(args.output_dir / "b_gs1_raw_free_candidates.ply", raw_free)
    bgs0.write_ply(args.output_dir / "b_gs1_visibility_selected_free.ply", selected_free)
    bgs0.write_ply(args.output_dir / "b_gs1_visibility_aware_combined.ply", va_combined)
    bgs0.write_ply(args.output_dir / "b_gs1_random_control_combined.ply", random_combined)
    np.savez_compressed(
        args.output_dir / "b_gs1_selection_diagnostics.npz",
        selected_idx=selected_idx,
        support=diagnostics["support"],
        outside=diagnostics["outside"],
        visible=diagnostics["visible"],
        score=diagnostics["score"],
        depth_margin=diagnostics["depth_margin"],
    )

    metrics = {
        "constrained": bgs0.render_gaussian_set("constrained", constrained, views, cameras, target_size=args.target_size, point_radius=args.point_radius, output_dir=args.output_dir),
        "visibility_aware": bgs0.render_gaussian_set("visibility_aware", va_combined, views, cameras, target_size=args.target_size, point_radius=args.point_radius, output_dir=args.output_dir),
        "raw_free_capped": bgs0.render_gaussian_set("raw_free_capped", raw_combined, views, cameras, target_size=args.target_size, point_radius=args.point_radius, output_dir=args.output_dir),
        "random_control": bgs0.render_gaussian_set("random_control", random_combined, views, cameras, target_size=args.target_size, point_radius=args.point_radius, output_dir=args.output_dir),
    }
    comparison = {
        "visibility_minus_constrained_iou": float(metrics["visibility_aware"]["mean_iou"] - metrics["constrained"]["mean_iou"]),
        "visibility_minus_constrained_overfill": float(metrics["visibility_aware"]["mean_overfill_ratio"] - metrics["constrained"]["mean_overfill_ratio"]),
        "visibility_minus_constrained_recall": float(metrics["visibility_aware"]["mean_target_recall"] - metrics["constrained"]["mean_target_recall"]),
        "visibility_rgb_better_than_constrained": bool(metrics["visibility_aware"]["mean_rgb_residual"] < metrics["constrained"]["mean_rgb_residual"]),
        "visibility_minus_random_iou": float(metrics["visibility_aware"]["mean_iou"] - metrics["random_control"]["mean_iou"]),
        "visibility_minus_raw_capped_iou": float(metrics["visibility_aware"]["mean_iou"] - metrics["raw_free_capped"]["mean_iou"]),
        "metrics": metrics,
    }
    success = (
        comparison["visibility_minus_constrained_iou"] >= 0.0
        and comparison["visibility_minus_constrained_overfill"] <= float(args.overfill_budget)
        and comparison["visibility_rgb_better_than_constrained"]
        and comparison["visibility_minus_random_iou"] > 0.0
    )
    decision = (
        "B-GS1 visibility-aware scoring meets the bounded local criteria, but remains research-only pending Open3D/strict review."
        if success
        else "B-GS1 produced new visibility-aware Gaussian artifacts, but did not satisfy IoU/overfill/control criteria; keep direction active but freeze this scoring recipe."
    )
    key_outputs = [
        str((args.output_dir / "b_gs1_constrained_baseline.ply").resolve()),
        str((args.output_dir / "b_gs1_raw_free_candidates.ply").resolve()),
        str((args.output_dir / "b_gs1_visibility_selected_free.ply").resolve()),
        str((args.output_dir / "b_gs1_visibility_aware_combined.ply").resolve()),
        str((args.output_dir / "b_gs1_random_control_combined.ply").resolve()),
        str((args.output_dir / "b_gs1_selection_diagnostics.npz").resolve()),
    ]
    summary = {
        "status": "research_only_visibility_backend_no_export",
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "view_indices": [int(view["view_index"]) for view in views],
            "target_size": int(args.target_size),
        },
        "parameters": {
            "min_support": int(args.min_support),
            "outside_penalty": float(args.outside_penalty),
            "max_selected_free": int(args.max_selected_free),
            "point_radius": int(args.point_radius),
        },
        "family_selection": diagnostics["family_rows"],
        "selected_free_count": int(selected_free["points"].shape[0]),
        "comparison": comparison,
        "success_local_bounded": bool(success),
        "decision": decision,
        "key_outputs": key_outputs,
    }
    write_json(args.output_dir / "b_gs1_summary.json", summary)
    write_json(args.output_dir / "b_gs1_render_comparison.json", comparison)
    write_report(args.output_dir / "b_gs1_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "success": success, "decision": decision}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
