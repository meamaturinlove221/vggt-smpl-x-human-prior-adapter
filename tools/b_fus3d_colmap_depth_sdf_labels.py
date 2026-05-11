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

from build_colmap_depth_teacher_targets import read_colmap_depth  # noqa: E402
from b_fus3d_query_evidence_cache import project_queries  # noqa: E402


DEFAULT_QUERY_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D7_query_sdf_smoke_hybrid6_layer23/"
    "b_fus3d_query_sdf_smoke_arrays.npz"
)
DEFAULT_A5_WORKSPACE = Path(
    "output/surface_research_preflight/"
    "A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/"
    "A5_known_camera_colmap_workspace/workspace"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D9_colmap_depth_sdf_labels_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_colmap_depth_sdf_labels_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "colmap_depth_label_precheck_only": True,
    "not_colmap_teacher": True,
    "not_sdf_decoder": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_state_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D9 COLMAP-depth weak SDF label precheck. "
            "It projects B-Fus3D query/shell samples into existing A5 known-direct "
            "PatchMatch depth maps to test whether per-view dense depths provide "
            "stronger front/on-surface/behind supervision than visual-hull masks. "
            "It never exports a teacher/candidate or writes strict pass state."
        )
    )
    parser.add_argument("--query-arrays", type=Path, default=DEFAULT_QUERY_ARRAYS)
    parser.add_argument("--a5-workspace", type=Path, default=DEFAULT_A5_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--near-threshold", type=float, default=0.035)
    parser.add_argument("--front-threshold", type=float, default=-0.060)
    parser.add_argument("--behind-threshold", type=float, default=0.060)
    parser.add_argument("--min-valid-views", type=int, default=2)
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
    return value


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return {"count": int(arr.size), "finite": int(finite.sum())}
    vals = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(finite.sum()),
        "min": float(np.min(vals)),
        "p05": float(np.percentile(vals, 5)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def load_query_arrays(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        required = {
            "query_positions",
            "query_families",
            "sample_positions",
            "sample_query_indices",
            "template_signed_distance",
            "selected_view_indices",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"Missing B-Fus3D7 arrays: {missing}")
        return {
            "query_positions": np.asarray(payload["query_positions"], dtype=np.float32),
            "query_families": np.asarray(payload["query_families"]).astype(str),
            "sample_positions": np.asarray(payload["sample_positions"], dtype=np.float32),
            "sample_query_indices": np.asarray(payload["sample_query_indices"], dtype=np.int64),
            "template_signed_distance": np.asarray(payload["template_signed_distance"], dtype=np.float32),
            "selected_view_indices": np.asarray(payload["selected_view_indices"], dtype=np.int32),
        }


def load_a5_views(workspace: Path) -> list[dict[str, Any]]:
    known_path = workspace / "known_camera_model.json"
    if not known_path.is_file():
        raise FileNotFoundError(known_path)
    known = json.loads(known_path.read_text(encoding="utf-8"))
    views = []
    depth_dir = workspace / "dense" / "stereo" / "depth_maps"
    for view in known.get("views", []):
        image_name = str(view["image_name"])
        depth_path = depth_dir / f"{image_name}.photometric.bin"
        if not depth_path.is_file():
            continue
        depth = read_colmap_depth(depth_path)
        views.append(
            {
                "view_index": int(view.get("view_index", len(views))),
                "camera_id": str(view.get("scene_camera_id", "")),
                "image_name": image_name,
                "intrinsic": np.asarray(view["intrinsic_3x3"], dtype=np.float32),
                "world_to_cam": np.asarray(view["world_to_cam_4x4"], dtype=np.float32),
                "depth": depth.astype(np.float32),
                "depth_path": str(depth_path),
            }
        )
    if not views:
        raise FileNotFoundError(f"No COLMAP photometric depth maps found in {depth_dir}")
    return views


def label_points(points: np.ndarray, views: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    n_points = int(points.shape[0])
    n_views = len(views)
    valid = np.zeros((n_points, n_views), dtype=bool)
    residual = np.full((n_points, n_views), np.nan, dtype=np.float32)
    colmap_depth = np.full((n_points, n_views), np.nan, dtype=np.float32)
    query_depth = np.full((n_points, n_views), np.nan, dtype=np.float32)
    uv = np.full((n_points, n_views, 2), np.nan, dtype=np.float32)
    for view_pos, view in enumerate(views):
        depth = np.asarray(view["depth"], dtype=np.float32)
        h, w = depth.shape
        uv_view, z_view, _cam = project_queries(points, view["world_to_cam"], view["intrinsic"])
        u = uv_view[:, 0]
        v = uv_view[:, 1]
        inside = (
            np.isfinite(uv_view).all(axis=1)
            & np.isfinite(z_view)
            & (z_view > 1e-6)
            & (u >= 0.0)
            & (u < w)
            & (v >= 0.0)
            & (v < h)
        )
        xi = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
        yi = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
        sampled_depth = depth[yi, xi]
        ok_depth = inside & np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
        valid[:, view_pos] = ok_depth
        colmap_depth[:, view_pos] = sampled_depth.astype(np.float32)
        query_depth[:, view_pos] = z_view.astype(np.float32)
        residual[:, view_pos] = (z_view - sampled_depth).astype(np.float32)
        uv[:, view_pos, :] = uv_view.astype(np.float32)
    valid_count = valid.sum(axis=1).astype(np.int32)
    near = valid & (np.abs(residual) <= float(args.near_threshold))
    front = valid & (residual <= float(args.front_threshold))
    behind = valid & (residual >= float(args.behind_threshold))
    near_count = near.sum(axis=1).astype(np.int32)
    front_count = front.sum(axis=1).astype(np.int32)
    behind_count = behind.sum(axis=1).astype(np.int32)
    enough = valid_count >= int(args.min_valid_views)
    surface_like = enough & (near_count >= int(args.min_valid_views))
    front_like = enough & (front_count >= int(args.min_valid_views)) & ~surface_like
    behind_like = enough & (behind_count >= int(args.min_valid_views)) & ~surface_like
    ambiguous = enough & ~(surface_like | front_like | behind_like)
    unknown = ~enough
    label = np.full((n_points,), -1, dtype=np.int8)
    label[front_like] = 0
    label[surface_like] = 1
    label[behind_like] = 2
    return {
        "valid": valid,
        "uv": uv,
        "residual": residual,
        "query_depth": query_depth,
        "colmap_depth": colmap_depth,
        "valid_count": valid_count,
        "near_count": near_count,
        "front_count": front_count,
        "behind_count": behind_count,
        "surface_like": surface_like,
        "front_like": front_like,
        "behind_like": behind_like,
        "ambiguous": ambiguous,
        "unknown": unknown,
        "label": label,
    }


def compact_label_summary(labels: dict[str, np.ndarray]) -> dict[str, Any]:
    valid_resid = labels["residual"][labels["valid"]]
    return {
        "count": int(labels["label"].shape[0]),
        "valid_ge_min": int((~labels["unknown"]).sum()),
        "surface_like": int(labels["surface_like"].sum()),
        "front_like": int(labels["front_like"].sum()),
        "behind_like": int(labels["behind_like"].sum()),
        "ambiguous": int(labels["ambiguous"].sum()),
        "unknown": int(labels["unknown"].sum()),
        "valid_count": scalar_stats(labels["valid_count"]),
        "near_count": scalar_stats(labels["near_count"]),
        "residual_on_valid": scalar_stats(valid_resid),
    }


def family_stats(
    query_families: np.ndarray,
    query_labels: dict[str, np.ndarray],
    sample_labels: dict[str, np.ndarray],
    sample_query_indices: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sample_families = query_families[np.asarray(sample_query_indices, dtype=np.int64)]
    for family in sorted(set(str(v) for v in query_families.tolist())):
        qmask = query_families == family
        smask = sample_families == family
        out[family] = {
            "query_count": int(qmask.sum()),
            "query_valid_ge_min": int((~query_labels["unknown"][qmask]).sum()),
            "query_surface_like": int(query_labels["surface_like"][qmask].sum()),
            "query_front_like": int(query_labels["front_like"][qmask].sum()),
            "query_behind_like": int(query_labels["behind_like"][qmask].sum()),
            "query_ambiguous": int(query_labels["ambiguous"][qmask].sum()),
            "query_unknown": int(query_labels["unknown"][qmask].sum()),
            "query_median_valid_residual": float(np.nanmedian(query_labels["residual"][qmask & (query_labels["valid"].any(axis=1))]))
            if np.any(qmask & (query_labels["valid"].any(axis=1)))
            else None,
            "sample_count": int(smask.sum()),
            "sample_valid_ge_min": int((~sample_labels["unknown"][smask]).sum()) if smask.any() else 0,
            "sample_surface_like": int(sample_labels["surface_like"][smask].sum()) if smask.any() else 0,
            "sample_front_like": int(sample_labels["front_like"][smask].sum()) if smask.any() else 0,
            "sample_behind_like": int(sample_labels["behind_like"][smask].sum()) if smask.any() else 0,
            "sample_ambiguous": int(sample_labels["ambiguous"][smask].sum()) if smask.any() else 0,
            "sample_unknown": int(sample_labels["unknown"][smask].sum()) if smask.any() else 0,
        }
    return out


def shell_signal(query_count: int, sample_query_indices: np.ndarray, sample_labels: dict[str, np.ndarray]) -> dict[str, Any]:
    informative = 0
    surface_hits = 0
    front_and_behind = 0
    for query_idx in range(query_count):
        member = np.asarray(sample_query_indices) == query_idx
        if not member.any():
            continue
        has_surface = bool(sample_labels["surface_like"][member].any())
        has_front = bool(sample_labels["front_like"][member].any())
        has_behind = bool(sample_labels["behind_like"][member].any())
        if has_surface:
            surface_hits += 1
        if has_front and has_behind:
            front_and_behind += 1
        if has_surface or (has_front and has_behind):
            informative += 1
    return {
        "query_triplets_with_any_surface_like_shell": int(surface_hits),
        "query_triplets_with_front_and_behind_shells": int(front_and_behind),
        "query_triplets_with_depth_boundary_signal": int(informative),
        "query_triplets_with_depth_boundary_signal_ratio": float(informative / max(query_count, 1)),
    }


def color_by_label(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels < 0] = np.asarray([145, 145, 145], dtype=np.uint8)
    colors[labels == 0] = np.asarray([65, 125, 230], dtype=np.uint8)
    colors[labels == 1] = np.asarray([40, 190, 95], dtype=np.uint8)
    colors[labels == 2] = np.asarray([230, 75, 55], dtype=np.uint8)
    return colors


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray, labels: dict[str, np.ndarray]) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int valid_count\nproperty int near_count\nproperty int label\n")
        handle.write("end_header\n")
        for idx, point in enumerate(points):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(colors[idx,0])} {int(colors[idx,1])} {int(colors[idx,2])} "
                f"{int(labels['valid_count'][idx])} {int(labels['near_count'][idx])} {int(labels['label'][idx])}\n"
            )


def write_markdown(path: Path, summary: dict[str, Any], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        "This is a research-only COLMAP-depth weak SDF label precheck. It is not",
        "a teacher, not a candidate, not a decoder success, and not a strict gate result.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Label Summary",
        "",
        "```json",
        json.dumps(summary["label_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Summary",
        "",
        "```json",
        json.dumps(summary["family_stats"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Shell Signal",
        "",
        "```json",
        json.dumps(summary["shell_signal"], indent=2, ensure_ascii=False),
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
    ]
    for value in summary["outputs"].values():
        lines.append(f"- `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_query_arrays(args.query_arrays)
    views = load_a5_views(args.a5_workspace.expanduser().resolve())
    query_labels = label_points(arrays["query_positions"], views, args)
    sample_labels = label_points(arrays["sample_positions"], views, args)
    fam_stats = family_stats(
        arrays["query_families"],
        query_labels,
        sample_labels,
        arrays["sample_query_indices"],
    )
    shell = shell_signal(
        int(arrays["query_positions"].shape[0]),
        arrays["sample_query_indices"],
        sample_labels,
    )
    has_signal = shell["query_triplets_with_depth_boundary_signal_ratio"] >= 0.10
    enough_surface_queries = compact_label_summary(query_labels)["surface_like"] >= 50
    if has_signal and enough_surface_queries:
        decision = (
            "A5 per-view COLMAP depths provide some query-shell depth boundary signal. "
            "They may be used only as a weak diagnostic term for a later B-Fus3D decoder; "
            "they remain teacher-negative because A5 fused/shared strict gate already fails."
        )
    else:
        decision = (
            "A5 per-view COLMAP depths do not provide enough stable query-shell SDF signal "
            "for B-Fus3D supervision under this setup. This closes the idea that failed "
            "A5 fusion can be rescued simply by using the per-view PatchMatch depths as "
            "B-Fus3D SDF labels."
        )

    arrays_path = output_dir / "b_fus3d_colmap_depth_sdf_labels_arrays.npz"
    np.savez_compressed(
        arrays_path,
        query_label=query_labels["label"],
        query_valid_count=query_labels["valid_count"],
        query_near_count=query_labels["near_count"],
        query_residual=query_labels["residual"].astype(np.float32),
        sample_label=sample_labels["label"],
        sample_valid_count=sample_labels["valid_count"],
        sample_near_count=sample_labels["near_count"],
        sample_residual=sample_labels["residual"].astype(np.float32),
        sample_query_indices=arrays["sample_query_indices"],
        template_signed_distance=arrays["template_signed_distance"],
    )
    query_ply = output_dir / "query_points_colored_by_colmap_depth_label.ply"
    sample_ply = output_dir / "normal_shell_samples_colored_by_colmap_depth_label.ply"
    write_point_ply(query_ply, arrays["query_positions"], color_by_label(query_labels["label"]), query_labels)
    write_point_ply(sample_ply, arrays["sample_positions"], color_by_label(sample_labels["label"]), sample_labels)

    summary_json = output_dir / "b_fus3d_colmap_depth_sdf_labels_summary.json"
    summary_md = output_dir / "b_fus3d_colmap_depth_sdf_labels_summary.md"
    status_report = args.status_report.expanduser().resolve()
    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d9_colmap_depth_sdf_label_precheck",
        "truthful_status": "research_only_colmap_depth_weak_label_precheck_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "query_arrays": str(args.query_arrays.expanduser().resolve()),
            "a5_workspace": str(args.a5_workspace.expanduser().resolve()),
            "a5_depth_view_count": len(views),
        },
        "configuration": {
            "near_threshold": float(args.near_threshold),
            "front_threshold": float(args.front_threshold),
            "behind_threshold": float(args.behind_threshold),
            "min_valid_views": int(args.min_valid_views),
        },
        "label_summary": {
            "query_points": compact_label_summary(query_labels),
            "normal_shell_samples": compact_label_summary(sample_labels),
        },
        "family_stats": fam_stats,
        "shell_signal": shell,
        "supervision_readout": {
            "has_depth_boundary_signal": bool(has_signal),
            "enough_surface_like_queries": bool(enough_surface_queries),
            "is_strict_teacher": False,
            "is_candidate": False,
            "insufficient_for_pass": True,
            "uses_failed_a5_colmap_depth_as_weak_diagnostic_only": True,
        },
        "decision": decision,
        "outputs": {
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "status_report": str(status_report),
            "query_ply": str(query_ply),
            "sample_ply": str(sample_ply),
        },
    }
    summary = json_ready(summary)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary_md, summary, "B-Fus3D9 COLMAP-Depth SDF Label Precheck")
    write_markdown(status_report, summary, "B-Fus3D9 COLMAP-Depth SDF Label Status")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
