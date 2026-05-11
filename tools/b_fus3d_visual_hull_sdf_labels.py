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

from b_fus3d_query_evidence_cache import load_scene_views, project_queries  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_QUERY_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D7_query_sdf_smoke_hybrid6_layer23/"
    "b_fus3d_query_sdf_smoke_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D8_visual_hull_sdf_labels_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_visual_hull_sdf_labels_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "visual_hull_label_precheck_only": True,
    "not_visual_hull_candidate": True,
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
    "writes_prediction_arrays": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D8 visual-hull weak SDF/occupancy label precheck. "
            "It projects B-Fus3D7 query and normal-shell samples into raw masks using "
            "known cameras, then reports whether visual-hull labels are informative "
            "enough to supervise a later query-level SDF/occupancy decoder. It never "
            "exports a teacher/candidate or writes strict pass state."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--query-arrays", type=Path, default=DEFAULT_QUERY_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-visible-views", type=int, default=2)
    parser.add_argument("--inside-ratio-threshold", type=float, default=0.98)
    parser.add_argument("--outside-ratio-threshold", type=float, default=0.95)
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
            "query_normals",
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
            "query_normals": np.asarray(payload["query_normals"], dtype=np.float32),
            "query_families": np.asarray(payload["query_families"]).astype(str),
            "sample_positions": np.asarray(payload["sample_positions"], dtype=np.float32),
            "sample_query_indices": np.asarray(payload["sample_query_indices"], dtype=np.int64),
            "template_signed_distance": np.asarray(payload["template_signed_distance"], dtype=np.float32),
            "selected_view_indices": np.asarray(payload["selected_view_indices"], dtype=np.int32),
        }


def label_points(
    points: np.ndarray,
    scene_rows: list[dict[str, Any]],
    target_size: int,
    min_visible_views: int,
    inside_ratio_threshold: float,
    outside_ratio_threshold: float,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    n_points = int(points.shape[0])
    visible = np.zeros((n_points, len(scene_rows)), dtype=bool)
    inside = np.zeros((n_points, len(scene_rows)), dtype=bool)
    depth = np.full((n_points, len(scene_rows)), np.nan, dtype=np.float32)
    uv = np.full((n_points, len(scene_rows), 2), np.nan, dtype=np.float32)
    for view_pos, row in enumerate(scene_rows):
        uv_view, z_view, _cam = project_queries(points, row["world_to_cam"], row["intrinsic"])
        u = uv_view[:, 0]
        v = uv_view[:, 1]
        valid = (
            np.isfinite(uv_view).all(axis=1)
            & np.isfinite(z_view)
            & (z_view > 1e-6)
            & (u >= 0.0)
            & (u < target_size)
            & (v >= 0.0)
            & (v < target_size)
        )
        xi = np.clip(np.floor(u).astype(np.int64), 0, target_size - 1)
        yi = np.clip(np.floor(v).astype(np.int64), 0, target_size - 1)
        mask = row["mask"].astype(bool)
        visible[:, view_pos] = valid
        inside[:, view_pos] = valid & mask[yi, xi]
        depth[:, view_pos] = z_view.astype(np.float32)
        uv[:, view_pos] = uv_view.astype(np.float32)
    visible_count = visible.sum(axis=1).astype(np.int32)
    inside_count = inside.sum(axis=1).astype(np.int32)
    outside_count = (visible & ~inside).sum(axis=1).astype(np.int32)
    denom = np.maximum(visible_count, 1).astype(np.float32)
    inside_ratio = inside_count.astype(np.float32) / denom
    outside_ratio = outside_count.astype(np.float32) / denom
    enough = visible_count >= int(min_visible_views)
    vh_inside = enough & (inside_ratio >= float(inside_ratio_threshold))
    vh_outside = enough & (outside_ratio >= float(outside_ratio_threshold))
    ambiguous = enough & ~(vh_inside | vh_outside)
    unknown = ~enough
    label = np.full((n_points,), -1, dtype=np.int8)
    label[vh_outside] = 0
    label[vh_inside] = 1
    return {
        "visible": visible,
        "inside": inside,
        "uv": uv,
        "depth": depth,
        "visible_count": visible_count,
        "inside_count": inside_count,
        "outside_count": outside_count,
        "inside_ratio": inside_ratio.astype(np.float32),
        "outside_ratio": outside_ratio.astype(np.float32),
        "vh_inside": vh_inside,
        "vh_outside": vh_outside,
        "ambiguous": ambiguous,
        "unknown": unknown,
        "label": label,
    }


def family_stats(
    families: np.ndarray,
    query_labels: dict[str, np.ndarray],
    sample_labels: dict[str, np.ndarray],
    sample_query_indices: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sample_families = families[np.asarray(sample_query_indices, dtype=np.int64)]
    for family in sorted(set(str(v) for v in families.tolist())):
        qmask = families == family
        smask = sample_families == family
        q_total = int(qmask.sum())
        s_total = int(smask.sum())
        if q_total == 0:
            continue
        out[family] = {
            "query_count": q_total,
            "query_visible_ge_min": int((~query_labels["unknown"][qmask]).sum()),
            "query_vh_inside": int(query_labels["vh_inside"][qmask].sum()),
            "query_vh_outside": int(query_labels["vh_outside"][qmask].sum()),
            "query_ambiguous": int(query_labels["ambiguous"][qmask].sum()),
            "query_unknown": int(query_labels["unknown"][qmask].sum()),
            "query_mean_inside_ratio": float(np.mean(query_labels["inside_ratio"][qmask])),
            "sample_count": s_total,
            "sample_visible_ge_min": int((~sample_labels["unknown"][smask]).sum()) if s_total else 0,
            "sample_vh_inside": int(sample_labels["vh_inside"][smask].sum()) if s_total else 0,
            "sample_vh_outside": int(sample_labels["vh_outside"][smask].sum()) if s_total else 0,
            "sample_ambiguous": int(sample_labels["ambiguous"][smask].sum()) if s_total else 0,
            "sample_unknown": int(sample_labels["unknown"][smask].sum()) if s_total else 0,
            "sample_mean_inside_ratio": float(np.mean(sample_labels["inside_ratio"][smask])) if s_total else 0.0,
        }
    return out


def shell_polarity_stats(
    query_count: int,
    sample_query_indices: np.ndarray,
    template_signed_distance: np.ndarray,
    sample_labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    signed = np.asarray(template_signed_distance, dtype=np.float32)
    sample_query_indices = np.asarray(sample_query_indices, dtype=np.int64)
    neg = signed < -1e-6
    pos = signed > 1e-6
    center = np.abs(signed) <= 1e-6
    rows = {}
    for name, mask in {"negative_shell": neg, "center": center, "positive_shell": pos}.items():
        rows[name] = {
            "sample_count": int(mask.sum()),
            "vh_inside": int(sample_labels["vh_inside"][mask].sum()),
            "vh_outside": int(sample_labels["vh_outside"][mask].sum()),
            "ambiguous": int(sample_labels["ambiguous"][mask].sum()),
            "unknown": int(sample_labels["unknown"][mask].sum()),
            "mean_inside_ratio": float(np.mean(sample_labels["inside_ratio"][mask])) if mask.any() else 0.0,
        }
    # Informative if, for a meaningful fraction of query triplets, one shell side is outside and the center/other side is not.
    informative = 0
    for query_idx in range(query_count):
        member = sample_query_indices == query_idx
        if not member.any():
            continue
        has_out = bool(sample_labels["vh_outside"][member].any())
        has_in_or_amb = bool((sample_labels["vh_inside"][member] | sample_labels["ambiguous"][member]).any())
        if has_out and has_in_or_amb:
            informative += 1
    rows["query_triplets_with_boundary_signal"] = int(informative)
    rows["query_triplets_with_boundary_signal_ratio"] = float(informative / max(query_count, 1))
    return rows


def color_by_label(labels: np.ndarray, inside_ratio: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    inside_ratio = np.asarray(inside_ratio, dtype=np.float32)
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels == 1] = np.asarray([40, 180, 80], dtype=np.uint8)
    colors[labels == 0] = np.asarray([220, 70, 50], dtype=np.uint8)
    colors[labels < 0] = np.asarray([145, 145, 145], dtype=np.uint8)
    amb = labels < 0
    # brighten ambiguous high-ratio points slightly.
    colors[amb, 2] = np.clip(colors[amb, 2].astype(np.float32) + inside_ratio[amb] * 80.0, 0, 255).astype(np.uint8)
    return colors


def write_point_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    labels: dict[str, np.ndarray],
    families: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    families = np.asarray(families).astype(str) if families is not None else np.asarray([""] * points.shape[0])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int visible_count\nproperty int inside_count\nproperty float inside_ratio\n")
        handle.write("property int visual_hull_label\n")
        handle.write("property uchar family_code\n")
        handle.write("end_header\n")
        family_codes = {name: idx for idx, name in enumerate(sorted(set(families.tolist())))}
        for idx in range(points.shape[0]):
            family_code = int(family_codes.get(str(families[idx]), 0))
            handle.write(
                f"{points[idx,0]:.7f} {points[idx,1]:.7f} {points[idx,2]:.7f} "
                f"{int(colors[idx,0])} {int(colors[idx,1])} {int(colors[idx,2])} "
                f"{int(labels['visible_count'][idx])} {int(labels['inside_count'][idx])} "
                f"{float(labels['inside_ratio'][idx]):.6f} {int(labels['label'][idx])} {family_code}\n"
            )


def write_markdown(path: Path, summary: dict[str, Any], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        "This is a research-only visual-hull weak-label precheck. It is not a",
        "visual-hull candidate, not a teacher, not a B-Fus3D decoder, and not a",
        "strict gate result.",
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
        "## Shell Polarity",
        "",
        "```json",
        json.dumps(summary["shell_polarity_stats"], indent=2, ensure_ascii=False),
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
        if isinstance(value, list):
            for item in value:
                lines.append(f"- `{item}`")
        else:
            lines.append(f"- `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_query_arrays(args.query_arrays)
    selected_view_indices = [int(v) for v in np.asarray(arrays["selected_view_indices"]).reshape(-1)]
    scene_rows, camera_source = load_scene_views(
        args.scene_dir.expanduser().resolve(),
        args.dataset_root,
        args.subset_name,
        selected_view_indices,
        int(args.target_size),
    )

    query_labels = label_points(
        arrays["query_positions"],
        scene_rows,
        int(args.target_size),
        int(args.min_visible_views),
        float(args.inside_ratio_threshold),
        float(args.outside_ratio_threshold),
    )
    sample_labels = label_points(
        arrays["sample_positions"],
        scene_rows,
        int(args.target_size),
        int(args.min_visible_views),
        float(args.inside_ratio_threshold),
        float(args.outside_ratio_threshold),
    )
    sample_families = arrays["query_families"][arrays["sample_query_indices"]]
    family_map = family_stats(arrays["query_families"], query_labels, sample_labels, arrays["sample_query_indices"])
    polarity = shell_polarity_stats(
        int(arrays["query_positions"].shape[0]),
        arrays["sample_query_indices"],
        arrays["template_signed_distance"],
        sample_labels,
    )

    def compact_label_summary(labels: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "count": int(labels["label"].shape[0]),
            "visible_ge_min": int((~labels["unknown"]).sum()),
            "vh_inside": int(labels["vh_inside"].sum()),
            "vh_outside": int(labels["vh_outside"].sum()),
            "ambiguous": int(labels["ambiguous"].sum()),
            "unknown": int(labels["unknown"].sum()),
            "inside_ratio": scalar_stats(labels["inside_ratio"]),
            "visible_count": scalar_stats(labels["visible_count"]),
        }

    label_summary = {
        "query_points": compact_label_summary(query_labels),
        "normal_shell_samples": compact_label_summary(sample_labels),
    }
    informative_ratio = float(polarity["query_triplets_with_boundary_signal_ratio"])
    has_some_signal = informative_ratio >= 0.05 and label_summary["normal_shell_samples"]["vh_outside"] > 0
    insufficient_for_teacher = True
    if has_some_signal:
        decision = (
            "Visual-hull labels provide some weak occupancy/SDF signal for query-shell samples, "
            "but this remains a mask-only constraint and cannot by itself create mentor-level "
            "face/hair/hand surface. It may be used as one weak supervision term for a later "
            "B-Fus3D query decoder, with full Open3D gates still required."
        )
    else:
        decision = (
            "Visual-hull labels are not informative enough under this query/shell setup to "
            "serve as the missing SDF supervision. A later B-Fus3D decoder still needs a "
            "stronger same-frame surface constraint or learned backend."
        )

    query_ply = output_dir / "query_points_colored_by_visual_hull_label.ply"
    sample_ply = output_dir / "normal_shell_samples_colored_by_visual_hull_label.ply"
    write_point_ply(
        query_ply,
        arrays["query_positions"],
        color_by_label(query_labels["label"], query_labels["inside_ratio"]),
        query_labels,
        arrays["query_families"],
    )
    write_point_ply(
        sample_ply,
        arrays["sample_positions"],
        color_by_label(sample_labels["label"], sample_labels["inside_ratio"]),
        sample_labels,
        sample_families,
    )

    arrays_path = output_dir / "b_fus3d_visual_hull_sdf_labels_arrays.npz"
    np.savez_compressed(
        arrays_path,
        query_label=query_labels["label"],
        query_visible_count=query_labels["visible_count"],
        query_inside_count=query_labels["inside_count"],
        query_inside_ratio=query_labels["inside_ratio"],
        sample_label=sample_labels["label"],
        sample_visible_count=sample_labels["visible_count"],
        sample_inside_count=sample_labels["inside_count"],
        sample_inside_ratio=sample_labels["inside_ratio"],
        sample_query_indices=arrays["sample_query_indices"],
        template_signed_distance=arrays["template_signed_distance"],
        selected_view_indices=np.asarray(selected_view_indices, dtype=np.int32),
    )
    summary_json = output_dir / "b_fus3d_visual_hull_sdf_labels_summary.json"
    summary_md = output_dir / "b_fus3d_visual_hull_sdf_labels_summary.md"
    status_report = args.status_report.expanduser().resolve()
    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d8_visual_hull_sdf_label_precheck",
        "truthful_status": "research_only_visual_hull_weak_label_precheck_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.expanduser().resolve()),
            "query_arrays": str(args.query_arrays.expanduser().resolve()),
            "selected_view_indices": selected_view_indices,
            "camera_source": str(camera_source),
        },
        "configuration": {
            "target_size": int(args.target_size),
            "min_visible_views": int(args.min_visible_views),
            "inside_ratio_threshold": float(args.inside_ratio_threshold),
            "outside_ratio_threshold": float(args.outside_ratio_threshold),
        },
        "label_summary": label_summary,
        "family_stats": family_map,
        "shell_polarity_stats": polarity,
        "supervision_readout": {
            "has_visual_hull_mask_label_signal": bool(has_some_signal),
            "query_triplet_boundary_signal_ratio": informative_ratio,
            "is_mask_only_constraint": True,
            "is_strict_teacher": False,
            "is_candidate": False,
            "insufficient_for_teacher_or_pass": insufficient_for_teacher,
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
    write_markdown(summary_md, summary, "B-Fus3D8 Visual-Hull SDF Label Precheck")
    write_markdown(status_report, summary, "B-Fus3D8 Visual-Hull SDF Label Status")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
