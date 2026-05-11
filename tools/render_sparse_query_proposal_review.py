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

from render_open3d_pointcloud import _save_open3d_renders, _save_projection_fallback  # noqa: E402


DEFAULT_POINTCLOUD = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D14_raw_image_offset_proposal_precheck_hybrid6_layer23/"
    "query_offset_proposal_points_colored_by_offset.ply"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D14_raw_image_offset_proposal_open3d_review"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_raw_image_offset_proposal_visual_status.md")

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
    "sparse_query_visual_review_only": True,
    "not_mesh": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a sparse B-Fus3D query proposal PLY for explicit visual review. "
            "This is not a mesh, teacher, candidate, or pass."
        )
    )
    parser.add_argument("--pointcloud", type=Path, default=DEFAULT_POINTCLOUD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--roi", choices=("full", "head", "face", "hands"), default="full")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
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


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY: {path}")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        points = []
        colors = []
        offsets = []
        selected = []
        for _ in range(vertex_count):
            parts = handle.readline().strip().split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            offsets.append(float(parts[6]) if len(parts) > 6 else 0.0)
            selected.append(int(parts[7]) if len(parts) > 7 else 1)
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(offsets, dtype=np.float32),
        np.asarray(selected, dtype=np.int32),
    )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Sparse Query Proposal Visual Review",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This renders sparse B-Fus3D query proposal points for visual inspection. It",
        "is not a mesh, teacher, candidate, or strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud train/infer/export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["pointcloud_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Screenshots",
        "",
    ]
    for item in summary.get("screenshots", []):
        lines.append(f"- `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors, offsets, selected = read_ply(args.pointcloud.expanduser().resolve())
    selected_mask = selected.astype(bool)
    summary = {
        "point_count": int(points.shape[0]),
        "selected_count": int(selected_mask.sum()),
        "selected_ratio": float(selected_mask.sum() / max(points.shape[0], 1)),
        "bbox_min": points.min(axis=0),
        "bbox_max": points.max(axis=0),
        "bbox_extent": np.ptp(points, axis=0),
        "negative_offset_count": int(np.sum(selected_mask & (offsets < -1e-8))),
        "positive_offset_count": int(np.sum(selected_mask & (offsets > 1e-8))),
    }
    screenshots = []
    try:
        screenshots = _save_open3d_renders(
            points=points,
            colors=colors,
            output_dir=output_dir,
            roi=args.roi,
            width=int(args.width),
            height=int(args.height),
            point_size=float(args.point_size),
            interactive=False,
        )
    except Exception as exc:
        summary["open3d_error"] = repr(exc)
    if not screenshots:
        screenshots = _save_projection_fallback(
            points=points,
            colors=colors,
            output_dir=output_dir,
            roi=args.roi,
            width=int(args.width),
            height=int(args.height),
        )
    payload = {
        "status": "research_only_sparse_query_visual_review_not_mesh_not_teacher_not_candidate",
        "contract": CONTRACT,
        "strict_gate_truth": STRICT_FACTS,
        "pointcloud": str(args.pointcloud.expanduser().resolve()),
        "output_dir": str(output_dir),
        "pointcloud_summary": summary,
        "screenshots": screenshots,
        "truthful_note": "Sparse query visual review only; no strict pass or cloud unblock.",
    }
    summary_path = output_dir / "sparse_query_proposal_visual_summary.json"
    md_path = output_dir / "sparse_query_proposal_visual_summary.md"
    summary_path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(payload))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(payload))
    print(json.dumps(json_ready({"summary": str(summary_path), "screenshots": screenshots}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
