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


DEFAULT_QUERY_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_support_open3d_hybrid6_layer23"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "open3d_precheck_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}

FAMILY_BASE_COLORS = {
    "full_body": np.asarray([170, 170, 170], dtype=np.float32),
    "face_core": np.asarray([245, 185, 40], dtype=np.float32),
    "hairline": np.asarray([155, 80, 225], dtype=np.float32),
    "left_hand": np.asarray([35, 145, 255], dtype=np.float32),
    "right_hand": np.asarray([255, 95, 55], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only Open3D visualization for B-Fus3D 3D query token support. "
            "It writes colored query point clouds and screenshots. It never exports "
            "a teacher/candidate, writes predictions, writes strict pass state, or calls cloud."
        )
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi", choices=("full", "head", "face", "hands"), default="full")
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
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


def write_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def color_by_family_support(families: np.ndarray, support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=np.float32)
    max_support = max(float(support.max()) if support.size else 0.0, 1.0)
    colors = np.zeros((support.shape[0], 3), dtype=np.float32)
    for idx, family in enumerate(families.astype(str).tolist()):
        base = FAMILY_BASE_COLORS.get(family, np.asarray([190, 190, 190], dtype=np.float32))
        intensity = 0.25 + 0.75 * float(support[idx]) / max_support
        colors[idx] = base * intensity
    zero = support <= 0
    colors[zero] = np.asarray([35, 35, 35], dtype=np.float32)
    return colors


def color_by_support_heat(support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=np.float32)
    max_support = max(float(support.max()) if support.size else 0.0, 1.0)
    t = np.clip(support / max_support, 0.0, 1.0)
    colors = np.zeros((support.shape[0], 3), dtype=np.float32)
    colors[:, 0] = (1.0 - t) * 255.0
    colors[:, 1] = t * 220.0
    colors[:, 2] = 60.0 + t * 90.0
    colors[support <= 0] = np.asarray([0, 0, 0], dtype=np.float32)
    return colors


def family_stats(families: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted(set(families.astype(str).tolist())):
        mask = families.astype(str) == family
        values = support[mask]
        out[family] = {
            "query_count": int(mask.sum()),
            "support_ge_1": int((values >= 1).sum()),
            "support_ge_2": int((values >= 2).sum()),
            "support_ge_3": int((values >= 3).sum()),
            "mean_support": float(values.mean()) if values.size else 0.0,
            "max_support": int(values.max()) if values.size else 0,
        }
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Query Support Open3D",
        "",
        "This is a research-only visualization of 3D query token support. It is",
        "not a decoder, not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Support",
        "",
        "```json",
        json.dumps(summary["family_support"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
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
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.query_cache.resolve(), allow_pickle=False) as payload:
        points = np.asarray(payload["query_positions"], dtype=np.float32)
        families = np.asarray(payload["query_families"])
        support = np.asarray(payload["support"], dtype=np.int32)

    family_colors = color_by_family_support(families, support)
    support_colors = color_by_support_heat(support)
    family_ply = output_dir / "query_support_family_colored.ply"
    heat_ply = output_dir / "query_support_heat_colored.ply"
    write_pointcloud_ply(family_ply, points, family_colors)
    write_pointcloud_ply(heat_ply, points, support_colors)

    renders: list[str] = []
    fallback: list[str] = []
    error = None
    try:
        renders = _save_open3d_renders(
            points,
            family_colors.astype(np.uint8),
            output_dir,
            args.roi,
            int(args.width),
            int(args.height),
            float(args.point_size),
            False,
        )
    except Exception as exc:  # pragma: no cover - Open3D display environment dependent.
        error = repr(exc)
        fallback = _save_projection_fallback(
            points,
            family_colors.astype(np.uint8),
            output_dir,
            args.roi,
            int(args.width),
            int(args.height),
        )

    stats = family_stats(families, support)
    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d_query_support_open3d",
        "truthful_status": "research_visualization_only_not_decoder_not_candidate_not_teacher",
        "contract": CONTRACT,
        "summary": {
            "query_cache": str(args.query_cache.resolve()),
            "query_count": int(points.shape[0]),
            "support_ge_1": int((support >= 1).sum()),
            "support_ge_2": int((support >= 2).sum()),
            "support_ge_3": int((support >= 3).sum()),
            "mean_support": float(support.mean()) if support.size else 0.0,
            "max_support": int(support.max()) if support.size else 0,
            "open3d_error": error,
        },
        "family_support": stats,
        "outputs": {
            "family_colored_ply": str(family_ply),
            "support_heat_ply": str(heat_ply),
            "open3d_renders": renders,
            "fallback_renders": fallback,
            "summary_json": str(output_dir / "b_fus3d_query_support_open3d_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_query_support_open3d_summary.md"),
        },
        "decision": (
            "The support visualization localizes B-Fus3D query evidence. Weak "
            "hairline/right-hand support must be treated as a decoder risk; this "
            "visualization is not a learned surface, visual pass, teacher, or candidate."
        ),
    }
    summary = json_ready(summary)
    (output_dir / "b_fus3d_query_support_open3d_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "b_fus3d_query_support_open3d_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
