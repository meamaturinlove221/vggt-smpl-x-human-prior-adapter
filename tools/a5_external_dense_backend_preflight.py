from __future__ import annotations

import argparse
import json
import shlex
import subprocess
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

from preflight_differentiable_renderer_backend import parse_view_indices  # noqa: E402
from prepare_4k4d_prior_training_case import load_scene_manifest, recover_legacy_crop_source_sizes  # noqa: E402
from render_research_ply_contact_sheet import mesh_components, read_ascii_ply  # noqa: E402
from research_scene_assets import localize_scene_manifest_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A5 research-only adapter contract for external same-frame dense backends. "
            "It exports/checks known-camera/mask workspace metadata and can import one shared mesh or a consistent "
            "depth set for review. It does not run formal VGGT train/infer/export and never writes strict passes."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["custom", "openmvs", "colmap_patchmatch"], default="custom")
    parser.add_argument("--backend-workspace", type=Path)
    parser.add_argument("--backend-command", default="", help="Optional command to run inside the backend workspace. Off by default.")
    parser.add_argument("--input-mesh", type=Path, help="Optional single shared mesh output from the backend.")
    parser.add_argument("--input-depth-dir", type=Path, help="Optional directory containing a consistent multi-view depth set.")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--eval-view-indices", default="")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A5 External Dense Backend Adapter Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only adapter contract. It does not claim backend success, does not export a strict teacher/candidate, and does not write pass state.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
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
    for item in summary["outputs"]:
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def depth_file_count(depth_dir: Path | None) -> int:
    if depth_dir is None:
        return 0
    if not depth_dir.is_dir():
        return 0
    suffixes = {".npy", ".npz", ".png", ".exr", ".tiff", ".tif"}
    return sum(1 for path in depth_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def maybe_run_backend(command: str, cwd: Path | None) -> dict[str, Any]:
    if not command.strip():
        return {"attempted": False, "returncode": None, "stdout_tail": "", "stderr_tail": ""}
    if cwd is None:
        raise ValueError("--backend-command requires --backend-workspace")
    result = subprocess.run(
        shlex.split(command),
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "attempted": True,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-6000:],
        "stderr_tail": result.stderr[-6000:],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    eval_view_indices = parse_view_indices(args.eval_view_indices, len(views)) if str(args.eval_view_indices).strip() else []
    workspace = args.backend_workspace.resolve() if args.backend_workspace else output_dir / "backend_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for index in view_indices:
        view = views[index]
        selected_rows.append(
            {
                "view_index": int(index),
                "camera_id": str(view.get("camera_id", "")),
                "image": str(view.get("image_path", "")),
                "mask": str(view.get("mask_path", "")),
                "target_size": int(args.target_size),
            }
        )
    workspace_manifest = {
        "research_only": True,
        "backend": args.backend,
        "scene_dir": str(scene_dir),
        "view_indices": view_indices,
        "eval_view_indices": eval_view_indices,
        "target_size": int(args.target_size),
        "views": selected_rows,
        "contract": {
            "required_backend_output": "one shared 3D mesh, or one consistent multi-view depth set tied to the same calibrated frame",
            "forbidden_as_success": "per-view unrelated patches, numeric-only coverage, visual-hull shell without Open3D full/head/face/hairline/hands pass",
            "strict_export_allowed": False,
        },
    }
    workspace_manifest_path = workspace / "a5_external_dense_backend_workspace_manifest.json"
    workspace_manifest_path.write_text(json.dumps(workspace_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(str(workspace_manifest_path))

    backend_run = maybe_run_backend(args.backend_command, workspace if args.backend_workspace else None)
    input_mesh_summary: dict[str, Any] = {"provided": False}
    if args.input_mesh:
        mesh_path = args.input_mesh.resolve()
        if mesh_path.is_file():
            vertices, faces, _colors = read_ascii_ply(mesh_path)
            input_mesh_summary = {
                "provided": True,
                "path": str(mesh_path),
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
                "bbox_min": [float(v) for v in vertices.min(axis=0)] if vertices.size else [0.0, 0.0, 0.0],
                "bbox_max": [float(v) for v in vertices.max(axis=0)] if vertices.size else [0.0, 0.0, 0.0],
                "components": mesh_components(int(vertices.shape[0]), faces),
            }
        else:
            input_mesh_summary = {"provided": True, "path": str(mesh_path), "exists": False}

    depth_count = depth_file_count(args.input_depth_dir.resolve() if args.input_depth_dir else None)
    has_shared_surface = bool(input_mesh_summary.get("provided") and input_mesh_summary.get("vertices", 0) > 0)
    has_depth_set = depth_count > 0
    blocked_reason = ""
    if backend_run["attempted"] and backend_run["returncode"] != 0:
        blocked_reason = "backend_command_failed"
    elif not has_shared_surface and not has_depth_set:
        blocked_reason = "blocked_no_backend_output"
    elif has_shared_surface:
        blocked_reason = "backend_output_imported_requires_visual_teacher_gate"
    else:
        blocked_reason = "depth_set_imported_requires_consistency_and_visual_teacher_gate"

    summary_path = output_dir / "a5_external_dense_backend_preflight_summary.json"
    report_path = output_dir / "a5_external_dense_backend_preflight_summary.md"
    decision = (
        "A5 is an adapter contract only. It can prepare/check a known-camera workspace and import a backend shared "
        "mesh/depth set, but no teacher/candidate export is allowed until a separate same-protocol Open3D full/head/"
        "face/hairline/hands strict teacher gate passes."
    )
    summary = {
        "status": blocked_reason,
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "backend": args.backend,
            "scene_dir": str(scene_dir),
            "workspace": str(workspace),
            "view_indices": view_indices,
            "eval_view_indices": eval_view_indices,
            "target_size": int(args.target_size),
            "backend_run": backend_run,
            "input_mesh": input_mesh_summary,
            "input_depth_dir": str(args.input_depth_dir.resolve()) if args.input_depth_dir else "",
            "input_depth_file_count": int(depth_count),
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "decision": decision,
        "outputs": outputs + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
