from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

FORBIDDEN_OUTPUT_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "predictions",
    "strict_gate_registry",
)
DEFAULT_SCENE_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/V9_cloud_asset_staging/assets/4k4d_scene"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_2dgs_colmap_scene"
DEFAULT_MUST3R_PLY = (
    REPO_ROOT
    / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend/must3r_run/scene_lowconf_0.0.ply"
)
DEFAULT_A5_WORKSPACE = (
    REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_2dgs_colmap_scene/_a5_known_camera_workspace"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a research-only 2DGS COLMAP scene from staged 4K4D known cameras. "
            "The output is a loader/backend decision-smoke input only; it is not a teacher, "
            "candidate, prediction export, or pass artifact."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--weak-pointcloud-ply", type=Path, default=DEFAULT_MUST3R_PLY)
    parser.add_argument("--a5-workspace-dir", type=Path, default=DEFAULT_A5_WORKSPACE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reject_formal_path(path: Path, flag_name: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if "surface_research_cloud_preflight" not in lowered:
        raise ValueError(f"{flag_name} must stay under surface_research_cloud_preflight: {path}")
    if any(token in lowered for token in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"{flag_name} looks like a formal/pass/export path: {path}")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_a5_known_camera_workspace(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/a5_known_camera_colmap_preflight.py"),
        "--scene-dir",
        str(args.scene_dir),
        "--output-dir",
        str(args.a5_workspace_dir),
        "--view-indices",
        str(args.view_indices),
        "--target-size",
        str(int(args.target_size)),
        "--overwrite",
        "--dry-run",
    ]
    start = time.perf_counter()
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    summary_path = args.a5_workspace_dir / "a5_known_camera_colmap_preflight_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "command": command,
        "returncode": int(result.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "summary_path": str(summary_path),
        "summary_status": summary.get("status", ""),
        "workspace": summary.get("workspace", {}),
        "asset_counts": summary.get("asset_counts", {}),
        "selected_view_count": len(summary.get("selected_views", [])),
    }


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def read_ply_header(path: Path) -> tuple[list[str], int, int]:
    with path.open("rb") as handle:
        header_lines: list[str] = []
        offset = 0
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header did not terminate with end_header: {path}")
            offset += len(line)
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
    vertex_count = 0
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            break
    return header_lines, vertex_count, offset


def convert_must3r_ply_to_2dgs_ply(src: Path, dst: Path, *, max_points: int | None = None) -> dict[str, Any]:
    if not src.is_file():
        return {
            "status": "missing_weak_pointcloud",
            "source": str(src),
            "output": str(dst),
            "vertex_count": 0,
        }
    header, vertex_count, offset = read_ply_header(src)
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Only binary_little_endian PLY is supported for MUSt3R weak pointcloud conversion: {src}")
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))
    expected = [("float", "x"), ("float", "y"), ("float", "z"), ("uchar", "red"), ("uchar", "green"), ("uchar", "blue")]
    if properties[:6] != expected:
        raise ValueError(f"Unexpected MUSt3R PLY vertex layout in {src}: {properties[:8]}")
    has_alpha = len(properties) >= 7 and properties[6] == ("uchar", "alpha")
    stride = 15 if has_alpha else 15
    # The current MUSt3R export has x/y/z float32 + rgba uchar. Preserve RGB and drop alpha.
    if not has_alpha:
        raise ValueError(f"Expected alpha property in MUSt3R PLY for fixed-size read: {src}")
    output_count = vertex_count if max_points is None else min(vertex_count, int(max_points))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as in_handle, dst.open("wb") as out_handle:
        in_handle.seek(offset)
        out_header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment converted_from_must3r_weak_pool_for_2dgs_research_loader_smoke",
                f"element vertex {output_count}",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
        )
        out_handle.write(out_header.encode("ascii") + b"\n")
        written = 0
        for _ in range(output_count):
            record = in_handle.read(stride)
            if len(record) != stride:
                break
            x, y, z, red, green, blue, _alpha = struct.unpack("<fffBBBB", record)
            out_handle.write(struct.pack("<ffffffBBB", x, y, z, 0.0, 0.0, 0.0, red, green, blue))
            written += 1
    return {
        "status": "converted",
        "source": str(src),
        "output": str(dst),
        "source_vertex_count": int(vertex_count),
        "vertex_count": int(written),
        "has_normals_added": True,
        "source_alignment_status": "weak_pool_only_not_known_camera_aligned",
    }


def build_scene(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    scene_root = output_dir / "2dgs_colmap_scene"
    sparse_dir = scene_root / "sparse" / "0"
    start = time.perf_counter()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    a5 = run_a5_known_camera_workspace(args)
    a5_workspace = Path(str(a5.get("workspace", {}).get("workspace_dir", "")))
    sparse_known = Path(str(a5.get("workspace", {}).get("sparse_known_text_dir", "")))
    images_src = Path(str(a5.get("workspace", {}).get("images_dir", "")))
    masks_src = Path(str(a5.get("workspace", {}).get("masks_dir", "")))
    if a5.get("returncode") != 0 or not sparse_known.is_dir() or not images_src.is_dir():
        status = "blocked_known_camera_workspace_failed"
        blockers = ["A5 known-camera workspace generation failed or did not produce images/sparse text model."]
    else:
        (scene_root / "images").mkdir(parents=True, exist_ok=True)
        (scene_root / "masks").mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        copy_tree_contents(images_src, scene_root / "images")
        if masks_src.is_dir():
            copy_tree_contents(masks_src, scene_root / "masks")
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            shutil.copy2(sparse_known / name, sparse_dir / name)
        known_camera_json = a5_workspace / "workspace" / "known_camera_model.json"
        if known_camera_json.is_file():
            shutil.copy2(known_camera_json, scene_root / "known_camera_model.json")
        weak = convert_must3r_ply_to_2dgs_ply(args.weak_pointcloud_ply.resolve(), sparse_dir / "points3D.ply")
        blockers = []
        if weak["status"] != "converted" or weak.get("vertex_count", 0) <= 0:
            status = "blocked_missing_nonempty_weak_pointcloud"
            blockers.append("2DGS loader can read cameras/images, but no non-empty initial points3D.ply was produced.")
        else:
            status = "ready_for_2dgs_research_loader_and_train_smoke"
            blockers.append("Initial pointcloud comes from MUSt3R weak pool and is not known-camera aligned, so this is not a teacher/candidate artifact.")

    summary = {
        "task": "v9_2dgs_colmap_scene_package",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "decision": (
            "2DGS COLMAP scene package is ready for research-only loader/train smoke."
            if status == "ready_for_2dgs_research_loader_and_train_smoke"
            else "2DGS COLMAP scene package is blocked; see blockers."
        ),
        "research_only": True,
        "formal_cloud_unblocked": False,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "output_dir": str(output_dir),
        "scene_root": str(scene_root),
        "images_dir": str(scene_root / "images"),
        "masks_dir": str(scene_root / "masks"),
        "sparse_dir": str(sparse_dir),
        "points3D_ply": str(sparse_dir / "points3D.ply"),
        "view_indices": str(args.view_indices),
        "target_size": int(args.target_size),
        "a5_known_camera_workspace": a5,
        "weak_pointcloud": weak if "weak" in locals() else {},
        "blockers": blockers,
        "elapsed_sec": float(time.perf_counter() - start),
    }
    (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# V9 2DGS COLMAP Scene Package",
                "",
                f"Status: `{summary['status']}`",
                "",
                summary["decision"],
                "",
                "Research-only package. No teacher, candidate, predictions, registry, or strict pass artifact was written.",
                "",
                "## Scene",
                "",
                f"- Scene root: `{summary['scene_root']}`",
                f"- Images: `{summary['images_dir']}`",
                f"- Sparse model: `{summary['sparse_dir']}`",
                f"- Initial PLY: `{summary['points3D_ply']}`",
                "",
                "## Blockers",
                "",
                *[f"- {item}" for item in blockers],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    args.scene_dir = args.scene_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.a5_workspace_dir = args.a5_workspace_dir.resolve()
    args.weak_pointcloud_ply = args.weak_pointcloud_ply.resolve()
    reject_formal_path(args.output_dir, "--output-dir")
    reject_formal_path(args.a5_workspace_dir, "--a5-workspace-dir")
    summary = build_scene(args)
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "ready_for_2dgs_research_loader_and_train_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
