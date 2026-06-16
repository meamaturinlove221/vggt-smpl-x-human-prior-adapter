from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
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

from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402
from tools.dna_4k4d import SUBSET_NAME, normalize_camera_id  # noqa: E402


STATUS_BLOCKED_NO_COLMAP = "blocked_no_colmap"
STATUS_READY = "ready_known_camera_workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A5 research-only known-camera COLMAP workspace preflight. It prepares same-frame "
            "images, masks, and calibrated COLMAP text metadata only; it does not run formal "
            "VGGT, create a teacher, create a candidate, or write pass state."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--colmap-command", default="colmap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and report the workspace, but skip even the lightweight COLMAP version/help probe.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def parse_view_indices(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            value = view_count + value
        if value < 0 or value >= view_count:
            raise IndexError(f"view index {raw} resolved to {value}, outside [0, {view_count})")
        out.append(value)
    if not out:
        out = list(range(min(6, view_count)))
    return sorted(dict.fromkeys(out))


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    image_size = view.get("image_size") or [target_size, target_size]
    native_size = int(image_size[0]) if len(image_size) >= 1 else int(target_size)
    meta = view.get("preprocess_meta") or {}
    if meta.get("transform") == "crop_pad_to_square" and native_size != int(target_size):
        native = align_intrinsics_for_scene_view(intrinsic, view, target_size=native_size)
        scale = float(target_size) / float(max(1, native_size))
        out = native.astype(np.float64).copy()
        out[0, :] *= scale
        out[1, :] *= scale
        return out
    return align_intrinsics_for_scene_view(intrinsic, view, target_size=target_size).astype(np.float64)


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal[1] > diagonal[2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0:
        qvec = -qvec
    qvec /= np.clip(np.linalg.norm(qvec), 1e-12, None)
    return qvec


def ensure_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"Expected camera transform with shape (3, 4) or (4, 4), got {matrix.shape}")


def command_parts(raw_command: str) -> list[str]:
    raw = str(raw_command).strip()
    if not raw:
        raise ValueError("--colmap-command must not be empty")
    unquoted = raw
    if len(unquoted) >= 2 and unquoted[0] == unquoted[-1] and unquoted[0] in {"'", '"'}:
        unquoted = unquoted[1:-1]
    if Path(unquoted).expanduser().exists():
        return [str(Path(unquoted).expanduser())]
    try:
        parts = shlex.split(raw, posix=(os.name != "nt"))
    except ValueError:
        parts = [raw]
    cleaned = [part.strip().strip("\"'") for part in parts if part.strip()]
    return cleaned or [raw]


def resolve_colmap_command(raw_command: str) -> dict[str, Any]:
    parts = command_parts(raw_command)
    first = parts[0]
    first_path = Path(first).expanduser()
    resolved: str | None = None
    if first_path.exists():
        resolved = str(first_path.resolve())
    else:
        which = shutil.which(first)
        if which:
            resolved = str(Path(which).resolve())
    available = resolved is not None
    resolved_parts = [resolved if resolved else first, *parts[1:]]
    return {
        "requested": raw_command,
        "parts": parts,
        "available": available,
        "resolved_executable": resolved or "",
        "resolved_parts": resolved_parts,
    }


def command_line(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in parts])
    return shlex.join([str(part) for part in parts])


def run_colmap_probe(resolved_parts: list[str], dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "attempted": False,
            "skipped_reason": "dry_run",
            "command": "",
            "returncode": None,
            "stdout_head": "",
            "stderr_head": "",
        }
    attempts = [["--version"], ["-h"]]
    last: dict[str, Any] | None = None
    for suffix in attempts:
        cmd = [*resolved_parts, *suffix]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        current = {
            "attempted": True,
            "command": command_line(cmd),
            "returncode": int(result.returncode),
            "stdout_head": result.stdout[:2000],
            "stderr_head": result.stderr[:2000],
        }
        if result.returncode == 0:
            return current
        last = current
    return last or {
        "attempted": False,
        "skipped_reason": "no_probe_attempted",
        "command": "",
        "returncode": None,
        "stdout_head": "",
        "stderr_head": "",
    }


def write_view_assets(
    view: dict[str, Any],
    *,
    image_name: str,
    images_dir: Path,
    masks_dir: Path,
    target_size: int,
) -> dict[str, Any]:
    image_path = Path(str(view.get("image_path", "")))
    mask_path = Path(str(view.get("mask_path", "")))
    if not image_path.is_file():
        raise FileNotFoundError(f"Scene image missing: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Scene mask missing: {mask_path}")

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    source_size = [int(image.width), int(image.height)]
    output_size = (int(target_size), int(target_size))
    resized = image.size != output_size
    if resized:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    output_image_path = images_dir / image_name
    output_mask_path = masks_dir / image_name
    image.save(output_image_path)
    mask.save(output_mask_path)
    mask_arr = np.asarray(mask) > 127
    return {
        "source_image_path": str(image_path),
        "source_mask_path": str(mask_path),
        "image_path": str(output_image_path),
        "mask_path": str(output_mask_path),
        "image_name": image_name,
        "mask_name": image_name,
        "source_size": source_size,
        "output_size": [int(image.width), int(image.height)],
        "resized_to_target_size": bool(resized),
        "mask_coverage": float(mask_arr.mean()),
    }


def selected_camera_params(
    scene_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    override = load_camera_params_sidecar(scene_dir)
    dataset_root = Path(str(manifest.get("dataset_root", ""))).expanduser()
    return resolve_scene_camera_params(manifest, dataset_root, SUBSET_NAME, override)


def build_next_commands(
    colmap_parts: list[str],
    *,
    workspace: Path,
    sparse_known_text_dir: Path,
    target_size: int,
) -> list[dict[str, Any]]:
    database_path = workspace / "database.db"
    images_dir = workspace / "images"
    image_list_path = workspace / "image_list.txt"
    sparse_triangulated_dir = workspace / "sparse_triangulated"
    dense_dir = workspace / "dense"
    fused_path = dense_dir / "fused.ply"
    commands = [
        (
            "feature_extractor",
            [
                *colmap_parts,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(images_dir),
                "--image_list_path",
                str(image_list_path),
                "--SiftExtraction.max_image_size",
                str(int(target_size)),
                "--SiftExtraction.use_gpu",
                "0",
            ],
        ),
        (
            "matcher",
            [
                *colmap_parts,
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--SiftMatching.use_gpu",
                "0",
            ],
        ),
        (
            "point_triangulator",
            [
                *colmap_parts,
                "point_triangulator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(images_dir),
                "--input_path",
                str(sparse_known_text_dir),
                "--output_path",
                str(sparse_triangulated_dir),
            ],
        ),
        (
            "image_undistorter",
            [
                *colmap_parts,
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(sparse_triangulated_dir),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(int(target_size)),
            ],
        ),
        (
            "patch_match",
            [
                *colmap_parts,
                "patch_match_stereo",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
                "--PatchMatchStereo.max_image_size",
                str(int(target_size)),
            ],
        ),
        (
            "stereo_fusion",
            [
                *colmap_parts,
                "stereo_fusion",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_path",
                str(fused_path),
            ],
        ),
    ]
    return [{"step": name, "argv": argv, "command": command_line(argv)} for name, argv in commands]


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A5 Known-Camera COLMAP Workspace Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only preflight. No formal VGGT run was launched, no teacher or candidate was exported, and no pass state was written.",
        "",
        "## Workspace",
        "",
        f"- Workspace: `{summary['workspace']['workspace_dir']}`",
        f"- Images: `{summary['workspace']['images_dir']}`",
        f"- Masks: `{summary['workspace']['masks_dir']}`",
        f"- Known-camera COLMAP text model: `{summary['workspace']['sparse_known_text_dir']}`",
        f"- Selected views: `{summary['workspace']['view_indices']}`",
        "",
        "## COLMAP",
        "",
        f"- Requested command: `{summary['colmap']['requested']}`",
        f"- Available: `{summary['colmap']['available']}`",
        f"- Resolved executable: `{summary['colmap']['resolved_executable']}`",
        f"- Probe command: `{summary['colmap_probe']['command']}`",
        f"- Probe return code: `{summary['colmap_probe']['returncode']}`",
        "",
        "## Camera Format",
        "",
        "The text model follows COLMAP's `cameras.txt`, `images.txt`, and empty `points3D.txt` sparse model layout. Cameras use `PINHOLE WIDTH HEIGHT fx fy cx cy` in the resized workspace image pixel coordinate system. Image poses are world-to-camera transforms encoded as Hamilton quaternions `QW QX QY QZ` plus translation `TX TY TZ`; every second image line is intentionally empty because no 2D observations are exported by this preflight.",
        "",
        "## Next Commands",
        "",
    ]
    if summary["next_commands"]:
        lines.extend(["```powershell"])
        lines.extend(command["command"] for command in summary["next_commands"])
        lines.extend(["```", ""])
    else:
        lines.extend(["No COLMAP commands are listed because the COLMAP executable was not found.", ""])
    lines.extend(
        [
            "## Outputs",
            "",
            *[f"- `{item}`" for item in summary["outputs"]],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if int(args.target_size) <= 0:
        raise ValueError("--target-size must be a positive integer")

    start = time.perf_counter()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    workspace = output_dir / "workspace"
    images_dir = workspace / "images"
    masks_dir = workspace / "masks"
    sparse_known_text_dir = workspace / "sparse_known_text"
    for directory in (images_dir, masks_dir, sparse_known_text_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = list(manifest.get("exported_views", []))
    if not views:
        raise ValueError(f"No exported_views in scene manifest under {scene_dir}")
    view_indices = parse_view_indices(args.view_indices, len(views))
    camera_params, camera_source = selected_camera_params(scene_dir, manifest)

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(view_indices)}",
    ]
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(view_indices)}, mean observations per image: 0",
    ]
    selected_views: list[dict[str, Any]] = []
    image_names: list[str] = []

    for image_id, view_index in enumerate(view_indices, start=1):
        view = views[view_index]
        camera_id = normalize_camera_id(view["camera_id"])
        if camera_id not in camera_params:
            raise KeyError(f"Camera {camera_id} missing from resolved camera params")

        image_name = f"{image_id:06d}_view{int(view_index):03d}_cam{camera_id}.png"
        asset_stats = write_view_assets(
            view,
            image_name=image_name,
            images_dir=images_dir,
            masks_dir=masks_dir,
            target_size=int(args.target_size),
        )
        image_names.append(image_name)

        intrinsic = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float64),
            view,
            int(args.target_size),
        )
        world_to_cam = ensure_homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float64))
        cam_to_world = ensure_homogeneous(np.asarray(camera_params[camera_id]["cam_to_world"], dtype=np.float64))
        qvec = rotation_matrix_to_qvec(world_to_cam[:3, :3])
        tvec = world_to_cam[:3, 3]
        width, height = [int(value) for value in asset_stats["output_size"]]

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        camera_lines.append(f"{image_id} PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}")
        image_lines.append(
            (
                f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} {image_id} {image_name}"
            )
        )
        image_lines.append("")
        selected_views.append(
            {
                "image_id": int(image_id),
                "colmap_camera_id": int(image_id),
                "view_index": int(view_index),
                "scene_camera_id": camera_id,
                "role": str(view.get("role", "")),
                "intrinsic_3x3": intrinsic,
                "world_to_cam_4x4": world_to_cam,
                "cam_to_world_4x4": cam_to_world,
                "qvec_hamilton_world_to_cam": qvec,
                "tvec_world_to_cam": tvec,
                "preprocess_variant": view.get("preprocess_variant", ""),
                "preprocess_meta": view.get("preprocess_meta", {}),
                **asset_stats,
            }
        )

    cameras_txt = sparse_known_text_dir / "cameras.txt"
    images_txt = sparse_known_text_dir / "images.txt"
    points3d_txt = sparse_known_text_dir / "points3D.txt"
    cameras_txt.write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    images_txt.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    points3d_txt.write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
                "# Number of points: 0, mean track length: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    image_list_path = workspace / "image_list.txt"
    image_list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")

    known_cameras_json = workspace / "known_camera_model.json"
    known_camera_payload = {
        "format": {
            "name": "COLMAP sparse text model plus JSON companion",
            "colmap_text_model_dir": str(sparse_known_text_dir),
            "camera_model": "PINHOLE",
            "camera_params": "fx fy cx cy in workspace image pixel coordinates after target-size resize",
            "pose_convention": "world_to_cam; COLMAP images.txt Hamilton quaternion QW QX QY QZ plus translation TX TY TZ",
            "points3D": "empty by design; this preflight exports known cameras only, not sparse features or tracks",
            "image_names_relative_to": str(images_dir),
        },
        "scene_dir": str(scene_dir),
        "camera_source": camera_source,
        "target_size": int(args.target_size),
        "view_indices": view_indices,
        "views": selected_views,
    }
    known_cameras_json.write_text(json.dumps(json_ready(known_camera_payload), indent=2, ensure_ascii=False), encoding="utf-8")

    colmap = resolve_colmap_command(str(args.colmap_command))
    colmap_probe = (
        run_colmap_probe(colmap["resolved_parts"], bool(args.dry_run))
        if bool(colmap["available"])
        else {
            "attempted": False,
            "skipped_reason": "colmap_unavailable",
            "command": "",
            "returncode": None,
            "stdout_head": "",
            "stderr_head": "",
        }
    )
    next_commands = (
        build_next_commands(
            colmap["resolved_parts"],
            workspace=workspace,
            sparse_known_text_dir=sparse_known_text_dir,
            target_size=int(args.target_size),
        )
        if bool(colmap["available"])
        else []
    )
    status = STATUS_READY if bool(colmap["available"]) else STATUS_BLOCKED_NO_COLMAP

    summary_json = output_dir / "a5_known_camera_colmap_preflight_summary.json"
    summary_md = output_dir / "a5_known_camera_colmap_preflight_summary.md"
    outputs = [
        str(images_dir),
        str(masks_dir),
        str(cameras_txt),
        str(images_txt),
        str(points3d_txt),
        str(image_list_path),
        str(known_cameras_json),
        str(summary_json),
        str(summary_md),
    ]
    summary = {
        "status": status,
        "research_only": True,
        "dry_run": bool(args.dry_run),
        "no_formal_vggt_run": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_pass_state_written": True,
        "scene": {
            "scene_dir": str(scene_dir),
            "seq_id": manifest.get("seq_id", ""),
            "frame_id": manifest.get("frame_id", ""),
            "same_frame_basis": "scene_manifest.seq_id/frame_id plus selected exported_views",
        },
        "workspace": {
            "output_dir": str(output_dir),
            "workspace_dir": str(workspace),
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "sparse_known_text_dir": str(sparse_known_text_dir),
            "view_indices": view_indices,
            "view_count": int(len(view_indices)),
            "target_size": int(args.target_size),
            "camera_source": camera_source,
        },
        "colmap": colmap,
        "colmap_probe": colmap_probe,
        "next_commands": next_commands,
        "notes": [
            "This preflight writes a known-camera workspace only. Run the listed commands manually after reviewing them.",
            "The feature/matcher/point_triangulator path is included to give COLMAP sparse points for automatic PatchMatch neighbor/depth selection.",
            "If sparse triangulation remains empty, COLMAP known-pose dense reconstruction may require manual PatchMatch source images and depth ranges.",
        ],
        "documentation": {
            "colmap_known_pose_faq": "https://colmap.readthedocs.io/en/latest/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses",
            "colmap_text_format": "https://colmap.readthedocs.io/en/latest/format.html#text-format",
            "colmap_cli": "https://colmap.readthedocs.io/en/latest/cli.html",
        },
        "elapsed_seconds": float(time.perf_counter() - start),
        "outputs": outputs,
    }
    summary_json.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary_md, json_ready(summary))
    print(json.dumps(json_ready({"status": status, "output_dir": str(output_dir), "summary_json": str(summary_json)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
