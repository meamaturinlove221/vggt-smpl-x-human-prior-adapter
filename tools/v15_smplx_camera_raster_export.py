from __future__ import annotations

import argparse
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

from tools.smplx_numpy import (  # noqa: E402
    compute_vertex_normals,
    forward_smplx_mesh,
    load_smplx_model,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)
from v15_common import LOCAL_ROOT, REPORTS, json_ready, scalar_stats, safe_v15_output_dir, utc_now, write_json  # noqa: E402


DEFAULT_SMPLX_ROOT = Path("G:/\u6570\u636e\u96c6/datasets/smplx")
DEFAULT_OUT = LOCAL_ROOT / "V15_SMPLX_native_camera_raster_export"
DEFAULT_JSON = REPORTS / "20260508_v15_smplx_camera_raster_export.json"
DEFAULT_MD = REPORTS / "20260508_v15_smplx_camera_raster_export.md"


def axis_angle_probe_pose(num_joints: int) -> np.ndarray:
    pose = np.zeros((num_joints, 3), dtype=np.float32)
    if num_joints > 16:
        pose[16, 2] = -0.15
    if num_joints > 17:
        pose[17, 2] = 0.15
    if num_joints > 20:
        pose[20, 1] = -0.18
    if num_joints > 21:
        pose[21, 1] = 0.18
    return pose


def look_at_world_to_cam(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32) if up is None else np.asarray(up, dtype=np.float32)
    forward = target - eye
    forward = forward / np.clip(np.linalg.norm(forward), 1e-8, None)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.clip(np.linalg.norm(right), 1e-8, None)
    true_up = np.cross(right, forward)
    true_up = true_up / np.clip(np.linalg.norm(true_up), 1e-8, None)
    rotation = np.stack([right, true_up, forward], axis=0).astype(np.float32)
    world_to_cam = np.eye(4, dtype=np.float32)
    world_to_cam[:3, :3] = rotation
    world_to_cam[:3, 3] = -(rotation @ eye)
    return world_to_cam


def build_cameras(vertices: np.ndarray, view_count: int, image_size: int, focal_ratio: float) -> list[dict[str, Any]]:
    center = ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5).astype(np.float32)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    radius = float(np.linalg.norm(extent) * 0.85)
    radius = max(radius, 2.2)
    focal = float(image_size) * float(focal_ratio)
    intrinsic = np.asarray([[focal, 0.0, image_size * 0.5], [0.0, focal, image_size * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    cameras: list[dict[str, Any]] = []
    for idx in range(int(view_count)):
        angle = 2.0 * np.pi * float(idx) / float(max(view_count, 1))
        eye = center + np.asarray([np.sin(angle) * radius, 0.12 * radius, np.cos(angle) * radius], dtype=np.float32)
        w2c = look_at_world_to_cam(eye, center)
        cameras.append(
            {
                "index": idx,
                "name": f"synthetic_cam_{idx:02d}",
                "width": int(image_size),
                "height": int(image_size),
                "intrinsic": intrinsic.copy(),
                "world_to_cam": w2c,
                "camera_center_world": eye,
            }
        )
    return cameras


def normalize_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(depth[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(depth[valid].min()), float(depth[valid].max())
    if hi <= lo:
        out[valid] = 1.0
    else:
        out[valid] = np.clip((depth[valid] - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    return out


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def vertex_macro_ids(model_path: Path, vertices: np.ndarray) -> np.ndarray:
    model = load_smplx_model(model_path)
    dominant = np.asarray(model["weights"], dtype=np.float32).argmax(axis=1)
    macro = np.zeros((vertices.shape[0],), dtype=np.int64)
    macro[(dominant >= 25) & (dominant <= 39)] = 4
    macro[(dominant >= 40) & (dominant <= 54)] = 5
    macro[np.isin(dominant, [15, 22, 23, 24])] = 1
    macro[np.isin(dominant, [13, 16, 18, 20])] = 2
    macro[np.isin(dominant, [14, 17, 19, 21])] = 3
    macro[np.isin(dominant, [1, 4, 7, 10])] = 6
    macro[np.isin(dominant, [2, 5, 8, 11])] = 7
    return macro


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = safe_v15_output_dir(args.output_dir)
    model_path = resolve_smplx_model_path(args.smplx_root, args.gender)
    model = load_smplx_model(model_path)
    num_joints = int(model["parents"].shape[0])
    betas = np.zeros((int(args.num_betas),), dtype=np.float32)
    expression = np.zeros((int(args.num_expression),), dtype=np.float32)
    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=betas,
        expression=expression,
        fullpose=axis_angle_probe_pose(num_joints),
        transl=np.zeros(3, dtype=np.float32),
        scale=1.0,
    )
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    normals = compute_vertex_normals(vertices, faces)
    macro_ids = vertex_macro_ids(model_path, vertices)
    vertex_features = np.concatenate([normals.astype(np.float32), macro_ids[:, None].astype(np.float32)], axis=1)
    cameras = build_cameras(vertices, int(args.view_count), int(args.image_size), float(args.focal_ratio))

    depth_maps = []
    point_maps = []
    normal_maps = []
    macro_maps = []
    masks = []
    raster_masks = []
    per_view: list[dict[str, Any]] = []
    for camera in cameras:
        depth, points, mask, features, raster_mask, meta = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=np.asarray(camera["world_to_cam"], dtype=np.float32),
            intrinsic=np.asarray(camera["intrinsic"], dtype=np.float32),
            image_hw=(int(camera["height"]), int(camera["width"])),
            silhouette_mask=None,
            fill_knn=0,
            vertex_features=vertex_features,
            return_vertex_features=True,
            return_raster_mask=True,
        )
        cam_normals = features[..., :3]
        normal_len = np.linalg.norm(cam_normals, axis=-1)
        valid_normal = mask & (normal_len > 1e-5)
        cam_normals[valid_normal] = cam_normals[valid_normal] / normal_len[valid_normal, None]
        macro_map = np.rint(features[..., 3]).astype(np.int16)

        depth_maps.append(depth.astype(np.float32))
        point_maps.append(points.astype(np.float32))
        normal_maps.append(cam_normals.astype(np.float32))
        macro_maps.append(macro_map)
        masks.append(mask.astype(bool))
        raster_masks.append(raster_mask.astype(bool))

        depth_png = out / f"{camera['name']}_depth.png"
        mask_png = out / f"{camera['name']}_mask.png"
        normal_png = out / f"{camera['name']}_normal.png"
        save_png(depth_png, normalize_depth(depth, mask))
        save_png(mask_png, mask.astype(np.float32))
        save_png(normal_png, (cam_normals + 1.0) * 0.5)
        per_view.append(
            {
                "index": int(camera["index"]),
                "name": str(camera["name"]),
                "visible_pixels": int(mask.sum()),
                "visible_ratio": float(mask.sum() / max(mask.size, 1)),
                "rasterized_pixels": int(raster_mask.sum()),
                "depth": scalar_stats(depth[mask]),
                "normal_length": scalar_stats(np.linalg.norm(cam_normals[valid_normal], axis=-1)),
                "raster_meta": meta,
                "preview_depth_png": str(depth_png.resolve()),
                "preview_mask_png": str(mask_png.resolve()),
                "preview_normal_png": str(normal_png.resolve()),
            }
        )

    depth_arr = np.stack(depth_maps, axis=0)
    point_arr = np.stack(point_maps, axis=0)
    normal_arr = np.stack(normal_maps, axis=0)
    macro_arr = np.stack(macro_maps, axis=0)
    mask_arr = np.stack(masks, axis=0)
    raster_arr = np.stack(raster_masks, axis=0)
    camera_names = np.asarray([camera["name"] for camera in cameras])
    world_to_cam = np.stack([np.asarray(camera["world_to_cam"], dtype=np.float32) for camera in cameras], axis=0)
    intrinsics = np.stack([np.asarray(camera["intrinsic"], dtype=np.float32) for camera in cameras], axis=0)
    npz_path = out / "v15_smplx_camera_raster_export.npz"
    np.savez_compressed(
        npz_path,
        view_names=camera_names,
        depth=depth_arr,
        points_world=point_arr,
        normals_world=normal_arr,
        macro_part_ids=macro_arr,
        mask=mask_arr,
        raw_raster_mask=raster_arr,
        world_to_cam=world_to_cam,
        intrinsics=intrinsics,
        vertices=vertices,
        faces=faces,
        research_only=np.asarray(True),
    )

    visible_total = int(mask_arr.sum())
    blockers = []
    if visible_total <= 0:
        blockers.append("No visible SMPL-X pixels were rasterized from synthetic cameras.")
    if len(per_view) != int(args.view_count):
        blockers.append("Raster export did not produce the requested view count.")
    metrics = {
        "view_count": int(len(per_view)),
        "image_size": int(args.image_size),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "visible_pixels_total": visible_total,
        "visible_ratio_mean": float(np.mean([row["visible_ratio"] for row in per_view])) if per_view else 0.0,
        "visible_ratio_min": float(np.min([row["visible_ratio"] for row in per_view])) if per_view else 0.0,
        "normal_valid_pixels": int(((np.linalg.norm(normal_arr, axis=-1) > 0.5) & mask_arr).sum()),
    }
    summary = {
        "task": "v15_smplx_camera_raster_export",
        "created_utc": utc_now(),
        "status": "v15_smplx_camera_raster_ready" if not blockers else "v15_smplx_camera_raster_blocked",
        "research_only": True,
        "smplx_native_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
        "inputs": {"smplx_root": str(Path(args.smplx_root).resolve()), "model_path": str(model_path), "gender": args.gender},
        "parameters": {"view_count": int(args.view_count), "image_size": int(args.image_size), "focal_ratio": float(args.focal_ratio)},
        "metrics": metrics,
        "per_view": per_view,
        "outputs": {"raster_npz": str(npz_path.resolve()), "summary": str((out / "summary.json").resolve())},
        "decision": (
            "SMPL-X native mesh was rasterized into bounded synthetic camera depth/point/normal/part maps for audit use only."
            if not blockers
            else "SMPL-X native camera raster export did not produce visible raster maps."
        ),
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 SMPL-X Camera Raster Export",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only synthetic-camera SMPL-X raster export. No package, cloud job, registry, teacher, candidate, or strict pass is produced.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Per View", "", "| View | Visible Pixels | Visible Ratio | Rasterized Pixels |", "|---|---:|---:|---:|"])
    for row in summary["per_view"]:
        lines.append(f"| {row['name']} | {row['visible_pixels']} | {row['visible_ratio']:.5f} | {row['rasterized_pixels']} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary["blockers"]] if summary["blockers"] else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 SMPL-X native synthetic-camera raster export.")
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-expression", type=int, default=10)
    parser.add_argument("--view-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--focal-ratio", type=float, default=1.35)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "output": args.output_dir}), ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
