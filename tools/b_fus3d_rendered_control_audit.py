from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    describe_cuda_device,
    import_nvdiffrast,
    load_view_rgb_mask,
    parse_view_indices,
    render_nvdiffrast_view,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_B16_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D16_latent_field_smoke_fixed_hybrid6_layer23"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D18_rendered_control_audit_b16_real_shuffle_zero"
)
DEFAULT_REPORT = Path("reports/20260507_b_fus3d_rendered_control_audit_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "rendered_control_audit_only": True,
    "no_train": True,
    "no_optimization": True,
    "no_mesh_extraction": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "no_cloud": True,
    "not_teacher": True,
    "not_candidate": True,
}

FORBIDDEN_PATH_TOKENS = ("strict_pass", "teacher_export", "candidate_export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-Fus3D18 rendered-control audit for B16 real/shuffle/zero meshes. "
            "It renders all controls with the same cameras/masks/RGB and checks "
            "whether real beats controls in rendered diagnostics. It does not "
            "train, optimize, extract a new mesh, export teacher/candidate "
            "artifacts, write predictions, update a registry, or call cloud."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--b16-dir", type=Path, default=DEFAULT_B16_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,24,36,45,57")
    parser.add_argument("--z-sign", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in text:
            raise ValueError(f"Refusing path containing forbidden token {token!r}: {path}")


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
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = True
        vertex_count = 0
        face_count = 0
        header_lines: list[str] = []
        for line in handle:
            if header:
                header_lines.append(line.rstrip("\n"))
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("element face"):
                    face_count = int(line.split()[-1])
                elif line.strip() == "end_header":
                    header = False
                    break
        if vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        for _ in range(vertex_count):
            parts = handle.readline().strip().split()
            if len(parts) < 3:
                raise ValueError(f"Bad vertex row in {path}")
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        for _ in range(face_count):
            parts = handle.readline().strip().split()
            if not parts:
                continue
            n = int(parts[0])
            if n < 3 or len(parts) < n + 1:
                continue
            idx = [int(v) for v in parts[1 : n + 1]]
            if n == 3:
                faces.append(idx)
            else:
                # Fan triangulate if needed.
                for i in range(1, n - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_mesh(path: Path) -> dict[str, np.ndarray]:
    vertices, faces = read_ascii_ply(path)
    if faces.size == 0:
        raise ValueError(f"PLY has no faces: {path}")
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    # Stable grey with a tiny normal tint for rendered RGB diagnostics.
    colors = np.clip(0.55 + normals * 0.18, 0.15, 0.95).astype(np.float32)
    return {"vertices": vertices, "faces": faces, "normals": normals, "colors": colors}


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_b = np.asarray(pred, dtype=bool)
    tgt_b = np.asarray(target, dtype=bool)
    inter = int((pred_b & tgt_b).sum())
    union = int((pred_b | tgt_b).sum())
    pred_count = int(pred_b.sum())
    target_count = int(tgt_b.sum())
    return {
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "intersection": inter,
        "union": union,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / target_count) if target_count else 0.0,
        "overfill_ratio": float(max(pred_count - inter, 0) / max(pred_count, 1)),
    }


def finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def render_control(
    *,
    dr: Any,
    ctx: Any,
    control_name: str,
    mesh_path: Path,
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    target_size: int,
    output_dir: Path,
    z_sign: float,
) -> dict[str, Any]:
    mesh = load_mesh(mesh_path)
    device = torch.device("cuda")
    vertices_t = torch.as_tensor(mesh["vertices"], dtype=torch.float32, device=device).contiguous()
    faces_t = torch.as_tensor(mesh["faces"], dtype=torch.int32, device=device).contiguous()
    normals_t = torch.as_tensor(mesh["normals"], dtype=torch.float32, device=device).contiguous()
    colors_t = torch.as_tensor(mesh["colors"], dtype=torch.float32, device=device).contiguous()

    control_dir = output_dir / control_name
    control_dir.mkdir(parents=True, exist_ok=True)
    view_rows: list[dict[str, Any]] = []
    ious: list[float] = []
    recalls: list[float] = []
    overfills: list[float] = []
    rgb_residuals: list[float] = []
    normal_grad_scores: list[float] = []

    for view in views:
        view_index = int(view["view_index"])
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, mask_np = load_view_rgb_mask(view, target_size)
        rgb_f = rgb_np.astype(np.float32) / 255.0

        render = render_nvdiffrast_view(
            dr,
            ctx,
            vertices_t,
            faces_t,
            normals_t,
            colors_t,
            torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device),
            torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device),
            target_size,
            target_size,
            z_sign=float(z_sign),
        )
        pred_mask = render["mask"].detach().cpu().numpy() > 0.5
        rendered_rgb = render["color"].detach().cpu().numpy().astype(np.float32)
        rendered_normal = render["normal"].detach().cpu().numpy().astype(np.float32)
        metrics = mask_metrics(pred_mask, mask_np)
        valid_rgb = pred_mask & mask_np
        if np.any(valid_rgb):
            rgb_res = float(np.mean(np.sqrt(np.sum((rendered_rgb[valid_rgb] - rgb_f[valid_rgb]) ** 2, axis=1) + 1e-6)))
        else:
            rgb_res = None
        normal_mag = np.linalg.norm(rendered_normal, axis=-1)
        normal_valid = pred_mask & np.isfinite(normal_mag) & (normal_mag > 0.1)
        if np.any(normal_valid):
            gy, gx = np.gradient(rendered_normal[..., 2])
            normal_grad = float(np.mean(np.sqrt(gx[normal_valid] ** 2 + gy[normal_valid] ** 2)))
        else:
            normal_grad = None

        save_image(control_dir / f"view_{view_index:02d}_render_mask.png", pred_mask.astype(np.float32))
        save_image(control_dir / f"view_{view_index:02d}_target_mask.png", mask_np.astype(np.float32))
        save_image(control_dir / f"view_{view_index:02d}_render_rgb.png", rendered_rgb)
        save_image(control_dir / f"view_{view_index:02d}_target_rgb.png", rgb_f)
        save_image(control_dir / f"view_{view_index:02d}_render_normal.png", rendered_normal * 0.5 + 0.5)
        delta = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
        delta[..., 0] = np.logical_and(pred_mask, ~mask_np).astype(np.float32)
        delta[..., 1] = np.logical_and(pred_mask, mask_np).astype(np.float32)
        delta[..., 2] = np.logical_and(~pred_mask, mask_np).astype(np.float32)
        save_image(control_dir / f"view_{view_index:02d}_mask_delta_rgb.png", delta)

        ious.append(float(metrics["iou"]))
        recalls.append(float(metrics["target_recall"]))
        overfills.append(float(metrics["overfill_ratio"]))
        if rgb_res is not None:
            rgb_residuals.append(rgb_res)
        if normal_grad is not None:
            normal_grad_scores.append(normal_grad)
        view_rows.append(
            {
                "view_index": view_index,
                "camera_id": camera_id,
                **metrics,
                "rgb_residual": rgb_res,
                "normal_grad_score": normal_grad,
            }
        )

    return {
        "control": control_name,
        "mesh_path": str(mesh_path.resolve()),
        "vertices": int(mesh["vertices"].shape[0]),
        "faces": int(mesh["faces"].shape[0]),
        "views": view_rows,
        "mean_iou": finite_mean(ious),
        "mean_target_recall": finite_mean(recalls),
        "mean_overfill_ratio": finite_mean(overfills),
        "mean_rgb_residual": finite_mean(rgb_residuals),
        "mean_normal_grad_score": finite_mean(normal_grad_scores),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D18 Rendered Control Audit",
        "",
        "Status: `research_only_rendered_control_audit_no_train_no_export`",
        "",
        "This audit renders B16 real/shuffle/zero meshes through the same cameras",
        "and raw masks/RGB. It is not a teacher, not a candidate, and not a strict",
        "pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        "formal_cloud_train_infer_export = blocked",
        "```",
        "",
        "## Control Metrics",
        "",
    ]
    for name, row in summary["controls"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                "```json",
                json.dumps(
                    {
                        "vertices": row["vertices"],
                        "faces": row["faces"],
                        "mean_iou": row["mean_iou"],
                        "mean_target_recall": row["mean_target_recall"],
                        "mean_overfill_ratio": row["mean_overfill_ratio"],
                        "mean_rgb_residual": row["mean_rgb_residual"],
                        "mean_normal_grad_score": row["mean_normal_grad_score"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            "```json",
            json.dumps(summary["comparison"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_safe_path(args.output_dir)
    ensure_safe_path(args.status_report)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to refresh")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dr, import_error = import_nvdiffrast()
    if dr is None:
        raise RuntimeError(f"nvdiffrast unavailable: {import_error}")
    cuda_info = describe_cuda_device()
    if not bool(cuda_info.get("available")):
        raise RuntimeError("CUDA unavailable for rendered control audit")

    b16_dir = args.b16_dir.resolve()
    mesh_paths = {
        "real": b16_dir / "real_latent_field_mesh.ply",
        "shuffle": b16_dir / "shuffle_latent_field_mesh.ply",
        "zero": b16_dir / "zero_latent_field_mesh.ply",
    }
    for name, path in mesh_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} mesh missing: {path}")

    manifest = recover_legacy_crop_source_sizes(args.scene_dir, load_scene_manifest(args.scene_dir))
    exported_views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(exported_views))
    views = [exported_views[idx] for idx in view_indices]
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)

    ctx = dr.RasterizeCudaContext(device=torch.device("cuda"))
    controls = {
        name: render_control(
            dr=dr,
            ctx=ctx,
            control_name=name,
            mesh_path=mesh_path,
            views=views,
            cameras=cameras,
            target_size=int(args.target_size),
            output_dir=args.output_dir,
            z_sign=float(args.z_sign),
        )
        for name, mesh_path in mesh_paths.items()
    }

    real = controls["real"]
    shuffle = controls["shuffle"]
    zero = controls["zero"]
    real_iou = real["mean_iou"] or 0.0
    shuffle_iou = shuffle["mean_iou"] or 0.0
    zero_iou = zero["mean_iou"] or 0.0
    real_rgb = real["mean_rgb_residual"]
    shuffle_rgb = shuffle["mean_rgb_residual"]
    zero_rgb = zero["mean_rgb_residual"]
    comparison = {
        "real_minus_shuffle_iou": float(real_iou - shuffle_iou),
        "real_minus_zero_iou": float(real_iou - zero_iou),
        "real_minus_shuffle_recall": float((real["mean_target_recall"] or 0.0) - (shuffle["mean_target_recall"] or 0.0)),
        "real_minus_zero_recall": float((real["mean_target_recall"] or 0.0) - (zero["mean_target_recall"] or 0.0)),
        "real_rgb_better_than_shuffle": None if real_rgb is None or shuffle_rgb is None else bool(real_rgb < shuffle_rgb),
        "real_rgb_better_than_zero": None if real_rgb is None or zero_rgb is None else bool(real_rgb < zero_rgb),
    }
    real_beats_controls = (
        comparison["real_minus_shuffle_iou"] > 0.02
        and comparison["real_minus_zero_iou"] > 0.02
        and comparison["real_rgb_better_than_shuffle"] is True
        and comparison["real_rgb_better_than_zero"] is True
    )
    decision = (
        "B16 real control has measurable rendered separation, but this audit is still not a pass; proceed only to a bounded visual review."
        if real_beats_controls
        else "B16 real does not robustly beat shuffle/zero in rendered diagnostics; freeze lightweight latent-field/control route."
    )
    summary = {
        "status": "research_only_rendered_control_audit_no_train_no_export",
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda": cuda_info,
            "nvdiffrast_import_error": import_error,
            "camera_source": camera_source,
            "scene_dir": str(args.scene_dir.resolve()),
            "view_indices": view_indices,
            "target_size": int(args.target_size),
        },
        "controls": controls,
        "comparison": comparison,
        "real_beats_controls": bool(real_beats_controls),
        "decision": decision,
        "blocked_actions": [
            "no_train",
            "no_optimization",
            "no_mesh_extraction",
            "no_teacher_export",
            "no_candidate_export",
            "no_predictions_write",
            "no_strict_registry_write",
            "no_cloud",
        ],
    }
    write_json(args.output_dir / "b_fus3d_rendered_control_audit_summary.json", summary)
    write_report(args.output_dir / "b_fus3d_rendered_control_audit_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "real_beats_controls": real_beats_controls, "decision": decision}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
