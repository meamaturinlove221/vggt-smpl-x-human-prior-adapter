from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    normalize_depth,
    parse_view_indices,
    render_nvdiffrast_view,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A1 research-only refinement of an A3 visual-hull mesh seed using raw masks/cameras. "
            "This is not a teacher, not a candidate, and not a strict-pass writer."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--mesh-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--offset-limit", type=float, default=0.018)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.6)
    parser.add_argument("--overfill-weight", type=float, default=0.75)
    parser.add_argument("--edge-reg-weight", type=float, default=0.04)
    parser.add_argument("--normal-reg-weight", type=float, default=0.02)
    parser.add_argument("--photometric-weight", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices [N,3] in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces [F,3] in {path}")
    return vertices, faces


def unique_edges(faces: np.ndarray) -> np.ndarray:
    faces_np = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8, strict=False):
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def mask_metrics(mask: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = np.asarray(mask) >= float(threshold)
    tgt = np.asarray(target).astype(bool)
    inter = int((pred & tgt).sum())
    union = int((pred | tgt).sum())
    pred_pixels = int(pred.sum())
    target_pixels = int(tgt.sum())
    return {
        "pred_pixels": pred_pixels,
        "target_pixels": target_pixels,
        "iou": float(inter / max(union, 1)),
        "target_recall": float(inter / max(target_pixels, 1)),
        "precision": float(inter / max(pred_pixels, 1)),
        "overfill_ratio": float(((pred & ~tgt).sum()) / max(pred_pixels, 1)),
    }


def sample_rgb(vertices: torch.Tensor, payload: dict[str, Any], height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    hom = torch.cat([vertices, ones], dim=1)
    cam = (payload["world_to_cam"] @ hom.T).T[:, :3]
    z = cam[:, 2]
    uvw = (payload["intrinsic"] @ cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-8)
    x = (uv[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    y = (uv[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=1)
    inside = (z > 1e-6) & (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    sample_grid = grid.view(1, -1, 1, 2)
    rgb = F.grid_sample(payload["rgb_tensor"], sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[
        0, :, :, 0
    ].T
    mask = F.grid_sample(payload["mask_tensor"], sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[
        0, 0, :, 0
    ]
    weight = inside.to(vertices.dtype) * (mask > 0.5).to(vertices.dtype)
    return rgb, weight


def photometric_variance(vertices: torch.Tensor, payloads: list[dict[str, Any]], height: int, width: int) -> tuple[torch.Tensor, dict[str, Any]]:
    colors = []
    weights = []
    for payload in payloads:
        rgb, weight = sample_rgb(vertices, payload, height, width)
        colors.append(rgb)
        weights.append(weight)
    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = weight_stack.sum(dim=0)
    valid = support >= 2.0
    if not valid.any():
        return vertices.sum() * 0.0, {"valid_vertices": 0, "mean_support": 0.0}
    weighted = weight_stack[:, :, None]
    mean = (color_stack * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    variance = (((color_stack - mean[None, :, :]) ** 2) * weighted).sum(dim=(0, 2)) / support.clamp_min(1.0)
    return variance[valid].mean(), {
        "valid_vertices": int(valid.detach().sum().cpu()),
        "mean_support": float(support[valid].detach().mean().cpu()),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A1 Visual Hull Mesh Refinement Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is research-only mesh-seed refinement. It is not a teacher, not a candidate, and not a cloud unblocker.",
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
        "## Loss Curve",
        "",
        "```json",
        json.dumps(summary["loss_curve"], indent=2, ensure_ascii=False),
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
    for output in summary["outputs"]:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    dr, import_error = import_nvdiffrast()
    cuda_info = describe_cuda_device()
    if dr is None or not torch.cuda.is_available():
        summary = {
            "status": "blocked_no_nvdiffrast_or_cuda",
            "summary": {
                "research_only": True,
                "strict_candidate_passes": 0,
                "strict_teacher_passes": 0,
                "formal_cloud": "blocked",
                "nvdiffrast_import_error": import_error,
                "cuda": cuda_info,
            },
            "loss_curve": [],
            "decision": "A1 mesh refinement requires nvdiffrast and CUDA. No teacher/candidate artifact was created.",
            "outputs": [],
        }
        summary_path = output_dir / "a1_refine_visual_hull_mesh_summary.json"
        report_path = output_dir / "a1_refine_visual_hull_mesh_summary.md"
        summary["outputs"] = [str(summary_path), str(report_path)]
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(report_path, summary)
        return 2

    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)

    vertices_np, faces_np = load_mesh_npz(args.mesh_seed)
    normals_np = compute_vertex_normals(vertices_np, faces_np).astype(np.float32)
    device = torch.device("cuda")
    vertices_base = torch.from_numpy(vertices_np).to(device=device, dtype=torch.float32)
    normals_base = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)
    faces_t = torch.from_numpy(faces_np.astype(np.int32)).to(device=device)
    edges_t = torch.from_numpy(unique_edges(faces_np)).to(device=device, dtype=torch.long)
    base_edge_lengths = torch.linalg.norm(vertices_base[edges_t[:, 0]] - vertices_base[edges_t[:, 1]], dim=1).detach()
    residual_raw = torch.zeros_like(vertices_base, requires_grad=True)

    height = width = int(args.target_size)
    view_payloads: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        rgb, mask = load_view_rgb_mask(view, height)
        intrinsic = align_intrinsics_for_loaded_scene_view(
            np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32),
            view,
            height,
        )
        world_to_cam = np.asarray(cameras[camera_id]["world_to_cam"], dtype=np.float32)
        rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        view_payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "target_mask_np": mask,
                "rgb_tensor": rgb_tensor,
                "mask_tensor": mask_tensor,
                "intrinsic": torch.from_numpy(intrinsic).to(device=device, dtype=torch.float32),
                "world_to_cam": torch.from_numpy(world_to_cam).to(device=device, dtype=torch.float32),
            }
        )

    optimizer = torch.optim.Adam([residual_raw], lr=float(args.lr))
    ctx = dr.RasterizeCudaContext()
    loss_curve: list[dict[str, Any]] = []

    def current_vertices() -> torch.Tensor:
        return vertices_base + torch.tanh(residual_raw) * float(args.offset_limit)

    def evaluate(vertices_eval: torch.Tensor, include_images: bool) -> tuple[list[dict[str, Any]], list[str]]:
        metrics: list[dict[str, Any]] = []
        image_paths: list[str] = []
        colors_t = torch.ones((vertices_eval.shape[0], 3), dtype=torch.float32, device=device)
        normals_eval = torch.from_numpy(compute_vertex_normals(vertices_eval.detach().cpu().numpy(), faces_np).astype(np.float32)).to(device)
        for payload in view_payloads:
            render = render_nvdiffrast_view(
                dr,
                ctx,
                vertices_eval,
                faces_t,
                normals_eval,
                colors_t,
                payload["world_to_cam"],
                payload["intrinsic"],
                height,
                width,
                z_sign=1.0,
            )
            mask_np = render["mask"].detach().cpu().numpy()
            depth_np = render["depth"].detach().cpu().numpy()
            target_np = payload["target_mask_np"]
            row = {
                "view_index": int(payload["view_index"]),
                "camera_id": payload["camera_id"],
                **mask_metrics(mask_np, target_np),
            }
            metrics.append(row)
            if include_images:
                stem = f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
                render_mask_path = output_dir / f"{stem}_render_mask.png"
                target_mask_path = output_dir / f"{stem}_target_mask.png"
                depth_path = output_dir / f"{stem}_depth.png"
                save_image(render_mask_path, mask_np)
                save_image(target_mask_path, target_np.astype(np.float32))
                save_image(depth_path, normalize_depth(depth_np, mask_np > 0.5))
                image_paths.extend([str(render_mask_path), str(target_mask_path), str(depth_path)])
        return metrics, image_paths

    initial_metrics, _ = evaluate(vertices_base, include_images=False)
    for step in range(max(1, int(args.steps)) + 1):
        vertices = current_vertices()
        optimizer.zero_grad(set_to_none=True)
        total_loss = vertices.sum() * 0.0
        mask_bce = vertices.sum() * 0.0
        target_recall_loss = vertices.sum() * 0.0
        overfill_loss = vertices.sum() * 0.0
        for payload in view_payloads:
            colors_t = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=device)
            render = render_nvdiffrast_view(
                dr,
                ctx,
                vertices,
                faces_t,
                normals_base,
                colors_t,
                payload["world_to_cam"],
                payload["intrinsic"],
                height,
                width,
                z_sign=1.0,
            )
            pred_mask = render["mask"].clamp(1e-5, 1.0 - 1e-5)
            target = payload["mask_tensor"][0, 0]
            mask_bce = mask_bce + F.binary_cross_entropy(pred_mask, target)
            intersection = (pred_mask * target).sum()
            target_recall_loss = target_recall_loss + (1.0 - intersection / target.sum().clamp_min(1.0))
            overfill_loss = overfill_loss + (pred_mask * (1.0 - target)).sum() / pred_mask.sum().clamp_min(1.0)
        denom = max(1, len(view_payloads))
        mask_bce = mask_bce / denom
        target_recall_loss = target_recall_loss / denom
        overfill_loss = overfill_loss / denom
        edge_lengths = torch.linalg.norm(vertices[edges_t[:, 0]] - vertices[edges_t[:, 1]], dim=1)
        edge_reg = ((edge_lengths - base_edge_lengths) ** 2).mean()
        offset_reg = ((vertices - vertices_base) ** 2).mean()
        photo, photo_meta = photometric_variance(vertices, view_payloads, height, width)
        total_loss = (
            float(args.mask_bce_weight) * mask_bce
            + float(args.target_recall_weight) * target_recall_loss
            + float(args.overfill_weight) * overfill_loss
            + float(args.edge_reg_weight) * edge_reg
            + float(args.normal_reg_weight) * offset_reg
            + float(args.photometric_weight) * photo
        )
        if step < int(args.steps):
            total_loss.backward()
            optimizer.step()
        if step in {0, int(args.steps)} or step % max(1, int(args.steps) // 5) == 0:
            loss_curve.append(
                {
                    "step": int(step),
                    "loss": float(total_loss.detach().cpu()),
                    "mask_bce": float(mask_bce.detach().cpu()),
                    "target_recall_loss": float(target_recall_loss.detach().cpu()),
                    "overfill_loss": float(overfill_loss.detach().cpu()),
                    "edge_reg": float(edge_reg.detach().cpu()),
                    "offset_reg": float(offset_reg.detach().cpu()),
                    "photometric_variance": float(photo.detach().cpu()),
                    "photometric_meta": photo_meta,
                }
            )

    optimized_vertices = current_vertices().detach()
    final_normals_np = compute_vertex_normals(optimized_vertices.cpu().numpy(), faces_np).astype(np.float32)
    final_metrics, image_paths = evaluate(optimized_vertices, include_images=True)
    displacement_np = (optimized_vertices - vertices_base).detach().cpu().numpy()
    disp_norm = np.linalg.norm(displacement_np, axis=1)
    disp_color = np.repeat((disp_norm / max(float(args.offset_limit), 1e-8))[:, None], 3, axis=1)
    initial_mesh_path = output_dir / "a1_visual_hull_seed_mesh.ply"
    refined_mesh_path = output_dir / "a1_refined_visual_hull_mesh.ply"
    displacement_mesh_path = output_dir / "a1_refined_displacement_mesh.ply"
    write_colored_ply(initial_mesh_path, vertices_np, faces_np, np.full((vertices_np.shape[0], 3), 0.65, dtype=np.float32))
    write_colored_ply(refined_mesh_path, optimized_vertices.cpu().numpy(), faces_np, (final_normals_np + 1.0) * 0.5)
    write_colored_ply(displacement_mesh_path, optimized_vertices.cpu().numpy(), faces_np, disp_color)
    np.savez_compressed(
        output_dir / "a1_refined_visual_hull_mesh.npz",
        vertices=optimized_vertices.cpu().numpy().astype(np.float32),
        faces=faces_np.astype(np.int32),
        displacement=displacement_np.astype(np.float32),
        view_indices=np.asarray(view_indices, dtype=np.int16),
    )
    avg_initial_iou = float(np.mean([row["iou"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([row["iou"] for row in final_metrics])) if final_metrics else 0.0
    avg_initial_precision = float(np.mean([row["precision"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_precision = float(np.mean([row["precision"] for row in final_metrics])) if final_metrics else 0.0
    avg_initial_recall = float(np.mean([row["target_recall"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_recall = float(np.mean([row["target_recall"] for row in final_metrics])) if final_metrics else 0.0
    summary_path = output_dir / "a1_refine_visual_hull_mesh_summary.json"
    report_path = output_dir / "a1_refine_visual_hull_mesh_summary.md"
    output_paths = [
        str(initial_mesh_path),
        str(refined_mesh_path),
        str(displacement_mesh_path),
        str(output_dir / "a1_refined_visual_hull_mesh.npz"),
        *image_paths,
    ]
    decision = (
        "A1 mesh refinement is research-only. It can diagnose whether raw-mask/camera losses reduce the "
        "A3 over-covering hull, but it remains disallowed as a teacher until a separate Open3D full/head/"
        "face/hairline/hands review and strict teacher gate pass."
    )
    summary = {
        "status": "a1_refine_visual_hull_mesh_preflight_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "scene_dir": str(scene_dir),
            "mesh_seed": str(args.mesh_seed.resolve()),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": height,
            "steps": int(args.steps),
            "seed_vertices": int(vertices_np.shape[0]),
            "seed_faces": int(faces_np.shape[0]),
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "avg_initial_precision": avg_initial_precision,
            "avg_final_precision": avg_final_precision,
            "avg_precision_delta": avg_final_precision - avg_initial_precision,
            "avg_initial_recall": avg_initial_recall,
            "avg_final_recall": avg_final_recall,
            "avg_recall_delta": avg_final_recall - avg_initial_recall,
            "max_vertex_delta": float(disp_norm.max()) if disp_norm.size else 0.0,
            "mean_vertex_delta": float(disp_norm.mean()) if disp_norm.size else 0.0,
            "cuda": cuda_info,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
