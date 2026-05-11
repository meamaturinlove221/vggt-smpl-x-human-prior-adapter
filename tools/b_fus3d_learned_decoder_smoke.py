from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from b_fus3d_decoder_skeleton_smoke import load_surface_features  # noqa: E402
from optimize_raw_surface_nvdiffrast import (  # noqa: E402
    multiview_photometric_variance_loss,
    project_vertices_to_grid,
    unique_edges,
    write_colored_ply,
)
from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    hard_reference,
    import_nvdiffrast,
    load_connected_mesh,
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
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_SURFACE_TOKEN_FEATURES = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D1_surface_token_smoke_hybrid6_layer23/surface_token_features.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D5_learned_decoder_smoke_hybrid6_layer23_t128"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "single_fixed_smoke": True,
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

PART_LIMITS = {
    0: 0.0010,
    1: 0.0030,
    2: 0.0030,
    3: 0.0020,
    4: 0.0040,
    5: 0.0025,
}
PART_TO_FAMILY = {
    0: "full_body",
    1: "left_hand",
    2: "right_hand",
    3: "face_core",
    4: "hairline",
    5: "full_body",
}
FAMILY_FALLBACK = {
    "full_body": ("full_body",),
    "left_hand": ("left_hand", "full_body"),
    "right_hand": ("right_hand", "full_body"),
    "face_core": ("face_core", "face", "head", "full_body"),
    "hairline": ("hairline", "head", "full_body"),
}


class TokenConditionedOffsetMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features)) * limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D learned local decoder smoke. It trains one "
            "tiny token-conditioned residual MLP for a fixed short schedule and "
            "renders initial/final mask/depth/normal diagnostics. It never exports "
            "teacher/candidate artifacts, writes strict pass state, or calls cloud."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--surface-token-features", type=Path, default=DEFAULT_SURFACE_TOKEN_FEATURES)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--view-indices", default="0,10,24,36,45,57")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.5)
    parser.add_argument("--overfill-weight", type=float, default=0.35)
    parser.add_argument("--offset-reg-weight", type=float, default=0.20)
    parser.add_argument("--edge-reg-weight", type=float, default=0.08)
    parser.add_argument("--photometric-variance-weight", type=float, default=0.05)
    parser.add_argument("--photometric-mask-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def part_limit_array(part_ids: np.ndarray) -> np.ndarray:
    limits = np.zeros((part_ids.shape[0],), dtype=np.float32)
    for part_id, limit in PART_LIMITS.items():
        limits[np.asarray(part_ids) == int(part_id)] = float(limit)
    limits[limits <= 0] = 0.001
    return limits


def choose_feature(features: dict[str, np.ndarray], family: str) -> tuple[str, np.ndarray | None]:
    for candidate in FAMILY_FALLBACK.get(family, (family, "full_body")):
        if candidate in features:
            return candidate, np.asarray(features[candidate], dtype=np.float32).reshape(-1)
    return family, None


def compressed_feature(feature: np.ndarray | None, out_dim: int = 16) -> np.ndarray:
    if feature is None or feature.size == 0:
        return np.zeros((out_dim,), dtype=np.float32)
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    chunks = np.array_split(arr, out_dim)
    vals = [float(np.mean(chunk)) if chunk.size else 0.0 for chunk in chunks]
    vec = np.asarray(vals, dtype=np.float32)
    vec = (vec - float(vec.mean())) / max(float(vec.std()), 1e-6)
    return np.clip(vec, -3.0, 3.0).astype(np.float32)


def build_vertex_features(
    vertices: np.ndarray,
    normals: np.ndarray,
    part_ids: np.ndarray,
    token_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    geom = centered / scale
    one_hot = np.zeros((vertices.shape[0], len(PART_LIMITS)), dtype=np.float32)
    clipped = np.clip(part_ids, 0, len(PART_LIMITS) - 1)
    one_hot[np.arange(vertices.shape[0]), clipped] = 1.0
    token_table = np.zeros((vertices.shape[0], 16), dtype=np.float32)
    provenance: dict[str, Any] = {}
    for part_id in sorted(int(v) for v in set(part_ids.tolist())):
        family = PART_TO_FAMILY.get(part_id, "full_body")
        used_family, feature = choose_feature(token_features, family)
        mask = part_ids == part_id
        token_vec = compressed_feature(feature)
        token_table[mask] = token_vec[None, :]
        provenance[str(part_id)] = {
            "requested_family": family,
            "used_family": used_family,
            "feature_available": feature is not None,
            "vertex_count": int(mask.sum()),
        }
    features = np.concatenate([geom, normals, one_hot, token_table], axis=1)
    return features.astype(np.float32), provenance


def load_view_payloads(
    scene_dir: Path,
    dataset_root: Path | None,
    subset_name: str,
    view_spec: str,
    target_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], str]:
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    views = manifest["exported_views"]
    view_indices = parse_view_indices(view_spec, len(views))
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    rows: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, mask_np = load_view_rgb_mask(view, target_size)
        rgb_tensor = (
            torch.as_tensor(rgb_np.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        mask_tensor = torch.as_tensor(mask_np.astype(np.float32), dtype=torch.float32, device=device).view(
            1, 1, target_size, target_size
        )
        rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "world_to_cam": torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device).contiguous(),
                "intrinsic": torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device).contiguous(),
                "target_mask": torch.as_tensor(mask_np.astype(np.float32), dtype=torch.float32, device=device),
                "target_mask_np": mask_np.astype(bool),
                "rgb_tensor": rgb_tensor,
                "mask_tensor": mask_tensor,
            }
        )
    return rows, camera_source


def mask_metrics(mask: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = np.asarray(mask) >= float(threshold)
    tgt = np.asarray(target).astype(bool)
    inter = int((pred & tgt).sum())
    union = int((pred | tgt).sum())
    return {
        "pred_pixels": int(pred.sum()),
        "target_pixels": int(tgt.sum()),
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / max(1, int(tgt.sum()))),
        "overfill_ratio": float(((pred & ~tgt).sum()) / max(1, int(pred.sum()))),
    }


def render_mask_depth(
    dr: Any,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    render = render_nvdiffrast_view(
        dr,
        ctx,
        vertices,
        faces,
        torch.nn.functional.normalize(vertices * 0.0 + torch.tensor([0.0, 0.0, 1.0], device=vertices.device), dim=-1),
        torch.ones_like(vertices).clamp(0.0, 1.0),
        world_to_cam,
        intrinsic,
        height,
        width,
        z_sign=1.0,
    )
    return render


def save_final_view_images(
    dr: Any,
    ctx: Any,
    vertices_np: np.ndarray,
    faces_t: torch.Tensor,
    normals_np: np.ndarray,
    colors_t: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    target_size: int,
    output_dir: Path,
    prefix_kind: str,
    device: torch.device,
) -> list[str]:
    vertices_t = torch.as_tensor(vertices_np, dtype=torch.float32, device=device).contiguous()
    normals_t = torch.as_tensor(normals_np, dtype=torch.float32, device=device).contiguous()
    out: list[str] = []
    for payload in view_payloads:
        render = render_nvdiffrast_view(
            dr,
            ctx,
            vertices_t,
            faces_t,
            normals_t,
            colors_t,
            payload["world_to_cam"],
            payload["intrinsic"],
            target_size,
            target_size,
            z_sign=1.0,
        )
        prefix = output_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}_{prefix_kind}"
        mask = render["mask"].detach().cpu().numpy().astype(np.float32)
        depth = render["depth"].detach().cpu().numpy().astype(np.float32)
        normal = render["normal"].detach().cpu().numpy().astype(np.float32)
        save_image(prefix.with_name(prefix.name + "_mask.png"), mask)
        save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(depth, mask > 0.5))
        save_image(prefix.with_name(prefix.name + "_normal.png"), (normal + 1.0) * 0.5)
        out.extend(
            [
                str(prefix.with_name(prefix.name + "_mask.png")),
                str(prefix.with_name(prefix.name + "_depth.png")),
                str(prefix.with_name(prefix.name + "_normal.png")),
            ]
        )
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Learned Decoder Smoke",
        "",
        "This is a fixed local smoke for one tiny token-conditioned residual MLP.",
        "It is not a mentor pass, not a teacher, not a candidate, and not a cloud unblocker.",
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
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Outputs",
        "",
    ]
    for item in summary.get("outputs", []):
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    dr, import_error = import_nvdiffrast()
    if dr is None:
        raise RuntimeError(f"nvdiffrast unavailable: {import_error}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for B-Fus3D learned decoder smoke")
    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    torch.manual_seed(20260507)
    np.random.seed(20260507)

    feature_payload = load_surface_features(args.surface_token_features, [])
    token_features = feature_payload["features"]
    mesh = load_connected_mesh(args.template_payload)
    base_vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    part_ids_np = np.asarray(mesh["part_ids"], dtype=np.int64)
    base_normals_np = np.asarray(mesh["normals"], dtype=np.float32)
    colors_np = np.asarray(mesh["part_colors"], dtype=np.float32)
    vertex_features_np, feature_provenance = build_vertex_features(
        base_vertices_np,
        base_normals_np,
        part_ids_np,
        token_features,
    )
    limits_np = part_limit_array(part_ids_np).reshape(-1, 1)
    edges_np = unique_edges(faces_np)

    faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    base_vertices = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=device).contiguous()
    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device).contiguous()
    limits = torch.as_tensor(limits_np, dtype=torch.float32, device=device).contiguous()
    edges = torch.as_tensor(edges_np, dtype=torch.long, device=device).contiguous()
    colors_t = torch.as_tensor(colors_np, dtype=torch.float32, device=device).contiguous()
    view_payloads, camera_source = load_view_payloads(
        args.scene_dir.resolve(),
        args.dataset_root,
        args.subset_name,
        args.view_indices,
        int(args.target_size),
        device,
    )

    model = TokenConditionedOffsetMLP(vertex_features.shape[1], int(args.hidden_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_curve: list[dict[str, Any]] = []
    start = time.perf_counter()
    height = width = int(args.target_size)
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = model(vertex_features, limits)
        vertices = base_vertices + delta
        mask_bce_total = torch.zeros((), dtype=torch.float32, device=device)
        recall_total = torch.zeros((), dtype=torch.float32, device=device)
        overfill_total = torch.zeros((), dtype=torch.float32, device=device)
        metrics_rows: list[dict[str, Any]] = []
        for payload in view_payloads:
            render = render_mask_depth(
                dr,
                ctx,
                vertices,
                faces_t,
                payload["world_to_cam"],
                payload["intrinsic"],
                height,
                width,
            )
            mask = render["mask_soft"].clamp(1e-4, 1.0 - 1e-4)
            target = payload["target_mask"]
            mask_bce_total = mask_bce_total + F.binary_cross_entropy(mask, target)
            recall_total = recall_total + (target * (1.0 - mask)).sum() / target.sum().clamp_min(1.0)
            overfill_total = overfill_total + ((1.0 - target) * mask).sum() / mask.sum().clamp_min(1.0)
            if step == 0 or step == int(args.steps):
                metrics_rows.append(
                    {
                        "view_index": payload["view_index"],
                        "camera_id": payload["camera_id"],
                        **mask_metrics(mask.detach().cpu().numpy(), payload["target_mask_np"]),
                    }
                )
        mask_bce = mask_bce_total / max(1, len(view_payloads))
        recall_loss = recall_total / max(1, len(view_payloads))
        overfill_loss = overfill_total / max(1, len(view_payloads))
        offset_reg = (delta / limits.clamp_min(1e-6)).pow(2).mean()
        edge_reg = (delta[edges[:, 0]] - delta[edges[:, 1]]).pow(2).mean() / (limits.mean().clamp_min(1e-6) ** 2)
        photo_loss, photo_meta = multiview_photometric_variance_loss(
            vertices,
            view_payloads,
            height,
            width,
            float(args.photometric_mask_threshold),
        )
        loss = (
            float(args.mask_bce_weight) * mask_bce
            + float(args.target_recall_weight) * recall_loss
            + float(args.overfill_weight) * overfill_loss
            + float(args.offset_reg_weight) * offset_reg
            + float(args.edge_reg_weight) * edge_reg
            + float(args.photometric_variance_weight) * photo_loss
        )
        if step < int(args.steps):
            loss.backward()
            optimizer.step()
        loss_curve.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "mask_bce": float(mask_bce.detach().cpu()),
                "target_recall_loss": float(recall_loss.detach().cpu()),
                "overfill_loss": float(overfill_loss.detach().cpu()),
                "offset_reg": float(offset_reg.detach().cpu()),
                "edge_reg": float(edge_reg.detach().cpu()),
                "photometric_variance": float(photo_loss.detach().cpu()),
                "photometric_meta": photo_meta,
                "metrics": metrics_rows,
            }
        )

    elapsed = float(time.perf_counter() - start)
    with torch.no_grad():
        final_delta = model(vertex_features, limits)
    optimized_vertices_np = (base_vertices + final_delta).detach().cpu().numpy().astype(np.float32)
    delta_np = final_delta.detach().cpu().numpy().astype(np.float32)
    optimized_normals_np = compute_vertex_normals(optimized_vertices_np, faces_np).astype(np.float32)
    normal_colors = (optimized_normals_np + 1.0) * 0.5
    mesh_path = output_dir / "b_fus3d5_learned_smoke_mesh.ply"
    carrier_path = output_dir / "b_fus3d5_carrier_initial_mesh.ply"
    normal_path = output_dir / "b_fus3d5_learned_smoke_normals.ply"
    write_colored_ply(mesh_path, optimized_vertices_np, faces_np, colors_np)
    write_colored_ply(carrier_path, base_vertices_np, faces_np, colors_np)
    write_colored_ply(normal_path, optimized_vertices_np, faces_np, normal_colors)
    output_paths = [str(mesh_path), str(carrier_path), str(normal_path)]
    output_paths.extend(
        save_final_view_images(
            dr,
            ctx,
            base_vertices_np,
            faces_t,
            base_normals_np,
            colors_t,
            view_payloads,
            height,
            output_dir,
            "initial",
            device,
        )
    )
    output_paths.extend(
        save_final_view_images(
            dr,
            ctx,
            optimized_vertices_np,
            faces_t,
            optimized_normals_np,
            colors_t,
            view_payloads,
            height,
            output_dir,
            "final",
            device,
        )
    )

    displacement_norm = np.linalg.norm(delta_np, axis=1)
    initial_metrics = loss_curve[0]["metrics"]
    final_metrics = loss_curve[-1]["metrics"]
    avg_initial_iou = float(np.mean([row["iou"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([row["iou"] for row in final_metrics])) if final_metrics else 0.0
    mean_delta = float(displacement_norm.mean()) if displacement_norm.size else 0.0
    max_delta = float(displacement_norm.max()) if displacement_norm.size else 0.0
    decision = (
        "The fixed B-Fus3D5 learned smoke ran a tiny token-conditioned residual MLP. "
        "Treat any mask or loss movement as a research signal only. This does not "
        "prove realistic face/hair/hand surface, does not satisfy Open3D visual "
        "review, and cannot unblock cloud or export."
    )
    summary: dict[str, Any] = {
        **STRICT_FACTS,
        "task": "b_fus3d_learned_decoder_smoke",
        "truthful_status": "research_fixed_learned_smoke_only_not_candidate_not_teacher",
        "contract": CONTRACT,
        "summary": {
            "scene_dir": str(args.scene_dir.resolve()),
            "surface_token_features": str(args.surface_token_features.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "views": [row["view_index"] for row in view_payloads],
            "target_size": int(args.target_size),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "vertices": int(base_vertices_np.shape[0]),
            "faces": int(faces_np.shape[0]),
            "feature_dim": feature_payload.get("feature_dim"),
            "vertex_feature_dim": int(vertex_features_np.shape[1]),
            "elapsed_seconds": elapsed,
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "initial_loss": float(loss_curve[0]["loss"]) if loss_curve else None,
            "final_loss": float(loss_curve[-1]["loss"]) if loss_curve else None,
            "loss_delta": float(loss_curve[-1]["loss"] - loss_curve[0]["loss"]) if loss_curve else None,
            "mean_vertex_delta": mean_delta,
            "max_vertex_delta": max_delta,
        },
        "feature_provenance": feature_provenance,
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": output_paths
        + [
            str(output_dir / "b_fus3d_learned_decoder_smoke_summary.json"),
            str(output_dir / "b_fus3d_learned_decoder_smoke_summary.md"),
        ],
    }
    summary = json_ready(summary)
    (output_dir / "b_fus3d_learned_decoder_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "b_fus3d_learned_decoder_smoke_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
