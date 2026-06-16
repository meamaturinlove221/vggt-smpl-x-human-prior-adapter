from __future__ import annotations

import argparse
import json
import sys
import time
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

from b_fus3d_decoder_skeleton_smoke import load_surface_features  # noqa: E402
from optimize_raw_surface_nvdiffrast import write_colored_ply  # noqa: E402
from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    compare_masks_and_depth,
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
    "B_Fus3D4_renderable_decoder_smoke_hybrid6_layer23_t128"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "deterministic_single_forward": True,
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
PART_LIMITS = {
    0: 0.0010,
    1: 0.0030,
    2: 0.0030,
    3: 0.0020,
    4: 0.0040,
    5: 0.0025,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D renderable decoder smoke. It maps pooled "
            "surface-token features to a tiny deterministic part-aware vertex "
            "displacement, renders baseline/displaced meshes with nvdiffrast, "
            "and writes diagnostic mask/depth/normal deltas. It never trains, "
            "writes predictions, exports a teacher/candidate, writes strict pass "
            "state, or calls cloud."
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
    parser.add_argument("--max-hard-raster-faces", type=int, default=-1)
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


def stable_code(name: str) -> float:
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    return float((total % 1009) / 1009.0)


def choose_feature(features: dict[str, np.ndarray], family: str) -> tuple[str, np.ndarray | None]:
    for candidate in FAMILY_FALLBACK.get(family, (family, "full_body")):
        if candidate in features:
            return candidate, np.asarray(features[candidate], dtype=np.float32).reshape(-1)
    return family, None


def feature_score(name: str, feature: np.ndarray | None) -> float:
    if feature is None or feature.size == 0:
        return 0.0
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    n = min(arr.size, 512)
    weights = np.sin(np.arange(1, n + 1, dtype=np.float32) * 0.017 + stable_code(name))
    projected = float(np.dot(arr[:n], weights) / max(np.sqrt(float(n)), 1.0))
    raw = projected * 0.08 + float(np.mean(arr)) * 1.5 + float(np.std(arr)) * 0.25 + (stable_code(name) - 0.5)
    return float(np.tanh(raw))


def build_token_displacement(
    vertices: np.ndarray,
    normals: np.ndarray,
    part_ids: np.ndarray,
    features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    displacement = np.zeros_like(vertices, dtype=np.float32)
    provenance: dict[str, Any] = {}
    for part_id in sorted(int(v) for v in set(part_ids.tolist())):
        family = PART_TO_FAMILY.get(part_id, "full_body")
        used_family, feature = choose_feature(features, family)
        score = feature_score(used_family, feature)
        limit = float(PART_LIMITS.get(part_id, 0.001))
        mask = part_ids == part_id
        direction = normals[mask].copy()
        if direction.size == 0:
            continue
        axis = np.asarray(
            [
                np.sin(0.71 + part_id),
                np.cos(1.13 + 0.5 * part_id),
                np.sin(1.71 + 0.25 * part_id),
            ],
            dtype=np.float32,
        )
        axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
        # Mostly normal-direction displacement, with a tiny deterministic tangential
        # component so downstream render deltas can detect part-specific motion.
        part_delta = direction * (score * limit) + axis[None, :] * (0.20 * score * limit)
        displacement[mask] = part_delta.astype(np.float32)
        provenance[str(part_id)] = {
            "requested_family": family,
            "used_family": used_family,
            "feature_available": feature is not None,
            "feature_score": score,
            "limit": limit,
            "vertex_count": int(mask.sum()),
            "mean_delta_norm": float(np.linalg.norm(part_delta, axis=1).mean()) if part_delta.size else 0.0,
            "max_delta_norm": float(np.linalg.norm(part_delta, axis=1).max()) if part_delta.size else 0.0,
        }
    return displacement, provenance


def load_view_payloads(
    scene_dir: Path,
    dataset_root: Path | None,
    subset_name: str,
    view_spec: str,
    target_size: int,
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
        intrinsic = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb, mask = load_view_rgb_mask(view, target_size)
        rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "intrinsic": intrinsic,
                "world_to_cam": world_to_cam,
                "rgb": rgb,
                "mask": mask,
            }
        )
    return rows, camera_source


def render_best(
    dr: Any,
    ctx: Any,
    vertices_t: torch.Tensor,
    faces_t: torch.Tensor,
    normals_t: torch.Tensor,
    colors_t: torch.Tensor,
    world_to_cam_t: torch.Tensor,
    intrinsic_t: torch.Tensor,
    hard_mask: np.ndarray,
    hard_depth: np.ndarray,
    target_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], float]:
    candidates: list[tuple[float, dict[str, torch.Tensor], dict[str, Any]]] = []
    for z_sign in (1.0, -1.0):
        render = render_nvdiffrast_view(
            dr,
            ctx,
            vertices_t,
            faces_t,
            normals_t,
            colors_t,
            world_to_cam_t,
            intrinsic_t,
            target_size,
            target_size,
            z_sign=z_sign,
        )
        mask_np = (render["mask"].detach().cpu().numpy() > 0.5)
        depth_np = render["depth"].detach().cpu().numpy().astype(np.float32)
        metrics = compare_masks_and_depth(mask_np, depth_np, hard_mask, hard_depth)
        candidates.append((z_sign, render, metrics))
    return max(
        candidates,
        key=lambda item: (
            item[2]["mask_iou"],
            -float(item[2]["median_abs_depth_residual"] if item[2]["median_abs_depth_residual"] is not None else 1e9),
        ),
    )


def render_delta_image(mask_a: np.ndarray, mask_b: np.ndarray, depth_a: np.ndarray, depth_b: np.ndarray) -> np.ndarray:
    a = np.asarray(mask_a) > 0.5
    b = np.asarray(mask_b) > 0.5
    out = np.zeros((*a.shape, 3), dtype=np.float32)
    out[a & ~b] = np.asarray([1.0, 0.25, 0.15], dtype=np.float32)
    out[b & ~a] = np.asarray([0.10, 0.45, 1.0], dtype=np.float32)
    common = a & b & np.isfinite(depth_a) & np.isfinite(depth_b) & (depth_a > 0) & (depth_b > 0)
    if np.any(common):
        diff = np.abs(depth_b - depth_a)
        hi = np.percentile(diff[common], 95)
        scaled = np.zeros_like(diff, dtype=np.float32)
        if hi > 1e-8:
            scaled[common] = np.clip(diff[common] / hi, 0.0, 1.0)
        out[common, 0] = scaled[common]
        out[common, 1] = 1.0 - 0.5 * scaled[common]
        out[common, 2] = 0.15
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Renderable Decoder Smoke",
        "",
        "This is a deterministic local geometry-touching smoke. It is not a",
        "trained decoder, not a teacher, not a candidate, and not a strict pass.",
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
        "## Part Displacement",
        "",
        "```json",
        json.dumps(summary["part_displacement"], indent=2, ensure_ascii=False),
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

    start = time.perf_counter()
    feature_payload = load_surface_features(args.surface_token_features, [])
    features = feature_payload["features"]
    mesh = load_connected_mesh(args.template_payload)
    base_vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    part_ids_np = np.asarray(mesh["part_ids"], dtype=np.int64)
    base_normals_np = np.asarray(mesh["normals"], dtype=np.float32)
    colors_np = np.asarray(mesh["part_colors"], dtype=np.float32)
    delta_np, delta_meta = build_token_displacement(base_vertices_np, base_normals_np, part_ids_np, features)
    displaced_vertices_np = (base_vertices_np + delta_np).astype(np.float32)
    displaced_normals_np = compute_vertex_normals(displaced_vertices_np, faces_np).astype(np.float32)

    views, camera_source = load_view_payloads(
        args.scene_dir.resolve(),
        args.dataset_root,
        args.subset_name,
        args.view_indices,
        int(args.target_size),
    )
    dr, import_error = import_nvdiffrast()
    if dr is None:
        raise RuntimeError(f"nvdiffrast unavailable: {import_error}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for B-Fus3D renderable decoder smoke")
    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    colors_t = torch.as_tensor(colors_np, dtype=torch.float32, device=device).contiguous()
    base_vertices_t = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=device).contiguous()
    displaced_vertices_t = torch.as_tensor(displaced_vertices_np, dtype=torch.float32, device=device).contiguous()
    base_normals_t = torch.as_tensor(base_normals_np, dtype=torch.float32, device=device).contiguous()
    displaced_normals_t = torch.as_tensor(displaced_normals_np, dtype=torch.float32, device=device).contiguous()

    view_rows: list[dict[str, Any]] = []
    output_paths: list[str] = []
    target_size = int(args.target_size)
    for payload in views:
        world_to_cam_t = torch.as_tensor(payload["world_to_cam"], dtype=torch.float32, device=device).contiguous()
        intrinsic_t = torch.as_tensor(payload["intrinsic"], dtype=torch.float32, device=device).contiguous()
        hard_depth, hard_mask, hard_meta = hard_reference(
            base_vertices_np,
            faces_np,
            payload["world_to_cam"],
            payload["intrinsic"],
            target_size,
            target_size,
            int(args.max_hard_raster_faces),
        )
        z_sign, base_render, base_metrics = render_best(
            dr,
            ctx,
            base_vertices_t,
            faces_t,
            base_normals_t,
            colors_t,
            world_to_cam_t,
            intrinsic_t,
            hard_mask,
            hard_depth,
            target_size,
        )
        displaced_render = render_nvdiffrast_view(
            dr,
            ctx,
            displaced_vertices_t,
            faces_t,
            displaced_normals_t,
            colors_t,
            world_to_cam_t,
            intrinsic_t,
            target_size,
            target_size,
            z_sign=z_sign,
        )
        displaced_mask_np = (displaced_render["mask"].detach().cpu().numpy() > 0.5)
        displaced_depth_np = displaced_render["depth"].detach().cpu().numpy().astype(np.float32)
        displaced_metrics = compare_masks_and_depth(displaced_mask_np, displaced_depth_np, hard_mask, hard_depth)
        base_mask_np = base_render["mask"].detach().cpu().numpy().astype(np.float32)
        base_depth_np = base_render["depth"].detach().cpu().numpy().astype(np.float32)
        delta_depth = np.zeros_like(base_depth_np, dtype=np.float32)
        common = (base_mask_np > 0.5) & displaced_mask_np & np.isfinite(base_depth_np) & np.isfinite(displaced_depth_np)
        if np.any(common):
            delta_depth[common] = np.abs(displaced_depth_np[common] - base_depth_np[common])
        prefix = output_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
        save_image(prefix.with_name(prefix.name + "_base_mask.png"), base_mask_np)
        save_image(prefix.with_name(prefix.name + "_displaced_mask.png"), displaced_mask_np.astype(np.float32))
        save_image(prefix.with_name(prefix.name + "_base_depth.png"), normalize_depth(base_depth_np, base_mask_np > 0.5))
        save_image(prefix.with_name(prefix.name + "_displaced_depth.png"), normalize_depth(displaced_depth_np, displaced_mask_np))
        save_image(prefix.with_name(prefix.name + "_base_normal.png"), (base_render["normal"].detach().cpu().numpy() + 1.0) * 0.5)
        save_image(prefix.with_name(prefix.name + "_displaced_normal.png"), (displaced_render["normal"].detach().cpu().numpy() + 1.0) * 0.5)
        save_image(prefix.with_name(prefix.name + "_depth_delta.png"), normalize_depth(delta_depth, delta_depth > 0))
        save_image(
            prefix.with_name(prefix.name + "_mask_depth_delta_rgb.png"),
            render_delta_image(base_mask_np, displaced_mask_np.astype(np.float32), base_depth_np, displaced_depth_np),
        )
        for suffix in (
            "base_mask",
            "displaced_mask",
            "base_depth",
            "displaced_depth",
            "base_normal",
            "displaced_normal",
            "depth_delta",
            "mask_depth_delta_rgb",
        ):
            output_paths.append(str(prefix.with_name(prefix.name + f"_{suffix}.png")))
        view_rows.append(
            {
                "view_index": int(payload["view_index"]),
                "camera_id": str(payload["camera_id"]),
                "z_sign": float(z_sign),
                "hard_raster_meta": hard_meta,
                "base": base_metrics,
                "displaced": displaced_metrics,
                "changed_mask_pixels": int(np.logical_xor(base_mask_np > 0.5, displaced_mask_np).sum()),
                "mean_abs_depth_delta_common": float(delta_depth[common].mean()) if np.any(common) else 0.0,
                "max_abs_depth_delta_common": float(delta_depth[common].max()) if np.any(common) else 0.0,
            }
        )

    write_colored_ply(output_dir / "carrier_base_mesh.ply", base_vertices_np, faces_np, colors_np)
    write_colored_ply(output_dir / "token_displaced_mesh.ply", displaced_vertices_np, faces_np, colors_np)
    displacement_norm = np.linalg.norm(delta_np, axis=1)
    delta_colors = np.zeros_like(colors_np, dtype=np.float32)
    if displacement_norm.size and float(displacement_norm.max()) > 1e-8:
        scaled = np.clip(displacement_norm / float(displacement_norm.max()), 0.0, 1.0)
        delta_colors[:, 0] = scaled
        delta_colors[:, 1] = 1.0 - scaled
        delta_colors[:, 2] = 0.2
    write_colored_ply(output_dir / "token_displacement_norm_mesh.ply", displaced_vertices_np, faces_np, delta_colors)
    output_paths.extend(
        [
            str(output_dir / "carrier_base_mesh.ply"),
            str(output_dir / "token_displaced_mesh.ply"),
            str(output_dir / "token_displacement_norm_mesh.ply"),
        ]
    )

    elapsed = float(time.perf_counter() - start)
    summary: dict[str, Any] = {
        **STRICT_FACTS,
        "task": "b_fus3d_renderable_decoder_smoke",
        "truthful_status": "research_geometry_touching_smoke_only_not_candidate_not_teacher",
        "contract": CONTRACT,
        "summary": {
            "scene_dir": str(args.scene_dir.resolve()),
            "surface_token_features": str(args.surface_token_features.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "views": [row["view_index"] for row in view_rows],
            "target_size": int(args.target_size),
            "vertices": int(base_vertices_np.shape[0]),
            "faces": int(faces_np.shape[0]),
            "feature_dim": feature_payload.get("feature_dim"),
            "selected_view_indices": feature_payload.get("selected_view_indices", []),
            "mean_vertex_delta": float(displacement_norm.mean()) if displacement_norm.size else 0.0,
            "max_vertex_delta": float(displacement_norm.max()) if displacement_norm.size else 0.0,
            "views_with_changed_mask": int(sum(1 for row in view_rows if int(row["changed_mask_pixels"]) > 0)),
            "elapsed_seconds": elapsed,
        },
        "part_displacement": delta_meta,
        "views": view_rows,
        "decision": (
            "The smoke verifies that pooled B-Fus3D token features can drive a "
            "deterministic part-aware displacement and that the changed geometry "
            "can be rendered as mask/depth/normal deltas. This is still not a "
            "learned SDF/surface decoder, not a visual pass, and not a teacher or "
            "candidate export."
        ),
        "outputs": output_paths
        + [
            str(output_dir / "b_fus3d_renderable_decoder_smoke_summary.json"),
            str(output_dir / "b_fus3d_renderable_decoder_smoke_summary.md"),
        ],
    }
    summary = json_ready(summary)
    (output_dir / "b_fus3d_renderable_decoder_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "b_fus3d_renderable_decoder_smoke_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
