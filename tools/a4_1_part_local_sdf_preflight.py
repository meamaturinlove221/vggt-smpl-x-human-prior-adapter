from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from a2_neural_field_reconstruction_preflight import (  # noqa: E402
    build_view_payloads,
    load_mesh_vertices,
    mask_metrics,
    ray_aabb_intersection,
    ray_weight_entropy,
    write_mesh_ply,
    write_point_ply,
)
from a4_neus_sdf_surface_preflight import (  # noqa: E402
    TinySDFSurfaceField,
    eikonal_loss,
    evaluate_view,
    render_sdf_volume,
)
from preflight_differentiable_renderer_backend import (  # noqa: E402
    describe_cuda_device,
    normalize_depth,
    parse_view_indices,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402


PART_ALIASES = {
    "head": "head",
    "face": "head",
    "hair": "hair",
    "hands": "hands",
    "hand": "hands",
    "left_hand": "hands",
    "right_hand": "hands",
}

PART_COLORS = {
    "head": np.asarray([0.20, 0.58, 0.95], dtype=np.float32),
    "hair": np.asarray([0.98, 0.70, 0.18], dtype=np.float32),
    "hands": np.asarray([0.35, 0.85, 0.48], dtype=np.float32),
    "unknown": np.asarray([0.66, 0.68, 0.72], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A4.1 research-only part-local SDF carrier/ROI diagnostics. "
            "This extends the A4 tiny SDF surface-field preflight with part-local "
            "held-out/train mask, RGB residual, ROI carrier, and component diagnostics. "
            "It never exports a teacher/candidate and never writes strict passes."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--mesh-seed", type=Path, required=True, help="A3/A4 mesh NPZ seed used only for bbox/optional part priors.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--eval-view-indices", default="", help="Optional held-out/eval views. Empty means training views.")
    parser.add_argument(
        "--part-carriers",
        default="head_hair_hands",
        help=(
            "Part carrier spec, e.g. head_hair_hands or comma-separated head,hair,hands. "
            "Currently diagnostics derive local ROIs from the seed bbox and optional seed part ids."
        ),
    )
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--ray-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-ray", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--pos-frequencies", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--color-weight", type=float, default=0.35)
    parser.add_argument("--background-color-weight", type=float, default=0.03)
    parser.add_argument("--surface-entropy-weight", type=float, default=0.01)
    parser.add_argument("--eikonal-weight", type=float, default=0.03)
    parser.add_argument("--sdf-l1-weight", type=float, default=0.0005)
    parser.add_argument("--sdf-beta", type=float, default=0.035)
    parser.add_argument("--density-scale", type=float, default=45.0)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--prior-radius", type=float, default=0.92)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--sdf-grid", type=int, default=48)
    parser.add_argument("--render-threshold", type=float, default=0.35)
    parser.add_argument("--part-min-pixels", type=int, default=16)
    parser.add_argument("--freeze-iou-threshold", type=float, default=0.18)
    parser.add_argument("--freeze-recall-threshold", type=float, default=0.30)
    parser.add_argument("--freeze-component-threshold", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_part_carriers(spec: str) -> list[str]:
    raw_items: list[str] = []
    for token in str(spec or "").replace("+", "_").replace(",", "_").split("_"):
        item = token.strip().lower()
        if item:
            raw_items.append(item)
    parts: list[str] = []
    for item in raw_items:
        part = PART_ALIASES.get(item)
        if part is None:
            raise ValueError(f"Unsupported part carrier {item!r}; supported: head, hair, hands")
        if part not in parts:
            parts.append(part)
    if not parts:
        parts = ["head", "hair", "hands"]
    return parts


def load_seed_payload(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "vertices" in data.files:
            vertices = np.asarray(data["vertices"], dtype=np.float32)
        elif "hybrid_vertices" in data.files:
            vertices = np.asarray(data["hybrid_vertices"], dtype=np.float32)
        else:
            raise KeyError(f"{path} does not contain vertices or hybrid_vertices")
        faces = None
        if "faces" in data.files:
            faces = np.asarray(data["faces"], dtype=np.int64)
        elif "hybrid_faces" in data.files:
            faces = np.asarray(data["hybrid_faces"], dtype=np.int64)
        part_ids = None
        if "part_ids" in data.files and data["part_ids"].shape[0] == vertices.shape[0]:
            part_ids = np.asarray(data["part_ids"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected mesh vertices [N,3] in {path}")
    payload: dict[str, np.ndarray] = {"vertices": vertices}
    if faces is not None and faces.ndim == 2 and faces.shape[1] == 3:
        payload["faces"] = faces
    if part_ids is not None:
        payload["part_ids"] = part_ids
    return payload


def carrier_roi_bounds(vertices: np.ndarray, parts: list[str]) -> dict[str, dict[str, Any]]:
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    y_norm = (vertices[:, 1] - bbox_min[1]) / span[1]
    x_norm_abs = np.abs((vertices[:, 0] - center[0]) / max(span[0], 1e-6))
    rois: dict[str, dict[str, Any]] = {}
    for part in parts:
        if part == "head":
            selector = y_norm >= 0.72
            fallback_min = np.asarray([center[0] - 0.36 * span[0], bbox_min[1] + 0.68 * span[1], center[2] - 0.36 * span[2]])
            fallback_max = np.asarray([center[0] + 0.36 * span[0], bbox_max[1] + 0.02 * span[1], center[2] + 0.36 * span[2]])
        elif part == "hair":
            selector = y_norm >= 0.84
            fallback_min = np.asarray([center[0] - 0.34 * span[0], bbox_min[1] + 0.80 * span[1], center[2] - 0.34 * span[2]])
            fallback_max = np.asarray([center[0] + 0.34 * span[0], bbox_max[1] + 0.03 * span[1], center[2] + 0.34 * span[2]])
        elif part == "hands":
            selector = (y_norm >= 0.22) & (y_norm <= 0.72) & (x_norm_abs >= 0.28)
            fallback_min = np.asarray([bbox_min[0] - 0.02 * span[0], bbox_min[1] + 0.20 * span[1], bbox_min[2] - 0.03 * span[2]])
            fallback_max = np.asarray([bbox_max[0] + 0.02 * span[0], bbox_min[1] + 0.75 * span[1], bbox_max[2] + 0.03 * span[2]])
        else:
            selector = np.ones((vertices.shape[0],), dtype=bool)
            fallback_min = bbox_min.copy()
            fallback_max = bbox_max.copy()

        if int(selector.sum()) >= 4:
            local_min = vertices[selector].min(axis=0)
            local_max = vertices[selector].max(axis=0)
            source = "seed_geometry_heuristic"
        else:
            local_min = fallback_min.astype(np.float32)
            local_max = fallback_max.astype(np.float32)
            source = "bbox_fallback"
        local_span = np.maximum(local_max - local_min, 1e-6)
        pad = np.maximum(local_span * 0.18, span * 0.025)
        rois[part] = {
            "bbox_min": (local_min - pad).astype(np.float32),
            "bbox_max": (local_max + pad).astype(np.float32),
            "source": source,
            "seed_vertex_count": int(selector.sum()),
        }
    return rois


def project_points(
    points: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (np.asarray(intrinsic, dtype=np.float32) @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    valid = np.isfinite(uv).all(axis=1) & (z > 1e-6)
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < width
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < height
    return uv, valid


def make_part_roi_mask(
    payload: dict[str, Any],
    roi: dict[str, Any],
    min_pixels: int,
    height: int,
    width: int,
) -> np.ndarray:
    local_min = np.asarray(roi["bbox_min"], dtype=np.float32)
    local_max = np.asarray(roi["bbox_max"], dtype=np.float32)
    corners = np.asarray(
        [
            [local_min[0], local_min[1], local_min[2]],
            [local_min[0], local_min[1], local_max[2]],
            [local_min[0], local_max[1], local_min[2]],
            [local_min[0], local_max[1], local_max[2]],
            [local_max[0], local_min[1], local_min[2]],
            [local_max[0], local_min[1], local_max[2]],
            [local_max[0], local_max[1], local_min[2]],
            [local_max[0], local_max[1], local_max[2]],
        ],
        dtype=np.float32,
    )
    uv, valid = project_points(corners, payload["world_to_cam_np"], payload["intrinsic_np"], height, width)
    target_mask = payload["target_mask_np"].astype(bool)
    if int(valid.sum()) >= 2:
        xy = uv[valid]
        x0 = int(np.floor(np.clip(xy[:, 0].min(), 0, width - 1)))
        x1 = int(np.ceil(np.clip(xy[:, 0].max(), 0, width - 1)))
        y0 = int(np.floor(np.clip(xy[:, 1].min(), 0, height - 1)))
        y1 = int(np.ceil(np.clip(xy[:, 1].max(), 0, height - 1)))
        pad_x = max(1, int(round((x1 - x0 + 1) * 0.15)))
        pad_y = max(1, int(round((y1 - y0 + 1) * 0.15)))
        x0 = max(0, x0 - pad_x)
        x1 = min(width - 1, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(height - 1, y1 + pad_y)
        roi_mask = np.zeros((height, width), dtype=bool)
        roi_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        roi_mask &= target_mask
        if int(roi_mask.sum()) >= int(min_pixels):
            return roi_mask

    ys, xs = np.nonzero(target_mask)
    roi_mask = np.zeros((height, width), dtype=bool)
    if ys.size == 0:
        return roi_mask
    y_cut = int(np.percentile(ys, 25))
    roi_mask[ys <= y_cut, xs] = True
    if int(roi_mask.sum()) < int(min_pixels):
        roi_mask[ys, xs] = True
    return roi_mask


def augment_payloads_with_camera_arrays(
    payloads: list[dict[str, Any]],
    manifest: dict[str, Any],
    cameras: dict[str, dict[str, np.ndarray]],
    view_indices: list[int],
    target_size: int,
) -> None:
    from preflight_differentiable_renderer_backend import align_intrinsics_for_loaded_scene_view

    views = manifest["exported_views"]
    for payload, view_index in zip(payloads, view_indices, strict=False):
        view = views[view_index]
        camera_id = str(view["camera_id"])
        payload["intrinsic_np"] = align_intrinsics_for_loaded_scene_view(
            np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size,
        )
        payload["cam_to_world_np"] = np.asarray(cameras[camera_id]["cam_to_world"], dtype=np.float32)
        payload["world_to_cam_np"] = np.asarray(cameras[camera_id]["world_to_cam"], dtype=np.float32)


def roi_metrics_for_view(
    result: dict[str, Any],
    payload: dict[str, Any],
    roi_mask: np.ndarray,
    render_threshold: float,
) -> dict[str, Any]:
    target = payload["target_mask_np"].astype(bool) & roi_mask.astype(bool)
    pred_opacity = np.asarray(result["opacity"], dtype=np.float32)
    metrics = mask_metrics(pred_opacity * roi_mask.astype(np.float32), target, threshold=render_threshold)
    metrics["roi_pixels"] = int(roi_mask.sum())
    metrics["target_roi_pixels"] = int(target.sum())
    visible = target.astype(bool)
    rgb_np = np.asarray(result["rgb"], dtype=np.float32)
    if np.any(visible):
        residual = np.linalg.norm(rgb_np[visible] - payload["rgb_np"][visible].astype(np.float32) / 255.0, axis=1)
        metrics["foreground_rgb_residual_mean"] = float(residual.mean())
        metrics["foreground_rgb_residual_p90"] = float(np.percentile(residual, 90))
    else:
        metrics["foreground_rgb_residual_mean"] = 0.0
        metrics["foreground_rgb_residual_p90"] = 0.0
    roi_values = pred_opacity[roi_mask.astype(bool)]
    metrics["opacity_roi_mean"] = float(roi_values.mean()) if roi_values.size else 0.0
    metrics["opacity_roi_p90"] = float(np.percentile(roi_values, 90)) if roi_values.size else 0.0
    return metrics


def grid_part_labels(grid_points: np.ndarray, rois: dict[str, dict[str, Any]]) -> np.ndarray:
    labels = np.full((grid_points.shape[0],), "unknown", dtype=object)
    volumes: dict[str, float] = {}
    for part, roi in rois.items():
        local_min = np.asarray(roi["bbox_min"], dtype=np.float32)
        local_max = np.asarray(roi["bbox_max"], dtype=np.float32)
        volumes[part] = float(np.prod(np.maximum(local_max - local_min, 1e-6)))
    for part in sorted(rois, key=lambda name: volumes[name]):
        local_min = np.asarray(rois[part]["bbox_min"], dtype=np.float32)
        local_max = np.asarray(rois[part]["bbox_max"], dtype=np.float32)
        inside = np.all((grid_points >= local_min[None, :]) & (grid_points <= local_max[None, :]), axis=1)
        labels[inside] = part
    return labels


def component_summary(
    active_grid: np.ndarray,
    grid_points: np.ndarray,
    labels_flat: np.ndarray,
    grid_resolution: int,
    parts: list[str],
) -> dict[str, Any]:
    active = np.asarray(active_grid, dtype=bool)
    visited = np.zeros(active.shape, dtype=bool)
    summaries: dict[str, Any] = {
        part: {
            "component_count": 0,
            "largest_component_voxels": 0,
            "active_voxels": int(np.sum(active & (labels_flat.reshape(active.shape) == part))),
            "components": [],
        }
        for part in parts
    }
    summaries["all"] = {
        "component_count": 0,
        "largest_component_voxels": 0,
        "active_voxels": int(active.sum()),
        "components": [],
    }
    flat_points = grid_points.reshape(grid_resolution, grid_resolution, grid_resolution, 3)
    labels_grid = labels_flat.reshape(grid_resolution, grid_resolution, grid_resolution)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    dims = active.shape
    for seed in np.argwhere(active):
        seed_tuple = tuple(int(v) for v in seed)
        if visited[seed_tuple]:
            continue
        queue: deque[tuple[int, int, int]] = deque([seed_tuple])
        visited[seed_tuple] = True
        coords: list[tuple[int, int, int]] = []
        part_counts: dict[str, int] = {}
        while queue:
            idx = queue.popleft()
            coords.append(idx)
            label = str(labels_grid[idx])
            part_counts[label] = part_counts.get(label, 0) + 1
            for delta in neighbors:
                nxt = (idx[0] + delta[0], idx[1] + delta[1], idx[2] + delta[2])
                if nxt[0] < 0 or nxt[1] < 0 or nxt[2] < 0 or nxt[0] >= dims[0] or nxt[1] >= dims[1] or nxt[2] >= dims[2]:
                    continue
                if active[nxt] and not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        count = len(coords)
        coord_array = np.asarray(coords, dtype=np.int64)
        points = flat_points[coord_array[:, 0], coord_array[:, 1], coord_array[:, 2]]
        dominant = max(part_counts, key=part_counts.get) if part_counts else "unknown"
        row = {
            "voxels": int(count),
            "dominant_part": dominant,
            "part_counts": {k: int(v) for k, v in sorted(part_counts.items())},
            "bbox_min": [float(v) for v in points.min(axis=0)],
            "bbox_max": [float(v) for v in points.max(axis=0)],
        }
        summaries["all"]["components"].append(row)
        for part in parts:
            part_voxels = int(part_counts.get(part, 0))
            if part_voxels <= 0:
                continue
            part_row = dict(row)
            part_row["part_voxels"] = part_voxels
            summaries[part]["components"].append(part_row)

    for key, value in summaries.items():
        components = sorted(value["components"], key=lambda item: int(item.get("part_voxels", item["voxels"])), reverse=True)
        value["components"] = components[:8]
        value["component_count"] = len(components)
        if components:
            value["largest_component_voxels"] = int(components[0].get("part_voxels", components[0]["voxels"]))
    return summaries


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A4.1 Part-Local SDF Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is research-only. It adds part-local carrier/ROI diagnostics on top of A4 tiny SDF; it is not a teacher, not a candidate, and not a cloud unblocker.",
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
        "## Train View Metrics",
        "",
        "```json",
        json.dumps(summary["view_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Eval View Metrics",
        "",
        "```json",
        json.dumps(summary["eval_view_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Part Metrics",
        "",
        "```json",
        json.dumps(summary["part_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Mesh/Component Summary",
        "",
        "```json",
        json.dumps(summary["mesh_component_summary"], indent=2, ensure_ascii=False),
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
    for item in summary["outputs"]:
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_no_cuda(args: argparse.Namespace, output_dir: Path, cuda_info: dict[str, Any], start_time: float) -> int:
    parts = parse_part_carriers(args.part_carriers)
    summary_path = output_dir / "a4_1_part_local_sdf_summary.json"
    report_path = output_dir / "a4_1_part_local_sdf_summary.md"
    summary = {
        "status": "blocked_no_cuda",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "part_carriers": parts,
            "cuda": cuda_info,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "view_metrics": [],
        "eval_view_metrics": [],
        "part_metrics": {},
        "mesh_component_summary": {},
        "loss_curve": [],
        "decision": "freeze: A4.1 part-local SDF preflight requires CUDA for the tiny SDF smoke. No teacher/candidate artifact was created.",
        "outputs": [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    print(json.dumps({"status": "blocked_no_cuda", "summary": str(summary_path), "report": str(report_path)}, ensure_ascii=False))
    return 2


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    cuda_info = describe_cuda_device()
    if not torch.cuda.is_available():
        return write_blocked_no_cuda(args, output_dir, cuda_info, start_time)

    device = torch.device("cuda")
    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    eval_view_indices = parse_view_indices(args.eval_view_indices, len(views)) if str(args.eval_view_indices).strip() else view_indices
    parts = parse_part_carriers(args.part_carriers)
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)
    height = width = int(args.target_size)

    seed_payload = load_seed_payload(args.mesh_seed)
    seed_vertices = load_mesh_vertices(args.mesh_seed)
    part_rois = carrier_roi_bounds(seed_payload["vertices"], parts)
    bbox_min_np = seed_vertices.min(axis=0)
    bbox_max_np = seed_vertices.max(axis=0)
    span_np = np.maximum(bbox_max_np - bbox_min_np, 1e-6)
    pad_np = span_np * float(args.bbox_pad_ratio)
    bbox_min_np = bbox_min_np - pad_np
    bbox_max_np = bbox_max_np + pad_np
    bbox_min = torch.as_tensor(bbox_min_np, dtype=torch.float32, device=device)
    bbox_max = torch.as_tensor(bbox_max_np, dtype=torch.float32, device=device)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_half = 0.5 * (bbox_max - bbox_min)

    payloads = build_view_payloads(scene_dir, manifest, cameras, view_indices, height, device)
    augment_payloads_with_camera_arrays(payloads, manifest, cameras, view_indices, height)
    for payload in payloads:
        near, far, valid = ray_aabb_intersection(payload["origins"], payload["dirs"], bbox_min, bbox_max)
        payload["near"] = near
        payload["far"] = far
        payload["valid"] = valid
        payload["valid_ray_count"] = int(valid.detach().sum().cpu())

    eval_payloads = build_view_payloads(scene_dir, manifest, cameras, eval_view_indices, height, device)
    augment_payloads_with_camera_arrays(eval_payloads, manifest, cameras, eval_view_indices, height)
    for payload in eval_payloads:
        near, far, valid = ray_aabb_intersection(payload["origins"], payload["dirs"], bbox_min, bbox_max)
        payload["near"] = near
        payload["far"] = far
        payload["valid"] = valid
        payload["valid_ray_count"] = int(valid.detach().sum().cpu())

    for payload in payloads + eval_payloads:
        payload["part_roi_masks"] = {
            part: make_part_roi_mask(payload, part_rois[part], int(args.part_min_pixels), height, width) for part in parts
        }

    field = TinySDFSurfaceField(
        int(args.pos_frequencies),
        int(args.hidden_dim),
        float(args.residual_scale),
        float(args.prior_radius),
    ).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=float(args.lr))
    rng = torch.Generator(device=device)
    rng.manual_seed(20260506)
    loss_curve: list[dict[str, Any]] = []

    for step in range(int(args.steps) + 1):
        field.train()
        payload = payloads[step % len(payloads)]
        valid_indices = torch.nonzero(payload["valid"], as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            raise RuntimeError(f"No rays intersect bbox for view {payload['view_index']}")
        sample_count = min(int(args.ray_batch_size), int(valid_indices.numel()))
        perm = torch.randperm(valid_indices.numel(), generator=rng, device=device)[:sample_count]
        ray_ids = valid_indices[perm]
        rendered = render_sdf_volume(
            field,
            payload["origins"][ray_ids],
            payload["dirs"][ray_ids],
            payload["near"][ray_ids],
            payload["far"][ray_ids],
            bbox_center,
            bbox_half,
            int(args.samples_per_ray),
            float(args.sdf_beta),
            float(args.density_scale),
        )
        target_mask = payload["mask"][ray_ids]
        target_rgb = payload["rgb"][ray_ids]
        opacity = rendered["opacity"].clamp(1e-5, 1.0 - 1e-5)
        rgb_pred = rendered["rgb"].clamp(0.0, 1.0)
        mask_loss = F.binary_cross_entropy(opacity, target_mask)
        fg = target_mask > 0.5
        if bool(fg.any()):
            color_loss = torch.sqrt(((rgb_pred[fg] - target_rgb[fg]) ** 2).sum(dim=1) + 1e-4).mean()
            entropy_loss = ray_weight_entropy(rendered["weights"][fg])
        else:
            color_loss = rgb_pred.sum() * 0.0
            entropy_loss = rgb_pred.sum() * 0.0
        bg = target_mask <= 0.5
        if bool(bg.any()):
            bg_color_loss = ((rgb_pred[bg] - 1.0) ** 2).mean()
        else:
            bg_color_loss = rgb_pred.sum() * 0.0
        eikonal = eikonal_loss(field, max(256, sample_count // 4), device)
        sdf_l1 = rendered["sdf"].abs().mean()
        loss = (
            mask_loss
            + float(args.color_weight) * color_loss
            + float(args.background_color_weight) * bg_color_loss
            + float(args.surface_entropy_weight) * entropy_loss
            + float(args.eikonal_weight) * eikonal
            + float(args.sdf_l1_weight) * sdf_l1
        )
        if step < int(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step in {0, int(args.steps)} or step % max(1, int(args.steps) // 10) == 0:
            loss_curve.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "mask_bce": float(mask_loss.detach().cpu()),
                    "foreground_color_charbonnier": float(color_loss.detach().cpu()),
                    "background_color_mse": float(bg_color_loss.detach().cpu()),
                    "surface_entropy": float(entropy_loss.detach().cpu()),
                    "eikonal": float(eikonal.detach().cpu()),
                    "sdf_l1": float(sdf_l1.detach().cpu()),
                    "train_view_index": int(payload["view_index"]),
                    "train_camera_id": payload["camera_id"],
                    "batch_foreground_fraction": float(target_mask.mean().detach().cpu()),
                    "batch_opacity_mean": float(opacity.mean().detach().cpu()),
                }
            )

    output_paths: list[str] = []
    part_metrics: dict[str, Any] = {
        part: {
            "carrier": part_rois[part]["source"],
            "roi_bbox_min": [float(v) for v in np.asarray(part_rois[part]["bbox_min"], dtype=np.float32)],
            "roi_bbox_max": [float(v) for v in np.asarray(part_rois[part]["bbox_max"], dtype=np.float32)],
            "seed_vertex_count": int(part_rois[part]["seed_vertex_count"]),
            "train": [],
            "eval": [],
        }
        for part in parts
    }

    def render_payloads_for_review(review_payloads: list[dict[str, Any]], prefix_label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for payload in review_payloads:
            result = evaluate_view(
                field,
                payload,
                bbox_center,
                bbox_half,
                int(args.samples_per_ray),
                int(args.ray_batch_size),
                height,
                width,
                float(args.render_threshold),
                float(args.sdf_beta),
                float(args.density_scale),
            )
            prefix = output_dir / f"{prefix_label}_view_{payload['view_index']:02d}_cam{payload['camera_id']}"
            save_image(prefix.with_name(prefix.name + "_target_mask.png"), payload["target_mask_np"].astype(np.float32))
            save_image(prefix.with_name(prefix.name + "_render_opacity.png"), result["opacity"])
            save_image(prefix.with_name(prefix.name + "_render_rgb.png"), result["rgb"])
            save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(result["depth"], result["opacity"] > 0.5))
            output_paths.extend(
                [
                    str(prefix.with_name(prefix.name + "_target_mask.png")),
                    str(prefix.with_name(prefix.name + "_render_opacity.png")),
                    str(prefix.with_name(prefix.name + "_render_rgb.png")),
                    str(prefix.with_name(prefix.name + "_depth.png")),
                ]
            )
            for part in parts:
                roi_mask = payload["part_roi_masks"][part]
                roi_path = prefix.with_name(prefix.name + f"_roi_{part}.png")
                save_image(roi_path, roi_mask.astype(np.float32))
                output_paths.append(str(roi_path))
                part_row = {
                    "view_index": int(payload["view_index"]),
                    "camera_id": payload["camera_id"],
                    **roi_metrics_for_view(result, payload, roi_mask, float(args.render_threshold)),
                }
                part_metrics[part][prefix_label].append(part_row)
            rows.append(
                {
                    "view_index": int(payload["view_index"]),
                    "camera_id": payload["camera_id"],
                    "valid_ray_count": int(payload["valid_ray_count"]),
                    **result["metrics"],
                }
            )
        return rows

    view_metrics = render_payloads_for_review(payloads, "train")
    eval_view_metrics = render_payloads_for_review(eval_payloads, "eval")

    for part in parts:
        for split in ("train", "eval"):
            rows = part_metrics[part][split]
            part_metrics[part][f"{split}_summary"] = {
                "avg_iou": float(np.mean([row["iou"] for row in rows])) if rows else 0.0,
                "avg_precision": float(np.mean([row["precision"] for row in rows])) if rows else 0.0,
                "avg_recall": float(np.mean([row["target_recall"] for row in rows])) if rows else 0.0,
                "avg_rgb_residual_mean": float(np.mean([row["foreground_rgb_residual_mean"] for row in rows])) if rows else 0.0,
                "avg_target_roi_pixels": float(np.mean([row["target_roi_pixels"] for row in rows])) if rows else 0.0,
            }

    grid_resolution = int(args.sdf_grid)
    xs = torch.linspace(float(bbox_min[0]), float(bbox_max[0]), grid_resolution, device=device)
    ys = torch.linspace(float(bbox_min[1]), float(bbox_max[1]), grid_resolution, device=device)
    zs = torch.linspace(float(bbox_min[2]), float(bbox_max[2]), grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
    sdf_chunks: list[torch.Tensor] = []
    color_chunks: list[torch.Tensor] = []
    field.eval()
    with torch.no_grad():
        for start in range(0, grid.shape[0], 65536):
            pts = grid[start : start + 65536]
            pts_norm = ((pts - bbox_center[None, :]) / bbox_half[None, :].clamp_min(1e-8)).clamp(-1.35, 1.35)
            sdf, color = field(pts_norm)
            sdf_chunks.append(sdf.detach().cpu())
            color_chunks.append(color.detach().cpu())
    sdf_grid = torch.cat(sdf_chunks, dim=0).numpy().reshape(grid_resolution, grid_resolution, grid_resolution).astype(np.float32)
    grid_np = grid.detach().cpu().numpy().astype(np.float32)
    grid_colors = torch.cat(color_chunks, dim=0).numpy().astype(np.float32)
    near_surface = np.abs(sdf_grid.reshape(-1)) <= float(args.sdf_beta)
    labels_flat = grid_part_labels(grid_np, part_rois)
    label_colors = np.stack([PART_COLORS.get(str(label), PART_COLORS["unknown"]) for label in labels_flat], axis=0)
    near_points = grid_np[near_surface]
    near_colors = 0.55 * grid_colors[near_surface] + 0.45 * label_colors[near_surface]
    points_path = output_dir / "a4_1_part_local_sdf_near_surface_points.ply"
    write_point_ply(points_path, near_points, near_colors)
    output_paths.append(str(points_path))
    npz_path = output_dir / "a4_1_part_local_sdf_grid.npz"
    np.savez_compressed(
        npz_path,
        sdf=sdf_grid,
        bbox_min=bbox_min_np.astype(np.float32),
        bbox_max=bbox_max_np.astype(np.float32),
        view_indices=np.asarray(view_indices, dtype=np.int16),
        eval_view_indices=np.asarray(eval_view_indices, dtype=np.int16),
        level=np.asarray(0.0, dtype=np.float32),
        sdf_beta=np.asarray(float(args.sdf_beta), dtype=np.float32),
        part_labels=np.asarray(labels_flat, dtype=str).reshape(grid_resolution, grid_resolution, grid_resolution),
        part_carriers=np.asarray(parts, dtype=str),
    )
    output_paths.append(str(npz_path))

    mesh_status = "not_attempted"
    mesh_path = output_dir / "a4_1_part_local_sdf_zero_level_mesh.ply"
    mesh_vertices = np.zeros((0, 3), dtype=np.float32)
    mesh_faces = np.zeros((0, 3), dtype=np.int64)
    try:
        from skimage import measure  # type: ignore

        if np.any(sdf_grid <= 0.0) and np.any(sdf_grid >= 0.0):
            verts, faces, _normals, _values = measure.marching_cubes(sdf_grid, level=0.0)
            scale = (bbox_max_np - bbox_min_np) / max(1, grid_resolution - 1)
            mesh_vertices = bbox_min_np[None, :] + verts.astype(np.float32) * scale[None, :]
            mesh_labels = grid_part_labels(mesh_vertices, part_rois)
            mesh_colors = np.stack([PART_COLORS.get(str(label), PART_COLORS["unknown"]) for label in mesh_labels], axis=0)
            mesh_faces = faces.astype(np.int64)
            write_mesh_ply(mesh_path, mesh_vertices, mesh_faces, mesh_colors)
            output_paths.append(str(mesh_path))
            mesh_status = "extracted"
        else:
            mesh_status = "empty_or_single_sign_sdf"
    except Exception as exc:  # noqa: BLE001
        mesh_status = f"skipped: {exc!r}"

    components = component_summary(
        near_surface.reshape(grid_resolution, grid_resolution, grid_resolution),
        grid_np,
        labels_flat,
        grid_resolution,
        parts,
    )
    mesh_component_summary = {
        "mesh_status": mesh_status,
        "mesh_vertices": int(mesh_vertices.shape[0]),
        "mesh_faces": int(mesh_faces.shape[0]),
        "near_surface_voxels": int(near_surface.sum()),
        "near_surface_fraction": float(near_surface.mean()) if near_surface.size else 0.0,
        "components": components,
    }

    sdf_values = sdf_grid.reshape(-1)
    sdf_quantiles = {
        "p01": float(np.percentile(sdf_values, 1)) if sdf_values.size else 0.0,
        "p05": float(np.percentile(sdf_values, 5)) if sdf_values.size else 0.0,
        "p50": float(np.percentile(sdf_values, 50)) if sdf_values.size else 0.0,
        "p95": float(np.percentile(sdf_values, 95)) if sdf_values.size else 0.0,
        "p99": float(np.percentile(sdf_values, 99)) if sdf_values.size else 0.0,
        "min": float(sdf_values.min()) if sdf_values.size else 0.0,
        "max": float(sdf_values.max()) if sdf_values.size else 0.0,
    }
    avg_iou = float(np.mean([row["iou"] for row in view_metrics])) if view_metrics else 0.0
    avg_precision = float(np.mean([row["precision"] for row in view_metrics])) if view_metrics else 0.0
    avg_recall = float(np.mean([row["target_recall"] for row in view_metrics])) if view_metrics else 0.0
    avg_rgb = float(np.mean([row["foreground_rgb_residual_mean"] for row in view_metrics])) if view_metrics else 0.0
    eval_avg_iou = float(np.mean([row["iou"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_precision = float(np.mean([row["precision"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_recall = float(np.mean([row["target_recall"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_rgb = (
        float(np.mean([row["foreground_rgb_residual_mean"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    )

    freeze_reasons: list[str] = []
    if eval_avg_iou < float(args.freeze_iou_threshold):
        freeze_reasons.append(f"eval_avg_render_iou {eval_avg_iou:.4f} < {float(args.freeze_iou_threshold):.4f}")
    if eval_avg_recall < float(args.freeze_recall_threshold):
        freeze_reasons.append(f"eval_avg_render_recall {eval_avg_recall:.4f} < {float(args.freeze_recall_threshold):.4f}")
    if int(mesh_component_summary["near_surface_voxels"]) < int(args.freeze_component_threshold):
        freeze_reasons.append(
            f"near_surface_voxels {int(mesh_component_summary['near_surface_voxels'])} < {int(args.freeze_component_threshold)}"
        )
    for part in parts:
        eval_summary = part_metrics[part]["eval_summary"]
        if float(eval_summary["avg_target_roi_pixels"]) < float(args.part_min_pixels):
            freeze_reasons.append(f"{part} eval ROI has too few target pixels")
        elif float(eval_summary["avg_recall"]) < float(args.freeze_recall_threshold):
            freeze_reasons.append(f"{part} eval ROI recall {float(eval_summary['avg_recall']):.4f} is below threshold")

    decision_mode = "freeze" if freeze_reasons else "go_research_only"
    if freeze_reasons:
        decision = (
            "freeze: A4.1 remains research-only and should not be wired into Modal/strict gates. "
            + "; ".join(freeze_reasons)
            + ". No teacher/candidate artifact was created."
        )
    else:
        decision = (
            "go_research_only: A4.1 part-local diagnostics produced enough tiny-SDF smoke signal to justify deeper "
            "local research review only. It is still not a teacher/candidate and cannot unblock cloud or strict gates."
        )

    summary_path = output_dir / "a4_1_part_local_sdf_summary.json"
    report_path = output_dir / "a4_1_part_local_sdf_summary.md"
    summary = {
        "status": "a4_1_part_local_sdf_research_preflight_complete",
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
            "eval_views": eval_view_indices,
            "part_carriers": parts,
            "part_roi_sources": {part: part_rois[part]["source"] for part in parts},
            "target_size": height,
            "steps": int(args.steps),
            "ray_batch_size": int(args.ray_batch_size),
            "samples_per_ray": int(args.samples_per_ray),
            "hidden_dim": int(args.hidden_dim),
            "pos_frequencies": int(args.pos_frequencies),
            "bbox_min": [float(v) for v in bbox_min_np],
            "bbox_max": [float(v) for v in bbox_max_np],
            "avg_render_iou": avg_iou,
            "avg_render_precision": avg_precision,
            "avg_render_recall": avg_recall,
            "avg_foreground_rgb_residual_mean": avg_rgb,
            "eval_avg_render_iou": eval_avg_iou,
            "eval_avg_render_precision": eval_avg_precision,
            "eval_avg_render_recall": eval_avg_recall,
            "eval_avg_foreground_rgb_residual_mean": eval_avg_rgb,
            "sdf_grid": grid_resolution,
            "sdf_beta": float(args.sdf_beta),
            "density_scale": float(args.density_scale),
            "render_threshold": float(args.render_threshold),
            "surface_entropy_weight": float(args.surface_entropy_weight),
            "eikonal_weight": float(args.eikonal_weight),
            "sdf_l1_weight": float(args.sdf_l1_weight),
            "sdf_quantiles": sdf_quantiles,
            "mesh_status": mesh_status,
            "freeze_go_decision": decision_mode,
            "freeze_reasons": freeze_reasons,
            "cuda": cuda_info,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "view_metrics": view_metrics,
        "eval_view_metrics": eval_view_metrics,
        "part_metrics": part_metrics,
        "mesh_component_summary": mesh_component_summary,
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    print(json.dumps({"status": summary["status"], "decision": decision_mode, "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
