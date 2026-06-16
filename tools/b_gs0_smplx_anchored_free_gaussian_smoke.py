from __future__ import annotations

import argparse
import json
import math
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

from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/B_GS0_smplx_anchored_free_gaussian_smoke")
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_gs0_smplx_anchored_free_gaussian_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "single_frame_smoke_only": True,
    "no_cloud": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "writes_checkpoint": False,
    "not_teacher": True,
    "not_candidate": True,
}
FORBIDDEN_PATH_TOKENS = ("strict_pass", "teacher_export", "candidate_export")
REQUIRED_MASKS = (
    "head_vertex_mask",
    "face_front_vertex_mask",
    "hairline_vertex_mask",
    "left_hand_vertex_mask",
    "right_hand_vertex_mask",
    "lower_clothing_vertex_mask",
)
FAMILY_COLORS = {
    "constrained": (190, 190, 190),
    "hairline_free": (165, 80, 235),
    "head_free": (90, 150, 250),
    "left_hand_free": (30, 180, 255),
    "right_hand_free": (255, 105, 55),
    "clothing_free": (50, 210, 120),
    "silhouette_free": (245, 190, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-GS0 research-only smoke for SMPL-X anchored plus free Gaussians. "
            "It initializes constrained surface Gaussians and template-outside free "
            "Gaussians, rasterizes them against local 4K4D masks/RGB, and writes "
            "Open3D-reviewable PLYs. It is not training, not a candidate, not a "
            "teacher, and never writes strict pass state."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,24,36,45,57")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--constrained-stride", type=int, default=4)
    parser.add_argument("--free-per-anchor", type=int, default=2)
    parser.add_argument("--silhouette-samples-per-view", type=int, default=160)
    parser.add_argument("--max-free-points", type=int, default=9000)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
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


def load_template(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [key for key in ("hybrid_vertices", "hybrid_faces", "part_ids", *REQUIRED_MASKS) if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
        faces = np.asarray(payload["hybrid_faces"], dtype=np.int32)
        part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
        masks = {key: np.asarray(payload[key], dtype=bool) for key in REQUIRED_MASKS}
    if part_ids.shape[0] != vertices.shape[0]:
        raise ValueError(f"part_ids length {part_ids.shape[0]} != vertices {vertices.shape[0]}")
    for name, mask in masks.items():
        if mask.shape[0] != vertices.shape[0]:
            raise ValueError(f"{name} length {mask.shape[0]} != vertices {vertices.shape[0]}")
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    return {"vertices": vertices, "faces": faces, "part_ids": part_ids, "normals": normals, "masks": masks}


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
    return list(dict.fromkeys(out))


def resolve_scene_path(scene_dir: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    scene_relative = scene_dir / path
    if scene_relative.exists():
        return scene_relative.resolve()
    return path.resolve()


def load_rgb_mask(view: dict[str, Any], target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(resolve_scene_path(Path("."), view["image_path"])).convert("RGB")
    mask = Image.open(resolve_scene_path(Path("."), view["mask_path"])).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8) > 127


def load_views(scene_dir: Path, dataset_root: Path | None, subset_name: str, view_spec: str, target_size: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], str]:
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    exported = manifest["exported_views"]
    view_indices = parse_view_indices(view_spec, len(exported))
    views: list[dict[str, Any]] = []
    for idx in view_indices:
        view = dict(exported[idx])
        view["view_index"] = int(idx)
        rgb, mask = load_rgb_mask(view, target_size)
        view["rgb"] = rgb
        view["mask"] = mask
        views.append(view)
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    return views, cameras, source


def sample_indices(mask: np.ndarray, stride: int, max_count: int | None = None) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return idx.astype(np.int64)
    stride = max(1, int(stride))
    idx = idx[::stride]
    if max_count is not None and idx.size > max_count:
        keep = np.linspace(0, idx.size - 1, int(max_count), dtype=np.int64)
        idx = idx[keep]
    return idx.astype(np.int64)


def make_constrained_gaussians(template: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    vertices = template["vertices"]
    normals = template["normals"]
    part_ids = template["part_ids"]
    idx = sample_indices(np.ones((vertices.shape[0],), dtype=bool), stride=stride)
    colors = np.zeros((idx.size, 3), dtype=np.uint8)
    for row, part in enumerate(part_ids[idx]):
        if part == 1:
            colors[row] = FAMILY_COLORS["left_hand_free"]
        elif part == 2:
            colors[row] = FAMILY_COLORS["right_hand_free"]
        elif part == 3:
            colors[row] = (250, 205, 80)
        elif part == 4:
            colors[row] = FAMILY_COLORS["hairline_free"]
        elif part == 5:
            colors[row] = FAMILY_COLORS["clothing_free"]
        else:
            colors[row] = FAMILY_COLORS["constrained"]
    return {
        "points": vertices[idx].astype(np.float32),
        "normals": normals[idx].astype(np.float32),
        "colors": colors,
        "family": np.asarray(["constrained"] * idx.size),
        "anchor_index": idx.astype(np.int64),
        "scale": np.full((idx.size,), 0.004, dtype=np.float32),
        "opacity": np.full((idx.size,), 0.7, dtype=np.float32),
    }


def _orthogonal_tangent(normals: np.ndarray) -> np.ndarray:
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    alt = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    tangent = np.cross(normals, up[None, :])
    small = np.linalg.norm(tangent, axis=1) < 1e-6
    if np.any(small):
        tangent[small] = np.cross(normals[small], alt[None, :])
    return tangent / np.clip(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8, None)


def make_free_family(
    template: dict[str, np.ndarray],
    mask_name: str,
    family: str,
    *,
    stride: int,
    per_anchor: int,
    normal_offsets: tuple[float, ...],
    tangent_offsets: tuple[float, ...],
    max_count: int,
) -> dict[str, np.ndarray]:
    mask = template["masks"][mask_name]
    idx = sample_indices(mask, stride=stride, max_count=max_count)
    if idx.size == 0:
        return empty_gaussians()
    vertices = template["vertices"][idx]
    normals = template["normals"][idx]
    tangents = _orthogonal_tangent(normals)
    points: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for slot in range(max(1, int(per_anchor))):
        no = float(normal_offsets[slot % len(normal_offsets)])
        to = float(tangent_offsets[slot % len(tangent_offsets)])
        points.append(vertices + normals * no + tangents * to)
        anchors.append(idx)
    pts = np.concatenate(points, axis=0).astype(np.float32)
    anc = np.concatenate(anchors, axis=0).astype(np.int64)
    return {
        "points": pts,
        "normals": np.repeat(normals, max(1, int(per_anchor)), axis=0).astype(np.float32),
        "colors": np.repeat(np.asarray(FAMILY_COLORS[family], dtype=np.uint8)[None, :], pts.shape[0], axis=0),
        "family": np.asarray([family] * pts.shape[0]),
        "anchor_index": anc,
        "scale": np.full((pts.shape[0],), 0.006, dtype=np.float32),
        "opacity": np.full((pts.shape[0],), 0.55, dtype=np.float32),
    }


def empty_gaussians() -> dict[str, np.ndarray]:
    return {
        "points": np.zeros((0, 3), dtype=np.float32),
        "normals": np.zeros((0, 3), dtype=np.float32),
        "colors": np.zeros((0, 3), dtype=np.uint8),
        "family": np.asarray([], dtype="<U32"),
        "anchor_index": np.zeros((0,), dtype=np.int64),
        "scale": np.zeros((0,), dtype=np.float32),
        "opacity": np.zeros((0,), dtype=np.float32),
    }


def merge_gaussians(chunks: list[dict[str, np.ndarray]], max_points: int | None = None) -> dict[str, np.ndarray]:
    nonempty = [chunk for chunk in chunks if chunk["points"].shape[0] > 0]
    if not nonempty:
        return empty_gaussians()
    merged = {
        "points": np.concatenate([chunk["points"] for chunk in nonempty], axis=0).astype(np.float32),
        "normals": np.concatenate([chunk["normals"] for chunk in nonempty], axis=0).astype(np.float32),
        "colors": np.concatenate([chunk["colors"] for chunk in nonempty], axis=0).astype(np.uint8),
        "family": np.concatenate([chunk["family"] for chunk in nonempty], axis=0),
        "anchor_index": np.concatenate([chunk["anchor_index"] for chunk in nonempty], axis=0).astype(np.int64),
        "scale": np.concatenate([chunk["scale"] for chunk in nonempty], axis=0).astype(np.float32),
        "opacity": np.concatenate([chunk["opacity"] for chunk in nonempty], axis=0).astype(np.float32),
    }
    if max_points is not None and merged["points"].shape[0] > int(max_points):
        keep = np.linspace(0, merged["points"].shape[0] - 1, int(max_points), dtype=np.int64)
        merged = {key: value[keep] for key, value in merged.items()}
    return merged


def project_points(points: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32)


def draw_points(height: int, width: int, uv: np.ndarray, depth: np.ndarray, colors: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros((height, width), dtype=bool)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    radius = max(0, int(radius))
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
    order = np.argsort(depth[valid])[::-1]
    valid_idx = np.flatnonzero(valid)[order]
    for idx in valid_idx:
        x = int(round(float(uv[idx, 0])))
        y = int(round(float(uv[idx, 1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                if (xx - x) * (xx - x) + (yy - y) * (yy - y) > radius * radius:
                    continue
                if depth[idx] <= zbuf[yy, xx]:
                    zbuf[yy, xx] = depth[idx]
                    mask[yy, xx] = True
                    rgb[yy, xx] = colors[idx].astype(np.float32) / 255.0
    return mask, rgb, zbuf


def mask_iou(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    inter = int(np.count_nonzero(pred & target))
    union = int(np.count_nonzero(pred | target))
    pred_count = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(target))
    overfill = int(np.count_nonzero(pred & ~target))
    return {
        "intersection": inter,
        "union": union,
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / target_count) if target_count else 0.0,
        "overfill_ratio": float(overfill / max(pred_count, 1)),
    }


def rgb_residual(pred_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 1.0
    diff = np.abs(pred_rgb[valid].astype(np.float32) - target_rgb[valid].astype(np.float32) / 255.0)
    return float(diff.mean())


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.asarray(rgb) * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def write_ply(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(gaussians["points"], dtype=np.float32)
    colors = np.asarray(gaussians["colors"], dtype=np.uint8)
    scales = np.asarray(gaussians["scale"], dtype=np.float32)
    opacity = np.asarray(gaussians["opacity"], dtype=np.float32)
    family = np.asarray(gaussians["family"]).astype(str)
    family_ids = {name: idx for idx, name in enumerate(sorted(set(family.tolist())))}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float scale\nproperty float opacity\nproperty int family_id\n")
        handle.write("end_header\n")
        for idx, point in enumerate(points):
            color = colors[idx]
            handle.write(
                f"{float(point[0])} {float(point[1])} {float(point[2])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(scales[idx])} {float(opacity[idx])} {int(family_ids[family[idx]])}\n"
            )


def summarize_families(gaussians: dict[str, np.ndarray]) -> dict[str, Any]:
    family = np.asarray(gaussians["family"]).astype(str)
    out: dict[str, Any] = {}
    for name in sorted(set(family.tolist())):
        mask = family == name
        points = gaussians["points"][mask]
        out[name] = {
            "count": int(mask.sum()),
            "bbox_min": points.min(axis=0).tolist() if points.size else None,
            "bbox_max": points.max(axis=0).tolist() if points.size else None,
        }
    return out


def render_gaussian_set(
    name: str,
    gaussians: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    *,
    target_size: int,
    point_radius: int,
    output_dir: Path,
) -> dict[str, Any]:
    points = np.asarray(gaussians["points"], dtype=np.float32)
    colors = np.asarray(gaussians["colors"], dtype=np.uint8)
    per_view: list[dict[str, Any]] = []
    for view in views:
        camera_id = normalize_camera_id(view["camera_id"])
        camera = cameras[camera_id]
        intrinsic = align_intrinsics_for_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size=target_size)
        uv, depth = project_points(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        pred_mask, pred_rgb, zbuf = draw_points(target_size, target_size, uv, depth, colors, radius=point_radius)
        target_mask = np.asarray(view["mask"], dtype=bool)
        target_rgb = np.asarray(view["rgb"], dtype=np.uint8)
        metrics = mask_iou(pred_mask, target_mask)
        metrics["rgb_residual"] = rgb_residual(pred_rgb, target_rgb, pred_mask & target_mask)
        metrics["view_index"] = int(view["view_index"])
        metrics["camera_id"] = str(view["camera_id"])
        valid_depth = zbuf[np.isfinite(zbuf)]
        metrics["depth_pixels"] = int(valid_depth.size)
        metrics["depth_min"] = float(valid_depth.min()) if valid_depth.size else None
        metrics["depth_max"] = float(valid_depth.max()) if valid_depth.size else None
        per_view.append(metrics)
        save_mask(output_dir / "renders" / name / f"view_{int(view['view_index']):02d}_mask.png", pred_mask)
        save_rgb(output_dir / "renders" / name / f"view_{int(view['view_index']):02d}_rgb.png", pred_rgb)
    return {
        "views": per_view,
        "mean_iou": float(np.mean([row["iou"] for row in per_view])) if per_view else 0.0,
        "mean_target_recall": float(np.mean([row["target_recall"] for row in per_view])) if per_view else 0.0,
        "mean_overfill_ratio": float(np.mean([row["overfill_ratio"] for row in per_view])) if per_view else 0.0,
        "mean_rgb_residual": float(np.mean([row["rgb_residual"] for row in per_view])) if per_view else 1.0,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    lines = [
        "# B-GS0 SMPL-X Anchored + Free Gaussian Smoke",
        "",
        "Status: `research_only_gaussian_smoke_no_export`",
        "",
        "This is a local single-frame representation smoke. It initializes constrained",
        "SMPL-X/hybrid-template Gaussians plus free Gaussians for hairline, hands,",
        "clothing, and silhouette support. It is not a teacher, candidate, strict pass,",
        "or cloud unblock.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['strict_facts']['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Gaussian Counts",
        "",
        "```json",
        json.dumps(summary["family_summary"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Render Comparison",
        "",
        "```json",
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
        "## Key Outputs",
        "",
        "```text",
        *summary["key_outputs"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_safe_path(args.output_dir)
    ensure_safe_path(args.status_report)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = load_template(args.template_payload)
    views, cameras, camera_source = load_views(
        args.scene_dir,
        args.dataset_root,
        args.subset_name,
        args.view_indices,
        args.target_size,
    )

    constrained = make_constrained_gaussians(template, stride=args.constrained_stride)
    free_chunks = [
        make_free_family(
            template,
            "hairline_vertex_mask",
            "hairline_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.018, 0.032),
            tangent_offsets=(-0.006, 0.006),
            max_count=max(64, args.max_free_points // 6),
        ),
        make_free_family(
            template,
            "head_vertex_mask",
            "head_free",
            stride=max(2, args.constrained_stride * 4),
            per_anchor=1,
            normal_offsets=(0.020,),
            tangent_offsets=(0.0,),
            max_count=max(64, args.max_free_points // 8),
        ),
        make_free_family(
            template,
            "left_hand_vertex_mask",
            "left_hand_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.012, 0.024),
            tangent_offsets=(-0.004, 0.004),
            max_count=max(64, args.max_free_points // 8),
        ),
        make_free_family(
            template,
            "right_hand_vertex_mask",
            "right_hand_free",
            stride=max(1, args.constrained_stride // 2),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.012, 0.024),
            tangent_offsets=(-0.004, 0.004),
            max_count=max(64, args.max_free_points // 8),
        ),
        make_free_family(
            template,
            "lower_clothing_vertex_mask",
            "clothing_free",
            stride=max(1, args.constrained_stride),
            per_anchor=args.free_per_anchor,
            normal_offsets=(0.018, 0.030),
            tangent_offsets=(-0.008, 0.008),
            max_count=max(64, args.max_free_points // 3),
        ),
    ]
    free = merge_gaussians(free_chunks, max_points=args.max_free_points)
    anchored_plus_free = merge_gaussians([constrained, free])

    write_ply(args.output_dir / "b_gs0_constrained_only_gaussians.ply", constrained)
    write_ply(args.output_dir / "b_gs0_free_gaussians.ply", free)
    write_ply(args.output_dir / "b_gs0_anchored_plus_free_gaussians.ply", anchored_plus_free)

    constrained_metrics = render_gaussian_set(
        "constrained_only",
        constrained,
        views,
        cameras,
        target_size=args.target_size,
        point_radius=args.point_radius,
        output_dir=args.output_dir,
    )
    combined_metrics = render_gaussian_set(
        "anchored_plus_free",
        anchored_plus_free,
        views,
        cameras,
        target_size=args.target_size,
        point_radius=args.point_radius,
        output_dir=args.output_dir,
    )
    free_metrics = render_gaussian_set(
        "free_only",
        free,
        views,
        cameras,
        target_size=args.target_size,
        point_radius=args.point_radius,
        output_dir=args.output_dir,
    )

    comparison = {
        "combined_minus_constrained_iou": float(combined_metrics["mean_iou"] - constrained_metrics["mean_iou"]),
        "combined_minus_constrained_recall": float(combined_metrics["mean_target_recall"] - constrained_metrics["mean_target_recall"]),
        "combined_minus_constrained_overfill": float(combined_metrics["mean_overfill_ratio"] - constrained_metrics["mean_overfill_ratio"]),
        "combined_rgb_better_than_constrained": bool(combined_metrics["mean_rgb_residual"] < constrained_metrics["mean_rgb_residual"]),
        "constrained": constrained_metrics,
        "anchored_plus_free": combined_metrics,
        "free_only": free_metrics,
    }
    improves_geometry = (
        comparison["combined_minus_constrained_iou"] > 0.005
        and comparison["combined_minus_constrained_recall"] > 0.005
        and comparison["combined_minus_constrained_overfill"] < 0.08
    )
    family_summary = {
        "constrained_only": summarize_families(constrained),
        "free_only": summarize_families(free),
        "anchored_plus_free": summarize_families(anchored_plus_free),
    }
    key_outputs = [
        str((args.output_dir / "b_gs0_constrained_only_gaussians.ply").resolve()),
        str((args.output_dir / "b_gs0_free_gaussians.ply").resolve()),
        str((args.output_dir / "b_gs0_anchored_plus_free_gaussians.ply").resolve()),
        str((args.output_dir / "renders").resolve()),
    ]
    decision = (
        "B-GS0 smoke shows bounded geometry improvement over constrained-only, but remains research-only and must pass Open3D/full protocol before any gate."
        if improves_geometry
        else "B-GS0 initialized anchored/free Gaussians and produced review assets, but this smoke does not prove non-template geometry; keep strict gates blocked."
    )
    summary = {
        "status": "research_only_gaussian_smoke_no_export",
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "view_indices": [int(view["view_index"]) for view in views],
            "target_size": int(args.target_size),
        },
        "parameters": {
            "constrained_stride": int(args.constrained_stride),
            "free_per_anchor": int(args.free_per_anchor),
            "max_free_points": int(args.max_free_points),
            "point_radius": int(args.point_radius),
            "seed": int(args.seed),
        },
        "family_summary": family_summary,
        "comparison": comparison,
        "improves_geometry_vs_constrained_only": bool(improves_geometry),
        "decision": decision,
        "key_outputs": key_outputs,
    }
    write_json(args.output_dir / "b_gs0_summary.json", summary)
    write_json(args.output_dir / "b_gs0_render_comparison.json", comparison)
    write_report(args.output_dir / "b_gs0_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "improves_geometry": improves_geometry, "decision": decision}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
