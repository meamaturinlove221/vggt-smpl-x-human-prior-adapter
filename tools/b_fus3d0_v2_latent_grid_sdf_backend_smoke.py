from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_CACHE = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_LATENT_GRID = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_QUERY_EVIDENCE = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D0_v2_latent_grid_sdf_backend_smoke"
)
DEFAULT_REPORT = REPO_ROOT / "reports/20260507_b_fus3d0_v2_latent_grid_sdf_backend_status.md"


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
    "registry_write": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "v6_backend_implementation": True,
    "backend_smoke": True,
    "writes_new_3d_artifact": True,
    "no_train": True,
    "no_cloud": True,
    "no_teacher": True,
    "no_candidate": True,
    "no_predictions": True,
    "no_checkpoint_write": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "no_b19_b2_b16_b18_tuning": True,
    "writes_predictions_npz": False,
    "writes_teacher": False,
    "writes_candidate": False,
    "writes_strict_registry": False,
}

CONTROL_NAMES = ("real", "shuffle", "zero", "random_view")
FAMILY_COLORS = {
    "full_body": np.asarray([115, 115, 115], dtype=np.uint8),
    "face_core": np.asarray([245, 185, 120], dtype=np.uint8),
    "hairline": np.asarray([165, 80, 220], dtype=np.uint8),
    "left_hand": np.asarray([35, 135, 255], dtype=np.uint8),
    "right_hand": np.asarray([255, 125, 35], dtype=np.uint8),
    "latent_grid": np.asarray([70, 185, 135], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-Fus3D0-v2 latent-grid SDF backend smoke. This uses the existing "
            "VGGT aggregator token cache, query evidence, and latent grid evidence "
            "to produce research-only SDF/occupancy/normal/visibility/confidence "
            "3D artifacts with real/shuffle/zero/random-view controls. It does not "
            "tune B19/B2/B16/B18 and never writes pass/export state."
        )
    )
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--latent-grid", type=Path, default=DEFAULT_LATENT_GRID)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--occupancy-threshold", type=float, default=0.56)
    parser.add_argument("--max-grid-points", type=int, default=5832)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=850)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if math.isfinite(val) else str(val)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def stat_row(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0, "finite": 0}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32)


def control_grid_features(latent: dict[str, np.ndarray], control: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    points = np.asarray(latent["points"], dtype=np.float32)
    occupancy = np.asarray(latent["occupancy_ratio"], dtype=np.float32)
    evidence = np.asarray(latent["evidence_score"], dtype=np.float32)
    token_cosine = np.asarray(latent["token_cosine"], dtype=np.float32)
    visible = np.asarray(latent["visible_count"], dtype=np.float32)
    mask_count = np.asarray(latent["mask_count"], dtype=np.float32)
    rgb_range = np.asarray(latent["rgb_range"], dtype=np.float32)
    rgb_var = np.asarray(latent["rgb_variance"], dtype=np.float32)
    boundary = np.asarray(latent["boundary_like"], dtype=bool)
    if control == "real":
        pass
    elif control == "shuffle":
        perm = rng.permutation(points.shape[0])
        evidence = evidence[perm]
        token_cosine = token_cosine[perm]
        rgb_range = rgb_range[perm]
        rgb_var = rgb_var[perm]
    elif control == "zero":
        evidence = np.zeros_like(evidence)
        token_cosine = np.zeros_like(token_cosine)
        rgb_range = np.zeros_like(rgb_range)
        rgb_var = np.zeros_like(rgb_var)
    elif control == "random_view":
        noise = rng.normal(0.0, 0.12, size=evidence.shape).astype(np.float32)
        evidence = np.roll(evidence, 137) + noise
        token_cosine = np.roll(token_cosine, -83)
        rgb_range = np.roll(rgb_range, 41)
        rgb_var = np.roll(rgb_var, -59)
    else:
        raise ValueError(control)
    return {
        "points": points,
        "occupancy_ratio": occupancy,
        "evidence_score": evidence,
        "token_cosine": token_cosine,
        "visible_count": visible,
        "mask_count": mask_count,
        "rgb_range": rgb_range,
        "rgb_variance": rgb_var,
        "boundary_like": boundary,
    }


def decode_latent_grid_sdf(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    points = np.asarray(features["points"], dtype=np.float32)
    occ = np.asarray(features["occupancy_ratio"], dtype=np.float32)
    evidence = normalize01(np.asarray(features["evidence_score"], dtype=np.float32))
    token_cos = normalize01(np.asarray(features["token_cosine"], dtype=np.float32))
    visible = np.asarray(features["visible_count"], dtype=np.float32) / 6.0
    mask_count = np.asarray(features["mask_count"], dtype=np.float32) / 6.0
    rgb_range = normalize01(np.asarray(features["rgb_range"], dtype=np.float32))
    rgb_var = normalize01(np.asarray(features["rgb_variance"], dtype=np.float32))
    boundary = np.asarray(features["boundary_like"], dtype=bool).astype(np.float32)
    center = np.median(points, axis=0)
    extent = np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)
    rel = (points - center[None, :]) / np.clip(extent[None, :], 1e-6, None)
    radial = np.linalg.norm(rel, axis=1)
    latent = (
        1.35 * occ
        + 0.95 * evidence
        + 0.50 * token_cos
        + 0.35 * visible
        + 0.40 * mask_count
        + 0.28 * boundary
        - 0.35 * rgb_range
        - 0.15 * rgb_var
        - 0.35 * radial
        - 1.10
    )
    occupancy = sigmoid(latent)
    confidence = np.clip(0.35 * visible + 0.35 * evidence + 0.20 * token_cos + 0.10 * mask_count, 0.0, 1.0)
    visibility = np.clip(0.55 * visible + 0.25 * mask_count + 0.20 * evidence, 0.0, 1.0)
    sdf = (0.56 - occupancy).astype(np.float32)
    normals = estimate_normals(points, occupancy)
    normal_residual = (normals * (confidence[:, None] - 0.5) * 0.08).astype(np.float32)
    return {
        "points": points,
        "sdf": sdf.astype(np.float32),
        "occupancy": occupancy.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "visibility": visibility.astype(np.float32),
        "normal": normals.astype(np.float32),
        "normal_residual": normal_residual,
        "latent_score": latent.astype(np.float32),
        "evidence": evidence.astype(np.float32),
        "support": mask_count.astype(np.float32),
    }


def estimate_normals(points: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    scalar = np.asarray(scalar, dtype=np.float32)
    n = int(round(points.shape[0] ** (1.0 / 3.0)))
    if n * n * n == points.shape[0]:
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        field = scalar[order].reshape(n, n, n)
        gx, gy, gz = np.gradient(field)
        grad = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1)
        grad = grad[inv]
    else:
        center = np.median(points, axis=0, keepdims=True)
        grad = points - center
    lens = np.linalg.norm(grad, axis=1, keepdims=True)
    fallback = points - np.median(points, axis=0, keepdims=True)
    grad = np.where(lens > 1e-6, grad, fallback)
    lens = np.linalg.norm(grad, axis=1, keepdims=True)
    return (grad / np.clip(lens, 1e-6, None)).astype(np.float32)


def query_family_metrics(decoded: dict[str, np.ndarray], query: dict[str, np.ndarray]) -> dict[str, Any]:
    points = np.asarray(decoded["points"], dtype=np.float32)
    occupancy = np.asarray(decoded["occupancy"], dtype=np.float32)
    confidence = np.asarray(decoded["confidence"], dtype=np.float32)
    qpos = np.asarray(query["query_positions"], dtype=np.float32)
    qfamilies = np.asarray(query["query_families"]).astype(str)
    if qpos.size == 0:
        return {}
    rows: dict[str, Any] = {}
    chunk = 256
    nearest = np.zeros((qpos.shape[0],), dtype=np.int64)
    for start in range(0, qpos.shape[0], chunk):
        stop = min(start + chunk, qpos.shape[0])
        d2 = ((qpos[start:stop, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        nearest[start:stop] = np.argmin(d2, axis=1)
    for family in sorted(set(qfamilies.tolist())):
        mask = qfamilies == family
        idx = nearest[mask]
        rows[family] = {
            "query_count": int(mask.sum()),
            "mean_nearest_occupancy": float(occupancy[idx].mean()) if idx.size else 0.0,
            "mean_nearest_confidence": float(confidence[idx].mean()) if idx.size else 0.0,
            "occupied_query_ratio": float((occupancy[idx] >= 0.56).mean()) if idx.size else 0.0,
        }
    return rows


def colors_for_points(decoded: dict[str, np.ndarray], occupied: np.ndarray) -> np.ndarray:
    occupancy = np.asarray(decoded["occupancy"], dtype=np.float32)
    confidence = np.asarray(decoded["confidence"], dtype=np.float32)
    visibility = np.asarray(decoded["visibility"], dtype=np.float32)
    colors = np.zeros((occupancy.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip(60 + 175 * occupancy, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(60 + 160 * confidence, 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(75 + 160 * visibility, 0, 255).astype(np.uint8)
    colors[~occupied] = np.asarray([185, 185, 185], dtype=np.uint8)
    return colors


def write_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray, extra: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    extra = np.asarray(extra, dtype=np.float32).reshape(-1) if extra is not None else None
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if extra is not None and extra.shape[0] == points.shape[0]:
            handle.write("property float scalar\n")
        handle.write("end_header\n")
        for idx, (point, color) in enumerate(zip(points, colors, strict=False)):
            line = (
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}"
            )
            if extra is not None and extra.shape[0] == points.shape[0]:
                line += f" {float(extra[idx]):.7f}"
            handle.write(line + "\n")


def projection_fallback(points: np.ndarray, colors: np.ndarray, out_path: Path, *, width: int, height: int, direction: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.size == 0:
        return
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    right = np.cross(up, direction)
    right = right / np.clip(np.linalg.norm(right), 1e-8, None)
    up = np.cross(direction, right)
    centered = points - np.median(points, axis=0, keepdims=True)
    xy = np.stack([centered @ right, centered @ up], axis=1)
    lo = np.quantile(xy, 0.01, axis=0)
    hi = np.quantile(xy, 0.99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo[None, :]) / span[None, :]
    px = np.clip((norm[:, 0] * (width - 1)).round().astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - norm[:, 1]) * (height - 1)).round().astype(np.int64), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    depth = centered @ direction
    for idx in np.argsort(depth):
        x, y = int(px[idx]), int(py[idx])
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = colors[idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(out_path)


def render_review(points: np.ndarray, colors: np.ndarray, output_dir: Path, *, width: int, height: int, point_size: float) -> tuple[list[str], str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        return [], "empty", "no occupied points"
    presets = {
        "front": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
        "side": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "top": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
        "iso": np.asarray([0.65, -0.25, -0.72], dtype=np.float64),
    }
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float64) / 255.0).clip(0.0, 1.0))
        bounds = pcd.get_axis_aligned_bounding_box()
        center = np.asarray(bounds.get_center(), dtype=np.float64)
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window("B-Fus3D0-v2 latent-grid SDF smoke", width=int(width), height=int(height), visible=False)
        if not ok:
            raise RuntimeError("Open3D Visualizer.create_window returned false")
        vis.add_geometry(pcd)
        option = vis.get_render_option()
        option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        option.point_size = float(point_size)
        option.light_on = True
        ctr = vis.get_view_control()
        saved: list[str] = []
        for name, direction in presets.items():
            ctr.set_front(direction.tolist())
            ctr.set_up([0.0, -1.0, 0.0] if name != "top" else [0.0, 0.0, -1.0])
            ctr.set_lookat(center.tolist())
            ctr.set_zoom(0.62)
            vis.poll_events()
            vis.update_renderer()
            path = output_dir / f"{name}.png"
            vis.capture_screen_image(str(path), do_render=True)
            saved.append(str(path))
        vis.destroy_window()
        return saved, "open3d_visualizer", None
    except Exception as exc:  # pragma: no cover - GUI/runtime dependent
        saved = []
        for name, direction in presets.items():
            path = output_dir / f"{name}.png"
            projection_fallback(points, colors, path, width=width, height=height, direction=direction)
            saved.append(str(path))
        return saved, "projection_fallback", repr(exc)


def make_contact_sheet(image_paths: list[str], out_path: Path, title: str) -> None:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for item in image_paths:
        path = Path(item)
        if path.is_file():
            thumbs.append(Image.open(path).convert("RGB").resize((280, 280), Image.Resampling.BICUBIC))
            labels.append(path.parent.name + "/" + path.stem)
    if not thumbs:
        return
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 280, rows * 312 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 280
        y = 34 + (idx // cols) * 312
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 286), labels[idx][:40], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D0-v2 Latent Grid SDF Backend Smoke",
        "",
        "Status: `research_only_backend_smoke_no_export`",
        "",
        "This run creates real/shuffle/zero/random-view latent-grid SDF artifacts from the local VGGT token cache and latent-grid evidence.",
        "It is not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        "formal cloud train/infer/export = blocked",
        "teacher/candidate/predictions/registry = none",
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Controls",
        "",
        "| control | occupied points | occupied ratio | mean occupancy | mean confidence | query occupied mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CONTROL_NAMES:
        row = summary["controls"][name]
        lines.append(
            f"| `{name}` | {row['occupied_points']} | {row['occupied_ratio']:.4f} | "
            f"{row['occupancy_stats']['mean']:.4f} | {row['confidence_stats']['mean']:.4f} | "
            f"{row['query_family_mean_occupied_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Real vs Controls",
        "",
        "```json",
        json.dumps(summary["comparison"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
        f"- output_dir: `{summary['outputs']['output_dir']}`",
        f"- contact_sheet: `{summary['outputs']['contact_sheet']}`",
        f"- field_npz: `{summary['outputs']['field_npz']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    token_cache = load_npz(args.token_cache)
    latent_grid = load_npz(args.latent_grid)
    query_evidence = load_npz(args.query_evidence)
    if "tokens" not in token_cache:
        raise KeyError("token cache missing tokens")
    if "points" not in latent_grid:
        raise KeyError("latent grid missing points")

    rng = np.random.default_rng(20260507)
    controls: dict[str, Any] = {}
    field_arrays: dict[str, np.ndarray] = {}
    render_images: list[str] = []
    threshold = float(args.occupancy_threshold)
    for control in CONTROL_NAMES:
        features = control_grid_features(latent_grid, control, rng)
        decoded = decode_latent_grid_sdf(features)
        occupied = np.asarray(decoded["occupancy"], dtype=np.float32) >= threshold
        colors = colors_for_points(decoded, occupied)
        all_ply = output_dir / f"b_fus3d0_v2_{control}_latent_grid_field_all_points.ply"
        occ_ply = output_dir / f"b_fus3d0_v2_{control}_latent_grid_sdf_occupied_points.ply"
        write_pointcloud_ply(all_ply, decoded["points"], colors, decoded["sdf"])
        occ_points = decoded["points"][occupied]
        occ_colors = colors[occupied]
        write_pointcloud_ply(occ_ply, occ_points, occ_colors, decoded["occupancy"][occupied])
        renders, render_mode, render_error = render_review(
            occ_points,
            occ_colors,
            output_dir / f"{control}_open3d_review",
            width=int(args.width),
            height=int(args.height),
            point_size=float(args.point_size),
        )
        render_images.extend(renders)
        family_metrics = query_family_metrics(decoded, query_evidence)
        family_mean = (
            float(np.mean([row["occupied_query_ratio"] for row in family_metrics.values()]))
            if family_metrics
            else 0.0
        )
        controls[control] = {
            "control": control,
            "all_points_ply": str(all_ply),
            "occupied_points_ply": str(occ_ply),
            "point_count": int(decoded["points"].shape[0]),
            "occupied_points": int(np.count_nonzero(occupied)),
            "occupied_ratio": float(np.mean(occupied)),
            "occupancy_stats": stat_row(decoded["occupancy"]),
            "sdf_stats": stat_row(decoded["sdf"]),
            "confidence_stats": stat_row(decoded["confidence"]),
            "visibility_stats": stat_row(decoded["visibility"]),
            "query_family_metrics": family_metrics,
            "query_family_mean_occupied_ratio": family_mean,
            "render_mode": render_mode,
            "render_error": render_error,
            "renders": renders,
        }
        for key in ("sdf", "occupancy", "confidence", "visibility", "normal_residual", "latent_score"):
            field_arrays[f"{control}_{key}"] = np.asarray(decoded[key])
    contact_sheet = output_dir / "b_fus3d0_v2_open3d_contact_sheet.png"
    make_contact_sheet(render_images, contact_sheet, "B-Fus3D0-v2 latent-grid SDF backend smoke")
    field_npz = output_dir / "b_fus3d0_v2_latent_grid_sdf_fields.npz"
    np.savez_compressed(
        field_npz,
        points=np.asarray(latent_grid["points"], dtype=np.float32),
        selected_view_indices=np.asarray(latent_grid["selected_view_indices"], dtype=np.int32),
        token_shape=np.asarray(token_cache["tokens"].shape, dtype=np.int64),
        **field_arrays,
    )
    real = controls["real"]
    comparison = {
        "real_minus_shuffle_occupied_ratio": float(real["occupied_ratio"] - controls["shuffle"]["occupied_ratio"]),
        "real_minus_zero_occupied_ratio": float(real["occupied_ratio"] - controls["zero"]["occupied_ratio"]),
        "real_minus_random_view_occupied_ratio": float(real["occupied_ratio"] - controls["random_view"]["occupied_ratio"]),
        "real_minus_shuffle_query_occupied": float(real["query_family_mean_occupied_ratio"] - controls["shuffle"]["query_family_mean_occupied_ratio"]),
        "real_minus_zero_query_occupied": float(real["query_family_mean_occupied_ratio"] - controls["zero"]["query_family_mean_occupied_ratio"]),
        "real_minus_random_view_query_occupied": float(real["query_family_mean_occupied_ratio"] - controls["random_view"]["query_family_mean_occupied_ratio"]),
        "real_confidence_minus_shuffle": float(real["confidence_stats"]["mean"] - controls["shuffle"]["confidence_stats"]["mean"]),
        "real_confidence_minus_zero": float(real["confidence_stats"]["mean"] - controls["zero"]["confidence_stats"]["mean"]),
        "real_confidence_minus_random_view": float(real["confidence_stats"]["mean"] - controls["random_view"]["confidence_stats"]["mean"]),
    }
    real_beats_controls = bool(
        comparison["real_confidence_minus_zero"] > 0.05
        and comparison["real_confidence_minus_shuffle"] > 0.0
        and comparison["real_minus_zero_query_occupied"] > 0.02
        and comparison["real_minus_shuffle_query_occupied"] > 0.0
    )
    success = False
    if real_beats_controls and real["occupied_points"] > 256:
        decision = (
            "RESEARCH_ONLY_SMOKE_PROGRESS: B-Fus3D0-v2 produced latent-grid SDF/occupancy artifacts and the real "
            "control improved over zero/shuffle on confidence/query occupancy, but no strict pass is written."
        )
    else:
        decision = (
            "FAIL: B-Fus3D0-v2 produced real/shuffle/zero/random-view latent-grid SDF artifacts, but the deterministic "
            "smoke does not establish a strict real-control win or visible non-template surface quality. Keep gate red."
        )
    summary = {
        "task": "b_fus3d0_v2_latent_grid_sdf_backend_smoke",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_backend_smoke_no_export",
        "truthful_status": "backend_smoke_not_candidate_not_teacher",
        "success": success,
        "pass": False,
        **STRICT_FACTS,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "contract": CONTRACT,
        "inputs": {
            "token_cache": str(args.token_cache.resolve()),
            "latent_grid": str(args.latent_grid.resolve()),
            "query_evidence": str(args.query_evidence.resolve()),
            "occupancy_threshold": threshold,
        },
        "observed_inputs": {
            "tokens_shape": list(token_cache["tokens"].shape),
            "patch_start_idx": np.asarray(token_cache.get("patch_start_idx", [])).tolist(),
            "selected_view_indices": np.asarray(token_cache.get("selected_view_indices", latent_grid.get("selected_view_indices", []))).tolist(),
            "latent_grid_points": int(np.asarray(latent_grid["points"]).shape[0]),
            "query_count": int(np.asarray(query_evidence["query_positions"]).shape[0]) if "query_positions" in query_evidence else 0,
            "query_families": sorted(set(np.asarray(query_evidence.get("query_families", np.asarray([], dtype=str))).astype(str).tolist())),
        },
        "controls": controls,
        "comparison": comparison,
        "real_beats_controls": real_beats_controls,
        "outputs": {
            "output_dir": str(output_dir),
            "summary_json": str((output_dir / "b_fus3d0_v2_latent_grid_sdf_backend_summary.json").resolve()),
            "report_md": str((output_dir / "b_fus3d0_v2_latent_grid_sdf_backend_report.md").resolve()),
            "field_npz": str(field_npz),
            "contact_sheet": str(contact_sheet),
        },
        "decision": decision,
        "blockers": [
            "Deterministic latent-grid decoder is a smoke, not a trained Fus3D backend.",
            "No strict visual full/head/face/hairline/hands pass.",
            "No teacher/candidate/predictions/export/cloud artifacts.",
            "If real does not robustly beat shuffle/zero after learned 2D-to-3D attention, freeze this recipe.",
        ],
    }
    write_json(output_dir / "b_fus3d0_v2_latent_grid_sdf_backend_summary.json", summary)
    write_markdown(output_dir / "b_fus3d0_v2_latent_grid_sdf_backend_report.md", summary)
    write_markdown(args.status_report.resolve(), summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "real_beats_controls": real_beats_controls,
                "success": success,
                "output_dir": str(output_dir),
                "decision": decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
