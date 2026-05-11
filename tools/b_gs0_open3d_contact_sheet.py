from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_INPUT_DIR = Path("output/surface_research_preflight_local/B_GS0_smplx_anchored_free_gaussian_smoke")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "open3d_contact_sheet"

PLY_FILES = (
    "b_gs0_constrained_only_gaussians.ply",
    "b_gs0_free_gaussians.ply",
    "b_gs0_anchored_plus_free_gaussians.ply",
)
STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render B-GS0 Gaussian PLYs into Open3D-review contact sheets. "
            "This is visual review only; it does not export a teacher/candidate or write pass state."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_open3d():
    import open3d as o3d

    return o3d


def crop_points(points: np.ndarray, colors: np.ndarray, roi: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if points.shape[0] == 0 or roi == "full":
        return points, colors, {"roi": roi, "points": int(points.shape[0])}
    y = points[:, 1]
    x = points[:, 0]
    z = points[:, 2]
    mask = np.ones((points.shape[0],), dtype=bool)
    if roi == "head_hair":
        mask &= y <= np.quantile(y, 0.22)
    elif roi == "hairline":
        purple = (colors[:, 0] > 0.45) & (colors[:, 2] > 0.55)
        mask &= purple | (y <= np.quantile(y, 0.16))
    elif roi == "hands":
        hand_color = ((colors[:, 2] > 0.65) & (colors[:, 0] < 0.35)) | ((colors[:, 0] > 0.75) & (colors[:, 1] < 0.55))
        lateral = (x <= np.quantile(x, 0.24)) | (x >= np.quantile(x, 0.76)) | (z <= np.quantile(z, 0.22))
        mask &= hand_color | lateral
    elif roi == "clothing":
        green = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.35)
        mask &= green | (y >= np.quantile(y, 0.45))
    if np.count_nonzero(mask) < 32:
        return points, colors, {"roi": roi, "fallback": "too_few_roi_points", "points": int(points.shape[0])}
    return points[mask], colors[mask], {"roi": roi, "points": int(np.count_nonzero(mask))}


def camera_presets() -> list[tuple[str, np.ndarray]]:
    return [
        ("front", np.asarray([0.0, 0.0, 1.0], dtype=np.float64)),
        ("side", np.asarray([1.0, 0.0, 0.0], dtype=np.float64)),
        ("top", np.asarray([0.0, -1.0, 0.0], dtype=np.float64)),
        ("iso", np.asarray([0.7, -0.35, 0.7], dtype=np.float64)),
    ]


def render_open3d(points: np.ndarray, colors: np.ndarray, out_path: Path, *, width: int, height: int, point_size: float) -> bool:
    try:
        o3d = load_open3d()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64).clip(0.0, 1.0))
        center = points.mean(axis=0)
        radius = float(np.linalg.norm(points - center[None, :], axis=1).max())
        radius = max(radius, 1e-3)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="B-GS0 Open3D Contact", width=width, height=height, visible=False)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        opt.point_size = float(point_size)
        opt.light_on = True
        ctr = vis.get_view_control()
        ctr.set_lookat(center.astype(float).tolist())
        ctr.set_up([0.0, -1.0, 0.0])
        # The caller has already rotated points into the desired camera frame.
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_zoom(0.72 if radius < 0.8 else 0.58)
        vis.poll_events()
        vis.update_renderer()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(out_path), do_render=True)
        vis.destroy_window()
        return out_path.is_file() and out_path.stat().st_size > 1000
    except Exception:
        return False


def projection_fallback(points: np.ndarray, colors: np.ndarray, out_path: Path, *, width: int, height: int, direction: np.ndarray) -> None:
    direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up, direction)
    right = right / np.clip(np.linalg.norm(right), 1e-8, None)
    up = np.cross(direction, right)
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    xy = np.stack([centered @ right, centered @ up], axis=1)
    lo = np.quantile(xy, 0.02, axis=0)
    hi = np.quantile(xy, 0.98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo[None, :]) / span[None, :]
    px = np.clip((norm[:, 0] * (width - 1)).round().astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - norm[:, 1]) * (height - 1)).round().astype(np.int64), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    depth = centered @ direction
    order = np.argsort(depth)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    for idx in order:
        x, y = int(px[idx]), int(py[idx])
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = rgb[idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(out_path)


def make_sheet(images: list[Path], out_path: Path, title: str) -> None:
    if not images:
        return
    thumbs = [Image.open(path).convert("RGB").resize((320, 320), Image.Resampling.BICUBIC) for path in images]
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 352 + 36), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 320
        y = 36 + (idx // cols) * 352
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 324), images[idx].stem, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    o3d = load_open3d()
    rendered: list[str] = []
    summaries: dict[str, Any] = {}
    for ply_name in PLY_FILES:
        ply_path = args.input_dir / ply_name
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if points.size == 0:
            summaries[ply_name] = {"points": 0, "empty": True}
            continue
        summaries[ply_name] = {
            "points": int(points.shape[0]),
            "has_colors": bool(pcd.has_colors()),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
        }
        ply_stem = Path(ply_name).stem
        stem_images: list[Path] = []
        for roi in ("full", "head_hair", "hairline", "hands", "clothing"):
            roi_points, roi_colors, roi_summary = crop_points(points, colors, roi)
            summaries.setdefault(ply_name + "_roi", {})[roi] = roi_summary
            for view_name, direction in camera_presets():
                out_path = args.output_dir / ply_stem / roi / f"{view_name}.png"
                ok = render_open3d(roi_points, roi_colors, out_path, width=args.width, height=args.height, point_size=args.point_size)
                if not ok:
                    projection_fallback(roi_points, roi_colors, out_path, width=args.width, height=args.height, direction=direction)
                stem_images.append(out_path)
                rendered.append(str(out_path))
        make_sheet(stem_images, args.output_dir / f"{ply_stem}_contact_sheet.png", title=ply_stem)
    payload = {
        "status": "research_only_open3d_contact_sheet_no_export",
        "strict_facts": STRICT_FACTS,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "rendered": rendered,
        "summaries": summaries,
    }
    (args.output_dir / "b_gs0_open3d_contact_sheet_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready({"status": payload["status"], "rendered": len(rendered), "output_dir": payload["output_dir"]}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
