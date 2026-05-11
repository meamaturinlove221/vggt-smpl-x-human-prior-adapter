from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PLY point cloud with Open3D offscreen screenshots.")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="open3d")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=2.0)
    return parser.parse_args()


def paste_grid(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths if path.is_file()]
    if not images:
        return
    w = max(image.width for image in images)
    h = max(image.height for image in images)
    canvas = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for idx, image in enumerate(images):
        canvas.paste(image.resize((w, h)), (idx * w, 0))
    canvas.save(output)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(args.input_ply))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise RuntimeError(f"Empty point cloud: {args.input_ply}")
    center = pcd.get_center()
    extent = pcd.get_max_bound() - pcd.get_min_bound()
    diag = float(np.linalg.norm(extent))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(diag * 0.08, 1e-3), origin=center)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="V11 Open3D PLY Render", width=args.width, height=args.height, visible=False)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    views = {
        "front": ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
        "side": ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        "top": ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        "iso": ([0.6, -0.4, -0.7], [0.0, -1.0, 0.0]),
    }
    saved: list[Path] = []
    ctr = vis.get_view_control()
    for name, (front, up) in views.items():
        ctr.set_lookat(center)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(0.72)
        vis.poll_events()
        vis.update_renderer()
        out_path = args.output_dir / f"{args.prefix}_{name}.png"
        vis.capture_screen_image(str(out_path), do_render=True)
        saved.append(out_path)
    vis.destroy_window()
    sheet = args.output_dir / f"{args.prefix}_contact_sheet.png"
    paste_grid(saved, sheet)
    summary = {
        "input_ply": str(args.input_ply),
        "point_count": int(len(points)),
        "has_colors": bool(pcd.has_colors()),
        "screenshots": [str(path) for path in saved],
        "contact_sheet": str(sheet),
    }
    (args.output_dir / f"{args.prefix}_open3d_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
