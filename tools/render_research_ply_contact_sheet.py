from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a research PLY mesh/point cloud into crash-safe projection contact sheets. "
            "This is visual review support only; it is not a strict pass writer."
        )
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--max-points", type=int, default=350000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        line = handle.readline().strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = 0
        face_count = 0
        properties: list[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
            elif stripped.startswith("element face"):
                face_count = int(stripped.split()[-1])
                in_vertex = False
            elif stripped.startswith("property") and in_vertex:
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = np.full((vertex_count, 3), 180, dtype=np.uint8)
        prop_to_index = {name: idx for idx, name in enumerate(properties)}
        has_color = all(key in prop_to_index for key in ("red", "green", "blue"))
        for idx in range(vertex_count):
            parts = handle.readline().strip().split()
            vertices[idx] = [
                float(parts[prop_to_index.get("x", 0)]),
                float(parts[prop_to_index.get("y", 1)]),
                float(parts[prop_to_index.get("z", 2)]),
            ]
            if has_color:
                colors[idx] = [
                    int(float(parts[prop_to_index["red"]])),
                    int(float(parts[prop_to_index["green"]])),
                    int(float(parts[prop_to_index["blue"]])),
                ]
        faces = np.zeros((face_count, 3), dtype=np.int64)
        for idx in range(face_count):
            parts = handle.readline().strip().split()
            if not parts:
                continue
            n = int(parts[0])
            if n < 3:
                faces[idx] = 0
            else:
                faces[idx] = [int(parts[1]), int(parts[2]), int(parts[3])]
    return vertices, faces, colors


def mesh_components(vertex_count: int, faces: np.ndarray) -> dict[str, Any]:
    if faces.size == 0:
        return {
            "component_count": int(vertex_count > 0),
            "largest_component_vertices": int(vertex_count),
            "largest_component_ratio": 1.0 if vertex_count else 0.0,
        }
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    valid_faces = faces[(faces >= 0).all(axis=1) & (faces < vertex_count).all(axis=1)]
    for tri in valid_faces:
        a, b, c = [int(v) for v in tri]
        adjacency[a].extend([b, c])
        adjacency[b].extend([a, c])
        adjacency[c].extend([a, b])
    seen = np.zeros((vertex_count,), dtype=bool)
    sizes: list[int] = []
    for start in range(vertex_count):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for nxt in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        sizes.append(size)
    sizes.sort(reverse=True)
    return {
        "component_count": len(sizes),
        "largest_component_vertices": int(sizes[0]) if sizes else 0,
        "largest_component_ratio": float(sizes[0] / max(1, vertex_count)) if sizes else 0.0,
        "component_sizes_top8": [int(v) for v in sizes[:8]],
    }


def projection_axes(name: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - np.median(points, axis=0, keepdims=True)
    if name == "front":
        return centered[:, 0], -centered[:, 1], centered[:, 2]
    if name == "back":
        return -centered[:, 0], -centered[:, 1], -centered[:, 2]
    if name == "left":
        return centered[:, 2], -centered[:, 1], -centered[:, 0]
    if name == "right":
        return -centered[:, 2], -centered[:, 1], centered[:, 0]
    if name == "top":
        return centered[:, 0], centered[:, 2], -centered[:, 1]
    if name == "iso":
        # A light-weight oblique projection.
        x = 0.82 * centered[:, 0] - 0.42 * centered[:, 2]
        y = -centered[:, 1] + 0.18 * centered[:, 0] + 0.18 * centered[:, 2]
        depth = 0.38 * centered[:, 0] + 0.70 * centered[:, 2] - 0.18 * centered[:, 1]
        return x, y, depth
    raise ValueError(name)


def render_projection(
    points: np.ndarray,
    colors: np.ndarray,
    view: str,
    size: int,
    title: str,
) -> Image.Image:
    x, y, depth = projection_axes(view, points)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
    x = x[finite]
    y = y[finite]
    depth = depth[finite]
    rgb = colors[finite]
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if x.size == 0:
        return Image.fromarray(canvas)
    x_lo, x_hi = np.percentile(x, [1.0, 99.0])
    y_lo, y_hi = np.percentile(y, [1.0, 99.0])
    span = max(float(x_hi - x_lo), float(y_hi - y_lo), 1e-6)
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    pad = 0.16 * span
    x_lo, x_hi = cx - 0.5 * span - pad, cx + 0.5 * span + pad
    y_lo, y_hi = cy - 0.5 * span - pad, cy + 0.5 * span + pad
    px = np.clip(((x - x_lo) / max(1e-6, x_hi - x_lo) * (size - 1)).round().astype(np.int32), 0, size - 1)
    py = np.clip(((1.0 - (y - y_lo) / max(1e-6, y_hi - y_lo)) * (size - 1)).round().astype(np.int32), 0, size - 1)
    order = np.argsort(depth)
    canvas[py[order], px[order]] = rgb[order]
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    label = f"{title} {view}".strip()
    draw.rectangle((0, 0, min(size, 12 + 7 * len(label)), 28), fill=(255, 255, 255))
    draw.text((8, 8), label, fill=(0, 0, 0))
    return image


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Research PLY Contact Sheet",
        "",
        "This is a crash-safe visual review helper. It does not write strict pass state.",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, colors = read_ascii_ply(args.ply)
    if int(args.max_points) > 0 and vertices.shape[0] > int(args.max_points):
        rng = np.random.default_rng(20260506)
        keep = np.sort(rng.choice(vertices.shape[0], size=int(args.max_points), replace=False))
        render_vertices = vertices[keep]
        render_colors = colors[keep]
    else:
        render_vertices = vertices
        render_colors = colors
    images: list[str] = []
    views = ["front", "back", "left", "right", "top", "iso"]
    for view in views:
        image = render_projection(render_vertices, render_colors, view, int(args.size), args.title)
        out = output_dir / f"{view}.png"
        image.save(out)
        images.append(str(out))
    thumbs = [Image.open(path).convert("RGB") for path in images]
    sheet = Image.new("RGB", (int(args.size) * 3, int(args.size) * 2), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % 3) * int(args.size), (idx // 3) * int(args.size)))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    bbox_min = vertices.min(axis=0) if vertices.size else np.zeros(3, dtype=np.float32)
    bbox_max = vertices.max(axis=0) if vertices.size else np.zeros(3, dtype=np.float32)
    extent = bbox_max - bbox_min
    summary = {
        "research_only": True,
        "strict_pass_write": False,
        "ply": str(args.ply.resolve()),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "extent": [float(v) for v in extent],
        "mesh_components": mesh_components(int(vertices.shape[0]), faces),
        "outputs": images + [str(sheet_path)],
    }
    summary_path = output_dir / "ply_contact_sheet_summary.json"
    report_path = output_dir / "ply_contact_sheet_summary.md"
    summary["outputs"].extend([str(summary_path), str(report_path)])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
