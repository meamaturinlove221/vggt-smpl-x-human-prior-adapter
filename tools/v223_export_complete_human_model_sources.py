from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
V15 = ROOT / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz"
V23 = ROOT / "output" / "surface_research_preflight_local" / "V23_residual_surface_v2" / "v23_residual_surface_v2_points.npz"
OUT = ROOT / "output" / "mentor_report_v50r2" / "complete_3d_model_sources"
IMG = OUT / "images"
REPORTS = ROOT / "reports"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def font(size: int):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    except Exception:
        return ImageFont.load_default()


def vertex_part_colors(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heuristic mentor-display colors for the complete SMPL-X mesh.

    These are display colors only. They avoid pretending the complete mesh has
    recovered RGB texture while still making the body structure readable.
    """
    v = np.asarray(vertices, dtype=np.float64)
    x = v[:, 0]
    y = v[:, 1]
    yn = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    ax = np.abs(x)
    colors = np.zeros((len(v), 3), dtype=np.uint8)
    labels = np.full(len(v), "body", dtype=object)

    skin = np.array([205, 155, 125], dtype=np.uint8)
    shirt = np.array([42, 42, 38], dtype=np.uint8)
    shorts = np.array([92, 118, 148], dtype=np.uint8)
    shoes = np.array([210, 210, 205], dtype=np.uint8)
    hair = np.array([18, 16, 14], dtype=np.uint8)

    colors[:] = shirt
    labels[:] = "torso"
    colors[(yn > 0.27) & (yn <= 0.43)] = shorts
    labels[(yn > 0.27) & (yn <= 0.43)] = "shorts"
    colors[yn <= 0.27] = skin
    labels[yn <= 0.27] = "legs"
    colors[yn <= 0.08] = shoes
    labels[yn <= 0.08] = "feet"

    arm_like = (ax > np.percentile(ax, 73)) & (yn > 0.28) & (yn < 0.72)
    colors[arm_like] = skin
    labels[arm_like] = "arms_hands"

    head_like = yn > 0.72
    colors[head_like] = skin
    labels[head_like] = "head_face"
    hair_like = yn > 0.84
    colors[hair_like] = hair
    labels[hair_like] = "hairline_head_top"
    return colors, labels


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.uint8)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    )
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header + "\n")
        for p, c in zip(vertices, colors):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def write_ply_points(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header + "\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = np.linalg.norm(np.cross(a, b), axis=1) * 0.5
    area = np.maximum(area, 1e-12)
    prob = area / area.sum()
    idx = rng.choice(len(faces), size=n, replace=True, p=prob)
    tri_sel = tri[idx]
    col_tri = colors[faces[idx]]
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    pts = tri_sel[:, 0] * w0 + tri_sel[:, 1] * w1 + tri_sel[:, 2] * w2
    col = col_tri[:, 0].astype(np.float64) * w0 + col_tri[:, 1].astype(np.float64) * w1 + col_tri[:, 2].astype(np.float64) * w2
    return pts.astype(np.float32), np.clip(col, 0, 255).astype(np.uint8)


def subset_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, keep_vertex: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_face = keep_vertex[faces].all(axis=1)
    sub_faces_old = faces[keep_face]
    used = np.unique(sub_faces_old.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[sub_faces_old], colors[used]


def basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up_global = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if view == "front":
        right = np.array([1.0, 0.0, 0.0])
        up = up_global
        depth = np.array([0.0, 0.0, 1.0])
    elif view == "side":
        right = np.array([0.0, 0.0, 1.0])
        up = up_global
        depth = np.array([1.0, 0.0, 0.0])
    elif view == "top":
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        depth = np.array([0.0, 1.0, 0.0])
    else:
        depth = np.array([0.65, -0.22, 0.72], dtype=np.float64)
        depth /= np.linalg.norm(depth)
        right = np.cross(up_global, depth)
        right /= np.linalg.norm(right)
        up = np.cross(depth, right)
        up /= np.linalg.norm(up)
    return right, up, depth


def render_points(path: Path, title: str, points: np.ndarray, colors: np.ndarray, view: str, size=(1500, 1200), point_radius: float = 1.6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) == 0:
        Image.new("RGB", size, "white").save(path)
        return
    center = np.median(points, axis=0)
    rel = points - center
    right, up, depth_axis = basis(view)
    x = rel @ right
    y = rel @ up
    d = rel @ depth_axis
    xy = np.stack([x, y], axis=1)
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad
    keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
    x = x[keep]
    y = y[keep]
    d = d[keep]
    c = colors[keep].astype(np.float64)
    order = np.argsort(d)
    x, y, d, c = x[order], y[order], d[order], c[order]
    # Mild depth shading makes the image visibly 3D without changing source colors.
    dn = (d - d.min()) / max(float(d.max() - d.min()), 1e-8)
    c = np.clip(c * (0.68 + 0.32 * dn[:, None]), 0, 255).astype(np.uint8)

    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img, "RGBA")
    scale = min((W - 150) / max(float(hi[0] - lo[0]), 1e-8), (H - 170) / max(float(hi[1] - lo[1]), 1e-8))
    px = (x - (lo[0] + hi[0]) / 2.0) * scale + W / 2.0
    py = H / 2.0 - (y - (lo[1] + hi[1]) / 2.0) * scale
    r = point_radius
    for xx, yy, cc in zip(px, py, c):
        draw.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(int(cc[0]), int(cc[1]), int(cc[2]), 220))
    draw.rectangle((0, 0, W, 78), fill=(255, 255, 255, 235))
    draw.text((36, 18), title, fill=(20, 20, 20, 255), font=font(34))
    draw.text((36, H - 46), f"complete SMPL-X mesh sampled point cloud, view={view}, points={len(points):,}", fill=(60, 60, 60, 255), font=font(20))
    img.save(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    with np.load(V15, allow_pickle=True) as z:
        vertices = np.asarray(z["vertices"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int32)

    colors, labels = vertex_part_colors(vertices)
    y = vertices[:, 1]
    x = vertices[:, 0]
    yn = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    ax = np.abs(x)

    full_mesh = OUT / "v50r2_COMPLETE_human_smplx_native_mesh_FULL_SEND.ply"
    write_ply_mesh(full_mesh, vertices, faces, colors)
    full_pts, full_cols = sample_mesh(vertices, faces, colors, 130000, seed=22350)
    full_pointcloud = OUT / "v50r2_COMPLETE_human_smplx_native_pointcloud_FULL_SEND.ply"
    write_ply_points(full_pointcloud, full_pts, full_cols)

    head_keep = yn > 0.70
    head_v, head_f, head_c = subset_mesh(vertices, faces, colors, head_keep)
    head_mesh = OUT / "v50r2_COMPLETE_head_face_smplx_native_mesh_SEND.ply"
    write_ply_mesh(head_mesh, head_v, head_f, head_c)
    head_pts, head_cols = sample_mesh(head_v, head_f, head_c, 32000, seed=22351)
    head_pc = OUT / "v50r2_COMPLETE_head_face_smplx_native_pointcloud_SEND.ply"
    write_ply_points(head_pc, head_pts, head_cols)

    hand_band = (yn > 0.23) & (yn < 0.72) & (ax > np.percentile(ax, 82))
    left_keep = hand_band & (x < 0)
    right_keep = hand_band & (x > 0)
    hand_outputs = []
    for side, keep, seed in [("left", left_keep, 22352), ("right", right_keep, 22353)]:
        hv, hf, hc = subset_mesh(vertices, faces, colors, keep)
        mesh_path = OUT / f"v50r2_COMPLETE_{side}_hand_forearm_smplx_native_mesh_SEND.ply"
        write_ply_mesh(mesh_path, hv, hf, hc)
        if len(hf) > 0:
            hp, hcol = sample_mesh(hv, hf, hc, 22000, seed=seed)
        else:
            hp, hcol = hv, hc
        pc_path = OUT / f"v50r2_COMPLETE_{side}_hand_forearm_smplx_native_pointcloud_SEND.ply"
        write_ply_points(pc_path, hp, hcol)
        hand_outputs.append({"side": side, "mesh": str(mesh_path.resolve()), "pointcloud": str(pc_path.resolve()), "vertices": int(len(hv)), "faces": int(len(hf)), "points": int(len(hp))})

    # Diagnostic residual surface, explicitly not a complete model.
    diagnostic = {}
    if V23.exists():
        with np.load(V23, allow_pickle=True) as z:
            residual_points = np.asarray(z["residual_surface_points"], dtype=np.float32)
            residual_colors = np.asarray(z["colors"], dtype=np.uint8)
        valid = np.isfinite(residual_points).all(axis=1) & (np.linalg.norm(residual_points, axis=1) > 1e-8)
        residual_points = residual_points[valid]
        residual_colors = residual_colors[valid]
        diag_path = OUT / "v50r2_DIAGNOSTIC_residual_visible_surface_points_NOT_COMPLETE_MODEL.ply"
        write_ply_points(diag_path, residual_points, residual_colors)
        diagnostic = {"residual_visible_surface_ply": str(diag_path.resolve()), "points": int(len(residual_points))}

    image_outputs = []
    for view in ["front", "side", "iso", "top"]:
        out = IMG / f"v50r2_complete_human_smplx_native_{view}.png"
        render_points(out, "V50R2 complete SMPL-X native human model", full_pts, full_cols, view=view, point_radius=1.45)
        image_outputs.append(str(out.resolve()))
    for name, pts, cols, views in [
        ("head_face", head_pts, head_cols, ["front", "side", "iso"]),
    ]:
        for view in views:
            out = IMG / f"v50r2_complete_{name}_{view}.png"
            render_points(out, f"V50R2 complete SMPL-X native {name}", pts, cols, view=view, point_radius=1.9)
            image_outputs.append(str(out.resolve()))

    report = {
        "task": "v223_complete_human_model_source_export",
        "created_utc": now(),
        "decision": "Complete 3D mentor-facing source is the SMPL-X native full mesh / mesh-sampled point cloud. Per-view VGGT point maps are visible-surface diagnostics and must not be presented as complete 3D human modeling.",
        "complete_model_policy": {
            "complete_human_mesh": "SMPL-X native vertices/faces from V15, part-colored for mentor display.",
            "complete_human_pointcloud": "Uniform samples from the complete mesh surface.",
            "visible_pointmaps": "diagnostic only; not full-body 3D model sources.",
            "strict_teacher_pass_claimed": False,
        },
        "inputs": {
            "v15_smplx_raster_npz": str(V15.resolve()),
            "v23_residual_surface_npz": str(V23.resolve()) if V23.exists() else None,
        },
        "outputs": {
            "full_mesh": str(full_mesh.resolve()),
            "full_pointcloud": str(full_pointcloud.resolve()),
            "head_face_mesh": str(head_mesh.resolve()),
            "head_face_pointcloud": str(head_pc.resolve()),
            "hand_outputs": hand_outputs,
            "diagnostic": diagnostic,
            "images": image_outputs,
        },
        "counts": {
            "full_vertices": int(len(vertices)),
            "full_faces": int(len(faces)),
            "full_sampled_points": int(len(full_pts)),
            "head_face_vertices": int(len(head_v)),
            "head_face_faces": int(len(head_f)),
            "head_face_points": int(len(head_pts)),
        },
        "bbox": {
            "full_min": [float(x) for x in vertices.min(axis=0)],
            "full_max": [float(x) for x in vertices.max(axis=0)],
            "full_extent": [float(x) for x in (vertices.max(axis=0) - vertices.min(axis=0))],
        },
        "invalidated_prior_outputs": [
            "output/mentor_report_v50r2/pointcloud_sources/invalid_do_not_send/v50r2_human_only_candidate_world_merged6v.ply",
            "output/surface_research_preflight_local/V32_candidate_inference_research/v32_candidate_open3d_review_points.ply",
        ],
    }
    json_path = REPORTS / "20260509_v50r2_complete_human_model_source_export.json"
    md_path = REPORTS / "20260509_v50r2_complete_human_model_source_export.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V50R2 Complete Human Model Source Export",
        "",
        "This fixes the previous visualization/export mistake: per-view VGGT point maps are visible-surface diagnostics, not complete 3D human models.",
        "",
        "Use these files for MeshLab mentor display:",
        f"- Full complete mesh: `{full_mesh.resolve()}`",
        f"- Full complete sampled point cloud: `{full_pointcloud.resolve()}`",
        f"- Head/face complete mesh: `{head_mesh.resolve()}`",
        f"- Head/face complete sampled point cloud: `{head_pc.resolve()}`",
    ]
    for item in hand_outputs:
        lines.append(f"- {item['side']} hand/forearm mesh: `{item['mesh']}`")
        lines.append(f"- {item['side']} hand/forearm sampled point cloud: `{item['pointcloud']}`")
    lines += [
        "",
        "Do not use the old merged/per-view visible-surface PLYs as complete 3D model evidence.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
