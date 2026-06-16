from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PAYLOAD = REPO_ROOT / (
    "output/normal_line_multiview_20260506/"
    "connected_surface_template_v28_semantic_detail_mouth_nose_fingers/"
    "connected_human_surface_template_payload.npz"
)
DEFAULT_HAND_ANCHOR_SUMMARY = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand3_smplx_wrist_arm_connected_precheck_hybrid6/"
    "raw_smplx_mesh_hand_anchor_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand4_connected_mesh_precheck_hybrid6"
)

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
    "connected_mesh_precheck_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_teacher": True,
    "no_candidate": True,
    "no_predictions": True,
    "no_registry": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_registry_write": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_teacher": False,
    "writes_candidate": False,
    "writes_strict_registry": False,
    "smplx_or_connected_template_hand_is_not_success": True,
}

PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline",
    5: "lower_clothing_proxy",
}

SIDE_PART_IDS = {"left": 1, "right": 2}

COLORS = {
    "left_hand": np.asarray([30, 145, 255], dtype=np.uint8),
    "right_hand": np.asarray([255, 125, 35], dtype=np.uint8),
    "wrist": np.asarray([255, 220, 80], dtype=np.uint8),
    "forearm": np.asarray([95, 185, 145], dtype=np.uint8),
    "body": np.asarray([175, 175, 175], dtype=np.uint8),
    "overlap": np.asarray([190, 110, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only B-hand4 connected mesh precheck. It extracts left/right "
            "hand+wrist/forearm topology from the existing connected template, "
            "writes Open3D-readable PLY proxies and diagnostic renders, and records "
            "a fail/zero-pass summary when the upstream hand gate did not pass. "
            "It never writes predictions, candidates, teachers, or strict registry state."
        )
    )
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--hand-anchor-summary", type=Path, default=DEFAULT_HAND_ANCHOR_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--body-expansion-rings", type=int, default=4)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
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
        return float(value) if math.isfinite(float(value)) else str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def require_allowed_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = DEFAULT_OUTPUT_DIR.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            "This branch is write-locked to "
            f"{allowed}; refusing output-dir {resolved}"
        ) from exc
    return resolved


def load_template(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        vertices_key = "hybrid_vertices" if "hybrid_vertices" in payload.files else "vertices"
        faces_key = "hybrid_faces" if "hybrid_faces" in payload.files else "faces"
        vertices = np.asarray(payload[vertices_key], dtype=np.float32)
        faces = np.asarray(payload[faces_key], dtype=np.int32)
        if "part_ids" in payload.files and payload["part_ids"].shape[0] == vertices.shape[0]:
            part_ids = np.asarray(payload["part_ids"], dtype=np.int64).reshape(-1)
        else:
            part_ids = np.zeros((vertices.shape[0],), dtype=np.int64)
        masks: dict[str, np.ndarray] = {}
        for key in ("left_hand_vertex_mask", "right_hand_vertex_mask"):
            if key in payload.files and payload[key].shape[0] == vertices.shape[0]:
                masks[key] = np.asarray(payload[key], dtype=bool).reshape(-1)
    return {
        "vertices": vertices,
        "faces": faces,
        "part_ids": part_ids,
        "vertices_key": np.asarray(vertices_key),
        "faces_key": np.asarray(faces_key),
        **masks,
    }


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"available": False, "path": str(resolved), "error": "missing"}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"available": False, "path": str(resolved), "error": "not_dict_json"}
    payload["_resolved_path"] = str(resolved)
    payload["available"] = True
    return payload


def bbox(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {"count": 0, "min": None, "max": None, "extent": None}
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "count": int(points.shape[0]),
        "min": [float(v) for v in lo],
        "max": [float(v) for v in hi],
        "extent": [float(v) for v in hi - lo],
    }


def connected_component_stats(faces: np.ndarray) -> dict[str, Any]:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return {
            "components": 0,
            "used_vertices": 0,
            "largest_component_vertices": 0,
            "largest_component_ratio": 0.0,
            "single_component": False,
        }
    used = np.unique(faces.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used.tolist())}
    parent = np.arange(len(used), dtype=np.int64)
    size = np.ones((len(used),), dtype=np.int64)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = int(parent[root])
        while parent[idx] != idx:
            nxt = int(parent[idx])
            parent[idx] = root
            idx = nxt
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for face in faces:
        a, b, c = [remap[int(v)] for v in face]
        union(a, b)
        union(b, c)
        union(c, a)
    roots = np.asarray([find(idx) for idx in range(len(used))], dtype=np.int64)
    _, counts = np.unique(roots, return_counts=True)
    largest = int(counts.max()) if counts.size else 0
    components = int(counts.size)
    return {
        "components": components,
        "used_vertices": int(len(used)),
        "largest_component_vertices": largest,
        "largest_component_ratio": float(largest / max(int(len(used)), 1)),
        "single_component": bool(components == 1),
    }


def build_side_proxy(
    vertices: np.ndarray,
    faces: np.ndarray,
    part_ids: np.ndarray,
    side: str,
    rings: int,
    template_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    part_id = SIDE_PART_IDS[side]
    mask_key = f"{side}_hand_vertex_mask"
    if mask_key in template_masks:
        hand_mask = np.asarray(template_masks[mask_key], dtype=bool).copy()
        hand_mask_source = mask_key
    else:
        hand_mask = part_ids == part_id
        hand_mask_source = f"part_ids == {part_id}"
    body_mask = part_ids == 0
    face_hand_hits = hand_mask[faces]
    face_body_hits = body_mask[faces]
    hand_face_any = face_hand_hits.any(axis=1)
    hand_face_all = face_hand_hits.all(axis=1)
    seam_face_mask = hand_face_any & face_body_hits.any(axis=1) & (~face_hand_hits).any(axis=1)

    wrist_mask = np.zeros((vertices.shape[0],), dtype=bool)
    if np.any(seam_face_mask):
        seam_vertices = faces[seam_face_mask].reshape(-1)
        wrist_mask[seam_vertices] = body_mask[seam_vertices]

    selected_body = wrist_mask.copy()
    frontier = wrist_mask.copy()
    expansion_counts: list[int] = []
    for _ in range(max(0, int(rings))):
        if not np.any(frontier):
            expansion_counts.append(0)
            continue
        neighbor_faces = frontier[faces].any(axis=1)
        candidates = np.unique(faces[neighbor_faces].reshape(-1))
        new_mask = np.zeros_like(selected_body)
        new_mask[candidates] = body_mask[candidates]
        new_mask &= ~selected_body
        selected_body |= new_mask
        frontier = new_mask
        expansion_counts.append(int(new_mask.sum()))

    forearm_mask = selected_body & ~wrist_mask
    selected_vertices = hand_mask | selected_body
    face_mask = selected_vertices[faces].all(axis=1) & (
        hand_mask[faces].any(axis=1) | selected_body[faces].any(axis=1)
    )
    selected_faces = faces[face_mask]
    components = connected_component_stats(selected_faces)
    topology_connected = bool(
        components["single_component"]
        and int(seam_face_mask.sum()) > 0
        and int(wrist_mask.sum()) > 0
        and int(forearm_mask.sum()) > 0
    )
    return {
        "side": side,
        "part_id": part_id,
        "part_name": PART_NAMES[part_id],
        "hand_mask_source": hand_mask_source,
        "hand_mask": hand_mask,
        "wrist_mask": wrist_mask,
        "forearm_mask": forearm_mask,
        "selected_vertex_mask": selected_vertices,
        "face_mask": face_mask,
        "hand_vertices": int(hand_mask.sum()),
        "hand_faces_any": int(hand_face_any.sum()),
        "hand_faces_all": int(hand_face_all.sum()),
        "seam_faces_to_body": int(seam_face_mask.sum()),
        "wrist_seed_vertices": int(wrist_mask.sum()),
        "forearm_proxy_vertices": int(forearm_mask.sum()),
        "body_expansion_rings": int(rings),
        "body_expansion_new_vertices_by_ring": expansion_counts,
        "selected_vertices": int(selected_vertices.sum()),
        "selected_faces": int(face_mask.sum()),
        "selected_bbox": bbox(vertices[selected_vertices]),
        "components": components,
        "topology_connected_to_body_scaffold": topology_connected,
    }


def colorize_vertices(proxy: dict[str, Any], vertex_count: int) -> np.ndarray:
    side = str(proxy["side"])
    colors = np.tile(COLORS["body"][None, :], (vertex_count, 1))
    colors[np.asarray(proxy["forearm_mask"], dtype=bool)] = COLORS["forearm"]
    colors[np.asarray(proxy["wrist_mask"], dtype=bool)] = COLORS["wrist"]
    hand_color_key = "left_hand" if side == "left" else "right_hand"
    colors[np.asarray(proxy["hand_mask"], dtype=bool)] = COLORS[hand_color_key]
    return colors.astype(np.uint8)


def make_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    global_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_faces = np.asarray(faces, dtype=np.int64)[np.asarray(face_mask, dtype=bool)]
    if selected_faces.size == 0:
        empty_v = np.zeros((0, 3), dtype=np.float32)
        empty_f = np.zeros((0, 3), dtype=np.int32)
        empty_c = np.zeros((0, 3), dtype=np.uint8)
        empty_ids = np.zeros((0,), dtype=np.int64)
        return empty_v, empty_f, empty_c, empty_ids
    used = np.unique(selected_faces.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used.tolist())}
    local_faces = np.asarray([[remap[int(v)] for v in face] for face in selected_faces], dtype=np.int32)
    return (
        vertices[used].astype(np.float32),
        local_faces,
        global_colors[used].astype(np.uint8),
        used.astype(np.int64),
    )


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors, strict=False):
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def pointcloud_from_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.uint8)
    if faces.size == 0:
        return vertices, colors
    centers = vertices[faces].mean(axis=1).astype(np.float32)
    center_colors = np.rint(colors[faces].astype(np.float32).mean(axis=1)).astype(np.uint8)
    return np.concatenate([vertices, centers], axis=0), np.concatenate([colors, center_colors], axis=0)


def write_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def projection_fallback(points: np.ndarray, colors: np.ndarray, output_dir: Path, width: int, height: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.size == 0:
        return []
    center = np.median(points, axis=0)
    centered = points - center[None, :]
    views = {
        "front": (centered[:, 0], -centered[:, 1], centered[:, 2]),
        "side": (centered[:, 2], -centered[:, 1], centered[:, 0]),
        "top": (centered[:, 0], centered[:, 2], -centered[:, 1]),
        "iso": (
            0.70 * centered[:, 0] + 0.30 * centered[:, 2],
            -0.82 * centered[:, 1] + 0.18 * centered[:, 2],
            centered[:, 0] + centered[:, 1] + centered[:, 2],
        ),
    }
    saved: list[str] = []
    for name, (axis_x, axis_y, depth) in views.items():
        order = np.argsort(depth)
        x = axis_x[order]
        y = axis_y[order]
        rgb = colors[order]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        rgb = rgb[finite]
        if len(x) == 0:
            continue
        lo_x, hi_x = np.percentile(x, [1.0, 99.0])
        lo_y, hi_y = np.percentile(y, [1.0, 99.0])
        pad_x = max(1e-6, float(hi_x - lo_x) * 0.10)
        pad_y = max(1e-6, float(hi_y - lo_y) * 0.10)
        lo_x -= pad_x
        hi_x += pad_x
        lo_y -= pad_y
        hi_y += pad_y
        px = np.clip(((x - lo_x) / max(1e-6, hi_x - lo_x) * (width - 1)).round().astype(np.int32), 0, width - 1)
        py = np.clip(((1.0 - (y - lo_y) / max(1e-6, hi_y - lo_y)) * (height - 1)).round().astype(np.int32), 0, height - 1)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[py, px] = rgb
        path = output_dir / f"{name}.png"
        Image.fromarray(canvas).save(path)
        saved.append(str(path))
    return saved


def image_has_nonwhite_pixels(path: Path) -> bool:
    try:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return False
    return bool(int(np.any(arr < 248, axis=-1).sum()) >= 16)


def open3d_renders(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    width: int,
    height: int,
    point_size: float,
) -> tuple[list[str], str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.size == 0:
        return [], "empty", "no points"
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0.0, 1.0).astype(np.float64))
        bounds = pcd.get_axis_aligned_bounding_box()
        center = np.asarray(bounds.get_center(), dtype=np.float64)
        extent = np.asarray(bounds.get_extent(), dtype=np.float64)
        radius = float(np.linalg.norm(extent) + 1e-6)
        presets = {
            "front": {"front": [0.0, 0.0, -1.0], "up": [0.0, -1.0, 0.0], "zoom": 0.86},
            "side": {"front": [1.0, 0.0, 0.0], "up": [0.0, -1.0, 0.0], "zoom": 0.84},
            "top": {"front": [0.0, -1.0, 0.0], "up": [0.0, 0.0, -1.0], "zoom": 0.82},
            "iso": {"front": [0.65, -0.25, -0.72], "up": [0.0, -1.0, 0.0], "zoom": 0.78},
        }
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(
            window_name="B-hand connected mesh precheck",
            width=int(width),
            height=int(height),
            visible=False,
        )
        if not ok:
            raise RuntimeError("Open3D Visualizer.create_window returned false")
        vis.add_geometry(pcd)
        option = vis.get_render_option()
        option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        option.point_size = float(point_size)
        option.light_on = True
        ctr = vis.get_view_control()
        saved: list[str] = []
        for name, preset in presets.items():
            lookat = center.copy()
            if name == "iso":
                lookat = center + np.asarray([0.0, -0.03 * radius, 0.03 * radius], dtype=np.float64)
            ctr.set_front(preset["front"])
            ctr.set_lookat(lookat.tolist())
            ctr.set_up(preset["up"])
            ctr.set_zoom(float(preset["zoom"]))
            vis.poll_events()
            vis.update_renderer()
            path = output_dir / f"{name}.png"
            vis.capture_screen_image(str(path), do_render=True)
            saved.append(str(path))
        vis.destroy_window()
        if not any(image_has_nonwhite_pixels(Path(path)) for path in saved):
            raise RuntimeError("Open3D produced blank/white render images")
        return saved, "open3d_visualizer", None
    except Exception as exc:  # pragma: no cover - display/runtime dependent
        fallback = projection_fallback(points, colors, output_dir, int(width), int(height))
        return fallback, "projection_fallback", repr(exc)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Hand4 Connected Mesh Precheck",
        "",
        "Status: `fail_precheck_only_not_candidate_not_teacher`",
        "",
        "This local-only precheck writes Open3D-readable hand+wrist/forearm mesh and pointcloud proxies.",
        "It does not create predictions, a teacher, a candidate, or strict registry state.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        "teacher/candidate/predictions/registry = none",
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Inputs",
        "",
        f"- template_payload: `{summary['inputs']['template_payload']}`",
        f"- hand_anchor_summary: `{summary['inputs']['hand_anchor_summary']}`",
        "",
        "## Upstream Gate Readout",
        "",
        "```json",
        json.dumps(summary["upstream_anchor_readout"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Side Proxies",
        "",
        "| side | selected vertices | selected faces | seam faces | wrist seed | forearm proxy | components | topology connected | gate decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for side in ("left", "right"):
        row = summary["sides"][side]
        lines.append(
            f"| `{side}` | {row['selected_vertices']} | {row['selected_faces']} | "
            f"{row['seam_faces_to_body']} | {row['wrist_seed_vertices']} | "
            f"{row['forearm_proxy_vertices']} | {row['components']['components']} | "
            f"`{row['topology_connected_to_body_scaffold']}` | `fail` |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            for item in value:
                lines.append(f"  - `{item}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "```json",
            json.dumps(summary["contract"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = require_allowed_output(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_template(args.template_payload)
    vertices = np.asarray(template["vertices"], dtype=np.float32)
    faces = np.asarray(template["faces"], dtype=np.int32)
    part_ids = np.asarray(template["part_ids"], dtype=np.int64)
    template_masks = {
        key: np.asarray(value, dtype=bool)
        for key, value in template.items()
        if key.endswith("_hand_vertex_mask")
    }
    anchor = load_json(args.hand_anchor_summary)
    body_gate = anchor.get("body_gate") if isinstance(anchor.get("body_gate"), dict) else {}
    hand_gate = anchor.get("hand_gate") if isinstance(anchor.get("hand_gate"), dict) else {}
    anchor_top_level_pass = bool(anchor.get("pass")) if anchor.get("available") else False
    hand_gate_pass = bool(hand_gate.get("pass")) if hand_gate else False
    body_gate_pass = bool(body_gate.get("pass")) if body_gate else False

    side_proxies = {
        side: build_side_proxy(vertices, faces, part_ids, side, int(args.body_expansion_rings), template_masks)
        for side in ("left", "right")
    }

    side_outputs: dict[str, Any] = {}
    combined_face_mask = np.zeros((faces.shape[0],), dtype=bool)
    combined_colors = np.tile(COLORS["body"][None, :], (vertices.shape[0], 1)).astype(np.uint8)
    for side, proxy in side_proxies.items():
        combined_face_mask |= np.asarray(proxy["face_mask"], dtype=bool)
        side_colors = colorize_vertices(proxy, vertices.shape[0])
        side_vertices, side_faces, side_vertex_colors, _ = make_submesh(
            vertices,
            faces,
            np.asarray(proxy["face_mask"], dtype=bool),
            side_colors,
        )
        side_points, side_point_colors = pointcloud_from_mesh(side_vertices, side_faces, side_vertex_colors)
        mesh_path = output_dir / f"{side}_hand_wrist_forearm_connected_mesh.ply"
        pointcloud_path = output_dir / f"{side}_hand_wrist_forearm_connected_pointcloud.ply"
        write_mesh_ply(mesh_path, side_vertices, side_faces, side_vertex_colors)
        write_pointcloud_ply(pointcloud_path, side_points, side_point_colors)
        renders, render_mode, render_error = open3d_renders(
            side_points,
            side_point_colors,
            output_dir / f"{side}_open3d_proxy",
            int(args.width),
            int(args.height),
            float(args.point_size),
        )
        side_outputs[side] = {
            "mesh_ply": str(mesh_path),
            "pointcloud_ply": str(pointcloud_path),
            "render_dir": str(output_dir / f"{side}_open3d_proxy"),
            "renders": renders,
            "render_mode": render_mode,
            "render_error": render_error,
            "mesh_vertices": int(side_vertices.shape[0]),
            "mesh_faces": int(side_faces.shape[0]),
            "pointcloud_points": int(side_points.shape[0]),
        }
        combined_colors[np.asarray(proxy["forearm_mask"], dtype=bool)] = COLORS["forearm"]
        combined_colors[np.asarray(proxy["wrist_mask"], dtype=bool)] = COLORS["wrist"]
        combined_colors[np.asarray(proxy["hand_mask"], dtype=bool)] = COLORS[f"{side}_hand"]

    overlap_mask = np.asarray(side_proxies["left"]["selected_vertex_mask"], dtype=bool) & np.asarray(
        side_proxies["right"]["selected_vertex_mask"], dtype=bool
    )
    combined_colors[overlap_mask] = COLORS["overlap"]
    combined_vertices, combined_faces, combined_vertex_colors, _ = make_submesh(
        vertices,
        faces,
        combined_face_mask,
        combined_colors,
    )
    combined_points, combined_point_colors = pointcloud_from_mesh(
        combined_vertices,
        combined_faces,
        combined_vertex_colors,
    )
    combined_mesh_path = output_dir / "combined_hand_wrist_forearm_connected_mesh.ply"
    combined_pointcloud_path = output_dir / "combined_hand_wrist_forearm_connected_pointcloud.ply"
    write_mesh_ply(combined_mesh_path, combined_vertices, combined_faces, combined_vertex_colors)
    write_pointcloud_ply(combined_pointcloud_path, combined_points, combined_point_colors)
    combined_renders, combined_render_mode, combined_render_error = open3d_renders(
        combined_points,
        combined_point_colors,
        output_dir / "combined_open3d_proxy",
        int(args.width),
        int(args.height),
        float(args.point_size),
    )

    upstream_anchor_readout = {
        "available": bool(anchor.get("available")),
        "summary_path": anchor.get("_resolved_path") or anchor.get("path"),
        "raw_smplx_anchor_top_level_pass": anchor_top_level_pass,
        "body_gate_pass": body_gate_pass,
        "body_gate": {
            "views_passing_body_anchor": body_gate.get("views_passing_body_anchor"),
            "min_body_views": body_gate.get("min_body_views"),
            "pass": body_gate.get("pass"),
        },
        "hand_gate_pass": hand_gate_pass,
        "hand_gate": {
            "eligible_views_with_hand_candidates": hand_gate.get("eligible_views_with_hand_candidates"),
            "views_passing_raw_hand_anchor": hand_gate.get("views_passing_raw_hand_anchor"),
            "views_with_compact_3d_hand_boxes": hand_gate.get("views_with_compact_3d_hand_boxes"),
            "implausible_hand_boxes": hand_gate.get("implausible_hand_boxes"),
            "min_hand_components": hand_gate.get("min_hand_components"),
            "pass": hand_gate.get("pass"),
        },
    }
    connected_proxy_built = all(
        bool(side_proxies[side]["topology_connected_to_body_scaffold"])
        for side in ("left", "right")
    )
    gate_decision = "fail_hand_gate_not_passed"
    if not anchor.get("available"):
        gate_decision = "fail_missing_upstream_hand_anchor_summary"
    elif hand_gate_pass and anchor_top_level_pass:
        gate_decision = "fail_proxy_only_no_candidate_teacher_or_strict_review"

    summary = {
        **STRICT_FACTS,
        "task": "b_hand4_connected_mesh_precheck",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "truthful_status": "fail_precheck_only_not_candidate_not_teacher",
        "pass": False,
        "gate_decision": gate_decision,
        "connected_proxy_built": bool(connected_proxy_built),
        "contract": CONTRACT,
        "inputs": {
            "template_payload": str(args.template_payload.expanduser().resolve()),
            "hand_anchor_summary": str(args.hand_anchor_summary.expanduser().resolve()),
            "body_expansion_rings": int(args.body_expansion_rings),
        },
        "template": {
            "vertices_key": str(template["vertices_key"]),
            "faces_key": str(template["faces_key"]),
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "part_ids_aligned": bool(part_ids.shape[0] == vertices.shape[0]),
            "part_counts": {
                PART_NAMES.get(int(part_id), str(part_id)): int((part_ids == int(part_id)).sum())
                for part_id in sorted(int(v) for v in set(part_ids.tolist()))
            },
        },
        "upstream_anchor_readout": upstream_anchor_readout,
        "sides": {
            side: {
                key: value
                for key, value in proxy.items()
                if key
                not in {
                    "hand_mask",
                    "wrist_mask",
                    "forearm_mask",
                    "selected_vertex_mask",
                    "face_mask",
                }
            }
            for side, proxy in side_proxies.items()
        },
        "outputs": {
            "summary_json": str(output_dir / "b_hand_connected_mesh_precheck_summary.json"),
            "report_md": str(output_dir / "b_hand_connected_mesh_precheck_report.md"),
            "combined_mesh_ply": str(combined_mesh_path),
            "combined_pointcloud_ply": str(combined_pointcloud_path),
            "combined_render_dir": str(output_dir / "combined_open3d_proxy"),
            "combined_renders": combined_renders,
            "side_outputs": side_outputs,
        },
        "render": {
            "combined_render_mode": combined_render_mode,
            "combined_render_error": combined_render_error,
            "side_render_modes": {
                side: row["render_mode"] for side, row in side_outputs.items()
            },
        },
        "blockers": [
            "B-hand3 upstream raw SMPL-X hand gate did not pass, so this is fail.",
            "Connected template/SMPL-X hand topology is a scaffold only, not hand success.",
            "No teacher, candidate, predictions, registry, training, inference, or cloud export was created.",
            "Open3D-readable connected proxy still requires human visual review before any future gate claim.",
        ],
        "decision": (
            "FAIL: left/right hand+wrist/forearm connected topology proxies were written from the existing "
            "connected template, but the upstream B-hand3 hand_gate is not passing and SMPL-X/connected-template "
            "hand parts are only weak scaffold evidence. strict_candidate_passes=0, strict_teacher_passes=0, "
            "and there is no teacher/candidate/predictions/registry output."
        ),
    }
    summary = json_ready(summary)
    summary_path = output_dir / "b_hand_connected_mesh_precheck_summary.json"
    report_path = output_dir / "b_hand_connected_mesh_precheck_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report_path, summary)
    print(json.dumps({"summary": str(summary_path), "status": summary["truthful_status"], "gate_decision": gate_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
