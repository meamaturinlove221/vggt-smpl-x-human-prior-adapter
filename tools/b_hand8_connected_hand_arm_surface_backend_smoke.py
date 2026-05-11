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
DEFAULT_TEMPLATE_PAYLOAD = REPO_ROOT / (
    "output/normal_line_multiview_20260506/"
    "connected_surface_template_v28_semantic_detail_mouth_nose_fingers/"
    "connected_human_surface_template_payload.npz"
)
DEFAULT_TEMPLATE_SUMMARY = DEFAULT_TEMPLATE_PAYLOAD.with_name("connected_human_surface_template_summary.json")
DEFAULT_HAND_ANCHOR_SUMMARY = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand3_smplx_wrist_arm_connected_precheck_hybrid6/"
    "raw_smplx_mesh_hand_anchor_summary.json"
)
DEFAULT_DEPTH_SUMMARY = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand6_colmap_depth_evidence_probe_hybrid12/"
    "b_hand_colmap_depth_evidence_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand8_connected_hand_arm_surface_backend_smoke"
)
DEFAULT_REPORT = REPO_ROOT / "reports/20260507_b_hand8_connected_hand_arm_backend_status.md"


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
    "backend_smoke": True,
    "writes_new_3d_artifact": True,
    "no_train": True,
    "no_cloud": True,
    "no_teacher": True,
    "no_candidate": True,
    "no_predictions": True,
    "no_registry": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_teacher": False,
    "writes_candidate": False,
    "writes_strict_registry": False,
    "template_or_smplx_scaffold_is_not_success": True,
    "weak_depth_or_landmark_evidence_is_not_success": True,
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
SIDE_COLORS = {
    "left": np.asarray([35, 135, 255], dtype=np.uint8),
    "right": np.asarray([255, 125, 35], dtype=np.uint8),
}
COLORS = {
    "forearm": np.asarray([80, 170, 130], dtype=np.uint8),
    "wrist": np.asarray([255, 220, 85], dtype=np.uint8),
    "outer_shell": np.asarray([185, 95, 230], dtype=np.uint8),
    "finger_seed": np.asarray([30, 30, 30], dtype=np.uint8),
    "body": np.asarray([170, 170, 170], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-hand8 connected hand-arm local surface backend smoke. "
            "It builds new Open3D-readable local hand+wrist+forearm surface artifacts "
            "from the existing connected template, weak hand anchors, and depth evidence. "
            "It never exports a teacher/candidate/predictions or writes strict pass state."
        )
    )
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--template-summary", type=Path, default=DEFAULT_TEMPLATE_SUMMARY)
    parser.add_argument("--hand-anchor-summary", type=Path, default=DEFAULT_HAND_ANCHOR_SUMMARY)
    parser.add_argument("--depth-summary", type=Path, default=DEFAULT_DEPTH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--body-expansion-rings", type=int, default=5)
    parser.add_argument("--outer-offset", type=float, default=0.012)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=850)
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


def load_template(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
        faces = np.asarray(payload["hybrid_faces"], dtype=np.int32)
        part_ids = np.asarray(payload["part_ids"], dtype=np.int64).reshape(-1)
        masks: dict[str, np.ndarray] = {}
        for key in ("left_hand_vertex_mask", "right_hand_vertex_mask"):
            if key in payload.files and payload[key].shape[0] == vertices.shape[0]:
                masks[key] = np.asarray(payload[key], dtype=bool).reshape(-1)
        seed_vertices = (
            np.asarray(payload["semantic_detail_seed_vertices"], dtype=np.int64).reshape(-1)
            if "semantic_detail_seed_vertices" in payload.files
            else np.zeros((0,), dtype=np.int64)
        )
    return {
        "vertices": vertices,
        "faces": faces,
        "part_ids": part_ids,
        "semantic_detail_seed_vertices": seed_vertices,
        **masks,
    }


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces].astype(np.float64)
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lens = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.clip(lens[:, None], 1e-8, None)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lens = np.linalg.norm(normals, axis=1)
    normals /= np.clip(lens[:, None], 1e-8, None)
    return normals.astype(np.float32)


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


def expand_body_from_wrist(
    vertices: np.ndarray,
    faces: np.ndarray,
    part_ids: np.ndarray,
    hand_mask: np.ndarray,
    rings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    body_mask = part_ids == 0
    face_hand_hits = hand_mask[faces]
    face_body_hits = body_mask[faces]
    seam_face_mask = face_hand_hits.any(axis=1) & face_body_hits.any(axis=1) & (~face_hand_hits).any(axis=1)
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
    return wrist_mask, forearm_mask, seam_face_mask, expansion_counts, selected_body


def make_submesh(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_faces = faces[np.asarray(face_mask, dtype=bool)]
    if selected_faces.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )
    used = np.unique(selected_faces.reshape(-1))
    inverse = {int(old): idx for idx, old in enumerate(used.tolist())}
    local_faces = np.asarray([[inverse[int(v)] for v in face] for face in selected_faces], dtype=np.int32)
    return vertices[used].astype(np.float32), local_faces, colors[used].astype(np.uint8), used.astype(np.int64)


def build_outer_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    hand_vertex_mask_local: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_v = vertices.astype(np.float32)
    base_f = faces.astype(np.int32)
    base_c = colors.astype(np.uint8)
    if base_v.size == 0 or base_f.size == 0 or not np.any(hand_vertex_mask_local):
        return base_v, base_f, base_c
    hand_used = np.where(hand_vertex_mask_local)[0]
    shell_index = {int(old): idx for idx, old in enumerate(hand_used.tolist())}
    shell_vertices = base_v[hand_used] + normals[hand_used] * float(offset)
    shell_colors = np.tile(COLORS["outer_shell"][None, :], (len(hand_used), 1)).astype(np.uint8)
    hand_face_mask = hand_vertex_mask_local[base_f].all(axis=1)
    shell_faces = []
    for face in base_f[hand_face_mask]:
        shell_faces.append([shell_index[int(face[0])], shell_index[int(face[2])], shell_index[int(face[1])]])
    if shell_faces:
        shell_faces_np = np.asarray(shell_faces, dtype=np.int32) + base_v.shape[0]
        vertices_out = np.concatenate([base_v, shell_vertices.astype(np.float32)], axis=0)
        colors_out = np.concatenate([base_c, shell_colors], axis=0)
        faces_out = np.concatenate([base_f, shell_faces_np], axis=0)
        return vertices_out, faces_out.astype(np.int32), colors_out
    return base_v, base_f, base_c


def semantic_hand_seeds(seed_vertices: np.ndarray, side_mask_global: np.ndarray) -> np.ndarray:
    seed_vertices = np.asarray(seed_vertices, dtype=np.int64)
    if seed_vertices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    valid = (seed_vertices >= 0) & (seed_vertices < side_mask_global.shape[0])
    seed_vertices = seed_vertices[valid]
    return seed_vertices[side_mask_global[seed_vertices]]


def side_quality_from_depth(depth_summary: dict[str, Any], side: str) -> dict[str, Any]:
    side_summary = depth_summary.get("side_summary") if isinstance(depth_summary.get("side_summary"), dict) else {}
    row = side_summary.get(side) if isinstance(side_summary.get(side), dict) else {}
    return {
        "available": bool(depth_summary.get("available")) and bool(row),
        "roi_count": int(row.get("roi_count", 0) or 0),
        "mapped_depth_roi_count": int(row.get("mapped_depth_roi_count", 0) or 0),
        "depth_valid_ratio_total": float(row.get("depth_valid_ratio_total", 0.0) or 0.0),
        "rois_ge_strong_present_ratio": int(row.get("rois_ge_strong_present_ratio", 0) or 0),
    }


def build_side_backend(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals_global: np.ndarray,
    part_ids: np.ndarray,
    template_masks: dict[str, np.ndarray],
    semantic_seed_vertices: np.ndarray,
    side: str,
    rings: int,
    outer_offset: float,
    depth_summary: dict[str, Any],
) -> dict[str, Any]:
    mask_key = f"{side}_hand_vertex_mask"
    hand_mask = template_masks.get(mask_key, part_ids == SIDE_PART_IDS[side]).astype(bool)
    wrist_mask, forearm_mask, seam_face_mask, expansion_counts, selected_body = expand_body_from_wrist(
        vertices, faces, part_ids, hand_mask, rings
    )
    selected_vertex_mask = hand_mask | selected_body
    face_mask = selected_vertex_mask[faces].all(axis=1) & (
        hand_mask[faces].any(axis=1) | selected_body[faces].any(axis=1)
    )
    colors = np.tile(COLORS["body"][None, :], (vertices.shape[0], 1)).astype(np.uint8)
    colors[forearm_mask] = COLORS["forearm"]
    colors[wrist_mask] = COLORS["wrist"]
    colors[hand_mask] = SIDE_COLORS[side]
    side_seed_global = semantic_hand_seeds(semantic_seed_vertices, hand_mask)
    if side_seed_global.size:
        colors[side_seed_global] = COLORS["finger_seed"]

    sub_vertices, sub_faces, sub_colors, used_global = make_submesh(vertices, faces, face_mask, colors)
    sub_normals = normals_global[used_global] if used_global.size else np.zeros_like(sub_vertices, dtype=np.float32)
    local_hand_mask = hand_mask[used_global] if used_global.size else np.zeros((0,), dtype=bool)
    shell_vertices, shell_faces, shell_colors = build_outer_shell(
        sub_vertices, sub_faces, local_hand_mask, sub_normals, sub_colors, outer_offset
    )
    components = connected_component_stats(sub_faces)
    shell_components = connected_component_stats(shell_faces)
    depth_quality = side_quality_from_depth(depth_summary, side)
    hand_bbox = bbox(vertices[hand_mask])
    side_bbox = bbox(vertices[selected_vertex_mask])
    ext = np.asarray(side_bbox["extent"] or [0.0, 0.0, 0.0], dtype=np.float32)
    # Finger visibility here is a backend proxy only. The smoke cannot call it a pass
    # because there is no learned hand decoder or same-frame dense hand surface.
    seed_count = int(side_seed_global.size)
    wrist_connected = bool(
        components["single_component"]
        and int(seam_face_mask.sum()) > 0
        and int(wrist_mask.sum()) > 0
        and int(forearm_mask.sum()) > 0
    )
    depth_range_normal_proxy = bool(depth_quality["mapped_depth_roi_count"] > 0 and depth_quality["depth_valid_ratio_total"] >= 0.20)
    scaffold_only = True
    finger_structure_visible = False
    palm_continuity_proxy = bool(components["single_component"] and int(hand_mask.sum()) > 512 and np.all(ext > 1e-3))
    failure_reasons = []
    if scaffold_only:
        failure_reasons.append("template_or_smplx_scaffold_only")
    if not finger_structure_visible:
        failure_reasons.append("finger_missing_no_learned_decoder")
    if not wrist_connected:
        failure_reasons.append("wrist_disconnected")
    if not palm_continuity_proxy:
        failure_reasons.append("palm_missing_or_fragmented")
    if not depth_range_normal_proxy:
        failure_reasons.append("depth_range_abnormal_or_too_sparse")
    return {
        "side": side,
        "part_id": SIDE_PART_IDS[side],
        "part_name": PART_NAMES[SIDE_PART_IDS[side]],
        "hand_mask_source": mask_key if mask_key in template_masks else f"part_ids == {SIDE_PART_IDS[side]}",
        "hand_mask": hand_mask,
        "wrist_mask": wrist_mask,
        "forearm_mask": forearm_mask,
        "selected_vertex_mask": selected_vertex_mask,
        "face_mask": face_mask,
        "used_global_vertices": used_global,
        "surface_vertices": sub_vertices,
        "surface_faces": sub_faces,
        "surface_colors": sub_colors,
        "shell_vertices": shell_vertices,
        "shell_faces": shell_faces,
        "shell_colors": shell_colors,
        "hand_vertices": int(hand_mask.sum()),
        "selected_vertices": int(selected_vertex_mask.sum()),
        "selected_faces": int(face_mask.sum()),
        "wrist_seed_vertices": int(wrist_mask.sum()),
        "forearm_proxy_vertices": int(forearm_mask.sum()),
        "seam_faces_to_body": int(seam_face_mask.sum()),
        "body_expansion_new_vertices_by_ring": expansion_counts,
        "semantic_finger_seed_vertices": seed_count,
        "hand_bbox": hand_bbox,
        "selected_bbox": side_bbox,
        "components": components,
        "shell_components": shell_components,
        "depth_quality": depth_quality,
        "wrist_connected_to_forearm": wrist_connected,
        "palm_continuity_proxy": palm_continuity_proxy,
        "finger_structure_visible": finger_structure_visible,
        "depth_range_normal_proxy": depth_range_normal_proxy,
        "largest_component_ratio_high": bool(components["largest_component_ratio"] >= 0.95),
        "not_detached_sheet_proxy": bool(wrist_connected and shell_components["largest_component_ratio"] >= 0.90),
        "scaffold_only": scaffold_only,
        "failure_reasons": failure_reasons,
    }


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


def projection_fallback(points: np.ndarray, colors: np.ndarray, out_path: Path, *, width: int, height: int, direction: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    right = np.cross(up, direction)
    right = right / np.clip(np.linalg.norm(right), 1e-8, None)
    up = np.cross(direction, right)
    center = np.median(points, axis=0, keepdims=True)
    centered = points - center
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


def render_open3d_or_projection(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    *,
    width: int,
    height: int,
    point_size: float,
) -> tuple[list[str], str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        return [], "empty", "no points"
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
        ok = vis.create_window(
            window_name="B-hand8 connected hand-arm backend",
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
        for name, direction in presets.items():
            up = [0.0, -1.0, 0.0] if name != "top" else [0.0, 0.0, -1.0]
            ctr.set_front(direction.tolist())
            ctr.set_up(up)
            ctr.set_lookat(center.tolist())
            ctr.set_zoom(0.78)
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
        if not path.is_file():
            continue
        thumbs.append(Image.open(path).convert("RGB").resize((300, 300), Image.Resampling.BICUBIC))
        labels.append(path.parent.name + "/" + path.stem)
    if not thumbs:
        return
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 300, rows * 334 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 300
        y = 34 + (idx // cols) * 334
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 306), labels[idx][:45], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def public_side_summary(side_payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {
        key: value
        for key, value in side_payload.items()
        if key
        not in {
            "hand_mask",
            "wrist_mask",
            "forearm_mask",
            "selected_vertex_mask",
            "face_mask",
            "used_global_vertices",
            "surface_vertices",
            "surface_faces",
            "surface_colors",
            "shell_vertices",
            "shell_faces",
            "shell_colors",
        }
    }
    return blocked


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Hand8 Connected Hand-Arm Surface Backend Smoke",
        "",
        "Status: `research_only_backend_smoke_no_export`",
        "",
        "This run writes new local 3D hand+wrist+forearm surface artifacts for Open3D review.",
        "It is still not a teacher, not a candidate, and not a strict pass because the surface is scaffold-derived and has no learned hand decoder or same-frame dense hand supervision.",
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
        "## Outputs",
        "",
        f"- combined_mesh_ply: `{summary['outputs']['combined_mesh_ply']}`",
        f"- combined_pointcloud_ply: `{summary['outputs']['combined_pointcloud_ply']}`",
        f"- contact_sheet: `{summary['outputs']['contact_sheet']}`",
        "",
        "## Side Checks",
        "",
        "| side | vertices | faces | wrist connected | palm proxy | finger visible | depth proxy | component ratio | reasons |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for side in ("left", "right"):
        row = summary["sides"][side]
        lines.append(
            f"| `{side}` | {row['selected_vertices']} | {row['selected_faces']} | "
            f"{row['wrist_connected_to_forearm']} | {row['palm_continuity_proxy']} | "
            f"{row['finger_structure_visible']} | {row['depth_range_normal_proxy']} | "
            f"{row['components']['largest_component_ratio']:.4f} | `{', '.join(row['failure_reasons'])}` |"
        )
    lines += [
        "",
        "## Blockers",
        "",
    ]
    lines += [f"- {item}" for item in summary["blockers"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_copy(summary: dict[str, Any], output_dir: Path, report_path: Path) -> None:
    write_markdown(output_dir / "b_hand8_connected_hand_arm_surface_backend_report.md", summary)
    write_markdown(report_path, summary)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
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
    semantic_seed_vertices = np.asarray(template["semantic_detail_seed_vertices"], dtype=np.int64)
    normals = vertex_normals(vertices, faces)
    template_summary = load_json(args.template_summary)
    anchor_summary = load_json(args.hand_anchor_summary)
    depth_summary = load_json(args.depth_summary)

    side_payloads = {
        side: build_side_backend(
            vertices,
            faces,
            normals,
            part_ids,
            template_masks,
            semantic_seed_vertices,
            side,
            int(args.body_expansion_rings),
            float(args.outer_offset),
            depth_summary,
        )
        for side in ("left", "right")
    }

    render_images: list[str] = []
    side_outputs: dict[str, Any] = {}
    combined_vertices: list[np.ndarray] = []
    combined_faces: list[np.ndarray] = []
    combined_colors: list[np.ndarray] = []
    vertex_offset = 0
    for side, payload in side_payloads.items():
        surface_vertices = np.asarray(payload["shell_vertices"], dtype=np.float32)
        surface_faces = np.asarray(payload["shell_faces"], dtype=np.int32)
        surface_colors = np.asarray(payload["shell_colors"], dtype=np.uint8)
        mesh_path = output_dir / f"b_hand8_{side}_connected_hand_arm_surface_mesh.ply"
        pcd_path = output_dir / f"b_hand8_{side}_connected_hand_arm_surface_pointcloud.ply"
        write_mesh_ply(mesh_path, surface_vertices, surface_faces, surface_colors)
        points, point_colors = pointcloud_from_mesh(surface_vertices, surface_faces, surface_colors)
        write_pointcloud_ply(pcd_path, points, point_colors)
        renders, render_mode, render_error = render_open3d_or_projection(
            points,
            point_colors,
            output_dir / f"{side}_open3d_review",
            width=int(args.width),
            height=int(args.height),
            point_size=float(args.point_size),
        )
        render_images.extend(renders)
        side_outputs[side] = {
            "mesh_ply": str(mesh_path),
            "pointcloud_ply": str(pcd_path),
            "mesh_vertices": int(surface_vertices.shape[0]),
            "mesh_faces": int(surface_faces.shape[0]),
            "pointcloud_points": int(points.shape[0]),
            "render_dir": str((output_dir / f"{side}_open3d_review").resolve()),
            "render_mode": render_mode,
            "render_error": render_error,
            "renders": renders,
        }
        combined_vertices.append(surface_vertices)
        combined_colors.append(surface_colors)
        combined_faces.append(surface_faces + vertex_offset)
        vertex_offset += surface_vertices.shape[0]

    combined_v = np.concatenate(combined_vertices, axis=0) if combined_vertices else np.zeros((0, 3), dtype=np.float32)
    combined_f = np.concatenate(combined_faces, axis=0) if combined_faces else np.zeros((0, 3), dtype=np.int32)
    combined_c = np.concatenate(combined_colors, axis=0) if combined_colors else np.zeros((0, 3), dtype=np.uint8)
    combined_mesh = output_dir / "b_hand8_combined_connected_hand_arm_surface_mesh.ply"
    combined_pcd = output_dir / "b_hand8_combined_connected_hand_arm_surface_pointcloud.ply"
    write_mesh_ply(combined_mesh, combined_v, combined_f, combined_c)
    combined_points, combined_point_colors = pointcloud_from_mesh(combined_v, combined_f, combined_c)
    write_pointcloud_ply(combined_pcd, combined_points, combined_point_colors)
    combined_renders, combined_render_mode, combined_render_error = render_open3d_or_projection(
        combined_points,
        combined_point_colors,
        output_dir / "combined_open3d_review",
        width=int(args.width),
        height=int(args.height),
        point_size=float(args.point_size),
    )
    render_images.extend(combined_renders)
    contact_sheet = output_dir / "b_hand8_open3d_contact_sheet.png"
    make_contact_sheet(render_images, contact_sheet, "B-hand8 connected hand-arm backend smoke")

    public_sides = {side: public_side_summary(payload) for side, payload in side_payloads.items()}
    hard_checks = {
        "left_hand_connected_to_wrist_forearm": bool(public_sides["left"]["wrist_connected_to_forearm"]),
        "right_hand_connected_to_wrist_forearm": bool(public_sides["right"]["wrist_connected_to_forearm"]),
        "left_palm_continuity": bool(public_sides["left"]["palm_continuity_proxy"]),
        "right_palm_continuity": bool(public_sides["right"]["palm_continuity_proxy"]),
        "left_finger_structure_visible": bool(public_sides["left"]["finger_structure_visible"]),
        "right_finger_structure_visible": bool(public_sides["right"]["finger_structure_visible"]),
        "left_depth_range_normal": bool(public_sides["left"]["depth_range_normal_proxy"]),
        "right_depth_range_normal": bool(public_sides["right"]["depth_range_normal_proxy"]),
        "largest_component_ratio_high": bool(
            public_sides["left"]["largest_component_ratio_high"]
            and public_sides["right"]["largest_component_ratio_high"]
        ),
        "not_smplx_scaffold_only": False,
    }
    success = bool(
        hard_checks["left_hand_connected_to_wrist_forearm"]
        and hard_checks["right_hand_connected_to_wrist_forearm"]
        and hard_checks["left_palm_continuity"]
        and hard_checks["right_palm_continuity"]
        and hard_checks["left_finger_structure_visible"]
        and hard_checks["right_finger_structure_visible"]
        and hard_checks["left_depth_range_normal"]
        and hard_checks["right_depth_range_normal"]
        and hard_checks["largest_component_ratio_high"]
        and hard_checks["not_smplx_scaffold_only"]
    )
    blockers = [
        "B-hand8 wrote new connected hand-arm local surface artifacts, but they are scaffold-derived.",
        "Finger structure is not counted as visible because no learned hand decoder or real multi-view hand SDF/Gaussian field was fit.",
        "Weak landmarks, SMPL-X topology, and bbox/depth evidence remain auxiliary only.",
        "B_hand7_continuous_connected_hand_surface_review is not produced as a pass artifact.",
        "No teacher, candidate, predictions, registry, formal cloud training/inference, or export was created.",
    ]
    decision = (
        "FAIL: B-hand8 produced new Open3D-readable connected hand-arm backend artifacts, "
        "including left/right/combined mesh and pointcloud outputs, but this is still a "
        "research-only scaffold smoke. It does not satisfy the learned connected hand "
        "surface gate because fingers are not reconstructed by a real hand decoder and "
        "the template scaffold remains the dominant support."
    )
    if success:
        decision = (
            "RESEARCH_ONLY_REVIEW_REQUIRED: B-hand8 hard checks passed locally, but no strict pass is written. "
            "A separate strict visual review would still be required."
        )

    summary = {
        "task": "b_hand8_connected_hand_arm_surface_backend_smoke",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_backend_smoke_no_export",
        "truthful_status": "fail_backend_smoke_not_candidate_not_teacher" if not success else "review_required_no_pass_written",
        "success": success,
        "pass": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        **STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "template_payload": str(args.template_payload.resolve()),
            "template_summary": str(args.template_summary.resolve()),
            "hand_anchor_summary": str(args.hand_anchor_summary.resolve()),
            "depth_summary": str(args.depth_summary.resolve()),
            "body_expansion_rings": int(args.body_expansion_rings),
            "outer_offset": float(args.outer_offset),
        },
        "template": {
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "part_counts": {
                PART_NAMES.get(int(pid), str(pid)): int(np.count_nonzero(part_ids == pid))
                for pid in sorted(np.unique(part_ids).tolist())
            },
            "semantic_detail_seed_vertices": int(semantic_seed_vertices.size),
            "template_summary_status": template_summary.get("truthful_status"),
        },
        "upstream_anchor_readout": {
            "available": bool(anchor_summary.get("available")),
            "top_level_pass": bool(anchor_summary.get("pass")) if anchor_summary.get("available") else False,
            "body_gate": anchor_summary.get("body_gate"),
            "hand_gate": anchor_summary.get("hand_gate"),
        },
        "depth_readout": {
            "available": bool(depth_summary.get("available")),
            "aggregate": depth_summary.get("aggregate"),
            "side_summary": depth_summary.get("side_summary"),
        },
        "hard_checks": hard_checks,
        "sides": public_sides,
        "outputs": {
            "summary_json": str((output_dir / "b_hand8_connected_hand_arm_surface_backend_summary.json").resolve()),
            "report_md": str((output_dir / "b_hand8_connected_hand_arm_surface_backend_report.md").resolve()),
            "combined_mesh_ply": str(combined_mesh),
            "combined_pointcloud_ply": str(combined_pcd),
            "combined_render_dir": str((output_dir / "combined_open3d_review").resolve()),
            "combined_render_mode": combined_render_mode,
            "combined_render_error": combined_render_error,
            "combined_renders": combined_renders,
            "contact_sheet": str(contact_sheet),
            "side_outputs": side_outputs,
        },
        "blockers": blockers,
        "decision": decision,
        "b_hand7_continuous_connected_hand_surface_review": {
            "produced": False,
            "reason": "B-hand8 smoke is scaffold-derived and has no learned connected hand-arm surface decoder.",
        },
    }
    write_json(output_dir / "b_hand8_connected_hand_arm_surface_backend_summary.json", summary)
    write_report_copy(summary, output_dir, args.report.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "success": success,
                "decision": decision,
                "output_dir": str(output_dir),
                "contact_sheet": str(contact_sheet),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
