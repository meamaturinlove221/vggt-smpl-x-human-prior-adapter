from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.smplx_numpy import build_smplx_vertex_features, resolve_smplx_model_path  # noqa: E402
from v15_common import LOCAL_ROOT, REPORTS, json_ready, scalar_stats, write_json  # noqa: E402


DEFAULT_SMPLX_ROOT = Path("G:/\u6570\u636e\u96c6/datasets/smplx")
DEFAULT_CASE_ROOT = REPO_ROOT / "output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_V15_RASTER_DIR = LOCAL_ROOT / "V15_SMPLX_native_camera_raster_export"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_smplx_native_region_roi_builder"
DEFAULT_JSON = REPORTS / "20260508_v16_smplx_native_region_roi_builder.json"
DEFAULT_MD = REPORTS / "20260508_v16_smplx_native_region_roi_builder.md"

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
SIDE_ORDER = ("left", "right")
CORE_ROIS = (
    "left_hand",
    "right_hand",
    "wrist_bridge_left",
    "wrist_bridge_right",
    "head",
    "face_front",
)

SIDE_PREFIX = {"left": "L", "right": "R"}
FINGER_NATIVE_NAMES = {
    "thumb": ("Thumb1", "Thumb2", "Thumb3"),
    "index": ("Index1", "Index2", "Index3"),
    "middle": ("Middle1", "Middle2", "Middle3"),
    "ring": ("Ring1", "Ring2", "Ring3"),
    "pinky": ("Pinky1", "Pinky2", "Pinky3"),
}

ROI_SAVE_ORDER = (
    "body_visible",
    "left_hand",
    "right_hand",
    "wrist_bridge_left",
    "wrist_bridge_right",
    "thumb_left",
    "index_left",
    "middle_left",
    "ring_left",
    "pinky_left",
    "thumb_right",
    "index_right",
    "middle_right",
    "ring_right",
    "pinky_right",
    "head",
    "face_front",
    "face_lmk_static",
    "face_lmk_dynamic",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_v16_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected = (LOCAL_ROOT / "V16_smplx_native_region_roi_builder").resolve()
    if resolved != expected:
        raise ValueError(f"Refusing output outside V16 ROI ownership: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_npz(path: Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_object_dict(npz_path: Path, key: str) -> dict[str, int]:
    with np.load(npz_path, allow_pickle=True) as payload:
        if key not in payload:
            return {}
        try:
            item = payload[key].item()
        except Exception:
            return {}
    if not isinstance(item, dict):
        return {}
    return {str(k): int(v) for k, v in item.items()}


def index_channels(names: list[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(names)}


def require_channels(channel_index: dict[str, int], names: tuple[str, ...]) -> list[str]:
    return [name for name in names if name not in channel_index]


def make_joint_groups(joint2num: dict[str, int]) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {}
    for side in SIDE_ORDER:
        prefix = SIDE_PREFIX[side]
        wrist = f"{prefix}_Wrist"
        hand_ids = {joint2num[wrist]} if wrist in joint2num else set()
        for finger, pieces in FINGER_NATIVE_NAMES.items():
            ids = {joint2num[f"{prefix}_{piece}"] for piece in pieces if f"{prefix}_{piece}" in joint2num}
            groups[f"{finger}_{side}"] = ids
            hand_ids |= ids
        groups[f"{side}_hand"] = hand_ids
        groups[f"wrist_bridge_{side}"] = {
            joint2num[name]
            for name in (wrist, f"{prefix}_Elbow", f"{prefix}_Shoulder", f"{prefix}_Collar")
            if name in joint2num
        }
    groups["head"] = {joint2num[name] for name in ("Head", "Jaw", "L_Eye", "R_Eye", "Neck") if name in joint2num}
    groups["face_front"] = {joint2num[name] for name in ("Head", "Jaw", "L_Eye", "R_Eye") if name in joint2num}
    return groups


def nearest_vertex_ids(canonical_xyz: np.ndarray, valid_mask: np.ndarray, canonical_tree: cKDTree) -> np.ndarray:
    out = np.full(valid_mask.shape, -1, dtype=np.int32)
    coords = np.asarray(canonical_xyz, dtype=np.float32)[valid_mask]
    if coords.size == 0:
        return out
    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        return out
    ids = np.full((coords.shape[0],), -1, dtype=np.int32)
    _, found = canonical_tree.query(coords[finite], k=1, workers=-1)
    ids[finite] = np.asarray(found, dtype=np.int32)
    out[valid_mask] = ids
    return out


def dilate2d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = grown
    return out


def erode2d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        shrunk = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                shrunk &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = shrunk
    return out


def close2d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return erode2d(dilate2d(mask, iterations=iterations), iterations=iterations)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return comps
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    height, width = mask.shape
    for start_y, start_x in zip(rows, cols):
        if seen[start_y, start_x]:
            continue
        comp = np.zeros_like(mask, dtype=bool)
        q: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        while q:
            y, x = q.popleft()
            comp[y, x] = True
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        comps.append(comp)
    comps.sort(key=lambda item: int(item.sum()), reverse=True)
    return comps


def largest_components(mask: np.ndarray, max_components: int = 2, min_pixels: int = 1) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for comp in connected_components(mask)[: max(1, int(max_components))]:
        if int(comp.sum()) >= int(min_pixels):
            out |= comp
    return out


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - int(pad))
    y0 = max(0, int(y0) - int(pad))
    x1 = min(width, int(x1) + int(pad))
    y1 = min(height, int(y1) + int(pad))
    out = np.zeros((height, width), dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def bbox_row(summary: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    bbox = bbox_from_mask(mask)
    summary["bbox_xyxy"] = list(bbox) if bbox else None
    return summary


def spatial_hand_fallbacks(
    hand_mask: np.ndarray,
    side: str,
    min_pixels: int,
) -> dict[str, np.ndarray]:
    hand_mask = np.asarray(hand_mask, dtype=bool)
    if hand_mask.ndim == 3:
        per_view = [spatial_hand_fallbacks(view, side, min_pixels) for view in hand_mask]
        return {
            f"{finger}_{side}": np.stack([row[f"{finger}_{side}"] for row in per_view], axis=0)
            for finger in FINGER_ORDER
        }
    if hand_mask.ndim != 2:
        raise ValueError(f"spatial_hand_fallbacks expects a 2D mask or [V,H,W] stack, got {hand_mask.shape}")
    hand_mask = largest_components(close2d(hand_mask, iterations=1), max_components=2, min_pixels=1)
    coords = np.column_stack(np.nonzero(hand_mask))
    out = {f"{finger}_{side}": np.zeros_like(hand_mask, dtype=bool) for finger in FINGER_ORDER}
    if coords.size == 0:
        return out
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h = max(1, int(y_max - y_min + 1))
    w = max(1, int(x_max - x_min + 1))
    local_y = (coords[:, 0] - y_min) / float(h)
    local_x = (coords[:, 1] - x_min) / float(w)
    distal = local_y < 0.76
    if side == "left":
        bins = {
            "thumb": local_x < 0.28,
            "index": (local_x >= 0.20) & (local_x < 0.45),
            "middle": (local_x >= 0.38) & (local_x < 0.62),
            "ring": (local_x >= 0.55) & (local_x < 0.80),
            "pinky": local_x >= 0.72,
        }
    else:
        bins = {
            "thumb": local_x >= 0.72,
            "index": (local_x >= 0.55) & (local_x < 0.80),
            "middle": (local_x >= 0.38) & (local_x < 0.62),
            "ring": (local_x >= 0.20) & (local_x < 0.45),
            "pinky": local_x < 0.28,
        }
    for finger, selector in bins.items():
        mask = np.zeros_like(hand_mask, dtype=bool)
        use = selector & distal
        if int(use.sum()) < int(min_pixels):
            use = selector
        mask[coords[use, 0], coords[use, 1]] = True
        out[f"{finger}_{side}"] = mask & hand_mask
    return out


def head_spatial_fallback(silhouette: np.ndarray, hand_mask: np.ndarray, prior_head: np.ndarray | None) -> np.ndarray:
    if prior_head is not None and bool(np.asarray(prior_head, dtype=bool).any()):
        return close2d(np.asarray(prior_head, dtype=bool), iterations=1)
    support = np.asarray(silhouette, dtype=bool) & ~dilate2d(hand_mask, iterations=2)
    coords = np.column_stack(np.nonzero(support))
    if coords.size == 0:
        return np.zeros_like(support, dtype=bool)
    y_min, _x_min = coords.min(axis=0)
    y_max, _x_max = coords.max(axis=0)
    cutoff = int(round(float(y_min) + 0.28 * float(max(1, y_max - y_min + 1))))
    return largest_components(support & (np.indices(support.shape)[0] <= cutoff), max_components=1, min_pixels=1)


def face_spatial_fallback(head_mask: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    head_mask = np.asarray(head_mask, dtype=bool)
    bbox = bbox_from_mask(head_mask)
    if bbox is None:
        return np.zeros_like(head_mask, dtype=bool)
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    inner = bbox_mask(
        head_mask.shape,
        (
            x0 + int(round(0.18 * width)),
            y0 + int(round(0.10 * height)),
            x1 - int(round(0.18 * width)),
            y0 + int(round(0.78 * height)),
        ),
        pad=1,
    )
    return close2d(inner & (head_mask | np.asarray(silhouette, dtype=bool)), iterations=1)


def landmark_vertex_mask(model_path: Path, vertex_count: int, key: str) -> np.ndarray:
    out = np.zeros((vertex_count,), dtype=bool)
    with np.load(model_path, allow_pickle=True) as payload:
        faces = np.asarray(payload["f"], dtype=np.int32)
        if key == "static":
            if "lmk_faces_idx" not in payload:
                return out
            face_ids = np.asarray(payload["lmk_faces_idx"], dtype=np.int64).reshape(-1)
        elif key == "dynamic":
            if "dynamic_lmk_faces_idx" not in payload:
                return out
            face_ids = np.asarray(payload["dynamic_lmk_faces_idx"], dtype=np.int64).reshape(-1)
        else:
            raise ValueError(key)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < faces.shape[0])]
    if face_ids.size:
        out[np.unique(faces[face_ids].reshape(-1))] = True
    return out


def row_stats(name: str, mask: np.ndarray, source: str) -> dict[str, Any]:
    per_view = [int(v) for v in np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1).sum(axis=1)]
    row = {
        "region": name,
        "source": source,
        "total_pixels": int(sum(per_view)),
        "nonempty_views": int(sum(1 for value in per_view if value > 0)),
        "per_view_pixels": per_view,
    }
    return row


def save_contact_sheet(path: Path, images: np.ndarray, roi_maps: dict[str, np.ndarray], view_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "left_hand": np.asarray([255, 84, 84], dtype=np.float32),
        "right_hand": np.asarray([76, 152, 255], dtype=np.float32),
        "wrist_bridge_left": np.asarray([255, 188, 54], dtype=np.float32),
        "wrist_bridge_right": np.asarray([63, 214, 153], dtype=np.float32),
        "head": np.asarray([205, 105, 255], dtype=np.float32),
        "face_front": np.asarray([255, 230, 80], dtype=np.float32),
    }
    thumbs = []
    for view_idx in range(images.shape[0]):
        base = images[view_idx].astype(np.float32).copy()
        for name, color in colors.items():
            mask = roi_maps[name][view_idx].astype(bool)
            base[mask] = 0.48 * base[mask] + 0.52 * color
        pil = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
        pil.thumbnail((260, 260))
        canvas = Image.new("RGB", (260, 284), (16, 16, 16))
        canvas.paste(pil, ((260 - pil.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 264), f"view {view_names[view_idx]}", fill=(230, 230, 230))
        thumbs.append(canvas)
    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / max(cols, 1)))
    sheet = Image.new("RGB", (cols * 260, rows * 284), (10, 10, 10))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 260, (idx // cols) * 284))
    sheet.save(path)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = safe_v16_output_dir(args.output_dir)
    model_path = resolve_smplx_model_path(args.smplx_root, args.gender)
    case_root = args.case_root.expanduser().resolve()
    manifest_path = case_root / "case_manifest.json"
    inputs_path = case_root / "inputs.npz"
    targets_path = case_root / "targets.npz"
    raster_part_path = args.v15_raster_dir / "prior_part_maps.npz"
    blockers: list[str] = []
    for path in (manifest_path, inputs_path, targets_path):
        if not path.is_file():
            blockers.append(f"Missing V15 case file: {path}")
    if blockers:
        return {
            "task": "v16_smplx_native_region_roi_builder",
            "status": "v16_smplx_native_region_roi_builder_blocked",
            "created_utc": utc_now(),
            "blockers": blockers,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = load_npz(inputs_path)
    targets = load_npz(targets_path)
    prior_maps = np.asarray(inputs["prior_maps"], dtype=np.float32)
    prior_mask = np.asarray(inputs.get("prior_mask", np.zeros(prior_maps.shape[:1] + prior_maps.shape[2:], dtype=bool)), dtype=bool)
    point_masks = np.asarray(inputs.get("point_masks", prior_mask), dtype=bool)
    images = np.asarray(inputs.get("images", np.zeros(prior_mask.shape + (3,), dtype=np.uint8)), dtype=np.uint8)
    camera_ids = [str(x) for x in np.asarray(inputs.get("camera_ids", np.arange(prior_maps.shape[0])))]
    channel_names = [str(name) for name in manifest.get("prior_input_meta", {}).get("channel_names", manifest.get("prior_channels", []))]
    if len(channel_names) != prior_maps.shape[1]:
        channel_names = [f"prior_channel_{idx:03d}" for idx in range(prior_maps.shape[1])]
    channel_index = index_channels(channel_names)
    missing = require_channels(
        channel_index,
        ("smplx_canonical_x", "smplx_canonical_y", "smplx_canonical_z", "silhouette"),
    )
    if missing:
        blockers.append(f"V15 prior maps are missing required canonical channels: {missing}")
    if blockers:
        return {
            "task": "v16_smplx_native_region_roi_builder",
            "status": "v16_smplx_native_region_roi_builder_blocked",
            "created_utc": utc_now(),
            "blockers": blockers,
        }

    silhouette = prior_maps[:, channel_index["silhouette"]] > 0.5
    visible_channel = channel_index.get("smplx_visible_mask")
    native_visible = prior_mask.copy()
    if visible_channel is not None:
        native_visible |= prior_maps[:, visible_channel] > 0.5

    vertex_payload = build_smplx_vertex_features(
        model_path=model_path,
        betas=np.zeros((int(args.num_betas),), dtype=np.float32),
        expression=np.zeros((int(args.num_expression),), dtype=np.float32),
        body_part_count=int(args.body_part_count),
    )
    canonical_vertices = np.asarray(vertex_payload["canonical_positions"], dtype=np.float32)
    canonical_tree = cKDTree(canonical_vertices)
    with np.load(model_path, allow_pickle=True) as model_payload:
        weights = np.asarray(model_payload["weights"], dtype=np.float32)
        faces = np.asarray(model_payload["f"], dtype=np.int32)
        native_lmk_keys = sorted([key for key in model_payload.files if "lmk" in key.lower()])
    dominant_joint = weights.argmax(axis=1).astype(np.int32)
    joint2num = load_object_dict(model_path, "joint2num")
    part2num = load_object_dict(model_path, "part2num")
    groups = make_joint_groups(joint2num)

    canonical_xyz = np.stack(
        [
            prior_maps[:, channel_index["smplx_canonical_x"]],
            prior_maps[:, channel_index["smplx_canonical_y"]],
            prior_maps[:, channel_index["smplx_canonical_z"]],
        ],
        axis=-1,
    ).astype(np.float32)
    visible_for_label = native_visible | silhouette
    nearest_ids = nearest_vertex_ids(canonical_xyz, visible_for_label, canonical_tree)
    valid_vertex = nearest_ids >= 0
    dominant_map = np.full(nearest_ids.shape, -1, dtype=np.int16)
    dominant_map[valid_vertex] = dominant_joint[nearest_ids[valid_vertex]].astype(np.int16)

    roi_maps: dict[str, np.ndarray] = {}
    roi_sources: dict[str, str] = {}
    roi_maps["body_visible"] = visible_for_label.astype(bool)
    roi_sources["body_visible"] = "v15_prior_mask_or_silhouette"

    for side in SIDE_ORDER:
        key = f"{side}_hand"
        joints = np.asarray(sorted(groups.get(key, set())), dtype=np.int16)
        native = np.isin(dominant_map, joints) & visible_for_label if joints.size else np.zeros_like(visible_for_label)
        anchor_key = f"smplx_{side}_hand_anchor_mask"
        if anchor_key in targets:
            native |= np.asarray(targets[anchor_key], dtype=bool)
        native = np.stack([largest_components(close2d(view, 1), max_components=2, min_pixels=1) for view in native], axis=0)
        roi_maps[key] = native
        roi_sources[key] = "smplx_native_joint2num_dominant_skinning_plus_v15_anchor"

        bridge_key = f"wrist_bridge_{side}"
        bridge_joints = np.asarray(sorted(groups.get(bridge_key, set())), dtype=np.int16)
        bridge = np.isin(dominant_map, bridge_joints) & visible_for_label if bridge_joints.size else np.zeros_like(visible_for_label)
        hand_grown = np.stack([dilate2d(view, iterations=4) for view in native], axis=0)
        bridge = (bridge | (hand_grown & visible_for_label)) & ~np.logical_or(
            roi_maps["left_hand"] if side == "right" and "left_hand" in roi_maps else np.zeros_like(bridge),
            roi_maps["right_hand"] if side == "left" and "right_hand" in roi_maps else np.zeros_like(bridge),
        )
        bridge = np.stack([largest_components(close2d(view, 1), max_components=2, min_pixels=1) for view in bridge], axis=0)
        roi_maps[bridge_key] = bridge
        roi_sources[bridge_key] = "smplx_native_wrist_elbow_skinning_with_dilated_hand_bridge"

    for side in SIDE_ORDER:
        spatial_fallback = spatial_hand_fallbacks(
            roi_maps[f"{side}_hand"].any(axis=0)[None].repeat(roi_maps[f"{side}_hand"].shape[0], axis=0)
            if False
            else roi_maps[f"{side}_hand"],
            side,
            int(args.min_finger_pixels),
        )
        for finger in FINGER_ORDER:
            key = f"{finger}_{side}"
            joints = np.asarray(sorted(groups.get(key, set())), dtype=np.int16)
            native = np.isin(dominant_map, joints) & roi_maps[f"{side}_hand"] if joints.size else np.zeros_like(visible_for_label)
            source = "smplx_native_finger_joint2num_dominant_skinning"
            if int(native.sum()) < int(args.min_finger_pixels):
                native = spatial_fallback[key]
                source = "documented_spatial_fallback_within_smplx_native_hand_roi"
            else:
                per_view_native = []
                for view_idx in range(native.shape[0]):
                    view = native[view_idx]
                    if int(view.sum()) < max(1, int(args.min_finger_pixels) // 3):
                        view = spatial_fallback[key][view_idx]
                        source = "mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views"
                    per_view_native.append(view)
                native = np.stack(per_view_native, axis=0)
            roi_maps[key] = native.astype(bool)
            roi_sources[key] = source

    raster_parts = load_npz(raster_part_path) if raster_part_path.is_file() else {}
    prior_head = np.asarray(raster_parts["head_map"], dtype=bool) if "head_map" in raster_parts else None
    prior_face = np.asarray(raster_parts["face_front_map"], dtype=bool) if "face_front_map" in raster_parts else None
    head_joints = np.asarray(sorted(groups.get("head", set())), dtype=np.int16)
    head_native = np.isin(dominant_map, head_joints) & visible_for_label if head_joints.size else np.zeros_like(visible_for_label)
    if prior_head is not None and prior_head.shape == head_native.shape:
        head_native |= prior_head
    hand_union = roi_maps["left_hand"] | roi_maps["right_hand"]
    head_rows = []
    for view_idx in range(head_native.shape[0]):
        view = head_native[view_idx]
        source = "smplx_native_head_jaw_eye_neck_skinning"
        if int(view.sum()) < int(args.min_head_pixels):
            view = head_spatial_fallback(silhouette[view_idx], hand_union[view_idx], prior_head[view_idx] if prior_head is not None and prior_head.shape == head_native.shape else None)
            source = "documented_spatial_fallback_top_silhouette_or_v15_head_map"
        head_rows.append(view & silhouette[view_idx])
    roi_maps["head"] = np.stack(head_rows, axis=0).astype(bool)
    roi_sources["head"] = "smplx_native_head_jaw_eye_neck_skinning_plus_spatial_fallback_when_sparse"

    static_lmk_vertices = landmark_vertex_mask(model_path, canonical_vertices.shape[0], "static")
    dynamic_lmk_vertices = landmark_vertex_mask(model_path, canonical_vertices.shape[0], "dynamic")
    face_lmk_static = np.zeros_like(visible_for_label)
    face_lmk_dynamic = np.zeros_like(visible_for_label)
    face_lmk_static[valid_vertex] = static_lmk_vertices[nearest_ids[valid_vertex]]
    face_lmk_dynamic[valid_vertex] = dynamic_lmk_vertices[nearest_ids[valid_vertex]]
    face_lmk_static &= roi_maps["head"]
    face_lmk_dynamic &= roi_maps["head"]
    face_lmk_static = np.stack([dilate2d(view, iterations=int(args.landmark_dilate)) for view in face_lmk_static], axis=0) & roi_maps["head"]
    face_lmk_dynamic = np.stack([dilate2d(view, iterations=int(args.landmark_dilate)) for view in face_lmk_dynamic], axis=0) & roi_maps["head"]
    roi_maps["face_lmk_static"] = face_lmk_static
    roi_maps["face_lmk_dynamic"] = face_lmk_dynamic
    roi_sources["face_lmk_static"] = "smplx_native_lmk_faces_idx_projected_via_nearest_canonical_vertex"
    roi_sources["face_lmk_dynamic"] = "smplx_native_dynamic_lmk_faces_idx_projected_via_nearest_canonical_vertex"

    face_front = face_lmk_static | face_lmk_dynamic
    if prior_face is not None and prior_face.shape == face_front.shape:
        face_front |= prior_face & roi_maps["head"]
    face_views = []
    for view_idx in range(face_front.shape[0]):
        view = face_front[view_idx] & roi_maps["head"][view_idx]
        if int(view.sum()) < int(args.min_face_pixels):
            view = face_spatial_fallback(roi_maps["head"][view_idx], silhouette[view_idx])
        face_views.append(view & roi_maps["head"][view_idx])
    roi_maps["face_front"] = np.stack(face_views, axis=0).astype(bool)
    roi_sources["face_front"] = "smplx_native_lmk_faces_dynamic_lmk_plus_v15_or_spatial_front_fallback"

    missing_required = [name for name in CORE_ROIS if int(roi_maps[name].sum()) <= 0]
    status = (
        "v16_smplx_native_region_roi_builder_ready_with_documented_fallbacks"
        if not missing_required
        else "v16_smplx_native_region_roi_builder_blocked"
    )
    if missing_required:
        blockers.append(f"Required ROI maps are empty: {missing_required}")

    roi_stack = np.stack([roi_maps[name].astype(np.uint8) for name in ROI_SAVE_ORDER], axis=1)
    view_support_rows = [row_stats(name, roi_maps[name], roi_sources.get(name, "unknown")) for name in ROI_SAVE_ORDER]
    for row in view_support_rows:
        bbox_row(row, roi_maps[row["region"]].any(axis=0))

    npz_path = out / "v16_smplx_native_region_roi_maps.npz"
    table_json_path = out / "view_support_table.json"
    table_md_path = out / "view_support_table.md"
    contact_sheet = out / "roi_contact_sheet.png"
    np.savez_compressed(
        npz_path,
        roi_maps=roi_stack,
        roi_names=np.asarray(ROI_SAVE_ORDER),
        view_names=np.asarray(camera_ids),
        nearest_vertex_ids=nearest_ids.astype(np.int32),
        dominant_joint_ids=dominant_map.astype(np.int16),
        body_visible=roi_maps["body_visible"].astype(np.uint8),
        research_only=np.asarray(True),
        smplx_native_only=np.asarray(True),
        no_mano=np.asarray(True),
        no_flame=np.asarray(True),
    )
    write_json(table_json_path, {"rows": view_support_rows})
    table_lines = [
        "# V16 SMPL-X Native Region ROI View Support",
        "",
        "| Region | Source | Total Pixels | Nonempty Views | Per View Pixels | Union BBox XYXY |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in view_support_rows:
        table_lines.append(
            f"| {row['region']} | {row['source']} | {row['total_pixels']} | {row['nonempty_views']} | "
            f"{row['per_view_pixels']} | {row.get('bbox_xyxy')} |"
        )
    table_md_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    save_contact_sheet(contact_sheet, images, roi_maps, camera_ids)

    metrics = {
        "view_count": int(prior_maps.shape[0]),
        "height": int(prior_maps.shape[2]),
        "width": int(prior_maps.shape[3]),
        "roi_count": int(len(ROI_SAVE_ORDER)),
        "required_nonempty": {name: int(roi_maps[name].sum()) > 0 for name in CORE_ROIS},
        "roi_pixels_total": {name: int(roi_maps[name].sum()) for name in ROI_SAVE_ORDER},
        "native_lmk_fields": native_lmk_keys,
        "smplx_joint_count": int(weights.shape[1]),
        "smplx_vertex_count": int(canonical_vertices.shape[0]),
        "smplx_face_count": int(faces.shape[0]),
        "canonical_query_valid_pixels": int(valid_vertex.sum()),
        "canonical_query_visible_pixels": int(visible_for_label.sum()),
        "canonical_query_coverage": float(valid_vertex.sum() / max(int(visible_for_label.sum()), 1)),
        "prior_mask_pixels": int(prior_mask.sum()),
        "silhouette_pixels": int(silhouette.sum()),
        "body_visible_pixels": int(roi_maps["body_visible"].sum()),
        "left_hand_pixels": int(roi_maps["left_hand"].sum()),
        "right_hand_pixels": int(roi_maps["right_hand"].sum()),
        "wrist_bridge_left_pixels": int(roi_maps["wrist_bridge_left"].sum()),
        "wrist_bridge_right_pixels": int(roi_maps["wrist_bridge_right"].sum()),
        "head_pixels": int(roi_maps["head"].sum()),
        "face_front_pixels": int(roi_maps["face_front"].sum()),
        "dominant_joint_stats": scalar_stats(dominant_map[valid_vertex]),
    }
    summary = {
        "task": "v16_smplx_native_region_roi_builder",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "smplx_native_only": True,
        "no_mano": True,
        "no_flame": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
        "inputs": {
            "smplx_root": str(Path(args.smplx_root).expanduser().resolve()),
            "model_path": str(model_path),
            "case_root": str(case_root),
            "v15_raster_dir": str(args.v15_raster_dir.expanduser().resolve()),
            "inputs_npz": str(inputs_path),
            "targets_npz": str(targets_path),
            "manifest": str(manifest_path),
        },
        "native_metadata": {
            "joint2num_keys": sorted(joint2num),
            "part2num_keys": sorted(part2num),
            "lmk_fields": native_lmk_keys,
            "finger_groups": {key: sorted(list(value)) for key, value in groups.items() if any(finger in key for finger in FINGER_ORDER)},
            "head_groups": {
                "head": sorted(list(groups.get("head", set()))),
                "face_front": sorted(list(groups.get("face_front", set()))),
            },
        },
        "fallback_policy": {
            "hand_and_wrist": "Primary labels use SMPL-X joint2num/dominant skinning and V15 native hand anchors; wrist bridges dilate hand ROIs onto wrist/elbow support.",
            "fingers": "If native per-finger skinning support is too sparse, split the SMPL-X-native hand ROI into thumb/index/middle/ring/pinky spatial bands and mark the source.",
            "head": "Primary label uses SMPL-X Head/Jaw/Eye/Neck skinning and V15 head map; sparse views fall back to top silhouette support excluding hands.",
            "face_front": "Primary label uses SMPL-X lmk_faces_idx and dynamic_lmk_faces_idx vertices; sparse views fall back to a centered crop inside the head ROI.",
        },
        "conditions": {
            "left_right_hand_nonempty": bool(int(roi_maps["left_hand"].sum()) > 0 and int(roi_maps["right_hand"].sum()) > 0),
            "wrist_bridge_nonempty": bool(int(roi_maps["wrist_bridge_left"].sum()) > 0 and int(roi_maps["wrist_bridge_right"].sum()) > 0),
            "head_nonempty": bool(int(roi_maps["head"].sum()) > 0),
            "face_front_nonempty": bool(int(roi_maps["face_front"].sum()) > 0),
            "strict_pass_claimed": False,
        },
        "metrics": metrics,
        "view_support_table": view_support_rows,
        "outputs": {
            "roi_maps_npz": str(npz_path.resolve()),
            "summary_json": str((out / "summary.json").resolve()),
            "view_support_table_json": str(table_json_path.resolve()),
            "view_support_table_md": str(table_md_path.resolve()),
            "roi_contact_sheet": str(contact_sheet.resolve()),
        },
        "decision": (
            "Runnable V16 SMPL-X-native ROI maps were produced without MANO/FLAME; sparse subregions use documented spatial fallbacks and no strict pass is claimed."
            if not blockers
            else "V16 SMPL-X-native ROI maps are blocked by empty required regions or missing inputs."
        ),
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "# V16 SMPL-X Native Region ROI Builder",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only ROI builder. It writes no predictions, teacher/candidate package, registry, or strict-pass state.",
        "",
        "## Decision",
        "",
        str(summary.get("decision", "")),
        "",
        "## Conditions",
        "",
    ]
    for key, value in summary.get("conditions", {}).items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Key Metrics", ""])
    for key in (
        "view_count",
        "height",
        "width",
        "roi_count",
        "left_hand_pixels",
        "right_hand_pixels",
        "wrist_bridge_left_pixels",
        "wrist_bridge_right_pixels",
        "head_pixels",
        "face_front_pixels",
        "canonical_query_coverage",
        "native_lmk_fields",
    ):
        if key in metrics:
            lines.append(f"- {key}: `{json_ready(metrics[key])}`")
    lines.extend(["", "## View Support", "", "| Region | Source | Total Pixels | Nonempty Views | Per View Pixels |", "|---|---|---:|---:|---|"])
    for row in summary.get("view_support_table", []):
        lines.append(
            f"| {row['region']} | {row['source']} | {row['total_pixels']} | {row['nonempty_views']} | {row['per_view_pixels']} |"
        )
    lines.extend(["", "## Fallback Policy", ""])
    for key, value in summary.get("fallback_policy", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V16 SMPL-X native hand/head/face ROI maps without MANO/FLAME.")
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--v15-raster-dir", type=Path, default=DEFAULT_V15_RASTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-expression", type=int, default=10)
    parser.add_argument("--body-part-count", type=int, default=12)
    parser.add_argument("--min-finger-pixels", type=int, default=12)
    parser.add_argument("--min-head-pixels", type=int, default=64)
    parser.add_argument("--min-face-pixels", type=int, default=24)
    parser.add_argument("--landmark-dilate", type=int, default=5)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.json_out, summary)
    write_markdown(summary, args.md_out)
    print(json.dumps(json_ready({"status": summary.get("status"), "metrics": summary.get("metrics", {}), "blockers": summary.get("blockers", [])}), indent=2, ensure_ascii=False))
    return 0 if not summary.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
