from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_fullbody_hand_integrity import (  # noqa: E402
    connected_component_stats_2d,
    create_hand_landmarker,
    hand_risk_mask,
    point_box_spatial_stats,
)
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_scene_local_prior_bundle_path,
    resolve_smplx_model_dir,
)
from research_scene_assets import load_camera_params_sidecar  # noqa: E402
from tools.smplx_numpy import load_smplx_model, resolve_smplx_model_path  # noqa: E402


STRICT_GATE_TRUTH = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}

LEFT_HAND_JOINTS = "20,25-39"
RIGHT_HAND_JOINTS = "21,40-54"
HAND_JOINTS = "20,21,25-54"
PRED_WORLD_KEYS = ("world_points", "points_world", "pred_world_points", "point_map", "points3d", "points_3d")
PRED_DEPTH_KEYS = ("depth", "depths", "depth_map", "depth_maps", "pred_depth")
PRED_CONF_KEYS = ("world_points_conf", "point_conf", "points_conf", "depth_conf")
PRED_INTRINSIC_KEYS = ("intrinsic", "intrinsics", "camera_intrinsics", "K")
PRED_EXTRINSIC_KEYS = ("extrinsic", "extrinsics", "camera_extrinsics", "world_to_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local-only B-hand evidence cache for HGGT-style hand-token "
            "preflight. The cache is diagnostic only: it does not train, infer, "
            "export a teacher/candidate, write a strict pass, launch Modal, or "
            "patch predictions."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=0, help="0 means infer from predictions/prior, else use this size.")
    parser.add_argument("--view-indices", default="auto", help="'auto', 'all', comma list, or simple ranges like 0-5,10.")
    parser.add_argument("--max-auto-views", type=int, default=12)
    parser.add_argument("--hand-landmarker-model", type=Path, default=REPO_ROOT / "external_models" / "hand_landmarker.task")
    parser.add_argument("--disable-mediapipe-hands", action="store_true")
    parser.add_argument("--hand-box-pad", type=int, default=24)
    parser.add_argument("--probe-real-cameras", action="store_true")
    parser.add_argument("--probe-annotations", action="store_true")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--patch-size", type=int, default=14)
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
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_existing_key(keys: list[str], aliases: tuple[str, ...]) -> str | None:
    available = set(keys)
    for alias in aliases:
        if alias in available:
            return alias
    return None


def normalize_camera_id(value: object) -> str:
    text = str(value)
    try:
        return str(int(text))
    except ValueError:
        return text.zfill(2)


def as_scalar_stack(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        return values[None].astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0][None].astype(np.float32, copy=False)
    if values.ndim == 3:
        return values.astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[-1] == 1:
        return values[..., 0].astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[1] == 1:
        return values[:, 0].astype(np.float32, copy=False)
    raise ValueError(f"Expected scalar stack, got {values.shape}")


def as_vector_stack(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] == 3:
        return values[None].astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[0] == 3:
        return np.transpose(values, (1, 2, 0))[None].astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[-1] == 3:
        return values.astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[1] == 3:
        return np.transpose(values, (0, 2, 3, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Expected vector stack, got {values.shape}")


def as_intrinsic_stack(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"Expected intrinsic stack, got {values.shape}")
    if values.shape[-2:] == (4, 4):
        values = values[:, :3, :3]
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"Expected 3x3 or 4x4 intrinsics, got {values.shape}")
    return values.astype(np.float32, copy=False)


def as_homogeneous_stack(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"Expected extrinsic stack, got {values.shape}")
    if values.shape[-2:] == (4, 4):
        return values.astype(np.float32, copy=False)
    if values.shape[-2:] == (3, 4):
        out = np.repeat(np.eye(4, dtype=np.float32)[None], values.shape[0], axis=0)
        out[:, :3, :4] = values
        return out
    raise ValueError(f"Expected 3x4 or 4x4 extrinsics, got {values.shape}")


def parse_joint_ids(spec: str) -> set[int]:
    out: set[int] = set()
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            out.update(range(int(left), int(right) + 1))
        else:
            out.add(int(item))
    return out


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def clip_box(box: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return [x0, y0, x1, y1]


def box_mask(box: list[int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def body_center_x(body_mask: np.ndarray) -> float:
    box = bbox_from_mask(body_mask)
    if box is None:
        return 0.5 * float(body_mask.shape[1] - 1)
    return 0.5 * float(box[0] + box[2])


def heuristic_side_for_box(box: list[int], body_mask: np.ndarray) -> str:
    center_x = 0.5 * float(box[0] + box[2])
    return "left" if center_x < body_center_x(body_mask) else "right"


def normalize_mediapipe_side(raw_label: str | None) -> str | None:
    text = str(raw_label or "").strip().lower()
    if text in {"left", "right"}:
        return text
    return None


def handedness_payload(result: Any, hand_idx: int) -> dict[str, Any]:
    handedness = getattr(result, "handedness", None) or []
    if hand_idx >= len(handedness):
        return {"raw_label": None, "score": None}
    categories = handedness[hand_idx] or []
    if not categories:
        return {"raw_label": None, "score": None}
    category = categories[0]
    label = getattr(category, "category_name", None) or getattr(category, "display_name", None)
    score = getattr(category, "score", None)
    return {
        "raw_label": None if label is None else str(label),
        "score": None if score is None else float(score),
    }


def mediapipe_hand_rois(
    rgb: np.ndarray,
    body_mask: np.ndarray,
    detector: Any | None,
    pad: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if detector is None:
        return None
    try:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb.astype(np.uint8)))
        result = detector.detect(image)
    except Exception as exc:
        return (
            [],
            {
                "roi_source": "mediapipe_error",
                "error": f"{type(exc).__name__}: {exc}",
                "detections": 0,
            },
        )

    height, width = body_mask.shape
    blocked = build_roi_masks(body_mask)["head"] | build_roi_masks(body_mask)["face"]
    rois: list[dict[str, Any]] = []
    for hand_idx, hand_landmarks in enumerate(getattr(result, "hand_landmarks", []) or []):
        xs = [float(landmark.x) * width for landmark in hand_landmarks]
        ys = [float(landmark.y) * height for landmark in hand_landmarks]
        if not xs or not ys:
            continue
        box = clip_box(
            [
                int(np.floor(min(xs) - int(pad))),
                int(np.floor(min(ys) - int(pad))),
                int(np.ceil(max(xs) + int(pad))),
                int(np.ceil(max(ys) + int(pad))),
            ],
            width,
            height,
        )
        support = box_mask(box, body_mask.shape) & body_mask.astype(bool) & ~blocked
        if int(support.sum()) <= 0:
            continue
        handedness = handedness_payload(result, hand_idx)
        side = normalize_mediapipe_side(handedness.get("raw_label"))
        side_source = "mediapipe_handedness"
        if side is None:
            side = heuristic_side_for_box(box, body_mask)
            side_source = "image_x_relative_to_body_bbox"
        rois.append(
            {
                "side": side,
                "side_assignment_source": side_source,
                "roi_source": "mediapipe_hand_landmarker",
                "bbox_xyxy": box,
                "mask": support,
                "landmark_bbox_xyxy": box,
                "raw_handedness": handedness,
            }
        )
    return (
        rois,
        {
            "roi_source": "mediapipe_hand_landmarker",
            "detections": int(len(rois)),
            "model_path": "provided",
        },
    )


def fallback_hand_rois(rgb: np.ndarray, body_mask: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hand_mask, summary = hand_risk_mask(rgb, body_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hand_mask.astype(np.uint8), connectivity=8)
    rois: list[dict[str, Any]] = []
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, label))
    components.sort(reverse=True)
    for area, label in components[:4]:
        support = labels == label
        box = bbox_from_mask(support)
        if box is None:
            continue
        rois.append(
            {
                "side": heuristic_side_for_box(box, body_mask),
                "side_assignment_source": "image_x_relative_to_body_bbox",
                "roi_source": "skin_extremity_fallback",
                "bbox_xyxy": box,
                "mask": support,
                "fallback_component_pixels": int(area),
                "raw_handedness": {"raw_label": None, "score": None},
            }
        )
    return rois, summary


def parse_view_indices(spec: str, view_count: int, max_auto_views: int) -> list[int]:
    text = str(spec or "auto").strip().lower()
    if text == "all":
        return list(range(view_count))
    if text == "auto":
        limit = max(1, min(int(max_auto_views), int(view_count)))
        if limit >= view_count:
            return list(range(view_count))
        return sorted(set(np.linspace(0, view_count - 1, num=limit, dtype=int).tolist()))
    out: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            values = list(range(start, stop + step, step))
        else:
            values = [int(item)]
        for value in values:
            if value < 0:
                value = view_count + value
            if value < 0 or value >= view_count:
                raise IndexError(f"View index {raw!r} resolved outside [0, {view_count})")
            out.append(int(value))
    return sorted(dict.fromkeys(out))


def load_prediction_probe(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "reason": "not_provided"}
    path = path.expanduser().resolve()
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "missing_file"}
    with np.load(path, allow_pickle=False) as payload:
        keys = list(payload.files)
        world_key = first_existing_key(keys, PRED_WORLD_KEYS)
        depth_key = first_existing_key(keys, PRED_DEPTH_KEYS)
        conf_key = first_existing_key(keys, PRED_CONF_KEYS)
        intrinsic_key = first_existing_key(keys, PRED_INTRINSIC_KEYS)
        extrinsic_key = first_existing_key(keys, PRED_EXTRINSIC_KEYS)
        world_points = as_vector_stack(payload[world_key]) if world_key else None
        depth = as_scalar_stack(payload[depth_key]) if depth_key else None
        conf = as_scalar_stack(payload[conf_key]) if conf_key else None
        intrinsics = as_intrinsic_stack(payload[intrinsic_key]) if intrinsic_key else None
        extrinsics = as_homogeneous_stack(payload[extrinsic_key]) if extrinsic_key else None

    view_count = 0
    hw: tuple[int, int] | None = None
    if world_points is not None:
        view_count = int(world_points.shape[0])
        hw = (int(world_points.shape[1]), int(world_points.shape[2]))
    elif depth is not None:
        view_count = int(depth.shape[0])
        hw = (int(depth.shape[1]), int(depth.shape[2]))
    return {
        "available": bool(world_points is not None or depth is not None),
        "path": str(path),
        "keys": {
            "world_points": world_key,
            "depth": depth_key,
            "confidence": conf_key,
            "intrinsic": intrinsic_key,
            "extrinsic": extrinsic_key,
        },
        "view_count": view_count,
        "image_hw": None if hw is None else [int(hw[0]), int(hw[1])],
        "world_points": world_points,
        "depth": depth,
        "confidence": conf,
        "intrinsic": intrinsics,
        "extrinsic": extrinsics,
    }


def load_smplx_prior_probe(scene_dir: Path, scene_manifest: dict[str, Any]) -> dict[str, Any]:
    prior_path = resolve_scene_local_prior_bundle_path(scene_dir, scene_manifest)
    if prior_path is None or not prior_path.is_file():
        return {"available": False, "reason": "missing_prior_maps_npz"}
    with np.load(prior_path, allow_pickle=False) as payload:
        keys = list(payload.files)
        prior_maps = np.asarray(payload["prior_maps"], dtype=np.float32) if "prior_maps" in keys else None
        prior_mask = np.asarray(payload["prior_mask"], dtype=bool) if "prior_mask" in keys else None
        channels = [str(item) for item in payload["prior_channels"].tolist()] if "prior_channels" in keys else []
        summary_tokens_shape = (
            list(np.asarray(payload["prior_summary_tokens"]).shape)
            if "prior_summary_tokens" in keys
            else None
        )
    if prior_maps is None or prior_mask is None:
        return {"available": False, "path": str(prior_path), "reason": "missing_prior_maps_or_mask"}
    channel_index = {name: idx for idx, name in enumerate(channels)}
    visible_idx = channel_index.get("smplx_visible_mask")
    posed_keys = ["smplx_posed_cam_x", "smplx_posed_cam_y", "smplx_posed_cam_z"]
    posed_indices = [channel_index.get(name) for name in posed_keys]
    visible = None
    if visible_idx is not None:
        visible = prior_mask & (prior_maps[:, visible_idx] > 0.5)
    smplx_cam = None
    if all(idx is not None for idx in posed_indices):
        smplx_cam = np.stack([prior_maps[:, int(idx)] for idx in posed_indices], axis=-1).astype(np.float32)
    return {
        "available": True,
        "path": str(prior_path),
        "prior_maps_shape": list(prior_maps.shape),
        "prior_mask_shape": list(prior_mask.shape),
        "prior_channels": channels,
        "has_smplx_visible_mask": visible is not None,
        "has_smplx_posed_cam_xyz": smplx_cam is not None,
        "prior_summary_tokens_shape": summary_tokens_shape,
        "smplx_visible": visible,
        "smplx_cam": smplx_cam,
    }


def probe_annotation_availability(
    enabled: bool,
    scene_manifest: dict[str, Any],
    dataset_root: Path | None,
    subset_name: str,
) -> dict[str, Any]:
    if not enabled:
        return {"available": False, "reason": "annotation_probe_disabled"}
    if dataset_root is None:
        return {"available": False, "reason": "missing_dataset_root"}
    try:
        smplx_params, keypoints3d = load_optional_annotation_payload(scene_manifest, dataset_root, subset_name)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "available": bool(smplx_params),
        "smplx_keys": sorted(str(key) for key in smplx_params.keys()),
        "has_required_mesh_params": {"betas", "fullpose"}.issubset(set(smplx_params.keys())),
        "keypoints3d_shape": None if keypoints3d is None else list(keypoints3d.shape),
        "smplx_params": smplx_params,
    }


def select_faces_for_joints(faces: np.ndarray, dominant_joint: np.ndarray, joints: set[int]) -> tuple[int, int]:
    vertex_mask = np.isin(dominant_joint, np.array(sorted(joints), dtype=np.int64))
    selected_faces = faces[np.any(vertex_mask[faces], axis=1)]
    return int(vertex_mask.sum()), int(selected_faces.shape[0])


def probe_smplx_hand_model(
    model_dir_arg: Path | None,
    smplx_gender: str,
    annotation_probe: dict[str, Any],
) -> dict[str, Any]:
    smplx_params = annotation_probe.get("smplx_params") if annotation_probe.get("available") else None
    model_dir = resolve_smplx_model_dir(None if model_dir_arg is None else str(model_dir_arg))
    if model_dir is None:
        return {
            "available": False,
            "reason": "missing_smplx_model_dir",
            "hand_joint_specs": {
                "all": HAND_JOINTS,
                "left": LEFT_HAND_JOINTS,
                "right": RIGHT_HAND_JOINTS,
            },
        }
    try:
        model_path = resolve_smplx_model_path(model_dir, smplx_gender)
        model = load_smplx_model(model_path)
        faces = np.asarray(model["faces"], dtype=np.int32)
        dominant_joint = np.asarray(model["weights"], dtype=np.float32).argmax(axis=1)
        all_vertices, all_faces = select_faces_for_joints(faces, dominant_joint, parse_joint_ids(HAND_JOINTS))
        left_vertices, left_faces = select_faces_for_joints(faces, dominant_joint, parse_joint_ids(LEFT_HAND_JOINTS))
        right_vertices, right_faces = select_faces_for_joints(faces, dominant_joint, parse_joint_ids(RIGHT_HAND_JOINTS))
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "model_dir": str(model_dir)}
    return {
        "available": bool(smplx_params),
        "model_path": str(model_path),
        "annotation_params_available": bool(smplx_params),
        "hand_joint_specs": {
            "all": HAND_JOINTS,
            "left": LEFT_HAND_JOINTS,
            "right": RIGHT_HAND_JOINTS,
        },
        "hand_topology_counts": {
            "all": {"vertices": all_vertices, "faces": all_faces},
            "left": {"vertices": left_vertices, "faces": left_faces},
            "right": {"vertices": right_vertices, "faces": right_faces},
        },
        "note": (
            "This only proves side-specific hand topology can be selected. "
            "This script does not rasterize it into a teacher or candidate."
        ),
    }


def prediction_target_size(prediction_probe: dict[str, Any]) -> int | None:
    image_hw = prediction_probe.get("image_hw")
    if image_hw and len(image_hw) >= 2 and int(image_hw[0]) == int(image_hw[1]):
        return int(image_hw[0])
    return None


def prior_target_size(prior_probe: dict[str, Any]) -> int | None:
    shape = prior_probe.get("prior_mask_shape")
    if shape and len(shape) == 3 and int(shape[1]) == int(shape[2]):
        return int(shape[1])
    return None


def build_camera_probe(
    args: argparse.Namespace,
    scene_manifest: dict[str, Any],
    prediction_probe: dict[str, Any],
    target_size: int,
) -> dict[str, Any]:
    pred_intrinsic = prediction_probe.get("intrinsic")
    pred_extrinsic = prediction_probe.get("extrinsic")
    if pred_intrinsic is not None:
        view_count = int(pred_intrinsic.shape[0])
        per_view = {}
        for idx in range(view_count):
            per_view[int(idx)] = {
                "intrinsic": pred_intrinsic[idx],
                "world_to_cam": pred_extrinsic[idx] if pred_extrinsic is not None and idx < pred_extrinsic.shape[0] else None,
            }
        return {
            "available": True,
            "source": "predictions_npz",
            "per_view": per_view,
            "world_to_cam_available": pred_extrinsic is not None,
        }

    sidecar = load_camera_params_sidecar(Path(args.scene_dir))
    if sidecar:
        per_view = {}
        for idx, view in enumerate(scene_manifest["exported_views"]):
            camera_id = normalize_camera_id(view["camera_id"])
            if camera_id not in sidecar:
                continue
            cam = sidecar[camera_id]
            per_view[idx] = {
                "camera_id": camera_id,
                "intrinsic": align_intrinsics_for_scene_view(cam["intrinsic"], view, target_size=target_size),
                "world_to_cam": cam["world_to_cam"],
            }
        return {
            "available": bool(per_view),
            "source": "camera_params_sidecar",
            "per_view": per_view,
            "world_to_cam_available": True,
        }

    if args.probe_real_cameras:
        dataset_root = args.dataset_root or Path(str(scene_manifest.get("dataset_root", ""))).expanduser()
        try:
            camera_params, camera_source = resolve_scene_camera_params(scene_manifest, dataset_root, args.subset_name)
            per_view = {}
            for idx, view in enumerate(scene_manifest["exported_views"]):
                camera_id = normalize_camera_id(view["camera_id"])
                cam = camera_params[camera_id]
                per_view[idx] = {
                    "camera_id": camera_id,
                    "intrinsic": align_intrinsics_for_scene_view(cam["intrinsic"], view, target_size=target_size),
                    "world_to_cam": cam["world_to_cam"],
                }
            return {
                "available": True,
                "source": camera_source,
                "per_view": per_view,
                "world_to_cam_available": True,
            }
        except Exception as exc:
            return {"available": False, "source": "real_camera_probe_error", "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "available": False,
        "source": "not_probed",
        "reason": "Provide predictions intrinsics/extrinsics, camera_params_sidecar.npz, or --probe-real-cameras.",
    }


def percentile_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "p50": None, "p10": None, "p90": None}
    return {
        "count": int(values.size),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def prediction_roi_metadata(prediction_probe: dict[str, Any], view_idx: int, mask: np.ndarray) -> dict[str, Any]:
    if not prediction_probe.get("available"):
        return {"available": False, "reason": prediction_probe.get("reason", "missing_predictions")}
    world_points = prediction_probe.get("world_points")
    depth = prediction_probe.get("depth")
    conf = prediction_probe.get("confidence")
    if world_points is not None and view_idx < int(world_points.shape[0]):
        support = mask & np.isfinite(world_points[view_idx]).all(axis=-1)
        spatial = point_box_spatial_stats(world_points[view_idx][support])
    elif depth is not None and view_idx < int(depth.shape[0]):
        support = mask & np.isfinite(depth[view_idx]) & (depth[view_idx] > 0.0)
        spatial = {"points": int(support.sum()), "source": "depth_valid_only"}
    else:
        return {"available": False, "reason": "view_outside_prediction_stack"}
    conf_stats = None
    if conf is not None and view_idx < int(conf.shape[0]):
        conf_stats = percentile_stats(conf[view_idx][support])
    return {
        "available": True,
        "support_pixels": int(support.sum()),
        "support_ratio_in_roi": float(support.sum() / max(int(mask.sum()), 1)),
        "confidence": conf_stats,
        "spatial": spatial,
    }


def smplx_prior_roi_metadata(prior_probe: dict[str, Any], view_idx: int, mask: np.ndarray) -> dict[str, Any]:
    if not prior_probe.get("available"):
        return {"available": False, "reason": prior_probe.get("reason", "missing_prior")}
    visible = prior_probe.get("smplx_visible")
    smplx_cam = prior_probe.get("smplx_cam")
    if visible is None or view_idx >= int(visible.shape[0]):
        return {"available": False, "reason": "missing_visible_mask_or_view_outside_prior_stack"}
    visible_mask = np.asarray(visible[view_idx], dtype=bool) & mask
    spatial = None
    if smplx_cam is not None and view_idx < int(smplx_cam.shape[0]):
        spatial = point_box_spatial_stats(smplx_cam[view_idx][visible_mask])
    return {
        "available": True,
        "visible_pixels": int(visible_mask.sum()),
        "visible_ratio_in_roi": float(visible_mask.sum() / max(int(mask.sum()), 1)),
        "spatial": spatial,
        "side_specific_prior": "placeholder_requires_raw_smplx_hand_joint_raster",
    }


def build_ray_metadata(
    camera_probe: dict[str, Any],
    view_idx: int,
    bbox: list[int],
) -> dict[str, Any]:
    if not camera_probe.get("available"):
        return {"available": False, "reason": camera_probe.get("reason", "missing_camera")}
    per_view = camera_probe.get("per_view", {})
    cam = per_view.get(view_idx) or per_view.get(int(view_idx))
    if cam is None or cam.get("intrinsic") is None:
        return {"available": False, "reason": "view_camera_missing"}
    intrinsic = np.asarray(cam["intrinsic"], dtype=np.float32)
    inv_k = np.linalg.inv(intrinsic)
    x0, y0, x1, y1 = bbox
    samples = [
        [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
        [x0, y0],
        [max(x0, x1 - 1), y0],
        [x0, max(y0, y1 - 1)],
        [max(x0, x1 - 1), max(y0, y1 - 1)],
    ]
    rays_cam = []
    for x, y in samples:
        ray = inv_k @ np.array([float(x), float(y), 1.0], dtype=np.float32)
        ray = ray / np.clip(np.linalg.norm(ray), 1e-8, None)
        rays_cam.append(ray.astype(np.float32))
    world_to_cam = cam.get("world_to_cam")
    rays_world = None
    if world_to_cam is not None:
        rotation = np.asarray(world_to_cam, dtype=np.float32)[:3, :3]
        rays_world = [rotation.T @ ray for ray in rays_cam]
        rays_world = [ray / np.clip(np.linalg.norm(ray), 1e-8, None) for ray in rays_world]
    return {
        "available": True,
        "camera_source": camera_probe.get("source"),
        "bbox_xyxy": bbox,
        "pixel_samples_xy": samples,
        "camera_ray_unit_samples": rays_cam,
        "world_ray_unit_samples": rays_world,
        "placeholder_note": "Ray samples are evidence-cache hooks only; no triangulation or teacher target is produced.",
    }


def build_vggt_token_hook(bbox: list[int], image_hw: tuple[int, int], patch_size: int) -> dict[str, Any]:
    height, width = image_hw
    patch = max(1, int(patch_size))
    grid_h = int(np.ceil(height / float(patch)))
    grid_w = int(np.ceil(width / float(patch)))
    x0, y0, x1, y1 = bbox
    px0 = max(0, min(grid_w, x0 // patch))
    py0 = max(0, min(grid_h, y0 // patch))
    px1 = max(0, min(grid_w, int(np.ceil(x1 / float(patch)))))
    py1 = max(0, min(grid_h, int(np.ceil(y1 / float(patch)))))
    token_ids = [int(y * grid_w + x) for y in range(py0, py1) for x in range(px0, px1)]
    return {
        "available": False,
        "hook_stage": "placeholder_pre_vggt_backend_wiring",
        "patch_size": patch,
        "patch_grid_hw": [grid_h, grid_w],
        "patch_range_xyxy": [px0, py0, px1, py1],
        "estimated_patch_token_count": int(len(token_ids)),
        "token_ids_preview": token_ids[:32],
        "required_backend_hook": "side-aware hand_roi_patch_mask -> hand token aggregator",
    }


def summarize_roi(
    roi: dict[str, Any],
    body_mask: np.ndarray,
    view_idx: int,
    camera_id: str,
    scene: Any,
    prior_probe: dict[str, Any],
    prediction_probe: dict[str, Any],
    camera_probe: dict[str, Any],
    patch_size: int,
) -> dict[str, Any]:
    mask = np.asarray(roi["mask"], dtype=bool)
    height, width = mask.shape
    bbox = clip_box(list(roi["bbox_xyxy"]), width, height)
    bbox_w = max(0, bbox[2] - bbox[0])
    bbox_h = max(0, bbox[3] - bbox[1])
    crop_pixels = int(bbox_w * bbox_h)
    mask_pixels = int(mask.sum())
    return {
        "view_index": int(view_idx),
        "camera_id": camera_id,
        "side": str(roi.get("side", "unknown")),
        "side_assignment_source": str(roi.get("side_assignment_source", "unknown")),
        "roi_source": str(roi.get("roi_source", "unknown")),
        "raw_handedness": roi.get("raw_handedness"),
        "image_path": str(scene.image_path),
        "mask_path": str(scene.mask_path),
        "bbox_xyxy": bbox,
        "crop_metadata": {
            "image_hw": [int(height), int(width)],
            "bbox_wh": [int(bbox_w), int(bbox_h)],
            "bbox_area_pixels": int(crop_pixels),
            "roi_pixels": mask_pixels,
            "roi_density_in_bbox": float(mask_pixels / max(crop_pixels, 1)),
            "body_pixels": int(body_mask.sum()),
            "body_overlap_ratio": float(mask_pixels / max(int(body_mask.sum()), 1)),
            "component_stats": connected_component_stats_2d(mask),
        },
        "smplx_prior": smplx_prior_roi_metadata(prior_probe, view_idx, mask),
        "prediction_support": prediction_roi_metadata(prediction_probe, view_idx, mask),
        "camera_rays": build_ray_metadata(camera_probe, view_idx, bbox),
        "vggt_token_hook": build_vggt_token_hook(bbox, (height, width), patch_size),
    }


def update_side_summary(side_summary: dict[str, Any], roi_rows: list[dict[str, Any]]) -> None:
    view_sets = {side: set(side_summary[side]["view_indices"]) for side in side_summary}
    smplx_sets = {side: set(side_summary[side]["smplx_visible_view_indices"]) for side in side_summary}
    pred_sets = {side: set(side_summary[side]["prediction_support_view_indices"]) for side in side_summary}
    for row in roi_rows:
        side = row.get("side", "unknown")
        if side not in side_summary:
            side_summary[side] = {
                "view_indices": [],
                "views_with_roi": 0,
                "roi_count": 0,
                "roi_pixels": 0,
                "smplx_visible_view_indices": [],
                "views_with_smplx_visible_prior": 0,
                "smplx_visible_pixels": 0,
                "prediction_support_view_indices": [],
                "views_with_prediction_support": 0,
                "prediction_support_pixels": 0,
            }
            view_sets[side] = set()
            smplx_sets[side] = set()
            pred_sets[side] = set()
        view_idx = int(row["view_index"])
        side_summary[side]["roi_count"] += 1
        side_summary[side]["roi_pixels"] += int(row["crop_metadata"]["roi_pixels"])
        view_sets[side].add(view_idx)
        smplx_visible_pixels = int(row["smplx_prior"].get("visible_pixels", 0) or 0)
        if smplx_visible_pixels > 0:
            side_summary[side]["smplx_visible_pixels"] += smplx_visible_pixels
            smplx_sets[side].add(view_idx)
        pred_pixels = int(row["prediction_support"].get("support_pixels", 0) or 0)
        if pred_pixels > 0:
            side_summary[side]["prediction_support_pixels"] += pred_pixels
            pred_sets[side].add(view_idx)
    for side in side_summary:
        side_summary[side]["view_indices"] = sorted(view_sets[side])
        side_summary[side]["views_with_roi"] = int(len(view_sets[side]))
        side_summary[side]["smplx_visible_view_indices"] = sorted(smplx_sets[side])
        side_summary[side]["views_with_smplx_visible_prior"] = int(len(smplx_sets[side]))
        side_summary[side]["prediction_support_view_indices"] = sorted(pred_sets[side])
        side_summary[side]["views_with_prediction_support"] = int(len(pred_sets[side]))


def source_inventory() -> dict[str, Any]:
    return {
        "hand_roi_and_visibility": [
            "tools/audit_fullbody_hand_integrity.py:create_hand_landmarker",
            "tools/audit_fullbody_hand_integrity.py:mediapipe_hand_mask",
            "tools/audit_fullbody_hand_integrity.py:hand_risk_mask",
            "tools/audit_smplx_weak_anchor.py:hand_visible = smplx_visible & hand_mask",
            "tools/audit_smplx_raw_mesh_hand_anchor.py:left/right hand joint specs",
        ],
        "scene_mask_crop": [
            "tools/normal_line_multiview_eval.py:load_scene_view",
            "tools/normal_line_multiview_eval.py:build_roi_masks",
            "tools/augment_case_with_image_roi_masks.py:_make_masks",
        ],
        "camera": [
            "predictions.npz intrinsic/extrinsic from tools/run_local_vggt_inference.py",
            "camera_params_sidecar.npz from tools/research_scene_assets.py",
            "real rgb_cams.smc via tools/prepare_4k4d_prior_training_case.py:resolve_scene_camera_params",
        ],
        "smplx_prior": [
            "scene prior_maps.npz smplx_visible_mask and smplx_posed_cam_xyz channels",
            "SMPLx annotation payload via tools/prepare_4k4d_prior_training_case.py:load_optional_annotation_payload",
            "side-specific hand topology via tools/smplx_numpy.py model weights and audit_smplx_raw_mesh_hand_anchor.py joint ids",
        ],
        "vggt_token_hook": [
            "tools/run_local_vggt_inference.py passes prior_maps/prior_summary_tokens into VGGT",
            "B-hand B0 still needs a dedicated side-aware hand ROI patch-token hook",
        ],
    }


def hggt_b0_contract(side_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "B-hand HGGT-style hand-token B0",
        "status": "pre_backend_diagnostic_skeleton",
        "minimal_structure": {
            "inputs": [
                "left/right hand ROI boxes and masks per visible view",
                "hand crop metadata, body-mask overlap, and ROI component stats",
                "camera ray samples for each hand crop",
                "VGGT patch-token ROI ranges as hook placeholders",
                "SMPL-X visible prior overlap and optional side-specific hand topology availability",
            ],
            "tokens": [
                "side token: left/right identity plus view support histogram",
                "wrist-arm anchor token: must overlap or connect to forearm/body support",
                "palm/finger local tokens: diagnostic only until a backend hook consumes ROI patch masks",
                "confidence token: visible-view count, SMPL-X overlap, prediction support, and ray availability",
            ],
            "outputs": [
                "diagnostic JSON/MD cache only",
                "no teacher target, no candidate target, no strict pass, no cloud action",
            ],
        },
        "current_left_right_view_summary": side_summary,
        "open3d_visual_stop_condition": (
            "Any future Open3D hand visualization must show hands connected to arms. "
            "Detached sheets, detached sticks, isolated palm blobs, or floating finger clusters are stop conditions."
        ),
    }


def stop_conditions(cache: dict[str, Any]) -> list[dict[str, Any]]:
    sides = cache["visible_view_summary"]
    camera_available = bool(cache["camera_availability"].get("available"))
    prior_available = bool(cache["smplx_prior_availability"].get("available"))
    left_roi = int(sides.get("left", {}).get("views_with_roi", 0))
    right_roi = int(sides.get("right", {}).get("views_with_roi", 0))
    return [
        {
            "name": "strict_gate_remains_zero",
            "must_stop": True,
            "reason": "Research-preflight cannot write candidate/teacher pass or unblock cloud.",
        },
        {
            "name": "missing_left_or_right_hand_roi",
            "must_stop": bool(left_roi == 0 or right_roi == 0),
            "left_views": left_roi,
            "right_views": right_roi,
            "reason": "B0 needs visible evidence for both hands before backend wiring.",
        },
        {
            "name": "missing_smplx_prior_or_side_specific_hand_anchor",
            "must_stop": True,
            "prior_available": prior_available,
            "reason": (
                "Dense prior overlap is diagnostic. Side-specific SMPL-X hand topology must remain a weak prior, "
                "not a success claim."
            ),
        },
        {
            "name": "missing_camera_ray_metadata",
            "must_stop": not camera_available,
            "camera_available": camera_available,
            "reason": "Hand tokens need camera-ray hooks before any cross-view geometry attempt.",
        },
        {
            "name": "vggt_hand_token_hook_not_implemented",
            "must_stop": True,
            "reason": "This cache only records patch-token ranges; it does not modify VGGT or train a backend.",
        },
        {
            "name": "open3d_hand_must_connect_arm",
            "must_stop": True,
            "reason": "Future Open3D review must reject detached sheets/sticks and floating hand fragments.",
        },
    ]


def write_markdown_report(path: Path, cache: dict[str, Any]) -> None:
    lines = [
        "# B-Hand Evidence Cache",
        "",
        "Status: `diagnostic_only_preflight_cache`",
        "",
        "This cache is local-only. It is not a teacher, not a candidate, not a strict pass, and not a cloud unblock signal.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_GATE_TRUTH, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Inputs",
        "",
        f"- scene_dir: `{cache['scene_dir']}`",
        f"- predictions_npz: `{cache['predictions_availability'].get('path')}`",
        f"- target_size: `{cache['target_size']}`",
        f"- view_indices: `{cache['view_indices']}`",
        "",
        "## Left/Right Visible View Summary",
        "",
        "```json",
        json.dumps(cache["visible_view_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## SMPL-X Prior Availability",
        "",
        "```json",
        json.dumps(cache["smplx_prior_availability"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Camera Availability",
        "",
        "```json",
        json.dumps(cache["camera_availability"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## HGGT B0 Contract",
        "",
        "```json",
        json.dumps(cache["hggt_b0_contract"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Stop Conditions",
        "",
        "```json",
        json.dumps(cache["stop_conditions"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
    ]
    for output in cache["outputs"]:
        lines.append(f"- `{output}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def public_probe_summary(probe: dict[str, Any], drop_arrays: bool = True) -> dict[str, Any]:
    if not drop_arrays:
        return probe
    return {
        key: value
        for key, value in probe.items()
        if key not in {"world_points", "depth", "confidence", "intrinsic", "extrinsic", "smplx_visible", "smplx_cam", "smplx_params", "per_view"}
    }


def public_camera_summary(camera_probe: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in camera_probe.items()
        if key != "per_view"
    } | {
        "view_count_with_camera": int(len(camera_probe.get("per_view", {}))) if isinstance(camera_probe.get("per_view"), dict) else 0
    }


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_path = output_dir / "b_hand_evidence_cache.json"
    report_path = output_dir / "b_hand_evidence_cache_report.md"
    if (cache_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{cache_path} or {report_path} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    prediction_probe = load_prediction_probe(args.predictions_npz)
    prior_probe = load_smplx_prior_probe(scene_dir, scene_manifest)
    inferred_target = int(args.target_size) if int(args.target_size) > 0 else None
    inferred_target = inferred_target or prediction_target_size(prediction_probe) or prior_target_size(prior_probe) or 518
    target_size = int(inferred_target)
    view_count = int(len(scene_manifest.get("exported_views", [])))
    view_indices = parse_view_indices(args.view_indices, view_count, int(args.max_auto_views))

    dataset_root = args.dataset_root
    if dataset_root is None and scene_manifest.get("dataset_root"):
        dataset_root = Path(str(scene_manifest["dataset_root"])).expanduser()
    camera_probe = build_camera_probe(args, scene_manifest, prediction_probe, target_size)
    annotation_probe = probe_annotation_availability(
        bool(args.probe_annotations),
        scene_manifest,
        dataset_root,
        args.subset_name,
    )
    smplx_hand_model_probe = probe_smplx_hand_model(args.smplx_model_dir, args.smplx_gender, annotation_probe)

    detector = create_hand_landmarker(args.hand_landmarker_model, bool(args.disable_mediapipe_hands))
    side_summary: dict[str, Any] = {
        "left": {
            "view_indices": [],
            "views_with_roi": 0,
            "roi_count": 0,
            "roi_pixels": 0,
            "smplx_visible_view_indices": [],
            "views_with_smplx_visible_prior": 0,
            "smplx_visible_pixels": 0,
            "prediction_support_view_indices": [],
            "views_with_prediction_support": 0,
            "prediction_support_pixels": 0,
        },
        "right": {
            "view_indices": [],
            "views_with_roi": 0,
            "roi_count": 0,
            "roi_pixels": 0,
            "smplx_visible_view_indices": [],
            "views_with_smplx_visible_prior": 0,
            "smplx_visible_pixels": 0,
            "prediction_support_view_indices": [],
            "views_with_prediction_support": 0,
            "prediction_support_pixels": 0,
        },
    }
    per_view: dict[str, Any] = {}
    all_roi_rows: list[dict[str, Any]] = []
    try:
        for view_idx in view_indices:
            scene = load_scene_view(scene_dir, int(view_idx), (target_size, target_size))
            body_mask = scene.mask.astype(bool)
            view = scene_manifest["exported_views"][view_idx]
            camera_id = normalize_camera_id(view.get("camera_id", view_idx))
            mp_rois = mediapipe_hand_rois(scene.rgb, body_mask, detector, pad=int(args.hand_box_pad))
            if mp_rois is None or not mp_rois[0]:
                fallback_rois, hand_summary = fallback_hand_rois(scene.rgb, body_mask)
                rois = fallback_rois if mp_rois is None else mp_rois[0] + fallback_rois
                if mp_rois is not None:
                    hand_summary = {
                        "roi_source": "mediapipe_empty_then_skin_extremity_fallback",
                        "mediapipe_summary": mp_rois[1],
                        "fallback_summary": hand_summary,
                    }
            else:
                rois, hand_summary = mp_rois
            roi_rows = [
                summarize_roi(
                    roi,
                    body_mask,
                    int(view_idx),
                    camera_id,
                    scene,
                    prior_probe,
                    prediction_probe,
                    camera_probe,
                    int(args.patch_size),
                )
                for roi in rois
            ]
            update_side_summary(side_summary, roi_rows)
            all_roi_rows.extend(roi_rows)
            per_view[str(view_idx)] = {
                "camera_id": camera_id,
                "body_pixels": int(body_mask.sum()),
                "body_bbox_xyxy": bbox_from_mask(body_mask),
                "hand_roi_summary": hand_summary,
                "hand_rois": roi_rows,
            }
    finally:
        if detector is not None:
            detector.close()

    cache: dict[str, Any] = {
        "task": "b_hand_evidence_cache",
        "status": "diagnostic_only_preflight_cache",
        "contract": {
            "local_only": True,
            "no_train": True,
            "no_infer": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "no_cloud": True,
            "not_mediapipe_patch_teacher": True,
            "not_smplx_hand_residual_success_claim": True,
        },
        "strict_gate_truth": STRICT_GATE_TRUTH,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "target_size": int(target_size),
        "view_indices": view_indices,
        "source_inventory": source_inventory(),
        "predictions_availability": public_probe_summary(prediction_probe),
        "smplx_prior_availability": {
            **public_probe_summary(prior_probe),
            "smplx_hand_model_probe": public_probe_summary(smplx_hand_model_probe),
            "annotation_probe": public_probe_summary(annotation_probe),
        },
        "camera_availability": public_camera_summary(camera_probe),
        "visible_view_summary": side_summary,
        "per_view": per_view,
        "hand_crop_metadata": all_roi_rows,
        "hggt_b0_contract": hggt_b0_contract(side_summary),
        "elapsed_seconds": float(time.perf_counter() - start),
        "outputs": [str(cache_path), str(report_path)],
    }
    cache["stop_conditions"] = stop_conditions(cache)
    write_json(cache_path, cache)
    write_markdown_report(report_path, json_ready(cache))
    print(json.dumps(json_ready(public_probe_summary(cache, drop_arrays=False)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
