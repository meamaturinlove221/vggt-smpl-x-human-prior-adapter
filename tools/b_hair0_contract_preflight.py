from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TOKEN_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_QUERY_EVIDENCE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/B_hair0_contract_preflight_hybrid6_layer23")
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_hair0_contract_preflight_status.md")

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT_FLAGS = {
    "research_only": True,
    "local_only": True,
    "fail_closed": True,
    "contract_preflight_only": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_cloud": True,
    "no_predictions_write": True,
    "no_checkpoint_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
}
FORBIDDEN_OUTPUT_TOKENS = (
    "prediction",
    "predictions",
    "teacher",
    "candidate",
    "strict_pass",
    "strict_registry",
    "checkpoint",
    "cloud",
)
ALLOWED_OUTPUT_FRAGMENT = "output/surface_research_preflight_local/b_hair0_"
ALLOWED_REPORT_PREFIX = "20260507_b_hair0_"
REQUIRED_TEMPLATE_KEYS = (
    "hybrid_vertices",
    "hybrid_faces",
    "part_ids",
    "head_vertex_mask",
    "hairline_vertex_mask",
)
REQUIRED_TOKEN_KEYS = ("tokens", "patch_start_idx", "selected_view_indices")
REQUIRED_QUERY_KEYS = (
    "query_positions",
    "query_families",
    "support",
    "token_ids",
    "uv",
    "depth",
    "selected_view_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only, fail-closed B-hair0 contract/preflight. It validates "
            "local scene/template/token/query inputs and reports scalp, hairline, "
            "and head-top support metrics. It never writes predictions, teachers, "
            "candidates, checkpoints, strict pass state, registries, or cloud jobs."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-support-views", type=int, default=2)
    parser.add_argument("--emit-diagnostic-ply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stat_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "finite": 0}
    if not np.issubdtype(arr.dtype, np.number):
        return {"count": int(arr.size), "finite": int(arr.size)}
    finite = np.isfinite(arr)
    if not finite.any():
        return {"count": int(arr.size), "finite": 0}
    data = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(data.size),
        "min": float(data.min()),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.quantile(data, 0.50)),
        "mean": float(data.mean()),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(data.max()),
    }


def resolve_existing_file(path: Path, errors: list[str], label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        errors.append(f"missing {label}: {resolved}")
    return resolved


def load_npz_optional(path: Path, required: tuple[str, ...], errors: list[str], label: str) -> dict[str, np.ndarray]:
    resolved = resolve_existing_file(path, errors, label)
    if not resolved.is_file():
        return {}
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            missing = [name for name in required if name not in payload.files]
            if missing:
                errors.append(f"{label} missing required arrays {missing}: {resolved}")
            return {name: np.asarray(payload[name]) for name in payload.files}
    except Exception as exc:
        errors.append(f"failed to load {label}: {resolved}: {exc!r}")
        return {}


def fail_closed_path_guard(output_dir: Path, status_report: Path) -> None:
    output_text = str(output_dir).replace("\\", "/").lower()
    report_name = status_report.name.lower()
    if ALLOWED_OUTPUT_FRAGMENT not in output_text:
        raise ValueError(f"Refusing B-hair0 output outside allowed output lane: {output_dir}")
    if not report_name.startswith(ALLOWED_REPORT_PREFIX):
        raise ValueError(f"Refusing B-hair0 report outside allowed report prefix: {status_report}")
    for path in (output_dir, status_report):
        text = str(path).replace("\\", "/").lower()
        bad = [token for token in FORBIDDEN_OUTPUT_TOKENS if token in text]
        if bad:
            raise ValueError(f"Refusing to write to path with forbidden tokens {bad}: {path}")


def output_manifest(output_dir: Path, status_report: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "b_hair0_contract_preflight_summary.json",
        "report_md": output_dir / "b_hair0_contract_preflight_report.md",
        "contract_json": output_dir / "b_hair0_contract.json",
        "diagnostic_arrays": output_dir / "b_hair0_diagnostic_arrays.npz",
        "blocked_next_artifact": output_dir / "b_hair0_blocked_next_artifact.json",
        "diagnostic_ply": output_dir / "b_hair0_diagnostic_support_points_research_only.ply",
        "status_report": status_report,
    }


def ensure_no_existing_outputs(paths: dict[str, Path], overwrite: bool, emit_ply: bool) -> None:
    if overwrite:
        return
    checked = [
        path
        for key, path in paths.items()
        if key != "diagnostic_ply" or emit_ply
    ]
    existing = [str(path) for path in checked if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing B-hair0 outputs without --overwrite:\n" + "\n".join(existing))


def sorted_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_scene_path(scene_dir: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    scene_relative = scene_dir / path
    if scene_relative.exists():
        return scene_relative.resolve()
    return path.resolve()


def load_json_optional(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        errors.append(f"missing {label}: {resolved}")
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse {label}: {resolved}: {exc!r}")
        return {}


def manifest_view_paths(scene_dir: Path, manifest: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    image_paths = [
        resolve_scene_path(scene_dir, view["image_path"])
        for view in manifest.get("exported_views", [])
        if isinstance(view, dict) and view.get("image_path")
    ]
    mask_paths = [
        resolve_scene_path(scene_dir, view["mask_path"])
        for view in manifest.get("exported_views", [])
        if isinstance(view, dict) and view.get("mask_path")
    ]
    if image_paths and mask_paths:
        return image_paths, mask_paths
    return sorted_image_files(scene_dir / "images"), sorted_image_files(scene_dir / "masks")


def align_intrinsics_for_pad_mode(intrinsic: np.ndarray, image_size_wh: list[int] | tuple[int, int], target_size: int) -> np.ndarray:
    width, height = int(image_size_wh[0]), int(image_size_wh[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for intrinsic alignment: {(width, height)}")
    if width >= height:
        new_width = int(target_size)
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(width * (new_height / height) / 14) * 14
    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))
    scale_x = new_width / float(width)
    scale_y = new_height / float(height)
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    aligned = intrinsic.astype(np.float32).copy()
    aligned[0, 0] *= scale_x
    aligned[1, 1] *= scale_y
    aligned[0, 2] = intrinsic[0, 2] * scale_x + pad_left
    aligned[1, 2] = intrinsic[1, 2] * scale_y + pad_top
    return aligned


def align_intrinsics_for_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    source_size = view.get("source_image_size") or view.get("image_size") or [target_size, target_size]
    aligned = align_intrinsics_for_pad_mode(intrinsic, source_size, int(target_size))
    meta = view.get("preprocess_meta") or {}
    transform = meta.get("transform")
    bbox = meta.get("crop_bbox_xyxy")
    if transform not in {"crop_pad_to_square", "raw_crop_pad_to_square"} or not bbox or len(bbox) != 4:
        return aligned

    x0, y0, x1, y1 = [float(v) for v in bbox]
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    if crop_w >= crop_h:
        new_w = float(target_size)
        new_h = float(round(crop_h * (new_w / crop_w) / 14.0) * 14)
    else:
        new_h = float(target_size)
        new_w = float(round(crop_w * (new_h / crop_h) / 14.0) * 14)
    new_w = max(14.0, new_w)
    new_h = max(14.0, new_h)
    scale_x = new_w / crop_w
    scale_y = new_h / crop_h
    pad_left = (float(target_size) - new_w) * 0.5
    pad_top = (float(target_size) - new_h) * 0.5
    base = intrinsic.astype(np.float32) if transform == "raw_crop_pad_to_square" else aligned.astype(np.float32)
    out = base.copy()
    out[0, 0] *= scale_x
    out[1, 1] *= scale_y
    out[0, 2] = (base[0, 2] - x0) * scale_x + pad_left
    out[1, 2] = (base[1, 2] - y0) * scale_y + pad_top
    return out


def load_camera_sidecar(scene_dir: Path, errors: list[str]) -> dict[str, Any]:
    path = scene_dir / "camera_params_sidecar.npz"
    payload = load_npz_optional(path, ("camera_ids", "intrinsics", "cam_to_world", "world_to_cam"), errors, "camera sidecar")
    if not payload:
        return {"path": str(path.resolve()), "exists": False, "camera_count": 0, "cameras": {}}
    camera_ids = np.asarray(payload["camera_ids"]).astype(str).reshape(-1)
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    cam_to_world = np.asarray(payload["cam_to_world"], dtype=np.float32)
    world_to_cam = np.asarray(payload["world_to_cam"], dtype=np.float32)
    if not (camera_ids.shape[0] == intrinsics.shape[0] == cam_to_world.shape[0] == world_to_cam.shape[0]):
        errors.append(f"camera sidecar view-count mismatch: {path.resolve()}")
    if intrinsics.shape[1:] != (3, 3):
        errors.append(f"camera sidecar intrinsics expected [V,3,3], got {intrinsics.shape}")
    if world_to_cam.shape[1:] != (4, 4):
        errors.append(f"camera sidecar world_to_cam expected [V,4,4], got {world_to_cam.shape}")
    finite = bool(np.isfinite(intrinsics).all() and np.isfinite(cam_to_world).all() and np.isfinite(world_to_cam).all())
    if not finite:
        errors.append("camera sidecar arrays contain non-finite values")
    cameras = {
        str(camera_id): {
            "intrinsic": intrinsics[idx].astype(np.float32),
            "cam_to_world": cam_to_world[idx].astype(np.float32),
            "world_to_cam": world_to_cam[idx].astype(np.float32),
        }
        for idx, camera_id in enumerate(camera_ids)
    }
    return {
        "path": str(path.resolve()),
        "exists": True,
        "camera_count": int(camera_ids.shape[0]),
        "camera_ids_first": camera_ids[:8].tolist(),
        "intrinsics_shape": list(intrinsics.shape),
        "world_to_cam_shape": list(world_to_cam.shape),
        "finite": finite,
        "cameras": cameras,
    }


def load_scene_summary(scene_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[Path], list[Path], dict[str, Any], list[str]]:
    errors: list[str] = []
    resolved = scene_dir.expanduser().resolve()
    if not resolved.is_dir():
        errors.append(f"missing scene_dir: {resolved}")
    manifest = load_json_optional(resolved / "scene_manifest.json", errors, "scene_manifest.json") if resolved.exists() else {}
    images, masks = manifest_view_paths(resolved, manifest) if manifest else (sorted_image_files(resolved / "images"), sorted_image_files(resolved / "masks"))
    if not images:
        errors.append(f"missing scene images under: {resolved / 'images'}")
    if not masks:
        errors.append(f"missing scene masks under: {resolved / 'masks'}")
    if images and masks and len(images) != len(masks):
        errors.append(f"scene image/mask count mismatch: {len(images)} vs {len(masks)}")
    missing_images = [str(path) for path in images if not path.is_file()]
    missing_masks = [str(path) for path in masks if not path.is_file()]
    if missing_images:
        errors.append(f"manifest references missing images: {missing_images[:4]}")
    if missing_masks:
        errors.append(f"manifest references missing masks: {missing_masks[:4]}")
    camera = load_camera_sidecar(resolved, errors) if resolved.exists() else {"exists": False, "camera_count": 0, "cameras": {}}
    summary = {
        "path": str(resolved),
        "image_count": len(images),
        "mask_count": len(masks),
        "manifest_exists": bool(manifest),
        "manifest_exported_view_count": len(manifest.get("exported_views", [])) if isinstance(manifest, dict) else 0,
        "manifest_seq_id": manifest.get("seq_id") if isinstance(manifest, dict) else None,
        "manifest_frame_id": manifest.get("frame_id") if isinstance(manifest, dict) else None,
        "sample_images": [path.name for path in images[:8]],
        "sample_masks": [path.name for path in masks[:8]],
        "camera_sidecar": {key: value for key, value in camera.items() if key != "cameras"},
        "errors": errors,
        "compatible": not errors and len(images) > 0 and len(images) == len(masks),
    }
    return summary, manifest, images, masks, camera, errors


def token_summary(path: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = load_npz_optional(path, REQUIRED_TOKEN_KEYS, errors, "token cache")
    if not payload:
        return {"path": str(path.expanduser().resolve()), "compatible": False, "errors": ["missing token cache"]}, {}
    local_errors: list[str] = []
    tokens = np.asarray(payload["tokens"])
    patch_start_idx = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        local_errors.append(f"tokens expected [1,V,T,C], got {tokens.shape}")
    view_count = int(tokens.shape[1]) if tokens.ndim == 4 else 0
    token_count = int(tokens.shape[2]) if tokens.ndim == 4 else 0
    feature_dim = int(tokens.shape[3]) if tokens.ndim == 4 else 0
    patch_count = max(0, token_count - patch_start_idx)
    patch_grid = int(round(math.sqrt(max(patch_count, 1))))
    if patch_grid * patch_grid != patch_count:
        local_errors.append(f"patch token count is not square: {patch_count}")
    if selected.shape[0] != view_count:
        local_errors.append(f"selected_view_indices length {selected.shape[0]} != token view_count {view_count}")
    if not np.issubdtype(tokens.dtype, np.floating):
        local_errors.append(f"tokens dtype should be floating, got {tokens.dtype}")
    if not np.isfinite(tokens.astype(np.float32)).all():
        local_errors.append("tokens contain non-finite values")
    errors.extend(local_errors)
    return {
        "path": str(path.expanduser().resolve()),
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "view_count": view_count,
        "token_count": token_count,
        "feature_dim": feature_dim,
        "patch_start_idx": patch_start_idx,
        "patch_count": patch_count,
        "patch_grid": patch_grid,
        "selected_view_indices": selected.astype(int).tolist(),
        "errors": local_errors,
        "compatible": not local_errors,
    }, payload


def template_summary(path: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = load_npz_optional(path, REQUIRED_TEMPLATE_KEYS, errors, "template payload")
    if not payload:
        return {"path": str(path.expanduser().resolve()), "compatible": False, "errors": ["missing template payload"]}, {}
    local_errors: list[str] = []
    vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
    faces = np.asarray(payload["hybrid_faces"], dtype=np.int64)
    part_ids = np.asarray(payload["part_ids"], dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        local_errors.append(f"hybrid_vertices expected [N,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        local_errors.append(f"hybrid_faces expected [F,3], got {faces.shape}")
    if part_ids.shape[0] != vertices.shape[0]:
        local_errors.append(f"part_ids length {part_ids.shape[0]} != vertex count {vertices.shape[0]}")
    if not np.isfinite(vertices).all():
        local_errors.append("hybrid_vertices contain non-finite values")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        local_errors.append("hybrid_faces contain out-of-range indices")
    mask_rows: dict[str, Any] = {}
    for name in ("head_vertex_mask", "hairline_vertex_mask", "face_front_vertex_mask"):
        if name not in payload:
            mask_rows[name] = {"present": False, "count": 0}
            if name != "face_front_vertex_mask":
                local_errors.append(f"missing required template mask: {name}")
            continue
        mask = np.asarray(payload[name], dtype=bool).reshape(-1)
        if mask.shape[0] != vertices.shape[0]:
            local_errors.append(f"{name} length {mask.shape[0]} != vertex count {vertices.shape[0]}")
        mask_rows[name] = {"present": True, "count": int(mask.sum())}
    part_names = np.asarray(payload["part_names"]).astype(str).tolist() if "part_names" in payload else []
    unique_part_ids, counts = np.unique(part_ids, return_counts=True) if part_ids.size else (np.zeros(0), np.zeros(0))
    errors.extend(local_errors)
    return {
        "path": str(path.expanduser().resolve()),
        "hybrid_vertex_count": int(vertices.shape[0]) if vertices.ndim == 2 else 0,
        "hybrid_face_count": int(faces.shape[0]) if faces.ndim == 2 else 0,
        "part_names": part_names,
        "part_counts": {str(int(pid)): int(count) for pid, count in zip(unique_part_ids, counts)},
        "required_mask_counts": mask_rows,
        "bbox_min": vertices.min(axis=0).astype(float).tolist() if vertices.ndim == 2 and vertices.shape[0] else [],
        "bbox_max": vertices.max(axis=0).astype(float).tolist() if vertices.ndim == 2 and vertices.shape[0] else [],
        "errors": local_errors,
        "compatible": not local_errors,
    }, payload


def query_summary(path: Path, token: dict[str, Any], errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = load_npz_optional(path, REQUIRED_QUERY_KEYS, errors, "query evidence")
    if not payload:
        return {"path": str(path.expanduser().resolve()), "compatible": False, "errors": ["missing query evidence"]}, {}
    local_errors: list[str] = []
    positions = np.asarray(payload["query_positions"], dtype=np.float32)
    families = np.asarray(payload["query_families"]).astype(str).reshape(-1)
    support = np.asarray(payload["support"], dtype=np.int64).reshape(-1)
    token_ids = np.asarray(payload["token_ids"], dtype=np.int64)
    uv = np.asarray(payload["uv"], dtype=np.float32)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    n = int(families.shape[0])
    view_count = int(token.get("view_count", 0))
    expected = {
        "query_positions": (n, 3),
        "support": (n,),
        "token_ids": (n, view_count),
        "uv": (n, view_count, 2),
        "depth": (n, view_count),
    }
    actual = {
        "query_positions": tuple(positions.shape),
        "support": tuple(support.shape),
        "token_ids": tuple(token_ids.shape),
        "uv": tuple(uv.shape),
        "depth": tuple(depth.shape),
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            local_errors.append(f"{name} expected {expected_shape}, got {actual[name]}")
    if selected.astype(int).tolist() != token.get("selected_view_indices", []):
        local_errors.append("query selected_view_indices do not match token cache")
    if not np.isfinite(positions).all():
        local_errors.append("query_positions contain non-finite values")
    family_rows: dict[str, Any] = {}
    for family in sorted(set(families.tolist()) | {"hairline"}):
        mask = families == family
        values = support[mask]
        family_rows[family] = {
            "count": int(mask.sum()),
            "present": bool(mask.sum() > 0),
            "support_stats": stat_array(values),
            "support_ge_1": int((values >= 1).sum()) if values.size else 0,
            "support_ge_2": int((values >= 2).sum()) if values.size else 0,
            "support_ge_3": int((values >= 3).sum()) if values.size else 0,
            "two_view_or_more_ratio": float((values >= 2).mean()) if values.size else 0.0,
        }
    errors.extend(local_errors)
    return {
        "path": str(path.expanduser().resolve()),
        "query_count": n,
        "selected_view_indices": selected.astype(int).tolist(),
        "family_rows": family_rows,
        "support_stats": stat_array(support),
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "errors": local_errors,
        "compatible": not local_errors,
    }, payload


def load_mask(path: Path, target_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 127


def project_points(points: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (np.asarray(intrinsic, dtype=np.float32) @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32)


def patch_ids_from_uv(
    uv: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    patch_start_idx: int,
    patch_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    u = uv[:, 0]
    v = uv[:, 1]
    inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-6)
        & (u >= 0.0)
        & (u < target_size)
        & (v >= 0.0)
        & (v < target_size)
    )
    xi = np.clip(np.floor(u).astype(np.int64), 0, target_size - 1)
    yi = np.clip(np.floor(v).astype(np.int64), 0, target_size - 1)
    inside &= mask[yi, xi]
    patch_size = float(target_size) / float(max(patch_grid, 1))
    patch_x = np.clip(np.floor(u / patch_size).astype(np.int64), 0, patch_grid - 1)
    patch_y = np.clip(np.floor(v / patch_size).astype(np.int64), 0, patch_grid - 1)
    patch_ids = patch_start_idx + patch_y * patch_grid + patch_x
    patch_ids[~inside] = -1
    return patch_ids.astype(np.int64), inside.astype(bool)


def region_masks(template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vertices = np.asarray(template["hybrid_vertices"], dtype=np.float32)
    head = np.asarray(template["head_vertex_mask"], dtype=bool).reshape(-1)
    hairline = np.asarray(template["hairline_vertex_mask"], dtype=bool).reshape(-1)
    y = vertices[:, 1]
    z = vertices[:, 2]
    head_y = y[head] if np.any(head) else y
    head_z = z[head] if np.any(head) else z
    head_top_y_threshold = float(np.quantile(head_y, 0.20)) if head_y.size else 0.0
    scalp_y_threshold = float(np.quantile(head_y, 0.38)) if head_y.size else 0.0
    scalp_z_low = float(np.quantile(head_z, 0.20)) if head_z.size else -np.inf
    scalp_z_high = float(np.quantile(head_z, 0.95)) if head_z.size else np.inf
    scalp = head & (y <= scalp_y_threshold) & (z >= scalp_z_low) & (z <= scalp_z_high)
    head_top = head & (y <= head_top_y_threshold)
    hair_ring = np.zeros((vertices.shape[0],), dtype=bool)
    if "hair_ring_vertex_ids" in template:
        ring_ids = np.asarray(template["hair_ring_vertex_ids"], dtype=np.int64).reshape(-1)
        ring_ids = ring_ids[(ring_ids >= 0) & (ring_ids < vertices.shape[0])]
        hair_ring[ring_ids] = True
    if not hair_ring.any():
        hair_ring = hairline.copy()
    return {
        "hairline": hairline,
        "scalp": scalp,
        "head_top": head_top,
        "hair_ring": hair_ring,
    }


def support_for_regions(
    regions: dict[str, np.ndarray],
    scene_manifest: dict[str, Any],
    mask_paths: list[Path],
    camera: dict[str, Any],
    token: dict[str, Any],
    template: dict[str, np.ndarray],
    target_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    vertices = np.asarray(template["hybrid_vertices"], dtype=np.float32)
    selected = [int(v) for v in token.get("selected_view_indices", [])]
    patch_start_idx = int(token.get("patch_start_idx", 0))
    patch_grid = int(token.get("patch_grid", 0))
    views = scene_manifest.get("exported_views", []) if isinstance(scene_manifest, dict) else []
    cameras = camera.get("cameras", {})
    region_support_counts: dict[str, np.ndarray] = {
        name: np.zeros((int(mask.sum()),), dtype=np.int32)
        for name, mask in regions.items()
    }
    region_token_union: dict[str, set[int]] = {name: set() for name in regions}
    region_view_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in regions}
    region_uv: dict[str, list[np.ndarray]] = {name: [] for name in regions}
    region_depth: dict[str, list[np.ndarray]] = {name: [] for name in regions}

    for slot, view_index in enumerate(selected):
        if view_index >= len(views) or view_index >= len(mask_paths):
            for name in regions:
                region_view_rows[name].append({"view_index": view_index, "slot": slot, "error": "view index outside scene assets"})
            continue
        view = views[view_index]
        camera_id = str(view.get("camera_id", "")).zfill(2)
        if camera_id not in cameras:
            for name in regions:
                region_view_rows[name].append({"view_index": view_index, "slot": slot, "camera_id": camera_id, "error": "missing camera"})
            continue
        intrinsic = align_intrinsics_for_scene_view(np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam = np.asarray(cameras[camera_id]["world_to_cam"], dtype=np.float32)
        mask_image = load_mask(mask_paths[view_index], target_size)
        for name, region_mask in regions.items():
            points = vertices[region_mask]
            if points.size == 0:
                region_view_rows[name].append({"view_index": view_index, "slot": slot, "camera_id": camera_id, "point_count": 0})
                continue
            uv, depth = project_points(points, world_to_cam, intrinsic)
            token_ids, inside = patch_ids_from_uv(uv, depth, mask_image, target_size, patch_start_idx, patch_grid)
            valid_ids = token_ids[inside & (token_ids >= patch_start_idx)]
            region_support_counts[name] += inside.astype(np.int32)
            region_token_union[name].update(int(v) for v in valid_ids.tolist())
            region_uv[name].append(uv.astype(np.float32))
            region_depth[name].append(depth.astype(np.float32))
            region_view_rows[name].append(
                {
                    "view_index": int(view_index),
                    "slot": int(slot),
                    "camera_id": camera_id,
                    "point_count": int(points.shape[0]),
                    "visible_mask_gated_points": int(inside.sum()),
                    "visible_ratio": float(inside.mean()) if inside.size else 0.0,
                    "unique_aggregator_tokens": int(np.unique(valid_ids).size) if valid_ids.size else 0,
                    "uv_bbox_xyxy": (
                        [
                            float(np.nanmin(uv[inside, 0])),
                            float(np.nanmin(uv[inside, 1])),
                            float(np.nanmax(uv[inside, 0])),
                            float(np.nanmax(uv[inside, 1])),
                        ]
                        if np.any(inside)
                        else []
                    ),
                    "depth_stats": stat_array(depth[inside]) if np.any(inside) else {"count": 0, "finite": 0},
                }
            )

    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, region_mask in regions.items():
        points = vertices[region_mask].astype(np.float32)
        support = region_support_counts[name]
        arrays[f"{name}_points"] = points
        arrays[f"{name}_support"] = support.astype(np.int32)
        arrays[f"{name}_mask_indices"] = np.flatnonzero(region_mask).astype(np.int64)
        rows[name] = {
            "point_count": int(points.shape[0]),
            "views": int(len(selected)),
            "support_stats": stat_array(support),
            "support_ge_1": int((support >= 1).sum()),
            "support_ge_2": int((support >= 2).sum()),
            "support_ge_3": int((support >= 3).sum()),
            "support_ge_1_ratio": float((support >= 1).mean()) if support.size else 0.0,
            "support_ge_2_ratio": float((support >= 2).mean()) if support.size else 0.0,
            "support_ge_3_ratio": float((support >= 3).mean()) if support.size else 0.0,
            "unique_aggregator_tokens": int(len(region_token_union[name])),
            "bbox_min": points.min(axis=0).astype(float).tolist() if points.size else [],
            "bbox_max": points.max(axis=0).astype(float).tolist() if points.size else [],
            "per_view": region_view_rows[name],
        }
    arrays["selected_view_indices"] = np.asarray(selected, dtype=np.int32)
    arrays["research_only"] = np.asarray([1], dtype=np.int32)
    arrays["strict_candidate_passes"] = np.asarray([0], dtype=np.int32)
    arrays["strict_teacher_passes"] = np.asarray([0], dtype=np.int32)
    return rows, arrays


def query_hairline_support(query: dict[str, np.ndarray], min_support_views: int) -> dict[str, Any]:
    if not query:
        return {"available": False, "reason": "query evidence unavailable"}
    families = np.asarray(query["query_families"]).astype(str).reshape(-1)
    support = np.asarray(query["support"], dtype=np.int64).reshape(-1)
    mask = families == "hairline"
    values = support[mask]
    return {
        "available": True,
        "family": "hairline",
        "query_count": int(mask.sum()),
        "support_stats": stat_array(values),
        "support_ge_1": int((values >= 1).sum()) if values.size else 0,
        "support_ge_min": int((values >= int(min_support_views)).sum()) if values.size else 0,
        "support_ge_min_ratio": float((values >= int(min_support_views)).mean()) if values.size else 0.0,
    }


def contract_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "B_hair0_contract_preflight_v1",
        "status": summary["status"],
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "allowed_inputs": {
            "scene": summary["scene"]["path"],
            "template_payload": summary["template_payload"]["path"],
            "token_cache": summary["token_cache"]["path"],
            "query_evidence": summary["query_evidence"]["path"],
        },
        "required_support_regions": ["hairline", "scalp", "head_top", "hair_ring"],
        "required_report_metrics": [
            "point_count",
            "support_ge_1_ratio",
            "support_ge_2_ratio",
            "unique_aggregator_tokens",
            "per_view_visible_mask_gated_points",
        ],
        "forbidden_outputs": [
            "predictions.npz",
            "teacher package",
            "candidate package",
            "strict registry entry",
            "checkpoint",
            "cloud job",
        ],
        "next_step_boundary": (
            "A later B-hair backend may consume this diagnostic contract, but this "
            "preflight cannot export teacher/candidate data or claim strict success."
        ),
    }


def blocked_next_artifact(errors: list[str], support_rows: dict[str, Any], min_support_views: int) -> dict[str, Any]:
    if errors:
        return {
            "status": "blocked_missing_or_invalid_inputs",
            "exact_next_artifact_needed": errors[0],
            "all_blockers": errors,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "teacher_candidate_export": "blocked",
            "cloud": "blocked",
            "predictions": "blocked",
        }
    weak_regions = [
        name
        for name, row in support_rows.items()
        if float(row.get("support_ge_2_ratio", 0.0)) <= 0.0 or int(row.get("point_count", 0)) <= 0
    ]
    if weak_regions:
        return {
            "status": "blocked_weak_hair_region_support",
            "exact_next_artifact_needed": (
                "A B-hair0-compatible scene/template/token/query bundle with mask-gated "
                f"{min_support_views}view support for regions: {', '.join(weak_regions)}"
            ),
            "weak_regions": weak_regions,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "teacher_candidate_export": "blocked",
            "cloud": "blocked",
            "predictions": "blocked",
        }
    return {
        "status": "inputs_valid_backend_not_implemented",
        "exact_next_artifact_needed": (
            "tools/b_hair0_backend_smoke.py implementing a research-only hair/scalp "
            "decoder smoke with real/shuffle/zero token controls and diagnostic renders only."
        ),
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "teacher_candidate_export": "blocked",
        "cloud": "blocked",
        "predictions": "blocked",
    }


def write_ply(path: Path, arrays: dict[str, np.ndarray]) -> None:
    colors = {
        "hairline": (230, 40, 80),
        "scalp": (40, 150, 230),
        "head_top": (245, 190, 40),
        "hair_ring": (120, 80, 220),
    }
    rows: list[tuple[float, float, float, int, int, int, int]] = []
    for name, rgb in colors.items():
        points = np.asarray(arrays.get(f"{name}_points", np.zeros((0, 3))), dtype=np.float32)
        support = np.asarray(arrays.get(f"{name}_support", np.zeros((points.shape[0],))), dtype=np.int32).reshape(-1)
        if points.shape[0] != support.shape[0]:
            support = np.zeros((points.shape[0],), dtype=np.int32)
        for point, count in zip(points, support):
            gain = min(1.0, 0.35 + 0.2 * float(count))
            color = tuple(int(round(channel * gain)) for channel in rgb)
            rows.append((float(point[0]), float(point[1]), float(point[2]), color[0], color[1], color[2], int(count)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment research_only B-hair0 diagnostic support points; not teacher, not candidate\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int support_views\n")
        handle.write("end_header\n")
        for row in rows:
            handle.write("%.7f %.7f %.7f %d %d %d %d\n" % row)


def build_markdown(summary: dict[str, Any]) -> str:
    paths = summary["written_paths"]
    lines = [
        "# B-hair0 Contract Preflight",
        "",
        "Status: `" + summary["status"] + "`",
        "",
        "This is a research-only fail-closed B-hair0 contract/preflight. It validates",
        "local scene, template, token, and query inputs and reports scalp, hairline,",
        "head-top, and hair-ring support metrics. It does not write predictions,",
        "teachers, candidates, checkpoints, strict pass state, registries, or cloud jobs.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud_train_infer_export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "```",
        "",
        "## Compatibility",
        "",
        "```json",
        json.dumps(json_ready(summary["compatibility"]), indent=2, sort_keys=True),
        "```",
        "",
        "## Observed Inputs",
        "",
        f"- Scene images/masks: {summary['scene']['image_count']} / {summary['scene']['mask_count']}",
        f"- Cameras: {summary['scene']['camera_sidecar'].get('camera_count')}",
        f"- VGGT tokens: shape={summary['token_cache'].get('shape')}, patch_grid={summary['token_cache'].get('patch_grid')}",
        f"- Template vertices/faces: {summary['template_payload'].get('hybrid_vertex_count')} / {summary['template_payload'].get('hybrid_face_count')}",
        f"- Query evidence: {summary['query_evidence'].get('query_count')} queries",
        "",
        "## Hair Support Metrics",
        "",
    ]
    for name in ("hairline", "scalp", "head_top", "hair_ring"):
        row = summary["support_metrics"].get(name, {})
        mean_support = row.get("support_stats", {}).get("mean")
        mean_text = "n/a" if mean_support is None else f"{mean_support:.3f}"
        lines.append(
            f"- `{name}`: points={row.get('point_count', 0)}, mean_support={mean_text}, "
            f"support_ge_1_ratio={row.get('support_ge_1_ratio', 0.0):.3f}, "
            f"support_ge_2_ratio={row.get('support_ge_2_ratio', 0.0):.3f}, "
            f"unique_tokens={row.get('unique_aggregator_tokens', 0)}"
        )
    lines.extend(
        [
            "",
            "## Query Hairline Readout",
            "",
            "```json",
            json.dumps(json_ready(summary["query_hairline_support"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Blocked Next Artifact",
            "",
            "```json",
            json.dumps(json_ready(summary["blocked_next_artifact"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Written Files",
            "",
        ]
    )
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    fail_closed_path_guard(args.output_dir, args.status_report)
    paths = output_manifest(args.output_dir, args.status_report)
    ensure_no_existing_outputs(paths, bool(args.overwrite), bool(args.emit_diagnostic_ply))

    all_errors: list[str] = []
    scene, manifest, _image_paths, mask_paths, camera, scene_errors = load_scene_summary(args.scene_dir)
    token, token_payload = token_summary(args.token_cache, all_errors)
    template, template_payload = template_summary(args.template_payload, all_errors)
    query, query_payload = query_summary(args.query_evidence, token, all_errors)
    all_errors = scene_errors + all_errors

    selected = token.get("selected_view_indices", [])
    if selected and scene.get("image_count", 0) and max(selected) >= int(scene.get("image_count", 0)):
        all_errors.append("token selected_view_indices reference views outside scene image count")
    if selected and int(camera.get("camera_count", 0)) and max(selected) >= int(camera.get("camera_count", 0)):
        all_errors.append("token selected_view_indices reference views outside camera sidecar count")

    support_rows: dict[str, Any] = {}
    diagnostic_arrays: dict[str, np.ndarray] = {}
    if not all_errors and template_payload and token_payload and query_payload:
        regions = region_masks(template_payload)
        support_rows, diagnostic_arrays = support_for_regions(
            regions,
            manifest,
            mask_paths,
            camera,
            token,
            template_payload,
            int(args.target_size),
        )
    else:
        support_rows = {
            name: {
                "point_count": 0,
                "views": 0,
                "support_stats": {"count": 0, "finite": 0},
                "support_ge_1": 0,
                "support_ge_2": 0,
                "support_ge_3": 0,
                "support_ge_1_ratio": 0.0,
                "support_ge_2_ratio": 0.0,
                "support_ge_3_ratio": 0.0,
                "unique_aggregator_tokens": 0,
                "per_view": [],
            }
            for name in ("hairline", "scalp", "head_top", "hair_ring")
        }

    compatibility = {
        "input_contract_compatible": not all_errors,
        "scene_compatible": bool(scene.get("compatible")),
        "token_cache_compatible": bool(token.get("compatible")),
        "template_payload_compatible": bool(template.get("compatible")),
        "query_evidence_compatible": bool(query.get("compatible")),
        "selected_view_indices_match": (
            token.get("selected_view_indices") == query.get("selected_view_indices")
            if token.get("selected_view_indices") is not None and query.get("selected_view_indices") is not None
            else False
        ),
        "backend_smoke_feasible": False,
        "errors": all_errors,
    }
    blocked = blocked_next_artifact(all_errors, support_rows, int(args.min_support_views))
    if all_errors:
        status = "research_only_fail_closed_blocked_missing_inputs_no_export"
        decision = "Blocked fail-closed on missing or invalid inputs; no teacher/candidate/cloud/prediction output was produced."
    else:
        status = "research_only_fail_closed_contract_preflight_no_export"
        decision = "B-hair0 local inputs and support metrics are recorded. Strict passes remain zero; backend/export/cloud remain blocked."

    summary = {
        "status": status,
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "scene": scene,
        "token_cache": token,
        "template_payload": template,
        "query_evidence": query,
        "support_metrics": support_rows,
        "query_hairline_support": query_hairline_support(query_payload, int(args.min_support_views)),
        "compatibility": compatibility,
        "blocked_next_artifact": blocked,
        "decision": decision,
        "written_paths": {
            key: str(path.resolve())
            for key, path in paths.items()
            if key != "diagnostic_ply" or bool(args.emit_diagnostic_ply)
        },
    }
    contract = contract_payload(summary)

    write_json(paths["contract_json"], contract)
    write_json(paths["blocked_next_artifact"], blocked)
    write_json(paths["summary_json"], summary)
    if diagnostic_arrays:
        paths["diagnostic_arrays"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths["diagnostic_arrays"], **diagnostic_arrays)
    else:
        paths["diagnostic_arrays"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths["diagnostic_arrays"],
            research_only=np.asarray([1], dtype=np.int32),
            strict_candidate_passes=np.asarray([0], dtype=np.int32),
            strict_teacher_passes=np.asarray([0], dtype=np.int32),
        )
    if args.emit_diagnostic_ply:
        write_ply(paths["diagnostic_ply"], diagnostic_arrays)
    report = build_markdown(summary)
    write_text(paths["report_md"], report)
    write_text(paths["status_report"], report)
    print(json.dumps(json_ready({"status": status, "written_paths": summary["written_paths"], "blocked_next_artifact": blocked}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
