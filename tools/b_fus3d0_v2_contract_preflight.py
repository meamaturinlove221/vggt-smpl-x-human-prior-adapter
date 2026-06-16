from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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
DEFAULT_LATENT_GRID = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D0_v2_contract_preflight_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d0_v2_contract_preflight_status.md")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_QUERY_FAMILIES = ("full_body", "face_core", "hairline", "left_hand", "right_hand")
EXPECTED_TEMPLATE_MASKS = (
    "face_front_vertex_mask",
    "head_vertex_mask",
    "hairline_vertex_mask",
    "left_hand_vertex_mask",
    "right_hand_vertex_mask",
    "lower_clothing_vertex_mask",
)
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
    "contract_preflight_only": True,
    "fail_closed": True,
    "representation_is_different_from_b19_query_to_carrier": True,
    "no_b19_tuning": True,
    "no_train": True,
    "no_optimization": True,
    "no_cloud": True,
    "no_predictions_write": True,
    "no_checkpoint_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "not_teacher": True,
    "not_candidate": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only fail-closed B-Fus3D0-v2 representation contract/preflight. "
            "It validates local VGGT token/RGB/mask/camera/template evidence for a genuinely "
            "different canonical latent-grid or local surface-token-grid SDF backend, then writes "
            "JSON/Markdown contract artifacts only. It never writes predictions, teachers, "
            "candidates, checkpoints, strict passes, registries, or cloud jobs."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--latent-grid", type=Path, default=DEFAULT_LATENT_GRID)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
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
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing required arrays: {missing}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def fail_closed_path_guard(output_dir: Path, status_report: Path) -> None:
    for path in (output_dir, status_report):
        text = str(path).replace("\\", "/").lower()
        bad = [token for token in FORBIDDEN_OUTPUT_TOKENS if token in text]
        if bad:
            raise ValueError(f"Refusing to write to path with forbidden tokens {bad}: {path}")


def output_manifest(output_dir: Path, status_report: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "b_fus3d0_v2_contract_preflight_summary.json",
        "report": output_dir / "b_fus3d0_v2_contract_preflight_report.md",
        "representation_contract": output_dir / "b_fus3d0_v2_representation_contract.json",
        "cross_attention_contract": output_dir / "b_fus3d0_v2_cross_attention_contract.json",
        "refinement_contract": output_dir / "b_fus3d0_v2_refinement_contract.json",
        "output_schema": output_dir / "b_fus3d0_v2_output_schema.json",
        "blocked_next_artifact": output_dir / "b_fus3d0_v2_blocked_next_artifact.json",
        "status_report": status_report,
    }


def ensure_no_existing_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        joined = "\n".join(existing)
        raise FileExistsError(f"Refusing to overwrite existing B-Fus3D0-v2 outputs without --overwrite:\n{joined}")


def scene_summary(scene_dir: Path) -> dict[str, Any]:
    resolved = scene_dir.expanduser().resolve()
    image_dir = resolved / "images"
    mask_dir = resolved / "masks"
    manifest_path = resolved / "scene_manifest.json"
    camera_path = resolved / "camera_params_sidecar.npz"
    errors: list[str] = []
    if not resolved.is_dir():
        errors.append(f"missing scene_dir: {resolved}")
    if not image_dir.is_dir():
        errors.append(f"missing images dir: {image_dir}")
    if not mask_dir.is_dir():
        errors.append(f"missing masks dir: {mask_dir}")
    if not manifest_path.is_file():
        errors.append(f"missing scene_manifest.json: {manifest_path}")
    if not camera_path.is_file():
        errors.append(f"missing camera_params_sidecar.npz: {camera_path}")

    images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
    masks = sorted(path for path in mask_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES) if mask_dir.is_dir() else []
    image_stems = {path.stem for path in images}
    mask_stems = {path.stem for path in masks}
    if images and masks and image_stems != mask_stems:
        errors.append("image/mask stem sets differ")

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    camera_info: dict[str, Any] = {"path": str(camera_path), "exists": camera_path.is_file()}
    if camera_path.is_file():
        payload = load_npz(camera_path, ("camera_ids", "intrinsics", "cam_to_world", "world_to_cam"))
        camera_ids = np.asarray(payload["camera_ids"]).astype(str).reshape(-1)
        intrinsics = np.asarray(payload["intrinsics"])
        cam_to_world = np.asarray(payload["cam_to_world"])
        world_to_cam = np.asarray(payload["world_to_cam"])
        camera_info.update(
            {
                "camera_count": int(camera_ids.shape[0]),
                "camera_ids_first": camera_ids[:8].tolist(),
                "intrinsics_shape": list(intrinsics.shape),
                "cam_to_world_shape": list(cam_to_world.shape),
                "world_to_cam_shape": list(world_to_cam.shape),
                "finite": bool(
                    np.isfinite(intrinsics).all()
                    and np.isfinite(cam_to_world).all()
                    and np.isfinite(world_to_cam).all()
                ),
            }
        )
        if intrinsics.shape != (camera_ids.shape[0], 3, 3):
            errors.append(f"intrinsics shape mismatch: {intrinsics.shape}")
        if cam_to_world.shape != (camera_ids.shape[0], 4, 4):
            errors.append(f"cam_to_world shape mismatch: {cam_to_world.shape}")
        if world_to_cam.shape != (camera_ids.shape[0], 4, 4):
            errors.append(f"world_to_cam shape mismatch: {world_to_cam.shape}")
        if not camera_info["finite"]:
            errors.append("camera arrays contain non-finite values")

    return {
        "path": str(resolved),
        "image_count": len(images),
        "mask_count": len(masks),
        "manifest_exists": manifest_path.is_file(),
        "manifest_seq_id": manifest.get("seq_id"),
        "manifest_frame_id": manifest.get("frame_id"),
        "manifest_exported_view_count": len(manifest.get("exported_views", [])),
        "sample_images": [path.name for path in images[:8]],
        "sample_masks": [path.name for path in masks[:8]],
        "camera_sidecar": camera_info,
        "errors": errors,
        "compatible": not errors and len(images) > 0 and len(images) == len(masks),
    }


def token_cache_summary(path: Path) -> dict[str, Any]:
    payload = load_npz(path, ("tokens", "patch_start_idx", "selected_view_indices"))
    tokens = np.asarray(payload["tokens"])
    patch_start_idx = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    errors: list[str] = []
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        errors.append(f"tokens expected [1,V,T,C], got {tokens.shape}")
    view_count = int(tokens.shape[1]) if tokens.ndim == 4 else 0
    token_count = int(tokens.shape[2]) if tokens.ndim == 4 else 0
    feature_dim = int(tokens.shape[3]) if tokens.ndim == 4 else 0
    patch_count = max(0, token_count - patch_start_idx)
    patch_grid = int(round(math.sqrt(max(patch_count, 1))))
    if patch_grid * patch_grid != patch_count:
        errors.append(f"patch token count is not square: {patch_count}")
    if selected.shape[0] != view_count:
        errors.append(f"selected_view_indices length {selected.shape[0]} != view_count {view_count}")
    if not np.issubdtype(tokens.dtype, np.floating):
        errors.append(f"tokens dtype should be floating, got {tokens.dtype}")

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
        "token_stats": stat_array(tokens.astype(np.float32)),
        "errors": errors,
        "compatible": not errors,
    }


def query_evidence_summary(path: Path, token_summary: dict[str, Any]) -> dict[str, Any]:
    required = (
        "query_indices",
        "query_positions",
        "query_part_ids",
        "query_families",
        "support",
        "token_ids",
        "uv",
        "depth",
        "mean_features",
        "variance_features",
        "selected_view_indices",
    )
    payload = load_npz(path, required)
    positions = np.asarray(payload["query_positions"])
    families = np.asarray(payload["query_families"]).astype(str).reshape(-1)
    support = np.asarray(payload["support"], dtype=np.int64).reshape(-1)
    token_ids = np.asarray(payload["token_ids"])
    uv = np.asarray(payload["uv"])
    depth = np.asarray(payload["depth"])
    mean_features = np.asarray(payload["mean_features"])
    variance_features = np.asarray(payload["variance_features"])
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    n = int(positions.shape[0]) if positions.ndim == 2 else int(families.shape[0])
    view_count = int(token_summary.get("view_count", 0))
    feature_dim = int(token_summary.get("feature_dim", 0))
    errors: list[str] = []

    expected_shapes = {
        "query_positions": (n, 3),
        "query_families": (n,),
        "support": (n,),
        "token_ids": (n, view_count),
        "uv": (n, view_count, 2),
        "depth": (n, view_count),
        "mean_features": (n, feature_dim),
        "variance_features": (n, feature_dim),
    }
    actual_shapes = {
        "query_positions": tuple(positions.shape),
        "query_families": tuple(families.shape),
        "support": tuple(support.shape),
        "token_ids": tuple(token_ids.shape),
        "uv": tuple(uv.shape),
        "depth": tuple(depth.shape),
        "mean_features": tuple(mean_features.shape),
        "variance_features": tuple(variance_features.shape),
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            errors.append(f"{name} expected {expected}, got {actual_shapes[name]}")
    if selected.astype(int).tolist() != token_summary.get("selected_view_indices", []):
        errors.append("query selected_view_indices do not match token cache")
    if not np.isfinite(positions).all():
        errors.append("query_positions contain non-finite values")
    if not np.isfinite(mean_features.astype(np.float32)).all():
        errors.append("mean_features contain non-finite values")

    family_rows: dict[str, Any] = {}
    for family in sorted(set(families.tolist()) | set(EXPECTED_QUERY_FAMILIES)):
        mask = families == family
        count = int(mask.sum())
        family_rows[family] = {
            "count": count,
            "present": count > 0,
            "support_stats": stat_array(support[mask]) if count else {"count": 0, "finite": 0},
            "two_view_or_more": int((support[mask] >= 2).sum()) if count else 0,
            "two_view_or_more_ratio": float((support[mask] >= 2).mean()) if count else 0.0,
        }
    missing = [family for family in EXPECTED_QUERY_FAMILIES if family_rows[family]["count"] <= 0]
    if missing:
        errors.append(f"missing query families: {missing}")

    return {
        "path": str(path.expanduser().resolve()),
        "query_count": n,
        "shapes": {name: list(shape) for name, shape in actual_shapes.items()},
        "selected_view_indices": selected.astype(int).tolist(),
        "family_rows": family_rows,
        "support_stats": stat_array(support),
        "position_bbox_min": positions.min(axis=0).astype(float).tolist() if positions.ndim == 2 and n else [],
        "position_bbox_max": positions.max(axis=0).astype(float).tolist() if positions.ndim == 2 and n else [],
        "errors": errors,
        "compatible": not errors,
    }


def latent_grid_summary(path: Path, token_summary: dict[str, Any]) -> dict[str, Any]:
    required = (
        "points",
        "visible_count",
        "mask_count",
        "token_count",
        "occupancy_ratio",
        "rgb_variance",
        "rgb_range",
        "token_cosine",
        "evidence_score",
        "boundary_like",
        "selected_view_indices",
    )
    payload = load_npz(path, required)
    points = np.asarray(payload["points"])
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    errors: list[str] = []
    if points.ndim != 2 or points.shape[1] != 3:
        errors.append(f"points expected [M,3], got {points.shape}")
    m = int(points.shape[0]) if points.ndim == 2 else 0
    for key in required:
        if key in ("points", "selected_view_indices"):
            continue
        arr = np.asarray(payload[key])
        if arr.shape[0] != m:
            errors.append(f"{key} expected length {m}, got {arr.shape}")
    if selected.astype(int).tolist() != token_summary.get("selected_view_indices", []):
        errors.append("latent-grid selected_view_indices do not match token cache")
    if points.size and not np.isfinite(points).all():
        errors.append("latent-grid points contain non-finite values")

    point_count = int(points.shape[0]) if points.ndim == 2 else 0
    grid_resolution_guess = int(round(point_count ** (1.0 / 3.0))) if point_count else 0
    return {
        "path": str(path.expanduser().resolve()),
        "point_count": point_count,
        "grid_resolution_guess": grid_resolution_guess,
        "points_shape": list(points.shape),
        "bbox_min": points.min(axis=0).astype(float).tolist() if points.ndim == 2 and point_count else [],
        "bbox_max": points.max(axis=0).astype(float).tolist() if points.ndim == 2 and point_count else [],
        "selected_view_indices": selected.astype(int).tolist(),
        "visible_count_stats": stat_array(payload["visible_count"]),
        "mask_count_stats": stat_array(payload["mask_count"]),
        "token_count_stats": stat_array(payload["token_count"]),
        "occupancy_ratio_stats": stat_array(payload["occupancy_ratio"]),
        "rgb_variance_stats": stat_array(payload["rgb_variance"]),
        "rgb_range_stats": stat_array(payload["rgb_range"]),
        "token_cosine_stats": stat_array(payload["token_cosine"]),
        "evidence_score_stats": stat_array(payload["evidence_score"]),
        "boundary_like_ratio": float(np.asarray(payload["boundary_like"], dtype=bool).mean()),
        "errors": errors,
        "compatible": not errors,
    }


def template_summary(path: Path) -> dict[str, Any]:
    required = ("hybrid_vertices", "hybrid_faces", "normals", "canonical_positions", "part_ids")
    payload = load_npz(path, required)
    vertices = np.asarray(payload["hybrid_vertices"])
    faces = np.asarray(payload["hybrid_faces"])
    normals = np.asarray(payload["normals"])
    canonical_positions = np.asarray(payload["canonical_positions"])
    part_ids = np.asarray(payload["part_ids"])
    errors: list[str] = []
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        errors.append(f"hybrid_vertices expected [N,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        errors.append(f"hybrid_faces expected [F,3], got {faces.shape}")
    if part_ids.ndim != 1 or part_ids.shape[0] != vertices.shape[0]:
        errors.append(f"part_ids expected [{vertices.shape[0]}], got {part_ids.shape}")
    if not np.isfinite(vertices).all():
        errors.append("hybrid_vertices contain non-finite values")
    if not np.isfinite(canonical_positions).all():
        errors.append("canonical_positions contain non-finite values")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        errors.append("hybrid_faces contain out-of-range indices")

    mask_rows: dict[str, Any] = {}
    missing_masks: list[str] = []
    for mask_name in EXPECTED_TEMPLATE_MASKS:
        if mask_name not in payload:
            missing_masks.append(mask_name)
            mask_rows[mask_name] = {"present": False, "count": 0}
            continue
        mask = np.asarray(payload[mask_name], dtype=bool).reshape(-1)
        if mask.shape[0] != vertices.shape[0]:
            errors.append(f"{mask_name} length {mask.shape[0]} != vertex count {vertices.shape[0]}")
        mask_rows[mask_name] = {"present": True, "count": int(mask.sum())}
    if missing_masks:
        errors.append(f"missing required template masks: {missing_masks}")

    unique_part_ids, counts = np.unique(part_ids.astype(np.int64), return_counts=True)
    return {
        "path": str(path.expanduser().resolve()),
        "hybrid_vertex_count": int(vertices.shape[0]) if vertices.ndim == 2 else 0,
        "hybrid_face_count": int(faces.shape[0]) if faces.ndim == 2 else 0,
        "normals_shape": list(normals.shape),
        "canonical_positions_shape": list(canonical_positions.shape),
        "bbox_min": vertices.min(axis=0).astype(float).tolist() if vertices.ndim == 2 and vertices.shape[0] else [],
        "bbox_max": vertices.max(axis=0).astype(float).tolist() if vertices.ndim == 2 and vertices.shape[0] else [],
        "part_counts": {str(int(pid)): int(count) for pid, count in zip(unique_part_ids, counts)},
        "required_mask_counts": mask_rows,
        "errors": errors,
        "compatible": not errors,
    }


def representation_contract(
    scene: dict[str, Any],
    token: dict[str, Any],
    query: dict[str, Any],
    latent: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D0_v2_representation_contract_v1",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "backend_family": "canonical_latent_grid_or_local_surface_token_grid_sdf",
        "explicitly_not": [
            "B19 bounded query-to-carrier renderer",
            "vertex-offset-only carrier patch",
            "template-only surface cleanup",
            "VGGT depth/point/normal hard-teacher export",
        ],
        "allowed_inputs": {
            "vggt_latent_tokens": {
                "source": token["path"],
                "shape": "[1, V, T, C]",
                "observed_shape": token["shape"],
                "selected_view_indices": token["selected_view_indices"],
                "patch_grid": token["patch_grid"],
            },
            "raw_rgb_masks_cameras": {
                "scene_dir": scene["path"],
                "image_count": scene["image_count"],
                "mask_count": scene["mask_count"],
                "camera_count": scene["camera_sidecar"].get("camera_count"),
            },
            "canonical_latent_grid_seed": {
                "source": latent["path"],
                "points": "[G, 3]",
                "observed_point_count": latent["point_count"],
                "grid_resolution_guess": latent["grid_resolution_guess"],
                "bbox_min": latent["bbox_min"],
                "bbox_max": latent["bbox_max"],
            },
            "local_surface_token_grid_seed": {
                "source": template["path"],
                "vertices": "[S, 3]",
                "faces": "[F, 3]",
                "observed_vertex_count": template["hybrid_vertex_count"],
                "observed_face_count": template["hybrid_face_count"],
                "required_local_families": list(EXPECTED_QUERY_FAMILIES),
            },
            "query_evidence": {
                "source": query["path"],
                "query_positions": "[N, 3]",
                "observed_query_count": query["query_count"],
                "families": query["family_rows"],
            },
        },
        "canonical_grid_contract": {
            "grid_features": "[B, Gx, Gy, Gz, Cg]",
            "grid_xyz": "[B, Gx, Gy, Gz, 3]",
            "grid_mask_or_validity": "[B, Gx, Gy, Gz, 1]",
            "grid_resolution_policy": "fixed small smoke first, then sparse/hash grid only after render controls pass",
        },
        "surface_token_grid_contract": {
            "surface_tokens": "[B, S, Cs]",
            "surface_xyz": "[B, S, 3]",
            "surface_normals": "[B, S, 3]",
            "surface_family_ids": "[B, S]",
            "local_neighbors": "[B, S, K]",
            "surface_token_policy": "local surface-token grid may seed geometry but cannot export a template-only carrier",
        },
    }


def cross_attention_contract(token: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D0_v2_2d_to_3d_cross_attention_contract_v1",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "input_shapes": {
            "vggt_tokens": "[B, V, T, C]",
            "observed_vggt_tokens": token["shape"],
            "query_xyz": "[B, N, 3]",
            "observed_query_count": query["query_count"],
            "projected_uv": "[B, N, V, 2]",
            "visibility_prior": "[B, N, V]",
            "token_indices": "[B, N, V]",
            "camera_params": "[B, V, intrinsics/extrinsics]",
        },
        "required_operations": [
            "project each canonical-grid or surface-token query into every selected view",
            "gather local VGGT patch-token neighborhoods around projected uv",
            "cross-attend 3D queries to per-view local 2D token neighborhoods",
            "condition attention with mask visibility, depth ordering, and query family",
            "produce fused 3D latent features without copying B19 carrier labels",
        ],
        "required_outputs": {
            "fused_query_features": "[B, N, Cf]",
            "attention_weights": "[B, N, V, K]",
            "visibility_logits": "[B, N, V]",
            "support_count": "[B, N, 1]",
            "cross_view_consistency": "[B, N, 1]",
        },
        "negative_controls_required_before_smoke": [
            "real_tokens",
            "patch_shuffle_tokens",
            "zero_tokens",
        ],
    }


def refinement_contract() -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D0_v2_3d_refinement_contract_v1",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "refiner_inputs": {
            "canonical_grid_or_surface_tokens": "[B, M, C]",
            "query_xyz": "[B, N, 3]",
            "fused_query_features": "[B, N, Cf]",
            "family_ids": "[B, N]",
            "local_neighbors": "[B, N, K]",
        },
        "required_modules_for_next_artifact": [
            "Sparse or dense 3D latent-grid encoder/refiner",
            "Surface-token local-neighborhood refiner",
            "2D-to-3D cross-attention block",
            "SDF/occupancy/normal residual heads",
            "Visibility/confidence heads",
        ],
        "refinement_outputs": {
            "refined_3d_features": "[B, N, Cr]",
            "neighbor_residuals": "[B, N, K, 3] optional",
            "latent_grid_delta": "[B, Gx, Gy, Gz, Cg] optional",
            "surface_token_delta": "[B, S, Cs] optional",
        },
        "fail_closed_rules": [
            "no export if refiner is absent",
            "no export if real tokens fail to beat shuffle and zero",
            "no export if visibility/confidence merely masks unsupported regions",
            "no export if output is a carrier/template-only deformation",
        ],
    }


def output_schema_contract() -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D0_v2_sdf_output_schema_v1",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "required_named_outputs": {
            "sdf": "[B, N, 1] signed distance or signed implicit field",
            "occupancy": "[B, N, 1] probability/logit",
            "normal_residual": "[B, N, 3] normalized or explicitly scaled residual",
            "visibility": "[B, N, V] per-view visibility probability/logit",
            "confidence": "[B, N, 1] geometry confidence separate from visibility",
            "family_logits": "[B, N, F]",
            "support_count": "[B, N, 1]",
        },
        "optional_render_outputs_after_backend_exists": {
            "rendered_mask": "[B, V, H, W]",
            "rendered_depth": "[B, V, H, W]",
            "rendered_normal": "[B, V, H, W, 3]",
            "rendered_rgb": "[B, V, H, W, 3]",
        },
        "forbidden_outputs_from_this_preflight": [
            "predictions.npz",
            "candidate package",
            "teacher package",
            "strict registry entry",
            "checkpoint",
            "cloud job spec",
        ],
    }


def blocked_next_artifact_summary(backend_smoke_feasible: bool) -> dict[str, Any]:
    if backend_smoke_feasible:
        status = "unexpected_ready"
        artifact = "Run a separate research smoke implementing B_Fus3D0_v2_latent_grid_sdf_backend_smoke.py."
    else:
        status = "blocked_backend_not_implemented"
        artifact = (
            "tools/b_fus3d0_v2_latent_grid_sdf_backend_smoke.py implementing: "
            "canonical latent-grid or local surface-token-grid construction, "
            "2D-to-3D cross-attention, 3D refinement, and SDF/occupancy/"
            "normal_residual/visibility/confidence heads with real/shuffle/zero controls."
        )
    return {
        "status": status,
        "full_smoke_feasible": bool(backend_smoke_feasible),
        "blocked_condition": (
            "Input contract can be validated locally, but no B-Fus3D0-v2 backend module exists "
            "under the allowed new-file scope. Running a full smoke would require implementing "
            "the next artifact listed below, not tuning B19."
        ),
        "exact_next_implementation_artifact": artifact,
        "teacher_candidate_export": "blocked",
        "cloud": "blocked",
        "predictions": "blocked",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    compatibility = summary["compatibility"]
    rows = summary["query_evidence"]["family_rows"]
    paths = summary["written_paths"]
    lines = [
        "# B-Fus3D0-v2 Representation Contract Preflight",
        "",
        "Status: `research_only_fail_closed_contract_preflight_no_export`",
        "",
        "This is a contract/preflight skeleton for a different B-Fus3D0-v2 latent-grid or",
        "surface-token-grid SDF backend. It does not tune B19 and does not write predictions,",
        "teachers, candidates, checkpoints, registry state, strict pass state, or cloud jobs.",
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
        json.dumps(json_ready(compatibility), indent=2, sort_keys=True),
        "```",
        "",
        "## Representation Path",
        "",
        "```text",
        "VGGT latent/raw RGB/masks/cameras",
        "  -> canonical 3D latent grid or local surface-token grid",
        "  -> 2D-to-3D cross-attention contract",
        "  -> 3D refinement contract",
        "  -> SDF / occupancy / normal residual / visibility / confidence schema",
        "```",
        "",
        "## Observed Local Inputs",
        "",
        f"- Scene images/masks: {summary['scene']['image_count']} / {summary['scene']['mask_count']}",
        f"- Cameras: {summary['scene']['camera_sidecar'].get('camera_count')}",
        f"- VGGT tokens: shape={summary['token_cache']['shape']}, patch_grid={summary['token_cache']['patch_grid']}",
        f"- Query evidence: {summary['query_evidence']['query_count']} queries",
        f"- Latent grid seed: {summary['latent_grid']['point_count']} points",
        (
            f"- Surface-token template seed: {summary['template_payload']['hybrid_vertex_count']} vertices, "
            f"{summary['template_payload']['hybrid_face_count']} faces"
        ),
        "",
        "## Query Families",
        "",
    ]
    for family in sorted(rows):
        row = rows[family]
        support_mean = row["support_stats"].get("mean")
        support_text = "n/a" if support_mean is None else f"{support_mean:.3f}"
        lines.append(
            f"- `{family}`: count={row['count']}, mean_support={support_text}, "
            f"two_view_or_more_ratio={row['two_view_or_more_ratio']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Full Smoke Status",
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
            "Contract/preflight is complete and fail-closed. Full B-Fus3D0-v2 smoke remains blocked until the named backend artifact exists.",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fail_closed_path_guard(args.output_dir, args.status_report)
    paths = output_manifest(args.output_dir, args.status_report)
    ensure_no_existing_outputs(paths, args.overwrite)

    scene = scene_summary(args.scene_dir)
    token = token_cache_summary(args.token_cache)
    query = query_evidence_summary(args.query_evidence, token)
    latent = latent_grid_summary(args.latent_grid, token)
    template = template_summary(args.template_payload)
    compatibility_errors = (
        scene["errors"]
        + token["errors"]
        + query["errors"]
        + latent["errors"]
        + template["errors"]
    )
    compatibility = {
        "input_contract_compatible": not compatibility_errors,
        "backend_smoke_feasible": False,
        "scene_compatible": scene["compatible"],
        "token_cache_compatible": token["compatible"],
        "query_evidence_compatible": query["compatible"],
        "latent_grid_seed_compatible": latent["compatible"],
        "surface_token_template_seed_compatible": template["compatible"],
        "selected_view_indices_match": (
            token["selected_view_indices"]
            == query["selected_view_indices"]
            == latent["selected_view_indices"]
        ),
        "errors": compatibility_errors,
    }

    rep_contract = representation_contract(scene, token, query, latent, template)
    xattn_contract = cross_attention_contract(token, query)
    refine_contract = refinement_contract()
    schema_contract = output_schema_contract()
    blocked = blocked_next_artifact_summary(backend_smoke_feasible=False)

    summary = {
        "status": "research_only_fail_closed_contract_preflight_no_export",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "scene": scene,
        "token_cache": token,
        "query_evidence": query,
        "latent_grid": latent,
        "template_payload": template,
        "compatibility": compatibility,
        "representation_contract": rep_contract,
        "cross_attention_contract": xattn_contract,
        "refinement_contract": refine_contract,
        "output_schema": schema_contract,
        "blocked_next_artifact": blocked,
        "written_paths": {key: str(path.resolve()) for key, path in paths.items()},
    }

    write_json(paths["representation_contract"], rep_contract)
    write_json(paths["cross_attention_contract"], xattn_contract)
    write_json(paths["refinement_contract"], refine_contract)
    write_json(paths["output_schema"], schema_contract)
    write_json(paths["blocked_next_artifact"], blocked)
    write_json(paths["summary"], summary)
    report = build_markdown(summary)
    write_text(paths["report"], report)
    write_text(paths["status_report"], report)
    print(json.dumps(json_ready({"status": summary["status"], "written_paths": summary["written_paths"]}), indent=2))


if __name__ == "__main__":
    main()
