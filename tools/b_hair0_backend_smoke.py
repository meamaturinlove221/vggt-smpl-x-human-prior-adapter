from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_HAIR0_ARRAYS = Path(
    "output/surface_research_preflight_local/B_hair0_contract_preflight_hybrid6_layer23/"
    "b_hair0_diagnostic_arrays.npz"
)
DEFAULT_QUERY_EVIDENCE = Path(
    "output/surface_research_preflight_local/B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_LATENT_REAL = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_SHUFFLE = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_ZERO = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/B_hair1_backend_smoke_v6")
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_hair1_backend_status.md")
DEFAULT_STATUS_JSON = Path("reports/20260507_b_hair1_backend_status.json")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "backend_smoke_only": True,
    "v6_backend_implementation": True,
    "no_cloud": True,
    "no_train": True,
    "no_predictions_write": True,
    "no_checkpoint_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "not_teacher": True,
    "not_candidate": True,
}
FORBIDDEN_PATH_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions",
    "checkpoint",
)
CONTROL_NAMES = ("real", "shuffle", "zero", "mask_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-hair1 research-only v6 backend smoke. Builds rooted hairline "
            "strand/Gaussian-chain primitives, compares real/shuffle/zero/mask-only "
            "controls, and writes diagnostic PLY/NPZ/contact sheets only."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--hair0-arrays", type=Path, default=DEFAULT_HAIR0_ARRAYS)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--latent-real", type=Path, default=DEFAULT_LATENT_REAL)
    parser.add_argument("--latent-shuffle", type=Path, default=DEFAULT_LATENT_SHUFFLE)
    parser.add_argument("--latent-zero", type=Path, default=DEFAULT_LATENT_ZERO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=160)
    parser.add_argument("--max-roots", type=int, default=320)
    parser.add_argument("--chain-steps", type=int, default=7)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    bad = [token for token in FORBIDDEN_PATH_TOKENS if token in text]
    if bad:
        raise ValueError(f"Refusing output path with forbidden tokens {bad}: {path}")


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


def stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "mean": float(arr.mean()),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def point_stats(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return {"count": 0}
    return {
        "count": int(pts.shape[0]),
        "bbox_min": pts.min(axis=0).tolist(),
        "bbox_max": pts.max(axis=0).tolist(),
        "extent": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
    }


def load_template(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [
            key
            for key in (
                "hybrid_vertices",
                "hybrid_faces",
                "head_vertex_mask",
                "hairline_vertex_mask",
                "face_front_vertex_mask",
            )
            if key not in payload.files
        ]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
        faces = np.asarray(payload["hybrid_faces"], dtype=np.int32)
        masks = {
            "head": np.asarray(payload["head_vertex_mask"], dtype=bool),
            "hairline": np.asarray(payload["hairline_vertex_mask"], dtype=bool),
            "face_front": np.asarray(payload["face_front_vertex_mask"], dtype=bool),
        }
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    return {"vertices": vertices, "faces": faces, "normals": normals, "masks": masks, "path": str(resolved)}


def load_hair0(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        required = (
            "hairline_points",
            "hairline_support",
            "hairline_mask_indices",
            "scalp_points",
            "head_top_points",
            "hair_ring_points",
            "selected_view_indices",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        out = {key: np.asarray(payload[key]) for key in payload.files}
    out["path"] = np.asarray(str(resolved))
    return out


def load_query(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        required = (
            "query_positions",
            "query_families",
            "support",
            "mean_features",
            "variance_features",
            "selected_view_indices",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        out = {key: np.asarray(payload[key]) for key in payload.files}
    out["path"] = np.asarray(str(resolved))
    return out


def load_latent(path: Path, name: str) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        required = (
            "points",
            "visible_count",
            "mask_count",
            "token_count",
            "occupancy_ratio",
            "token_cosine",
            "evidence_score",
            "selected_view_indices",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        out = {key: np.asarray(payload[key]) for key in payload.files}
    out["control_name"] = np.asarray(name)
    out["path"] = np.asarray(str(resolved))
    return out


def load_rgb_mask(view: dict[str, Any], target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(Path(str(view["image_path"])).expanduser().resolve()).convert("RGB")
    mask = Image.open(Path(str(view["mask_path"])).expanduser().resolve()).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8) > 127


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    image_size = view.get("image_size") or [target_size, target_size]
    native_size = int(image_size[0]) if len(image_size) >= 1 else int(target_size)
    meta = view.get("preprocess_meta") or {}
    if meta.get("transform") == "crop_pad_to_square" and native_size != int(target_size):
        native = align_intrinsics_for_scene_view(intrinsic, view, target_size=native_size)
        scale = float(target_size) / float(max(1, native_size))
        out = native.astype(np.float32).copy()
        out[0, :] *= scale
        out[1, :] *= scale
        return out
    return align_intrinsics_for_scene_view(intrinsic, view, target_size=target_size)


def load_views(
    scene_dir: Path,
    selected_view_indices: np.ndarray,
    dataset_root: Path | None,
    subset_name: str,
    target_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], str]:
    scene_dir = scene_dir.expanduser().resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    exported = manifest["exported_views"]
    view_indices = [int(v) for v in np.asarray(selected_view_indices).reshape(-1).tolist()]
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    rows: list[dict[str, Any]] = []
    for view_index in view_indices:
        if view_index < 0 or view_index >= len(exported):
            raise IndexError(f"selected view {view_index} outside scene view count {len(exported)}")
        view = dict(exported[view_index])
        view["view_index"] = int(view_index)
        rgb, mask = load_rgb_mask(view, target_size)
        view["rgb"] = rgb
        view["mask"] = mask
        rows.append(view)
    return rows, cameras, camera_source


def nearest_indices(query: np.ndarray, reference: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    all_idx: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []
    for start in range(0, query.shape[0], int(chunk_size)):
        chunk = query[start : start + int(chunk_size)]
        dist2 = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
        idx = np.argmin(dist2, axis=1).astype(np.int64)
        all_idx.append(idx)
        all_dist.append(np.sqrt(dist2[np.arange(chunk.shape[0]), idx]).astype(np.float32))
    return np.concatenate(all_idx), np.concatenate(all_dist)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    if not np.any(finite):
        return out
    lo, hi = np.quantile(arr[finite], [0.05, 0.95])
    if hi <= lo:
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
    if hi <= lo:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - float(lo)) / float(hi - lo), 0.0, 1.0)
    return out


def color_by_scalar(values: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    t = robust_normalize(values)[:, None]
    lo = np.asarray(low, dtype=np.float32)[None, :]
    hi = np.asarray(high, dtype=np.float32)[None, :]
    return np.clip(lo * (1.0 - t) + hi * t, 0, 255).astype(np.uint8)


def select_roots(
    hair0: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    *,
    max_roots: int,
    min_support: int,
    seed: int,
) -> dict[str, np.ndarray]:
    root_points = np.asarray(hair0["hairline_points"], dtype=np.float32)
    support = np.asarray(hair0["hairline_support"], dtype=np.float32).reshape(-1)
    root_vertex_ids = np.asarray(hair0["hairline_mask_indices"], dtype=np.int64).reshape(-1)
    normals = np.asarray(template["normals"], dtype=np.float32)[root_vertex_ids]
    keep = np.flatnonzero(support >= float(min_support))
    if keep.size < max(32, min(max_roots, 64)):
        keep = np.flatnonzero(support >= 1.0)
    if keep.size == 0:
        keep = np.arange(root_points.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    scores = support[keep] + 0.01 * rng.random(keep.size)
    order = keep[np.argsort(scores)[::-1]]
    if order.size > int(max_roots):
        stride_pick = np.linspace(0, order.size - 1, int(max_roots)).round().astype(np.int64)
        order = order[stride_pick]
    return {
        "root_points": root_points[order].astype(np.float32),
        "root_support": support[order].astype(np.float32),
        "root_vertex_ids": root_vertex_ids[order].astype(np.int64),
        "root_normals": normals[order].astype(np.float32),
        "source_indices": order.astype(np.int64),
    }


def query_hairline_readout(query: dict[str, np.ndarray], roots: np.ndarray) -> dict[str, np.ndarray]:
    families = np.asarray(query["query_families"]).astype(str)
    hair_mask = families == "hairline"
    if not np.any(hair_mask):
        return {
            "query_support": np.zeros((roots.shape[0],), dtype=np.float32),
            "feature_energy": np.zeros((roots.shape[0],), dtype=np.float32),
            "feature_uncertainty": np.ones((roots.shape[0],), dtype=np.float32),
            "nearest_distance": np.full((roots.shape[0],), np.inf, dtype=np.float32),
        }
    qpos = np.asarray(query["query_positions"], dtype=np.float32)[hair_mask]
    qsupport = np.asarray(query["support"], dtype=np.float32)[hair_mask]
    mean_features = np.asarray(query["mean_features"], dtype=np.float32)[hair_mask]
    variance_features = np.asarray(query["variance_features"], dtype=np.float32)[hair_mask]
    idx, dist = nearest_indices(roots, qpos)
    energy = np.linalg.norm(mean_features[idx], axis=1) / math.sqrt(float(mean_features.shape[1]))
    uncertainty = np.mean(variance_features[idx], axis=1)
    return {
        "query_support": qsupport[idx].astype(np.float32),
        "feature_energy": energy.astype(np.float32),
        "feature_uncertainty": uncertainty.astype(np.float32),
        "nearest_distance": dist.astype(np.float32),
    }


def latent_root_readout(roots: np.ndarray, latent: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    points = np.asarray(latent["points"], dtype=np.float32)
    idx, dist = nearest_indices(roots, points)
    out: dict[str, np.ndarray] = {"nearest_distance": dist.astype(np.float32)}
    for key in ("visible_count", "mask_count", "token_count", "occupancy_ratio", "token_cosine", "evidence_score"):
        out[key] = np.asarray(latent[key])[idx].astype(np.float32)
    return out


def make_control_root_scores(
    roots: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    latents: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    root_points = roots["root_points"]
    support_norm = np.clip(roots["root_support"] / 6.0, 0.0, 1.0).astype(np.float32)
    query_readout = query_hairline_readout(query, root_points)
    query_norm = np.clip(query_readout["query_support"] / 6.0, 0.0, 1.0).astype(np.float32)
    feature_energy_norm = robust_normalize(query_readout["feature_energy"])
    feature_uncert_norm = 1.0 - robust_normalize(query_readout["feature_uncertainty"])
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in ("real", "shuffle", "zero"):
        readout = latent_root_readout(root_points, latents[name])
        evidence = robust_normalize(readout["evidence_score"])
        token = robust_normalize(np.nan_to_num(readout["token_cosine"], nan=0.0))
        occupancy = np.clip(readout["occupancy_ratio"], 0.0, 1.0)
        mask = np.clip(readout["mask_count"] / 6.0, 0.0, 1.0)
        score = (
            0.34 * support_norm
            + 0.20 * query_norm
            + 0.18 * evidence
            + 0.11 * token
            + 0.10 * occupancy
            + 0.05 * feature_energy_norm
            + 0.02 * feature_uncert_norm
        ).astype(np.float32)
        out[name] = {
            "root_score": np.clip(score, 0.0, 1.0).astype(np.float32),
            "support_norm": support_norm,
            "query_support_norm": query_norm,
            "latent_evidence_norm": evidence.astype(np.float32),
            "token_cosine_norm": token.astype(np.float32),
            "occupancy_ratio": occupancy.astype(np.float32),
            "mask_count_norm": mask.astype(np.float32),
            "latent_nearest_distance": readout["nearest_distance"],
            "query_nearest_distance": query_readout["nearest_distance"],
        }
    mask_only_score = (0.60 * support_norm + 0.25 * query_norm + 0.15 * out["real"]["mask_count_norm"]).astype(np.float32)
    out["mask_only"] = {
        "root_score": np.clip(mask_only_score, 0.0, 1.0).astype(np.float32),
        "support_norm": support_norm,
        "query_support_norm": query_norm,
        "latent_evidence_norm": np.zeros_like(support_norm),
        "token_cosine_norm": np.zeros_like(support_norm),
        "occupancy_ratio": out["real"]["occupancy_ratio"],
        "mask_count_norm": out["real"]["mask_count_norm"],
        "latent_nearest_distance": out["real"]["latent_nearest_distance"],
        "query_nearest_distance": query_readout["nearest_distance"],
    }
    return out


def tangent_basis(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(norm, 1e-8, None)
    center = pts.mean(axis=0, keepdims=True)
    radial = pts - center
    radial = radial - np.sum(radial * n, axis=1, keepdims=True) * n
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    fallback = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (pts.shape[0], 1))
    radial = np.where(radial_norm > 1e-6, radial / np.clip(radial_norm, 1e-8, None), fallback)
    side = np.cross(n, radial)
    side_norm = np.linalg.norm(side, axis=1, keepdims=True)
    side = side / np.clip(side_norm, 1e-8, None)
    return radial.astype(np.float32), side.astype(np.float32)


def build_strand_chain(
    roots: dict[str, np.ndarray],
    score_row: dict[str, np.ndarray],
    *,
    control: str,
    chain_steps: int,
) -> dict[str, np.ndarray]:
    root_points = np.asarray(roots["root_points"], dtype=np.float32)
    normals = np.asarray(roots["root_normals"], dtype=np.float32)
    root_support = np.asarray(roots["root_support"], dtype=np.float32)
    score = np.asarray(score_row["root_score"], dtype=np.float32)
    radial, side = tangent_basis(root_points, normals)
    vertical = np.tile(np.array([[0.0, -1.0, 0.0]], dtype=np.float32), (root_points.shape[0], 1))
    bend_phase = np.sin(np.arange(root_points.shape[0], dtype=np.float32) * 0.6180339)[:, None]
    direction = (
        0.48 * normals
        + 0.34 * vertical
        + 0.16 * radial
        + 0.05 * bend_phase * side
    )
    direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)
    if control == "shuffle":
        direction = 0.82 * direction + 0.18 * np.roll(direction, shift=max(1, direction.shape[0] // 7), axis=0)
        direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)
    elif control == "zero":
        direction = 0.72 * normals + 0.28 * vertical
        direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)
    elif control == "mask_only":
        direction = 0.58 * normals + 0.42 * vertical
        direction = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8, None)

    score_spread = 0.006 + 0.024 * score
    support_scale = 0.85 + 0.20 * np.clip(root_support / 6.0, 0.0, 1.0)
    length = np.clip(score_spread * float(chain_steps - 1) * support_scale, 0.010, 0.120).astype(np.float32)
    t = np.linspace(0.0, 1.0, int(chain_steps), dtype=np.float32)[None, :, None]
    bend = (np.sin(t * math.pi) * (0.003 + 0.009 * score[:, None, None]) * side[:, None, :]).astype(np.float32)
    sag = (t * t * 0.010 * vertical[:, None, :] * (0.4 + score[:, None, None])).astype(np.float32)
    points = root_points[:, None, :] + direction[:, None, :] * length[:, None, None] * t + bend + sag
    n_roots = root_points.shape[0]
    chain_ids = np.repeat(np.arange(n_roots, dtype=np.int32), int(chain_steps))
    step_ids = np.tile(np.arange(int(chain_steps), dtype=np.int32), n_roots)
    point_scores = np.repeat(score, int(chain_steps)).astype(np.float32)
    root_support_rep = np.repeat(root_support, int(chain_steps)).astype(np.float32)
    opacity_root = np.clip(0.22 + 0.68 * score, 0.05, 0.95).astype(np.float32)
    scale_root = np.clip(0.0025 + 0.0060 * score, 0.0020, 0.0100).astype(np.float32)
    fade = (1.0 - 0.45 * (step_ids.astype(np.float32) / max(1, int(chain_steps) - 1))).astype(np.float32)
    opacity = np.repeat(opacity_root, int(chain_steps)) * fade
    scale = np.repeat(scale_root, int(chain_steps)) * (0.90 + 0.25 * fade)
    colors = color_by_scalar(point_scores, (55, 35, 85), (235, 175, 80))
    colors[step_ids == 0] = np.array([245, 80, 150], dtype=np.uint8)
    return {
        "points": points.reshape(-1, 3).astype(np.float32),
        "chain_ids": chain_ids,
        "step_ids": step_ids,
        "root_points": root_points.astype(np.float32),
        "root_score": score.astype(np.float32),
        "root_support": root_support.astype(np.float32),
        "root_vertex_ids": np.asarray(roots["root_vertex_ids"], dtype=np.int64),
        "root_normals": normals.astype(np.float32),
        "colors": colors.astype(np.uint8),
        "opacity": opacity.astype(np.float32),
        "scale": scale.astype(np.float32),
        "point_score": point_scores,
        "point_support": root_support_rep,
    }


def chain_edges(chain_count: int, chain_steps: int) -> np.ndarray:
    edges: list[tuple[int, int]] = []
    for chain_id in range(int(chain_count)):
        base = chain_id * int(chain_steps)
        for step in range(int(chain_steps) - 1):
            edges.append((base + step, base + step + 1))
    return np.asarray(edges, dtype=np.int32)


def write_chain_ply(path: Path, chain: dict[str, np.ndarray], chain_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(chain["points"], dtype=np.float32)
    colors = np.asarray(chain["colors"], dtype=np.uint8)
    opacity = np.asarray(chain["opacity"], dtype=np.float32)
    scale = np.asarray(chain["scale"], dtype=np.float32)
    chain_ids = np.asarray(chain["chain_ids"], dtype=np.int32)
    step_ids = np.asarray(chain["step_ids"], dtype=np.int32)
    scores = np.asarray(chain["point_score"], dtype=np.float32)
    support = np.asarray(chain["point_support"], dtype=np.float32)
    edges = chain_edges(int(chain["root_points"].shape[0]), int(chain_steps))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float opacity\n")
        handle.write("property float scale\n")
        handle.write("property int chain_id\n")
        handle.write("property int step_id\n")
        handle.write("property float score\n")
        handle.write("property float support\n")
        handle.write(f"element edge {edges.shape[0]}\n")
        handle.write("property int vertex1\n")
        handle.write("property int vertex2\n")
        handle.write("end_header\n")
        for idx, point in enumerate(points):
            color = colors[idx]
            handle.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(opacity[idx]):.6f} {float(scale[idx]):.6f} "
                f"{int(chain_ids[idx])} {int(step_ids[idx])} {float(scores[idx]):.6f} {float(support[idx]):.6f}\n"
            )
        for edge in edges:
            handle.write(f"{int(edge[0])} {int(edge[1])}\n")


def write_support_ply(path: Path, template: dict[str, np.ndarray], hair0: dict[str, np.ndarray], real_chain: dict[str, np.ndarray]) -> None:
    head = np.asarray(template["vertices"], dtype=np.float32)[np.asarray(template["masks"]["head"], dtype=bool)]
    head_top = np.asarray(hair0["head_top_points"], dtype=np.float32)
    scalp = np.asarray(hair0["scalp_points"], dtype=np.float32)
    roots = np.asarray(real_chain["root_points"], dtype=np.float32)
    points = np.concatenate([head[::6], scalp, head_top, roots], axis=0).astype(np.float32)
    colors = np.concatenate(
        [
            np.tile(np.array([[95, 120, 180]], dtype=np.uint8), (head[::6].shape[0], 1)),
            np.tile(np.array([[80, 190, 150]], dtype=np.uint8), (scalp.shape[0], 1)),
            np.tile(np.array([[245, 210, 85]], dtype=np.uint8), (head_top.shape[0], 1)),
            np.tile(np.array([[245, 70, 150]], dtype=np.uint8), (roots.shape[0], 1)),
        ],
        axis=0,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def project_points(points: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32)


def draw_points(height: int, width: int, uv: np.ndarray, depth: np.ndarray, colors: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros((height, width), dtype=bool)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
    valid_idx = np.flatnonzero(valid)
    order = valid_idx[np.argsort(depth[valid_idx])[::-1]]
    radius = max(0, int(radius))
    for idx in order:
        x = int(round(float(uv[idx, 0])))
        y = int(round(float(uv[idx, 1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                if (xx - x) * (xx - x) + (yy - y) * (yy - y) > radius * radius:
                    continue
                if depth[idx] <= zbuf[yy, xx]:
                    zbuf[yy, xx] = depth[idx]
                    mask[yy, xx] = True
                    rgb[yy, xx] = colors[idx].astype(np.float32) / 255.0
    return mask, rgb, zbuf


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    inter = int(np.count_nonzero(pred & target))
    union = int(np.count_nonzero(pred | target))
    pred_count = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(target))
    overfill = int(np.count_nonzero(pred & ~target))
    miss = int(np.count_nonzero((~pred) & target))
    return {
        "intersection": inter,
        "union": union,
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "miss_pixels": miss,
        "overfill_pixels": overfill,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / target_count) if target_count else 0.0,
        "overfill_ratio": float(overfill / max(pred_count, 1)),
    }


def rgb_residual(pred_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 1.0
    target = np.asarray(target_rgb, dtype=np.float32) / 255.0
    diff = np.sqrt(np.sum((np.asarray(pred_rgb, dtype=np.float32)[valid] - target[valid]) ** 2, axis=1) + 1e-6)
    return float(diff.mean())


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.asarray(rgb) * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def crop_bbox(mask: np.ndarray, pad: int, size: int) -> tuple[int, int, int, int]:
    yy, xx = np.where(np.asarray(mask, dtype=bool))
    if yy.size == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0 = max(0, int(xx.min()) - int(pad))
    y0 = max(0, int(yy.min()) - int(pad))
    x1 = min(mask.shape[1], int(xx.max()) + int(pad) + 1)
    y1 = min(mask.shape[0], int(yy.max()) + int(pad) + 1)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = max((x1 - x0), (y1 - y0), 1) // 2
    half = max(half, int(size) // 3)
    x0 = max(0, cx - half)
    x1 = min(mask.shape[1], cx + half)
    y0 = max(0, cy - half)
    y1 = min(mask.shape[0], cy + half)
    return (x0, y0, x1, y1)


def render_control(
    name: str,
    chain: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    *,
    target_size: int,
    point_radius: int,
    output_dir: Path,
) -> dict[str, Any]:
    control_dir = output_dir / "renders" / name
    control_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(chain["points"], dtype=np.float32)
    colors = np.asarray(chain["colors"], dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for view in views:
        camera_id = normalize_camera_id(view["camera_id"])
        camera = cameras[camera_id]
        intrinsic = align_intrinsics_for_loaded_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size)
        uv, depth = project_points(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        pred_mask, pred_rgb, zbuf = draw_points(target_size, target_size, uv, depth, colors, point_radius)
        target_mask = np.asarray(view["mask"], dtype=bool)
        target_rgb = np.asarray(view["rgb"], dtype=np.uint8)
        metrics = mask_metrics(pred_mask, target_mask)
        metrics["rgb_residual"] = rgb_residual(pred_rgb, target_rgb, pred_mask & target_mask)
        metrics["view_index"] = int(view["view_index"])
        metrics["camera_id"] = str(view["camera_id"])
        valid_depth = zbuf[np.isfinite(zbuf)]
        metrics["depth_pixels"] = int(valid_depth.size)
        metrics["depth_min"] = float(valid_depth.min()) if valid_depth.size else None
        metrics["depth_max"] = float(valid_depth.max()) if valid_depth.size else None
        rows.append(metrics)
        save_mask(control_dir / f"view_{int(view['view_index']):02d}_strand_mask.png", pred_mask)
        save_rgb(control_dir / f"view_{int(view['view_index']):02d}_strand_rgb.png", pred_rgb)
        delta = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
        delta[..., 0] = np.logical_and(pred_mask, ~target_mask).astype(np.float32)
        delta[..., 1] = np.logical_and(pred_mask, target_mask).astype(np.float32)
        delta[..., 2] = np.logical_and(~pred_mask, target_mask).astype(np.float32)
        save_rgb(control_dir / f"view_{int(view['view_index']):02d}_mask_delta_rgb.png", delta)
    return {
        "control": name,
        "views": rows,
        "mean_iou": float(np.mean([row["iou"] for row in rows])) if rows else 0.0,
        "mean_target_recall": float(np.mean([row["target_recall"] for row in rows])) if rows else 0.0,
        "mean_overfill_ratio": float(np.mean([row["overfill_ratio"] for row in rows])) if rows else 0.0,
        "mean_rgb_residual": float(np.mean([row["rgb_residual"] for row in rows])) if rows else 1.0,
        "mean_pred_pixels": float(np.mean([row["pred_pixels"] for row in rows])) if rows else 0.0,
    }


def make_contact_sheet(
    path: Path,
    views: list[dict[str, Any]],
    controls: dict[str, dict[str, np.ndarray]],
    cameras: dict[str, dict[str, np.ndarray]],
    *,
    target_size: int,
    point_radius: int,
) -> None:
    cols = ["target", *CONTROL_NAMES]
    rows = min(3, len(views))
    tile = 160
    header = 24
    sheet = Image.new("RGB", (len(cols) * tile, rows * (tile + header)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for row_idx, view in enumerate(views[:rows]):
        target = Image.fromarray(np.asarray(view["rgb"], dtype=np.uint8), mode="RGB")
        target_mask = np.asarray(view["mask"], dtype=bool)
        bbox = crop_bbox(target_mask, pad=18, size=target_size)
        target = target.crop(bbox).resize((tile, tile), Image.Resampling.BICUBIC)
        yoff = row_idx * (tile + header)
        sheet.paste(target, (0, yoff + header))
        draw.text((6, yoff + 5), f"target v{int(view['view_index']):02d}", fill=(235, 235, 235))
        for col_idx, name in enumerate(CONTROL_NAMES, start=1):
            camera_id = normalize_camera_id(view["camera_id"])
            camera = cameras[camera_id]
            intrinsic = align_intrinsics_for_loaded_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size)
            chain = controls[name]
            uv, depth = project_points(chain["points"], np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
            pred_mask, pred_rgb, _ = draw_points(target_size, target_size, uv, depth, chain["colors"], point_radius)
            overlay = np.asarray(view["rgb"], dtype=np.float32) / 255.0
            overlay = overlay * 0.58 + pred_rgb * 0.72
            delta = np.logical_and(pred_mask, ~target_mask)
            overlay[delta] = np.array([0.95, 0.18, 0.15], dtype=np.float32)
            img = Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8), mode="RGB")
            img = img.crop(bbox).resize((tile, tile), Image.Resampling.BICUBIC)
            xoff = col_idx * tile
            sheet.paste(img, (xoff, yoff + header))
            draw.text((xoff + 6, yoff + 5), name, fill=(235, 235, 235))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def mean_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def summarize_control(name: str, chain: dict[str, np.ndarray], render_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "control": name,
        "root_count": int(chain["root_points"].shape[0]),
        "chain_point_count": int(chain["points"].shape[0]),
        "root_score": stats(chain["root_score"]),
        "root_support": stats(chain["root_support"]),
        "chain_points": point_stats(chain["points"]),
        "mean_iou": render_metrics["mean_iou"],
        "mean_target_recall": render_metrics["mean_target_recall"],
        "mean_overfill_ratio": render_metrics["mean_overfill_ratio"],
        "mean_rgb_residual": render_metrics["mean_rgb_residual"],
        "mean_pred_pixels": render_metrics["mean_pred_pixels"],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    lines = [
        "# B-hair1 Backend Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only v6 backend implementation for B-hair1. This builds a rooted",
        "hairline strand/Gaussian-chain primitive from local hairline support and",
        "compares real/shuffle/zero/mask-only controls. It is not a teacher,",
        "candidate, strict pass, export, checkpoint, registry update, or cloud job.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['strict_facts']['formal_cloud_train_infer_export']}",
        f"teacher_export = {summary['strict_facts']['teacher_export']}",
        f"candidate_export = {summary['strict_facts']['candidate_export']}",
        "```",
        "",
        "## Primitive",
        "",
        "```json",
        json.dumps(json_ready(summary["primitive"]), indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Control Metrics",
        "",
        "| control | root_score_mean | mean_iou | mean_recall | overfill | rgb_residual | pred_px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CONTROL_NAMES:
        row = summary["controls"][name]
        lines.append(
            "| "
            f"{name} | "
            f"{row['root_score']['mean']:.6f} | "
            f"{row['mean_iou']:.6f} | "
            f"{row['mean_target_recall']:.6f} | "
            f"{row['mean_overfill_ratio']:.6f} | "
            f"{row['mean_rgb_residual']:.6f} | "
            f"{row['mean_pred_pixels']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Real vs Controls",
            "",
            "```json",
            json.dumps(json_ready(comparison), indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
            "## Outputs",
            "",
            "```text",
            *summary["outputs"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (args.output_dir, args.status_report, args.status_json):
        ensure_safe_path(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = load_template(args.template_payload)
    hair0 = load_hair0(args.hair0_arrays)
    query = load_query(args.query_evidence)
    latents = {
        "real": load_latent(args.latent_real, "real"),
        "shuffle": load_latent(args.latent_shuffle, "shuffle"),
        "zero": load_latent(args.latent_zero, "zero"),
    }
    selected_view_indices = np.asarray(hair0["selected_view_indices"], dtype=np.int32)
    query_views = np.asarray(query["selected_view_indices"], dtype=np.int32)
    if not np.array_equal(selected_view_indices, query_views):
        raise ValueError(f"hair0/query selected views differ: {selected_view_indices.tolist()} vs {query_views.tolist()}")
    for name, latent in latents.items():
        latent_views = np.asarray(latent["selected_view_indices"], dtype=np.int32)
        if not np.array_equal(selected_view_indices, latent_views):
            raise ValueError(f"hair0/{name} latent selected views differ: {selected_view_indices.tolist()} vs {latent_views.tolist()}")

    views, cameras, camera_source = load_views(
        args.scene_dir,
        selected_view_indices,
        args.dataset_root,
        args.subset_name,
        args.target_size,
    )
    roots = select_roots(
        hair0,
        template,
        max_roots=args.max_roots,
        min_support=args.min_support,
        seed=args.seed,
    )
    control_scores = make_control_root_scores(roots, query, latents)
    chains = {
        name: build_strand_chain(roots, control_scores[name], control=name, chain_steps=args.chain_steps)
        for name in CONTROL_NAMES
    }

    control_metrics: dict[str, dict[str, Any]] = {}
    control_outputs: dict[str, dict[str, str]] = {}
    for name, chain in chains.items():
        ply_path = args.output_dir / f"b_hair1_backend_{name}_strand_gaussian_chain.ply"
        write_chain_ply(ply_path, chain, args.chain_steps)
        render_metrics = render_control(
            name,
            chain,
            views,
            cameras,
            target_size=args.target_size,
            point_radius=args.point_radius,
            output_dir=args.output_dir,
        )
        control_metrics[name] = render_metrics
        control_outputs[name] = {"ply": str(ply_path.resolve())}

    npz_path = args.output_dir / "b_hair1_backend_strand_gaussian_chain_artifact.npz"
    np.savez_compressed(
        npz_path,
        selected_view_indices=selected_view_indices.astype(np.int32),
        real_points=chains["real"]["points"].astype(np.float32),
        real_chain_ids=chains["real"]["chain_ids"].astype(np.int32),
        real_step_ids=chains["real"]["step_ids"].astype(np.int32),
        real_root_points=chains["real"]["root_points"].astype(np.float32),
        real_root_score=chains["real"]["root_score"].astype(np.float32),
        real_root_support=chains["real"]["root_support"].astype(np.float32),
        real_root_vertex_ids=chains["real"]["root_vertex_ids"].astype(np.int64),
        real_opacity=chains["real"]["opacity"].astype(np.float32),
        real_scale=chains["real"]["scale"].astype(np.float32),
        shuffle_root_score=chains["shuffle"]["root_score"].astype(np.float32),
        zero_root_score=chains["zero"]["root_score"].astype(np.float32),
        mask_only_root_score=chains["mask_only"]["root_score"].astype(np.float32),
        strict_candidate_passes=np.asarray([0], dtype=np.int32),
        strict_teacher_passes=np.asarray([0], dtype=np.int32),
        research_only=np.asarray([1], dtype=np.int32),
    )
    support_ply = args.output_dir / "b_hair1_backend_head_hairline_headtop_support.ply"
    write_support_ply(support_ply, template, hair0, chains["real"])
    contact_sheet = args.output_dir / "b_hair1_backend_head_hairline_headtop_contact_sheet.png"
    make_contact_sheet(
        contact_sheet,
        views,
        chains,
        cameras,
        target_size=args.target_size,
        point_radius=args.point_radius,
    )

    controls_summary = {
        name: summarize_control(name, chains[name], control_metrics[name])
        for name in CONTROL_NAMES
    }
    real = controls_summary["real"]
    shuffle = controls_summary["shuffle"]
    zero = controls_summary["zero"]
    mask_only = controls_summary["mask_only"]
    comparison = {
        "real_minus_shuffle_root_score": float(real["root_score"]["mean"] - shuffle["root_score"]["mean"]),
        "real_minus_zero_root_score": float(real["root_score"]["mean"] - zero["root_score"]["mean"]),
        "real_minus_mask_only_root_score": float(real["root_score"]["mean"] - mask_only["root_score"]["mean"]),
        "real_minus_shuffle_iou": float(real["mean_iou"] - shuffle["mean_iou"]),
        "real_minus_zero_iou": float(real["mean_iou"] - zero["mean_iou"]),
        "real_minus_mask_only_iou": float(real["mean_iou"] - mask_only["mean_iou"]),
        "real_rgb_better_than_shuffle": bool(real["mean_rgb_residual"] < shuffle["mean_rgb_residual"]),
        "real_rgb_better_than_zero": bool(real["mean_rgb_residual"] < zero["mean_rgb_residual"]),
        "real_rgb_better_than_mask_only": bool(real["mean_rgb_residual"] < mask_only["mean_rgb_residual"]),
    }
    real_beats_controls = (
        comparison["real_minus_shuffle_root_score"] > 0.01
        and comparison["real_minus_zero_root_score"] > 0.01
        and comparison["real_minus_shuffle_iou"] >= -0.002
        and comparison["real_minus_zero_iou"] >= -0.002
    )
    gate_color = "red"
    decision = (
        "B-hair1 backend artifact exists, but this remains research-only and gate-red; strict review/export/cloud stay blocked."
        if real_beats_controls
        else "B-hair1 real does not beat the requested controls strongly enough; keep gates red and do not promote this backend."
    )
    status = "research_only_b_hair1_backend_smoke_no_export_gate_red"
    outputs = [
        str((args.output_dir / f"b_hair1_backend_{name}_strand_gaussian_chain.ply").resolve())
        for name in CONTROL_NAMES
    ]
    outputs.extend(
        [
            str(npz_path.resolve()),
            str(support_ply.resolve()),
            str(contact_sheet.resolve()),
            str((args.output_dir / "b_hair1_backend_summary.json").resolve()),
            str((args.output_dir / "b_hair1_backend_report.md").resolve()),
            str(args.status_json.resolve()),
            str(args.status_report.resolve()),
        ]
    )
    summary = {
        "status": status,
        "gate_color": gate_color,
        "real_beats_controls": bool(real_beats_controls),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "hair0_arrays": str(args.hair0_arrays.resolve()),
            "query_evidence": str(args.query_evidence.resolve()),
            "latent_real": str(args.latent_real.resolve()),
            "latent_shuffle": str(args.latent_shuffle.resolve()),
            "latent_zero": str(args.latent_zero.resolve()),
            "camera_source": camera_source,
            "selected_view_indices": selected_view_indices.astype(int).tolist(),
            "target_size": int(args.target_size),
        },
        "primitive": {
            "backend": "v6_rooted_hairline_strand_gaussian_chain",
            "not_cap": True,
            "not_floating_dots": True,
            "has_chain_edges": True,
            "root_source": "B_hair0 hairline vertices with support gating",
            "root_count": int(chains["real"]["root_points"].shape[0]),
            "chain_steps": int(args.chain_steps),
            "chain_point_count": int(chains["real"]["points"].shape[0]),
            "real_chain_extent": point_stats(chains["real"]["points"]),
            "hairline_support": stats(np.asarray(hair0["hairline_support"], dtype=np.float32)),
            "selected_root_support": stats(chains["real"]["root_support"]),
        },
        "controls": controls_summary,
        "control_outputs": control_outputs,
        "comparison": comparison,
        "decision": decision,
        "outputs": outputs,
    }
    write_json(args.output_dir / "b_hair1_backend_summary.json", summary)
    write_report(args.output_dir / "b_hair1_backend_report.md", summary)
    write_json(args.status_json, summary)
    write_report(args.status_report, summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": status,
                    "gate_color": gate_color,
                    "real_beats_controls": real_beats_controls,
                    "real_mean_iou": real["mean_iou"],
                    "real_root_score_mean": real["root_score"]["mean"],
                    "comparison": comparison,
                    "decision": decision,
                }
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
