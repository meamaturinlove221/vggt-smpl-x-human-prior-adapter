from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
V29_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V29_normal_route_rescue"
V24_TARGETS = (
    REPO_ROOT
    / "output"
    / "surface_research_preflight_local"
    / "V24_residual_teacher_v2"
    / "v24_residual_teacher_targets_v2.npz"
)
V26_TARGETS = (
    REPO_ROOT
    / "output"
    / "surface_research_preflight_local"
    / "V26_temporal_canonical_teacher"
    / "v26_temporal_canonical_teacher_targets.npz"
)
V25_POINTS = (
    REPO_ROOT
    / "output"
    / "surface_research_cloud_preflight"
    / "V25_research_vggt_predictions"
    / "research_points_world.npz"
)
REGION_NAMES = ("body", "head", "face", "left_hand", "right_hand")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_npz(path: Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def normalize_vectors(vectors: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vectors, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    length = np.linalg.norm(arr, axis=-1)
    valid = np.isfinite(arr).all(axis=-1) & np.isfinite(length) & (length > eps)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    out[valid] = arr[valid] / length[valid, None]
    return out, valid


def normals_from_point_map(points: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 4 or pts.shape[-1] != 3:
        raise ValueError(f"Expected point map [V,H,W,3], got {pts.shape}")
    dx = np.zeros_like(pts, dtype=np.float32)
    dy = np.zeros_like(pts, dtype=np.float32)
    dx[:, :, 1:-1] = pts[:, :, 2:] - pts[:, :, :-2]
    dx[:, :, 0] = pts[:, :, 1] - pts[:, :, 0]
    dx[:, :, -1] = pts[:, :, -1] - pts[:, :, -2]
    dy[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[:, 0] = pts[:, 1] - pts[:, 0]
    dy[:, -1] = pts[:, -1] - pts[:, -2]
    raw = np.cross(dx, dy)
    return normalize_vectors(raw, mask=mask, eps=eps)


def vector_length_stats(vectors: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(vectors, dtype=np.float32)
    length = np.linalg.norm(arr, axis=-1)
    if mask is not None:
        length = length[np.asarray(mask, dtype=bool)]
    finite = length[np.isfinite(length)]
    if finite.size == 0:
        return {"count": int(length.size), "finite": 0}
    return {
        "count": int(length.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
    }


def region_support_from_masks(valid: np.ndarray, region_masks: np.ndarray, region_names: tuple[str, ...] = REGION_NAMES) -> dict[str, int]:
    support: dict[str, int] = {}
    for idx, name in enumerate(region_names):
        if idx >= region_masks.shape[0]:
            support[name] = 0
        else:
            support[name] = int((np.asarray(valid, dtype=bool) & np.asarray(region_masks[idx], dtype=bool)).sum())
    return support


def region_support_from_id_map(valid: np.ndarray, region_id_map: np.ndarray, region_names: tuple[str, ...] = REGION_NAMES) -> dict[str, int]:
    support: dict[str, int] = {}
    for idx, name in enumerate(region_names, start=1):
        support[name] = int((np.asarray(valid, dtype=bool) & (np.asarray(region_id_map) == idx)).sum())
    return support


def all_regions_nonempty(support: dict[str, int], region_names: tuple[str, ...] = REGION_NAMES) -> bool:
    return all(int(support.get(name, 0)) > 0 for name in region_names)


def geometric_consistency(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    an, av = normalize_vectors(a)
    bn, bv = normalize_vectors(b)
    valid = av & bv
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return {"valid": 0}
    dot = np.sum(an[valid] * bn[valid], axis=-1)
    abs_dot = np.abs(dot)
    return {
        "valid": int(valid.sum()),
        "mean_dot": float(dot.mean()),
        "mean_abs_dot": float(abs_dot.mean()),
        "median_abs_dot": float(np.median(abs_dot)),
        "p10_abs_dot": float(np.percentile(abs_dot, 10.0)),
    }
