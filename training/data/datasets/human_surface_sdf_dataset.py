from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FAMILY_ORDER = ("full_body", "face_core", "hairline", "left_hand", "right_hand")
FAMILY_TO_PARTS = {
    "full_body": (0, 5),
    "left_hand": (1,),
    "right_hand": (2,),
    "face_core": (3,),
    "hairline": (4,),
}
CONTROL_NAMES = ("real", "zero", "shuffle", "random")


@dataclass(frozen=True)
class HumanSurfaceSDFCase:
    case_id: str
    features: np.ndarray
    labels: np.ndarray
    sdf: np.ndarray
    positions: np.ndarray
    families: np.ndarray
    normals: np.ndarray
    query_indices: np.ndarray
    sample_offsets: np.ndarray
    source: str
    genealogy: dict[str, Any]


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


def load_npz(path: Path, required: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def safe_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.clip(norm, eps, None)).astype(np.float32)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float32)
    if vertices.size == 0 or faces.size == 0:
        return normals
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return safe_normalize(normals)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.float32)
    if not finite.any():
        return out
    center = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - center))
    scale = float(1.4826 * mad)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values[finite]))
    if not np.isfinite(scale) or scale < 1e-6:
        return out
    out[finite] = np.clip((values[finite] - center) / scale, -5.0, 5.0)
    return out.astype(np.float32)


def robust_01(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.full(values.shape, float(default), dtype=np.float32)
    if not finite.any():
        return out
    lo, hi = np.percentile(values[finite], [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = np.clip((values[finite] - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    return out.astype(np.float32)


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "finite": 0}
    if not np.issubdtype(arr.dtype, np.number):
        return {"count": int(arr.size)}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _reduce_features(features: np.ndarray, bins: int, mode: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    bins = max(1, int(bins))
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {features.shape}")
    chunks = np.array_split(np.arange(features.shape[1]), bins)
    reduced = []
    for chunk in chunks:
        if chunk.size == 0:
            row = np.zeros((features.shape[0],), dtype=np.float32)
        elif mode == "rms":
            row = np.sqrt(np.mean(np.square(features[:, chunk]), axis=1)).astype(np.float32)
        else:
            row = np.mean(features[:, chunk], axis=1).astype(np.float32)
        reduced.append(row)
    mat = np.stack(reduced, axis=1).astype(np.float32)
    if mode == "mean":
        return np.stack([robust_z(mat[:, idx]) for idx in range(mat.shape[1])], axis=1).astype(np.float32)
    return np.stack([robust_01(mat[:, idx]) for idx in range(mat.shape[1])], axis=1).astype(np.float32)


def _family_one_hot(families: np.ndarray) -> np.ndarray:
    families = np.asarray(families).astype(str).reshape(-1)
    out = np.zeros((families.shape[0], len(FAMILY_ORDER)), dtype=np.float32)
    for idx, family in enumerate(FAMILY_ORDER):
        out[:, idx] = families == family
    return out


def _template_normals(template_payload: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template = load_npz(template_payload, ("hybrid_vertices", "hybrid_faces", "part_ids"))
    vertices = np.asarray(template["hybrid_vertices"], dtype=np.float32)
    faces = np.asarray(template["hybrid_faces"], dtype=np.int64)
    part_ids = np.asarray(template["part_ids"], dtype=np.int64)
    if "normals" in template and np.asarray(template["normals"]).shape == vertices.shape:
        normals = safe_normalize(np.asarray(template["normals"], dtype=np.float32))
    else:
        normals = compute_vertex_normals(vertices, faces)
    return vertices, normals, part_ids


def load_query_cache_case(
    query_cache: Path,
    template_payload: Path,
    *,
    case_id: str,
    shell_offsets: tuple[float, ...],
    feature_bins: int,
    seed: int,
) -> HumanSurfaceSDFCase:
    cache = load_npz(
        query_cache,
        (
            "query_indices",
            "query_positions",
            "query_families",
            "support",
            "token_ids",
            "uv",
            "depth",
            "mean_features",
            "variance_features",
        ),
    )
    vertices, template_normals, _part_ids = _template_normals(template_payload)
    query_indices = np.asarray(cache["query_indices"], dtype=np.int64).reshape(-1)
    query_positions = np.asarray(cache["query_positions"], dtype=np.float32)
    if query_indices.size != query_positions.shape[0]:
        raise ValueError("query_indices/query_positions count mismatch")
    if query_indices.max(initial=-1) >= template_normals.shape[0]:
        raise ValueError("query index exceeds template normal count")

    normals = safe_normalize(template_normals[query_indices])
    support = np.asarray(cache["support"], dtype=np.float32).reshape(-1)
    token_ids = np.asarray(cache["token_ids"], dtype=np.int32)
    uv = np.asarray(cache["uv"], dtype=np.float32)
    depth = np.asarray(cache["depth"], dtype=np.float32)
    mean_features = np.nan_to_num(np.asarray(cache["mean_features"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    variance_features = np.nan_to_num(
        np.asarray(cache["variance_features"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    view_count = max(1, int(token_ids.shape[1]))
    valid_view = (token_ids >= 0) & np.isfinite(depth) & np.isfinite(uv).all(axis=2)
    depth_mean = np.zeros((query_positions.shape[0],), dtype=np.float32)
    depth_std = np.zeros_like(depth_mean)
    uv_spread = np.zeros_like(depth_mean)
    for idx in range(query_positions.shape[0]):
        mask = valid_view[idx]
        if not mask.any():
            continue
        local_depth = depth[idx, mask]
        local_uv = uv[idx, mask]
        depth_mean[idx] = float(np.mean(local_depth))
        depth_std[idx] = float(np.std(local_depth)) if local_depth.size > 1 else 0.0
        if local_uv.shape[0] > 1:
            centered = local_uv - local_uv.mean(axis=0, keepdims=True)
            uv_spread[idx] = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))) / 518.0)

    base_features = np.concatenate(
        [
            _reduce_features(mean_features, feature_bins, "mean"),
            _reduce_features(variance_features, feature_bins, "rms"),
            np.clip(support[:, None] / float(view_count), 0.0, 1.0).astype(np.float32),
            robust_z(depth_mean)[:, None],
            robust_01(depth_std)[:, None],
            robust_01(uv_spread)[:, None],
            _family_one_hot(np.asarray(cache["query_families"]).astype(str)),
        ],
        axis=1,
    ).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    rows = []
    labels = []
    sdf = []
    positions = []
    normals_out = []
    families_out = []
    query_indices_out = []
    sample_offsets = []
    for offset in shell_offsets:
        offset_value = float(offset)
        offset_feature = np.full((base_features.shape[0], 1), offset_value, dtype=np.float32)
        abs_feature = np.full((base_features.shape[0], 1), abs(offset_value), dtype=np.float32)
        signed_feature = np.full((base_features.shape[0], 1), np.sign(offset_value), dtype=np.float32)
        jitter = rng.normal(0.0, 0.002, size=(base_features.shape[0], 1)).astype(np.float32)
        rows.append(np.concatenate([base_features, offset_feature, abs_feature, signed_feature, jitter], axis=1))
        labels.append(np.full((base_features.shape[0],), 1 if offset_value <= 0.0 else 0, dtype=np.float32))
        sdf.append(np.full((base_features.shape[0],), offset_value, dtype=np.float32))
        positions.append((query_positions + normals * offset_value).astype(np.float32))
        normals_out.append(normals)
        families_out.append(np.asarray(cache["query_families"]).astype("<U32"))
        query_indices_out.append(query_indices)
        sample_offsets.append(np.full((base_features.shape[0],), offset_value, dtype=np.float32))

    return HumanSurfaceSDFCase(
        case_id=case_id,
        features=np.concatenate(rows, axis=0).astype(np.float32),
        labels=np.concatenate(labels, axis=0).astype(np.float32),
        sdf=np.concatenate(sdf, axis=0).astype(np.float32),
        positions=np.concatenate(positions, axis=0).astype(np.float32),
        families=np.concatenate(families_out, axis=0).astype("<U32"),
        normals=np.concatenate(normals_out, axis=0).astype(np.float32),
        query_indices=np.concatenate(query_indices_out, axis=0).astype(np.int64),
        sample_offsets=np.concatenate(sample_offsets, axis=0).astype(np.float32),
        source="query_cache_mesh_shell",
        genealogy={
            "case_builder": "load_query_cache_case",
            "query_cache": str(query_cache.expanduser().resolve()),
            "template_payload": str(template_payload.expanduser().resolve()),
            "shell_offsets": [float(v) for v in shell_offsets],
            "feature_bins": int(feature_bins),
            "seed": int(seed),
            "template_vertex_count": int(vertices.shape[0]),
            "query_count": int(query_positions.shape[0]),
        },
    )


def _maybe_load_json(path: Path) -> dict[str, Any]:
    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_external_case(
    case_root: Path,
    *,
    case_id: str | None = None,
    feature_bins: int = 32,
) -> HumanSurfaceSDFCase:
    root = case_root.expanduser().resolve()
    manifest = _maybe_load_json(root / "case_manifest.json")
    payload_path = root / str(manifest.get("surface_sdf_npz", "surface_sdf.npz"))
    if not payload_path.is_file():
        payload_path = root / "b_fus3d2_surface_sdf_case.npz"
    data = load_npz(payload_path, ("features", "labels", "sdf", "positions"))
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim == 2 and features.shape[1] > feature_bins * 4:
        features = np.concatenate(
            [
                _reduce_features(features, feature_bins, "mean"),
                _reduce_features(features, feature_bins, "rms"),
            ],
            axis=1,
        ).astype(np.float32)
    n = int(features.shape[0])
    families = np.asarray(data["families"]).astype("<U32") if "families" in data else np.asarray(["external"] * n, dtype="<U32")
    normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data else np.zeros((n, 3), dtype=np.float32)
    return HumanSurfaceSDFCase(
        case_id=case_id or str(manifest.get("case_id", root.name)),
        features=features.astype(np.float32),
        labels=np.asarray(data["labels"], dtype=np.float32).reshape(-1),
        sdf=np.asarray(data["sdf"], dtype=np.float32).reshape(-1),
        positions=np.asarray(data["positions"], dtype=np.float32),
        families=families,
        normals=normals,
        query_indices=np.asarray(data["query_indices"], dtype=np.int64) if "query_indices" in data else np.arange(n, dtype=np.int64),
        sample_offsets=np.asarray(data["sample_offsets"], dtype=np.float32) if "sample_offsets" in data else np.asarray(data["sdf"], dtype=np.float32).reshape(-1),
        source="external_surface_sdf_case",
        genealogy={
            "case_builder": "load_external_case",
            "case_root": str(root),
            "payload": str(payload_path),
            "manifest": manifest,
        },
    )


class HumanSurfaceSDFDataset:
    """Small query-level SDF dataset used by the V8-A Fus3D-Human smoke.

    It intentionally returns arrays instead of VGGT image batches. The formal
    trainer remains untouched; this adapter is a research-only bridge from
    mesh/token evidence to bounded dataset-level supervision.
    """

    def __init__(
        self,
        *,
        query_cache: Path,
        template_payload: Path,
        external_case_roots: list[Path] | None = None,
        max_cases: int = 4,
        shell_offsets: tuple[float, ...] = (-0.012, -0.006, 0.0, 0.006, 0.012),
        feature_bins: int = 32,
        seed: int = 20260507,
    ) -> None:
        self.query_cache = query_cache
        self.template_payload = template_payload
        self.external_case_roots = [Path(v) for v in (external_case_roots or [])]
        self.max_cases = max(1, int(max_cases))
        self.shell_offsets = tuple(float(v) for v in shell_offsets)
        self.feature_bins = int(feature_bins)
        self.seed = int(seed)
        self.cases = self._load_cases()

    def _load_cases(self) -> list[HumanSurfaceSDFCase]:
        cases: list[HumanSurfaceSDFCase] = []
        for idx, root in enumerate(self.external_case_roots[: self.max_cases]):
            cases.append(load_external_case(root, case_id=f"external_{idx:03d}_{root.name}", feature_bins=self.feature_bins))
        if cases:
            return cases[: self.max_cases]

        # No external dataset is present: synthesize deterministic dataset-level
        # cases from the same mesh/query supervision using small shell/feature
        # perturbations. This keeps the smoke bounded while exercising case-level
        # iteration, aggregation, and control comparisons.
        for idx in range(self.max_cases):
            scale = 1.0 + (idx - max(self.max_cases - 1, 1) * 0.5) * 0.08
            offsets = tuple(float(v) * scale for v in self.shell_offsets)
            cases.append(
                load_query_cache_case(
                    self.query_cache,
                    self.template_payload,
                    case_id=f"synthetic_mesh_case_{idx:03d}",
                    shell_offsets=offsets,
                    feature_bins=self.feature_bins,
                    seed=self.seed + idx,
                )
            )
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> HumanSurfaceSDFCase:
        return self.cases[int(idx) % len(self.cases)]

    def as_arrays(self) -> dict[str, np.ndarray]:
        features = np.concatenate([case.features for case in self.cases], axis=0).astype(np.float32)
        labels = np.concatenate([case.labels for case in self.cases], axis=0).astype(np.float32)
        sdf = np.concatenate([case.sdf for case in self.cases], axis=0).astype(np.float32)
        positions = np.concatenate([case.positions for case in self.cases], axis=0).astype(np.float32)
        families = np.concatenate([case.families for case in self.cases], axis=0).astype("<U32")
        normals = np.concatenate([case.normals for case in self.cases], axis=0).astype(np.float32)
        case_ids = np.concatenate(
            [np.asarray([case.case_id] * case.features.shape[0], dtype="<U96") for case in self.cases],
            axis=0,
        )
        sample_offsets = np.concatenate([case.sample_offsets for case in self.cases], axis=0).astype(np.float32)
        return {
            "features": features,
            "labels": labels,
            "sdf": sdf,
            "positions": positions,
            "families": families,
            "normals": normals,
            "case_ids": case_ids,
            "sample_offsets": sample_offsets,
        }

    def genealogy(self) -> dict[str, Any]:
        return {
            "dataset": "HumanSurfaceSDFDataset",
            "case_count": len(self.cases),
            "synthetic_mesh_supervision": not bool(self.external_case_roots),
            "external_case_roots": [str(v.expanduser().resolve()) for v in self.external_case_roots],
            "query_cache": str(self.query_cache.expanduser().resolve()),
            "template_payload": str(self.template_payload.expanduser().resolve()),
            "shell_offsets": [float(v) for v in self.shell_offsets],
            "feature_bins": int(self.feature_bins),
            "seed": int(self.seed),
            "cases": [case.genealogy for case in self.cases],
        }


def apply_control_features(features: np.ndarray, control: str, seed: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if control == "real":
        return features.copy()
    if control == "zero":
        out = features.copy()
        # Preserve the final signed shell columns so zero-control remains a
        # learnability baseline over geometry-only offsets, not label leakage.
        preserve = min(4, out.shape[1])
        out[:, : max(0, out.shape[1] - preserve)] = 0.0
        return out
    if control == "shuffle":
        out = features.copy()
        preserve = min(4, out.shape[1])
        perm = rng.permutation(out.shape[0])
        out[:, : max(0, out.shape[1] - preserve)] = out[perm, : max(0, out.shape[1] - preserve)]
        return out
    if control == "random":
        out = features.copy()
        preserve = min(4, out.shape[1])
        width = max(0, out.shape[1] - preserve)
        if width > 0:
            mean = np.mean(out[:, :width], axis=0, keepdims=True)
            std = np.std(out[:, :width], axis=0, keepdims=True)
            out[:, :width] = rng.normal(mean, np.clip(std, 0.05, None), size=(out.shape[0], width)).astype(np.float32)
        return out
    raise ValueError(f"Unknown control {control!r}; expected {CONTROL_NAMES}")
