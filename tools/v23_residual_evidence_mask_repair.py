from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"

DEFAULT_CASE_ROOT = REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_ROI_NPZ = LOCAL_ROOT / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V23_residual_surface_v2"
DEFAULT_MASK_NPZ = DEFAULT_OUTPUT_DIR / "v23_repaired_region_evidence_masks.npz"
DEFAULT_MASK_JSON = DEFAULT_OUTPUT_DIR / "v23_repaired_region_evidence_masks_summary.json"

REGION_ORDER = ("body", "head", "face", "left_hand", "right_hand")
ROI_BY_REGION = {
    "body": "body_visible",
    "head": "head",
    "face": "face_front",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
}
REGION_ID = {name: idx + 1 for idx, name in enumerate(REGION_ORDER)}

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "strict_pass",
    "candidate_gate",
    "package",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def safe_v23_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    expected_root = (LOCAL_ROOT / "V23_residual_surface_v2").resolve().as_posix().lower()
    if lower != expected_root and not lower.startswith(expected_root + "/"):
        raise ValueError(f"Refusing non-V23 research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_npz(path: Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_vectors(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arr, dtype=np.float32)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    out = np.divide(values, np.maximum(length, 1e-6), out=np.zeros_like(values), where=length > 1e-6)
    return out.astype(np.float32), length[..., 0].astype(np.float32)


def finite_stats(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values.reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
    }


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def load_roi_maps(roi_npz: Path, expected_shape: tuple[int, int, int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta: dict[str, Any] = {
        "roi_npz": roi_npz,
        "exists": roi_npz.is_file(),
        "used": False,
        "required_roi_names": ROI_BY_REGION,
    }
    if not roi_npz.is_file():
        raise FileNotFoundError(f"Missing V16 ROI map NPZ: {roi_npz}")
    payload = load_npz(roi_npz)
    if "roi_maps" not in payload or "roi_names" not in payload:
        raise KeyError(f"V16 ROI NPZ lacks roi_maps/roi_names: {roi_npz}")
    roi_maps = np.asarray(payload["roi_maps"], dtype=bool)
    roi_names = [str(x) for x in np.asarray(payload["roi_names"])]
    if roi_maps.shape[0] != expected_shape[0] or roi_maps.shape[-2:] != expected_shape[-2:]:
        raise ValueError(f"V16 ROI shape {roi_maps.shape} does not match case shape {expected_shape}")
    missing = [roi_name for roi_name in ROI_BY_REGION.values() if roi_name not in roi_names]
    if missing:
        raise KeyError(f"V16 ROI NPZ missing required regions: {missing}")
    regions = {name: roi_maps[:, roi_names.index(roi_name)].copy() for name, roi_name in ROI_BY_REGION.items()}
    meta.update(
        {
            "used": True,
            "roi_names": roi_names,
            "roi_shape": list(roi_maps.shape),
            "missing_required_roi_names": missing,
        }
    )
    return regions, meta


def make_region_masks_exclusive(candidate_masks: dict[str, np.ndarray], raw_support: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw_support = np.asarray(raw_support, dtype=bool)
    assigned = np.zeros(raw_support.shape, dtype=bool)
    exclusive: dict[str, np.ndarray] = {}
    priority = ("face", "head", "left_hand", "right_hand", "body")
    for name in priority:
        mask = np.asarray(candidate_masks[name], dtype=bool) & raw_support & ~assigned
        exclusive[name] = mask
        assigned |= mask
    exclusive["body"] = (exclusive["body"] | (raw_support & ~assigned)) & raw_support

    region_id_map = np.zeros(raw_support.shape, dtype=np.uint8)
    for name in REGION_ORDER:
        region_id_map[exclusive[name]] = np.uint8(REGION_ID[name])
    return {name: exclusive[name] for name in REGION_ORDER}, region_id_map


def build_repaired_region_evidence(
    case_root: Path,
    roi_npz: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    inputs = load_npz(case_root / "inputs.npz")
    targets = load_npz(case_root / "targets.npz")
    manifest = load_manifest(case_root / "case_manifest.json")

    prior_points = np.asarray(targets["prior_points"], dtype=np.float32)
    world_points = np.asarray(targets["world_points"], dtype=np.float32)
    prior_normals, prior_normal_len = normalize_vectors(np.asarray(targets["prior_normals"], dtype=np.float32))
    raw_point_mask = np.asarray(inputs["point_masks"], dtype=bool)
    teacher_mask = np.asarray(targets.get("teacher_mask", raw_point_mask), dtype=bool)
    soft_alpha = np.asarray(inputs.get("soft_alpha", raw_point_mask.astype(np.float32)), dtype=np.float32)
    native_visible = np.asarray(targets.get("smplx_native_visible_mask", np.zeros(raw_point_mask.shape, dtype=bool)), dtype=bool)
    conf = np.asarray(targets.get("world_points_conf", np.ones(raw_point_mask.shape, dtype=np.float32)), dtype=np.float32)

    raw_silhouette_support = (
        raw_point_mask
        & teacher_mask
        & (soft_alpha > 0.0)
        & (conf > 0.05)
        & np.isfinite(prior_points).all(axis=-1)
        & np.isfinite(world_points).all(axis=-1)
        & (prior_normal_len > 0.5)
    )

    roi_regions, roi_meta = load_roi_maps(roi_npz, raw_silhouette_support.shape)
    roi_supported = {name: roi_regions[name] & raw_silhouette_support for name in REGION_ORDER}
    repaired_regions, region_id_map = make_region_masks_exclusive(roi_supported, raw_silhouette_support)

    per_view_region_pixels: dict[str, list[int]] = {}
    region_summary: dict[str, Any] = {}
    blockers: list[str] = []
    for name in REGION_ORDER:
        mask = repaired_regions[name]
        raw_overlap = roi_regions[name] & raw_silhouette_support
        per_view = [int(mask[idx].sum()) for idx in range(mask.shape[0])]
        per_view_region_pixels[name] = per_view
        raw_overlap_pixels = int(raw_overlap.sum())
        final_pixels = int(mask.sum())
        if raw_overlap_pixels > 0 and final_pixels <= 0:
            blockers.append(f"{name} raw silhouette support exists but repaired evidence is empty")
        region_summary[name] = {
            "roi_name": ROI_BY_REGION[name],
            "roi_pixels": int(roi_regions[name].sum()),
            "raw_silhouette_overlap_pixels": raw_overlap_pixels,
            "repaired_pixels": final_pixels,
            "per_view_pixels": per_view,
            "bbox_xyxy_per_view": [bbox_from_mask(mask[idx]) for idx in range(mask.shape[0])],
            "source": "v16_roi_intersect_v15_raw_silhouette_support_exclusive",
            "native_visible_overlap_pixels": int((roi_regions[name] & native_visible).sum()),
        }

    summary = {
        "task": "v23_residual_evidence_mask_repair",
        "created_utc": utc_now(),
        "status": "v23_region_evidence_masks_repaired",
        "case": {
            "case_id": manifest.get("case_id", case_root.name),
            "camera_ids": [str(x) for x in np.asarray(inputs.get("camera_ids", np.arange(raw_point_mask.shape[0])))],
            "smplx_native_only": True,
            "no_mano": True,
            "no_flame": True,
            "no_hairgs": True,
        },
        "inputs": {
            "case_root": case_root,
            "inputs_npz": case_root / "inputs.npz",
            "targets_npz": case_root / "targets.npz",
            "roi_npz": roi_npz,
        },
        "mask_policy": {
            "bug_fixed": "V17 intersected every region with smplx_native_visible_mask; V23 uses per-region raw silhouette support so head and face are not erased.",
            "raw_support_formula": "point_masks & teacher_mask & soft_alpha>0 & conf>0.05 & finite prior/world points & nonzero prior normals",
            "exclusive_priority": ["face", "head", "left_hand", "right_hand", "body"],
            "required_nonempty_if_raw_supported": list(REGION_ORDER),
        },
        "global_counts": {
            "raw_point_mask_pixels": int(raw_point_mask.sum()),
            "teacher_mask_pixels": int(teacher_mask.sum()),
            "raw_silhouette_support_pixels": int(raw_silhouette_support.sum()),
            "smplx_native_visible_pixels": int(native_visible.sum()),
            "exclusive_region_pixels": int(sum(int(repaired_regions[name].sum()) for name in REGION_ORDER)),
            "unassigned_raw_support_pixels": int((raw_silhouette_support & (region_id_map == 0)).sum()),
        },
        "regions": region_summary,
        "roi_meta": roi_meta,
        "blockers": blockers,
    }

    arrays = {
        "raw_silhouette_support": raw_silhouette_support,
        "region_id_map": region_id_map,
        "region_names": np.asarray(REGION_ORDER),
        "region_ids": np.asarray([REGION_ID[name] for name in REGION_ORDER], dtype=np.uint8),
        "prior_normals_unit": prior_normals.astype(np.float32),
    }
    for name in REGION_ORDER:
        arrays[f"{name}_evidence_mask"] = repaired_regions[name].astype(np.uint8)
        arrays[f"{name}_roi_raw_overlap_mask"] = roi_supported[name].astype(np.uint8)
    return arrays, summary


def write_repaired_masks(mask_npz: Path, summary_json: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    mask_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_npz, **arrays)
    summary = dict(summary)
    summary["outputs"] = {
        "mask_npz": mask_npz,
        "summary_json": summary_json,
    }
    write_json(summary_json, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="V23 repair of per-region residual evidence masks.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--roi-npz", type=Path, default=DEFAULT_ROI_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask-npz", type=Path, default=DEFAULT_MASK_NPZ)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_MASK_JSON)
    args = parser.parse_args()

    output_dir = safe_v23_output_dir(args.output_dir)
    mask_npz = args.mask_npz if args.mask_npz.is_absolute() else output_dir / args.mask_npz
    summary_json = args.summary_json if args.summary_json.is_absolute() else output_dir / args.summary_json
    if not (args.case_root / "inputs.npz").is_file() or not (args.case_root / "targets.npz").is_file():
        raise FileNotFoundError(f"Missing V15 case inputs/targets under {args.case_root}")
    arrays, summary = build_repaired_region_evidence(args.case_root, args.roi_npz)
    write_repaired_masks(mask_npz, summary_json, arrays, summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "mask_npz": mask_npz,
                    "summary_json": summary_json,
                    "region_pixels": {name: summary["regions"][name]["repaired_pixels"] for name in REGION_ORDER},
                    "blockers": summary["blockers"],
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
