from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from v23_residual_evidence_mask_repair import (
    DEFAULT_CASE_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROI_NPZ,
    REGION_ID,
    REGION_ORDER,
    build_repaired_region_evidence,
    finite_stats,
    json_ready,
    load_manifest,
    load_npz,
    normalize_vectors,
    safe_v23_output_dir,
    write_json,
    write_repaired_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_JSON = REPORTS / "20260508_v23_residual_surface_v2.json"
DEFAULT_MD = REPORTS / "20260508_v23_residual_surface_v2.md"
DEFAULT_MASK_NPZ = DEFAULT_OUTPUT_DIR / "v23_repaired_region_evidence_masks.npz"
DEFAULT_MASK_JSON = DEFAULT_OUTPUT_DIR / "v23_repaired_region_evidence_masks_summary.json"
DEFAULT_SURFACE_NPZ = DEFAULT_OUTPUT_DIR / "v23_residual_surface_v2_points.npz"
DEFAULT_SAPIENS_NORMAL_NPZ = (
    LOCAL_ROOT / "V15_S_sapiens_normal_convention_solver" / "v15_sapiens_normals_best_camera_convention.npz"
)

REGION_COLOR = {
    "body": (145, 188, 226),
    "head": (202, 129, 236),
    "face": (255, 218, 93),
    "left_hand": (255, 147, 93),
    "right_hand": (105, 221, 178),
}
REGION_WEIGHT = {
    "body": 0.10,
    "head": 0.16,
    "face": 0.20,
    "left_hand": 0.24,
    "right_hand": 0.24,
}
REGION_CLIP_M = {
    "body": 0.025,
    "head": 0.020,
    "face": 0.015,
    "left_hand": 0.030,
    "right_hand": 0.030,
}
REGION_RELIEF_M = {
    "body": 0.0015,
    "head": 0.0020,
    "face": 0.0025,
    "left_hand": 0.0022,
    "right_hand": 0.0022,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def camera_id_from_name(name: str) -> str | None:
    match = re.search(r"cam(\d+)", str(name))
    if match:
        return match.group(1).zfill(2)
    parts = re.findall(r"\d+", str(name))
    return parts[-1].zfill(2) if parts else None


def resize_nearest_2d(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(arr)
    dst_h, dst_w = int(shape[0]), int(shape[1])
    if src.shape[:2] == (dst_h, dst_w):
        return src.copy()
    y = np.clip(np.rint((np.arange(dst_h) + 0.5) * src.shape[0] / dst_h - 0.5).astype(np.int64), 0, src.shape[0] - 1)
    x = np.clip(np.rint((np.arange(dst_w) + 0.5) * src.shape[1] / dst_w - 0.5).astype(np.int64), 0, src.shape[1] - 1)
    return src[y[:, None], x[None, :]]


def match_sapiens_to_case(
    path: Path,
    case_camera_ids: list[str],
    target_hw: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    meta: dict[str, Any] = {"sapiens_normal_npz": path, "exists": path.is_file(), "used": False}
    if not path.is_file():
        return None, None, meta
    payload = load_npz(path)
    if "normal" not in payload or "mask" not in payload or "image_names" not in payload:
        meta["warning"] = "sapiens_npz_missing_normal_mask_or_image_names"
        return None, None, meta
    normals = np.asarray(payload["normal"], dtype=np.float32)
    masks = np.asarray(payload["mask"], dtype=bool)
    names = [str(x) for x in payload["image_names"]]
    sap_ids = [camera_id_from_name(name) for name in names]
    rows: list[dict[str, Any]] = []
    matched_normals: list[np.ndarray] = []
    matched_masks: list[np.ndarray] = []
    for case_idx, camera_id in enumerate(case_camera_ids):
        selected = None
        exact = [idx for idx, sap_id in enumerate(sap_ids) if sap_id == str(camera_id).zfill(2)]
        if exact:
            selected = exact[0]
            mode = "exact_camera_id"
            delta = 0
        else:
            numeric = int(camera_id) if str(camera_id).isdigit() else None
            candidates = [
                (abs(int(sap_id) - numeric), idx)
                for idx, sap_id in enumerate(sap_ids)
                if numeric is not None and sap_id and sap_id.isdigit()
            ]
            candidates.sort()
            if candidates and candidates[0][0] <= 1:
                delta, selected = candidates[0]
                mode = "nearest_camera_id_delta_le_1"
            else:
                delta = None
                mode = "unmatched"
        rows.append(
            {
                "case_index": int(case_idx),
                "case_camera_id": str(camera_id),
                "sapiens_index": None if selected is None else int(selected),
                "sapiens_name": None if selected is None else names[selected],
                "camera_id_delta": delta,
                "mode": mode,
            }
        )
        if selected is None:
            matched_normals.append(np.zeros(target_hw + (3,), dtype=np.float32))
            matched_masks.append(np.zeros(target_hw, dtype=bool))
            continue
        matched_normals.append(resize_nearest_2d(normals[selected], target_hw).astype(np.float32))
        matched_masks.append(resize_nearest_2d(masks[selected], target_hw).astype(bool))
    stacked_normals = np.stack(matched_normals, axis=0)
    stacked_masks = np.stack(matched_masks, axis=0)
    stacked_normals, normal_len = normalize_vectors(stacked_normals)
    stacked_masks &= normal_len > 0.5
    meta.update(
        {
            "used": bool(int(stacked_masks.sum()) > 0),
            "matched_view_count": int(sum(row["sapiens_index"] is not None for row in rows)),
            "valid_pixels": int(stacked_masks.sum()),
            "convention_name": str(payload.get("convention_name", "unknown")),
            "match_rows": rows,
        }
    )
    return stacked_normals, stacked_masks, meta


def normal_agreement_metrics(a: np.ndarray, b: np.ndarray | None, mask: np.ndarray) -> dict[str, Any]:
    if b is None:
        return {"pixels": 0}
    valid = np.asarray(mask, dtype=bool)
    if int(valid.sum()) == 0:
        return {"pixels": 0}
    au, alen = normalize_vectors(a)
    bu, blen = normalize_vectors(b)
    valid &= (alen > 0.5) & (blen > 0.5) & np.isfinite(au).all(axis=-1) & np.isfinite(bu).all(axis=-1)
    if int(valid.sum()) == 0:
        return {"pixels": 0}
    dots = np.clip(np.sum(au[valid] * bu[valid], axis=-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(dots))
    return {
        "pixels": int(dots.size),
        "cosine_mean": float(np.mean(dots)),
        "cosine_median": float(np.median(dots)),
        "angle_mean_deg": float(np.mean(angle)),
        "angle_median_deg": float(np.median(angle)),
        "negative_dot_fraction": float(np.mean(dots < 0.0)),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if colors is None:
        cols = np.full((pts.shape[0], 3), 210, dtype=np.uint8)
    else:
        cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[finite]
    nrm = None
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)[finite]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {pts.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if nrm is not None:
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        if nrm is None:
            for point, color in zip(pts, cols):
                handle.write(
                    f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        else:
            for point, normal, color in zip(pts, nrm, cols):
                handle.write(
                    f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                    f"{float(normal[0]):.6f} {float(normal[1]):.6f} {float(normal[2]):.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )


def clip_vectors(vec: np.ndarray, limit: float) -> np.ndarray:
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(limit) / np.maximum(length, 1e-6))
    return (vec * scale).astype(np.float32)


def sample_indices_by_region(region_masks: dict[str, np.ndarray], sample_limit: int, seed: int) -> np.ndarray:
    all_indices: list[np.ndarray] = []
    total_pixels = sum(int(region_masks[name].sum()) for name in REGION_ORDER)
    if total_pixels <= int(sample_limit):
        for name in REGION_ORDER:
            all_indices.append(np.flatnonzero(region_masks[name].reshape(-1)))
        return np.sort(np.concatenate(all_indices)) if all_indices else np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    remaining = int(sample_limit)
    remaining_regions = [name for name in REGION_ORDER if int(region_masks[name].sum()) > 0]
    for idx, name in enumerate(remaining_regions):
        flat = np.flatnonzero(region_masks[name].reshape(-1))
        if idx == len(remaining_regions) - 1:
            take = min(flat.size, remaining)
        else:
            proportional = max(1, int(round(sample_limit * (flat.size / max(total_pixels, 1)))))
            take = min(flat.size, max(1, min(remaining - (len(remaining_regions) - idx - 1), proportional)))
        selected = flat if flat.size <= take else np.sort(rng.choice(flat, size=take, replace=False))
        all_indices.append(selected)
        remaining -= int(selected.size)
    return np.sort(np.concatenate(all_indices)) if all_indices else np.asarray([], dtype=np.int64)


def colorize(region_names: np.ndarray, residual_norm: np.ndarray) -> np.ndarray:
    colors = np.zeros((region_names.shape[0], 3), dtype=np.uint8)
    finite = residual_norm[np.isfinite(residual_norm)]
    scale = float(np.percentile(finite, 95.0)) if finite.size else 1.0
    strength = np.clip(residual_norm / max(scale, 1e-6), 0.0, 1.0)
    hot = np.asarray([255, 42, 36], dtype=np.float32)
    for name in REGION_ORDER:
        mask = region_names == name
        if not bool(mask.any()):
            continue
        base = np.asarray(REGION_COLOR[name], dtype=np.float32)
        colors[mask] = np.clip(base * (1.0 - strength[mask, None] * 0.55) + hot * (strength[mask, None] * 0.55), 0, 255).astype(np.uint8)
    return colors


def build_residual_surface_v2(
    case_root: Path,
    output_dir: Path,
    roi_npz: Path,
    sapiens_normal_npz: Path,
    sample_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = load_npz(case_root / "inputs.npz")
    targets = load_npz(case_root / "targets.npz")
    manifest = load_manifest(case_root / "case_manifest.json")
    mask_arrays, mask_summary = build_repaired_region_evidence(case_root, roi_npz)

    prior_points = np.asarray(targets["prior_points"], dtype=np.float32)
    world_points = np.asarray(targets["world_points"], dtype=np.float32)
    prior_normals, prior_normal_len = normalize_vectors(np.asarray(targets["prior_normals"], dtype=np.float32))
    region_masks = {name: np.asarray(mask_arrays[f"{name}_evidence_mask"], dtype=bool) for name in REGION_ORDER}
    evidence_mask = np.zeros(prior_points.shape[:3], dtype=bool)
    region_id_map = np.zeros(prior_points.shape[:3], dtype=np.uint8)
    for name in REGION_ORDER:
        evidence_mask |= region_masks[name]
        region_id_map[region_masks[name]] = np.uint8(REGION_ID[name])

    case_camera_ids = [str(x) for x in np.asarray(inputs.get("camera_ids", np.arange(prior_points.shape[0])))]
    sapiens_normals, sapiens_mask, sapiens_meta = match_sapiens_to_case(
        sapiens_normal_npz,
        case_camera_ids,
        prior_points.shape[1:3],
    )

    normal_guidance = prior_normals.copy()
    normal_evidence_mask = evidence_mask.copy()
    if sapiens_normals is not None and sapiens_mask is not None and int(sapiens_mask.sum()) > 0:
        shared = evidence_mask & sapiens_mask
        blended = prior_normals.copy()
        blended[shared] = 0.75 * prior_normals[shared] + 0.25 * sapiens_normals[shared]
        normal_guidance, _ = normalize_vectors(blended)
        normal_evidence_mask = shared

    raw_residual = np.zeros_like(prior_points, dtype=np.float32)
    raw_residual[evidence_mask] = world_points[evidence_mask] - prior_points[evidence_mask]
    bounded_residual = np.zeros_like(prior_points, dtype=np.float32)
    weak_residual = np.zeros_like(prior_points, dtype=np.float32)
    normal_relief = np.zeros_like(prior_points, dtype=np.float32)
    for name in REGION_ORDER:
        mask = region_masks[name]
        if int(mask.sum()) == 0:
            continue
        clipped = clip_vectors(raw_residual[mask], REGION_CLIP_M[name])
        bounded_residual[mask] = clipped
        weak_residual[mask] = clipped * np.float32(REGION_WEIGHT[name])
        normal_relief[mask] = normal_guidance[mask] * np.float32(REGION_RELIEF_M[name])

    residual_surface = prior_points + weak_residual + normal_relief
    applied_residual = residual_surface - prior_points
    raw_residual_norm = np.linalg.norm(raw_residual, axis=-1)
    applied_residual_norm = np.linalg.norm(applied_residual, axis=-1)

    valid_indices = sample_indices_by_region(region_masks, int(sample_limit), 20260523)
    flat_prior = prior_points.reshape(-1, 3)
    flat_surface = residual_surface.reshape(-1, 3)
    flat_normals = normal_guidance.reshape(-1, 3)
    flat_raw_residual = raw_residual.reshape(-1, 3)
    flat_applied_residual = applied_residual.reshape(-1, 3)
    flat_region_id = region_id_map.reshape(-1)
    flat_applied_norm = applied_residual_norm.reshape(-1)
    sample_region_names = np.full(valid_indices.shape, "unknown", dtype=object)
    for name in REGION_ORDER:
        sample_region_names[flat_region_id[valid_indices] == REGION_ID[name]] = name
    colors = colorize(sample_region_names, flat_applied_norm[valid_indices])

    residual_ply = output_dir / "v23_residual_surface_v2_points.ply"
    prior_ply = output_dir / "v23_prior_region_sample_points.ply"
    surface_npz = output_dir / "v23_residual_surface_v2_points.npz"
    mask_npz = output_dir / "v23_repaired_region_evidence_masks.npz"
    mask_json = output_dir / "v23_repaired_region_evidence_masks_summary.json"
    write_ply(residual_ply, flat_surface[valid_indices], colors, flat_normals[valid_indices])
    write_ply(prior_ply, flat_prior[valid_indices], np.full_like(colors, 178, dtype=np.uint8), flat_normals[valid_indices])
    np.savez_compressed(
        surface_npz,
        residual_surface_points=flat_surface[valid_indices].astype(np.float32),
        prior_sample_points=flat_prior[valid_indices].astype(np.float32),
        raw_residual=flat_raw_residual[valid_indices].astype(np.float32),
        applied_residual=flat_applied_residual[valid_indices].astype(np.float32),
        normals=flat_normals[valid_indices].astype(np.float32),
        colors=colors.astype(np.uint8),
        sample_indices=valid_indices.astype(np.int64),
        sample_region_ids=flat_region_id[valid_indices].astype(np.uint8),
        sample_region_names=sample_region_names.astype(str),
        region_names=np.asarray(REGION_ORDER),
        region_ids=np.asarray([REGION_ID[name] for name in REGION_ORDER], dtype=np.uint8),
        evidence_mask=evidence_mask.astype(np.uint8),
        region_id_map=region_id_map.astype(np.uint8),
        raw_residual_norm=raw_residual_norm.astype(np.float32),
        applied_residual_norm=applied_residual_norm.astype(np.float32),
    )
    write_repaired_masks(mask_npz, mask_json, mask_arrays, mask_summary)

    raw_to_applied_ratio = np.divide(
        applied_residual_norm,
        np.maximum(raw_residual_norm, 1e-6),
        out=np.zeros_like(applied_residual_norm),
        where=evidence_mask,
    )
    region_metrics: dict[str, Any] = {}
    blockers: list[str] = []
    for name in REGION_ORDER:
        mask = region_masks[name]
        sampled = int(np.sum(sample_region_names == name))
        if mask_summary["regions"][name]["raw_silhouette_overlap_pixels"] > 0 and int(mask.sum()) <= 0:
            blockers.append(f"{name} evidence is empty despite raw silhouette overlap")
        if int(mask.sum()) > 0 and sampled <= 0:
            blockers.append(f"{name} sampled_points is zero despite repaired evidence")
        region_metrics[name] = {
            "pixels": int(mask.sum()),
            "sampled_points": sampled,
            "raw_silhouette_overlap_pixels": int(mask_summary["regions"][name]["raw_silhouette_overlap_pixels"]),
            "native_visible_overlap_pixels": int(mask_summary["regions"][name]["native_visible_overlap_pixels"]),
            "per_view_pixels": mask_summary["regions"][name]["per_view_pixels"],
            "source": mask_summary["regions"][name]["source"],
            "weak_weight": float(REGION_WEIGHT[name]),
            "clip_limit_m": float(REGION_CLIP_M[name]),
            "normal_relief_m": float(REGION_RELIEF_M[name]),
            "raw_residual_norm": finite_stats(raw_residual_norm, mask),
            "applied_residual_norm": finite_stats(applied_residual_norm, mask),
        }

    normal_metrics = {
        "prior_normal_length": finite_stats(prior_normal_len, evidence_mask),
        "sapiens_available": bool(sapiens_meta.get("used", False)),
        "sapiens_matched_view_count": int(sapiens_meta.get("matched_view_count", 0)),
        "sapiens_valid_pixels_after_resize": int(sapiens_meta.get("valid_pixels", 0)),
        "prior_vs_sapiens": normal_agreement_metrics(prior_normals, sapiens_normals, normal_evidence_mask),
    }
    metrics = {
        "view_count": int(prior_points.shape[0]),
        "height": int(prior_points.shape[1]),
        "width": int(prior_points.shape[2]),
        "evidence_pixels": int(evidence_mask.sum()),
        "sampled_points": int(valid_indices.size),
        "raw_residual_norm": finite_stats(raw_residual_norm, evidence_mask),
        "applied_residual_norm": finite_stats(applied_residual_norm, evidence_mask),
        "applied_to_raw_residual_ratio": finite_stats(raw_to_applied_ratio, evidence_mask & (raw_residual_norm > 1e-6)),
        "region_metrics": region_metrics,
        "normal_metrics": normal_metrics,
        "max_applied_residual_m": float(applied_residual_norm[evidence_mask].max()) if int(evidence_mask.sum()) else 0.0,
        "mean_applied_residual_m": float(applied_residual_norm[evidence_mask].mean()) if int(evidence_mask.sum()) else 0.0,
    }

    audit = {
        "inputs": {
            "case_root": case_root,
            "inputs_npz": case_root / "inputs.npz",
            "targets_npz": case_root / "targets.npz",
            "case_manifest": case_root / "case_manifest.json",
            "roi_npz": roi_npz,
            "sapiens_normal_npz": sapiens_normal_npz,
        },
        "case": {
            "case_id": manifest.get("case_id", case_root.name),
            "camera_ids": case_camera_ids,
            "smplx_native_only": True,
            "no_mano": True,
            "no_flame": True,
            "no_hairgs": True,
        },
        "mask_summary": mask_summary,
        "residual_policy": {
            "anchor": "V15 SMPL-X native prior points with V23 repaired raw-silhouette region evidence",
            "mode": "weak_bounded_residual_plus_optional_normal_relief",
            "v17_fix": "head/face/hand/body sampling is driven by per-region V16 ROI raw silhouette support, not global smplx_native_visible_mask",
            "weak_weights": REGION_WEIGHT,
            "clip_limits_m": REGION_CLIP_M,
            "normal_relief_m": REGION_RELIEF_M,
            "sapiens_usage": "optional image-normal blend at 25 percent where matched; never a metric teacher or formal pass",
        },
        "outputs": {
            "residual_surface_ply": residual_ply,
            "prior_sample_ply": prior_ply,
            "residual_surface_npz": surface_npz,
            "mask_npz": mask_npz,
            "mask_summary_json": mask_json,
            "summary_json": output_dir / "summary.json",
        },
        "blockers": blockers,
    }
    return metrics, audit


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V23 Residual Surface V2",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only/no formal pass. This run writes only V23 residual evidence, PLY, NPZ, audit, and report artifacts.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Key Metrics",
        "",
    ]
    metrics = summary.get("metrics", {})
    for key in ("evidence_pixels", "sampled_points", "mean_applied_residual_m", "max_applied_residual_m"):
        if key in metrics:
            lines.append(f"- {key}: `{json_ready(metrics[key])}`")
    lines.extend(["", "## Region Metrics", ""])
    for name in REGION_ORDER:
        row = metrics.get("region_metrics", {}).get(name, {})
        raw = row.get("raw_residual_norm", {})
        applied = row.get("applied_residual_norm", {})
        lines.append(
            "- "
            f"{name}: pixels=`{row.get('pixels')}`, sampled=`{row.get('sampled_points')}`, "
            f"raw_support=`{row.get('raw_silhouette_overlap_pixels')}`, "
            f"native_overlap=`{row.get('native_visible_overlap_pixels')}`, "
            f"raw_mean_m=`{raw.get('mean')}`, applied_mean_m=`{applied.get('mean')}`, "
            f"applied_p95_m=`{applied.get('p95')}`"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Guardrails", ""])
    for key in (
        "research_only",
        "smplx_native_only",
        "no_mano",
        "no_flame",
        "no_hairgs",
        "no_predictions_write",
        "no_teacher_export",
        "no_candidate_export",
        "no_registry_write",
        "no_package_write",
        "no_strict_pass_claim",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V23 SMPL-X residual surface optimizer v2 with repaired per-region evidence.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-npz", type=Path, default=DEFAULT_ROI_NPZ)
    parser.add_argument("--sapiens-normal-npz", type=Path, default=DEFAULT_SAPIENS_NORMAL_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--sample-limit", type=int, default=70000)
    args = parser.parse_args()

    output_dir = safe_v23_output_dir(args.output_dir)
    if not (args.case_root / "inputs.npz").is_file() or not (args.case_root / "targets.npz").is_file():
        raise FileNotFoundError(f"Missing V15 native prior case inputs/targets under {args.case_root}")

    metrics, audit = build_residual_surface_v2(
        args.case_root,
        output_dir,
        args.roi_npz,
        args.sapiens_normal_npz,
        int(args.sample_limit),
    )
    nonempty_regions = {
        name: bool(metrics["region_metrics"][name]["pixels"] > 0 and metrics["region_metrics"][name]["sampled_points"] > 0)
        for name in REGION_ORDER
    }
    blockers = list(audit.get("blockers", []))
    missing = [name for name, ok in nonempty_regions.items() if not ok]
    if missing:
        blockers.append(f"required V23 regions empty or unsampled: {missing}")

    status = "DONE_PASS" if not blockers else "FAIL_ROUTED"
    summary = {
        "task": "v23_residual_surface_v2",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "smplx_native_only": True,
        "no_mano": True,
        "no_flame": True,
        "no_hairgs": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_package_write": True,
        "no_strict_pass_claim": True,
        "decision": (
            "V23 repaired the V17 zero head/face sampling bug by building per-region evidence from V16 ROI maps "
            "intersected with V15 raw silhouette support. The resulting research surface has nonempty body/head/face/"
            "left_hand/right_hand evidence and samples."
        ),
        "inputs": {
            "case_root": args.case_root,
            "roi_npz": args.roi_npz,
            "sapiens_normal_npz": args.sapiens_normal_npz,
        },
        "metrics": metrics,
        "outputs": audit["outputs"],
        "blockers": blockers,
        "route_token": status,
    }
    audit["summary_status"] = status
    audit["metrics"] = metrics

    write_json(args.output_json, summary)
    write_json(output_dir / "summary.json", summary)
    write_markdown(args.output_md, summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": status,
                    "summary_json": args.output_json,
                    "residual_surface_ply": audit["outputs"]["residual_surface_ply"],
                    "residual_surface_npz": audit["outputs"]["residual_surface_npz"],
                    "region_pixels": {k: v["pixels"] for k, v in metrics["region_metrics"].items()},
                    "region_sampled_points": {k: v["sampled_points"] for k, v in metrics["region_metrics"].items()},
                    "blockers": blockers,
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0 if status == "DONE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
