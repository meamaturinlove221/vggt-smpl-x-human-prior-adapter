from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from v23_residual_evidence_mask_repair import (
    DEFAULT_CASE_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROI_NPZ,
    REGION_ORDER,
    build_repaired_region_evidence,
    json_ready,
    load_npz,
    safe_v23_output_dir,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_SUMMARY_JSON = REPORTS / "20260508_v23_residual_surface_v2.json"
DEFAULT_AUDIT_JSON = REPORTS / "20260508_v23_residual_surface_v2_region_audit.json"
DEFAULT_AUDIT_MD = REPORTS / "20260508_v23_residual_surface_v2_region_audit.md"
DEFAULT_SURFACE_NPZ = DEFAULT_OUTPUT_DIR / "v23_residual_surface_v2_points.npz"
DEFAULT_SURFACE_PLY = DEFAULT_OUTPUT_DIR / "v23_residual_surface_v2_points.ply"
DEFAULT_MASK_NPZ = DEFAULT_OUTPUT_DIR / "v23_repaired_region_evidence_masks.npz"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
    }


def read_ply_vertex_count(path: Path) -> tuple[int, str]:
    if not path.is_file():
        return 0, "missing"
    fmt = "unknown"
    vertex_count = 0
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "format":
                fmt = parts[1]
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            if line == "end_header":
                break
    return vertex_count, fmt


def build_region_audit(
    case_root: Path,
    roi_npz: Path,
    summary_json: Path,
    surface_npz: Path,
    surface_ply: Path,
    mask_npz: Path,
) -> dict[str, Any]:
    summary = read_json(summary_json)
    repaired_arrays, repaired_summary = build_repaired_region_evidence(case_root, roi_npz)
    surface_exists = surface_npz.is_file()
    surface_payload = load_npz(surface_npz) if surface_exists else {}
    mask_payload = load_npz(mask_npz) if mask_npz.is_file() else {}
    ply_vertices, ply_format = read_ply_vertex_count(surface_ply)

    blockers: list[str] = []
    region_rows: dict[str, Any] = {}
    sample_region_names = np.asarray(surface_payload.get("sample_region_names", np.asarray([], dtype=str))).astype(str)
    applied_norm = None
    if "applied_residual" in surface_payload:
        applied_norm = np.linalg.norm(np.asarray(surface_payload["applied_residual"], dtype=np.float32), axis=-1)
    for name in REGION_ORDER:
        repaired_mask = np.asarray(repaired_arrays[f"{name}_evidence_mask"], dtype=bool)
        persisted_mask = np.asarray(mask_payload.get(f"{name}_evidence_mask", repaired_mask.astype(np.uint8)), dtype=bool)
        sampled = int(np.sum(sample_region_names == name))
        raw_supported = int(repaired_summary["regions"][name]["raw_silhouette_overlap_pixels"])
        pixels = int(repaired_mask.sum())
        if raw_supported > 0 and pixels <= 0:
            blockers.append(f"{name} raw silhouette support exists but repaired pixels are zero")
        if pixels > 0 and sampled <= 0:
            blockers.append(f"{name} repaired pixels exist but sampled points are zero")
        if persisted_mask.shape != repaired_mask.shape or not np.array_equal(persisted_mask, repaired_mask):
            blockers.append(f"{name} persisted mask differs from rebuilt repaired mask")
        sample_norm_stats = {"count": 0, "finite": 0}
        if applied_norm is not None and sampled > 0:
            sample_norm_stats = finite_stats(applied_norm[sample_region_names == name])
        region_rows[name] = {
            "raw_silhouette_overlap_pixels": raw_supported,
            "repaired_pixels": pixels,
            "sampled_points": sampled,
            "per_view_pixels": repaired_summary["regions"][name]["per_view_pixels"],
            "persisted_mask_matches_rebuild": bool(
                persisted_mask.shape == repaired_mask.shape and np.array_equal(persisted_mask, repaired_mask)
            ),
            "native_visible_overlap_pixels": int(repaired_summary["regions"][name]["native_visible_overlap_pixels"]),
            "sample_applied_residual_norm": sample_norm_stats,
        }

    if not surface_exists:
        blockers.append(f"missing residual surface NPZ: {surface_npz}")
    if not surface_ply.is_file():
        blockers.append(f"missing residual surface PLY: {surface_ply}")
    expected_samples = int(surface_payload.get("sample_indices", np.asarray([], dtype=np.int64)).shape[0]) if surface_payload else 0
    if ply_vertices != expected_samples:
        blockers.append(f"PLY vertex count {ply_vertices} does not match NPZ sample count {expected_samples}")
    missing_outputs = [
        key
        for key, value in {
            "summary_json": summary_json,
            "surface_npz": surface_npz,
            "surface_ply": surface_ply,
            "mask_npz": mask_npz,
        }.items()
        if not Path(value).is_file()
    ]
    if missing_outputs:
        blockers.append(f"missing outputs: {missing_outputs}")

    status = "DONE_PASS" if not blockers else "FAIL_ROUTED"
    return {
        "task": "v23_residual_region_audit",
        "created_utc": utc_now(),
        "status": status,
        "route_token": status,
        "research_only": True,
        "formal_pass": False,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_package_write": True,
        "no_strict_pass_claim": True,
        "inputs": {
            "case_root": case_root,
            "roi_npz": roi_npz,
            "summary_json": summary_json,
            "surface_npz": surface_npz,
            "surface_ply": surface_ply,
            "mask_npz": mask_npz,
        },
        "summary_status": summary.get("status"),
        "ply": {
            "path": surface_ply,
            "format": ply_format,
            "vertex_count": ply_vertices,
            "npz_sample_count": expected_samples,
        },
        "regions": region_rows,
        "blockers": blockers,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# V23 Residual Surface V2 Region Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "Research-only audit. No predictions, candidate/teacher package, registry, or strict pass state is written.",
        "",
        "## Regions",
        "",
    ]
    for name in REGION_ORDER:
        row = audit["regions"][name]
        lines.append(
            "- "
            f"{name}: raw_support=`{row['raw_silhouette_overlap_pixels']}`, "
            f"repaired=`{row['repaired_pixels']}`, sampled=`{row['sampled_points']}`, "
            f"native_overlap=`{row['native_visible_overlap_pixels']}`, "
            f"persisted_match=`{row['persisted_mask_matches_rebuild']}`"
        )
    lines.extend(["", "## PLY", ""])
    for key, value in audit.get("ply", {}).items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Inputs", ""])
    for key, value in audit.get("inputs", {}).items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in audit.get("blockers", [])] or ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V23 residual surface v2 region evidence and outputs.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--roi-npz", type=Path, default=DEFAULT_ROI_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--surface-npz", type=Path, default=DEFAULT_SURFACE_NPZ)
    parser.add_argument("--surface-ply", type=Path, default=DEFAULT_SURFACE_PLY)
    parser.add_argument("--mask-npz", type=Path, default=DEFAULT_MASK_NPZ)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    args = parser.parse_args()

    safe_v23_output_dir(args.output_dir)
    audit = build_region_audit(
        args.case_root,
        args.roi_npz,
        args.summary_json,
        args.surface_npz,
        args.surface_ply,
        args.mask_npz,
    )
    write_json(args.audit_json, audit)
    write_markdown(args.audit_md, audit)
    print(
        json.dumps(
            json_ready(
                {
                    "status": audit["status"],
                    "audit_json": args.audit_json,
                    "audit_md": args.audit_md,
                    "ply_vertices": audit["ply"]["vertex_count"],
                    "regions": {
                        name: {
                            "raw_support": audit["regions"][name]["raw_silhouette_overlap_pixels"],
                            "repaired": audit["regions"][name]["repaired_pixels"],
                            "sampled": audit["regions"][name]["sampled_points"],
                        }
                        for name in REGION_ORDER
                    },
                    "blockers": audit["blockers"],
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0 if audit["status"] == "DONE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
