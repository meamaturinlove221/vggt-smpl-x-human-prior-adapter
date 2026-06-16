from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V33_head_face_detail_route"
REFINED = OUT_DIR / "v33_head_face_refined_teacher.npz"
V24_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
REPORT_JSON = ROOT / "reports" / "20260508_v33_face_normal_relief_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v33_face_normal_relief_audit.md"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stats(values: np.ndarray) -> dict:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "finite": 0}
    return {
        "count": int(vals.size),
        "finite": int(vals.size),
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refined = np.load(REFINED, allow_pickle=True)
    v24 = np.load(V24_TARGETS, allow_pickle=True)
    points = refined["refined_points_world"].astype(np.float32)
    normals = refined["refined_normals_world"].astype(np.float32)
    region = refined["refined_region_id_map"].astype(np.uint8)
    base_points = v24["teacher_points_world"].astype(np.float32)
    base_normals = v24["teacher_normals_world"].astype(np.float32)

    audits = {}
    for name, rid in [("head", 1), ("face", 2)]:
        mask = region == rid
        normal_len = np.linalg.norm(normals[mask], axis=-1) if int(mask.sum()) else np.array([])
        base_dot = np.sum(normals[mask] * base_normals[mask], axis=-1) if int(mask.sum()) else np.array([])
        relief = np.linalg.norm((points - base_points)[mask], axis=-1) if int(mask.sum()) else np.array([])
        audits[name] = {
            "pixels": int(mask.sum()),
            "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
            "normal_length": _stats(normal_len),
            "normal_alignment_to_v24": _stats(base_dot),
            "relief_norm": _stats(relief),
            "normal_pass": bool(normal_len.size > 0 and float(np.median(normal_len)) > 0.9 and float(np.median(base_dot)) > 0.2),
            "relief_nonexplosive": bool(relief.size > 0 and float(np.percentile(relief, 95)) < 1.0),
        }

    status = "DONE_PASS" if all(v["pixels"] > 0 and v["normal_pass"] and v["relief_nonexplosive"] for v in audits.values()) else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v33_face_normal_relief_audit",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_flame": True,
        "metrics": audits,
        "inputs": {"refined_npz": str(REFINED), "v24_targets": str(V24_TARGETS)},
        "outputs": {"report_json": str(REPORT_JSON), "report_md": str(REPORT_MD)},
        "blockers": [] if status == "DONE_PASS" else ["head_face_normal_or_relief_audit_failed"],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V33 Face Normal Relief Audit",
                "",
                f"Status: `{status}`",
                "",
                f"- Head pixels: {audits['head']['pixels']}, normal pass: {audits['head']['normal_pass']}",
                f"- Face pixels: {audits['face']['pixels']}, normal pass: {audits['face']['normal_pass']}",
                "",
                "No formal output was written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
