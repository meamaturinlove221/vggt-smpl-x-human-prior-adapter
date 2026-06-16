from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V34_smplx_native_hand_route"
PATCH = OUT_DIR / "v34_smplx_native_hand_continuity_patch.npz"
REPORT_JSON = ROOT / "reports" / "20260508_v34_hand_region_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v34_hand_region_audit.md"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stats(vals: np.ndarray) -> dict:
    vals = np.asarray(vals)
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
    patch = np.load(PATCH, allow_pickle=True)
    pts = patch["hand_points_world"].astype(np.float32)
    normals = patch["hand_normals_world"].astype(np.float32)
    region = patch["hand_region_id_map"].astype(np.uint8)
    audit = {}
    for name, rid in [("left_hand", 1), ("right_hand", 2)]:
        mask = region == rid
        p = pts[mask]
        n = normals[mask]
        if p.size:
            bbox = p.max(axis=0) - p.min(axis=0)
            normal_len = np.linalg.norm(n, axis=-1)
        else:
            bbox = np.array([])
            normal_len = np.array([])
        audit[name] = {
            "pixels": int(mask.sum()),
            "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
            "bbox_extent": [float(x) for x in bbox] if bbox.size else [],
            "bbox_extent_norm": float(np.linalg.norm(bbox)) if bbox.size else 0.0,
            "normal_length": _stats(normal_len),
            "not_detached_sparse_patch": bool(mask.sum() > 200 and (float(np.linalg.norm(bbox)) if bbox.size else 0.0) > 0.01),
            "normal_depth_consistency_pass": bool(normal_len.size > 0 and float(np.median(normal_len)) > 0.8),
        }
    status = "DONE_PASS" if all(v["not_detached_sparse_patch"] and v["normal_depth_consistency_pass"] for v in audit.values()) else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v34_hand_region_audit",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_mano": True,
        "metrics": audit,
        "inputs": {"hand_patch": str(PATCH)},
        "outputs": {"report_json": str(REPORT_JSON), "report_md": str(REPORT_MD)},
        "blockers": [] if status == "DONE_PASS" else ["hand_patch_sparse_or_normal_inconsistent"],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V34 Hand Region Audit",
                "",
                f"Status: `{status}`",
                "",
                f"- Left hand pixels: {audit['left_hand']['pixels']}",
                f"- Right hand pixels: {audit['right_hand']['pixels']}",
                "",
                "Research-only audit; no formal output was written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
