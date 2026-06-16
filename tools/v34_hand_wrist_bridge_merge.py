from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V34_smplx_native_hand_route"
PATCH = OUT_DIR / "v34_smplx_native_hand_continuity_patch.npz"
REPORT_JSON = ROOT / "reports" / "20260508_v34_hand_wrist_bridge_merge.json"
REPORT_MD = ROOT / "reports" / "20260508_v34_hand_wrist_bridge_merge.md"

OPTIONAL_V32 = [
    ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research" / "candidate_points_world_research.npz",
    ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference" / "candidate_points_world_research.npz",
    ROOT / "output" / "surface_research_cloud_preflight" / "V32_candidate_inference" / "candidate_points_world_research.npz",
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _candidate() -> tuple[np.ndarray | None, str | None]:
    for p in OPTIONAL_V32:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            for key in ("candidate_points_world", "points_world", "research_points_world"):
                if key in z.files:
                    return z[key].astype(np.float32), str(p)
    return None, None


def main() -> None:
    patch = np.load(PATCH, allow_pickle=True)
    hand_points = patch["hand_points_world"].astype(np.float32)
    hand_normals = patch["hand_normals_world"].astype(np.float32)
    region = patch["hand_region_id_map"].astype(np.uint8)
    cand, cand_path = _candidate()
    candidate_optional_missing = cand is None or cand.shape != hand_points.shape
    if candidate_optional_missing:
        merged = hand_points.copy()
    else:
        merged = cand.copy()
        merged[region > 0] = hand_points[region > 0]
    np.savez_compressed(
        OUT_DIR / "v34_hand_wrist_bridge_merged_research.npz",
        merged_points_world=merged,
        merged_normals_world=hand_normals,
        merged_region_source=region,
        candidate_optional_missing=np.array(candidate_optional_missing),
        research_only=np.array(True),
    )
    pts = merged[region > 0]
    regs = region[region > 0]
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[regs == 1] = np.array([80, 255, 130], dtype=np.uint8)
    colors[regs == 2] = np.array([255, 80, 130], dtype=np.uint8)
    ply = OUT_DIR / "v34_hand_wrist_bridge_merged_research.ply"
    with ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    status = "DONE_PASS" if int((region == 1).sum()) > 0 and int((region == 2).sum()) > 0 else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v34_hand_wrist_bridge_merge",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_mano": True,
        "candidate_optional_missing": candidate_optional_missing,
        "candidate_path": cand_path,
        "metrics": {
            "left_region_pixels": int((region == 1).sum()),
            "right_region_pixels": int((region == 2).sum()),
            "merged_region_pixels": int((region > 0).sum()),
        },
        "outputs": {
            "merged_npz": str(OUT_DIR / "v34_hand_wrist_bridge_merged_research.npz"),
            "merged_ply": str(ply),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "blockers": [] if status == "DONE_PASS" else ["left_or_right_hand_merge_empty"],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V34 Hand Wrist Bridge Merge",
                "",
                f"Status: `{status}`",
                "",
                f"- Candidate optional missing: `{candidate_optional_missing}`",
                f"- Left pixels: {summary['metrics']['left_region_pixels']}",
                f"- Right pixels: {summary['metrics']['right_region_pixels']}",
                "",
                "Research-only merge; no formal candidate output was written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
