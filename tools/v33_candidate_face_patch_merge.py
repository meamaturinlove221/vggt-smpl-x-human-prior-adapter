from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V33_head_face_detail_route"
REFINED = OUT_DIR / "v33_head_face_refined_teacher.npz"
REPORT_JSON = ROOT / "reports" / "20260508_v33_candidate_face_patch_merge.json"
REPORT_MD = ROOT / "reports" / "20260508_v33_candidate_face_patch_merge.md"

OPTIONAL_V32 = [
    ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research" / "candidate_points_world_research.npz",
    ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference" / "candidate_points_world_research.npz",
    ROOT / "output" / "surface_research_cloud_preflight" / "V32_candidate_inference" / "candidate_points_world_research.npz",
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_optional_candidate() -> tuple[np.ndarray | None, str | None]:
    for p in OPTIONAL_V32:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            for key in ("candidate_points_world", "points_world", "research_points_world"):
                if key in z.files:
                    return z[key].astype(np.float32), str(p)
    return None, None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refined = np.load(REFINED, allow_pickle=True)
    refined_points = refined["refined_points_world"].astype(np.float32)
    refined_normals = refined["refined_normals_world"].astype(np.float32)
    refined_region = refined["refined_region_id_map"].astype(np.uint8)
    candidate, candidate_path = _load_optional_candidate()

    if candidate is not None and candidate.shape == refined_points.shape:
        merged = candidate.copy()
        source = np.zeros(refined_region.shape, dtype=np.uint8)
        mask = refined_region > 0
        merged[mask] = refined_points[mask]
        source[mask] = refined_region[mask]
        candidate_optional_missing = False
    else:
        merged = refined_points.copy()
        source = refined_region.copy()
        candidate_optional_missing = True

    np.savez_compressed(
        OUT_DIR / "v33_candidate_face_patch_merged_research.npz",
        merged_points_world=merged,
        merged_normals_world=refined_normals,
        merged_region_source=source,
        candidate_optional_missing=np.array(candidate_optional_missing),
        research_only=np.array(True),
        no_formal_outputs=np.array(True),
    )

    pts = merged[source > 0]
    regs = source[source > 0]
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[regs == 1] = np.array([80, 180, 255], dtype=np.uint8)
    colors[regs == 2] = np.array([255, 170, 80], dtype=np.uint8)
    ply = OUT_DIR / "v33_candidate_face_patch_merged_research.ply"
    with ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    status = "DONE_PASS" if int((source == 1).sum()) > 0 and int((source == 2).sum()) > 0 else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v33_candidate_face_patch_merge",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_flame": True,
        "candidate_optional_missing": candidate_optional_missing,
        "candidate_path": candidate_path,
        "metrics": {
            "head_patch_pixels": int((source == 1).sum()),
            "face_patch_pixels": int((source == 2).sum()),
            "merged_patch_points": int((source > 0).sum()),
        },
        "outputs": {
            "merged_npz": str(OUT_DIR / "v33_candidate_face_patch_merged_research.npz"),
            "merged_ply": str(ply),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "blockers": [] if status == "DONE_PASS" else ["head_or_face_patch_empty"],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V33 Candidate Face Patch Merge",
                "",
                f"Status: `{status}`",
                "",
                f"- Candidate optional missing: `{candidate_optional_missing}`",
                f"- Head patch pixels: {summary['metrics']['head_patch_pixels']}",
                f"- Face patch pixels: {summary['metrics']['face_patch_pixels']}",
                "",
                "The output is a research-only patch merge. It is not a formal candidate package.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
