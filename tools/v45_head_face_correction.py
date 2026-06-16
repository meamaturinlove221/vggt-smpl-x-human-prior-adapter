from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, file_row, read_json, scan_forbidden, utc_now, write_json, write_md


OUT = LOCAL_OUT / "V45_head_face_correction"
JSON = REPORTS / "20260509_v45_head_face_correction.json"
MD = REPORTS / "20260509_v45_head_face_correction.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v33 = read_json(REPORTS / "20260508_v33_head_face_detail_route.json")
    metrics = v33.get("metrics") if isinstance(v33.get("metrics"), dict) else {}
    coverage = metrics.get("coverage") if isinstance(metrics.get("coverage"), dict) else {}
    head = coverage.get("head", {}) if isinstance(coverage.get("head"), dict) else {}
    face = coverage.get("face", {}) if isinstance(coverage.get("face"), dict) else {}
    outputs = v33.get("outputs") if isinstance(v33.get("outputs"), dict) else {}
    refined = Path(str(outputs.get("refined_npz", "")))
    refined_ply = Path(str(outputs.get("refined_ply", "")))
    blockers: list[str] = []
    if v33.get("status") != "DONE_PASS":
        blockers.append("V33 head/face correction source did not pass")
    if int(head.get("pixels", 0) or 0) <= 0:
        blockers.append("head correction support empty")
    if int(face.get("pixels", 0) or 0) <= 0:
        blockers.append("face correction support empty")
    if not refined.is_file():
        blockers.append("head/face refined NPZ missing")
    if not refined_ply.is_file():
        blockers.append("head/face refined PLY missing")
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V45 output detected")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v45_head_face_correction",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "smplx_native_only": True,
        "head_pixels": int(head.get("pixels", 0) or 0),
        "face_pixels": int(face.get("pixels", 0) or 0),
        "normal_relief_source": "V33 SMPL-X-native head/face detail route",
        "artifacts": {
            "refined_npz": file_row(refined),
            "refined_ply": file_row(refined_ply),
        },
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V45 head/face correction is usable for research pre-promotion." if status == "DONE_PASS" else "V45 routed; head/face correction remains insufficient.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V45 Head/Face Correction", summary, [
        f"- head_pixels: `{summary['head_pixels']}`",
        f"- face_pixels: `{summary['face_pixels']}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
