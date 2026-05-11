from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, file_row, read_json, scan_forbidden, utc_now, write_json, write_md


OUT = LOCAL_OUT / "V48_temporal_correction"
JSON = REPORTS / "20260509_v48_temporal_correction.json"
MD = REPORTS / "20260509_v48_temporal_correction.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v26 = read_json(REPORTS / "20260508_v26_temporal_canonical_teacher.json")
    v29 = read_json(REPORTS / "20260508_v29_normal_route_rescue.json")
    targets = Path(str(v26.get("output_targets", v26.get("target_path", ""))))
    if not targets.is_file():
        targets = Path("output/surface_research_preflight_local/V26_temporal_canonical_teacher/v26_temporal_canonical_teacher_targets.npz")
    temporal_normals = Path(str((v29.get("outputs") or {}).get("temporal_normals", "")))
    temporal_summary = v29.get("temporal_summary") if isinstance(v29.get("temporal_summary"), dict) else {}
    region_support = temporal_summary.get("region_normal_support") if isinstance(temporal_summary.get("region_normal_support"), dict) else {}
    blockers: list[str] = []
    if v26.get("status") != "DONE_PASS":
        blockers.append("V26 temporal canonical teacher source did not pass")
    if not targets.is_file():
        blockers.append("V26 temporal target file missing")
    if not bool(v29.get("temporal_normal_available")):
        blockers.append("V29 temporal normal reconstruction unavailable")
    for region in ("body", "head", "face", "left_hand", "right_hand"):
        if int(region_support.get(region, 0) or 0) <= 0:
            blockers.append(f"{region}_temporal_normal_support_empty")
    if not temporal_normals.is_file():
        blockers.append("V29 temporal normals file missing")
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V48 output detected")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v48_temporal_correction",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "v26_status": v26.get("status"),
        "temporal_normal_available": bool(v29.get("temporal_normal_available")),
        "temporal_region_normal_support": region_support,
        "artifacts": {
            "v26_targets": file_row(targets),
            "v29_temporal_normals": file_row(temporal_normals),
        },
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V48 temporal correction is usable for research pre-promotion." if status == "DONE_PASS" else "V48 routed; temporal correction remains insufficient.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V48 Temporal Correction", summary, [
        f"- temporal_normal_available: `{summary['temporal_normal_available']}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
