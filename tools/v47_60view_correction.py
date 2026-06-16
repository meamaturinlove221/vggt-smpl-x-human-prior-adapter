from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, file_row, read_json, scan_forbidden, utc_now, write_json, write_md


OUT = LOCAL_OUT / "V47_60view_correction"
JSON = REPORTS / "20260509_v47_60view_correction.json"
MD = REPORTS / "20260509_v47_60view_correction.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v35 = read_json(REPORTS / "20260508_v35_60view_support_expansion.json")
    reraster = v35.get("reraster_summary") if isinstance(v35.get("reraster_summary"), dict) else {}
    inv = reraster.get("scene_level_60v_prior_inventory") if isinstance(reraster.get("scene_level_60v_prior_inventory"), dict) else {}
    prior_maps = inv.get("prior_maps") if isinstance(inv.get("prior_maps"), dict) else {}
    region_pass = v35.get("region_6v_support_pass") if isinstance(v35.get("region_6v_support_pass"), dict) else {}
    blockers: list[str] = []
    if v35.get("status") != "DONE_PASS":
        blockers.append("V35 60-view source did not pass")
    if not bool(v35.get("has_60v_scene")):
        blockers.append("60-view scene missing")
    if not bool(v35.get("has_scene_level_60v_prior")):
        blockers.append("scene-level 60-view prior missing")
    if int(inv.get("views_with_prior_mask", 0) or 0) < 60:
        blockers.append("60-view prior does not cover all views")
    for region, ok in region_pass.items():
        if not ok:
            blockers.append(f"{region}_6v_support_failed")
    prior_path = Path(str(prior_maps.get("path", "")))
    if not prior_path.is_file():
        blockers.append("60-view prior_maps.npz missing")
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V47 output detected")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v47_60view_correction",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "has_60v_scene": bool(v35.get("has_60v_scene")),
        "scene_level_60v_prior_usable": bool(inv),
        "views_with_prior_mask": int(inv.get("views_with_prior_mask", 0) or 0),
        "channel_count": int(inv.get("channel_count", 0) or 0),
        "prior_maps": file_row(prior_path),
        "region_6v_support_pass": region_pass,
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V47 60-view support correction is usable for research pre-promotion." if status == "DONE_PASS" else "V47 routed; 60-view support remains insufficient.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V47 60-View Correction", summary, [
        f"- views_with_prior_mask: `{summary['views_with_prior_mask']}`",
        f"- channel_count: `{summary['channel_count']}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
