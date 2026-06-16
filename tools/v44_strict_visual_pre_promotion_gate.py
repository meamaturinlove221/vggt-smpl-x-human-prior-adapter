from __future__ import annotations

from pathlib import Path

from v44_v50_common import (
    LOCAL_OUT,
    REPORTS,
    REQUIRED_REGIONS,
    base_stage_statuses,
    file_row,
    read_json,
    region_metric_pass,
    scan_forbidden,
    utc_now,
    v30_prior_prediction_ready,
    write_json,
    write_md,
)


OUT = LOCAL_OUT / "V44_strict_visual_pre_promotion_gate"
JSON = REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.json"
MD = REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v32 = read_json(REPORTS / "20260508_v32_candidate_inference_region_audit.json")
    v29 = read_json(REPORTS / "20260508_v29_normal_route_rescue.json")
    v30_ready, v30_blockers, _ = v30_prior_prediction_ready()
    metrics = v32.get("region_metrics") if isinstance(v32.get("region_metrics"), dict) else {}
    region_ok, region_blockers = region_metric_pass(metrics)
    contact_sheet = Path(str(v32.get("contact_sheet", "")))
    candidate_points = Path(str(v32.get("candidate_points_world", "")))
    candidate_normals = Path(str(v32.get("candidate_normals_geometric", "")))
    visual_items = {
        "full_body_open3d_source": file_row(contact_sheet),
        "candidate_points": file_row(candidate_points),
        "candidate_geometric_normals": file_row(candidate_normals),
    }
    blockers: list[str] = []
    blockers.extend(region_blockers)
    if not contact_sheet.is_file():
        blockers.append("V32 contact sheet missing")
    if not candidate_points.is_file():
        blockers.append("V32 candidate points missing")
    if not candidate_normals.is_file():
        blockers.append("V32 candidate normals missing")
    if v29.get("status") != "DONE_PASS":
        blockers.append("V29 normal route not pass")
    if not v30_ready:
        blockers.extend(v30_blockers)
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V44 output detected")
    # V44 is intentionally strict: without prior-enabled prediction pass, visual pre-promotion fails.
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    review = {
        "status": "PASS" if status == "DONE_PASS" else "FAIL",
        "regions": {region: bool(metrics.get(region, {}).get("pixel_count", metrics.get(region, {}).get("pixels", 0))) for region in REQUIRED_REGIONS},
        "requires_prior_enabled_prediction_pass": True,
        "v30_prior_prediction_ready": v30_ready,
        "blockers": blockers,
    }
    review_path = OUT / ("visual_review_codex_pass.json" if status == "DONE_PASS" else "visual_review_codex_fail.json")
    write_json(review_path, review)
    summary = {
        "task": "v44_strict_visual_pre_promotion_gate",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "visual_review_path": review_path,
        "region_support_pass": region_ok,
        "v30_prior_prediction_ready": v30_ready,
        "visual_items": visual_items,
        "stage_statuses": base_stage_statuses(),
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V44 pre-promotion visual gate cannot pass until prior-enabled predictions are available." if blockers else "V44 visual pre-promotion gate passed.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V44 Strict Visual Pre-Promotion Gate", summary, [
        f"- visual_review_path: `{review_path}`",
        f"- v30_prior_prediction_ready: `{v30_ready}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
