from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, file_row, read_json, scan_forbidden, utc_now, v30_prior_prediction_ready, write_json, write_md


OUT = LOCAL_OUT / "V49_package_dry_run"
JSON = REPORTS / "20260509_v49_package_dry_run.json"
MD = REPORTS / "20260509_v49_package_dry_run.md"


def stage_status(name: str) -> str:
    return str(read_json(REPORTS / f"20260509_{name}.json").get("status", "MISSING"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v30_ready, v30_blockers, _ = v30_prior_prediction_ready()
    dryrun_candidate = OUT / "candidate_package_dryrun"
    dryrun_teacher = OUT / "teacher_package_dryrun"
    dryrun_candidate.mkdir(parents=True, exist_ok=True)
    dryrun_teacher.mkdir(parents=True, exist_ok=True)
    stages = {
        "v44_strict_visual_pre_promotion_gate": read_json(REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.json").get("status", "MISSING"),
        "v45_head_face_correction": read_json(REPORTS / "20260509_v45_head_face_correction.json").get("status", "MISSING"),
        "v46_hand_correction": read_json(REPORTS / "20260509_v46_hand_correction.json").get("status", "MISSING"),
        "v47_60view_correction": read_json(REPORTS / "20260509_v47_60view_correction.json").get("status", "MISSING"),
        "v48_temporal_correction": read_json(REPORTS / "20260509_v48_temporal_correction.json").get("status", "MISSING"),
    }
    required_files = {
        "candidate_points": Path("output/surface_research_preflight_local/V32_candidate_inference_research/candidate_points_world_research.npz"),
        "candidate_normals": Path("output/surface_research_preflight_local/V32_candidate_inference_research/candidate_normals_geometric_research.npz"),
        "head_face_patch": Path("output/surface_research_preflight_local/V33_head_face_detail_route/v33_head_face_refined_teacher.npz"),
        "hand_patch": Path("output/surface_research_preflight_local/V34_smplx_native_hand_route/v34_smplx_native_hand_continuity_patch.npz"),
        "temporal_teacher": Path("output/surface_research_preflight_local/V26_temporal_canonical_teacher/v26_temporal_canonical_teacher_targets.npz"),
        "visual_review": Path("output/surface_research_preflight_local/V44_strict_visual_pre_promotion_gate/visual_review_codex_pass.json"),
    }
    blockers: list[str] = []
    for stage, status in stages.items():
        if status != "DONE_PASS":
            blockers.append(f"{stage} not pass: {status}")
    for name, path in required_files.items():
        if not path.is_file():
            blockers.append(f"dry-run required file missing: {name}")
    if not v30_ready:
        blockers.extend(v30_blockers)
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V49 output detected")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    dryrun_entry = {
        "dry_run_only": True,
        "would_write_formal_package": status == "DONE_PASS",
        "strict_registry_not_written": True,
        "candidate_files": {name: str(path) for name, path in required_files.items()},
        "v30_prior_prediction_ready": v30_ready,
        "stage_statuses": stages,
        "blockers": blockers,
    }
    write_json(OUT / "registry_entry_dryrun.json", dryrun_entry)
    write_json(dryrun_candidate / "manifest_dryrun.json", dryrun_entry)
    write_json(dryrun_teacher / "manifest_dryrun.json", dryrun_entry)
    summary = {
        "task": "v49_package_dry_run",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "writes_formal_package": False,
        "writes_strict_registry": False,
        "candidate_package_dryrun": file_row(dryrun_candidate / "manifest_dryrun.json"),
        "teacher_package_dryrun": file_row(dryrun_teacher / "manifest_dryrun.json"),
        "registry_entry_dryrun": file_row(OUT / "registry_entry_dryrun.json"),
        "stage_statuses": stages,
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V49 dry-run passed; final promotion may evaluate strict conditions." if status == "DONE_PASS" else "V49 dry-run routed; formal promotion remains illegal.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V49 Package Dry Run", summary, [
        f"- candidate_package_dryrun: `{dryrun_candidate}`",
        f"- teacher_package_dryrun: `{dryrun_teacher}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
