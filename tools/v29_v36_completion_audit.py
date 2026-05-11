from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def status(path: str) -> Any:
    data = read_json(ROOT / path)
    return data.get("status") or data.get("final_status") or data.get("_error")


def add(rows: list[dict[str, Any]], stage: str, item: str, ok: bool, evidence: str = "") -> None:
    rows.append({"stage": stage, "item": item, "ok": bool(ok), "evidence": evidence})


def main() -> int:
    rows: list[dict[str, Any]] = []
    required = {
        "V29": [
            "tools/v29_teacher_normal_route_rescue.py",
            "tools/v29_temporal_normal_accumulator.py",
            "tools/v29_candidate_geometric_normal_builder.py",
            "tools/v29_normal_gate_audit.py",
            "reports/20260508_v29_normal_route_rescue.md",
            "reports/20260508_v29_normal_route_rescue.json",
            "output/surface_research_preflight_local/V29_normal_route_rescue/v29_teacher_normals_world.npz",
            "output/surface_research_preflight_local/V29_normal_route_rescue/v29_temporal_normals_world.npz",
            "output/surface_research_preflight_local/V29_normal_route_rescue/v29_candidate_geometric_normals.npz",
        ],
        "V30": [
            "modal_v30_prior_enabled_vggt_predictions.py",
            "tools/v30_prior_enabled_prediction_intake.py",
            "tools/v30_prior_channel_verifier.py",
            "tools/v30_prediction_control_audit.py",
            "reports/20260508_v30_prior_enabled_vggt_predictions.md",
            "reports/20260508_v30_prior_enabled_vggt_predictions.json",
        ],
        "V31": [
            "modal_v31_teacher_supervised_vggt_candidate_train.py",
            "training/config/4k4d_smplx_residual_teacher_candidate_v31.yaml",
            "tools/v31_training_curve_audit.py",
            "tools/v31_checkpoint_safety_scanner.py",
            "reports/20260508_v31_teacher_supervised_candidate_train.md",
            "reports/20260508_v31_teacher_supervised_candidate_train.json",
            "output/surface_research_preflight_local/V31_teacher_supervised_candidate_train/v31_candidate_research_checkpoint.npz",
        ],
        "V32": [
            "modal_v32_candidate_inference_research.py",
            "tools/v32_candidate_region_audit.py",
            "tools/v32_candidate_open3d_review.py",
            "tools/v32_candidate_normal_depth_audit.py",
            "reports/20260508_v32_candidate_inference_region_audit.md",
            "reports/20260508_v32_candidate_inference_region_audit.json",
            "output/surface_research_preflight_local/V32_candidate_inference_research/candidate_depths_research.npz",
            "output/surface_research_preflight_local/V32_candidate_inference_research/candidate_points_world_research.npz",
            "output/surface_research_preflight_local/V32_candidate_inference_research/candidate_normals_geometric_research.npz",
            "output/surface_research_preflight_local/V32_candidate_inference_research/candidate_visibility_research.npz",
        ],
        "V33": [
            "tools/v33_head_face_region_teacher_refiner.py",
            "tools/v33_face_normal_relief_audit.py",
            "tools/v33_candidate_face_patch_merge.py",
            "reports/20260508_v33_head_face_detail_route.md",
            "reports/20260508_v33_head_face_detail_route.json",
        ],
        "V34": [
            "tools/v34_smplx_native_hand_continuity_refiner.py",
            "tools/v34_hand_region_audit.py",
            "tools/v34_hand_wrist_bridge_merge.py",
            "reports/20260508_v34_smplx_native_hand_route.md",
            "reports/20260508_v34_smplx_native_hand_route.json",
        ],
        "V35": [
            "tools/v35_60view_smplx_prior_reraster.py",
            "tools/v35_60view_teacher_candidate_support_audit.py",
            "reports/20260508_v35_60view_support_expansion.md",
            "reports/20260508_v35_60view_support_expansion.json",
        ],
        "V36": [
            "tools/v36_final_strict_promotion_transaction.py",
            "tools/v36_forbidden_output_scanner.py",
            "tools/v36_final_mentor_packet_builder.py",
            "reports/20260508_v36_final_promotion_report.md",
            "reports/20260508_v36_final_promotion_report.json",
        ],
    }
    for stage, paths in required.items():
        for p in paths:
            add(rows, stage, p, (ROOT / p).exists())

    expected_status = {
        "V29": ("reports/20260508_v29_normal_route_rescue.json", {"DONE_PASS"}),
        "V30": ("reports/20260508_v30_prior_enabled_vggt_predictions.json", {"DONE_PASS", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"}),
        "V31": ("reports/20260508_v31_teacher_supervised_candidate_train.json", {"DONE_PASS"}),
        "V32": ("reports/20260508_v32_candidate_inference_region_audit.json", {"DONE_PASS"}),
        "V33": ("reports/20260508_v33_head_face_detail_route.json", {"DONE_PASS"}),
        "V34": ("reports/20260508_v34_smplx_native_hand_route.json", {"DONE_PASS"}),
        "V35": ("reports/20260508_v35_60view_support_expansion.json", {"DONE_PASS"}),
        "V36": ("reports/20260508_v36_final_promotion_report.json", {"DONE_PASS", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"}),
    }
    for stage, (path, allowed) in expected_status.items():
        st = status(path)
        add(rows, stage, f"{path} status in {sorted(allowed)}", st in allowed, str(st))

    v30 = read_json(ROOT / "reports/20260508_v30_prior_enabled_vggt_predictions.json")
    add(rows, "V30", "code prior route verified", bool((v30.get("verifier") or {}).get("code_supports_prior_adapter")), "")
    add(rows, "V30", "hard-impossible has checkpoint blocker evidence", "No usable prior-enabled VGGT checkpoint" in json.dumps(v30, ensure_ascii=False), "")
    v31 = read_json(ROOT / "reports/20260508_v31_teacher_supervised_candidate_train.json")
    add(rows, "V31", "checkpoint_exists true", bool(v31.get("checkpoint_exists")), str(v31.get("checkpoint_path")))
    v32 = read_json(ROOT / "reports/20260508_v32_candidate_inference_region_audit.json")
    add(rows, "V32", "candidate artifacts exist", bool(v32.get("candidate_depths") and v32.get("candidate_points_world") and v32.get("candidate_normals_geometric")), "")
    v36 = read_json(ROOT / "reports/20260508_v36_final_promotion_report.json")
    add(rows, "V36", "all routes executed", bool(v36.get("all_routes_executed")), str(v36.get("all_routes_executed")))
    add(rows, "V36", "no strict writes unless pass", (v36.get("strict_candidate_passes", 0) > 0) == bool(v36.get("writes_package")), str({k: v36.get(k) for k in ("strict_candidate_passes", "writes_package", "writes_strict_registry")}))

    # Current-run forbidden scan should be clean.
    scan = read_json(ROOT / "reports/20260508_v36_forbidden_output_scan.json")
    add(rows, "V36", "forbidden scan clean", scan.get("hit_count") == 0, str(scan.get("hit_count")))

    failed = [row for row in rows if not row["ok"]]
    out = {
        "task": "v29_v36_completion_audit",
        "status": "COMPLETE_AUDIT_PASS" if not failed else "COMPLETE_AUDIT_FAIL",
        "total_checks": len(rows),
        "failed_checks": len(failed),
        "failed": failed,
        "checks": rows,
    }
    write_json(REPORTS / "20260508_v29_v36_completion_audit.json", out)
    lines = [
        "# V29-V36 Completion Audit",
        "",
        f"Status: `{out['status']}`",
        f"Total checks: `{out['total_checks']}`",
        f"Failed checks: `{out['failed_checks']}`",
        "",
        "## Failed Checks",
    ]
    lines.extend([f"- {row['stage']} {row['item']}: {row['evidence']}" for row in failed] or ["- none"])
    lines.extend(["", "## Stage Summary"])
    for stage in required:
        stage_rows = [row for row in rows if row["stage"] == stage]
        lines.append(f"- {stage}: `{sum(row['ok'] for row in stage_rows)}/{len(stage_rows)}` checks passed")
    (REPORTS / "20260508_v29_v36_completion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "total_checks": len(rows), "failed_checks": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
