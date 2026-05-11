from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def report_status(path: Path) -> Any:
    data = read_json(ROOT / path)
    return data.get("status") or data.get("final_status") or data.get("_error")


def add(checks: list[dict[str, Any]], stage: str, item: str, ok: bool, evidence: str = "") -> None:
    checks.append({"stage": stage, "item": item, "ok": bool(ok), "evidence": evidence})


def main() -> int:
    checks: list[dict[str, Any]] = []

    # V21
    for f in [
        "tools/v21_completion_contract.py",
        "tools/v21_dline_after_router.py",
        "reports/20260508_v21_completion_contract.md",
        "reports/20260508_v21_dline_after_router.md",
        "reports/20260508_v21_completion_contract.json",
        "reports/20260508_v21_dline_after_router.json",
    ]:
        add(checks, "V21", f, (ROOT / f).exists())
    add(checks, "V21", "completion status routed", report_status(Path("reports/20260508_v21_completion_contract.json")) == "DONE_FAIL_ROUTED", str(report_status(Path("reports/20260508_v21_completion_contract.json"))))
    add(checks, "V21", "after-router status routed", report_status(Path("reports/20260508_v21_dline_after_router.json")) == "DONE_FAIL_ROUTED", str(report_status(Path("reports/20260508_v21_dline_after_router.json"))))

    # V22
    for f in [
        "modal_v22_true_vggt_smplx_microfit.py",
        "tools/v22_microfit_result_auditor.py",
        "reports/20260508_v22_true_vggt_smplx_microfit.md",
        "reports/20260508_v22_true_vggt_smplx_microfit.json",
    ]:
        add(checks, "V22", f, (ROOT / f).exists())
    v22 = read_json(ROOT / "reports/20260508_v22_true_vggt_smplx_microfit.json")
    add(checks, "V22", "final_status DONE_PASS", v22.get("final_status") == "DONE_PASS", str(v22.get("final_status")))
    metric_root = ROOT / "output/surface_research_preflight_local/V22_true_vggt_smplx_microfit"
    metric_files = list(metric_root.glob("*.metrics.json"))
    add(checks, "V22", "30 metric cells exist", len(metric_files) >= 30, str(len(metric_files)))
    missing = []
    for method in ["M2", "M3"]:
        for viewset in ["existing6", "hand_head6", "balanced12"]:
            for control in ["real", "zero", "shuffle", "random-region", "prior-dropout"]:
                if not (metric_root / f"{method}_{viewset}_{control}.metrics.json").exists():
                    missing.append(f"{method}_{viewset}_{control}")
    add(checks, "V22", "M2/M3 x 3 viewsets x 5 controls complete", not missing, ",".join(missing[:5]) if missing else "complete")

    # V23
    for f in [
        "tools/v23_residual_evidence_mask_repair.py",
        "tools/v23_smplx_residual_surface_optimizer_v2.py",
        "tools/v23_residual_region_audit.py",
        "reports/20260508_v23_residual_surface_v2.md",
        "reports/20260508_v23_residual_surface_v2.json",
        "output/surface_research_preflight_local/V23_residual_surface_v2/v23_residual_surface_v2_points.npz",
    ]:
        add(checks, "V23", f, (ROOT / f).exists())
    v23 = read_json(ROOT / "reports/20260508_v23_residual_surface_v2.json")
    regions = v23.get("metrics", {}).get("region_metrics", {}) or v23.get("region_coverage", {})
    for region in ["body", "head", "face", "left_hand", "right_hand"]:
        row = regions.get(region, {})
        pixels = row.get("pixels") or row.get("sample_points") or 0
        add(checks, "V23", f"{region} nonempty", pixels > 0, str(pixels))
    add(checks, "V23", "status DONE_PASS", report_status(Path("reports/20260508_v23_residual_surface_v2.json")) == "DONE_PASS", str(report_status(Path("reports/20260508_v23_residual_surface_v2.json"))))

    # V24
    for f in [
        "tools/v24_residual_teacher_distillation_case_v2.py",
        "tools/v24_teacher_reprojection_audit.py",
        "tools/v24_teacher_visual_review_pack.py",
        "reports/20260508_v24_residual_teacher_v2.md",
        "reports/20260508_v24_residual_teacher_v2.json",
        "reports/20260508_v24_teacher_reprojection_audit.json",
        "reports/20260508_v24_teacher_visual_review_pack.json",
        "output/surface_research_preflight_local/V24_residual_teacher_v2/v24_residual_teacher_targets_v2.npz",
        "output/surface_research_preflight_local/V24_residual_teacher_v2/v24_teacher_region_contact_sheet.png",
    ]:
        add(checks, "V24", f, (ROOT / f).exists())
    v24 = read_json(ROOT / "reports/20260508_v24_residual_teacher_v2.json")
    add(checks, "V24", "status DONE_PASS", report_status(Path("reports/20260508_v24_residual_teacher_v2.json")) == "DONE_PASS", str(report_status(Path("reports/20260508_v24_residual_teacher_v2.json"))))
    for region, row in v24.get("region_coverage", {}).items():
        add(checks, "V24", f"{region} teacher pixels nonempty", row.get("pixels", 0) > 0, str(row.get("pixels")))
    v24a = read_json(ROOT / "reports/20260508_v24_teacher_reprojection_audit.json")
    support_keys = set(v24a.get("support_audits", {}).keys())
    for key in ["6v", "12v", "60v"]:
        add(checks, "V24", f"{key} audit present", key in support_keys, str(sorted(support_keys)))

    # V25
    for f in [
        "modal_v25_research_vggt_predictions_3frames.py",
        "tools/v25_research_prediction_intake.py",
        "tools/v25_prediction_safety_scanner.py",
        "reports/20260508_v25_research_predictions_3frames.md",
        "reports/20260508_v25_research_predictions_3frames.json",
    ]:
        add(checks, "V25", f, (ROOT / f).exists())
    v25root = ROOT / "output/surface_research_cloud_preflight/V25_research_vggt_predictions"
    for f in ["research_depths.npz", "research_points_world.npz", "research_normals.npz", "research_confidence.npz", "research_summary.json"]:
        p = v25root / f
        add(checks, "V25", f"output {f}", p.exists(), str(p.stat().st_size if p.exists() else 0))
    frame_keys = ["frame0000", "frame0001", "frame0002"]
    expected_keys = {
        "research_depths.npz": frame_keys,
        "research_points_world.npz": frame_keys,
        "research_confidence.npz": [f"{frame}_depth_conf" for frame in frame_keys] + [f"{frame}_world_points_conf" for frame in frame_keys],
    }
    for file_name, keys in expected_keys.items():
        p = v25root / file_name
        ok = False
        evidence = "missing"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                ok = all(key in z.files for key in keys)
                evidence = str(list(z.files))
        add(checks, "V25", f"{file_name} has required frame arrays", ok, evidence[:240])
    add(checks, "V25", "no predictions.npz in V25 root", not list(v25root.rglob("predictions.npz")))

    # V26
    for f in [
        "tools/v26_temporal_canonical_smplx_residual_teacher.py",
        "tools/v26_temporal_consistency_audit.py",
        "reports/20260508_v26_temporal_canonical_teacher.md",
        "reports/20260508_v26_temporal_canonical_teacher.json",
        "reports/20260508_v26_temporal_consistency_audit.json",
        "output/surface_research_preflight_local/V26_temporal_canonical_teacher/v26_temporal_canonical_teacher_targets.npz",
    ]:
        add(checks, "V26", f, (ROOT / f).exists())
    v26 = read_json(ROOT / "reports/20260508_v26_temporal_canonical_teacher.json")
    add(checks, "V26", "status DONE_PASS", report_status(Path("reports/20260508_v26_temporal_canonical_teacher.json")) == "DONE_PASS", str(report_status(Path("reports/20260508_v26_temporal_canonical_teacher.json"))))
    for region, row in v26.get("region_support", {}).items():
        add(checks, "V26", f"{region} temporal support nonempty", row.get("canonical_support", 0) > 0, str(row.get("canonical_support")))
    add(checks, "V26", "3 frame keys", v26.get("metrics", {}).get("frame_keys") == frame_keys, str(v26.get("metrics", {}).get("frame_keys")))

    # V27
    for f in [
        "modal_v27_teacher_supervised_vggt_research.py",
        "training/config/4k4d_smplx_residual_teacher_research.yaml",
        "tools/v27_teacher_supervised_audit.py",
        "reports/20260508_v27_teacher_supervised_training.md",
        "reports/20260508_v27_teacher_supervised_training.json",
    ]:
        add(checks, "V27", f, (ROOT / f).exists())
    v27 = read_json(ROOT / "reports/20260508_v27_teacher_supervised_training.json")
    add(checks, "V27", "status DONE_PASS", report_status(Path("reports/20260508_v27_teacher_supervised_training.json")) == "DONE_PASS", str(report_status(Path("reports/20260508_v27_teacher_supervised_training.json"))))
    add(checks, "V27", "research_positive true", bool(v27.get("research_positive")), str(v27.get("research_positive")))
    add(checks, "V27", "positive V22 cell exists", bool(v27.get("v22_positive_cells")), str(list(v27.get("v22_positive_cells", {}).keys())))

    # V28
    for f in [
        "tools/v28_strict_teacher_candidate_gate.py",
        "tools/v28_final_package_builder.py",
        "reports/20260508_v28_final_mentor_report.md",
        "reports/20260508_v28_strict_teacher_candidate_gate.json",
        "reports/20260508_v28_final_package_builder.json",
        "reports/20260508_v28_final_mentor_report.json",
    ]:
        add(checks, "V28", f, (ROOT / f).exists())
    v28 = read_json(ROOT / "reports/20260508_v28_strict_teacher_candidate_gate.json")
    add(checks, "V28", "all required branches executed", bool(v28.get("all_required_branches_executed")), str(v28.get("all_required_branches_executed")))
    add(checks, "V28", "failure proof complete", bool(v28.get("failure_proof_complete")), str(v28.get("failure_proof_complete")))
    add(checks, "V28", "forbidden scan hit_count zero", v28.get("forbidden_scan", {}).get("hit_count") == 0, str(v28.get("forbidden_scan", {}).get("hit_count")))
    add(
        checks,
        "V28",
        "strict passes remain zero without package write",
        v28.get("strict_teacher_passes") == 0 and v28.get("strict_candidate_passes") == 0 and not v28.get("writes_package") and not v28.get("writes_strict_registry"),
        str({k: v28.get(k) for k in ["strict_teacher_passes", "strict_candidate_passes", "writes_package", "writes_strict_registry"]}),
    )

    failed = [c for c in checks if not c["ok"]]
    out = {
        "task": "v21_v28_completion_audit",
        "status": "COMPLETE_AUDIT_PASS" if not failed else "COMPLETE_AUDIT_FAIL",
        "total_checks": len(checks),
        "failed_checks": len(failed),
        "failed": failed,
        "checks": checks,
    }
    write_json(REPORTS / "20260508_v21_v28_completion_audit.json", out)
    md = [
        "# V21-V28 Completion Audit",
        "",
        f"Status: `{out['status']}`",
        f"Total checks: `{len(checks)}`",
        f"Failed checks: `{len(failed)}`",
        "",
        "## Failed Checks",
    ]
    md.extend([f"- {c['stage']} {c['item']}: {c['evidence']}" for c in failed] or ["- none"])
    md.extend(["", "## Stage Summary"])
    for stage in ["V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]:
        stage_checks = [c for c in checks if c["stage"] == stage]
        md.append(f"- {stage}: `{sum(c['ok'] for c in stage_checks)}/{len(stage_checks)}` checks passed")
    (REPORTS / "20260508_v21_v28_completion_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "total_checks": len(checks), "failed_checks": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
