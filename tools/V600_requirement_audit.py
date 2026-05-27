from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
GOAL = REPO / "docs" / "goals" / "V20010000000000000_V60000000000000000_high_density_detail_fidelity_goal.md"
MANIFEST = REPORTS / "V20010000000000000_goal_file_manifest.json"
FINAL = "V60000000000000000_HIGH_DENSITY_DETAIL_FIDELITY_MENTOR_READY_NOT_PROMOTED"
HARD_BLOCK = "V60000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def npz_ok(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            return bool(z.files)
    except Exception:
        return False


def ply_ok(path: Path) -> bool:
    try:
        head = path.read_bytes()[:512]
        return head.startswith(b"ply") and b"property uchar red" in head
    except Exception:
        return False


def main() -> None:
    manifest = read_json(MANIFEST, {})
    goal_sha = sha256_file(GOAL)
    goal_lines = len(GOAL.read_text(encoding="utf-8").splitlines())
    v210 = read_json(REPORTS / "V21000000000000000_current_evidence_decision.json", {})
    v220 = read_json(REPORTS / "V22000000000000000_full_forward_effect_decision.json", {})
    v230 = read_json(REPORTS / "V23000000000000000_per_case_full_forward_decision.json", {})
    v240 = read_json(REPORTS / "V24000000000000000_point_budget_plan.json", {})
    v310 = read_json(REPORTS / "V31000000000000000_local_detail_decision_v3.json", {})
    v330 = read_json(REPORTS / "V33000000000000000_claim_boundary_v5.json", {})
    v340 = read_json(REPORTS / "V34000000000000000_visual_judge_proxy.json", {})
    v350 = read_json(REPORTS / "V35000000000000000_multisequence_high_density_gate.json", {})
    v400 = read_json(REPORTS / "V40000000000000000_final_high_density_mentor_gate.json", {})
    v550 = read_json(REPORTS / "V55000000000000000_bundle_integrity.json", {})
    v580 = read_json(REPORTS / "V58000000000000000_post_push_cleanup.json", {})
    final = read_json(REPORTS / "V60000000000000000_final_status.json", {})
    density_rows = csv_rows(REPORTS / "V25000000000000000_density_quality_metrics.csv")
    true_rows = [r for r in density_rows if r["config"] == "high_density_smpl_true"]
    bundle_ok = []
    for row in v550.get("bundles", []):
        path = REPO / row["path"]
        try:
            with zipfile.ZipFile(path, "r") as zf:
                clean = zf.testzip() is None
                entries = zf.namelist()
            bundle_ok.append(clean and bool(entries) and path.stat().st_size < 500 * 1024 * 1024)
        except Exception:
            bundle_ok.append(False)
    sample_pred = OUTPUT / "V25000000000000000_high_density_predictions" / "current_v895_0021_03" / "high_density_smpl_true" / "predictions.npz"
    sample_ply = OUTPUT / "V25000000000000000_high_density_predictions" / "current_v895_0021_03" / "high_density_smpl_true" / "full_scene_rgb_pointcloud.ply"
    checks = {
        "goal_file_exists": GOAL.exists(),
        "manifest_sha_matches": manifest.get("sha256") == goal_sha,
        "manifest_line_count_matches": manifest.get("line_count") == goal_lines,
        "no_agent_rule_present": bool(manifest.get("no_agent_rule")),
        "v200_downgraded": (REPORTS / "V20010000000000000_v200_checkpoint_freeze.json").exists(),
        "current_artifact_audit_pass": bool(v210.get("artifact_audit_pass")),
        "per_case_full_forward_effect_pass": bool(v220.get("per_case_full_forward_effect_pass") and v230.get("per_case_full_forward_effect_pass")),
        "effect_not_single_smoke_reuse": bool(v220.get("effects_case_specific")),
        "high_density_point_budget_pass": bool(v240.get("high_density_budget_pass") and len(true_rows) == 4 and all(int(r["human_points"]) >= 30000 for r in true_rows)),
        "true_preferred_60k_pass": all(int(r["human_points"]) >= 60000 for r in true_rows),
        "environment_points_pass": all(int(r["environment_points"]) >= 8000 for r in true_rows),
        "sample_npz_readable": npz_ok(sample_pred),
        "sample_ply_readable": ply_ok(sample_ply),
        "v300_main_board_readable": image_ok(BOARDS / "V30000000000000000_advisor_high_density_main.png"),
        "local_detail_non_regression_4_of_4": v310.get("non_regression_count") == 4,
        "local_visible_improvement_3_of_4": v310.get("actual_visible_improvement_count", 0) >= 3,
        "head_hair_improvement_2_of_4": v310.get("head_hair_improvement_count", 0) >= 2,
        "hand_arm_improvement_2_of_4": v310.get("hand_arm_improvement_count", 0) >= 2,
        "clothing_improvement_2_of_4": v310.get("clothing_improvement_count", 0) >= 2,
        "active_count_only_rejected": bool(v310.get("active_count_only_rejected")),
        "no_facial_detail_overclaim": bool(v310.get("facial_detail_overclaim_forbidden")),
        "hard_controls_pass": bool(v330.get("hard_controls_pass")),
        "visual_judge_proxy_pass": bool(v340.get("visual_judge_proxy_pass")),
        "multisequence_gate_pass": bool(v350.get("at_least_four_cases_retained") and v350.get("at_least_three_strong_visual_pass") and v350.get("at_least_three_local_visible_improvement")),
        "final_mentor_gate_pass": bool(v400.get("all_pass")),
        "advisor_report_exists": (REPORTS / "V50000000000000000_high_density_detail_advisor_report.md").exists(),
        "bundles_all_clean": bool(v550.get("all_zip_clean") and v550.get("all_under_500mb") and v550.get("all_non_empty") and all(bundle_ok)),
        "cleanup_no_agent": bool(v580.get("no_agent_subagent")),
        "final_status_allowed": final.get("status") in {FINAL, HARD_BLOCK},
        "final_status_ready": final.get("status") == FINAL and final.get("all_pass") is True,
        "no_promotion": bool(final.get("checks", {}).get("no_promotion")),
        "no_registry": bool(final.get("checks", {}).get("no_registry")),
        "no_v50_v50r2_change": bool(final.get("checks", {}).get("no_v50_v50r2_change")),
        "active_candidate_unchanged": bool(final.get("checks", {}).get("active_candidate_unchanged")),
    }
    failures = [k for k, v in checks.items() if not v]
    payload = {
        "created_at": now(),
        "all_ok": not failures,
        "error_count": len(failures),
        "failures": failures,
        "checks": checks,
        "goal_sha256": goal_sha,
        "goal_line_count": goal_lines,
        "final_status": final.get("status"),
    }
    write_json(REPORTS / "V60000000000000000_requirement_by_requirement_audit.json", payload)
    print(json.dumps({"all_ok": payload["all_ok"], "error_count": payload["error_count"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
