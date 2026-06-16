from __future__ import annotations

import csv
import hashlib
import json
import subprocess
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
ARCHIVE = REPO / "archive"
VIEWER = REPO / "viewer"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]
TRUE_STATUS = "V1800000000000000_REAL_VGGT_SMPL_FEATURE_DETAIL_MENTOR_READY_NOT_PROMOTED"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    return {"args": args, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def check_file(path: Path, *, min_bytes: int = 1) -> dict[str, Any]:
    return {"path": str(path.relative_to(REPO)), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0, "pass": path.exists() and path.stat().st_size >= min_bytes}


def check_png(path: Path) -> dict[str, Any]:
    row = check_file(path, min_bytes=1000)
    if row["pass"]:
        with Image.open(path) as im:
            row |= {"width": im.width, "height": im.height, "mode": im.mode, "png_readable": True}
            row["pass"] = im.width >= 200 and im.height >= 200
    return row


def check_npz(path: Path, required: list[str] | None = None) -> dict[str, Any]:
    row = check_file(path, min_bytes=100)
    if row["pass"]:
        with np.load(path, allow_pickle=True) as z:
            keys = list(z.files)
            missing = [k for k in (required or []) if k not in keys]
            row |= {"keys": keys[:20], "key_count": len(keys), "missing_required": missing, "npz_readable": True}
            row["pass"] = not missing
    return row


def check_ply(path: Path) -> dict[str, Any]:
    row = check_file(path, min_bytes=100)
    if row["pass"]:
        header = []
        with path.open("rb") as f:
            for _ in range(20):
                line = f.readline().decode("ascii", errors="replace").strip()
                header.append(line)
                if line == "end_header":
                    break
        row |= {"header": header, "ply_readable": header[:2] == ["ply", "format ascii 1.0"], "has_rgb": any("red" in h for h in header) and any("green" in h for h in header) and any("blue" in h for h in header)}
        row["pass"] = row["ply_readable"] and row["has_rgb"]
    return row


def main() -> None:
    rows: list[dict[str, Any]] = []

    goal = REPO / "docs/goals/V900100000000000_V1800000000000000_real_vggt_smpl_feature_detail_goal.md"
    manifest = read_json(REPORTS / "V900100000000000_goal_file_manifest.json")
    goal_hash = sha256(goal)
    goal_line_count = len(goal.read_text(encoding="utf-8").splitlines())
    rows.append({"requirement": "goal_file_manifest_matches_written_goal", "pass": goal_hash == manifest["sha256"] and goal_line_count == manifest["line_count"], "evidence": str(goal.relative_to(REPO)), "details": {"sha256": goal_hash, "line_count": goal_line_count}})

    final_status = read_json(REPORTS / "V1800000000000000_final_status.json")
    rows.append({"requirement": "allowed_final_status_reached", "pass": final_status.get("status") == TRUE_STATUS and final_status.get("all_pass") is True, "evidence": "reports/V1800000000000000_final_status.json", "details": final_status})

    for rel in [
        "reports/V900100000000000_v900_checkpoint_freeze.json",
        "reports/V900100000000000_why_v900_is_not_final.md",
        "reports/V900100000000000_target_drift_risk.md",
        "reports/V910000000000000_current_artifact_index.csv",
        "reports/V910000000000000_current_evidence_decision.json",
        "reports/V920000000000000_real_vggt_path_inventory.json",
        "reports/V920000000000000_real_vggt_adapter_feasibility.md",
        "reports/V920000000000000_required_patch_list.md",
    ]:
        rows.append({"requirement": f"checkpoint_or_audit_artifact_exists:{rel}", **check_file(REPO / rel)})

    v930 = read_json(REPORTS / "V930000000000000_token_shape_audit.json")
    rows.append({"requirement": "real_vggt_or_aggregator_tokens_used", "pass": bool(v930.get("decision", {}).get("pass")), "evidence": "reports/V930000000000000_token_shape_audit.json", "details": v930.get("decision")})
    for case in CASES:
        rows.append({"requirement": f"real_token_npz_readable:{case}", **check_npz(OUTPUT / "V930000000000000_real_vggt_tokens" / case / "real_vggt_tokens_and_predictions.npz", ["real_aggregator_tokens_with_smpl_prior_last", "real_aggregator_token_delta_last", "world_points", "input_images"])})
        rows.append({"requirement": f"smpl_feature_bank_npz_readable:{case}", **check_npz(OUTPUT / "V940000000000000_smpl_feature_bank" / case / "smpl_feature_bank.npz", ["world_points", "smpl_feature_image", "skinning_weights", "graph_knn_ring", "voxel_coords_32"])})

    v950 = read_json(REPORTS / "V950000000000000_forward_gradient_smoke.json")
    v950_decision = v950.get("decision", {})
    rows.append({"requirement": "smpl_feature_binding_gradient_on_real_tokens", "pass": all(bool(v950_decision.get(k)) for k in ["pass", "real_vggt_tokens_affect_output", "smpl_feature_binding_affects_output", "binding_delta_nonzero", "teacher_raw_kinect_rejected"]) and not v950_decision.get("tiny_v330_or_synthetic_tokens_used"), "evidence": "reports/V950000000000000_forward_gradient_smoke.json", "details": v950_decision})
    rows.append({"requirement": "v950_model_file_exists", **check_file(REPO / "models/v950_real_vggt_smpl_feature_adapter.py", min_bytes=1000)})

    v960_manifest = read_json(REPORTS / "V960000000000000_training_manifest.json")
    metric_rows = list(csv.DictReader((REPORTS / "V960000000000000_seed_metrics.csv").open("r", encoding="utf-8", newline="")))
    rows.append({"requirement": "modal_real_vggt_matrix_completed", "pass": v960_manifest.get("rows") == 48 and not v960_manifest.get("failures") and len(metric_rows) == 48, "evidence": "reports/V960000000000000_training_manifest.json", "details": {"rows": v960_manifest.get("rows"), "failures": v960_manifest.get("failures")}})
    for case in CASES:
        for cfg in ["real_vggt_smpl_feature_true_full", "real_vggt_baseline_only", "posthoc_surfel_only", "tiny_v330_synthetic_token_control"]:
            rows.append({"requirement": f"v960_prediction_npz_readable:{case}:{cfg}", **check_npz(OUTPUT / "V960000000000000_real_vggt_matrix" / case / f"{cfg}_seed0" / "predictions.npz", ["student_points", "active_mask", "rgb", "full_scene_points", "full_scene_rgb"])})
            rows.append({"requirement": f"v960_full_scene_ply_readable:{case}:{cfg}", **check_ply(OUTPUT / "V960000000000000_real_vggt_matrix" / case / f"{cfg}_seed0" / "full_scene_rgb.ply")})

    gate = read_json(REPORTS / "V1100000000000000_final_mentor_gate.json")
    gate_pass = all(v for k, v in gate.items() if k not in {"created_at", "all_pass"}) and gate.get("all_pass") is True
    rows.append({"requirement": "mentor_final_gate_all_items_pass", "pass": gate_pass, "evidence": "reports/V1100000000000000_final_mentor_gate.json", "details": gate})
    for rel in [
        "boards/V970000000000000_real_vggt_advisor_main_board.png",
        "boards/V970000000000000_same_scene_controls_board.png",
        "boards/V970000000000000_cloudcompare_style_board.png",
        "boards/V980000000000000_head_face_hair_detail_board.png",
        "boards/V980000000000000_hand_arm_detail_board.png",
        "boards/V980000000000000_clothing_boundary_board.png",
        "boards/V990000000000000_control_firewall_v2_board.png",
    ]:
        rows.append({"requirement": f"visual_board_png_readable:{rel}", **check_png(REPO / rel)})

    control_rows = list(csv.DictReader((REPORTS / "V990000000000000_control_firewall_v2.csv").open("r", encoding="utf-8", newline="")))
    controls_pass = len(control_rows) == 4 and all(r["controls_weaker"].lower() == "true" and r["posthoc_surfel_beaten"].lower() == "true" and r["tiny_synthetic_beaten"].lower() == "true" for r in control_rows)
    rows.append({"requirement": "true_beats_baseline_controls_posthoc_synthetic", "pass": controls_pass, "evidence": "reports/V990000000000000_control_firewall_v2.csv", "details": control_rows})

    detail = read_json(REPORTS / "V980000000000000_local_detail_decision.json")
    rows.append({"requirement": "local_detail_non_regression_and_improvement_gate", "pass": bool(detail.get("decision", {}).get("non_regression_pass") and detail.get("decision", {}).get("at_least_one_local_improvement_pass")), "evidence": "reports/V980000000000000_local_detail_decision.json", "details": detail.get("decision")})

    for rel in [
        "reports/V1400000000000000_real_vggt_smpl_feature_advisor_report.md",
        "reports/V1400000000000000_one_page.md",
        "reports/V1400000000000000_limitations.md",
        "viewer/V1600000000000000_real_vggt_smpl_feature_viewer.html",
        "reports/V1800000000000000_migration_handoff.md",
    ]:
        rows.append({"requirement": f"report_or_viewer_exists:{rel}", **check_file(REPO / rel, min_bytes=100)})

    bundle_integrity = read_json(REPORTS / "V1600000000000000_bundle_integrity.json")
    bundle_pass = bundle_integrity.get("all_zip_clean") and bundle_integrity.get("all_under_500mb") and bundle_integrity.get("all_non_empty")
    for b in bundle_integrity.get("bundles", []):
        zp = REPO / b["path"]
        ok = False
        if zp.exists():
            with zipfile.ZipFile(zp, "r") as zf:
                ok = zf.testzip() is None and len(zf.namelist()) == int(b["entry_count"])
        rows.append({"requirement": f"bundle_verified:{b['bundle']}", "pass": bool(ok and b["zip_clean"] and b["under_500mb"] and b["non_empty"]), "evidence": b["path"], "details": b})
    rows.append({"requirement": "bundle_integrity_summary_pass", "pass": bool(bundle_pass), "evidence": "reports/V1600000000000000_bundle_integrity.json", "details": {k: bundle_integrity.get(k) for k in ["all_zip_clean", "all_under_500mb", "all_non_empty"]}})

    cleanup = read_json(REPORTS / "V1700000000000000_post_push_cleanup.json")
    v50_diff = run_cmd(["git", "diff", "--name-only", "--", "V50", "V50R2", "reports/V50", "reports/V50R2"])
    promotion_registry_diff = run_cmd(["powershell", "-NoProfile", "-Command", "git diff --name-only -- . | Select-String -Pattern 'registry|promotion|V50|V50R2'"])
    rows.append({"requirement": "no_agent_no_promotion_no_registry_active_candidate_v50", "pass": final_status.get("no_agent_subagent") and final_status.get("no_promotion") and final_status.get("no_registry") and final_status.get("active_candidate") == "V11700_gap_reduction_branch_520" and v50_diff["stdout"] == "" and promotion_registry_diff["stdout"] == "", "evidence": "reports/V1700000000000000_post_push_cleanup.json + git diff", "details": {"cleanup_flags": {k: cleanup.get(k) for k in ["agents_or_subagents_launched", "promotion_performed", "registry_modified", "v50_v50r2_modified_by_this_script"]}, "v50_diff": v50_diff, "promotion_registry_diff": promotion_registry_diff}})

    failed = [r for r in rows if not r.get("pass")]
    payload = {
        "created_at": now(),
        "status": TRUE_STATUS if not failed else "INCOMPLETE",
        "all_requirements_proven": not failed,
        "requirement_count": len(rows),
        "failed_count": len(failed),
        "failed_requirements": failed,
        "rows": rows,
    }
    write_json(REPORTS / "V1800000000000000_completion_audit.json", payload)
    print(json.dumps({"all_requirements_proven": not failed, "requirement_count": len(rows), "failed_count": len(failed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
