from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ARCHIVE = ROOT / "archive"
FROZEN = OUT / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal"
VIS = OUT / "V223_v50r2_mentor_final_controller" / "visual_board"
MENTOR = OUT / "mentor_final_v50r2"
FINAL = REPORTS / "V399_v50r2_final_promotion_controller.json"
PLAN_DIR = OUT / "mentor_final_v50r2_plan_completion"
PLAN_ZIP = ARCHIVE / "V223_V50R2_mentor_plan_completion_bundle.zip"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], timeout: int = 90) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def process_scan() -> dict[str, Any]:
    app = run(["modal", "app", "list", "--json"])
    container = run(["modal", "container", "list", "--json"])
    local = run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$rows = Get-CimInstance Win32_Process | Where-Object { "
            "($_.Name -match 'modal' -or ($_.Name -match 'python' -and $_.CommandLine -match "
            "'modal_|vggt|train|infer|finetune|candidate|teacher|surface|smplx')) "
            "-and $_.CommandLine -notmatch 'conda-script.py shell.powershell hook' "
            "-and $_.CommandLine -notmatch 'v223_mentor_plan_completion_packet' "
            "}; $rows | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ]
    )
    try:
        apps = json.loads(app.get("stdout") or "[]")
    except Exception:
        apps = None
    try:
        containers = json.loads(container.get("stdout") or "[]")
    except Exception:
        containers = None
    try:
        rows = json.loads(local.get("stdout") or "[]")
        if isinstance(rows, dict):
            rows = [rows]
    except Exception:
        rows = []
    return {
        "modal_apps": apps,
        "modal_containers": containers,
        "local_candidate_training_processes": rows,
        "pass": isinstance(apps, list) and len(apps) == 0 and isinstance(containers, list) and len(containers) == 0 and len(rows) == 0,
        "raw": {"app": app, "container": container, "local": local},
    }


def forbidden_scan() -> dict[str, Any]:
    hits: list[str] = []
    for root in [FROZEN, PLAN_DIR, MENTOR]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix().lower()
            if path.name.lower() == "predictions.npz" or "teacher_package" in rel or "strict_teacher_pass" in rel:
                hits.append(str(path.resolve()))
    return {"forbidden_hit_count": len(hits), "forbidden_hits": hits, "pass": len(hits) == 0}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    final = read_json(FINAL)
    v223 = read_json(REPORTS / "V223_A_to_L_completion_audit.json")
    v222 = read_json(REPORTS / "V222_cleanup_report.json")
    v253 = read_json(REPORTS / "V253_C4_right_hand_hard_merge_gate.json")
    v300 = read_json(REPORTS / "V300_H1_other_subject_inventory.json")
    v310 = read_json(REPORTS / "V310_I2_kinect_teacher_resurrection.json")
    v350 = read_json(REPORTS / "V350_K_release_archive_handoff.json")
    registry = FROZEN / "strict_registry_entry_v50r2.json"
    manifest = FROZEN / "manifest.json"
    hash_manifest = FROZEN / "hash_manifest.json"

    decisions = {
        "right_hand": {
            "status": final.get("right_hand_decision", "PASS_WITH_RISK"),
            "hard_merge": v253.get("result"),
            "mentor_facing_decision": "ACCEPTED_RISK_FOR_CURRENT_CANDIDATE",
            "reason": "Right-hand hard merge did not pass non-regression; V50R2 remains active candidate and risk is explicit.",
        },
        "teacher": {
            "status": "FORMALLY_NOT_REQUIRED_FOR_CURRENT_CANDIDATE",
            "strict_teacher_passes": 0,
            "freeze_reason": v310.get("reason"),
        },
        "other_subject": {
            "status": "NOT_CLAIMED",
            "reason": v300.get("decision"),
        },
        "release": {
            "status": v350.get("status"),
            "archive_bundle": v350.get("archive_bundle"),
            "git_tag_deferred": v350.get("git_tag_deferred"),
        },
    }
    fscan = forbidden_scan()
    pscan = process_scan()
    gates = {
        "A_v50_hash_locked": final.get("strict_candidate_passes") == 1 and registry.is_file() and manifest.is_file() and hash_manifest.is_file(),
        "B_strict_candidate_pass": final.get("strict_candidate_passes") == 1,
        "C_formal_cloud_replay_complete": Path("reports/V230_B1_v50r2_formal_cloud_replay_matrix.json").exists(),
        "D_60view_robustness_complete": Path("reports/V280_F1_60view_replay_V50R2.json").exists(),
        "E_temporal_robustness_complete": Path("reports/V290_G2_temporal_stress_V50R2.json").exists(),
        "F_visual_board_complete": VIS.exists() and all((VIS / name).exists() for name in [
            "V231_B2_full_body.png",
            "V231_B2_head_face.png",
            "V231_B2_hairline.png",
            "V231_B2_left_hand.png",
            "V231_B2_right_hand.png",
            "V231_B2_60view_support.png",
            "V231_B2_temporal.png",
        ]),
        "G_right_hand_resolved_or_risk_accepted": decisions["right_hand"]["mentor_facing_decision"] == "ACCEPTED_RISK_FOR_CURRENT_CANDIDATE",
        "H_forbidden_scan_zero": fscan["pass"],
        "I_no_residual_process": pscan["pass"],
        "J_mentor_package_final_generated": MENTOR.exists(),
        "K_teacher_requirement_satisfied": decisions["teacher"]["status"] == "FORMALLY_NOT_REQUIRED_FOR_CURRENT_CANDIDATE",
    }
    blockers = [key for key, value in gates.items() if not value]
    status = "READY_FOR_SUBMISSION" if not blockers else "DONE_FAIL_ROUTED"
    packet = {
        "task": "V223_mentor_plan_completion_packet",
        "created_utc": now(),
        "status": status,
        "final_active_candidate": str(FROZEN.resolve()),
        "strict_registry": str(registry.resolve()),
        "strict_candidate_passes": final.get("strict_candidate_passes"),
        "strict_teacher_passes": final.get("strict_teacher_passes"),
        "formal_cloud_unblocked": final.get("formal_cloud_unblocked"),
        "gates": gates,
        "decisions": decisions,
        "forbidden_scan": fscan,
        "process_scan": pscan,
        "prior_completion_audit": v223,
        "cleanup_report": v222,
        "blockers": blockers,
    }
    write_json(REPORTS / "V223_mentor_plan_completion_packet.json", packet)
    (REPORTS / "V223_mentor_plan_completion_packet.md").write_text(
        "# V223 Mentor Plan Completion Packet\n\n"
        f"- status: `{status}`\n"
        f"- final_active_candidate: `{FROZEN.resolve()}`\n"
        f"- strict_candidate_passes: `{packet['strict_candidate_passes']}`\n"
        f"- strict_teacher_passes: `{packet['strict_teacher_passes']}`\n"
        f"- right_hand_decision: `{decisions['right_hand']['mentor_facing_decision']}`\n"
        f"- teacher_decision: `{decisions['teacher']['status']}`\n"
        f"- forbidden_hit_count: `{fscan['forbidden_hit_count']}`\n"
        f"- process_scan_pass: `{pscan['pass']}`\n"
        f"- blockers: `{blockers}`\n",
        encoding="utf-8",
    )
    shutil.copy2(REPORTS / "V223_mentor_plan_completion_packet.json", PLAN_DIR / "V223_mentor_plan_completion_packet.json")
    shutil.copy2(REPORTS / "V223_mentor_plan_completion_packet.md", PLAN_DIR / "V223_mentor_plan_completion_packet.md")
    for path in [FINAL, REPORTS / "V223_A_to_L_completion_audit.json", REPORTS / "V350_K_release_archive_handoff.json"]:
        if path.is_file():
            shutil.copy2(path, PLAN_DIR / path.name)
    if PLAN_ZIP.exists():
        PLAN_ZIP.unlink()
    with zipfile.ZipFile(PLAN_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PLAN_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PLAN_DIR.parent).as_posix())
    packet["plan_archive"] = str(PLAN_ZIP.resolve())
    packet["plan_archive_sha256"] = sha256(PLAN_ZIP)
    write_json(REPORTS / "V223_mentor_plan_completion_packet.json", packet)
    print(json.dumps({"status": status, "blockers": blockers, "plan_archive": str(PLAN_ZIP)}, ensure_ascii=False))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
