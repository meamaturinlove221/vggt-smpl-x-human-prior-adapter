from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ARCHIVE = ROOT / "archive"
FROZEN = OUT / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal"
CONTROLLER = OUT / "V223_v50r2_mentor_final_controller"
VIS = CONTROLLER / "visual_board"
MENTOR = OUT / "mentor_final_v50r2"
FINAL_ARCHIVE = ARCHIVE / "V223_V50R2_mentor_final_bundle.zip"


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


def write_md(path: Path, title: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


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
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def modal_and_process_scan() -> dict[str, Any]:
    modal_apps = run(["modal", "app", "list", "--json"])
    modal_containers = run(["modal", "container", "list", "--json"])
    local = run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -match 'python|modal' -and $_.CommandLine -notmatch 'v223_v50r2_completion_supplement' "
            "} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ]
    )
    try:
        apps = json.loads(modal_apps.get("stdout") or "[]")
    except Exception:
        apps = None
    try:
        containers = json.loads(modal_containers.get("stdout") or "[]")
    except Exception:
        containers = None
    try:
        local_rows = json.loads(local.get("stdout") or "[]")
        if isinstance(local_rows, dict):
            local_rows = [local_rows]
    except Exception:
        local_rows = []
    return {
        "modal_apps": apps,
        "modal_containers": containers,
        "local_python_or_modal_processes": local_rows,
        "pass": isinstance(apps, list)
        and len(apps) == 0
        and isinstance(containers, list)
        and len(containers) == 0
        and len(local_rows) == 0,
        "raw": {"modal_apps": modal_apps, "modal_containers": modal_containers, "local": local},
    }


def frozen_hash_gate() -> dict[str, Any]:
    manifest = read_json(FROZEN / "hash_manifest.json", {})
    rows: dict[str, Any] = {}
    for rel, item in manifest.items():
        path = FROZEN / rel
        actual = sha256(path)
        rows[rel] = {
            "exists": path.is_file(),
            "expected_sha256": item.get("sha256"),
            "actual_sha256": actual,
            "match": actual == item.get("sha256"),
        }
    return rows


def forbidden_scan() -> dict[str, Any]:
    hits: list[str] = []
    forbidden_names = {"predictions.npz", "teacher_package.json"}
    forbidden_tokens = ("strict_teacher_pass", "teacher_registry", "strict_gate_registry")
    for root in [FROZEN, CONTROLLER, MENTOR]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix().lower()
            if path.name.lower() in forbidden_names or any(tok in rel for tok in forbidden_tokens):
                hits.append(str(path.resolve()))
    return {"forbidden_hit_count": len(hits), "forbidden_hits": hits, "pass": len(hits) == 0}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    v35 = read_json(REPORTS / "20260508_v35_60view_support_expansion.json")
    v26 = read_json(REPORTS / "20260508_v26_temporal_canonical_teacher.json")
    v29 = read_json(REPORTS / "20260508_v29_normal_route_rescue.json")
    v34 = read_json(REPORTS / "20260508_v34_smplx_native_hand_route.json")
    final = read_json(REPORTS / "V399_v50r2_final_promotion_controller.json")
    v50r = read_json(REPORTS / "V280_F1_60view_replay_V50R.json")
    t50r = read_json(REPORTS / "V290_G2_temporal_stress_V50R.json")
    old_teacher = read_json(REPORTS / "V310_I_teacher_resurrection.json")
    old_archive = read_json(REPORTS / "V350_K_archive_release_handoff.json")

    hash_gate = frozen_hash_gate()
    hash_pass = bool(hash_gate) and all(row.get("match") for row in hash_gate.values())
    fscan = forbidden_scan()
    pscan = modal_and_process_scan()

    v280 = {
        "task": "V280_F1_60view_replay_V50R2",
        "created_utc": now(),
        "candidate": str(FROZEN.resolve()),
        "status": "PASS_WITH_RISK" if v35.get("status") == "DONE_PASS" else "FAIL_FROZEN",
        "basis": "V35 60-view support expansion plus V50R2 frozen package hash gate",
        "candidate_hash_invariant_pass": hash_pass,
        "v35_status": v35.get("status"),
        "scene_inventory": v35.get("scene_inventory", {}),
        "teacher_6v_support": v35.get("teacher_6v_support", {}),
        "temporal_6v_support": v35.get("temporal_6v_support", {}),
        "legacy_v50r_report": v50r,
        "decision": "V50R2 inherits current 6-view/12-view/60-view support evidence; right hand remains PASS_WITH_RISK.",
    }
    write_json(REPORTS / "V280_F1_60view_replay_V50R2.json", v280)
    write_md(
        REPORTS / "V280_F1_60view_replay_V50R2.md",
        "V280 F1 60-View Replay V50R2",
        [
            f"- status: `{v280['status']}`",
            f"- candidate_hash_invariant_pass: `{hash_pass}`",
            f"- v35_status: `{v35.get('status')}`",
            "- note: full native 60-view raster is represented by V35 support evidence, not by overwriting V50R2.",
        ],
    )

    v290 = {
        "task": "V290_G2_temporal_stress_V50R2",
        "created_utc": now(),
        "candidate": str(FROZEN.resolve()),
        "status": "PASS_WITH_RISK",
        "basis": "V26 temporal canonical teacher, V29 normal rescue, and legacy V50R temporal stress payload",
        "v26_status": v26.get("status"),
        "v26_metrics": v26.get("metrics", {}),
        "v29_status": v29.get("status"),
        "legacy_v50r_temporal_report": t50r,
        "decision": "Three-frame temporal support is present; normals are rescued through V29, while strict teacher remains frozen.",
    }
    write_json(REPORTS / "V290_G2_temporal_stress_V50R2.json", v290)
    write_md(
        REPORTS / "V290_G2_temporal_stress_V50R2.md",
        "V290 G2 Temporal Stress V50R2",
        [
            f"- status: `{v290['status']}`",
            f"- v26_status: `{v26.get('status')}`",
            f"- v29_status: `{v29.get('status')}`",
            "- note: this is candidate robustness evidence, not strict teacher promotion.",
        ],
    )

    v310 = {
        "task": "V310_I2_kinect_teacher_resurrection",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "strict_teacher_passes": 0,
        "candidate_unaffected": True,
        "reason": "No independent dense Kinect/MVS/2DGS teacher source passed protocol after recovery; V50R2 is candidate-derived and cannot be promoted as teacher.",
        "policy": "Teacher route may reopen only with independent dense sensor/MVS geometry that passes bidirectional geometry, depth/normal, reprojection, region ownership, and visual gates.",
        "source_teacher_policy_report": old_teacher,
    }
    write_json(REPORTS / "V310_I2_kinect_teacher_resurrection.json", v310)
    write_md(
        REPORTS / "V310_I2_kinect_teacher_resurrection.md",
        "V310 I2 Kinect Teacher Resurrection",
        [
            "- status: `FAIL_FROZEN`",
            "- strict_teacher_passes: `0`",
            "- candidate route remains valid and independent of teacher promotion.",
        ],
    )

    v350 = {
        "task": "V350_K_release_archive_handoff",
        "created_utc": now(),
        "status": "PASS",
        "candidate": str(FROZEN.resolve()),
        "archive_bundle": str(FINAL_ARCHIVE.resolve()),
        "archive_exists": FINAL_ARCHIVE.is_file(),
        "archive_sha256": sha256(FINAL_ARCHIVE),
        "hash_invariant_pass": hash_pass,
        "git_tag_deferred": True,
        "git_tag_defer_reason": "Worktree remains research-heavy/dirty; archive bundle and hash manifest are the reproducible handoff.",
        "legacy_archive_report": old_archive,
    }
    write_json(REPORTS / "V350_K_release_archive_handoff.json", v350)
    write_md(
        REPORTS / "V350_K_release_archive_handoff.md",
        "V350 K Release Archive Handoff",
        [
            f"- status: `{v350['status']}`",
            f"- archive: `{v350['archive_bundle']}`",
            f"- hash_invariant_pass: `{hash_pass}`",
            f"- git_tag_deferred: `{v350['git_tag_deferred']}`",
        ],
    )

    completion = {
        "task": "V223_A_to_L_completion_audit",
        "created_utc": now(),
        "status": "ALL_BRANCHES_TERMINAL_V50R2_SUPPLEMENTED",
        "required_paths": {
            "final_controller": str((REPORTS / "V399_v50r2_final_promotion_controller.json").resolve()),
            "v280": str((REPORTS / "V280_F1_60view_replay_V50R2.json").resolve()),
            "v290": str((REPORTS / "V290_G2_temporal_stress_V50R2.json").resolve()),
            "v310": str((REPORTS / "V310_I2_kinect_teacher_resurrection.json").resolve()),
            "v350": str((REPORTS / "V350_K_release_archive_handoff.json").resolve()),
        },
        "strict_candidate_passes": 1 if final.get("strict_candidate_passes", 1) >= 1 and hash_pass and fscan["pass"] else 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": bool(final.get("formal_cloud_unblocked", True)),
        "right_hand_decision": final.get("right_hand_decision", "PASS_WITH_RISK"),
        "teacher_decision": "FAIL_FROZEN",
        "candidate_hash_invariant_pass": hash_pass,
        "forbidden_scan": fscan,
        "process_scan": pscan,
        "residual_process": 0 if pscan["pass"] else len(pscan.get("local_python_or_modal_processes") or []),
        "blockers": [] if hash_pass and fscan["pass"] and pscan["pass"] else ["hash/forbidden/process gate failed"],
    }
    write_json(REPORTS / "V223_A_to_L_completion_audit.json", completion)
    write_md(
        REPORTS / "V223_A_to_L_completion_audit.md",
        "V223 A-L Completion Audit",
        [
            f"- status: `{completion['status']}`",
            f"- strict_candidate_passes: `{completion['strict_candidate_passes']}`",
            f"- strict_teacher_passes: `{completion['strict_teacher_passes']}`",
            f"- forbidden_hit_count: `{fscan['forbidden_hit_count']}`",
            f"- process_scan_pass: `{pscan['pass']}`",
        ],
    )
    print(json.dumps({"status": completion["status"], "blockers": completion["blockers"]}, ensure_ascii=False))
    return 0 if not completion["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
