import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

SUBOBJECTIVE_POSTMORTEM_JSON = OUTPUT_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"
VERDICT_JSON = OUTPUT_ROOT / "candidate_verdict.json"

POSTMORTEM_JSON = OUTPUT_ROOT / "camera_depth_objective_coupling_audit_postmortem.20260403.json"
POSTMORTEM_MD = OUTPUT_ROOT / "camera_depth_objective_coupling_audit_postmortem.20260403.md"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def render_md(payload: dict) -> str:
    lines = [
        "# Camera-Depth Objective Coupling Audit Postmortem (2026-04-03)",
        "",
        f"- family: `{payload['family']}`",
        f"- latest_closed_ticket: `{payload['latest_closed_ticket']['family']}`",
        f"- latest_verdict: `{payload['latest_closed_ticket']['status']}`",
        f"- gate_stage_reached: `{payload['latest_closed_ticket']['gate_stage_reached']}`",
        "",
        "## Boundary",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["boundary_readout"]])
    lines.extend(["", "## Next Question", "", f"- {payload['next_question']}", ""])
    return "\n".join(lines)


def main() -> int:
    subobjective = load_json(SUBOBJECTIVE_POSTMORTEM_JSON)
    verdict = load_json(VERDICT_JSON)

    family = str(verdict.get("family", "")).strip()
    status = str(verdict.get("status", "")).strip()
    if family != "camera_objective_coupling_rebalancing":
        raise RuntimeError("candidate_verdict family must be camera_objective_coupling_rebalancing for the audit postmortem.")
    if status not in {"dead_same_day", "failed_long_gate"}:
        raise RuntimeError("camera_depth_objective_coupling_audit postmortem expects a closed coupling verdict.")

    short_gate = verdict.get("short_gate_vs_lead", {}) or {}
    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "camera_depth_objective_coupling_audit_postmortem",
        "family": "camera_depth_objective_coupling_audit",
        "closed_axes": {
            "camera_focal_objective_isolation": subobjective.get("focal_isolation", {}),
            "camera_translation_objective_isolation": subobjective.get("translation_isolation", {}),
            "camera_objective_coupling_rebalancing": {
                "family": family,
                "status": status,
                "shape": verdict.get("first_candidate_shape", ""),
                "reason": verdict.get("reason", ""),
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
                "short_gate_vs_lead": short_gate,
            },
        },
        "latest_closed_ticket": {
            "family": family,
            "status": status,
            "shape": verdict.get("first_candidate_shape", ""),
            "candidate_config": verdict.get("active_candidate", ""),
            "reason": verdict.get("reason", ""),
            "gate_stage_reached": verdict.get("gate_stage_reached", ""),
        },
        "boundary_readout": [
            "Close the focal-only axis for this round.",
            "Close the translation-only axis for this round.",
            "Close the bounded FL/T coupling axis for this round.",
            "Depth did not emerge as the broken side of the objective during these tickets; the plateau now reads as a camera-vs-depth coupling question at the global objective level.",
            "Do not reopen focal-only, translation-only, FL/T coupling, tail-contract derivative, source-policy, or bucket cousins from this boundary.",
        ],
        "next_question": (
            "Can one bounded global camera-weight relief on the stable lead improve the joint camera-depth gate "
            "balance without changing dataset routing, trainer plumbing, or cloud scope?"
        ),
        "recommended_next_problem": "camera_depth_objective_coupling_audit",
        "recommended_first_candidate_shape": "stablelead_global_cameraweight095_depthhold100",
        "recommended_first_candidate_constraints": [
            "keep the current stable lead fixed",
            "do not change dataset or trainer plumbing",
            "reduce only loss.camera.weight from 1.0 to 0.95",
            "hold loss.depth.weight at 1.0",
            "keep cloud off",
        ],
        "supporting_refs": {
            "camera_subobjective_isolation_postmortem": repo_rel(SUBOBJECTIVE_POSTMORTEM_JSON),
            "candidate_verdict": repo_rel(VERDICT_JSON),
        },
    }

    write_json(POSTMORTEM_JSON, payload)
    write_text(POSTMORTEM_MD, render_md(payload))
    print(json.dumps({"camera_depth_objective_coupling_audit_postmortem": repo_rel(POSTMORTEM_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
