import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

VERDICT_JSON = OUTPUT_ROOT / "candidate_verdict.json"
POSTMORTEM_JSON = OUTPUT_ROOT / "camera_focal_vs_translation_postmortem.20260401.json"
POSTMORTEM_MD = OUTPUT_ROOT / "camera_focal_vs_translation_postmortem.20260401.md"


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
        "# Camera Focal vs Translation Postmortem (2026-04-01)",
        "",
        f"- family: `{payload['family']}`",
        f"- candidate_shape: `{payload['candidate_shape']}`",
        f"- verdict: `{payload['verdict']}`",
        f"- gate_stage_reached: `{payload['gate_stage_reached']}`",
        "",
        "## Core Readout",
        "",
    ]
    for key, value in payload["core_readout"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Conclusion", ""])
    for item in payload["conclusion"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Question", "", f"- {payload['next_question']}", ""])
    return "\n".join(lines)


def main() -> int:
    verdict = load_json(VERDICT_JSON)
    short_gate = verdict.get("short_gate_vs_lead", {}) or {}

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "camera_focal_vs_translation_postmortem",
        "family": str(verdict.get("family", "")),
        "candidate_shape": str(verdict.get("first_candidate_shape", "")),
        "candidate_config": str(verdict.get("active_candidate", "")),
        "verdict": str(verdict.get("status", "")),
        "gate_stage_reached": str(verdict.get("gate_stage_reached", "")),
        "core_readout": {
            "delta_camera": short_gate.get("delta_camera"),
            "delta_T": short_gate.get("delta_T"),
            "delta_conf_depth": short_gate.get("delta_conf_depth"),
            "delta_reg_depth": short_gate.get("delta_reg_depth"),
        },
        "conclusion": [
            "Pure FL isolation did not pass the multi-metric gate and is closed for this round.",
            "The executed lossflisolation0 ticket still produced a large camera improvement, so the run was informative rather than null.",
            "The dominant residual blocker shifted from a broad camera tax to translation-specific gate failure: T regressed while depth terms stayed flat.",
            "That makes camera_translation_objective_isolation the next honest single-variable follow-up, not another tail/source/bucket cousin.",
        ],
        "next_question": (
            "Can a single translation-objective isolation candidate on the current stable lead remove the remaining T blocker "
            "without reopening tail-contract or source-policy families?"
        ),
        "supporting_refs": {
            "candidate_verdict": repo_rel(VERDICT_JSON),
        },
    }

    write_json(POSTMORTEM_JSON, payload)
    write_text(POSTMORTEM_MD, render_md(payload))
    print(json.dumps({"camera_focal_vs_translation_postmortem": repo_rel(POSTMORTEM_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
