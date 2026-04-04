import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

FAMILY_STOP_REASON_JSON = OUTPUT_ROOT / "family_stop_reason.json"
FOCAL_BLUEPRINT_JSON = OUTPUT_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
TRANSLATION_BLUEPRINT_JSON = OUTPUT_ROOT / "family_blueprint.camera_translation_objective_isolation.json"
POSTMORTEM_JSON = OUTPUT_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"
POSTMORTEM_MD = OUTPUT_ROOT / "camera_subobjective_isolation_postmortem.20260402.md"


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
        "# Camera Subobjective Isolation Postmortem (2026-04-02)",
        "",
        "## Focal Isolation",
        "",
    ]
    for key, value in payload["focal_isolation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Translation Isolation", ""])
    for key, value in payload["translation_isolation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## New Boundary", ""])
    for item in payload["new_boundary"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    focal_blueprint = load_json(FOCAL_BLUEPRINT_JSON)
    translation_blueprint = load_json(TRANSLATION_BLUEPRINT_JSON)
    focal_fallback = (family_stop.get("latest_family_outcomes", {}) or {}).get("camera_focal_objective_isolation", {})
    translation_fallback = (family_stop.get("latest_family_outcomes", {}) or {}).get("camera_translation_objective_isolation", {})
    focal = focal_blueprint.get("latest_verdict", {}) or focal_fallback
    translation = translation_blueprint.get("latest_verdict", {}) or translation_fallback

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "camera_subobjective_isolation_postmortem",
        "focal_isolation": {
            "family": "camera_focal_objective_isolation",
            "status": focal.get("status", focal_fallback.get("latest_status", "")),
            "shape": focal_blueprint.get("first_candidate_shape", focal_fallback.get("first_candidate_shape", "")),
            "reason": focal.get("reason", ""),
            "gate_stage_reached": focal.get("gate_stage_reached", ""),
        },
        "translation_isolation": {
            "family": "camera_translation_objective_isolation",
            "status": translation.get("status", translation_fallback.get("latest_status", "")),
            "shape": translation_blueprint.get("first_candidate_shape", translation_fallback.get("first_candidate_shape", "")),
            "reason": translation.get("reason", ""),
            "gate_stage_reached": translation.get("gate_stage_reached", ""),
        },
        "new_boundary": [
            "Close the single-subobjective isolation axis for this round.",
            "Do not reopen focal-only or translation-only cousins.",
            "Next problem: camera_objective_coupling_rebalancing.",
        ],
        "next_problem": "camera_objective_coupling_rebalancing",
        "supporting_refs": {
            "family_stop_reason": repo_rel(FAMILY_STOP_REASON_JSON),
            "focal_blueprint": repo_rel(FOCAL_BLUEPRINT_JSON),
            "translation_blueprint": repo_rel(TRANSLATION_BLUEPRINT_JSON),
        },
    }

    write_json(POSTMORTEM_JSON, payload)
    write_text(POSTMORTEM_MD, render_md(payload))
    print(json.dumps({"camera_subobjective_isolation_postmortem": repo_rel(POSTMORTEM_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
