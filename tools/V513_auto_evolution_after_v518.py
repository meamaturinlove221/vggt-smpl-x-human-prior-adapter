from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GOALS = ROOT / "docs" / "goals"

V512 = REPORTS / "V5120000000000000000000_manual_mentor_gate.json"
V517 = REPORTS / "V5170000000000000000000_full_scene_clarity_decision.json"
V518 = REPORTS / "V5180000000000000000000_canonical_surfel_graph_repair_decision.json"
V518_ROUTE = GOALS / "V5180000000000000000000_auto_evolved_canonical_surfel_graph_repair_route.md"
OUT = REPORTS / "V5130000000000000000000_auto_evolution_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def main() -> int:
    v512 = read_json(V512)
    v517 = read_json(V517)
    v518 = read_json(V518)
    v517_gates = v517.get("gates", {})
    v518_gates = v518.get("gates", {})
    payload = {
        "task": "V513_auto_evolution_after_v512_fail_closed",
        "status": "V513_AUTO_EVOLUTION_V518_REPRESENTATION_SWITCH_EXECUTED_V519_REQUIRED_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "trigger": v512.get("status"),
        "route_file": str(V518_ROUTE),
        "executed_steps": [
            {
                "name": "V516 paired visible-surface control-separation adjudication",
                "result": "V516_PAIRED_VISIBLE_SURFACE_CONTROL_SEPARATION_PASS_NEEDS_V509_V512_NOT_PROMOTED",
                "evidence": str(REPORTS / "V5160000000000000000000_paired_visible_surface_decision.json"),
            },
            {
                "name": "V517 full-scene clarity composer",
                "result": v517.get("status"),
                "evidence": str(V517),
                "main_decision": "candidate generated but manual mentor visual remains fail-closed",
            },
            {
                "name": "V518 canonical SMPL-X surfel/graph representation repair smoke",
                "result": v518.get("status"),
                "evidence": str(V518),
                "main_decision": "representation switch executed; not accepted for V509 until V519 training/adjudication",
            },
        ],
        "pending_steps": [
            "V519 train/adjudicate canonical SMPL-X surfel graph with VGGT feature sampling, local body-part heads, anti-blob/anti-sheet losses",
            "V519 same-scene hard controls: no-SMPL-graph, shuffled semantic, weak semantic, VGGT visible baseline, SMPL graph only",
            "V519 local morphology boards vs V50R2 visual floor",
            "V509 full-scene insertion retry only if V519 manual visual improves over V517 and controls",
            "V510 local fidelity retry after accepted V519 student",
            "V511 anti-2D retry after accepted V519 student",
            "V512 manual mentor gate retry",
        ],
        "gates": {
            "auto_evolution_executed_not_route_only": True,
            "v517_model_owned_full_scene_candidate_ready": bool(v517_gates.get("accepted_for_v509", False)),
            "v517_no_teacher_copy": bool(v517_gates.get("no_teacher_copy", False)),
            "v517_manual_visual_body_readable": False,
            "v518_representation_switch_executed": bool(v518_gates.get("representation_switch_executed", False)),
            "v518_canonical_graph_body_parts_present": bool(v518_gates.get("canonical_graph_body_parts_present", False)),
            "v518_no_teacher_copy": bool(v518_gates.get("no_teacher_copy", False)),
            "v518_accepted_for_v509": bool(v518_gates.get("accepted_for_v509", False)),
            "mentor_ready": False,
            "external_hard_block": False,
            "not_promoted": True,
        },
        "decision": (
            "Auto-evolution did not stop at V517 visual failure. V518 executed the required representation switch "
            "to canonical SMPL-X surfel/graph smoke evidence. It is still not mentor-ready; proceed to V519 training "
            "and strict control adjudication before any V509/V512 pass claim."
        ),
        "blockers": [
            "V517 full-scene candidate remains below V50R2 human morphology floor",
            "V518 smoke is topology-oriented but still visually line-like and untrained",
            "V519 must train/adjudicate body-part surfel heads before mentor-ready evidence is possible",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
