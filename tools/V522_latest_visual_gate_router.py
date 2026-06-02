from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
DOCS = ROOT / "docs" / "goals"

V521_DECISION = REPORTS / "V5210000000000000000000_observation_anchored_visible_student_decision.json"
V519_DECISION = REPORTS / "V5190000000000000000000_canonical_surfel_graph_training_decision.json"
V520_DECISION = REPORTS / "V5200000000000000000000_pose_aligned_surfel_graph_decision.json"

V509_DECISION = REPORTS / "V5090000000000000000000_full_scene_decision.json"
V510_DECISION = REPORTS / "V5100000000000000000000_local_fidelity_decision.json"
V511_DECISION = REPORTS / "V5110000000000000000000_anti_2d_decision.json"
V512_DECISION = REPORTS / "V5120000000000000000000_manual_mentor_gate.json"
V513_DECISION = REPORTS / "V5130000000000000000000_auto_evolution_decision.json"
V522_DECISION = REPORTS / "V5220000000000000000000_latest_visual_gate_router_decision.json"
V512_BOARD = BOARDS / "V5120000000000000000000_manual_gate_annotated.png"
V523_ROUTE = DOCS / "V5230000000000000000000_auto_evolved_observation_anchor_control_part_binding_repair_route.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def status(path: Path) -> str:
    try:
        return str(read_json(path).get("status", "missing"))
    except Exception:
        return "missing"


def make_manual_board(v521: dict[str, Any], gate_rows: list[dict[str, Any]]) -> None:
    src = Path(v521["boards"]["main"])
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1500), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V512 manual mentor gate after V521 visual recovery", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Human readability recovered, but baseline/control separation is not mentor-convincing. Not promoted.", fill=(140, 0, 0), font=font)
    if src.exists():
        with Image.open(src).convert("RGB") as im:
            im.thumbnail((1760, 1000), Image.Resampling.LANCZOS)
            board.paste(im, (18, 70))
    y = 1080
    for row in gate_rows:
        mark = "PASS" if row["pass"] else "FAIL"
        color = (0, 110, 0) if row["pass"] else (150, 0, 0)
        draw.text((18, y), f"{mark} {row['gate']}: {row['reason']}", fill=color, font=font)
        y += 30
    V512_BOARD.parent.mkdir(parents=True, exist_ok=True)
    board.save(V512_BOARD)


def write_v523_route() -> None:
    V523_ROUTE.parent.mkdir(parents=True, exist_ok=True)
    V523_ROUTE.write_text(
        """# V523 Auto-Evolved Observation-Anchor Control/Part-Binding Repair Route

Repo: `D:\\vggt\\vggt-canonical-surfel-adapter`

Branch: `codex/volume-aware-3d-morphology`

No promotion. No registry. No V50/V50R2 modification. No active candidate replacement. No `git add .`.

## Trigger

V521 recovered the central visual direction: the human is readable again in full-scene RGB point-cloud panels with partial environment. However it remains fail-closed because the V42/VGGT visible anchor is itself very strong and the student does not visibly or metrically beat the competitive visible-anchor baseline.

## Required Repair

1. Separate source roles:
   - V50R2: visual floor / teacher / reference only.
   - V42 visible observation: input anchor, not a claimed VGGT baseline win.
   - VGGT baseline/control: use the actual prior student/VGGT baseline artifacts from V514/V517/V520 for mentor comparison.
2. Repair body-part binding:
   - head/hair, torso/clothing, arm/hand, leg/foot labels must be robust enough for local gates.
   - Do not use coarse y/x-only labels as final body-part evidence.
3. Preserve V521 visual readability:
   - Do not return to V519/V520 free-template smear.
   - Any new residual/completion must be identity-safe when it would degrade the visible anchor.
4. Re-run gates:
   - V509 full-scene insertion.
   - V510 local fidelity.
   - V511 anti-2D.
   - V512 manual mentor gate.

## Forbidden Success Claims

Do not claim success for raw visible-anchor recovery alone, teacher-only/crop-only output, metric-only improvement, projection-only evidence, or route-created-only status.

## Allowed Next Statuses

- `V523_OBSERVATION_ANCHOR_CONTROL_PART_BINDING_REPAIR_READY_FOR_V512_NOT_PROMOTED`
- `V523_OBSERVATION_ANCHOR_REPAIR_FAIL_CLOSED_CONTINUE_MODEL_REPAIR_NOT_PROMOTED`
""",
        encoding="utf-8",
    )


def main() -> int:
    v521 = read_json(V521_DECISION)
    v521_boards = v521["boards"]
    v521_gates = v521["gates"]
    created = now()

    v509 = {
        "task": "V509_full_scene_insertion_gate",
        "status": "V509_FULL_SCENE_INSERTION_V521_VISUAL_RECOVERED_CONTROLS_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "latest_candidate": "V521_observation_anchored_visible_student",
        "full_scene_student_board": v521_boards["main"],
        "same_scene_controls_board": v521_boards["same_scene_controls"],
        "input_v519": str(V519_DECISION),
        "input_v520": str(V520_DECISION),
        "input_v521": str(V521_DECISION),
        "input_statuses": {
            "v519": status(V519_DECISION),
            "v520": status(V520_DECISION),
            "v521": v521["status"],
        },
        "gates": {
            "model_owned_full_scene_candidate_ready": True,
            "full_scene_student_inserted": True,
            "partial_environment_visible": True,
            "manual_visual_body_readable": True,
            "same_scene_controls_generated_as_pass_evidence": False,
            "v521_no_teacher_copy": bool(v521_gates["no_teacher_copy"]),
            "v521_true_beats_required_controls": bool(v521_gates["true_beats_required_controls"]),
            "mentor_visual_gate_pass": False,
            "not_promoted": True,
            "auto_evolve_required": True,
        },
        "decision": "V521 recovers a readable human-main full-scene visual direction and preserves partial environment, but it does not clearly beat the competitive visible-anchor/VGGT-style baseline. Keep fail-closed and repair control separation plus body-part binding.",
        "blockers": [
            "V521 human is readable, but true does not clearly beat the strong visible-anchor baseline",
            "Same-scene controls are generated but not mentor-convincing as pass evidence",
            "V523 must preserve V521 readability while repairing baseline/control separation",
        ],
    }
    write_json(V509_DECISION, v509)

    v510 = {
        "task": "V510_local_fidelity_gate",
        "status": "V510_LOCAL_FIDELITY_V521_VISUAL_FLOOR_RECOVERED_PART_BINDING_INCOMPLETE_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "input_v509_decision": str(V509_DECISION),
        "input_v521_decision": str(V521_DECISION),
        "boards": {
            "v50r2_visual_floor_comparison": v521_boards["v50r2_visual_floor_comparison"],
            "main": v521_boards["main"],
        },
        "gates": {
            "head_hair_readability_recovered": True,
            "torso_clothing_readability_recovered": True,
            "leg_foot_readability_recovered": True,
            "arm_hand_readability_incomplete": True,
            "body_part_binding_proxy_pass": False,
            "local_fidelity_complete": False,
            "teacher_floor_available": True,
            "not_promoted": True,
            "auto_evolve_required": True,
        },
        "decision": "V521 visually matches the V50R2 floor much better than V519/V520, but local fidelity is not complete because arm/hand evidence and robust body-part binding remain incomplete.",
        "blockers": [
            "V521 body-part labels are too coarse for final local morphology evidence",
            "Arm/hand and local part gates need a stronger SMPL-X/body-part binding route",
        ],
    }
    write_json(V510_DECISION, v510)

    v511 = {
        "task": "V511_anti_2d_multiview_gate",
        "status": "V511_ANTI_2D_V521_PROXY_PASS_CONTROLS_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "input_v509_decision": str(V509_DECISION),
        "input_v521_decision": str(V521_DECISION),
        "board": v521_boards["anti_2d"],
        "gates": {
            "turntable_generated": True,
            "side_depth_generated": True,
            "cross_section_generated": True,
            "anti_2d_proxy_pass": bool(v521_gates["anti_2d_proxy_pass"]),
            "anti_2d_pass": bool(v521_gates["anti_2d_proxy_pass"]),
            "full_scene_is_not_crop_only": True,
            "accepted_student_available": False,
            "not_promoted": True,
            "auto_evolve_required": True,
        },
        "decision": "V521 has non-flat side/depth evidence and full-scene context. Anti-2D is no longer the primary blocker; control separation and body-part/local fidelity remain blockers.",
        "blockers": [
            "Anti-2D proxy passes, but V512 still fails due control separation",
            "Per-camera visible surfaces must not be overclaimed as final full fused completion",
        ],
    }
    write_json(V511_DECISION, v511)

    gate_rows = [
        {"gate": "full_scene_main", "pass": True, "reason": "V521 main board is full-scene, human-main, RGB, and environment-visible"},
        {"gate": "true_greater_than_vggt_baseline", "pass": False, "reason": "V521 true does not clearly beat the competitive visible-anchor/VGGT-style baseline"},
        {"gate": "true_greater_than_hard_controls", "pass": False, "reason": "Same-scene controls exist, but visual separation is not enough for mentor pass"},
        {"gate": "student_close_to_v50r2_floor", "pass": True, "reason": "V521 restores V50R2-like human readability in the visual floor comparison"},
        {"gate": "no_teacher_copy", "pass": True, "reason": "V521 teacher-copy detector reports no direct V50R2/teacher copy"},
        {"gate": "local_fidelity_complete", "pass": False, "reason": "V521 part binding is too coarse; arm/hand and local gates are incomplete"},
        {"gate": "face_policy_honest", "pass": True, "reason": "No fine facial detail is claimed; face/head policy remains contour/region-only"},
        {"gate": "environment_visible", "pass": True, "reason": "V521 keeps partial real VGGT environment in each full-scene panel"},
        {"gate": "anti_2d_pass", "pass": True, "reason": "V521 side/depth proxy passes, but this alone is not enough"},
    ]
    make_manual_board(v521, gate_rows)
    v512 = {
        "task": "V512_manual_mentor_visual_gate",
        "status": "V512_MANUAL_MENTOR_GATE_V521_VISUAL_RECOVERED_CONTROLS_FAIL_CLOSED_AUTO_EVOLVE_REQUIRED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "gate_rows": gate_rows,
        "annotated_board": str(V512_BOARD),
        "source_decisions": {
            "v509": str(V509_DECISION),
            "v510": str(V510_DECISION),
            "v511": str(V511_DECISION),
            "v519": str(V519_DECISION),
            "v520": str(V520_DECISION),
            "v521": str(V521_DECISION),
        },
        "input_statuses": {
            "v509": v509["status"],
            "v510": v510["status"],
            "v511": v511["status"],
            "v519": status(V519_DECISION),
            "v520": status(V520_DECISION),
            "v521": v521["status"],
        },
        "gates": {
            "manual_mentor_gate_pass": False,
            "full_scene_main_pass": True,
            "same_scene_controls_pass": False,
            "student_close_to_v50r2_floor": True,
            "no_teacher_copy": True,
            "local_fidelity_complete": False,
            "face_policy_honest": True,
            "environment_visible_with_student": True,
            "anti_2d_pass": True,
            "auto_evolve_required": True,
            "not_promoted": True,
        },
        "decision": "Manual mentor gate remains fail-closed. V521 fixes the largest visual regression by restoring human readability, but final pass is blocked by control separation against the strong visible anchor and incomplete body-part/local fidelity.",
        "blockers": [
            "True is not clearly better than the competitive visible-anchor/VGGT-style baseline",
            "Hard-control visual separation is not mentor-convincing enough",
            "Body-part/local fidelity is incomplete",
        ],
    }
    write_json(V512_DECISION, v512)

    write_v523_route()
    v513 = {
        "task": "V513_auto_evolution_after_v512_fail_closed",
        "status": "V513_AUTO_EVOLUTION_V521_VISUAL_RECOVERED_V523_CONTROL_REPAIR_REQUIRED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "trigger": v512["status"],
        "route_file": str(V523_ROUTE),
        "executed_steps": [
            {
                "name": "V519 canonical SMPL-X surfel graph training/adjudication",
                "result": status(V519_DECISION),
                "evidence": str(V519_DECISION),
                "main_decision": "automatic metrics improved but manual visual remained smeared/blob-like",
            },
            {
                "name": "V520 pose-aligned surfel graph repair",
                "result": status(V520_DECISION),
                "evidence": str(V520_DECISION),
                "main_decision": "pose/axis alignment improved thickness and metrics but not mentor-readable morphology",
            },
            {
                "name": "V521 observation-anchored visible student",
                "result": v521["status"],
                "evidence": str(V521_DECISION),
                "main_decision": "human readability recovered with full-scene context, but control separation and part binding failed closed",
            },
            {
                "name": "V522 latest visual gate router",
                "result": "V522_GATE_ROUTER_RECORDED_V521_VISUAL_RECOVERY_AND_V523_REQUIRED_NOT_PROMOTED",
                "evidence": str(V522_DECISION),
            },
        ],
        "pending_steps": [
            "V523 repair control separation without degrading V521 visual readability",
            "V523 replace coarse y/x part labels with robust SMPL-X/body-part binding",
            "V509-V512 rerun after V523",
        ],
        "gates": {
            "auto_evolution_executed_not_route_only": True,
            "v521_human_readability_recovered": True,
            "v521_no_teacher_copy": bool(v521_gates["no_teacher_copy"]),
            "v521_true_beats_required_controls": bool(v521_gates["true_beats_required_controls"]),
            "mentor_ready": False,
            "external_hard_block": False,
            "not_promoted": True,
        },
        "decision": "Auto-evolution did not stop at V519/V520 failures. V521 recovered the visual floor direction, but V512 still fails closed because controls and local body-part fidelity are incomplete. Continue with V523; do not promote.",
        "blockers": [
            "V521 true does not clearly beat the competitive visible-anchor baseline",
            "V521 body-part/local gates are incomplete",
            "V523 must keep V521 readability while repairing controls",
        ],
    }
    write_json(V513_DECISION, v513)

    payload = {
        "task": "V522_latest_visual_gate_router",
        "status": "V522_GATE_ROUTER_RECORDED_V521_VISUAL_RECOVERY_AND_V523_REQUIRED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "updated_reports": [str(V509_DECISION), str(V510_DECISION), str(V511_DECISION), str(V512_DECISION), str(V513_DECISION)],
        "route_file": str(V523_ROUTE),
        "annotated_board": str(V512_BOARD),
        "gates": {
            "v521_visual_recovery_recorded": True,
            "mentor_ready": False,
            "auto_evolve_required": True,
            "not_promoted": True,
        },
        "artifact_hashes": {
            "v509": sha256(V509_DECISION),
            "v510": sha256(V510_DECISION),
            "v511": sha256(V511_DECISION),
            "v512": sha256(V512_DECISION),
            "v513": sha256(V513_DECISION),
            "v512_board": sha256(V512_BOARD),
            "v523_route": sha256(V523_ROUTE),
        },
    }
    write_json(V522_DECISION, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
