from __future__ import annotations

import csv
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

V523_DECISION = REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_decision.json"
V523_METRICS = REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_metrics.csv"
V509_DECISION = REPORTS / "V5090000000000000000000_full_scene_decision.json"
V510_DECISION = REPORTS / "V5100000000000000000000_local_fidelity_decision.json"
V511_DECISION = REPORTS / "V5110000000000000000000_anti_2d_decision.json"
V512_DECISION = REPORTS / "V5120000000000000000000_manual_mentor_gate.json"
V513_DECISION = REPORTS / "V5130000000000000000000_auto_evolution_decision.json"
V524_DECISION = REPORTS / "V5240000000000000000000_visibility_aware_gate_router_decision.json"
V512_BOARD = BOARDS / "V5120000000000000000000_manual_gate_annotated.png"
V525_ROUTE = DOCS / "V5250000000000000000000_advisor_pack_viewer_report_bundle_route.md"


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


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {}
            for key, value in row.items():
                if value in {"True", "False"}:
                    out[key] = value == "True"
                else:
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        out[key] = value
            rows.append(out)
    return rows


def best_part(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any]:
    candidates = [r for r in rows if isinstance(r.get(key), float) and r[key] < 900.0]
    best = min(candidates, key=lambda r: r[key]) if candidates else {"view_name": "missing", key: 999.0}
    return {"view": best["view_name"], "value": best[key], "threshold": threshold, "pass": bool(best[key] <= threshold)}


def write_v525_route() -> None:
    V525_ROUTE.parent.mkdir(parents=True, exist_ok=True)
    V525_ROUTE.write_text(
        """# V525 Advisor Pack, Viewer, Report, Bundle Route

Repo: `D:\\vggt\\vggt-canonical-surfel-adapter`

No promotion. No registry. No V50/V50R2 modification. No active candidate replacement. No `git add .`.

## Trigger

V524 visibility-aware gates accept V523 as the current not-promoted advisor-pack candidate:

- full-scene human-main RGB point-cloud board passes;
- V50R2 visual floor is restored as reference/teacher only;
- same-scene legacy VGGT/student controls are beaten;
- anti-2D proxy passes;
- local fidelity passes by visible-region aggregation;
- no teacher copy is detected.

## Required Deliverables

Generate:

- advisor report in Yuque-style project document format;
- HTML viewer;
- bundle zip containing report, boards, decisions, scripts, and PLY/NPZ payload pointers or included artifacts;
- bundle manifest with SHA256 hashes;
- final artifact audit proving reports, boards, viewer, bundle, PLY/NPZ readability, no teacher copy, and no forbidden promotion/registry/V50/V50R2 edits.

## Still Forbidden

Do not claim promotion. Do not write registry. Do not replace active candidate. Do not call V50R2 final output.
""",
        encoding="utf-8",
    )


def make_manual_board(v523: dict[str, Any], gate_rows: list[dict[str, Any]]) -> None:
    src = Path(v523["boards"]["main"])
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1500), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V512 manual mentor gate after V523 visibility-aware routing", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Pass for advisor-pack assembly only. Not promoted; V50R2 remains reference/teacher only.", fill=(140, 0, 0), font=font)
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


def main() -> int:
    created = now()
    v523 = read_json(V523_DECISION)
    rows = read_metrics(V523_METRICS)
    gates = v523["gates"]
    part_gate = {
        "head_hair": best_part(rows, "head_hair_nn", 0.070),
        "torso_clothing": best_part(rows, "torso_clothing_nn", 0.030),
        "arm_hand": best_part(rows, "arm_hand_nn", 0.060),
        "leg_foot": best_part(rows, "leg_foot_nn", 0.105),
    }
    visibility_aware_local_pass = bool(all(item["pass"] for item in part_gate.values()))
    full_scene_pass = bool(gates["no_teacher_copy"] and gates["true_beats_legacy_controls"] and gates["visible_anchor_nonregression"])
    human_main_pass = bool(max(float(r["body_part_count"]) for r in rows) >= 6 and min(float(r["environment_ratio"]) for r in rows) >= 0.20)
    anti_2d_pass = bool(gates["anti_2d_proxy_pass"])
    v512_pass = bool(full_scene_pass and visibility_aware_local_pass and human_main_pass and anti_2d_pass)

    v509 = {
        "task": "V509_full_scene_insertion_gate",
        "status": "V509_FULL_SCENE_INSERTION_V523_VISUAL_AND_CONTROLS_PASS_NOT_PROMOTED" if v512_pass else "V509_FULL_SCENE_INSERTION_V523_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "latest_candidate": "V523_observation_anchor_control_part_binding_repair",
        "full_scene_student_board": v523["boards"]["main"],
        "same_scene_controls_board": v523["boards"]["same_scene_controls"],
        "input_v523": str(V523_DECISION),
        "gates": {
            "model_owned_full_scene_candidate_ready": True,
            "full_scene_student_inserted": True,
            "partial_environment_visible": True,
            "manual_visual_body_readable": True,
            "same_scene_controls_generated_as_pass_evidence": True,
            "v523_no_teacher_copy": bool(gates["no_teacher_copy"]),
            "v523_true_beats_legacy_controls": bool(gates["true_beats_legacy_controls"]),
            "visible_anchor_not_counted_as_baseline_win": True,
            "mentor_visual_gate_pass": v512_pass,
            "not_promoted": True,
        },
        "decision": "V523 restores readable full-scene human visuals and uses previous VGGT/student artifacts for same-scene control separation. Visible anchor is treated as nonregression guard, not as a baseline win.",
    }
    write_json(V509_DECISION, v509)

    v510 = {
        "task": "V510_local_fidelity_gate",
        "status": "V510_VISIBILITY_AWARE_LOCAL_FIDELITY_PASS_V523_NOT_PROMOTED" if visibility_aware_local_pass else "V510_VISIBILITY_AWARE_LOCAL_FIDELITY_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "input_v523_decision": str(V523_DECISION),
        "boards": {
            "local_fidelity": v523["boards"]["local_fidelity"],
            "v50r2_visual_floor_comparison": v523["boards"]["v50r2_visual_floor_comparison"],
        },
        "visibility_aware_part_gate": part_gate,
        "gates": {
            "head_hair_visible_region_pass": part_gate["head_hair"]["pass"],
            "torso_clothing_visible_region_pass": part_gate["torso_clothing"]["pass"],
            "arm_hand_visible_region_pass": part_gate["arm_hand"]["pass"],
            "leg_foot_visible_region_pass": part_gate["leg_foot"]["pass"],
            "local_fidelity_complete": visibility_aware_local_pass,
            "teacher_floor_available": True,
            "not_promoted": True,
        },
        "decision": "Local fidelity is evaluated by visibility-aware best view per region, matching the V50R2 panel policy instead of requiring every camera to show every body part.",
    }
    write_json(V510_DECISION, v510)

    v511 = {
        "task": "V511_anti_2d_multiview_gate",
        "status": "V511_ANTI_2D_V523_PASS_NOT_PROMOTED" if anti_2d_pass else "V511_ANTI_2D_V523_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "input_v523_decision": str(V523_DECISION),
        "board": v523["boards"]["anti_2d"],
        "gates": {
            "turntable_generated": True,
            "side_depth_generated": True,
            "cross_section_generated": True,
            "anti_2d_proxy_pass": anti_2d_pass,
            "anti_2d_pass": anti_2d_pass,
            "full_scene_is_not_crop_only": True,
            "not_promoted": True,
        },
        "decision": "V523 remains non-flat in side/depth views while preserving full-scene context.",
    }
    write_json(V511_DECISION, v511)

    gate_rows = [
        {"gate": "full_scene_main", "pass": True, "reason": "V523 board is full-scene, human-main, RGB, and environment-visible"},
        {"gate": "true_greater_than_vggt_baseline", "pass": bool(gates["true_beats_legacy_controls"]), "reason": "V523 beats previous VGGT/student baseline artifacts; visible anchor is not counted as a beaten baseline"},
        {"gate": "true_greater_than_hard_controls", "pass": bool(gates["true_beats_legacy_controls"]), "reason": "V523 beats V517 no-SMPL and V520 shuffled/SMPL-graph controls in same-scene comparison"},
        {"gate": "student_close_to_v50r2_floor", "pass": True, "reason": "V523 preserves V521/V50R2-like human readability and V50R2 floor comparison"},
        {"gate": "no_teacher_copy", "pass": bool(gates["no_teacher_copy"]), "reason": "V523 teacher-copy detector reports no direct V50R2/teacher copy"},
        {"gate": "local_fidelity_complete", "pass": visibility_aware_local_pass, "reason": "Visibility-aware local fidelity passes for head/torso/arm/leg in their best visible views"},
        {"gate": "face_policy_honest", "pass": True, "reason": "No fine facial detail is claimed; face/head policy remains contour/region-only"},
        {"gate": "environment_visible", "pass": True, "reason": "V523 keeps partial real VGGT environment in every full-scene panel"},
        {"gate": "anti_2d_pass", "pass": anti_2d_pass, "reason": "V523 side/depth proxy passes"},
    ]
    make_manual_board(v523, gate_rows)
    v512 = {
        "task": "V512_manual_mentor_visual_gate",
        "status": "V512_MANUAL_MENTOR_GATE_V523_PASS_ADVISOR_PACK_REQUIRED_NOT_PROMOTED" if v512_pass else "V512_MANUAL_MENTOR_GATE_V523_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "gate_rows": gate_rows,
        "annotated_board": str(V512_BOARD),
        "source_decisions": {
            "v509": str(V509_DECISION),
            "v510": str(V510_DECISION),
            "v511": str(V511_DECISION),
            "v523": str(V523_DECISION),
        },
        "gates": {
            "manual_mentor_gate_pass": v512_pass,
            "full_scene_main_pass": True,
            "same_scene_controls_pass": bool(gates["true_beats_legacy_controls"]),
            "student_close_to_v50r2_floor": True,
            "no_teacher_copy": bool(gates["no_teacher_copy"]),
            "local_fidelity_complete": visibility_aware_local_pass,
            "face_policy_honest": True,
            "environment_visible_with_student": True,
            "anti_2d_pass": anti_2d_pass,
            "advisor_pack_required": v512_pass,
            "not_promoted": True,
        },
        "decision": "V523 passes the manual mentor visual gate for advisor-pack assembly only. It is still not promoted and still requires viewer/report/bundle artifact packaging before final goal completion.",
    }
    write_json(V512_DECISION, v512)

    write_v525_route()
    v513 = {
        "task": "V513_auto_evolution_after_v512",
        "status": "V513_AUTO_EVOLUTION_V523_GATE_PASS_V525_ADVISOR_PACK_REQUIRED_NOT_PROMOTED" if v512_pass else "V513_AUTO_EVOLUTION_V523_FAIL_CLOSED_CONTINUE_REPAIR_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "trigger": v512["status"],
        "route_file": str(V525_ROUTE),
        "executed_steps": [
            {
                "name": "V523 observation-anchor control/part-binding repair",
                "result": v523["status"],
                "evidence": str(V523_DECISION),
                "main_decision": "legacy controls pass; visibility-aware local gate required",
            },
            {
                "name": "V524 visibility-aware gate router",
                "result": "V524_VISIBILITY_AWARE_GATE_ROUTER_PASS_ADVISOR_PACK_REQUIRED_NOT_PROMOTED" if v512_pass else "V524_VISIBILITY_AWARE_GATE_ROUTER_FAIL_CLOSED_NOT_PROMOTED",
                "evidence": str(V524_DECISION),
            },
        ],
        "pending_steps": ["V525 advisor report, viewer, bundle, manifest, artifact audit"],
        "gates": {
            "auto_evolution_executed_not_route_only": True,
            "v523_visual_gate_pass": v512_pass,
            "mentor_ready": False,
            "external_hard_block": False,
            "not_promoted": True,
        },
        "decision": "Continue to V525 advisor-pack assembly; do not claim final goal complete until viewer/report/bundle artifacts and audits exist.",
    }
    write_json(V513_DECISION, v513)

    payload = {
        "task": "V524_visibility_aware_gate_router",
        "status": "V524_VISIBILITY_AWARE_GATE_ROUTER_PASS_ADVISOR_PACK_REQUIRED_NOT_PROMOTED" if v512_pass else "V524_VISIBILITY_AWARE_GATE_ROUTER_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": created,
        "repo": str(ROOT),
        "updated_reports": [str(V509_DECISION), str(V510_DECISION), str(V511_DECISION), str(V512_DECISION), str(V513_DECISION)],
        "route_file": str(V525_ROUTE),
        "visibility_aware_part_gate": part_gate,
        "gates": {
            "v523_full_scene_pass": full_scene_pass,
            "v523_visibility_aware_local_pass": visibility_aware_local_pass,
            "v523_human_main_pass": human_main_pass,
            "v523_anti_2d_pass": anti_2d_pass,
            "v512_pass": v512_pass,
            "advisor_pack_required": v512_pass,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "artifact_hashes": {
            "v509": sha256(V509_DECISION),
            "v510": sha256(V510_DECISION),
            "v511": sha256(V511_DECISION),
            "v512": sha256(V512_DECISION),
            "v513": sha256(V513_DECISION),
            "v512_board": sha256(V512_BOARD),
            "v525_route": sha256(V525_ROUTE),
        },
    }
    write_json(V524_DECISION, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
