from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"

INPUTS = {
    "v501": REPORTS / "V5010000000000000000000_v50r2_visual_floor_decision.json",
    "v502": REPORTS / "V5020000000000000000000_v50r2_panel_decision.json",
    "v503": REPORTS / "V5030000000000000000000_regression_decision.json",
    "v504": REPORTS / "V5040000000000000000000_teacher_bank_decision.json",
    "v505": REPORTS / "V5050000000000000000000_firewall_smoke.json",
    "v506": REPORTS / "V5060000000000000000000_forward_smoke.json",
    "v507": REPORTS / "V5070000000000000000000_loss_smoke.json",
    "v508": REPORTS / "V5080000000000000000000_hash_reconciliation.json",
    "v508_modal": REPORTS / "V5080000000000000000000_modal_smoke_result.json",
    "v509": REPORTS / "V5090000000000000000000_full_scene_decision.json",
    "v510": REPORTS / "V5100000000000000000000_local_fidelity_decision.json",
    "v511": REPORTS / "V5110000000000000000000_anti_2d_decision.json",
    "v514": REPORTS / "V5140000000000000000000_checkpoint_adjudication_decision.json",
    "v516": REPORTS / "V5160000000000000000000_paired_visible_surface_decision.json",
    "v517": REPORTS / "V5170000000000000000000_full_scene_clarity_decision.json",
}

BOARD_INPUTS = [
    BOARDS / "V5010000000000000000000_v50r2_visual_floor_contact_sheet.png",
    BOARDS / "V5030000000000000000000_student_vs_v50r2_contact_sheet.png",
    BOARDS / "V5170000000000000000000_human_main_full_scene.png",
    BOARDS / "V5170000000000000000000_same_scene_controls.png",
    BOARDS / "V5170000000000000000000_turntable_side_depth_cross_section.png",
    BOARDS / "V5140000000000000000000_checkpoint_adjudication_board.png",
    BOARDS / "V5160000000000000000000_paired_visible_surface_board.png",
    BOARDS / "V5090000000000000000000_full_scene_student.png",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {"missing": str(path)}


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if path.is_file():
        with Image.open(path).convert("RGB") as im:
            im.thumbnail((size[0] - 30, size[1] - 70), Image.Resampling.LANCZOS)
            canvas.paste(im, ((size[0] - im.width) // 2, 44 + (size[1] - 70 - im.height) // 2))
    else:
        ImageDraw.Draw(canvas).text((20, 20), f"missing: {path}", fill=(140, 0, 0), font=ImageFont.load_default())
    return canvas


def make_board(output: Path, gate_rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1580), "white")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), "V512 manual mentor visual gate: FAIL CLOSED", fill=(140, 0, 0), font=font)
    draw.text((16, 34), "V517 full-scene candidate exists, but human morphology remains below the V50R2 visual floor.", fill=(0, 0, 0), font=font)
    cell_w, cell_h = 600, 300
    for i, path in enumerate(BOARD_INPUTS):
        x = (i % 3) * cell_w
        y = 68 + (i // 3) * cell_h
        panel = fit(path, (cell_w, cell_h - 12))
        board.paste(panel, (x, y))
        draw.rectangle([x + 8, y + 8, x + cell_w - 8, y + cell_h - 18], outline=(150, 80, 80), width=2)
        draw.text((x + 18, y + 14), path.name[:80], fill=(0, 0, 0), font=font)
    y0 = 990
    draw.text((16, y0), "Gate Summary", fill=(0, 0, 0), font=font)
    y = y0 + 28
    for row in gate_rows:
        color = (0, 110, 0) if row["pass"] else (150, 0, 0)
        draw.text((24, y), f'{row["gate"]}: pass={row["pass"]} | {row["reason"]}', fill=color, font=font)
        y += 28
    draw.text((16, 1530), "Decision: not mentor-ready; switch to canonical SMPL-X surfel/graph representation repair. No promotion / no registry.", fill=(140, 0, 0), font=font)
    board.save(output)


def main() -> int:
    data = {name: read_json(path) for name, path in INPUTS.items()}
    v514_gates = data["v514"].get("gates", {})
    v516_gates = data["v516"].get("gates", {})
    v517_gates = data["v517"].get("gates", {})
    gate_rows = [
        {"gate": "full_scene_main", "pass": False, "reason": "V517 generated a model-owned full-scene candidate with environment, but the human remains blob-like and not limb-readable"},
        {"gate": "true_greater_than_vggt_baseline", "pass": bool(v516_gates.get("paired_true_improves_vggt_baseline", False)), "reason": f"V516 paired pass={v516_gates.get('paired_true_improves_vggt_baseline')} but V514 full-scene visual remains unaccepted"},
        {"gate": "true_greater_than_hard_controls", "pass": False, "reason": "V517 same-scene controls exist, but visual separation is not mentor-convincing"},
        {"gate": "student_close_to_v50r2_floor", "pass": False, "reason": "V517 paired metrics are close, but the visible human morphology is still worse than the V50R2 visual floor"},
        {"gate": "no_teacher_copy", "pass": bool(v517_gates.get("no_teacher_copy", v514_gates.get("no_teacher_copy", False))), "reason": "V517/V505 copy checks show no direct teacher copy"},
        {"gate": "local_fidelity_complete", "pass": False, "reason": "V510 failed closed: local head/hair, limbs, hands/feet remain below V50R2 readability"},
        {"gate": "face_policy_honest", "pass": True, "reason": "V502 forbids fine face detail claims"},
        {"gate": "environment_visible", "pass": bool(v517_gates.get("partial_environment_visible", False)), "reason": "V517 preserves partial VGGT environment, but environment alone cannot pass mentor gate"},
        {"gate": "anti_2d_pass", "pass": False, "reason": "V517 anti-2D proxy passes, but proxy thickness does not replace readable body morphology"},
    ]
    board = BOARDS / "V5120000000000000000000_manual_gate_annotated.png"
    make_board(board, gate_rows)
    payload = {
        "task": "V512_manual_mentor_visual_gate",
        "status": "V512_MANUAL_MENTOR_GATE_FAIL_CLOSED_AUTO_EVOLVE_REQUIRED_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "gate_rows": gate_rows,
        "annotated_board": str(board),
        "source_decisions": {name: str(path) for name, path in INPUTS.items()},
        "input_statuses": {name: value.get("status") for name, value in data.items()},
        "gates": {
            "manual_mentor_gate_pass": False,
            "full_scene_main_pass": False,
            "same_scene_controls_pass": False,
            "v517_candidate_available_for_review": bool(v517_gates.get("accepted_for_v509", False)),
            "paired_surface_close_to_v50r2": bool(v516_gates.get("paired_surface_pass", False)),
            "student_close_to_v50r2_floor": False,
            "no_teacher_copy": bool(v517_gates.get("no_teacher_copy", v514_gates.get("no_teacher_copy", False))),
            "local_fidelity_complete": False,
            "face_policy_honest": True,
            "environment_visible_with_student": bool(v517_gates.get("partial_environment_visible", False)),
            "anti_2d_proxy_pass": bool(v517_gates.get("anti_2d_proxy_pass", False)),
            "anti_2d_pass": False,
            "auto_evolve_required": True,
            "not_promoted": True
        },
        "decision": "Manual mentor gate fails closed. V517 is a model-owned full-scene candidate, but the visible human is still not as readable as V50R2; continue with canonical SMPL-X surfel/graph representation repair. Do not claim teacher-only, crop-only, metric-only, route-created-only, or external-hard-block success.",
        "blockers": [
            "V517 human-main full-scene candidate remains blob-like / limb-unreadable",
            "V517 visual morphology remains below V50R2 visual floor despite paired metric pass",
            "V510 local fidelity failed",
            "V511 anti-2D proxy cannot substitute for mentor-readable human morphology"
        ]
    }
    out = REPORTS / "V5120000000000000000000_manual_mentor_gate.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "manual_gate_json": str(out), "annotated_board": str(board)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
