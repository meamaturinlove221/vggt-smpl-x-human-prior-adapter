from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"

V50R2_FLOOR = BOARDS / "V5020000000000000000000_v50r2_panel_annotated.png"
V910_STATUS = REPORTS / "V9102000000000000000000_final_reconciliation_status.json"

CANDIDATES = [
    {
        "candidate": "V352_human_main_full_scene",
        "path": BOARDS / "V352000000000000000000_human_main_full_scene.png",
        "source_status": "previous_best_student_board",
        "human_roi_quality_vs_v50r2": "major_regression",
        "rgb_realism_vs_v50r2": "weaker_fragmented",
        "head_hair_vs_v50r2": "weaker",
        "clothing_vs_v50r2": "weaker",
        "arm_hand_vs_v50r2": "weaker",
        "leg_foot_vs_v50r2": "weaker",
        "anti_2d_vs_v50r2": "not_sufficient",
        "speckle_hole_vs_v50r2": "worse",
        "silhouette_continuity_vs_v50r2": "worse",
        "mentor_ready_under_v50r2_floor": False,
        "route_action": "send_to_distillation_repair",
        "notes": "Human is not upright/readable like V50R2; side-lying/fragmented presentation is a morphology regression.",
    },
    {
        "candidate": "V353_manual_gate_assembly",
        "path": BOARDS / "V353000000000000000000_manual_gate_annotated.png",
        "source_status": "previous_manual_gate_board",
        "human_roi_quality_vs_v50r2": "major_regression",
        "rgb_realism_vs_v50r2": "weaker_fragmented",
        "head_hair_vs_v50r2": "weaker",
        "clothing_vs_v50r2": "weaker",
        "arm_hand_vs_v50r2": "weaker",
        "leg_foot_vs_v50r2": "weaker",
        "anti_2d_vs_v50r2": "not_sufficient",
        "speckle_hole_vs_v50r2": "worse",
        "silhouette_continuity_vs_v50r2": "worse",
        "mentor_ready_under_v50r2_floor": False,
        "route_action": "send_to_distillation_repair",
        "notes": "Contains controls and artifact narrative, but the true panel remains visually below V50R2 human morphology floor.",
    },
    {
        "candidate": "V300_target_isolated_topology_surfel_scene",
        "path": BOARDS / "V300000000000000000000_human_main_full_scene.png",
        "source_status": "V300_series_present",
        "human_roi_quality_vs_v50r2": "major_regression",
        "rgb_realism_vs_v50r2": "partial_but_pose_orientation_regressed",
        "head_hair_vs_v50r2": "mixed_weaker_overall",
        "clothing_vs_v50r2": "mixed_but_less_mentor_readable",
        "arm_hand_vs_v50r2": "weaker",
        "leg_foot_vs_v50r2": "weaker",
        "anti_2d_vs_v50r2": "not_sufficient",
        "speckle_hole_vs_v50r2": "worse",
        "silhouette_continuity_vs_v50r2": "worse",
        "mentor_ready_under_v50r2_floor": False,
        "route_action": "send_to_distillation_repair",
        "notes": "Contains recognizable RGB human evidence, but orientation/scene presentation is less mentor-readable than V50R2 standing panels.",
    },
    {
        "candidate": "V910_control_safe_subject_clarity",
        "path": BOARDS / "V910000000000000000000_human_main_full_scene.png",
        "source_status": "V910_series_present_previous_status_only",
        "human_roi_quality_vs_v50r2": "major_regression",
        "rgb_realism_vs_v50r2": "weaker_fragmented",
        "head_hair_vs_v50r2": "weaker",
        "clothing_vs_v50r2": "weaker",
        "arm_hand_vs_v50r2": "weaker",
        "leg_foot_vs_v50r2": "weaker",
        "anti_2d_vs_v50r2": "not_sufficient",
        "speckle_hole_vs_v50r2": "worse",
        "silhouette_continuity_vs_v50r2": "worse",
        "mentor_ready_under_v50r2_floor": False,
        "route_action": "send_to_distillation_repair",
        "notes": "Under the new V50R2 non-regression guard, prior V910 ready status is historical only and cannot pass.",
    },
    {
        "candidate": "V900_series",
        "path": "",
        "source_status": "not_found_in_current_boards_scan",
        "human_roi_quality_vs_v50r2": "missing",
        "rgb_realism_vs_v50r2": "missing",
        "head_hair_vs_v50r2": "missing",
        "clothing_vs_v50r2": "missing",
        "arm_hand_vs_v50r2": "missing",
        "leg_foot_vs_v50r2": "missing",
        "anti_2d_vs_v50r2": "missing",
        "speckle_hole_vs_v50r2": "missing",
        "silhouette_continuity_vs_v50r2": "missing",
        "mentor_ready_under_v50r2_floor": False,
        "route_action": "do_not_use_as_pass_signal",
        "notes": "No V900 board was found in the current boards directory during this audit.",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fit(path: Path, box: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", box, "white")
    if not path.is_file():
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), f"missing: {path}", fill=(140, 0, 0), font=ImageFont.load_default())
        return canvas
    with Image.open(path).convert("RGB") as im:
        im.thumbnail((box[0] - 24, box[1] - 52), Image.Resampling.LANCZOS)
        x = (box[0] - im.width) // 2
        y = 42 + (box[1] - 52 - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def make_contact_sheet(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    width = 1800
    with Image.open(V50R2_FLOOR).convert("RGB") as floor:
        floor.thumbnail((width, 820), Image.Resampling.LANCZOS)
        top = Image.new("RGB", (width, floor.height + 66), "white")
        d = ImageDraw.Draw(top)
        d.text((14, 10), "V50R2 visual floor: standing, multi-view, RGB human morphology baseline", fill=(0, 0, 0), font=font)
        d.text((14, 30), "New guard: candidate human ROI must not visibly regress below this floor.", fill=(120, 0, 0), font=font)
        top.paste(floor, ((width - floor.width) // 2, 58))

    cell_w, cell_h = 900, 560
    grid = Image.new("RGB", (width, cell_h * 2 + 40), "white")
    d = ImageDraw.Draw(grid)
    shown = [c for c in CANDIDATES if c["path"]][:4]
    for i, cand in enumerate(shown):
        x = (i % 2) * cell_w
        y = (i // 2) * cell_h + 28
        img = fit(Path(cand["path"]), (cell_w, cell_h - 10))
        grid.paste(img, (x, y))
        d.rectangle([x + 8, y + 8, x + cell_w - 8, y + cell_h - 18], outline=(190, 80, 80), width=2)
        d.text((x + 18, y + 14), f'{cand["candidate"]}: V50R2 regression -> {cand["human_roi_quality_vs_v50r2"]}', fill=(120, 0, 0), font=font)
    d.text((14, 8), "Current student/main boards under V50R2 regression audit", fill=(0, 0, 0), font=font)

    board = Image.new("RGB", (width, top.height + grid.height + 18), "white")
    board.paste(top, (0, 0))
    board.paste(grid, (0, top.height + 18))
    board.save(output)


def previous_status() -> dict[str, Any]:
    if not V910_STATUS.is_file():
        return {"status": None, "exists": False}
    data = json.loads(V910_STATUS.read_text(encoding="utf-8"))
    return {"status": data.get("status"), "mentor_ready": data.get("mentor_ready"), "exists": True, "path": str(V910_STATUS)}


def main() -> int:
    audit_csv = REPORTS / "V5030000000000000000000_student_vs_v50r2_regression_audit.csv"
    board_png = BOARDS / "V5030000000000000000000_student_vs_v50r2_contact_sheet.png"
    decision_json = REPORTS / "V5030000000000000000000_regression_decision.json"

    write_csv(audit_csv, CANDIDATES)
    make_contact_sheet(board_png)

    payload = {
        "task": "V503_student_vs_v50r2_regression_audit",
        "status": "V503_STUDENT_VS_V50R2_REGRESSION_FAIL_CLOSED_TO_DISTILLATION_REPAIR_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "audit_csv": str(audit_csv),
        "contact_sheet": str(board_png),
        "previous_status": previous_status(),
        "gates": {
            "v50r2_visual_floor_established": True,
            "student_meets_or_exceeds_v50r2_floor": False,
            "human_roi_quality_regression_found": True,
            "rgb_observation_fidelity_regression_found": True,
            "controls_or_report_can_override_visual_floor": False,
            "mentor_ready_allowed_under_new_guard": False,
            "enter_distillation_repair": True,
            "external_hard_block": False,
            "not_promoted": True,
        },
        "failed_candidates": [c["candidate"] for c in CANDIDATES if not c["mentor_ready_under_v50r2_floor"]],
        "decision": (
            "Current student/main-scene boards are visibly weaker than the V50R2 human morphology floor. "
            "The previous V910 ready state is retained only as historical previous_status, not as a pass under the new V50R2 guard. "
            "Fail closed and continue into V504 teacher bank, V505 firewall, and V506 observation-distilled student repair."
        ),
        "blockers": [],
    }
    write_json(decision_json, payload)
    print(json.dumps({"status": payload["status"], "audit_csv": str(audit_csv), "contact_sheet": str(board_png), "decision_json": str(decision_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
