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
V501_DECISION = REPORTS / "V5010000000000000000000_v50r2_visual_floor_decision.json"

POINTCLOUD_SHEET = Path(
    r"D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud"
    r"\images\V223_V50R2_full_body_pointcloud_v42_consistent.png"
)


ROWS = [
    {
        "camera_id": "cam00",
        "orientation": "back",
        "body_readability": "strong",
        "head_hair_contour": "strong",
        "shoulder_neck": "strong",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "partially_visible_but_coherent",
        "leg_foot": "strong",
        "rgb_realism": "strong",
        "speckle_hole": "minor_speckle_not_dominant",
        "view_consistency": "strong",
        "face_visibility_category": "back_view_head_hair_contour_only",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Back-view full body is upright and readable; head, hair, shoulders, shirt, shorts, legs, and shoes remain continuous.",
    },
    {
        "camera_id": "cam01",
        "orientation": "back",
        "body_readability": "strong",
        "head_hair_contour": "strong",
        "shoulder_neck": "strong",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "partially_visible_but_coherent",
        "leg_foot": "strong",
        "rgb_realism": "strong",
        "speckle_hole": "minor_speckle_not_dominant",
        "view_consistency": "strong",
        "face_visibility_category": "back_view_head_hair_contour_only",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Cleanest back panel; body proportions and clothing boundary are natural enough to serve as morphology floor.",
    },
    {
        "camera_id": "cam06",
        "orientation": "side",
        "body_readability": "strong",
        "head_hair_contour": "strong",
        "shoulder_neck": "good",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "good",
        "leg_foot": "good",
        "rgb_realism": "strong",
        "speckle_hole": "small_floating_speckles",
        "view_consistency": "strong",
        "face_visibility_category": "side_view_profile_region_only",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Side profile keeps standing posture, torso thickness, phone/hand pose, shorts, legs, and shoes readable.",
    },
    {
        "camera_id": "cam11",
        "orientation": "side",
        "body_readability": "strong",
        "head_hair_contour": "strong",
        "shoulder_neck": "good",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "good",
        "leg_foot": "strong",
        "rgb_realism": "strong",
        "speckle_hole": "small_floating_speckles",
        "view_consistency": "strong",
        "face_visibility_category": "side_view_profile_region_only",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Side silhouette is upright, not a sheet; legs and shoes remain connected, with visible hand/phone endpoint.",
    },
    {
        "camera_id": "cam16",
        "orientation": "side-oblique",
        "body_readability": "strong",
        "head_hair_contour": "strong",
        "shoulder_neck": "good",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "good",
        "leg_foot": "strong",
        "rgb_realism": "strong",
        "speckle_hole": "minor_speckle_not_dominant",
        "view_consistency": "strong",
        "face_visibility_category": "side_oblique_face_region_shape_only",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Side-oblique view exposes head/hair, jaw-side region, arm, torso, shorts, legs, and shoes without template-shell dominance.",
    },
    {
        "camera_id": "cam21",
        "orientation": "front-oblique",
        "body_readability": "strong",
        "head_hair_contour": "good",
        "shoulder_neck": "good",
        "torso_clothing": "strong",
        "arm_hand_endpoint": "good",
        "leg_foot": "good",
        "rgb_realism": "strong",
        "speckle_hole": "minor_speckle_not_dominant",
        "view_consistency": "strong",
        "face_visibility_category": "front_oblique_face_region_shape_only_no_fine_detail",
        "face_detail_claim_allowed": False,
        "visual_floor_pass": True,
        "notes": "Front-oblique body is readable and useful for the floor, but only face-region shape is allowed; eyes/nose/mouth detail is not claimed.",
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


def make_annotated_board(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    with Image.open(POINTCLOUD_SHEET).convert("RGB") as src:
        src.thumbnail((1900, 1120), Image.Resampling.LANCZOS)
        board = Image.new("RGB", (1900, src.height + 260), "white")
        board.paste(src, ((1900 - src.width) // 2, 34))
    draw = ImageDraw.Draw(board)
    draw.text((16, 10), "V502 panel-level visual audit: V50R2 is ROI visual floor only, not final full-scene student", fill=(0, 0, 0), font=font)
    y = src.height + 54
    cols = [
        ("cam00", "back: head/hair only; body floor pass"),
        ("cam01", "back: strongest full-body floor"),
        ("cam06", "side: profile/hand/leg readable"),
        ("cam11", "side: upright, not sheet"),
        ("cam16", "side-oblique: face region only"),
        ("cam21", "front-oblique: no fine face detail claim"),
    ]
    col_w = 316
    for i, (cam, txt) in enumerate(cols):
        x = i * col_w + 10
        draw.rectangle([x, y, x + col_w - 18, y + 140], outline=(80, 120, 80), width=2)
        draw.text((x + 8, y + 10), cam, fill=(0, 0, 0), font=font)
        draw.text((x + 8, y + 34), txt, fill=(0, 0, 0), font=font)
        draw.text((x + 8, y + 74), "visual_floor_pass=true", fill=(0, 90, 0), font=font)
        draw.text((x + 8, y + 100), "face_detail_claim=false", fill=(130, 60, 0), font=font)
    draw.text(
        (16, y + 168),
        "Gate note: this passes the V50R2 human morphology floor; it does not satisfy final mentor full-scene evidence by itself.",
        fill=(0, 0, 0),
        font=font,
    )
    board.save(output)


def main() -> int:
    v501 = json.loads(V501_DECISION.read_text(encoding="utf-8"))
    audit_csv = REPORTS / "V5020000000000000000000_v50r2_panel_audit.csv"
    decision_json = REPORTS / "V5020000000000000000000_v50r2_panel_decision.json"
    board_png = BOARDS / "V5020000000000000000000_v50r2_panel_annotated.png"

    write_csv(audit_csv, ROWS)
    make_annotated_board(board_png)

    payload = {
        "task": "V502_v50r2_panel_visual_audit",
        "status": "V502_V50R2_PANEL_AUDIT_COMPLETE_VISUAL_FLOOR_PASS_CONTINUE_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "input_v501_decision": str(V501_DECISION),
        "source_visual_floor_png": v501["source_visual_floor_png"],
        "audit_csv": str(audit_csv),
        "annotated_board": str(board_png),
        "gates": {
            "all_six_panels_present": True,
            "body_readability_floor_pass": all(r["visual_floor_pass"] for r in ROWS),
            "head_hair_floor_pass": True,
            "torso_clothing_floor_pass": True,
            "arm_hand_endpoint_floor_pass": True,
            "leg_foot_floor_pass": True,
            "rgb_realism_floor_pass": True,
            "view_consistency_floor_pass": True,
            "face_policy_honest": True,
            "face_detail_claim_allowed": False,
            "full_scene_final_ready": False,
            "not_promoted": True,
        },
        "face_policy": {
            "back_views": "head/hair contour only",
            "side_views": "side profile / jaw-side / hairline region only",
            "front_oblique": "face region shape only; no eyes/nose/mouth fine detail claim",
        },
        "decision": (
            "V50R2 passes as the human morphology visual floor across the six requested panels. "
            "It remains ROI/white-background reference evidence, not final full-scene student output. "
            "Continue to V503 student-vs-V50R2 regression audit."
        ),
        "blockers": [],
    }
    write_json(decision_json, payload)
    print(json.dumps({"status": payload["status"], "audit_csv": str(audit_csv), "annotated_board": str(board_png), "decision_json": str(decision_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
