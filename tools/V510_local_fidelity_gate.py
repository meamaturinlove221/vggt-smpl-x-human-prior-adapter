from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
V504_REGIONS = BOARDS / "V5040000000000000000000_teacher_bank_regions.png"
V503_REGRESSION = BOARDS / "V5030000000000000000000_student_vs_v50r2_contact_sheet.png"
V509_DECISION = REPORTS / "V5090000000000000000000_full_scene_decision.json"
V517_DECISION = REPORTS / "V5170000000000000000000_full_scene_clarity_decision.json"
V517_MAIN = BOARDS / "V5170000000000000000000_human_main_full_scene.png"

REGIONS = [
    ("head_hair", "head/hair vs V50R2", "V5100000000000000000000_head_hair_fidelity.png"),
    ("shoulder_neck", "shoulder/neck vs V50R2", "V5100000000000000000000_shoulder_neck_fidelity.png"),
    ("clothing_boundary", "clothing boundary vs V50R2", "V5100000000000000000000_clothing_boundary_fidelity.png"),
    ("hand_arm", "hand/arm vs V50R2", "V5100000000000000000000_hand_arm_fidelity.png"),
    ("leg_foot", "leg/foot vs V50R2", "V5100000000000000000000_leg_foot_fidelity.png"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if not path.is_file():
        ImageDraw.Draw(canvas).text((20, 20), f"missing: {path}", fill=(140, 0, 0), font=ImageFont.load_default())
        return canvas
    with Image.open(path).convert("RGB") as im:
        im.thumbnail((size[0] - 32, size[1] - 70), Image.Resampling.LANCZOS)
        canvas.paste(im, ((size[0] - im.width) // 2, 48 + (size[1] - 70 - im.height) // 2))
    return canvas


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def make_region_board(region_key: str, title: str, output: Path, candidate_ready: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    width, height = 1600, 820
    board = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), f"V510 local fidelity gate: {title}", fill=(0, 0, 0), font=font)
    draw.text((16, 32), "Fail-closed: V50R2 teacher floor exists; V517 candidate is not locally body-part readable enough.", fill=(140, 0, 0), font=font)
    left = fit(V504_REGIONS, (800, 700))
    right = fit(V517_MAIN if candidate_ready else V503_REGRESSION, (800, 700))
    board.paste(left, (0, 70))
    board.paste(right, (800, 70))
    draw.rectangle([8, 76, 792, 770], outline=(80, 120, 80), width=2)
    draw.rectangle([808, 76, 1592, 770], outline=(160, 80, 80), width=2)
    draw.text((22, 88), f"V50R2 teacher/reference region source: {region_key}", fill=(0, 80, 0), font=font)
    draw.text((822, 88), "V517 candidate: full-scene exists, local limbs/head are still not robustly readable", fill=(140, 0, 0), font=font)
    draw.text((16, 790), "Result: local fidelity not pass; switch to canonical SMPL-X surfel/graph representation repair before claiming this region.", fill=(0, 0, 0), font=font)
    board.save(output)


def main() -> int:
    v517 = read_json(V517_DECISION)
    candidate_ready = bool(v517.get("gates", {}).get("accepted_for_v509", False))
    outputs = {}
    for region_key, title, filename in REGIONS:
        path = BOARDS / filename
        make_region_board(region_key, title, path, candidate_ready)
        outputs[region_key] = str(path)

    decision = {
        "task": "V510_local_fidelity_gate",
        "status": "V510_LOCAL_FIDELITY_FAIL_CLOSED_V517_LOCAL_MORPHOLOGY_BELOW_V50R2_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "input_v509_decision": str(V509_DECISION),
        "input_v517_decision": str(V517_DECISION),
        "v517_status": v517.get("status"),
        "boards": outputs,
        "gates": {
            "head_hair_fidelity_pass": False,
            "shoulder_neck_fidelity_pass": False,
            "clothing_boundary_fidelity_pass": False,
            "hand_arm_fidelity_pass": False,
            "leg_foot_fidelity_pass": False,
            "face_visible_subset_honest": True,
            "v517_candidate_available_for_review": candidate_ready,
            "accepted_student_available": False,
            "teacher_floor_available": True,
            "not_promoted": True,
            "auto_evolve_required": True
        },
        "decision": "V517 provides a model-owned full-scene candidate, but local head/hair, shoulder/neck, clothing, hand/arm, and leg/foot readability is still below the V50R2 visual floor. Fail closed and continue with canonical surfel/graph representation repair.",
        "blockers": [
            "V517 full-scene candidate remains blob-like at local body-part level",
            "V50R2 local morphology is not matched in the mentor-facing view"
        ]
    }
    out = REPORTS / "V5100000000000000000000_local_fidelity_decision.json"
    out.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "decision_json": str(out), "boards": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
