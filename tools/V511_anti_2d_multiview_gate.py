from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
V509_BOARD = BOARDS / "V5090000000000000000000_full_scene_student.png"
V503_BOARD = BOARDS / "V5030000000000000000000_student_vs_v50r2_contact_sheet.png"
V509_DECISION = REPORTS / "V5090000000000000000000_full_scene_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if path.is_file():
        with Image.open(path).convert("RGB") as im:
            im.thumbnail((size[0] - 24, size[1] - 60), Image.Resampling.LANCZOS)
            canvas.paste(im, ((size[0] - im.width) // 2, 42 + (size[1] - 60 - im.height) // 2))
    else:
        ImageDraw.Draw(canvas).text((20, 20), f"missing: {path}", fill=(140, 0, 0), font=ImageFont.load_default())
    return canvas


def main() -> int:
    board_path = BOARDS / "V5110000000000000000000_turntable_side_depth_cross_section.png"
    decision_path = REPORTS / "V5110000000000000000000_anti_2d_decision.json"
    font = ImageFont.load_default()
    board = Image.new("RGB", (1700, 980), "white")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), "V511 anti-2D / multiview gate: fail closed", fill=(0, 0, 0), font=font)
    draw.text((16, 34), "No accepted full-scene model-owned student exists; turntable/side-depth/cross-section cannot be passed.", fill=(140, 0, 0), font=font)
    board.paste(fit(V509_BOARD, (850, 820)), (0, 70))
    board.paste(fit(V503_BOARD, (850, 820)), (850, 70))
    draw.rectangle([8, 78, 842, 880], outline=(160, 80, 80), width=2)
    draw.rectangle([858, 78, 1692, 880], outline=(160, 80, 80), width=2)
    draw.text((24, 90), "V509 insertion failed: no accepted student", fill=(140, 0, 0), font=font)
    draw.text((874, 90), "V503 regression: old boards below V50R2", fill=(140, 0, 0), font=font)
    draw.text((16, 914), "Result: anti-2D gate cannot pass until a V508 target-matrix student is inserted and rendered in side-depth/cross-section.", fill=(0, 0, 0), font=font)
    board.save(board_path)

    payload = {
        "task": "V511_anti_2d_multiview_gate",
        "status": "V511_ANTI_2D_MULTIVIEW_FAIL_CLOSED_NO_ACCEPTED_STUDENT_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "input_v509_decision": str(V509_DECISION),
        "board": str(board_path),
        "gates": {
            "turntable_generated": False,
            "side_depth_generated": False,
            "cross_section_generated": False,
            "anti_2d_pass": False,
            "full_scene_is_not_crop_only": False,
            "accepted_student_available": False,
            "not_promoted": True,
            "auto_evolve_required": True
        },
        "decision": "Anti-2D/multiview gate fails because no accepted model-owned student checkpoint is available for full-scene rendering. Continue V508 target matrix and V506 repair.",
        "blockers": [
            "No accepted V508 student checkpoint",
            "V509 full-scene insertion failed closed"
        ]
    }
    decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "board": str(board_path), "decision_json": str(decision_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
