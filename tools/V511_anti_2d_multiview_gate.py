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
V517_DECISION = REPORTS / "V5170000000000000000000_full_scene_clarity_decision.json"
V517_ANTI2D = BOARDS / "V5170000000000000000000_turntable_side_depth_cross_section.png"


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def main() -> int:
    v517 = read_json(V517_DECISION)
    v517_candidate_ready = bool(v517.get("gates", {}).get("accepted_for_v509", False))
    v517_anti2d_proxy_pass = bool(v517.get("gates", {}).get("anti_2d_proxy_pass", False))
    board_path = BOARDS / "V5110000000000000000000_turntable_side_depth_cross_section.png"
    decision_path = REPORTS / "V5110000000000000000000_anti_2d_decision.json"
    font = ImageFont.load_default()
    board = Image.new("RGB", (1700, 980), "white")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), "V511 anti-2D / multiview gate: proxy pass, manual visual fail closed", fill=(0, 0, 0), font=font)
    draw.text((16, 34), "V517 generated turntable/side-depth/cross-section, but the human is still not mentor-readable enough.", fill=(140, 0, 0), font=font)
    board.paste(fit(V517_ANTI2D if v517_candidate_ready else V509_BOARD, (850, 820)), (0, 70))
    board.paste(fit(V503_BOARD, (850, 820)), (850, 70))
    draw.rectangle([8, 78, 842, 880], outline=(160, 80, 80), width=2)
    draw.rectangle([858, 78, 1692, 880], outline=(160, 80, 80), width=2)
    draw.text((24, 90), "V517 anti-2D proxy: generated, but not final mentor pass", fill=(140, 0, 0), font=font)
    draw.text((874, 90), "V503 regression: old boards below V50R2", fill=(140, 0, 0), font=font)
    draw.text((16, 914), "Result: proxy anti-2D evidence exists; manual mentor gate still fails because morphology remains below V50R2 visual floor.", fill=(0, 0, 0), font=font)
    board.save(board_path)

    payload = {
        "task": "V511_anti_2d_multiview_gate",
        "status": "V511_ANTI_2D_PROXY_PASS_MANUAL_VISUAL_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "input_v509_decision": str(V509_DECISION),
        "input_v517_decision": str(V517_DECISION),
        "v517_status": v517.get("status"),
        "board": str(board_path),
        "gates": {
            "turntable_generated": v517_candidate_ready,
            "side_depth_generated": v517_candidate_ready,
            "cross_section_generated": v517_candidate_ready,
            "anti_2d_proxy_pass": v517_anti2d_proxy_pass,
            "anti_2d_pass": False,
            "full_scene_is_not_crop_only": v517_candidate_ready,
            "accepted_student_available": False,
            "not_promoted": True,
            "auto_evolve_required": True
        },
        "decision": "V517 provides anti-2D proxy renders and a full-scene candidate, but manual visual inspection still finds body morphology insufficient. Anti-2D cannot be used as replacement for mentor-readable human shape.",
        "blockers": [
            "V517 human remains blob-like / locally unreadable despite non-flat proxy thickness",
            "Representation repair is required before mentor pass"
        ]
    }
    decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "board": str(board_path), "decision_json": str(decision_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
