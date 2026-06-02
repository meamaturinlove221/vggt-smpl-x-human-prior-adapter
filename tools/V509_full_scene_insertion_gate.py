from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"

V508_HASH = REPORTS / "V5080000000000000000000_hash_reconciliation.json"
V508_MODAL = REPORTS / "V5080000000000000000000_modal_smoke_result.json"
V508_A10G_4000 = REPORTS / "V5080000000000000000000_modal_a10g_4000_result.json"
V508_A100_4000 = REPORTS / "V5080000000000000000000_modal_a100_4000_result.json"
V514_DECISION = REPORTS / "V5140000000000000000000_checkpoint_adjudication_decision.json"
V514_BOARD = BOARDS / "V5140000000000000000000_checkpoint_adjudication_board.png"
V50R2_FLOOR = BOARDS / "V5020000000000000000000_v50r2_panel_annotated.png"
V503_REGRESSION = BOARDS / "V5030000000000000000000_student_vs_v50r2_contact_sheet.png"
RGB_SCENE = Path(r"D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_12views_tmf_v223_repaired\rgb_contact_sheet.png")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    if not path.is_file():
        draw.text((20, 20), f"missing: {path}", fill=(140, 0, 0), font=ImageFont.load_default())
        return canvas
    try:
        with Image.open(path).convert("RGB") as im:
            im.thumbnail((size[0] - 24, size[1] - 54), Image.Resampling.LANCZOS)
            canvas.paste(im, ((size[0] - im.width) // 2, 42 + (size[1] - 54 - im.height) // 2))
    except Exception:
        font = ImageFont.load_default()
        text = path.read_text(encoding="utf-8", errors="replace")[:900]
        y = 52
        draw.text((18, y), str(path.name), fill=(0, 0, 0), font=font)
        y += 28
        for line in text.replace("{", "").replace("}", "").splitlines()[:16]:
            draw.text((18, y), line[:82], fill=(60, 60, 60), font=font)
            y += 22
    return canvas


def make_board(output: Path, title: str, panels: list[tuple[str, Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    width, cell_w, cell_h = 1800, 600, 520
    board = Image.new("RGB", (width, 1120), "white")
    draw = ImageDraw.Draw(board)
    draw.text((16, 10), title, fill=(0, 0, 0), font=font)
    draw.text((16, 30), "Fail-closed: no teacher/raw RGB/V50R2 copy is inserted as final student.", fill=(140, 0, 0), font=font)
    for i, (label, path, note) in enumerate(panels[:6]):
        x = (i % 3) * cell_w
        y = 60 + (i // 3) * cell_h
        img = fit(path, (cell_w, cell_h - 24))
        board.paste(img, (x, y))
        draw.rectangle([x + 8, y + 8, x + cell_w - 8, y + cell_h - 28], outline=(160, 80, 80), width=2)
        draw.text((x + 16, y + 14), label, fill=(0, 0, 0), font=font)
        draw.text((x + 16, y + cell_h - 42), note, fill=(120, 0, 0), font=font)
    board.save(output)


def main() -> int:
    v508_hash = read_json(V508_HASH)
    v508_modal = read_json(V508_MODAL)
    v508_a10g_4000 = read_json(V508_A10G_4000)
    v508_a100_4000 = read_json(V508_A100_4000)
    v514 = read_json(V514_DECISION)
    a10g_checkpoints_complete = bool(
        v508_hash.get("modal_matrix_complete")
        and int(v508_hash.get("local_smoke_steps") or 0) >= 4000
        and v508_a10g_4000.get("gates", {}).get("a10g_checkpoints_complete", False)
    )
    a100_required = bool(v508_hash.get("a10_a100_required", False))
    a100_complete = bool(v508_a100_4000.get("gates", {}).get("a100_checkpoints_complete", False))
    v508_target_matrix_complete = bool(a10g_checkpoints_complete and (not a100_required or a100_complete))
    v514_accepted = bool(v514.get("gates", {}).get("accepted_model_owned_student", False))
    status = (
        "V509_FULL_SCENE_INSERTION_READY_FOR_STRICT_VISUAL_GATE_NOT_PROMOTED"
        if v514_accepted
        else
        "V509_FULL_SCENE_INSERTION_FAIL_CLOSED_TARGET_MATRIX_COMPLETE_ACCEPTED_STUDENT_PENDING_NOT_PROMOTED"
        if v508_target_matrix_complete
        else "V509_FULL_SCENE_INSERTION_FAIL_CLOSED_A10G_COMPLETE_A100_ACCEPTED_STUDENT_PENDING_NOT_PROMOTED"
    )
    matrix_note = "A10G/A100 complete" if v508_target_matrix_complete else "A10G complete, A100 pending"

    full_scene_board = BOARDS / "V5090000000000000000000_full_scene_student.png"
    controls_board = BOARDS / "V5090000000000000000000_same_scene_controls.png"
    decision_json = REPORTS / "V5090000000000000000000_full_scene_decision.json"

    make_board(
        full_scene_board,
        "V509 full-scene insertion gate: failed closed, no accepted model-owned student yet",
        [
            ("V50R2 visual floor", V50R2_FLOOR, "reference only"),
            ("Full-scene RGB observation", RGB_SCENE, "environment source only"),
            ("Current-student regression", V503_REGRESSION, "student below V50R2"),
            ("V508 A10G 4000", V508_A10G_4000, matrix_note),
            ("V508 A100 4000", V508_A100_4000, "target matrix evidence"),
            ("V514 adjudication", V514_BOARD, "candidate failed controls" if not v514_accepted else "candidate handoff"),
        ],
    )
    make_board(
        controls_board,
        "V509 same-scene controls gate: not generated as pass evidence",
        [
            ("V50R2 visual floor", V50R2_FLOOR, "reference only"),
            ("Current controls/regression", V503_REGRESSION, "controls cannot override visual floor"),
            ("V514 controls adjudication", V514_BOARD, "true not better than controls" if not v514_accepted else "requires V512"),
        ],
    )

    payload = {
        "task": "V509_full_scene_insertion_gate",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "full_scene_student_board": str(full_scene_board),
        "same_scene_controls_board": str(controls_board),
        "input_v508_hash": str(V508_HASH),
        "input_v508_modal": str(V508_MODAL),
        "input_v508_a10g_4000": str(V508_A10G_4000),
        "input_v508_a100_4000": str(V508_A100_4000),
        "input_v514_adjudication": str(V514_DECISION),
        "v508_status": v508_hash.get("status"),
        "v508_modal_status": v508_modal.get("status"),
        "v508_a10g_4000_status": v508_a10g_4000.get("status"),
        "v508_a100_4000_status": v508_a100_4000.get("status"),
        "v514_status": v514.get("status"),
        "gates": {
            "model_owned_student_checkpoint_ready": v514_accepted,
            "v508_a10g_checkpoints_complete": a10g_checkpoints_complete,
            "v508_target_checkpoints_complete": v508_target_matrix_complete,
            "modal_a10g_smoke_pass": v508_modal.get("gates", {}).get("modal_a10g_smoke_pass", False),
            "modal_a100_target_complete": a100_complete,
            "v514_no_teacher_copy": v514.get("gates", {}).get("no_teacher_copy", False),
            "v514_true_improves_vggt_baseline": v514.get("gates", {}).get("true_improves_vggt_baseline", False),
            "v514_true_improves_no_smpl": v514.get("gates", {}).get("true_improves_no_smpl", False),
            "v514_true_improves_shuffled_semantic": v514.get("gates", {}).get("true_improves_shuffled_semantic", False),
            "full_scene_student_inserted": v514_accepted,
            "same_scene_controls_generated_as_pass_evidence": v514_accepted,
            "teacher_or_rgb_copy_used_as_student": False,
            "mentor_visual_gate_pass": False,
            "not_promoted": True,
            "auto_evolve_required": not v514_accepted,
        },
        "decision": (
            "Do not generate fake insertion from V50R2 or raw RGB. Full-scene insertion must wait for a model-owned "
            "student checkpoint from the V508 target matrix that passes V505 copy firewall and V50R2 non-regression. "
            "Continue repair/training; this is not an external hard block."
        ),
        "blockers": [
            "V514 checkpoint adjudication produced model-owned full-scene PLYs, but true did not beat VGGT/no-SMPL/shuffled controls under the V50R2 floor",
            "No accepted model-owned student checkpoint is available for V509 mentor insertion"
        ] if not v514_accepted else [],
    }
    decision_json.parent.mkdir(parents=True, exist_ok=True)
    decision_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "full_scene_board": str(full_scene_board), "controls_board": str(controls_board), "decision_json": str(decision_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
