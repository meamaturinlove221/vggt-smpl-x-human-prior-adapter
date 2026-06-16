from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT / "output/surface_research_preflight_local"
REPORTS = REPO_ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    hand = load_json(LOCAL_ROOT / "V11_HHand_B_vggt_decoder/summary.json")
    hair = load_json(LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian/summary.json")
    g3 = load_json(LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/summary.json")
    dline = load_json(LOCAL_ROOT / "DLine_V11_promotion_transaction/summary.json")
    hand_positive = bool(hand.get("bounded_overfit_positive"))
    hand_ownership = bool(hand.get("ownership_pass"))
    hair_positive = bool(hair.get("bounded_overfit_positive"))
    hair_ownership = bool(hair.get("ownership_pass"))
    dline_blocked = int(dline.get("strict_candidate_passes", 0) or 0) == 0 and int(dline.get("strict_teacher_passes", 0) or 0) == 0
    trigger = bool(g3.get("anchor_surface_precheck_pass") and dline_blocked and (not hand_ownership or not hair_ownership))
    summary = {
        "task": "v11_to_v12_tmf_trigger_audit",
        "status": "v12_tmf_recommended_after_v11_fail_closed" if trigger else "v11_continue_without_v12_trigger",
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "inputs": {
            "g3_anchor_pass": bool(g3.get("anchor_surface_precheck_pass")),
            "hand_bounded_positive": hand_positive,
            "hand_ownership_pass": hand_ownership,
            "hair_bounded_positive": hair_positive,
            "hair_ownership_pass": hair_ownership,
            "dline_blocked": dline_blocked,
        },
        "evidence": {
            "hand_metrics": hand.get("metrics"),
            "hand_controls": hand.get("controls"),
            "hair_metrics": hair.get("metrics"),
            "hair_controls": hair.get("controls"),
            "dline_blockers": dline.get("blockers"),
        },
        "v12_tmf_scope": [
            "temporal adjacent frames for same identity",
            "SMPL-X/4K4D pose canonicalization to shared canonical pose",
            "multi-frame hand/hair visibility fusion",
            "surface teacher back-projection to target frame",
        ],
        "decision": (
            "Do not promote V11. Keep G3 anchor; use V12-TMF only after V11 hard gaps are confirmed by the current D-line block."
            if trigger
            else "V11 still has a path without V12 trigger."
        ),
    }
    out = LOCAL_ROOT / "V11_to_V12_TMF_trigger_audit"
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v11_to_v12_tmf_trigger_audit.json", summary)
    lines = [
        "# V11 to V12-TMF Trigger Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        summary["decision"],
        "",
        "```json",
        json.dumps(summary["inputs"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (REPORTS / "20260508_v11_to_v12_tmf_trigger_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(out)}, ensure_ascii=False))
    return 0 if trigger else 2


if __name__ == "__main__":
    raise SystemExit(main())
