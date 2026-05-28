from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    created_at = now()
    payload = {
        "created_at": created_at,
        "status": "V10230_CURRENT_ROUTE_STATE_ACTIVE_NOT_COMPLETE",
        "active_parent_goal": "V950100000000000000_to_V3000000000000000000_visual_supervised_residual",
        "latest_user_correction": "Current boards looked too 2D; face detail remains not applicable.",
        "current_findings": [
            {
                "stage": "V10170",
                "finding": "V10150 board renderer was effectively 2D/orthographic because it used points[:, :2].",
                "evidence": str(REPORTS / "V10170000000000000000_flatness_and_depth_render_decision.json"),
            },
            {
                "stage": "V10180",
                "finding": "Render repair is necessary but cannot be final; candidate geometry was not meaningfully more 3D than baseline.",
                "evidence": str(REPORTS / "V10180000000000000000_depth_cue_and_geometry_repair_decision.json"),
            },
            {
                "stage": "V10190",
                "finding": "Thickness-aware candidate improved thickness over baseline internally.",
                "evidence": str(REPORTS / "V10190000000000000000_thickness_aware_geometry_candidate_decision.json"),
            },
            {
                "stage": "V10200",
                "finding": "Thickness-only gate failed closed because shuffled control had stronger thickness; need topology/part-continuity/control-separation training.",
                "evidence": str(REPORTS / "V10200000000000000000_thickness_geometry_visual_gate_decision.json"),
            },
            {
                "stage": "V10210",
                "finding": "True 3D geometry training payload assembled for four eligible cases.",
                "evidence": str(REPORTS / "V10210000000000000000_true_3d_geometry_training_payload_decision.json"),
            },
            {
                "stage": "V10220",
                "finding": "Canonical surfel/graph student smoke passed with teacher-key rejection; not a mentor visual pass.",
                "evidence": str(REPORTS / "V10220000000000000000_true_3d_geometry_model_smoke.json"),
            },
        ],
        "not_complete_reasons": [
            "No trained V10230 multi-case student predictions yet.",
            "No new full-scene mentor board from trained model yet.",
            "No hard-control visual separation gate pass yet.",
            "No upload-safe bundles for this route yet.",
            "Dirty worktree remains.",
        ],
        "next_action": "Build and run V10230 tiny/multicase training execution, then generate oblique human-main full-scene boards and hard-control gates.",
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
        "no_agent_used": True,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "V10230000000000000000_current_route_state.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
