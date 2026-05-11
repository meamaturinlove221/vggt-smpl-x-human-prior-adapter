from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify_reason_classes(dline: dict[str, Any], source_graph: dict[str, Any], v14_rollup: dict[str, Any]) -> list[str]:
    gate = dline.get("gate_results", {}) if isinstance(dline.get("gate_results"), dict) else {}
    classes: list[str] = []
    if not gate.get("k14_teacher_precheck_pass"):
        classes.append("K_PROTOCOL_FAIL")
    if not gate.get("g14_teacher_precheck_pass"):
        classes.append("G_PROTOCOL_FAIL")
    if gate.get("s14_supervision_ready"):
        classes.append("S_SUPERVISION_ONLY")
    if not gate.get("hand_ownership_ready"):
        classes.append("H_ASSET_FAIL")
    if not gate.get("hair_ownership_ready"):
        classes.append("R_ASSET_FAIL")
    if not gate.get("t14_canonical_teacher_ready"):
        classes.append("T_PRED_MISSING")
    if not gate.get("f14_body_head_face_ready"):
        classes.append("F_TEACHER_MISSING")
    eligible_regions = set(source_graph.get("promotion_eligible_regions", []))
    needed = {"body", "head", "face", "hair", "left_hand", "right_hand"}
    if not needed.issubset(eligible_regions):
        classes.append("UNIFIED_ILLEGAL")
    return sorted(set(classes))


def dispatch_for(classes: list[str]) -> dict[str, str]:
    dispatch: dict[str, str] = {}
    if "K_PROTOCOL_FAIL" in classes:
        dispatch["agent_k15"] = "Run Kinect visible-surface teacher audit, protocol-objective alignment, and raw-sensor fullbody/hand audit."
    if "G_PROTOCOL_FAIL" in classes:
        dispatch["agent_g15"] = "Run view contract remap, true 2DGS normal rasterizer, and Sapiens/2D-SuGaR-guided surface precheck."
    if "S_SUPERVISION_ONLY" in classes:
        dispatch["agent_s15"] = "Solve Sapiens normal convention and calibrate relative depth against K/G anchors; keep supervision-only."
    if "H_ASSET_FAIL" in classes:
        dispatch["agent_h15"] = "Acquire/probe WiLoR/HaMeR/OSX/SMPLer routes, detect MANO assets, and produce hand anchor readiness or asset request."
    if "R_ASSET_FAIL" in classes:
        dispatch["agent_r15"] = "Acquire/probe HairGS/GaussianHaircut, detect FLAME/hair inputs, and produce topology readiness or native fallback queue."
    if "T_PRED_MISSING" in classes:
        dispatch["agent_t15"] = "Generate or dispatch bounded research-only frame0001/frame0002 VGGT predictions and canonical teacher audit."
    if "F_TEACHER_MISSING" in classes:
        dispatch["agent_f15"] = "Prepare Fus3D region backend with K/G/S/T supervision, but do not claim full candidate without hand/hair ownership."
    if "UNIFIED_ILLEGAL" in classes:
        dispatch["agent_u15"] = "Hold unified merge until required ownership regions become promotion eligible; scan for forbidden/proxy parent use."
    dispatch["agent_d15"] = "Refresh source graph and rerun D-line promotion after branch artifacts land."
    return dispatch


def main() -> int:
    parser = argparse.ArgumentParser(description="Route V15 work after D-line blocks promotion.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_dline_router_queue.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_dline_router_status.md")
    args = parser.parse_args()
    dline = read_summary(REPORTS / "20260508_v14_dline_promotion_report.json")
    source_graph = read_summary(REPORTS / "20260508_v15_source_graph.json")
    v14_rollup = read_summary(REPORTS / "20260508_v14_execution_rollup.json")
    classes = classify_reason_classes(dline, source_graph, v14_rollup)
    dispatch = dispatch_for(classes)
    strict_candidate = int(dline.get("strict_candidate_passes", 0) or 0)
    strict_teacher = int(dline.get("strict_teacher_passes", 0) or 0)
    return_allowed = strict_candidate > 0 or strict_teacher > 0
    summary = {
        "task": "v15_dline_router",
        "created_utc": utc_now(),
        "status": "v15_router_dispatch_required" if not return_allowed else "v15_router_strict_pass_observed",
        "dline_status": dline.get("status"),
        "strict_candidate_passes": strict_candidate,
        "strict_teacher_passes": strict_teacher,
        "formal_cloud_unblocked": bool(dline.get("formal_cloud_unblocked")),
        "reason_classes": classes,
        "dispatch": dispatch,
        "return_allowed": return_allowed,
        "next_work_queue": [
            {"id": key, "task": value, "status": "pending"} for key, value in dispatch.items()
        ],
        "decision": (
            "D-line remains the strict judge. Because promotion is blocked, V15 router generated the next branch queue instead of treating blocked as terminal."
            if not return_allowed
            else "D-line strict pass observed; return is allowed."
        ),
        "blockers": dline.get("blockers", []),
    }
    write_json(args.output_json, summary)
    write_json(REPORTS / "20260508_v15_next_work_queue.json", summary)
    write_report(args.output_md, "V15 D-line Router", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
