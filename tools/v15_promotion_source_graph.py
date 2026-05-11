from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


REGIONS = ("body", "head", "face", "hair", "left_hand", "right_hand", "cloth", "none")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def first_summary(*paths: Path) -> dict[str, Any]:
    for path in paths:
        data = read_summary(path)
        if data:
            return data
    return {}


def artifact(
    name: str,
    *,
    source_type: str,
    ownership_region: str,
    path: str | None,
    promotion_eligible: bool,
    reason: str,
    status: str | None = None,
) -> dict[str, Any]:
    if ownership_region not in REGIONS:
        raise ValueError(f"unknown ownership region {ownership_region!r}")
    return {
        "name": name,
        "source_type": source_type,
        "ownership_region": ownership_region,
        "path": path,
        "promotion_eligible": bool(promotion_eligible),
        "status": status,
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V15 promotion source graph from V14/V15 reports.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_source_graph.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_source_graph.md")
    args = parser.parse_args()

    truth = read_summary(REPORTS / "20260508_v14_truth_registry.json")
    k14 = first_summary(
        REPORTS / "V14_K14/20260508_k14_kinect_teacher_gate_autopsy.json",
        REPORTS / "20260508_v14_kinect_teacher_gate_autopsy.json",
    )
    g14 = read_summary(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json")
    s14 = read_summary(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json")
    h14 = first_summary(REPORTS / "V14_H14_R14/readiness.json", REPORTS / "20260508_v14_external_hand_hair_asset_manager.json")
    f14 = read_summary(REPORTS / "20260508_v14_fus3d_region_backend_readiness.json")
    t14 = read_summary(REPORTS / "20260508_v14_tmf_prediction_readiness.json")
    dline = read_summary(REPORTS / "20260508_v14_dline_promotion_report.json")

    sources: list[dict[str, Any]] = []
    for entry in truth.get("artifacts", []):
        name = str(entry.get("name", "unknown"))
        forbidden = bool(entry.get("forbidden_for_promotion"))
        sources.append(
            artifact(
                name,
                source_type="forbidden" if forbidden else str(entry.get("source_type", "diagnostic_only")),
                ownership_region="none",
                path=entry.get("path"),
                promotion_eligible=False,
                status=entry.get("status"),
                reason=str(entry.get("reason") or "V14 truth lock entry is not a V15 ownership-pass source."),
            )
        )

    k14_pass = bool(k14.get("strict_teacher_precheck_pass") or k14.get("visible_surface_teacher_pass"))
    sources.append(
        artifact(
            "K14 Kinect/raw sensor",
            source_type="raw_sensor",
            ownership_region="body" if k14_pass else "none",
            path=k14.get("artifacts", {}).get("output_summary") if isinstance(k14.get("artifacts"), dict) else None,
            promotion_eligible=k14_pass,
            status=k14.get("status"),
            reason="strict teacher precheck passed" if k14_pass else "K14 failed official depth/alignment teacher protocol.",
        )
    )
    g14_pass = bool(g14.get("strict_teacher_precheck_pass"))
    sources.append(
        artifact(
            "G14 2DGS surface",
            source_type="gaussian_surface",
            ownership_region="body" if g14_pass else "none",
            path=str((REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json").resolve()),
            promotion_eligible=g14_pass,
            status=g14.get("status"),
            reason="strict 2DGS teacher precheck passed" if g14_pass else "2DGS protocol remains below strict threshold and normal export was diagnostic.",
        )
    )
    sources.append(
        artifact(
            "S14 Sapiens normal/depth",
            source_type="supervision_only",
            ownership_region="none",
            path=str((REPORTS / "20260508_v14_sapiens_normal_depth_qa.json").resolve()),
            promotion_eligible=False,
            status=s14.get("status"),
            reason="Sapiens is 2D supervision only and cannot be promoted directly.",
        )
    )
    hand_ready = bool(h14.get("hand_ownership_ready"))
    hair_ready = bool(h14.get("hair_ownership_ready"))
    sources.append(
        artifact(
            "H14 hand routes",
            source_type="licensed_hand_model",
            ownership_region="left_hand" if hand_ready else "none",
            path=str((REPORTS / "V14_H14_R14/readiness.json").resolve()),
            promotion_eligible=hand_ready,
            status=h14.get("status"),
            reason="hand ownership ready" if hand_ready else "No runnable hand route with MANO/checkpoint/ownership assets.",
        )
    )
    sources.append(
        artifact(
            "R14 hair routes",
            source_type="licensed_hair_model",
            ownership_region="hair" if hair_ready else "none",
            path=str((REPORTS / "V14_H14_R14/readiness.json").resolve()),
            promotion_eligible=hair_ready,
            status=h14.get("status"),
            reason="hair topology ownership ready" if hair_ready else "No runnable hair route with FLAME/HairGS/GaussianHaircut topology assets.",
        )
    )
    f14_ready = bool(f14.get("body_head_face_ready"))
    sources.append(
        artifact(
            "F14 Fus3D body/head/face",
            source_type="learned_surface",
            ownership_region="body" if f14_ready else "none",
            path=str((REPORTS / "20260508_v14_fus3d_region_backend_readiness.json").resolve()),
            promotion_eligible=f14_ready,
            status=f14.get("status"),
            reason="body/head/face ownership ready" if f14_ready else "Fus3D region backend is research-prep only; strict teacher/anchor missing.",
        )
    )
    t14_ready = bool(t14.get("canonical_teacher_ready"))
    sources.append(
        artifact(
            "T14 temporal canonical teacher",
            source_type="raw_sensor" if t14_ready else "diagnostic_only",
            ownership_region="body" if t14_ready else "none",
            path=str((REPORTS / "20260508_v14_tmf_prediction_readiness.json").resolve()),
            promotion_eligible=t14_ready,
            status=t14.get("status"),
            reason="canonical teacher ready" if t14_ready else "frame0001/frame0002 predictions missing; no canonical teacher pass.",
        )
    )

    eligible = [src for src in sources if src["promotion_eligible"]]
    summary = {
        "task": "v15_promotion_source_graph",
        "created_utc": utc_now(),
        "status": "v15_source_graph_no_promotable_full_candidate",
        "strict_candidate_passes": int(dline.get("strict_candidate_passes", 0) or 0),
        "strict_teacher_passes": int(dline.get("strict_teacher_passes", 0) or 0),
        "promotion_eligible_count": len(eligible),
        "promotion_eligible_regions": sorted({src["ownership_region"] for src in eligible}),
        "sources": sources,
        "decision": "V15 source graph built. No complete promotion-eligible body/head/face/hair/hand region set exists yet.",
        "blockers": [
            "No strict K/G/T teacher source.",
            "No hand ownership source.",
            "No hair topology ownership source.",
            "Unified source set is illegal until required ownership regions are promotion eligible.",
        ],
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V15 Promotion Source Graph", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
