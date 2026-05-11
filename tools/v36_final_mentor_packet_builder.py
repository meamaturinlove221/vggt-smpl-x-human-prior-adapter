from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V36_final_promotion" / "mentor_packet"
OUT_JSON = REPORTS / "20260508_v36_final_mentor_packet.json"
OUT_MD = REPORTS / "20260508_v36_final_mentor_packet.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def copy(path: Path, subdir: str) -> str | None:
    if not path.is_file():
        return None
    dst_dir = OUT / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    shutil.copy2(path, dst)
    return str(dst.resolve())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    promotion = read_json(REPORTS / "20260508_v36_final_promotion_report.json")
    copied = {}
    for report in [
        "20260508_v29_normal_route_rescue.md",
        "20260508_v30_prior_enabled_vggt_predictions.md",
        "20260508_v31_teacher_supervised_candidate_train.md",
        "20260508_v32_candidate_inference_region_audit.md",
        "20260508_v33_head_face_detail_route.md",
        "20260508_v34_smplx_native_hand_route.md",
        "20260508_v35_60view_support_expansion.md",
        "20260508_v36_final_promotion_report.md",
    ]:
        copied[report] = copy(REPORTS / report, "reports")
    for visual in [
        REPO_ROOT / "output/surface_research_preflight_local/V24_residual_teacher_v2/v24_teacher_region_contact_sheet.png",
        REPO_ROOT / "output/surface_research_preflight_local/V24_residual_teacher_v2/v24_teacher_depth_contact_sheet.png",
    ]:
        copied[visual.name] = copy(visual, "visuals")
    summary = {
        "task": "v36_final_mentor_packet_builder",
        "created_utc": utc_now(),
        "status": "DONE_PASS",
        "strict_candidate_passes": promotion.get("strict_candidate_passes", 0),
        "strict_teacher_passes": promotion.get("strict_teacher_passes", 0),
        "formal_cloud_unblocked": promotion.get("formal_cloud_unblocked", False),
        "packet_dir": str(OUT.resolve()),
        "copied": copied,
        "remaining_blockers": promotion.get("remaining_blockers", []),
    }
    write_json(OUT_JSON, summary)
    lines = [
        "# V36 Final Mentor Packet",
        "",
        f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        f"- formal_cloud_unblocked: `{summary['formal_cloud_unblocked']}`",
        f"- packet_dir: `{summary['packet_dir']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend([f"- {b}" for b in summary["remaining_blockers"]] or ["- none"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE_PASS", "packet_dir": summary["packet_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
