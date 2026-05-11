from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOCAL = ROOT / "output" / "surface_research_preflight_local"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def file_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(str(path)).is_file()


def md(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", "", f"Status: `{summary.get('status')}`", "", f"Blockers: `{summary.get('blockers', [])}`"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    v33 = read_json(REPORTS / "20260508_v33_head_face_detail_route.json")
    cov = ((v33.get("metrics") or {}).get("coverage") or {})
    v45_blockers: list[str] = []
    if v33.get("status") != "DONE_PASS":
        v45_blockers.append("v33_not_pass")
    if int((cov.get("head") or {}).get("pixels", 0) or 0) <= 0:
        v45_blockers.append("head_empty")
    if int((cov.get("face") or {}).get("pixels", 0) or 0) <= 0:
        v45_blockers.append("face_empty")
    outs33 = v33.get("outputs") or {}
    if not file_exists(outs33.get("refined_npz")):
        v45_blockers.append("refined_npz_missing")
    if not file_exists(outs33.get("refined_ply")):
        v45_blockers.append("refined_ply_missing")
    v45 = {
        "task": "v45_head_face_correction",
        "created_utc": now(),
        "status": "DONE_PASS" if not v45_blockers else "DONE_FAIL_ROUTED",
        "research_only": True,
        "smplx_native_only": True,
        "head_pixels": int((cov.get("head") or {}).get("pixels", 0) or 0),
        "face_pixels": int((cov.get("face") or {}).get("pixels", 0) or 0),
        "source_report": str((REPORTS / "20260508_v33_head_face_detail_route.json").resolve()),
        "blockers": v45_blockers,
    }
    write_json(REPORTS / "20260509_v45_head_face_correction.json", v45)
    write_json(LOCAL / "V45_head_face_correction" / "summary.json", v45)
    md(REPORTS / "20260509_v45_head_face_correction.md", "V45 Head Face Correction", v45)

    v34 = read_json(REPORTS / "20260508_v34_smplx_native_hand_route.json")
    mets = v34.get("metrics") or {}
    left = mets.get("left") or {}
    right = mets.get("right") or {}
    outs34 = v34.get("outputs") or {}
    v46_blockers: list[str] = []
    if v34.get("status") != "DONE_PASS":
        v46_blockers.append("v34_not_pass")
    for side, row in (("left", left), ("right", right)):
        if int(row.get("pixels", 0) or 0) <= 0:
            v46_blockers.append(f"{side}_pixels_empty")
        if int(row.get("wrist_bridge_pixels", 0) or 0) <= 0:
            v46_blockers.append(f"{side}_wrist_bridge_empty")
        if int(row.get("finger_bands_nonempty", 0) or 0) < 5:
            v46_blockers.append(f"{side}_finger_bands_incomplete")
    if not file_exists(outs34.get("hand_patch_npz")):
        v46_blockers.append("hand_patch_npz_missing")
    if not file_exists(outs34.get("hand_patch_ply")):
        v46_blockers.append("hand_patch_ply_missing")
    v46 = {
        "task": "v46_hand_correction",
        "created_utc": now(),
        "status": "DONE_PASS" if not v46_blockers else "DONE_FAIL_ROUTED",
        "research_only": True,
        "smplx_native_only": True,
        "left_hand_pixels": int(left.get("pixels", 0) or 0),
        "right_hand_pixels": int(right.get("pixels", 0) or 0),
        "left_finger_bands_nonempty": int(left.get("finger_bands_nonempty", 0) or 0),
        "right_finger_bands_nonempty": int(right.get("finger_bands_nonempty", 0) or 0),
        "source_report": str((REPORTS / "20260508_v34_smplx_native_hand_route.json").resolve()),
        "blockers": v46_blockers,
    }
    write_json(REPORTS / "20260509_v46_hand_correction.json", v46)
    write_json(LOCAL / "V46_hand_correction" / "summary.json", v46)
    md(REPORTS / "20260509_v46_hand_correction.md", "V46 Hand Correction", v46)

    v35 = read_json(REPORTS / "20260508_v35_60view_support_expansion.json")
    v47_blockers: list[str] = []
    if v35.get("status") != "DONE_PASS":
        v47_blockers.append("v35_not_pass")
    if not bool(v35.get("has_gdrive_raw_4k4d_smc_triplet")):
        v47_blockers.append("gdrive_raw_4k4d_smc_triplet_missing")
    for region, ok in (v35.get("region_6v_support_pass") or {}).items():
        if not ok:
            v47_blockers.append(f"{region}_6v_support_failed")
    v47 = {
        "task": "v47_60view_correction",
        "created_utc": now(),
        "status": "DONE_PASS" if not v47_blockers else "DONE_FAIL_ROUTED",
        "research_only": True,
        "has_60v_scene": bool(v35.get("has_60v_scene")),
        "has_gdrive_raw_4k4d_smc_triplet": bool(v35.get("has_gdrive_raw_4k4d_smc_triplet")),
        "region_6v_support_pass": v35.get("region_6v_support_pass") or {},
        "source_report": str((REPORTS / "20260508_v35_60view_support_expansion.json").resolve()),
        "blockers": v47_blockers,
    }
    write_json(REPORTS / "20260509_v47_60view_correction.json", v47)
    write_json(LOCAL / "V47_60view_correction" / "summary.json", v47)
    md(REPORTS / "20260509_v47_60view_correction.md", "V47 60view Correction", v47)

    v26 = read_json(REPORTS / "20260508_v26_temporal_canonical_teacher.json")
    v29 = read_json(REPORTS / "20260508_v29_normal_route_rescue.json")
    temporal_support = (((v29.get("temporal_summary") or {}).get("region_normal_support")) or {})
    v48_blockers: list[str] = []
    if v26.get("status") != "DONE_PASS":
        v48_blockers.append("v26_not_pass")
    if not bool(v29.get("temporal_normal_available")):
        v48_blockers.append("temporal_normal_unavailable")
    for region in ("body", "head", "face", "left_hand", "right_hand"):
        if int(temporal_support.get(region, 0) or 0) <= 0:
            v48_blockers.append(f"{region}_temporal_normal_support_empty")
    v48 = {
        "task": "v48_temporal_correction",
        "created_utc": now(),
        "status": "DONE_PASS" if not v48_blockers else "DONE_FAIL_ROUTED",
        "research_only": True,
        "v26_status": v26.get("status"),
        "temporal_normal_available": bool(v29.get("temporal_normal_available")),
        "temporal_region_normal_support": temporal_support,
        "source_reports": [
            str((REPORTS / "20260508_v26_temporal_canonical_teacher.json").resolve()),
            str((REPORTS / "20260508_v29_normal_route_rescue.json").resolve()),
        ],
        "blockers": v48_blockers,
    }
    write_json(REPORTS / "20260509_v48_temporal_correction.json", v48)
    write_json(LOCAL / "V48_temporal_correction" / "summary.json", v48)
    md(REPORTS / "20260509_v48_temporal_correction.md", "V48 Temporal Correction", v48)

    statuses = {name: row["status"] for name, row in {"v45": v45, "v46": v46, "v47": v47, "v48": v48}.items()}
    print(json.dumps(statuses, indent=2))
    return 0 if all(status == "DONE_PASS" for status in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
