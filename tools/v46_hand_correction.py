from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, file_row, read_json, scan_forbidden, utc_now, write_json, write_md


OUT = LOCAL_OUT / "V46_hand_correction"
JSON = REPORTS / "20260509_v46_hand_correction.json"
MD = REPORTS / "20260509_v46_hand_correction.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v34 = read_json(REPORTS / "20260508_v34_smplx_native_hand_route.json")
    metrics = v34.get("metrics") if isinstance(v34.get("metrics"), dict) else {}
    left = metrics.get("left", {}) if isinstance(metrics.get("left"), dict) else {}
    right = metrics.get("right", {}) if isinstance(metrics.get("right"), dict) else {}
    outputs = v34.get("outputs") if isinstance(v34.get("outputs"), dict) else {}
    patch_npz = Path(str(outputs.get("hand_patch_npz", "")))
    patch_ply = Path(str(outputs.get("hand_patch_ply", "")))
    blockers: list[str] = []
    if v34.get("status") != "DONE_PASS":
        blockers.append("V34 hand route source did not pass")
    for side, row in (("left", left), ("right", right)):
        if int(row.get("pixels", 0) or 0) <= 0:
            blockers.append(f"{side}_hand_pixels_empty")
        if int(row.get("wrist_bridge_pixels", 0) or 0) <= 0:
            blockers.append(f"{side}_wrist_bridge_empty")
        if int(row.get("finger_bands_nonempty", 0) or 0) < 5:
            blockers.append(f"{side}_finger_bands_incomplete")
    if not patch_npz.is_file():
        blockers.append("hand continuity NPZ missing")
    if not patch_ply.is_file():
        blockers.append("hand continuity PLY missing")
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V46 output detected")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v46_hand_correction",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "smplx_native_only": True,
        "left_hand_pixels": int(left.get("pixels", 0) or 0),
        "right_hand_pixels": int(right.get("pixels", 0) or 0),
        "left_finger_bands_nonempty": int(left.get("finger_bands_nonempty", 0) or 0),
        "right_finger_bands_nonempty": int(right.get("finger_bands_nonempty", 0) or 0),
        "artifacts": {
            "hand_patch_npz": file_row(patch_npz),
            "hand_patch_ply": file_row(patch_ply),
        },
        "forbidden_hit_count": len(forbidden),
        "blockers": blockers,
        "decision": "V46 hand correction is usable for research pre-promotion." if status == "DONE_PASS" else "V46 routed; hand continuity remains insufficient.",
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V46 Hand Correction", summary, [
        f"- left_hand_pixels: `{summary['left_hand_pixels']}`",
        f"- right_hand_pixels: `{summary['right_hand_pixels']}`",
        f"- left_finger_bands_nonempty: `{summary['left_finger_bands_nonempty']}`",
        f"- right_finger_bands_nonempty: `{summary['right_finger_bands_nonempty']}`",
    ])
    print({"status": status, "blockers": blockers})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
