from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V35_60view_support_expansion"
REPORT_JSON = ROOT / "reports" / "20260508_v35_60view_support_expansion.json"
REPORT_MD = ROOT / "reports" / "20260508_v35_60view_support_expansion.md"

V24_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
RERASTER_SUMMARY = OUT_DIR / "v35_60view_smplx_prior_reraster_summary.json"
SCENE_60V = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _support(mask: np.ndarray) -> dict:
    return {
        "pixels": int(mask.sum()),
        "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
        "per_view_pixels": [int(x) for x in mask.reshape(mask.shape[0], -1).sum(axis=1)],
    }


def _scene_inventory(scene: Path) -> dict:
    if not scene.exists():
        return {"exists": False, "path": str(scene)}
    files = list(scene.rglob("*"))
    suffix_counts = {}
    for f in files:
        if f.is_file():
            suffix_counts[f.suffix.lower() or "<none>"] = suffix_counts.get(f.suffix.lower() or "<none>", 0) + 1
    return {
        "exists": True,
        "path": str(scene),
        "file_count": int(sum(suffix_counts.values())),
        "suffix_counts": suffix_counts,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v24 = np.load(V24_TARGETS, allow_pickle=True)
    v26 = np.load(V26_TARGETS, allow_pickle=True)
    names = [str(x) for x in v24["teacher_region_names"]]
    region_masks = v24["teacher_region_masks"].astype(bool)
    teacher_support = {name: _support(region_masks[idx]) for idx, name in enumerate(names)}
    temporal_support = {}
    rid = v26["target_frame_region_id_map"].astype(np.uint8)
    for idx, name in enumerate(names, start=1):
        temporal_support[name] = _support(rid == idx)

    reraster = json.loads(RERASTER_SUMMARY.read_text(encoding="utf-8")) if RERASTER_SUMMARY.exists() else {"status": "missing"}
    scene_inv = _scene_inventory(SCENE_60V)
    has_60v_scene = bool(scene_inv.get("exists"))
    has_native_60v_prior = bool(reraster.get("found_native_60v_prior_case", False))

    region_6v_pass = {
        name: bool(teacher_support[name]["pixels"] > 0 and temporal_support[name]["pixels"] > 0)
        for name in names
    }
    # V35 support expansion is a pass only when actual native 60-view priors exist. Otherwise
    # the 6-view teacher support is useful evidence but remains routed for reraster.
    status = "DONE_PASS" if has_native_60v_prior and all(region_6v_pass.values()) else "DONE_FAIL_ROUTED"
    blockers = []
    if not has_native_60v_prior:
        blockers.append("native_60view_smplx_prior_missing")
    if not has_60v_scene:
        blockers.append("60view_scene_missing")
    if not all(region_6v_pass.values()):
        blockers.append("6view_teacher_or_temporal_region_support_missing")

    summary = {
        "task": "v35_60view_teacher_candidate_support_audit",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_formal_outputs": True,
        "has_60v_scene": has_60v_scene,
        "has_native_60v_prior": has_native_60v_prior,
        "scene_60v_inventory": scene_inv,
        "teacher_6v_support": teacher_support,
        "temporal_6v_support": temporal_support,
        "region_6v_support_pass": region_6v_pass,
        "reraster_summary": reraster,
        "outputs": {
            "summary_json": str(OUT_DIR / "summary.json"),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "blockers": blockers,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V35 60-view Support Expansion",
                "",
                f"Status: `{status}`",
                "",
                f"- 60-view scene exists: `{has_60v_scene}`",
                f"- Native 60-view SMPL-X prior exists: `{has_native_60v_prior}`",
                f"- 6-view region support pass: `{all(region_6v_pass.values())}`",
                "",
                "Region 6-view support:",
                *[f"- {name}: teacher={teacher_support[name]['pixels']} temporal={temporal_support[name]['pixels']} pass={region_6v_pass[name]}" for name in names],
                "",
                "No formal output was written. If native 60-view prior is missing, this stage routes to the upstream SMPL-X native 60-view raster exporter.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
