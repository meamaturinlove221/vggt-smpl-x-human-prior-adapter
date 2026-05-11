from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V35_60view_support_expansion"
REPORT_JSON = ROOT / "reports" / "20260508_v35_60view_smplx_prior_reraster.json"
REPORT_MD = ROOT / "reports" / "20260508_v35_60view_smplx_prior_reraster.md"

CANDIDATE_60V = [
    ROOT / "output" / "training_cases" / "0012_11_frame0000_60views_smplx_native_prior_v15",
    ROOT / "output" / "training_cases" / "0012_11_frame0000_60views_v15_smplx_native_prior",
    ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views",
]
V16_6V_ROI = ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
V15_6V_RASTER = ROOT / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_npz_keys(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    try:
        z = np.load(path, allow_pickle=True)
        return {"exists": True, "keys": list(z.files), "shapes": {k: list(z[k].shape) for k in z.files}}
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def _case_inventory(case_dir: Path) -> dict:
    files = {}
    for name in ["inputs.npz", "targets.npz", "prior_maps.npz", "case_manifest.json"]:
        p = case_dir / name
        files[name] = {"exists": p.exists(), "path": str(p), "length": p.stat().st_size if p.exists() else 0}
        if p.suffix == ".npz" and p.exists():
            files[name].update(_safe_npz_keys(p))
    return {"exists": case_dir.exists(), "path": str(case_dir), "files": files}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = {p.name: _case_inventory(p) for p in CANDIDATE_60V}
    found_native_60v = any(
        item["exists"] and (item["files"].get("inputs.npz", {}).get("exists") or item["files"].get("targets.npz", {}).get("exists"))
        for name, item in inventory.items()
        if "60views_smplx_native" in name or "60views_v15_smplx_native" in name
    )

    roi = np.load(V16_6V_ROI, allow_pickle=True)
    roi_names = [str(x) for x in roi["roi_names"]]
    roi_maps = roi["roi_maps"].astype(bool)
    six_view_support = {}
    for idx, name in enumerate(roi_names):
        m = roi_maps[:, idx]
        six_view_support[name] = {
            "pixels": int(m.sum()),
            "views_with_pixels": int(np.count_nonzero(m.reshape(m.shape[0], -1).sum(axis=1) > 0)),
            "per_view_pixels": [int(x) for x in m.reshape(m.shape[0], -1).sum(axis=1)],
        }

    raster_keys = _safe_npz_keys(V15_6V_RASTER)
    status = "DONE_PASS" if found_native_60v else "DONE_FAIL_ROUTED"
    blockers = [] if found_native_60v else ["60v_smplx_native_prior_case_missing; reraster requires V15 raster exporter to be run on 60 views"]
    summary = {
        "task": "v35_60view_smplx_prior_reraster",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_formal_outputs": True,
        "found_native_60v_prior_case": found_native_60v,
        "inventory": inventory,
        "six_view_roi_support_baseline": six_view_support,
        "six_view_raster_npz": raster_keys,
        "outputs": {
            "summary_json": str(OUT_DIR / "v35_60view_smplx_prior_reraster_summary.json"),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "blockers": blockers,
    }
    (OUT_DIR / "v35_60view_smplx_prior_reraster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V35 60-view SMPL-X Prior Reraster",
                "",
                f"Status: `{status}`",
                "",
                f"- Native 60-view prior case found: `{found_native_60v}`",
                f"- 6-view ROI baseline path: `{V16_6V_ROI}`",
                "",
                "This tool does not fabricate 60-view SMPL-X priors. If the 60-view native prior case is absent, it records a routed blocker for the upstream V15 raster exporter.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
