from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOCAL = ROOT / "output" / "surface_research_preflight_local"
OUT = LOCAL / "V35_60view_support_expansion"
V24 = LOCAL / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26 = LOCAL / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
SCENE = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views"
DATASET = ROOT.parent / "datasets_g" / "data_used_in_4K4D"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def support(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "pixels": int(mask.sum()),
        "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
        "per_view_pixels": [int(x) for x in mask.reshape(mask.shape[0], -1).sum(axis=1)],
    }


def scene_inventory() -> dict[str, Any]:
    image_files = sorted((SCENE / "images").glob("*")) if (SCENE / "images").is_dir() else []
    mask_files = sorted((SCENE / "masks").glob("*")) if (SCENE / "masks").is_dir() else []
    data_files = {
        "main_smc": DATASET / "main" / "0012_11.smc",
        "annotations_smc": DATASET / "annotations" / "0012_11_annots.smc",
        "kinect_smc": DATASET / "kinect" / "0012_11_kinect.smc",
    }
    return {
        "scene_root": SCENE,
        "scene_exists": SCENE.is_dir(),
        "scene_image_count": len([p for p in image_files if p.is_file()]),
        "scene_mask_count": len([p for p in mask_files if p.is_file()]),
        "raw_dataset_files": {
            name: {"path": path, "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}
            for name, path in data_files.items()
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    if not V24.is_file():
        blockers.append(f"missing V24 targets: {V24}")
    if not V26.is_file():
        blockers.append(f"missing V26 targets: {V26}")
    if blockers:
        summary = {"status": "DONE_FAIL_ROUTED", "blockers": blockers}
        write_json(REPORTS / "20260508_v35_60view_support_expansion.json", summary)
        print(summary["status"])
        return 2

    with np.load(V24, allow_pickle=False) as z:
        names = [str(x) for x in z["teacher_region_names"]]
        region_masks = z["teacher_region_masks"].astype(bool)
    with np.load(V26, allow_pickle=False) as z:
        rid = z["target_frame_region_id_map"].astype(np.uint8)
    teacher_support = {name: support(region_masks[idx]) for idx, name in enumerate(names)}
    temporal_support = {name: support(rid == idx) for idx, name in enumerate(names, start=1)}
    region_pass = {name: teacher_support[name]["pixels"] > 0 and temporal_support[name]["pixels"] > 0 for name in names}
    inv = scene_inventory()
    raw_ready = all(row["exists"] for row in inv["raw_dataset_files"].values())
    scene_or_raw_ready = raw_ready or (inv["scene_image_count"] >= 60 and inv["scene_mask_count"] >= 60)
    if not scene_or_raw_ready:
        blockers.append("neither filled local 60-view scene nor G-drive raw 4K4D SMC triplet is available")
    if not all(region_pass.values()):
        blockers.append("6-view teacher/temporal region support missing")

    summary = {
        "task": "v35_60view_support_expansion",
        "created_utc": now(),
        "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
        "recovery_mode": "60view_support_from_G_drive_raw_4K4D_plus_V24_V26_region_support",
        "research_only": True,
        "smplx_native_only": True,
        "no_formal_outputs": True,
        "has_60v_scene": bool(inv["scene_exists"]),
        "has_filled_60v_scene_files": bool(inv["scene_image_count"] >= 60 and inv["scene_mask_count"] >= 60),
        "has_gdrive_raw_4k4d_smc_triplet": bool(raw_ready),
        "scene_inventory": inv,
        "teacher_6v_support": teacher_support,
        "temporal_6v_support": temporal_support,
        "region_6v_support_pass": region_pass,
        "blockers": blockers,
    }
    write_json(OUT / "summary.json", summary)
    write_json(REPORTS / "20260508_v35_60view_support_expansion.json", summary)
    write_json(REPORTS / "20260508_v35_60view_smplx_prior_reraster.json", {
        "status": summary["status"],
        "recovery_mode": summary["recovery_mode"],
        "has_gdrive_raw_4k4d_smc_triplet": summary["has_gdrive_raw_4k4d_smc_triplet"],
        "blockers": blockers,
    })
    md = [
        "# V35 60-View Support Expansion",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- local 60-view scene exists: `{summary['has_60v_scene']}`",
        f"- local scene filled: `{summary['has_filled_60v_scene_files']}`",
        f"- G-drive raw SMC triplet exists: `{summary['has_gdrive_raw_4k4d_smc_triplet']}`",
        f"- blockers: `{blockers}`",
    ]
    (REPORTS / "20260508_v35_60view_support_expansion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "20260508_v35_60view_smplx_prior_reraster.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(summary["status"])
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
