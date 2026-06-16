from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output" / "V31000000000_camera_bound_dataset"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prediction_sources() -> list[dict[str, Any]]:
    items = []
    items.append({
        "family": "baseline",
        "group": "V11700",
        "path": str(AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"),
        "resolution": "518x518",
    })
    v145 = AUX / "output" / "V14500000000_control_prediction_samples"
    for p in sorted(v145.glob("*/*predictions_sample.npz")):
        items.append({"family": "V145_sample", "group": p.parent.name.replace("_seed0", ""), "path": str(p), "resolution": "130x130"})
    v300 = AUX / "output" / "V30000000000_bundle_samples"
    for p in sorted(v300.glob("*_predictions_sample.npz")):
        group = p.name.replace("_seed0_predictions_sample.npz", "").replace("_predictions_sample.npz", "")
        items.append({"family": "V300_sample", "group": group, "path": str(p), "resolution": "65x65"})
    return [x for x in items if Path(x["path"]).exists()]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    binding = json.loads((REPORTS / "V35000000000_learned_binding_eval.json").read_text(encoding="utf-8"))
    sources = prediction_sources()
    copied = []
    for item in sources:
        src = Path(item["path"])
        dst_dir = OUTPUT / f"{item['family']}__{item['group']}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "predictions.npz"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        source_manifest = {
            "created_utc": now(),
            "family": item["family"],
            "group": item["group"],
            "source_path": str(src),
            "local_prediction_path": str(dst),
            "resolution": item["resolution"],
            "camera_ids": CAMERA_IDS,
            "binding_source": str(REPORTS / "V35000000000_learned_binding_eval.json"),
            "best_binding": binding["best_binding"],
            "no_promotion": True,
            "active_candidate_replaced": False,
        }
        write_json(dst_dir / "source_manifest.json", source_manifest)
        copied.append(source_manifest)
    manifest = {
        "created_utc": now(),
        "dataset_root": str(OUTPUT),
        "binding": binding["best_binding"],
        "items": copied,
        "limitations": [
            "This dataset combines full V11700 518x518 baseline with V145 130x130 and V300 65x65 sampled predictions.",
            "It is suitable for camera-bound verification and repair routing, not final paper-grade full-resolution matrix by itself.",
        ],
    }
    write_json(REPORTS / "V31000000000_dataset_manifest.json", manifest)
    write_json(REPORTS / "V31000000000_pairing_audit.json", {
        "created_utc": now(),
        "groups": [x["group"] for x in copied],
        "camera_ids": CAMERA_IDS,
        "same_binding_for_all_groups": True,
        "controls_present": sorted([x["group"] for x in copied if x["group"] != "true_surface_transformer"]),
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
