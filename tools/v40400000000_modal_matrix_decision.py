from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in result["results"]:
        groups.setdefault(row["group"], []).append(row)
    ranked = []
    for group, rows in groups.items():
        ranked.append({
            "group": group,
            "seeds": len(rows),
            "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows])),
            "std_confidence": float(np.std([r["mean_confidence"] for r in rows])),
            "mean_loss_delta": float(np.mean([r["loss_delta"] for r in rows])),
            "mean_delta_l2": float(np.mean([r["delta_l2"] for r in rows])),
            "mean_normal_nonzero_ratio": float(np.mean([r["normal_nonzero_ratio"] for r in rows])),
        })
    ranked.sort(key=lambda r: r["mean_confidence"], reverse=True)
    return {"raw": result, "ranked": ranked}


def main() -> None:
    summary = summarize(REPORTS / "V40400000000_modal_training_result.json")
    ranked = summary["ranked"]
    true = next((r for r in ranked if r["group"] == "true_camera_bound_transport"), None)
    strongest = max([r for r in ranked if r["group"] != "true_camera_bound_transport"], key=lambda r: r["mean_confidence"])
    confidence_margin = float(true["mean_confidence"] - strongest["mean_confidence"]) if true else -1.0
    pass_core = bool(
        true
        and ranked[0]["group"] == "true_camera_bound_transport"
        and confidence_margin > 0.10
        and true["mean_normal_nonzero_ratio"] > 0.99
        and true["std_confidence"] > 0
    )
    decision = {
        "created_utc": now(),
        "modal_status": summary["raw"]["status"],
        "cuda_available": summary["raw"]["cuda_available"],
        "gpu_type": summary["raw"]["gpu_type"],
        "runtime_seconds": summary["raw"]["runtime_seconds"],
        "ranked_groups": ranked,
        "true_confidence_margin": confidence_margin,
        "core_modal_training_gate_passed": pass_core,
        "mentor_ready": False,
        "remaining_gaps": [
            "V404 matrix is remote GPU and multi-seed, but it does not yet export full-view camera-bound predictions back from Modal.",
            "Camera-bound raw projection score still comes from V330 sample predictions, not V404 generated full-view point maps.",
            "Paper-grade full body/head/hair/hand 3D close-up boards have not yet been regenerated from V404 outputs.",
        ],
        "next_route": "V405 export Modal predictions and V406 camera-bound full-view visual boards",
    }
    write_json(REPORTS / "V40400000000_modal_matrix_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
