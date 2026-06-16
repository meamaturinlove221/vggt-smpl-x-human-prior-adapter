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


def main() -> None:
    result_path = REPORTS / "V40200000000_modal_training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in result["results"]:
        groups.setdefault(row["group"], []).append(row)
    summary = []
    for group, rows in groups.items():
        summary.append({
            "group": group,
            "seeds": len(rows),
            "mean_loss_delta": float(np.mean([r["loss_delta"] for r in rows])),
            "std_loss_delta": float(np.std([r["loss_delta"] for r in rows])),
            "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows])),
            "mean_delta_l2": float(np.mean([r["delta_l2"] for r in rows])),
            "mean_normal_nonzero_ratio": float(np.mean([r["normal_nonzero_ratio"] for r in rows])),
        })
    summary.sort(key=lambda r: (r["mean_confidence"], r["mean_loss_delta"]), reverse=True)
    true = next((r for r in summary if r["group"] == "true_camera_bound_transport"), None)
    controls = [r for r in summary if r["group"] != "true_camera_bound_transport"]
    strongest = max(controls, key=lambda r: r["mean_confidence"]) if controls else None
    pass_gate = bool(true and strongest and true["mean_confidence"] > strongest["mean_confidence"] and true["mean_normal_nonzero_ratio"] > 0.1)
    decision = {
        "created_utc": now(),
        "modal_status": result["status"],
        "cuda_available": result["cuda_available"],
        "gpu_type": result["gpu_type"],
        "runtime_seconds": result["runtime_seconds"],
        "ranked_groups": summary,
        "true_confidence_margin": float(true["mean_confidence"] - strongest["mean_confidence"]) if true and strongest else None,
        "learned_normal_valid": bool(true and true["mean_normal_nonzero_ratio"] > 0.1),
        "mentor_ready": False,
        "pass_training_signal_gate": pass_gate,
        "next_route": "V404 longer Modal matrix with camera-bound loss" if pass_gate else "V404 longer Modal matrix plus stronger semantic contrastive",
        "notes": [
            "V402 is a Modal A10G sanity matrix, not final paper-grade full matrix.",
            "It establishes remote GPU viability after local crash and confirms learned normal output is nonzero.",
        ],
    }
    write_json(REPORTS / "V40300000000_modal_result_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
