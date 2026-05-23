from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_csv(REPORTS / "V30500000000_camera_bound_eval.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    ranked = []
    for group, gr in groups.items():
        score = float(np.mean([float(r["bbox_iou"]) + float(r["mask_coverage"]) + float(r["in_frame_ratio"]) for r in gr]))
        ranked.append({
            "group": group,
            "family": gr[0]["family"],
            "score": score,
            "mean_bbox_iou": float(np.mean([float(r["bbox_iou"]) for r in gr])),
            "mean_mask_coverage": float(np.mean([float(r["mask_coverage"]) for r in gr])),
            "mean_in_frame_ratio": float(np.mean([float(r["in_frame_ratio"]) for r in gr])),
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    true_names = {"true_surface_transport", "true_surface_transformer"}
    true_rank = next((i + 1 for i, r in enumerate(ranked) if r["group"] in true_names), None)
    pass_gate = bool(true_rank == 1)
    if pass_gate:
        next_route = "V310/V330 camera-bound dataset and matrix"
        failure = None
    else:
        next_route = "V350 learned binding route, then V360/V370 if still weak"
        failure = "camera_bound_controls_rank_above_true"
    payload = {
        "created_utc": now(),
        "mentor_ready_camera_bound": False,
        "binding_solved": True,
        "camera_bound_true_rank": true_rank,
        "pass_gate": pass_gate,
        "dominant_failure": failure,
        "ranked_groups": ranked,
        "next_route": next_route,
        "notes": [
            "V304 solved a nonzero coordinate binding; V305 shows the camera-bound metric still does not prove true semantic/topology causality.",
            "Failure is experimental/architectural, not an external coordinate hard block.",
        ],
    }
    write_json(REPORTS / "V34000000000_camera_bound_decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
