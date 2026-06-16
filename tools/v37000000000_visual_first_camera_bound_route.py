from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_matrix_rows() -> list[dict[str, str]]:
    with (REPORTS / "V33000000000_camera_bound_matrix.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_board(rows: list[dict[str, str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: float(r["mean_camera_bound_score"]), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    xs = np.arange(len(rows))
    axes[0].bar(xs, [float(r["mean_camera_bound_score"]) for r in rows])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([r["group"] for r in rows], rotation=60, ha="right", fontsize=8)
    axes[0].set_title("Camera-bound score (existing predictions)")
    axes[1].bar(xs, [float(r["std_camera_bound_score"]) for r in rows])
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([r["group"] for r in rows], rotation=60, ha="right", fontsize=8)
    axes[1].set_title("Seed variance")
    fig.suptitle("V370 visual-first routing board: binding solved, mentor-ready not yet satisfied")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V37000000000_visual_first_mentor_board.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_matrix_rows()
    make_board(rows)
    ranked = sorted(rows, key=lambda r: float(r["mean_camera_bound_score"]), reverse=True)
    true = next((r for r in ranked if r["group"] == "true_surface_transformer"), None)
    top = ranked[0] if ranked else None
    report = {
        "created_utc": now(),
        "visual_first_board": str(BOARDS / "V37000000000_visual_first_mentor_board.png"),
        "binding_solved": True,
        "mentor_ready": False,
        "true_rank_raw_camera_bound": (ranked.index(true) + 1) if true in ranked else None,
        "top_raw_camera_bound_group": top["group"] if top else None,
        "decision": "visual_first_insufficient_for_mentor_ready",
        "limitations": [
            "Raw camera-bound score still ranks random_semantic above true_surface_transformer.",
            "V360 SDF-style score ranks true first, but that is a proxy backend score rather than a trained SDF model.",
            "Current visual board is metric-oriented; it is not yet a paper-grade 3D close-up board with full-resolution true/control predictions.",
        ],
        "next_route": "V400 auto-evolved route: point-transformer/differentiable-renderer camera-bound full-resolution training",
    }
    write_json(REPORTS / "V37000000000_visual_first_eval.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
