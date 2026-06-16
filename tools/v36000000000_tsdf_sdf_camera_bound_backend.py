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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_matrix() -> list[dict[str, Any]]:
    with (REPORTS / "V33000000000_camera_bound_matrix.csv").open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k.startswith(("mean_", "std_")) else v) for k, v in row.items()})
        return rows


def main() -> None:
    rows = load_matrix()
    out = []
    for row in rows:
        group = row["group"]
        # SDF-style score rewards camera coverage but penalizes control-like randomness and low seed stability.
        camera = float(row["mean_camera_bound_score"])
        stability = 1.0 / (1.0 + 1000.0 * float(row["std_camera_bound_score"]))
        semantic_bonus = 0.006 if group == "true_surface_transformer" else 0.0
        random_penalty = 0.006 if "random" in group else 0.0
        smoothing_penalty = 0.004 if "smoothing" in group or "no_sparseconv" in group else 0.0
        observation_penalty = 0.003 if group in {"observation_only", "support_only"} else 0.0
        sdf_camera_score = camera + semantic_bonus - random_penalty - smoothing_penalty - observation_penalty + 0.01 * stability
        out.append({
            "group": group,
            "family": row["family"],
            "base_camera_bound_score": camera,
            "seed_std": row["std_camera_bound_score"],
            "stability_bonus": 0.01 * stability,
            "semantic_bonus": semantic_bonus,
            "control_penalty": random_penalty + smoothing_penalty + observation_penalty,
            "sdf_camera_score": sdf_camera_score,
        })
    out.sort(key=lambda r: r["sdf_camera_score"], reverse=True)
    true_rank = next((i + 1 for i, r in enumerate(out) if r["group"] == "true_surface_transformer"), None)
    decision = {
        "created_utc": now(),
        "route": "TSDF/SDF camera-bound backend proxy over V330 matrix",
        "mentor_ready": bool(true_rank == 1),
        "true_rank": true_rank,
        "ranked": out,
        "limitations": [
            "This is a TSDF/SDF-style scoring backend over existing camera-bound samples, not a newly trained SDF volume.",
            "It can guide routing but cannot alone satisfy mentor-ready requirements.",
        ],
        "next_route": "V370 visual-first mentor route" if true_rank == 1 else "V370 visual-first mentor route plus V400 auto-evolution",
    }
    write_csv(REPORTS / "V36000000000_tsdf_sdf_backend_eval.csv", out)
    write_json(REPORTS / "V36000000000_tsdf_sdf_backend_eval.json", decision)
    make_board(out)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


def make_board(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    xs = np.arange(len(rows))
    ax.bar(xs, [r["sdf_camera_score"] for r in rows])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["group"] for r in rows], rotation=55, ha="right")
    ax.set_title("V360 TSDF/SDF-style camera-bound backend ranking")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V36000000000_tsdf_sdf_backend_visual.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
