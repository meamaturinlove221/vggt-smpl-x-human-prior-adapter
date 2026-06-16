from __future__ import annotations

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


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((REPORTS / "V40500000000_modal_prediction_samples.json").read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if r.get("seed") == 0 and r.get("sample_delta") is not None]
    fig = plt.figure(figsize=(18, 10))
    for i, row in enumerate(rows[:6]):
        delta = np.asarray(row["sample_delta"], dtype=np.float32)
        normal = np.asarray(row["sample_normal"], dtype=np.float32)
        conf = np.asarray(row["sample_confidence"], dtype=np.float32)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        # Visualize residual/normal cloud in model-output space. This is a sampled board, not full point map.
        pts = delta + 0.03 * normal
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=conf, s=2, cmap="viridis")
        ax.set_title(f"{row['group']}\nconf={row['mean_confidence']:.3f} normal={row['normal_nonzero_ratio']:.1f}")
        ax.set_axis_off()
    fig.suptitle("V406 Modal camera-bound point-transformer sampled outputs (seed0)")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    out = BOARDS / "V40600000000_camera_bound_modal_samples.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    report = {
        "created_utc": now(),
        "board": str(out),
        "groups": [r["group"] for r in rows],
        "visual_type": "sampled_delta_plus_learned_normal_3d_scatter",
        "limitations": [
            "This is sampled Modal output, not a full 518x518 rendered point-map board.",
            "It is useful to inspect learned normal/residual behavior but cannot alone satisfy the final visual mentor gate.",
        ],
    }
    write_json(REPORTS / "V40600000000_visual_board_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
