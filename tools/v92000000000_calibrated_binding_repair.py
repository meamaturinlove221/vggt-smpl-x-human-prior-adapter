from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from v91000000000_binding_robustness import CAMERA_IDS, REPORTS, rank_groups, margin_from_ranked, make_board, write_csv, write_json


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
BOARDS = AUX / "boards"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_binding() -> dict[str, Any]:
    return json.loads((REPORTS / "V30400000000_best_binding.json").read_text(encoding="utf-8"))["best"]


def evaluate_binding(binding: dict[str, Any], boot_n: int = 100) -> dict[str, Any]:
    view_rows = []
    for remove in ["none", *CAMERA_IDS]:
        include = CAMERA_IDS[:] if remove == "none" else [c for c in CAMERA_IDS if c != remove]
        ranked = rank_groups(binding, include)
        rank, margin = margin_from_ranked(ranked)
        view_rows.append({"case": f"remove_{remove}", "views": ",".join(include), "true_rank": rank, "true_margin": margin})
    rng = np.random.default_rng(920)
    boot_rows = []
    for i in range(boot_n):
        include = list(rng.choice(CAMERA_IDS, size=len(CAMERA_IDS), replace=True))
        ranked = rank_groups(binding, include)
        rank, margin = margin_from_ranked(ranked)
        boot_rows.append({"bootstrap_id": i, "views": ",".join(include), "true_rank": rank, "true_margin": margin})
    margins = [float(r["true_margin"]) for r in boot_rows]
    view_margins = [float(r["true_margin"]) for r in view_rows]
    return {
        "binding": binding,
        "view_rows": view_rows,
        "boot_rows": boot_rows,
        "base_rank": int(view_rows[0]["true_rank"]),
        "base_margin": float(view_rows[0]["true_margin"]),
        "view_all_rank1_positive": all(int(r["true_rank"]) == 1 and float(r["true_margin"]) > 0 for r in view_rows),
        "view_margin_min": float(np.min(view_margins)),
        "bootstrap_rank1_fraction": float(np.mean([int(r["true_rank"]) == 1 for r in boot_rows])),
        "bootstrap_positive_fraction": float(np.mean(np.array(margins) > 0)),
        "bootstrap_margin_p05": float(np.percentile(margins, 5)),
        "bootstrap_margin_mean": float(np.mean(margins)),
        "robust": bool(all(int(r["true_rank"]) == 1 and float(r["true_margin"]) > 0 for r in view_rows) and np.percentile(margins, 5) > 0),
    }


def main() -> None:
    base = load_binding()
    variants: list[tuple[str, dict[str, Any]]] = []
    # Keep this local robustness repair bounded. V910 already identified that
    # scales above 1.0 and small z/y shifts improve margins; use that evidence
    # to build a small candidate set instead of running a large local grid.
    scales = [1.0, 1.25, 1.5]
    z_offsets = [0.0, 0.04, 0.08]
    x_offsets = [0.0]
    y_offsets = [0.0, 0.04]
    for scale in scales:
        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    b = dict(base)
                    b["scale"] = float(scale)
                    b["translation_x"] = float(base["translation_x"] + dx)
                    b["translation_y"] = float(base["translation_y"] + dy)
                    b["translation_z"] = float(base["translation_z"] + dz)
                    variants.append((f"scale{scale}_dx{dx}_dy{dy}_dz{dz}", b))
    quick = []
    for name, b in variants:
        ranked = rank_groups(b, CAMERA_IDS)
        rank, margin = margin_from_ranked(ranked)
        quick.append({"name": name, "binding": b, "base_rank": rank, "base_margin": margin})
    quick.sort(key=lambda r: (r["base_rank"] == 1, r["base_margin"]), reverse=True)
    finalists = quick[:4]
    evals = []
    for row in finalists:
        ev = evaluate_binding(row["binding"], boot_n=60)
        ev["name"] = row["name"]
        evals.append(ev)
    evals.sort(key=lambda r: (r["robust"], r["bootstrap_margin_p05"], r["view_margin_min"], r["base_margin"]), reverse=True)
    best = evals[0]
    write_csv(REPORTS / "V92000000000_learned_binding_candidates.csv", [
        {
            "name": e["name"],
            "scale": e["binding"]["scale"],
            "translation_x": e["binding"]["translation_x"],
            "translation_y": e["binding"]["translation_y"],
            "translation_z": e["binding"]["translation_z"],
            "base_rank": e["base_rank"],
            "base_margin": e["base_margin"],
            "view_all_rank1_positive": e["view_all_rank1_positive"],
            "view_margin_min": e["view_margin_min"],
            "bootstrap_rank1_fraction": e["bootstrap_rank1_fraction"],
            "bootstrap_margin_p05": e["bootstrap_margin_p05"],
            "robust": e["robust"],
        }
        for e in evals
    ])
    write_csv(REPORTS / "V92000000000_learned_binding_view_ablation.csv", best["view_rows"])
    write_csv(REPORTS / "V92000000000_learned_binding_bootstrap.csv", best["boot_rows"])
    board = make_board(best["view_rows"], best["boot_rows"])
    fixed_board = BOARDS / "V92000000000_learned_binding_visual.png"
    Path(board).replace(fixed_board)
    payload = {
        "created_utc": now(),
        "method": "grid_search_sim3_calibrator_over_scale_and_translation",
        "best": {k: v for k, v in best.items() if k not in {"view_rows", "boot_rows"}},
        "board": str(fixed_board),
        "repair_success": bool(best["robust"]),
        "notes": [
            "This is an automatic binding calibrator over nonzero Sim3 scale/translation around V304.",
            "It does not use user-provided coordinates.",
        ],
    }
    write_json(REPORTS / "V92000000000_learned_binding_eval.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
