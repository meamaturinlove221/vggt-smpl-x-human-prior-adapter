from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
V360 = OUTPUT / "V3600000000_fullview_dataset_v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_semantic(group: str) -> np.ndarray:
    with np.load(V360 / f"{group}.npz", allow_pickle=False) as z:
        return z["semantic"].astype(np.float32)


def score_semantic(sem: np.ndarray) -> dict[str, float]:
    valid = sem[:, 15] > 0.5
    bary = sem[:, 22:25]
    skin = sem[:, 27:35]
    canon = np.moveaxis(sem[:, 0:3], 1, -1)
    posed = np.moveaxis(sem[:, 3:6], 1, -1)
    tangent_u = np.moveaxis(sem[:, 62:65], 1, -1)
    tangent_v = np.moveaxis(sem[:, 65:68], 1, -1)
    local_n = np.moveaxis(sem[:, 68:71], 1, -1)
    surf = sem[:, 77]
    surf_valid = sem[:, 78] > 0.5
    curv = sem[:, 79]
    body_conf = sem[:, 74]
    mask = valid
    if mask.sum() == 0:
        mask = np.isfinite(sem[:, 0])
    bary_err = np.abs(bary.sum(axis=1) - 1.0)
    skin_err = np.abs(skin.sum(axis=1) - 1.0)
    uv_dot = np.abs((tangent_u * tangent_v).sum(axis=-1))
    n_dot_u = np.abs((local_n * tangent_u).sum(axis=-1))
    n_dot_v = np.abs((local_n * tangent_v).sum(axis=-1))
    n_norm = np.abs(np.linalg.norm(local_n, axis=-1) - 1.0)
    displacement = np.linalg.norm(posed - canon, axis=-1)
    finite = np.isfinite(sem).mean(axis=1)
    # Higher score is better. This deliberately avoids using group labels.
    score_map = (
        np.exp(-5.0 * bary_err)
        + np.exp(-3.0 * skin_err)
        + np.exp(-3.0 * (uv_dot + n_dot_u + n_dot_v + n_norm))
        + 0.1 * np.tanh(displacement)
        + 0.1 * np.tanh(curv)
        + 0.1 * body_conf
        + 0.1 * surf_valid.astype(np.float32) * np.exp(-np.abs(surf))
        + 0.1 * finite
    )
    return {
        "coherence_score": float(score_map[mask].mean()),
        "bary_err_mean": float(bary_err[mask].mean()),
        "skinning_err_mean": float(skin_err[mask].mean()),
        "frame_orth_error_mean": float((uv_dot + n_dot_u + n_dot_v + n_norm)[mask].mean()),
        "surface_valid_ratio": float(surf_valid[mask].mean()),
        "finite_ratio": float(finite[mask].mean()),
        "valid_pixels": int(mask.sum()),
    }


def main() -> None:
    route = Path(r"D:\vggt\vggt-feature-adapter\docs\goals\V18000000000_field_consistency_supervision_route.md")
    route.write_text(
        "# V18000000000 Field Consistency Supervision Route\n\n"
        "Evaluate whether current 81-channel semantic fields have enough intrinsic SMPL-X consistency to reject random/shuffled controls before more geometry training.\n",
        encoding="utf-8",
    )
    groups = ["true_full", "same_support_random_semantic", "same_support_shuffled_semantic", "local_knn_smoothing", "random_adjacency_sparseconv", "no_sparseconv_mlp"]
    rows = []
    for group in groups:
        sem = load_semantic(group)
        row = {"group": group, **score_semantic(sem)}
        rows.append(row)
    rows.sort(key=lambda r: r["coherence_score"], reverse=True)
    write_csv(REPORTS / "V18000000000_field_consistency_metrics.csv", rows)
    true = next(r for r in rows if r["group"] == "true_full")
    random = next(r for r in rows if r["group"] == "same_support_random_semantic")
    shuffled = next(r for r in rows if r["group"] == "same_support_shuffled_semantic")
    passed = true["coherence_score"] > random["coherence_score"] and true["coherence_score"] > shuffled["coherence_score"]
    payload = {
        "created_utc": now(),
        "field_consistency_gate_passed": passed,
        "ranking": rows,
        "true_minus_random": true["coherence_score"] - random["coherence_score"],
        "true_minus_shuffled": true["coherence_score"] - shuffled["coherence_score"],
        "interpretation": "If this fails, current schema cannot support independent semantic causality without additional renderer/supervision assets.",
    }
    write_json(REPORTS / "V18000000000_field_consistency_decision.json", payload)
    if true["coherence_score"] == shuffled["coherence_score"]:
        # Build a stronger shuffled-control manifest. The existing shuffled
        # group preserves field coherence, so it is not a valid adversarial
        # semantic-control for causality.
        stronger = {
            "created_utc": now(),
            "issue": "same_support_shuffled_semantic preserves intrinsic semantic field coherence",
            "required_control": "independently shuffle face/barycentric/skinning/joint/local-frame blocks while keeping support and observation fixed",
            "next_route": "V19000000000_STRONG_SHUFFLED_CONTROL_REBUILD",
            "hard_gate": "true must beat strong_shuffled_control before any semantic causality claim",
        }
        write_json(REPORTS / "V19000000000_strong_shuffled_control_plan.json", stronger)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
