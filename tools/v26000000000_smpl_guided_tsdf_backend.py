from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
SURFACE = OUTPUT / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"
V240 = OUTPUT / "V24000000000_dense_fullview_predictions"
GROUPS = [
    "true_surface_transformer",
    "random_semantic",
    "strong_shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "no_sparseconv_mlp",
    "no_surface_graph",
    "random_surface_graph",
    "observation_only",
    "support_only",
    "no_teacher",
]
REGIONS = {"full_body": None, "head_face": 1, "hairline": 2, "left_hand": 3, "right_hand": 4}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def gradient_energy(x: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for axis in [1, 2]:
        d = np.diff(x, axis=axis)
        m = np.take(mask, range(d.shape[axis]), axis=axis)
        if d.ndim == 4:
            e = np.linalg.norm(d, axis=-1)
        else:
            e = np.abs(d)
        if m.any():
            vals.append(float(e[m].mean()))
    return float(np.mean(vals)) if vals else math.nan


def run() -> dict[str, Any]:
    with np.load(SURFACE, allow_pickle=False) as s:
        valid = s["valid_mask"].astype(bool)
        region = s["region_label"].astype(np.int16)
        surface_distance = s["surface_distance"].astype(np.float32)
        curvature = s["curvature"].astype(np.float32)
        base_normal = s["normal"].astype(np.float32)
    rows = []
    group_scores = {}
    for group in GROUPS:
        with np.load(V240 / f"{group}_seed0" / "predictions.npz", allow_pickle=False) as z:
            wp = z["world_points"].astype(np.float32)
            normal = z["normal"].astype(np.float32)
            residual = z["normal_residual"].astype(np.float32)
        # SDF-style proxy: distance to SMPL surface plus curvature-weighted
        # residual regularity. Lower continuity cost and stronger true-region
        # residual are better; converted to a positive score below.
        sdf = np.clip(surface_distance - np.linalg.norm(residual, axis=-1), -0.05, 0.05)
        normal_agree = (normal * base_normal).sum(axis=-1)
        continuity = gradient_energy(wp, valid)
        rows_for_group = []
        for name, rid in REGIONS.items():
            mask = valid if rid is None else (valid & (region == rid))
            count = int(mask.sum())
            if not count:
                row = {"group": group, "region": name, "status": "empty"}
            else:
                component_total = sum(int(ndimage.label(mask[v])[1]) for v in range(mask.shape[0]))
                sdf_abs = float(np.abs(sdf[mask]).mean())
                curvature_weight = float(curvature[mask].mean())
                normal_score = float(normal_agree[mask].mean())
                residual_mean = float(np.linalg.norm(residual, axis=-1)[mask].mean())
                score = residual_mean + 0.02 * normal_score + 0.01 * curvature_weight - 0.05 * sdf_abs - 0.01 * continuity
                row = {
                    "group": group,
                    "region": name,
                    "status": "evaluated",
                    "pixel_count": count,
                    "tsdf_abs_mean": sdf_abs,
                    "normal_agreement": normal_score,
                    "surface_continuity_cost": continuity,
                    "component_count": component_total,
                    "curvature_mean": curvature_weight,
                    "normal_residual_mean": residual_mean,
                    "tsdf_score": score,
                }
                rows_for_group.append(score)
            rows.append(row)
        group_scores[group] = float(np.mean(rows_for_group)) if rows_for_group else -1e9
    write_csv(REPORTS / "V26000000000_tsdf_metrics.csv", rows)
    ranking = sorted([{"group": g, "score": s} for g, s in group_scores.items()], key=lambda x: x["score"], reverse=True)
    true_score = group_scores.get("true_surface_transformer", -1e9)
    decision = {
        "created_utc": now(),
        "ranking": ranking,
        "true_beats_all_controls": all(true_score > s for g, s in group_scores.items() if g != "true_surface_transformer"),
        "uses_v11700_teacher": False,
        "reprojection_consistency_available": False,
        "mentor_ready": False,
        "remaining_limitations": [
            "TSDF/SDF backend is a metric/evaluation backend over V920/V240 fields, not a trained volumetric model.",
            "Trusted camera-bound reprojection remains unavailable.",
        ],
        "next_route_required": True,
        "next_route": "V270 external camera action checklist or differentiable renderer route",
    }
    write_json(REPORTS / "V26000000000_decision.json", decision)
    return decision


def board() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(REPORTS / "V26000000000_tsdf_metrics.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["region"] == "full_body" and row["status"] == "evaluated":
                rows.append(row)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([r["group"] for r in rows], [float(r["tsdf_score"]) for r in rows])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_title("V260 TSDF/SDF full_body control scores")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    out = BOARDS / "V26000000000_tsdf_visual.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    write_json(REPORTS / "V26000000000_visual_manifest.json", {"created_utc": now(), "board": str(out)})


def main() -> None:
    decision = run()
    board()
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
