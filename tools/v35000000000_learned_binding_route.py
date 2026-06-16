from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import importlib.util


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V304 = load_module("v304", REPO / "tools" / "v30400000000_coordinate_binding_search.py")
V305 = load_module("v305", REPO / "tools" / "v30500000000_camera_bound_eval.py")


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


def sample_points(points: np.ndarray, valid: np.ndarray, max_points: int = 1600) -> list[np.ndarray]:
    rng = np.random.default_rng(350)
    out = []
    for vi in range(points.shape[0]):
        idx = np.flatnonzero(valid[vi].reshape(-1))
        if idx.size > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        out.append(points[vi].reshape(-1, 3)[idx].astype(np.float64))
    return out


def build_cache(base_binding: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    smc_path = V304.SMC_DIR / base_binding["smc"]
    items = []
    signs = np.array(V304.AXIS_FLIPS[base_binding["axis_flip"]], dtype=np.float64)
    unit = float(V304.UNIT_SCALES[base_binding["unit_name"]])
    for family, group, path in V305.prediction_sources():
        pts, valid = V305.load_points(path)
        view_samples = sample_points(pts * signs[None, None, None, :] * unit, valid)
        for vi, cid in enumerate(CAMERA_IDS):
            if vi >= len(view_samples) or view_samples[vi].size == 0:
                continue
            mask_info = V304.try_read_mask(smc_path, cid)
            if mask_info is None:
                continue
            h, w = pts.shape[1], pts.shape[2]
            mask = V305.resize_mask_for_points(mask_info, (h, w))
            cam = V305.scaled_camera(smc_path, cid, mask_info["source_hw"], (h, w))
            items.append({
                "family": family,
                "group": group,
                "camera_id": cid,
                "points": view_samples[vi],
                "mask": mask,
                "K": cam["K"],
                "RT": cam["RT"],
            })
    return items, smc_path


def eval_cached(items: list[dict[str, Any]], binding: dict[str, Any]) -> list[dict[str, Any]]:
    t = np.array([binding["translation_x"], binding["translation_y"], binding["translation_z"]], dtype=np.float64)
    scale = float(binding["scale"])
    rows = []
    for item in items:
        pts = item["points"] * scale + t[None, :]
        xy, z = V304.project(pts, item["K"], item["RT"], binding["rt_convention"])
        stats = V304.mask_iou_from_xy(xy, item["mask"])
        stats["positive_depth_ratio"] = float((z > 0).mean()) if z.size else 0.0
        rows.append({
            "family": item["family"],
            "group": item["group"],
            "camera_id": item["camera_id"],
            **stats,
        })
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    out = []
    for group, gr in groups.items():
        score = float(np.mean([r["bbox_iou"] + r["mask_coverage"] + r["in_frame_ratio"] for r in gr]))
        out.append({
            "group": group,
            "family": gr[0]["family"],
            "score": score,
            "mean_bbox_iou": float(np.mean([r["bbox_iou"] for r in gr])),
            "mean_mask_coverage": float(np.mean([r["mask_coverage"] for r in gr])),
            "mean_center_error": float(np.mean([r["center_error"] for r in gr])),
            "views": len(gr),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def true_margin(ranked: list[dict[str, Any]]) -> tuple[float, float, float]:
    true = next((r for r in ranked if r["group"] == "true_surface_transformer"), None)
    if true is None:
        return -1e9, -1.0, -1.0
    controls = [r for r in ranked if r["group"] != "true_surface_transformer"]
    strongest = max(controls, key=lambda r: r["score"]) if controls else {"score": 0.0}
    return float(true["score"] - strongest["score"]), float(true["score"]), float(strongest["score"])


def search(items: list[dict[str, Any]], base: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tx, ty, tz = float(base["translation_x"]), float(base["translation_y"]), float(base["translation_z"])
    # Coarse then local. Keep this bounded; this is an automatic calibrator, not full training.
    candidates = []
    for scale_mult in [0.5, 0.8, 1.0, 1.25, 2.0]:
        for dx in [-0.25, 0.0, 0.25]:
            for dy in [-0.25, 0.0, 0.25]:
                for dz in [-0.5, 0.0, 0.5]:
                    candidates.append((scale_mult, dx, dy, dz))
    coarse_rows = []
    best_binding = dict(base)
    best_ranked: list[dict[str, Any]] = []
    best_margin = -1e9
    for scale_mult, dx, dy, dz in candidates:
        cand = dict(base)
        cand["scale"] = float(base["scale"] * scale_mult)
        cand["translation_x"] = tx + dx
        cand["translation_y"] = ty + dy
        cand["translation_z"] = tz + dz
        ranked = aggregate(eval_cached(items, cand))
        margin, true_score, control_score = true_margin(ranked)
        row = {
            "phase": "coarse",
            "scale": cand["scale"],
            "translation_x": cand["translation_x"],
            "translation_y": cand["translation_y"],
            "translation_z": cand["translation_z"],
            "true_score": true_score,
            "strongest_control_score": control_score,
            "true_margin": margin,
        }
        coarse_rows.append(row)
        if margin > best_margin:
            best_margin = margin
            best_binding = cand
            best_ranked = ranked
    fine_rows = []
    btx, bty, btz = best_binding["translation_x"], best_binding["translation_y"], best_binding["translation_z"]
    for scale_mult in [0.9, 1.0, 1.1]:
        for dx in [-0.05, 0.0, 0.05]:
            for dy in [-0.05, 0.0, 0.05]:
                for dz in [-0.1, 0.0, 0.1]:
                    cand = dict(best_binding)
                    cand["scale"] = float(best_binding["scale"] * scale_mult)
                    cand["translation_x"] = btx + dx
                    cand["translation_y"] = bty + dy
                    cand["translation_z"] = btz + dz
                    ranked = aggregate(eval_cached(items, cand))
                    margin, true_score, control_score = true_margin(ranked)
                    row = {
                        "phase": "fine",
                        "scale": cand["scale"],
                        "translation_x": cand["translation_x"],
                        "translation_y": cand["translation_y"],
                        "translation_z": cand["translation_z"],
                        "true_score": true_score,
                        "strongest_control_score": control_score,
                        "true_margin": margin,
                    }
                    fine_rows.append(row)
                    if margin > best_margin:
                        best_margin = margin
                        best_binding = cand
                        best_ranked = ranked
    rows = sorted(coarse_rows + fine_rows, key=lambda r: r["true_margin"], reverse=True)
    return best_binding, rows, best_ranked


def make_board(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[:60]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot([r["true_margin"] for r in top], marker="o")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("V350 calibrator search: true margin over strongest control")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("true margin")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V35000000000_learned_binding_visual.png", dpi=180)
    plt.close(fig)


def main() -> None:
    base = json.loads((REPORTS / "V30400000000_best_binding.json").read_text(encoding="utf-8"))["best"]
    items, smc_path = build_cache(base)
    best_binding, rows, ranked = search(items, base)
    write_csv(REPORTS / "V35000000000_learned_binding_search.csv", rows)
    eval_rows = eval_cached(items, best_binding)
    write_csv(REPORTS / "V35000000000_learned_binding_eval.csv", eval_rows)
    make_board(rows)
    margin, true_score, control_score = true_margin(ranked)
    decision = {
        "created_utc": now(),
        "smc": smc_path.name,
        "base_binding": base,
        "best_binding": best_binding,
        "ranked_groups": ranked,
        "true_beats_all_controls_after_calibration": bool(margin > 0),
        "best_true_margin": margin,
        "true_score": true_score,
        "strongest_control_score": control_score,
        "next_route": "V310/V330 camera-bound dataset and matrix" if margin > 0 else "V360 TSDF/SDF camera-bound backend and V370 visual-first route",
        "notes": [
            "This is a bounded Sim3 scale/translation calibrator around V304, with cached projection inputs.",
            "It optimizes true-vs-control margin; if it cannot make true beat controls, the route must evolve.",
        ],
    }
    write_json(REPORTS / "V35000000000_learned_binding_eval.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
