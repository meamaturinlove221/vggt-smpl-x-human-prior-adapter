from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]
GROUPS = [
    "true_camera_bound_transport",
    "random_surface_semantic",
    "shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "no_surface_graph",
    "random_surface_graph",
    "observation_only",
    "support_only",
    "no_sparseconv_mlp",
    "no_teacher",
]


def load_v304():
    path = REPO / "tools" / "v30400000000_coordinate_binding_search.py"
    spec = importlib.util.spec_from_file_location("v304", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V304 = load_v304()


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


def load_best_binding() -> dict[str, Any]:
    doc = json.loads((REPORTS / "V30400000000_best_binding.json").read_text(encoding="utf-8"))
    if not doc.get("binding_passed") or not doc.get("best"):
        raise RuntimeError("V304 binding did not pass")
    return doc["best"]


def resize_mask_for_points(mask_info: dict[str, Any], hw: tuple[int, int]) -> np.ndarray:
    source_mask = mask_info["mask"].astype(np.uint8) * 255
    return V304.resize_mask(source_mask, hw)


def scaled_camera(path: Path, cid: str, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    with h5py.File(str(path), "r") as f:
        g = f[f"Camera_Parameter/{cid}"]
        return {
            "K": V304.resize_intrinsic(g["K"][()], source_hw, target_hw),
            "RT": g["RT"][()],
        }


def sample_eval_points(points: np.ndarray, valid: np.ndarray, max_points: int = 30000) -> np.ndarray:
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size > max_points:
        rng = np.random.default_rng(414)
        idx = rng.choice(idx, max_points, replace=False)
    return points.reshape(-1, 3)[idx]


def score_group(group: str, points: np.ndarray, confidence: np.ndarray, binding: dict[str, Any], smc_path: Path) -> list[dict[str, Any]]:
    signs = np.array(V304.AXIS_FLIPS[binding["axis_flip"]], dtype=np.float64)
    transform_t = np.array([binding["translation_x"], binding["translation_y"], binding["translation_z"]], dtype=np.float64)
    unit_scale = float(V304.UNIT_SCALES[binding["unit_name"]])
    scale = float(binding["scale"])
    valid = confidence > 0
    rows: list[dict[str, Any]] = []
    for vi, cid in enumerate(CAMERA_IDS):
        mask_info = V304.try_read_mask(smc_path, cid)
        if mask_info is None:
            continue
        h, w = points.shape[1], points.shape[2]
        mask = resize_mask_for_points(mask_info, (h, w))
        cam = scaled_camera(smc_path, cid, mask_info["source_hw"], (h, w))
        view_points = points[vi].astype(np.float64) * signs[None, None, :] * unit_scale * scale + transform_t[None, None, :]
        eval_pts = sample_eval_points(view_points, valid[vi])
        xy, z = V304.project(eval_pts, cam["K"], cam["RT"], binding["rt_convention"])
        stats = V304.mask_iou_from_xy(xy, mask)
        stats["positive_depth_ratio"] = float((z > 0).mean()) if z.size else 0.0
        stats["camera_bound_score"] = float(stats["bbox_iou"] + stats["mask_coverage"] + stats["in_frame_ratio"] + 0.25 * stats["positive_depth_ratio"])
        stats.update({"group": group, "camera_id": cid, "smc": smc_path.name})
        rows.append(stats)
    return rows


def make_board(ranked: list[dict[str, Any]], prefix: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    path = BOARDS / f"{prefix}_camera_bound_projection_ranking.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    xs = np.arange(len(ranked))
    ax.bar(xs, [r["mean_camera_bound_score"] for r in ranked])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["group"] for r in ranked], rotation=55, ha="right")
    ax.set_ylabel("camera-bound score")
    ax.set_title(f"{prefix} Modal full-view camera-bound projection ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--prefix", default="V41400000000")
    args = parser.parse_args()
    pred_path = Path(args.pred)
    prefix = args.prefix
    with zipfile.ZipFile(pred_path) as zf:
        bad_member = zf.testzip()
    binding = load_best_binding()
    smc_path = V304.SMC_DIR / binding["smc"]
    rows: list[dict[str, Any]] = []
    with np.load(pred_path, allow_pickle=False) as z:
        confidence = z["confidence"].astype(np.float32)
        for group in GROUPS:
            key = f"{group}_world_points"
            if key not in z:
                continue
            points = z[key].astype(np.float32)
            rows.extend(score_group(group, points, confidence, binding, smc_path))
    write_csv(REPORTS / f"{prefix}_camera_bound_projection_metrics.csv", rows)
    ranked: list[dict[str, Any]] = []
    for group in sorted({r["group"] for r in rows}):
        gr = [r for r in rows if r["group"] == group]
        ranked.append({
            "group": group,
            "views": len(gr),
            "mean_camera_bound_score": float(np.mean([r["camera_bound_score"] for r in gr])),
            "mean_bbox_iou": float(np.mean([r["bbox_iou"] for r in gr])),
            "mean_mask_coverage": float(np.mean([r["mask_coverage"] for r in gr])),
            "mean_in_frame_ratio": float(np.mean([r["in_frame_ratio"] for r in gr])),
            "mean_positive_depth_ratio": float(np.mean([r["positive_depth_ratio"] for r in gr])),
            "mean_center_error": float(np.mean([r["center_error"] for r in gr])),
        })
    ranked.sort(key=lambda r: r["mean_camera_bound_score"], reverse=True)
    true = next((r for r in ranked if r["group"] == "true_camera_bound_transport"), None)
    strongest = next((r for r in ranked if r["group"] != "true_camera_bound_transport"), None)
    true_rank = next((i + 1 for i, r in enumerate(ranked) if r["group"] == "true_camera_bound_transport"), None)
    board = make_board(ranked, prefix)
    decision = {
        "created_utc": now(),
        "predictions_npz": str(pred_path),
        "npz_testzip": bad_member,
        "binding": binding,
        "ranked_groups": ranked,
        "true_rank": true_rank,
        "true_camera_bound_margin": float(true["mean_camera_bound_score"] - strongest["mean_camera_bound_score"]) if true and strongest else None,
        "true_beats_all_controls_camera_bound": bool(true_rank == 1),
        "board": board,
        "notes": [
            "Projection metrics use V304 best nonzero binding and full-view Modal predictions.",
            "This score complements residual and region metrics; it does not by itself establish mentor readiness.",
        ],
    }
    write_json(REPORTS / f"{prefix}_camera_bound_projection_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
