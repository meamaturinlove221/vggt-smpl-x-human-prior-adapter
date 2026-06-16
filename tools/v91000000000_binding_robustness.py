from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
PRED = AUX / "output" / "V41500000000_modal_camera_mask_fullview_core_controls" / "predictions.npz"
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


def load_binding() -> dict[str, Any]:
    return json.loads((REPORTS / "V30400000000_best_binding.json").read_text(encoding="utf-8"))["best"]


def load_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = REPORTS / "V30400000000_coordinate_binding_candidates.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows[:5]


def scaled_camera(path: Path, cid: str, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    with h5py.File(str(path), "r") as f:
        g = f[f"Camera_Parameter/{cid}"]
        return {"K": V304.resize_intrinsic(g["K"][()], source_hw, target_hw), "RT": g["RT"][()]}


def score_one(points: np.ndarray, confidence: np.ndarray, binding: dict[str, Any], include_views: list[str]) -> dict[str, float]:
    smc_path = V304.SMC_DIR / binding["smc"]
    signs = np.array(V304.AXIS_FLIPS[binding["axis_flip"]], dtype=np.float64)
    t = np.array([float(binding["translation_x"]), float(binding["translation_y"]), float(binding["translation_z"])], dtype=np.float64)
    unit_scale = float(V304.UNIT_SCALES[binding["unit_name"]])
    scale = float(binding["scale"])
    rows = []
    for vi, cid in enumerate(CAMERA_IDS):
        if cid not in include_views:
            continue
        mask_info = V304.try_read_mask(smc_path, cid)
        if mask_info is None:
            continue
        h, w = points.shape[1], points.shape[2]
        mask = V304.resize_mask(mask_info["mask"].astype(np.uint8) * 255, (h, w))
        cam = scaled_camera(smc_path, cid, mask_info["source_hw"], (h, w))
        valid = confidence[vi] > 0
        idx = np.flatnonzero(valid.reshape(-1))
        if idx.size > 30000:
            rng = np.random.default_rng(910 + vi)
            idx = rng.choice(idx, 30000, replace=False)
        pts = points[vi].reshape(-1, 3)[idx].astype(np.float64)
        pts = pts * signs[None, :] * unit_scale * scale + t[None, :]
        xy, z = V304.project(pts, cam["K"], cam["RT"], binding["rt_convention"])
        stats = V304.mask_iou_from_xy(xy, mask)
        stats["positive_depth_ratio"] = float((z > 0).mean()) if z.size else 0.0
        stats["score"] = float(stats["bbox_iou"] + stats["mask_coverage"] + stats["in_frame_ratio"] + 0.25 * stats["positive_depth_ratio"])
        rows.append(stats)
    return {
        "score": float(np.mean([r["score"] for r in rows])) if rows else 0.0,
        "bbox_iou": float(np.mean([r["bbox_iou"] for r in rows])) if rows else 0.0,
        "mask_coverage": float(np.mean([r["mask_coverage"] for r in rows])) if rows else 0.0,
        "in_frame_ratio": float(np.mean([r["in_frame_ratio"] for r in rows])) if rows else 0.0,
        "views": float(len(rows)),
    }


def rank_groups(binding: dict[str, Any], include_views: list[str]) -> list[dict[str, Any]]:
    with np.load(PRED, allow_pickle=False) as z:
        confidence = z["confidence"].astype(np.float32)
        ranked = []
        for group in GROUPS:
            points = z[f"{group}_world_points"].astype(np.float32)
            stats = score_one(points, confidence, binding, include_views)
            ranked.append({"group": group, **stats})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def margin_from_ranked(ranked: list[dict[str, Any]]) -> tuple[int, float]:
    true_idx = next(i for i, r in enumerate(ranked) if r["group"] == "true_camera_bound_transport")
    strongest = next(r for r in ranked if r["group"] != "true_camera_bound_transport")
    true = ranked[true_idx]
    return true_idx + 1, float(true["score"] - strongest["score"])


def make_board(view_rows: list[dict[str, Any]], boot_rows: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    path = BOARDS / "V91000000000_binding_robustness.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r["case"] for r in view_rows]
    margins = [float(r["true_margin"]) for r in view_rows]
    axes[0].bar(range(len(labels)), margins)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_title("View ablation true margin")
    axes[0].set_ylabel("score margin")
    boot = [float(r["true_margin"]) for r in boot_rows]
    axes[1].hist(boot, bins=12)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Bootstrap margin distribution")
    axes[1].set_xlabel("score margin")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    binding = load_binding()
    top5 = load_candidates()
    view_rows = []
    for remove in ["none", *CAMERA_IDS]:
        include = CAMERA_IDS[:] if remove == "none" else [c for c in CAMERA_IDS if c != remove]
        ranked = rank_groups(binding, include)
        rank, margin = margin_from_ranked(ranked)
        view_rows.append({"case": f"remove_{remove}", "views": ",".join(include), "true_rank": rank, "true_margin": margin})
    write_csv(REPORTS / "V91000000000_view_ablation.csv", view_rows)

    rng = np.random.default_rng(910)
    boot_rows = []
    for i in range(200):
        include = list(rng.choice(CAMERA_IDS, size=len(CAMERA_IDS), replace=True))
        ranked = rank_groups(binding, include)
        rank, margin = margin_from_ranked(ranked)
        boot_rows.append({"bootstrap_id": i, "views": ",".join(include), "true_rank": rank, "true_margin": margin})
    write_csv(REPORTS / "V91000000000_binding_bootstrap.csv", boot_rows)

    perturb_rows = []
    base_t = np.array([binding["translation_x"], binding["translation_y"], binding["translation_z"]], dtype=float)
    variants: list[tuple[str, dict[str, Any]]] = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        b = dict(binding)
        b["scale"] = scale
        variants.append((f"scale_{scale}", b))
    for axis in ["identity", "flip_x", "flip_y", "flip_z", "flip_xy", "flip_xz", "flip_yz", "flip_xyz"]:
        b = dict(binding)
        b["axis_flip"] = axis
        variants.append((f"axis_{axis}", b))
    for conv in ["rt_world_to_camera", "inverse_rt_camera_to_world", "rt_transposed_rotation"]:
        b = dict(binding)
        b["rt_convention"] = conv
        variants.append((f"rt_{conv}", b))
    for j, off in enumerate([(-0.05, 0, 0), (0.05, 0, 0), (0, -0.05, 0), (0, 0.05, 0), (0, 0, -0.05), (0, 0, 0.05)]):
        b = dict(binding)
        t = base_t + np.array(off)
        b["translation_x"], b["translation_y"], b["translation_z"] = [float(x) for x in t]
        variants.append((f"translation_{j}", b))
    for name, b in variants:
        try:
            ranked = rank_groups(b, CAMERA_IDS)
            rank, margin = margin_from_ranked(ranked)
        except Exception:
            rank, margin = 999, -999.0
        perturb_rows.append({"case": name, "true_rank": rank, "true_margin": margin})
    write_csv(REPORTS / "V91000000000_binding_perturbation.csv", perturb_rows)
    board = make_board(view_rows, boot_rows)
    margins = [float(r["true_margin"]) for r in boot_rows]
    view_margins = [float(r["true_margin"]) for r in view_rows]
    summary = {
        "created_utc": now(),
        "top5_binding_candidates": top5,
        "base_binding": binding,
        "base_true_rank": int(view_rows[0]["true_rank"]),
        "base_true_margin": float(view_rows[0]["true_margin"]),
        "view_ablation_all_positive": all(float(r["true_margin"]) > 0 and int(r["true_rank"]) == 1 for r in view_rows),
        "bootstrap_positive_fraction": float(np.mean(np.array(margins) > 0)),
        "bootstrap_rank1_fraction": float(np.mean([int(r["true_rank"]) == 1 for r in boot_rows])),
        "bootstrap_margin_mean": float(np.mean(margins)),
        "bootstrap_margin_p05": float(np.percentile(margins, 5)),
        "bootstrap_margin_p95": float(np.percentile(margins, 95)),
        "view_margin_min": float(np.min(view_margins)),
        "perturbation_rows": perturb_rows,
        "binding_robust": bool(all(float(r["true_margin"]) > 0 and int(r["true_rank"]) == 1 for r in view_rows) and np.percentile(margins, 5) > 0),
        "board": board,
    }
    write_json(REPORTS / "V91000000000_binding_robustness.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
