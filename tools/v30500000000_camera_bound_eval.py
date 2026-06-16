from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import importlib.util


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
OUTPUT = AUX / "output"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]


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
        raise RuntimeError("V304 binding did not pass; V305 cannot run")
    return doc["best"]


def prediction_sources() -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = [
        ("baseline", "V11700", AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"),
    ]
    v145 = OUTPUT / "V14500000000_control_prediction_samples"
    for p in sorted(v145.glob("*/*predictions_sample.npz")):
        items.append(("V145_sample", p.parent.name.replace("_seed0", ""), p))
    v300 = OUTPUT / "V30000000000_bundle_samples"
    for p in sorted(v300.glob("*_predictions_sample.npz")):
        group = p.name.replace("_seed0_predictions_sample.npz", "").replace("_predictions_sample.npz", "")
        items.append(("V300_sample", group, p))
    return [(family, group, path) for family, group, path in items if path.exists()]


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        points = z["world_points"].astype(np.float64)
        if "valid_mask" in z.files:
            valid = z["valid_mask"].astype(bool)
        elif "confidence" in z.files:
            valid = z["confidence"].astype(np.float32) > 0
        elif "world_points_conf" in z.files:
            valid = z["world_points_conf"].astype(np.float32) > 0
        else:
            valid = np.isfinite(points).all(axis=-1)
    return points, valid


def axis_signs(axis_flip: str) -> np.ndarray:
    return np.array(V304.AXIS_FLIPS[axis_flip], dtype=np.float64)


def resize_mask_for_points(mask_info: dict[str, Any], hw: tuple[int, int]) -> np.ndarray:
    source_mask = mask_info["mask"].astype(np.uint8) * 255
    return V304.resize_mask(source_mask, hw)


def scaled_camera(path: Path, cid: str, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    with h5py.File(str(path), "r") as f:
        g = f[f"Camera_Parameter/{cid}"]
        return {
            "K": V304.resize_intrinsic(g["K"][()], source_hw, target_hw),
            "RT": g["RT"][()],
            "D": g["D"][()] if "D" in g else np.zeros(5),
        }


def sample_eval_points(points: np.ndarray, valid: np.ndarray, max_points: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size > max_points:
        rng = np.random.default_rng(305)
        idx = rng.choice(idx, max_points, replace=False)
    return points.reshape(-1, 3)[idx], idx


def eval_one(family: str, group: str, pred_path: Path, binding: dict[str, Any], smc_path: Path) -> list[dict[str, Any]]:
    points, valid = load_points(pred_path)
    signs = axis_signs(binding["axis_flip"])
    transform_t = np.array([binding["translation_x"], binding["translation_y"], binding["translation_z"]], dtype=np.float64)
    unit_scale = float(V304.UNIT_SCALES[binding["unit_name"]])
    scale = float(binding["scale"])
    rows: list[dict[str, Any]] = []
    for vi, cid in enumerate(CAMERA_IDS):
        if vi >= points.shape[0]:
            continue
        mask_info = V304.try_read_mask(smc_path, cid)
        if mask_info is None:
            continue
        h, w = points.shape[1], points.shape[2]
        mask = resize_mask_for_points(mask_info, (h, w))
        cam = scaled_camera(smc_path, cid, mask_info["source_hw"], (h, w))
        view_points = points[vi] * signs[None, None, :] * unit_scale * scale + transform_t[None, None, :]
        eval_pts, _ = sample_eval_points(view_points, valid[vi], 12000)
        xy, z = V304.project(eval_pts, cam["K"], cam["RT"], binding["rt_convention"])
        stats = V304.mask_iou_from_xy(xy, mask)
        stats["positive_depth_ratio"] = float((z > 0).mean()) if z.size else 0.0
        stats.update({
            "family": family,
            "group": group,
            "prediction_path": str(pred_path),
            "camera_id": cid,
            "prediction_hw": f"{h}x{w}",
            "smc": smc_path.name,
        })
        rows.append(stats)
    return rows


def make_board(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    groups = sorted({r["group"] for r in rows})
    means = []
    for g in groups:
        gr = [r for r in rows if r["group"] == g]
        means.append({
            "group": g,
            "score": float(np.mean([r["bbox_iou"] + r["mask_coverage"] + r["in_frame_ratio"] for r in gr])),
            "bbox": float(np.mean([r["bbox_iou"] for r in gr])),
            "coverage": float(np.mean([r["mask_coverage"] for r in gr])),
        })
    fig, ax = plt.subplots(figsize=(14, 6))
    xs = np.arange(len(means))
    ax.bar(xs, [m["score"] for m in means], label="camera-bound score")
    ax.set_xticks(xs)
    ax.set_xticklabels([m["group"] for m in means], rotation=55, ha="right")
    ax.set_ylabel("bbox_iou + mask_coverage + in_frame")
    ax.set_title("V305 camera-bound verification under V304 best binding")
    ax.legend()
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V30500000000_camera_bound_visual.png", dpi=180)
    plt.close(fig)


def main() -> None:
    binding = load_best_binding()
    smc_path = V304.SMC_DIR / binding["smc"]
    rows: list[dict[str, Any]] = []
    for family, group, path in prediction_sources():
        rows.extend(eval_one(family, group, path, binding, smc_path))
    write_csv(REPORTS / "V30500000000_camera_bound_eval.csv", rows)
    grouped: list[dict[str, Any]] = []
    for group in sorted({r["group"] for r in rows}):
        gr = [r for r in rows if r["group"] == group]
        grouped.append({
            "group": group,
            "family": gr[0]["family"],
            "views": len(gr),
            "mean_bbox_iou": float(np.mean([r["bbox_iou"] for r in gr])),
            "mean_mask_coverage": float(np.mean([r["mask_coverage"] for r in gr])),
            "mean_in_frame_ratio": float(np.mean([r["in_frame_ratio"] for r in gr])),
            "mean_positive_depth_ratio": float(np.mean([r["positive_depth_ratio"] for r in gr])),
            "mean_center_error": float(np.mean([r["center_error"] for r in gr])),
            "camera_bound_score": float(np.mean([r["bbox_iou"] + r["mask_coverage"] + r["in_frame_ratio"] for r in gr])),
        })
    grouped.sort(key=lambda r: r["camera_bound_score"], reverse=True)
    true_rows = [r for r in grouped if r["group"] in {"true_surface_transformer", "true_surface_transport", "V11700"}]
    decision = {
        "created_utc": now(),
        "binding": binding,
        "groups_ranked": grouped,
        "true_improves_under_camera_bound_metrics": bool(true_rows and grouped and grouped[0]["group"] in {r["group"] for r in true_rows}),
        "next_route": "V310/V330 camera-bound dataset and matrix" if true_rows and grouped and grouped[0]["group"] in {r["group"] for r in true_rows} else "V350 learned binding / V360 alternative camera route",
        "notes": [
            "V305 evaluates existing predictions/samples under V304 best binding; it is verification, not a new formal matrix.",
            "V145/V300 samples may be downsampled, so metrics use prediction-specific K/mask scaling.",
        ],
    }
    write_json(REPORTS / "V30500000000_camera_bound_decision.json", decision)
    make_board(rows)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
