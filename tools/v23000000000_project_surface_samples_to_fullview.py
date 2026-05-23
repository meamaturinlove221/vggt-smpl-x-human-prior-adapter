from __future__ import annotations

import csv
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"

SURFACE = OUTPUT / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"
V220 = OUTPUT / "V22000000000_full_mentor_gate_predictions"
OUT = OUTPUT / "V23000000000_fullview_predictions"

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
REGION_IDS = {
    "full_body": None,
    "head_face": 1,
    "hairline": 2,
    "left_hand": 3,
    "right_hand": 4,
}


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def scatter(ax, points: np.ndarray, title: str, max_points: int = 5000, mask: np.ndarray | None = None) -> None:
    pts = points.reshape(-1, 3)
    if mask is not None and mask.shape == points.shape[:3]:
        pts = points[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(23000000000)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    if pts.size == 0:
        ax.set_title(title + " empty", fontsize=7)
        ax.set_axis_off()
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.35, c=pts[:, 2], cmap="viridis", linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_axis_off()


def build_fullview() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SURFACE, allow_pickle=False) as s:
        valid = s["valid_mask"].astype(bool)
        base_wp = s["world_points"].astype(np.float32)
        base_depth = s["depth"].astype(np.float32)
        base_conf = s["confidence"].astype(np.float32)
        geom_normal = s["normal"].astype(np.float32)
        region_label = s["region_label"].astype(np.int16)

    valid_idx = np.flatnonzero(valid.reshape(-1))
    report: dict[str, Any] = {
        "created_utc": now(),
        "surface_index": str(SURFACE),
        "valid_pixel_count": int(valid_idx.size),
        "groups": {},
        "projection_limitation": "V220 predictions cover sampled valid pixels; dense full-view maps are reconstructed by assigning samples to the first valid pixels from V920 valid_mask.",
    }
    for group in GROUPS:
        src = V220 / f"{group}_seed0" / "predictions.npz"
        if not src.exists():
            report["groups"][group] = {"exists": False}
            continue
        with zipfile.ZipFile(src, "r") as zf:
            bad = zf.testzip()
        with np.load(src, allow_pickle=False) as z:
            sample_wp = z["world_points_sample"].astype(np.float32)
            sample_normal = z["normal_sample"].astype(np.float32)
            sample_geom = z["geometric_normal_sample"].astype(np.float32)
            sample_res = z["normal_residual_sample"].astype(np.float32)
            gate = z["gate_sample"].astype(np.float32)
        n = min(sample_wp.shape[0], valid_idx.size)
        wp = base_wp.copy().reshape(-1, 3)
        normal = geom_normal.copy().reshape(-1, 3)
        learned_geom = geom_normal.copy().reshape(-1, 3)
        residual = np.zeros_like(wp, dtype=np.float32)
        normal_conf = np.zeros(base_conf.shape, dtype=np.float32).reshape(-1)
        confidence = base_conf.copy().reshape(-1)
        wp[valid_idx[:n]] = sample_wp[:n]
        normal[valid_idx[:n]] = sample_normal[:n]
        learned_geom[valid_idx[:n]] = sample_geom[:n]
        residual[valid_idx[:n]] = sample_res[:n]
        normal_conf[valid_idx[:n]] = np.clip(gate[:n, 0], 0.0, 1.0)
        out_dir = OUT / f"{group}_seed0"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "predictions.npz",
            world_points=wp.reshape(base_wp.shape).astype(np.float32),
            depth=base_depth.astype(np.float32),
            confidence=confidence.reshape(base_conf.shape).astype(np.float32),
            normal=normal.reshape(base_wp.shape).astype(np.float32),
            normal_conf=normal_conf.reshape(base_conf.shape).astype(np.float32),
            learned_normal=normal.reshape(base_wp.shape).astype(np.float32),
            geometric_normal=learned_geom.reshape(base_wp.shape).astype(np.float32),
            normal_residual=residual.reshape(base_wp.shape).astype(np.float32),
            valid_mask=valid,
            region_label=region_label,
        )
        eval_doc = {
            "created_utc": now(),
            "group": group,
            "seed": 0,
            "source_v220_prediction": str(src),
            "fullview_shape": list(base_wp.shape),
            "npz_bad_member": bad,
            "valid_assigned_samples": int(n),
            "projection_coverage": float(n / max(1, valid_idx.size)),
            "normal_nonzero_ratio": float((np.linalg.norm(normal.reshape(base_wp.shape), axis=-1)[valid] > 1e-6).mean()),
            "learned_normal_residual_mean": float(np.linalg.norm(residual.reshape(base_wp.shape), axis=-1)[valid].mean()),
            "projection_limitation": report["projection_limitation"],
        }
        write_json(out_dir / "eval.json", eval_doc)
        write_json(out_dir / "source_manifest.json", {
            "created_utc": now(),
            "route": "V230 surface sample to full-view projection",
            "old_residual_composer": False,
            "teacher_postcompose": False,
            "active_candidate_replaced": False,
            "source": str(src),
        })
        report["groups"][group] = eval_doc
    write_json(REPORTS / "V23000000000_projection_report.json", report)
    return report


def region_metrics() -> dict[str, Any]:
    with np.load(SURFACE, allow_pickle=False) as s:
        base_wp = s["world_points"].astype(np.float32)
        valid = s["valid_mask"].astype(bool)
        region_label = s["region_label"].astype(np.int16)
    rows: list[dict[str, Any]] = []
    base_spread = np.linalg.norm(base_wp - base_wp[valid].mean(axis=0), axis=-1)
    for group in GROUPS:
        pred = OUT / f"{group}_seed0" / "predictions.npz"
        if not pred.exists():
            continue
        with np.load(pred, allow_pickle=False) as z:
            wp = z["world_points"]
            normal = z["normal"]
            residual = z["normal_residual"]
            normal_conf = z["normal_conf"]
        delta_to_base = np.linalg.norm(wp - base_wp, axis=-1)
        spread = np.linalg.norm(wp - wp[valid].mean(axis=0), axis=-1)
        normal_nonzero = np.linalg.norm(normal, axis=-1) > 1e-6
        residual_norm = np.linalg.norm(residual, axis=-1)
        for region, rid in REGION_IDS.items():
            mask = valid if rid is None else (valid & (region_label == rid))
            count = int(mask.sum())
            rows.append({
                "group": group,
                "seed": 0,
                "region": region,
                "status": "evaluated" if count > 0 else "empty_mask",
                "pixel_count": count,
                "mean_delta_to_v920_base": float(delta_to_base[mask].mean()) if count else math.nan,
                "mean_spatial_spread": float(spread[mask].mean()) if count else math.nan,
                "base_spatial_spread": float(base_spread[mask].mean()) if count else math.nan,
                "normal_nonzero_ratio": float(normal_nonzero[mask].mean()) if count else math.nan,
                "normal_residual_mean": float(residual_norm[mask].mean()) if count else math.nan,
                "normal_conf_mean": float(normal_conf[mask].mean()) if count else math.nan,
                "region_metric_source": "V920 full-view region_label; connected-components/reprojection not yet available",
            })
    write_csv(REPORTS / "V23000000000_region_metrics.csv", rows)
    summary = {
        "created_utc": now(),
        "rows": len(rows),
        "not_evaluated_count": sum(1 for r in rows if r["status"] != "evaluated"),
        "regions": list(REGION_IDS),
        "proxy_limitations": ["connected_component_not_evaluated", "reprojection_consistency_not_available"],
    }
    write_json(REPORTS / "V23000000000_region_summary.json", summary)
    return summary


def visual_boards() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    groups = [
        "true_surface_transformer",
        "random_semantic",
        "strong_shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "observation_only",
    ]
    data = {}
    for group in groups:
        p = OUT / f"{group}_seed0" / "predictions.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                data[group] = {k: z[k] for k in ["world_points", "normal", "normal_residual", "valid_mask", "region_label"]}

    fullbody = BOARDS / "V23000000000_fullbody.png"
    fig = plt.figure(figsize=(18, 4))
    for i, group in enumerate(groups, start=1):
        ax = fig.add_subplot(1, len(groups), i, projection="3d")
        scatter(ax, data[group]["world_points"], group, max_points=6000)
    fig.tight_layout()
    fig.savefig(fullbody, dpi=180)
    plt.close(fig)

    closeup = BOARDS / "V23000000000_head_hair_hand.png"
    regions = [("head_face", 1), ("hairline", 2), ("left_hand", 3), ("right_hand", 4)]
    fig = plt.figure(figsize=(18, 10))
    panel = 1
    for region, rid in regions:
        for group in groups[:4]:
            mask = data[group]["valid_mask"] & (data[group]["region_label"] == rid)
            ax = fig.add_subplot(len(regions), 4, panel, projection="3d")
            scatter(ax, data[group]["world_points"], f"{region} {group}", mask=mask, max_points=3000)
            panel += 1
    fig.tight_layout()
    fig.savefig(closeup, dpi=180)
    plt.close(fig)

    controls = BOARDS / "V23000000000_controls.png"
    fig = plt.figure(figsize=(18, 4))
    for i, group in enumerate(groups, start=1):
        ax = fig.add_subplot(1, len(groups), i, projection="3d")
        scatter(ax, data[group]["world_points"], f"control {group}", max_points=5000)
    fig.tight_layout()
    fig.savefig(controls, dpi=180)
    plt.close(fig)

    normals = BOARDS / "V23000000000_normals.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, group in zip(axes.flat, groups):
        arr = np.linalg.norm(data[group]["normal_residual"], axis=-1)[0]
        ax.imshow(arr, cmap="magma")
        ax.set_title(group, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(normals, dpi=180)
    plt.close(fig)

    manifest = {
        "created_utc": now(),
        "boards": {
            "fullbody": str(fullbody),
            "head_hair_hand": str(closeup),
            "controls": str(controls),
            "normals": str(normals),
        },
    }
    write_json(REPORTS / "V23000000000_visual_boards_manifest.json", manifest)
    return manifest


def decision() -> dict[str, Any]:
    validation = json.loads((REPORTS / "V22000000000_matrix_validation.json").read_text(encoding="utf-8"))
    ranking = validation.get("ranking", [])
    top = ranking[0]["group"] if ranking else None
    projection = json.loads((REPORTS / "V23000000000_projection_report.json").read_text(encoding="utf-8"))
    region = json.loads((REPORTS / "V23000000000_region_summary.json").read_text(encoding="utf-8"))
    mentor_ready = (
        top == "true_surface_transformer"
        and projection.get("groups", {}).get("true_surface_transformer", {}).get("projection_coverage", 0) >= 0.99
        and region.get("not_evaluated_count") == 0
        and not region.get("proxy_limitations")
    )
    payload = {
        "created_utc": now(),
        "mentor_ready": bool(mentor_ready),
        "top_group": top,
        "true_beats_all_v220_controls": top == "true_surface_transformer",
        "fullview_predictions_created": True,
        "fullview_prediction_shape": [6, 518, 518, 3],
        "remaining_limitations": [] if mentor_ready else [
            "V230 is projection from sampled surface predictions, not dense end-to-end full-view inference.",
            "Region metrics still lack connected-component and reprojection proof.",
            "Visual boards are full-view 3D boards, but mentor-ready visible improvement requires dense route validation.",
        ],
        "next_route_required": not mentor_ready,
        "next_route": "V240 dense full-view surface transport inference" if not mentor_ready else None,
    }
    write_json(REPORTS / "V23000000000_decision.json", payload)
    return payload


def main() -> None:
    build_fullview()
    region_metrics()
    visual_boards()
    print(json.dumps(decision(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
