from __future__ import annotations

import csv
import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
DEFAULT_PRED = AUX / "output" / "V41100000000_modal_fullview_core_controls" / "predictions.npz"
SURFACE = AUX / "output" / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"
SOURCE_POINTS = AUX / "output" / "V3600000000_fullview_dataset_v2" / "true_full.npz"

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
REGIONS = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]


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


def region_masks(region_label: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "full_body": valid,
        "head_face": (region_label == 1) & valid,
        "hairline": (region_label == 2) & valid,
        "left_hand": (region_label == 3) & valid,
        "right_hand": (region_label == 4) & valid,
    }


def sample_region(points: np.ndarray, mask: np.ndarray, max_points: int = 3000) -> np.ndarray:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if idx.size > max_points:
        rng = np.random.default_rng(412)
        idx = rng.choice(idx, size=max_points, replace=False)
    return points.reshape(-1, 3)[idx].astype(np.float32)


def load_source_points() -> np.ndarray | None:
    if not SOURCE_POINTS.exists():
        return None
    with np.load(SOURCE_POINTS, allow_pickle=False) as z:
        if "world_points" not in z:
            return None
        return z["world_points"].astype(np.float32)


def plot_boards(
    points_by_group: dict[str, np.ndarray],
    normals_by_group: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    selected = [
        "true_camera_bound_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "observation_only",
        "support_only",
    ]
    fullbody_path = BOARDS / f"{prefix}_fullbody_core_controls.png"
    fig = plt.figure(figsize=(4.2 * len(selected), 4.8))
    for i, group in enumerate(selected, start=1):
        ax = fig.add_subplot(1, len(selected), i, projection="3d")
        pts = sample_region(points_by_group[group], masks["full_body"], max_points=4000)
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.8, alpha=0.5)
        ax.set_title(group, fontsize=8)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fullbody_path, dpi=180)
    plt.close(fig)

    closeup_path = BOARDS / f"{prefix}_head_hair_hand_core_controls.png"
    closeup_groups = ["true_camera_bound_transport", "random_surface_semantic", "local_knn_smoothing_surface", "observation_only", "support_only"]
    closeup_regions = ["head_face", "hairline", "left_hand", "right_hand"]
    fig = plt.figure(figsize=(4.0 * len(closeup_groups), 3.8 * len(closeup_regions)))
    plot_i = 1
    for region in closeup_regions:
        for group in closeup_groups:
            ax = fig.add_subplot(len(closeup_regions), len(closeup_groups), plot_i, projection="3d")
            pts = sample_region(points_by_group[group], masks[region], max_points=1800)
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, alpha=0.6)
            ax.set_title(f"{region}\n{group}", fontsize=8)
            ax.set_axis_off()
            plot_i += 1
    fig.tight_layout()
    fig.savefig(closeup_path, dpi=180)
    plt.close(fig)

    normal_path = BOARDS / f"{prefix}_normal_core_controls.png"
    normal_groups = ["true_camera_bound_transport", "random_surface_semantic", "local_knn_smoothing_surface", "observation_only", "support_only"]
    fig, axes = plt.subplots(1, len(normal_groups), figsize=(4.2 * len(normal_groups), 4.2))
    for ax, group in zip(axes, normal_groups):
        n = normals_by_group[group][0].astype(np.float32)
        img = ((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(group, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(normal_path, dpi=180)
    plt.close(fig)

    return {
        "fullbody": str(fullbody_path),
        "head_hair_hand": str(closeup_path),
        "normal": str(normal_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=str(DEFAULT_PRED))
    parser.add_argument("--prefix", default="V41200000000")
    args = parser.parse_args()
    pred_path = Path(args.pred)
    prefix = args.prefix

    with zipfile.ZipFile(pred_path) as zf:
        bad_member = zf.testzip()
    with np.load(SURFACE, allow_pickle=False) as z:
        region_label = z["region_label"].astype(np.int16)
        valid = z["valid_mask"].astype(bool)
    masks = region_masks(region_label, valid)
    source_points = load_source_points()

    rows: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    points_by_group: dict[str, np.ndarray] = {}
    normals_by_group: dict[str, np.ndarray] = {}
    with np.load(pred_path, allow_pickle=False) as z:
        for group in GROUPS:
            wp_key = f"{group}_world_points"
            n_key = f"{group}_normal"
            if wp_key not in z or n_key not in z:
                rows.append({"group": group, "region": "all", "status": "missing_keys"})
                continue
            points = z[wp_key].astype(np.float32)
            normals = z[n_key].astype(np.float32)
            points_by_group[group] = points
            normals_by_group[group] = normals
            full_scores = []
            for region in REGIONS:
                mask = masks[region]
                if not mask.any():
                    rows.append({"group": group, "region": region, "status": "empty"})
                    continue
                normal_norm = np.linalg.norm(normals[mask], axis=-1)
                normal_nonzero = float((normal_norm > 0.1).mean())
                mean_delta = None
                if source_points is not None:
                    delta = np.linalg.norm(points[mask] - source_points[mask], axis=-1)
                    mean_delta = float(delta.mean())
                    p95_delta = float(np.percentile(delta, 95))
                else:
                    p95_delta = None
                rows.append({
                    "group": group,
                    "region": region,
                    "status": "ok",
                    "points": int(mask.sum()),
                    "mean_delta_vs_input": mean_delta,
                    "p95_delta_vs_input": p95_delta,
                    "normal_nonzero_ratio": normal_nonzero,
                })
                if region == "full_body":
                    full_scores.append(mean_delta if mean_delta is not None else 0.0)
            summary = next((s for s in full_scores if s is not None), 0.0)
            ranking.append({
                "group": group,
                "full_body_mean_delta_vs_input": float(summary),
                "normal_nonzero_ratio": float((np.linalg.norm(normals[valid], axis=-1) > 0.1).mean()),
            })
    write_csv(REPORTS / f"{prefix}_region_metrics.csv", rows)

    # A smaller residual against input is better for this Modal full-view export
    # because the trained head is a constrained camera-bound residual, not a
    # free confidence-only scorer.
    ranking.sort(key=lambda r: r["full_body_mean_delta_vs_input"])
    true_row = next(r for r in ranking if r["group"] == "true_camera_bound_transport")
    stronger_controls = [r for r in ranking if r["group"] != "true_camera_bound_transport" and r["full_body_mean_delta_vs_input"] <= true_row["full_body_mean_delta_vs_input"]]
    regions_ok = all(row["status"] == "ok" for row in rows if row["region"] in REGIONS)
    normals_ok = all(r["normal_nonzero_ratio"] > 0.99 for r in ranking)
    boards = plot_boards(points_by_group, normals_by_group, masks, prefix)
    decision = {
        "created_utc": now(),
        "predictions_npz": str(pred_path),
        "npz_testzip": bad_member,
        "groups": GROUPS,
        "regions": REGIONS,
        "regions_ok": bool(regions_ok),
        "normals_ok": bool(normals_ok),
        "ranked_by_full_body_delta_vs_input_low_is_better": ranking,
        "true_beats_all_controls_on_delta": len(stronger_controls) == 0,
        "controls_not_beaten": stronger_controls,
        "mentor_ready_visual_gate": bool(regions_ok and normals_ok and len(stronger_controls) == 0),
        "boards": boards,
        "notes": [
            "V411 is a Modal GPU full-view export for the 10 core controls.",
            "This evaluates full body, head-face, hairline, left hand, and right hand using V920 region labels.",
            "Delta-vs-input is a conservative residual ranking; camera-bound mentor readiness also depends on projection and source-manifest gates.",
        ],
    }
    write_json(REPORTS / f"{prefix}_core_controls_eval.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
