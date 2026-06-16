from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
PRED = AUX / "output" / "V40800000000_modal_fullview_predictions" / "predictions.npz"
SURFACE = AUX / "output" / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"


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


def sample_region(points: np.ndarray, mask: np.ndarray, max_points: int = 3500) -> np.ndarray:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if idx.size > max_points:
        rng = np.random.default_rng(409)
        idx = rng.choice(idx, max_points, replace=False)
    return points.reshape(-1, 3)[idx].astype(np.float32)


def region_masks(
    region_label: np.ndarray,
    part_id: np.ndarray | None = None,
    canonical_xyz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Infer route regions from V920 labels with conservative fallbacks.

    V920 writes region labels directly from semantic masks:
    1=head_face, 2=hairline, 3=left_hand, 4=right_hand.  Earlier board code
    treated hand labels as 6..11, which made the hand metrics falsely empty.
    """
    masks = {
        "full_body": region_label >= 0,
        "head_face": region_label == 1,
        "hairline": region_label == 2,
        "left_hand": region_label == 3,
        "right_hand": region_label == 4,
    }
    if part_id is None:
        return masks

    # Fallback for older surface datasets that did not preserve hand region
    # labels: part_id 4/5 correspond to left/right hands in the V920 crosstab.
    if not masks["left_hand"].any():
        masks["left_hand"] = part_id == 4
    if not masks["right_hand"].any():
        masks["right_hand"] = part_id == 5

    # Last fallback: split generic hand-like part ids by canonical x position.
    if canonical_xyz is not None:
        hand_like = np.isin(part_id, [4, 5, 6])
        if hand_like.any() and (not masks["left_hand"].any() or not masks["right_hand"].any()):
            x = canonical_xyz[..., 0]
            median_x = np.median(x[hand_like])
            if not masks["left_hand"].any():
                masks["left_hand"] = hand_like & (x >= median_x)
            if not masks["right_hand"].any():
                masks["right_hand"] = hand_like & (x < median_x)
    return masks


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(PRED, allow_pickle=False) as z:
        groups = ["true_camera_bound_transport", "random_surface_semantic", "local_knn_smoothing_surface"]
        points = {g: z[f"{g}_world_points"].astype(np.float32) for g in groups}
        normals = {g: z[f"{g}_normal"].astype(np.float32) for g in groups}
    with np.load(SURFACE, allow_pickle=False) as z:
        region = z["region_label"].astype(np.int16)
        valid = z["valid_mask"].astype(bool)
        part_id = z["part_id"].astype(np.int16) if "part_id" in z else None
        canonical_xyz = z["canonical_surface_xyz"].astype(np.float32) if "canonical_surface_xyz" in z else None
    masks = region_masks(region, part_id=part_id, canonical_xyz=canonical_xyz)
    rows = []
    baseline = points["random_surface_semantic"]
    for group in groups:
        for name, mask in masks.items():
            m = mask & valid
            if not m.any():
                rows.append({"group": group, "region": name, "status": "empty"})
                continue
            delta = np.linalg.norm(points[group][m] - baseline[m], axis=-1)
            normal_nonzero = (np.linalg.norm(normals[group][m], axis=-1) > 0.1).mean()
            rows.append({
                "group": group,
                "region": name,
                "status": "ok",
                "points": int(m.sum()),
                "mean_delta_vs_random": float(delta.mean()),
                "p95_delta_vs_random": float(np.percentile(delta, 95)),
                "normal_nonzero_ratio": float(normal_nonzero),
            })
    write_csv(REPORTS / "V40900000000_region_metrics.csv", rows)
    # Full body and close-up scatter boards.
    for board_name, region_names in [
        ("V40900000000_fullbody_pointcloud.png", ["full_body"]),
        ("V40900000000_head_hair_hand_closeups.png", ["head_face", "hairline", "left_hand", "right_hand"]),
    ]:
        fig = plt.figure(figsize=(5 * len(groups), 4 * len(region_names)))
        plot_i = 1
        for rn in region_names:
            mask = masks[rn] & valid
            for group in groups:
                ax = fig.add_subplot(len(region_names), len(groups), plot_i, projection="3d")
                pts = sample_region(points[group], mask)
                if pts.size:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.55)
                ax.set_title(f"{rn}\n{group}")
                ax.set_axis_off()
                plot_i += 1
        fig.tight_layout()
        out = BOARDS / board_name
        BOARDS.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        plt.close(fig)
    # Normal comparison board.
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 4))
    for ax, group in zip(axes, groups):
        n = normals[group][0]
        img = ((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(group)
        ax.axis("off")
    fig.tight_layout()
    normal_board = BOARDS / "V40900000000_normal_comparison.png"
    fig.savefig(normal_board, dpi=180)
    plt.close(fig)
    report = {
        "created_utc": now(),
        "predictions": str(PRED),
        "boards": {
            "fullbody": str(BOARDS / "V40900000000_fullbody_pointcloud.png"),
            "head_hair_hand": str(BOARDS / "V40900000000_head_hair_hand_closeups.png"),
            "normal": str(normal_board),
        },
        "groups": groups,
        "mentor_visual_gate": "partial",
        "limitations": [
            "Boards use V408 Modal full-view predictions and derived region labels.",
            "They are new full-view point-cloud boards, but mentor-ready still depends on final decision gates and raw camera-bound projection ranking.",
        ],
    }
    write_json(REPORTS / "V40900000000_visual_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
