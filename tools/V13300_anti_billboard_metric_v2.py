from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center[None]
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = centered @ vecs
    return center, vals, vecs, proj


def _entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def connected_component_proxy(points: np.ndarray, bins: tuple[int, int, int] = (16, 12, 8)) -> dict[str, float | int]:
    _center, _vals, _axes, proj = pca_frame(points)
    ranges = np.maximum(np.ptp(proj, axis=0), 1e-9)
    ijk = np.clip(np.floor((proj - proj.min(axis=0)[None]) / ranges[None] * np.array(bins)).astype(int), 0, np.array(bins) - 1)
    occupied = {tuple(v.tolist()) for v in ijk}
    if not occupied:
        return {"occupied_voxels": 0, "largest_component_ratio": 0.0, "component_count": 0}
    remaining = set(occupied)
    sizes: list[int] = []
    nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        size = 1
        while stack:
            cur = stack.pop()
            for d in nbrs:
                nxt = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    size += 1
        sizes.append(size)
    return {
        "occupied_voxels": len(occupied),
        "largest_component_ratio": float(max(sizes) / max(1, len(occupied))),
        "component_count": len(sizes),
    }


def anti_billboard_metric_v2(points: np.ndarray, body_part: np.ndarray | None = None) -> dict[str, Any]:
    _center, vals, _axes, proj = pca_frame(points)
    ranges = np.maximum(np.ptp(proj, axis=0), 1e-9)
    long_range, mid_range, thin_range = [float(x) for x in ranges]
    bins_a = np.clip(np.floor((proj[:, 0] - proj[:, 0].min()) / long_range * 18).astype(int), 0, 17)
    bins_b = np.clip(np.floor((proj[:, 1] - proj[:, 1].min()) / mid_range * 14).astype(int), 0, 13)
    bins_t = np.clip(np.floor((proj[:, 2] - proj[:, 2].min()) / thin_range * 10).astype(int), 0, 9)

    section_count = 0
    multi = 0
    dense = 0
    front_back = 0
    layers: list[int] = []
    entropies: list[float] = []
    areas: list[float] = []
    for a in range(18):
        for b in range(14):
            m = (bins_a == a) & (bins_b == b)
            count = int(m.sum())
            if count < 10:
                continue
            section_count += 1
            counts = np.bincount(bins_t[m], minlength=10)
            occ = int(np.count_nonzero(counts))
            layers.append(occ)
            entropies.append(_entropy(counts) / np.log2(10))
            multi += int(occ >= 3)
            dense += int(occ >= 4)
            front_back += int(counts[:2].sum() > 0 and counts[-2:].sum() > 0)
            areas.append(float(count))

    multi_ratio = multi / max(section_count, 1)
    dense_ratio = dense / max(section_count, 1)
    front_back_ratio = front_back / max(section_count, 1)
    mean_layers = float(np.mean(layers)) if layers else 0.0
    entropy = float(np.mean(entropies)) if entropies else 0.0
    area_cv = float(np.std(areas) / max(np.mean(areas), 1e-9)) if areas else 1.0
    comp = connected_component_proxy(points)

    part_balance = 1.0
    part_component_penalty = 0.0
    if body_part is not None and len(body_part) == len(points):
        body = np.asarray(body_part)
        ratios = []
        for part in np.unique(body):
            m = body == part
            if int(m.sum()) >= 100:
                ratios.append(connected_component_proxy(points[m])["largest_component_ratio"])
        if ratios:
            part_balance = float(np.mean(ratios))
            part_component_penalty = float(np.mean([max(0.0, 0.70 - r) for r in ratios]))

    score = (
        0.22 * min(thin_range / long_range / 0.45, 1.0)
        + 0.20 * multi_ratio
        + 0.16 * dense_ratio
        + 0.15 * front_back_ratio
        + 0.12 * entropy
        + 0.10 * float(comp["largest_component_ratio"])
        + 0.05 * part_balance
    )
    risk = 1.0 - score + 0.08 * min(area_cv, 2.0) + 0.08 * part_component_penalty
    fail = bool(score < 0.58 or multi_ratio < 0.35 or front_back_ratio < 0.22 or comp["largest_component_ratio"] < 0.58)
    return {
        "pca_thickness_ratio": float(thin_range / long_range),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
        "section_count": int(section_count),
        "multi_layer_section_ratio": float(multi_ratio),
        "dense_section_ratio": float(dense_ratio),
        "front_back_separation_ratio": float(front_back_ratio),
        "mean_thin_axis_layers": float(mean_layers),
        "depth_layer_entropy": float(entropy),
        "torso_cross_section_area_cv_proxy": float(area_cv),
        "occupied_voxels": comp["occupied_voxels"],
        "largest_component_ratio": comp["largest_component_ratio"],
        "component_count": comp["component_count"],
        "part_largest_component_ratio_mean": float(part_balance),
        "anti_billboard_score_v2": float(score),
        "billboard_risk_v2": float(risk),
        "billboard_fail_v2": fail,
    }


def load_npz_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as z:
        points = np.asarray(z["human_points"], dtype=np.float64)
        body = np.asarray(z["body_part_id"]) if "body_part_id" in z.files else None
    return points, body


def main() -> int:
    path = Path(r"D:\vggt\vggt-canonical-surfel-adapter\output\V10700000000000000000_volume_aware_training_matrix\0012_11_frame001\real_vggt_baseline_only\predictions.npz")
    points, body = load_npz_points(path)
    print(json.dumps(anti_billboard_metric_v2(points, body), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
