from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAIN = Path(r"D:\vggt\vggt-main")
ROOT = MAIN / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUTPUT = ROOT / "output"
FEATURE_MAPS = OUTPUT / "V8100000_V9000000_smplx_feature_encoding" / "V8200000_smplx_feature_raster" / "feature_maps.npz"
V770 = OUTPUT / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V999 = OUTPUT / "V9400000_V9990000_longrun_feature_adapter" / "V9800000_candidates" / "cand_129_triplane_only_w080" / "predictions.npz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jdump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def load_points(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "world_points" in z.files:
            return z["world_points"].astype(np.float32)
        return z["points"].astype(np.float32)


def load_feature_masks() -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(FEATURE_MAPS, allow_pickle=True) as z:
        fm = z["feature_maps"].astype(np.float32)
        names = [str(x) for x in z["channel_names"].tolist()]
    ch = {name: i for i, name in enumerate(names)}
    masks = {
        "full_body": fm[:, ch["semantic_foreground"]] > 0.25,
        "head_face": fm[:, ch["semantic_head_face"]] > 0.25,
        "hairline": fm[:, ch["semantic_hairline"]] > 0.20,
        "left_hand": fm[:, ch["semantic_left_hand"]] > 0.20,
        "right_hand": fm[:, ch["semantic_right_hand"]] > 0.20,
    }
    return masks, names


def sample_region(points: np.ndarray, mask: np.ndarray, view: int, limit: int, seed: int) -> np.ndarray:
    pts = points[view][mask[view]]
    pts = pts[np.isfinite(pts).all(axis=-1)]
    if pts.shape[0] == 0:
        return pts.reshape(0, 3)
    if pts.shape[0] > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=limit, replace=False)
        pts = pts[idx]
    return pts


def axis_bounds(regions: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts = np.concatenate([r for r in regions if r.size], axis=0) if any(r.size for r in regions) else np.zeros((1, 3), dtype=np.float32)
    bounds = []
    for axis in range(3):
        lo = float(np.quantile(pts[:, axis], 0.01))
        hi = float(np.quantile(pts[:, axis], 0.99))
        pad = max((hi - lo) * 0.08, 1e-3)
        bounds.append((lo - pad, hi + pad))
    return bounds[0], bounds[1], bounds[2]


def scatter_panel(ax: Any, pts: np.ndarray, title: str, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.55)
    ax.set_title(title)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.view_init(elev=16, azim=-72)
    ax.tick_params(labelsize=6)


def delta_metrics(base: np.ndarray, prev: np.ndarray, new: np.ndarray, mask: np.ndarray) -> dict[str, float | int | bool]:
    d_base = np.linalg.norm(new - base, axis=-1)
    d_prev = np.linalg.norm(new - prev, axis=-1)
    region = mask & np.isfinite(d_base)
    if not bool(region.any()):
        return {
            "pixels": 0,
            "mean_delta_vs_v770": 0.0,
            "p95_delta_vs_v770": 0.0,
            "mean_delta_vs_v999": 0.0,
            "changed_vs_v999": 0,
            "visible_proxy_pass": False,
        }
    mean = float(d_base[region].mean())
    p95 = float(np.quantile(d_base[region], 0.95))
    mean_prev = float(d_prev[region].mean())
    changed_prev = int((d_prev[region] > 1e-6).sum())
    return {
        "pixels": int(region.sum()),
        "mean_delta_vs_v770": mean,
        "p95_delta_vs_v770": p95,
        "mean_delta_vs_v999": mean_prev,
        "changed_vs_v999": changed_prev,
        "visible_proxy_pass": bool(changed_prev > 50 and (p95 > 0.001 or mean > 0.0005)),
    }


def run(candidate: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    v770 = load_points(V770)
    v999 = load_points(V999)
    new = load_points(candidate)
    masks, channels = load_feature_masks()
    regions = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]
    variants = [("V770", v770), ("V999", v999), ("SparseConv", new)]
    fig = plt.figure(figsize=(12, 18))
    fig.suptitle("V12100000 mentor visual gate: true 3D region point cloud closeups")
    panel = 1
    metrics: dict[str, Any] = {}
    for ridx, region in enumerate(regions):
        sampled = [sample_region(arr, masks[region], view=0, limit=2200 if region == "full_body" else 1600, seed=1200 + ridx) for _, arr in variants]
        bounds = axis_bounds(sampled)
        for label, pts in zip([v[0] for v in variants], sampled):
            ax = fig.add_subplot(len(regions), len(variants), panel, projection="3d")
            scatter_panel(ax, pts, f"{region}: {label}", bounds)
            panel += 1
        metrics[region] = delta_metrics(v770, v999, new, masks[region])
    fig.tight_layout()
    board_path = out_dir / "V12100000_true_region_pointcloud_closeups.png"
    fig.savefig(board_path, dpi=150)
    plt.close(fig)

    all_regions_pass = all(metrics[r]["visible_proxy_pass"] for r in ("head_face", "hairline", "left_hand", "right_hand"))
    at_least_two_local = sum(bool(metrics[r]["visible_proxy_pass"]) for r in ("head_face", "hairline", "left_hand", "right_hand")) >= 2
    report = {
        "created_utc": now(),
        "candidate": str(candidate),
        "board": str(board_path),
        "metrics": metrics,
        "channels": channels,
        "all_regions_pass": all_regions_pass,
        "at_least_two_local_regions_pass": at_least_two_local,
        "mentor_visual_gate": "PASS_REVIEW_READY_PROXY" if at_least_two_local else "FAIL_NEEDS_NEXT_ROUTE",
        "note": "This is a local visual hardening gate over the Modal spconv candidate; it does not promote or write a strict registry.",
    }
    jdump(REPORTS / "V12100000_sparseconv_mentor_visual_gate.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default=str(
            OUTPUT
            / "V10000000_V12000000_modal_sparseconv"
            / "V100_formal_20260520"
            / "candidates"
            / "cand_099_spconv_v129_guarded_mix_s1p25"
            / "predictions.npz"
        ),
    )
    parser.add_argument("--out-dir", default=str(BOARDS))
    args = parser.parse_args()
    report = run(Path(args.candidate), Path(args.out_dir))
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
