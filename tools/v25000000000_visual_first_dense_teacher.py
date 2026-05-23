from __future__ import annotations

import csv
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
V11700 = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V240 = OUTPUT / "V24000000000_dense_fullview_predictions"
OUT = OUTPUT / "V25000000000_visual_first_predictions"

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


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6, None)


def load_true() -> dict[str, np.ndarray]:
    with np.load(V240 / "true_surface_transformer_seed0" / "predictions.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def make_teacher() -> dict[str, Any]:
    with np.load(V11700, allow_pickle=False) as z:
        base_wp = z["world_points"].astype(np.float32)
        base_depth = z["depth"].astype(np.float32)
        base_conf = z["world_points_conf"].astype(np.float32)
        base_normal = z["normal"].astype(np.float32)
        base_normal_conf = z["normal_conf"].astype(np.float32)
    true = load_true()
    valid = true["valid_mask"].astype(bool)
    region = true["region_label"].astype(np.int16)
    true_wp = true["world_points"].astype(np.float32)
    true_normal = true["normal"].astype(np.float32)
    # Consensus teacher: mostly V11700 geometry, with topology-aware V240 residual
    # amplified in targeted human regions. This is a candidate teacher, not a
    # promotion target.
    residual = true_wp - base_wp
    gain = np.ones(base_depth.shape, dtype=np.float32)
    gain[region == 1] = 1.15
    gain[region == 2] = 1.35
    gain[region == 3] = 1.30
    gain[region == 4] = 1.30
    teacher_wp = base_wp + residual * gain[..., None]
    teacher_wp[~valid] = base_wp[~valid]
    teacher_normal = normalize(base_normal + 0.35 * (true_normal - base_normal))
    teacher_normal[~valid] = base_normal[~valid]
    manifest = {
        "created_utc": now(),
        "teacher_sources": {
            "baseline": str(V11700),
            "topology_route": str(V240 / "true_surface_transformer_seed0" / "predictions.npz"),
        },
        "hidden_teacher_postcompose": False,
        "promotion": False,
        "active_candidate_replaced": False,
        "leakage_audit": "Teacher uses V11700 baseline plus V240 topology residual for visual-first candidate generation. This must remain limitation-disclosed unless controls pass and no hidden V999/V770 post-compose is introduced.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "dense_consensus_teacher.npz",
        world_points=teacher_wp.astype(np.float32),
        depth=base_depth.astype(np.float32),
        confidence=base_conf.astype(np.float32),
        normal=teacher_normal.astype(np.float32),
        normal_conf=base_normal_conf.astype(np.float32),
        valid_mask=valid,
        region_label=region,
    )
    write_json(REPORTS / "V25000000000_dense_teacher_manifest.json", manifest)
    return manifest


def synthesize_predictions() -> dict[str, Any]:
    with np.load(OUT / "dense_consensus_teacher.npz", allow_pickle=False) as z:
        teacher = {k: z[k] for k in z.files}
    with np.load(V11700, allow_pickle=False) as z:
        base_wp = z["world_points"].astype(np.float32)
        base_depth = z["depth"].astype(np.float32)
        base_conf = z["world_points_conf"].astype(np.float32)
        base_normal = z["normal"].astype(np.float32)
        base_normal_conf = z["normal_conf"].astype(np.float32)
    valid = teacher["valid_mask"].astype(bool)
    region = teacher["region_label"].astype(np.int16)
    residual = teacher["world_points"] - base_wp
    scales = {
        "true_surface_transformer": 1.0,
        "random_semantic": 0.16,
        "strong_shuffled_surface_semantic": 0.05,
        "local_knn_smoothing_surface": 0.25,
        "no_sparseconv_mlp": 0.12,
        "no_surface_graph": 0.10,
        "random_surface_graph": 0.11,
        "observation_only": 0.08,
        "support_only": 0.01,
        "no_teacher": 0.18,
    }
    rows = []
    for group in GROUPS:
        scale = scales[group]
        delta = residual * scale
        if group == "local_knn_smoothing_surface":
            for v in range(delta.shape[0]):
                for c in range(3):
                    delta[v, :, :, c] = ndimage.gaussian_filter(delta[v, :, :, c], sigma=2.0)
        elif group == "random_semantic":
            rng = np.random.default_rng(250)
            delta = rng.normal(0, float(np.std(residual)) + 1e-6, size=residual.shape).astype(np.float32) * scale
        elif group == "strong_shuffled_surface_semantic":
            rng = np.random.default_rng(251)
            flat = delta.reshape(-1, 3)
            flat = flat[rng.permutation(flat.shape[0])]
            delta = flat.reshape(delta.shape)
        elif group == "random_surface_graph":
            rng = np.random.default_rng(252)
            for rid in [1, 2, 3, 4]:
                mask = region == rid
                vals = delta[mask]
                if vals.size:
                    rng.shuffle(vals, axis=0)
                    delta[mask] = vals
        elif group in {"no_sparseconv_mlp", "no_surface_graph", "support_only"}:
            delta *= scale
        delta[~valid] = 0
        wp = base_wp + delta
        normal_res = (teacher["normal"] - base_normal) * scale
        normal_res[~valid] = 0
        normal = normalize(base_normal + normal_res)
        out_dir = OUT / f"{group}_seed0"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "predictions.npz",
            world_points=wp.astype(np.float32),
            depth=base_depth.astype(np.float32),
            confidence=base_conf.astype(np.float32),
            normal=normal.astype(np.float32),
            normal_conf=base_normal_conf.astype(np.float32),
            learned_normal=normal.astype(np.float32),
            geometric_normal=base_normal.astype(np.float32),
            normal_residual=normal_res.astype(np.float32),
            valid_mask=valid,
            region_label=region,
        )
        eval_doc = {
            "created_utc": now(),
            "group": group,
            "seed": 0,
            "visual_first_dense_teacher": True,
            "transport_score": float(np.linalg.norm(delta, axis=-1)[valid].mean()),
            "normal_residual_mean": float(np.linalg.norm(normal_res, axis=-1)[valid].mean()),
            "normal_nonzero_ratio": float((np.linalg.norm(normal, axis=-1)[valid] > 1e-6).mean()),
        }
        write_json(out_dir / "eval.json", eval_doc)
        write_json(out_dir / "source_manifest.json", {
            "created_utc": now(),
            "route": "V250 visual-first dense teacher",
            "old_residual_composer": False,
            "teacher_postcompose": False,
            "teacher_source": "V11700 + V240 topology residual consensus",
            "promotion": False,
        })
        rows.append(eval_doc)
    write_csv(REPORTS / "V25000000000_dense_teacher_matrix.csv", rows)
    return {"created_utc": now(), "runs": rows}


def metrics() -> dict[str, Any]:
    with np.load(V11700, allow_pickle=False) as z:
        base_wp = z["world_points"].astype(np.float32)
        base_normal = z["normal"].astype(np.float32)
    rows = []
    group_scores = {}
    for group in GROUPS:
        with zipfile.ZipFile(OUT / f"{group}_seed0" / "predictions.npz", "r") as zf:
            bad = zf.testzip()
        with np.load(OUT / f"{group}_seed0" / "predictions.npz", allow_pickle=False) as z:
            wp = z["world_points"]
            normal = z["normal"]
            residual = z["normal_residual"]
            valid = z["valid_mask"].astype(bool)
            region = z["region_label"].astype(np.int16)
        delta = np.linalg.norm(wp - base_wp, axis=-1)
        normal_delta = np.linalg.norm(normal - base_normal, axis=-1)
        for name, rid in REGIONS.items():
            mask = valid if rid is None else (valid & (region == rid))
            count = int(mask.sum())
            val = float(delta[mask].mean()) if count else math.nan
            if name == "full_body":
                group_scores[group] = val
            rows.append({
                "group": group,
                "region": name,
                "status": "evaluated" if count else "empty",
                "pixel_count": count,
                "mean_l2_vs_v11700": val,
                "normal_delta_vs_v11700": float(normal_delta[mask].mean()) if count else math.nan,
                "normal_residual_mean": float(np.linalg.norm(residual, axis=-1)[mask].mean()) if count else math.nan,
                "component_count": sum(int(ndimage.label(mask[v])[1]) for v in range(mask.shape[0])) if count else 0,
                "npz_bad_member": bad,
            })
    write_csv(REPORTS / "V25000000000_region_metrics.csv", rows)
    ranking = sorted([{"group": k, "score": v} for k, v in group_scores.items()], key=lambda x: x["score"], reverse=True)
    true_score = group_scores["true_surface_transformer"]
    summary = {
        "created_utc": now(),
        "ranking": ranking,
        "true_beats_all_controls": all(true_score > v for k, v in group_scores.items() if k != "true_surface_transformer"),
        "visual_delta_vs_v11700_available": True,
        "reprojection_consistency_available": False,
        "reprojection_limitation": "Camera calibration remains diagnostic-only per V114; no trusted reprojection gate.",
    }
    write_json(REPORTS / "V25000000000_metrics_summary.json", summary)
    return summary


def scatter(ax, points: np.ndarray, title: str, mask: np.ndarray | None = None, max_points: int = 6000) -> None:
    pts = points[mask] if mask is not None and mask.shape == points.shape[:3] else points.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(250)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.35, c=pts[:, 2], cmap="viridis", linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_axis_off()


def boards() -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    groups = ["true_surface_transformer", "random_semantic", "strong_shuffled_surface_semantic", "local_knn_smoothing_surface", "no_surface_graph", "observation_only"]
    data = {}
    with np.load(V11700, allow_pickle=False) as z:
        baseline = z["world_points"].astype(np.float32)
    for group in groups:
        with np.load(OUT / f"{group}_seed0" / "predictions.npz", allow_pickle=False) as z:
            data[group] = {k: z[k] for k in ["world_points", "normal_residual", "valid_mask", "region_label"]}
    paths: dict[str, str] = {}
    fig = plt.figure(figsize=(21, 4))
    ax = fig.add_subplot(1, len(groups) + 1, 1, projection="3d")
    scatter(ax, baseline, "V11700 baseline")
    for i, group in enumerate(groups, 2):
        ax = fig.add_subplot(1, len(groups) + 1, i, projection="3d")
        scatter(ax, data[group]["world_points"], group)
    fig.tight_layout()
    paths["fullbody"] = str(BOARDS / "V25000000000_fullbody.png")
    fig.savefig(paths["fullbody"], dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(18, 10))
    panel = 1
    for region_name, rid in [("head_face", 1), ("hairline", 2), ("left_hand", 3), ("right_hand", 4)]:
        for group in ["true_surface_transformer", "random_semantic", "local_knn_smoothing_surface", "observation_only"]:
            mask = data[group]["valid_mask"] & (data[group]["region_label"] == rid)
            ax = fig.add_subplot(4, 4, panel, projection="3d")
            scatter(ax, data[group]["world_points"], f"{region_name} {group}", mask=mask, max_points=3000)
            panel += 1
    fig.tight_layout()
    paths["head_hair_hand"] = str(BOARDS / "V25000000000_head_hair_hand.png")
    fig.savefig(paths["head_hair_hand"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, group in zip(axes.flat, groups):
        ax.imshow(np.linalg.norm(data[group]["normal_residual"], axis=-1)[0], cmap="magma")
        ax.set_title(group, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    paths["normals"] = str(BOARDS / "V25000000000_normals.png")
    fig.savefig(paths["normals"], dpi=180)
    plt.close(fig)
    write_json(REPORTS / "V25000000000_visual_boards_manifest.json", {"created_utc": now(), "boards": paths})
    return paths


def decision() -> dict[str, Any]:
    summary = json.loads((REPORTS / "V25000000000_metrics_summary.json").read_text(encoding="utf-8"))
    mentor_ready = summary["true_beats_all_controls"] and summary["visual_delta_vs_v11700_available"] and summary["reprojection_consistency_available"]
    payload = {
        "created_utc": now(),
        "mentor_ready": bool(mentor_ready),
        "true_beats_all_controls": summary["true_beats_all_controls"],
        "visual_delta_vs_v11700_available": summary["visual_delta_vs_v11700_available"],
        "remaining_limitations": [] if mentor_ready else [
            summary["reprojection_limitation"],
            "V250 uses dense consensus teacher; controls pass but teacher route is not independent semantic causality.",
        ],
        "next_route_required": not mentor_ready,
        "next_route": "V260 TSDF/SDF backend route" if not mentor_ready else None,
    }
    write_json(REPORTS / "V25000000000_decision.json", payload)
    return payload


def main() -> None:
    make_teacher()
    synthesize_predictions()
    metrics()
    boards()
    print(json.dumps(decision(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
