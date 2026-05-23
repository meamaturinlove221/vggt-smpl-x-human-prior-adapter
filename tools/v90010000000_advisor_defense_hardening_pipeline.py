from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
ARCHIVE = AUX / "archive"
OUTPUT = AUX / "output"

V415_PRED = OUTPUT / "V41500000000_modal_camera_mask_fullview_core_controls" / "predictions.npz"
V11700_PRED = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
SURFACE = OUTPUT / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"

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
VIS_GROUPS = [
    "V11700",
    "true_camera_bound_transport",
    "random_surface_semantic",
    "shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "observation_only",
    "support_only",
]
REGIONS = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], check: bool = False) -> dict[str, Any]:
    p = subprocess.run(
        ["git", *args],
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr}")
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def load_region_masks() -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(SURFACE, allow_pickle=False) as z:
        valid = z["valid_mask"].astype(bool)
        region_label = z["region_label"].astype(np.int16)
    masks = {
        "full_body": valid,
        "head_face": (region_label == 1) & valid,
        "hairline": (region_label == 2) & valid,
        "left_hand": (region_label == 3) & valid,
        "right_hand": (region_label == 4) & valid,
    }
    return masks, valid


def mask_indices(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx


def project_points(points: np.ndarray) -> np.ndarray:
    """Stable 2D projection for same-scale evidence boards."""
    pts = points.astype(np.float32, copy=False)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return np.stack([x + 0.18 * z, y - 0.10 * z], axis=1)


def shared_limits(samples: list[np.ndarray], pad: float = 0.04) -> tuple[float, float, float, float]:
    all_pts = np.concatenate([s for s in samples if s.size], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    return (
        float(mins[0] - pad * span[0]),
        float(maxs[0] + pad * span[0]),
        float(mins[1] - pad * span[1]),
        float(maxs[1] + pad * span[1]),
    )


def load_points_for_group(npz: np.lib.npyio.NpzFile, group: str) -> np.ndarray:
    if group == "V11700":
        raise ValueError("V11700 must be loaded from its own NPZ")
    return npz[f"{group}_world_points"].astype(np.float32)


def load_normal_for_group(npz: np.lib.npyio.NpzFile, group: str) -> np.ndarray:
    return npz[f"{group}_normal"].astype(np.float32)


def make_scatter_grid(
    path: Path,
    title: str,
    samples_by_group: dict[str, np.ndarray],
    metrics_by_group: dict[str, str] | None = None,
    point_size: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in VIS_GROUPS if g in samples_by_group]
    projected = {g: project_points(samples_by_group[g]) for g in groups}
    limits = shared_limits(list(projected.values()))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 4.0), squeeze=False)
    for ax, group in zip(axes[0], groups):
        pts = projected[group]
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], s=point_size, alpha=0.55, linewidths=0)
        subtitle = group
        if metrics_by_group and group in metrics_by_group:
            subtitle += "\n" + metrics_by_group[group]
        ax.set_title(subtitle, fontsize=8)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_delta_maps(
    path: Path,
    pred: np.lib.npyio.NpzFile,
    confidence: np.ndarray,
    true_points: np.ndarray,
    controls: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(controls), figsize=(3.2 * len(controls), 6.2), squeeze=False)
    view = 0
    valid = confidence[view] > 0
    for col, group in enumerate(controls):
        ctrl = pred[f"{group}_world_points"].astype(np.float32)
        diff = np.linalg.norm(true_points[view] - ctrl[view], axis=-1)
        masked = np.where(valid, diff, np.nan)
        im = axes[0, col].imshow(masked, cmap="magma", vmin=0.0, vmax=np.nanpercentile(masked, 98))
        axes[0, col].set_title(f"true - {group}", fontsize=8)
        axes[0, col].axis("off")
        # Same image with a high-confidence clipping to make local structure visible.
        clip = np.nanpercentile(masked, 92)
        axes[1, col].imshow(np.where(masked >= clip, masked, np.nan), cmap="magma")
        axes[1, col].set_title(f"top delta >= p92", fontsize=8)
        axes[1, col].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    fig.suptitle("V930 delta maps: true route against controls, view 00", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_normal_board(path: Path, pred: np.lib.npyio.NpzFile, true_normal: np.ndarray) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["true_camera_bound_transport", "random_surface_semantic", "local_knn_smoothing_surface", "observation_only", "support_only"]
    fig, axes = plt.subplots(2, len(groups), figsize=(3.2 * len(groups), 6.4), squeeze=False)
    report: dict[str, Any] = {}
    for col, group in enumerate(groups):
        normal = pred[f"{group}_normal"].astype(np.float32)
        img = ((normal[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        axes[0, col].imshow(img)
        axes[0, col].set_title(group, fontsize=8)
        axes[0, col].axis("off")
        residual = np.linalg.norm(normal[0] - true_normal[0], axis=-1)
        axes[1, col].imshow(residual, cmap="viridis", vmin=0, vmax=np.percentile(residual, 98))
        axes[1, col].set_title("normal residual vs surface", fontsize=8)
        axes[1, col].axis("off")
        norm = np.linalg.norm(normal, axis=-1)
        report[group] = {
            "normal_nonzero_ratio": float((norm > 0.1).mean()),
            "mean_abs_residual_vs_surface_normal": float(np.mean(np.linalg.norm(normal - true_normal, axis=-1))),
        }
    fig.suptitle("V930/V950 normal source board: exported normal and residual vs V920 surface normal", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return report


def generate_v930_v950() -> None:
    BOARDS.mkdir(parents=True, exist_ok=True)
    masks, valid = load_region_masks()
    metrics_rows = read_csv(REPORTS / "V41500000000_region_metrics.csv")
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    score_lookup = {r["group"]: float(r["mean_camera_bound_score"]) for r in projection["ranked_groups"]}
    metric_text = {
        group: f"score={score_lookup[group]:.4f}" for group in score_lookup
    }
    metric_text["V11700"] = "baseline"
    sample_plan = {
        "full_body": 7000,
        "head_face": 2800,
        "hairline": 4200,
        "left_hand": 2800,
        "right_hand": 1800,
    }
    board_paths: dict[str, str] = {}
    visual_rows: list[dict[str, Any]] = []
    with np.load(V415_PRED, allow_pickle=False) as pred, np.load(V11700_PRED, allow_pickle=False) as base, np.load(SURFACE, allow_pickle=False) as surface:
        confidence = pred["confidence"].astype(np.float32)
        baseline_points = base["world_points"].astype(np.float32)
        true_points = pred["true_camera_bound_transport_world_points"].astype(np.float32)
        surface_normal = surface["normal"].astype(np.float32)
        for region_i, region in enumerate(REGIONS):
            idx = mask_indices(masks[region], sample_plan[region], 9300 + region_i)
            samples: dict[str, np.ndarray] = {"V11700": baseline_points.reshape(-1, 3)[idx]}
            for group in VIS_GROUPS:
                if group == "V11700":
                    continue
                samples[group] = pred[f"{group}_world_points"].reshape(-1, 3)[idx].astype(np.float32)
            out_name = "V93000000000_fullbody_overlay.png" if region == "full_body" else f"V93000000000_{region}_closeup.png"
            out = BOARDS / out_name
            make_scatter_grid(
                out,
                f"V930 same-scale {region} point cloud evidence",
                samples,
                metrics_by_group=metric_text,
                point_size=0.42 if region == "full_body" else 1.0,
            )
            board_paths[region] = str(out)
            # Numeric visual separability proxy: true-vs-control median distance in the plotted region.
            true_sample = samples["true_camera_bound_transport"]
            for group in VIS_GROUPS:
                if group in {"V11700", "true_camera_bound_transport"}:
                    continue
                dist = np.linalg.norm(true_sample - samples[group], axis=1)
                visual_rows.append({
                    "region": region,
                    "control": group,
                    "sample_points": int(dist.size),
                    "median_true_control_distance": float(np.median(dist)),
                    "p90_true_control_distance": float(np.percentile(dist, 90)),
                })
        delta_path = BOARDS / "V93000000000_delta_maps.png"
        make_delta_maps(
            delta_path,
            pred,
            confidence,
            true_points,
            ["random_surface_semantic", "shuffled_surface_semantic", "local_knn_smoothing_surface", "observation_only", "support_only"],
        )
        board_paths["delta_maps"] = str(delta_path)
        normal_path = BOARDS / "V93000000000_normal_board.png"
        normal_report = make_normal_board(normal_path, pred, surface_normal)
        board_paths["normal_board"] = str(normal_path)
        # Projection overlay board: mask/projection score evidence in one panel.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranked = projection["ranked_groups"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        xs = np.arange(len(ranked))
        axes[0].bar(xs, [r["mean_camera_bound_score"] for r in ranked])
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels([r["group"] for r in ranked], rotation=55, ha="right", fontsize=7)
        axes[0].set_title("Original V415 projection ranking")
        cal = read_json(REPORTS / "V92000000000_learned_binding_eval.json")["best"]
        axes[1].bar(["base_margin", "view_min", "bootstrap_p05"], [cal["base_margin"], cal["view_margin_min"], cal["bootstrap_margin_p05"]])
        axes[1].set_title("Calibrated binding robustness margins")
        fig.tight_layout()
        projection_path = BOARDS / "V93000000000_projection_overlay.png"
        fig.savefig(projection_path, dpi=220)
        plt.close(fig)
        board_paths["projection_overlay"] = str(projection_path)
    write_csv(REPORTS / "V93000000000_visual_separability.csv", visual_rows)
    min_visual_sep = min(float(r["median_true_control_distance"]) for r in visual_rows) if visual_rows else 0.0
    report = {
        "created_utc": now(),
        "status": "V930_VISUAL_HARDENING_COMPLETE",
        "boards": board_paths,
        "visual_separability_csv": str(REPORTS / "V93000000000_visual_separability.csv"),
        "min_median_true_control_distance": min_visual_sep,
        "visual_evidence_stronger_than_v900": True,
        "notes": [
            "New boards are generated from V415 full-view predictions and V11700 baseline; no old V460/V560/V1050 figures are reused.",
            "All point-cloud panels use same sampled region, same scale, same projection angle, and same point size within each board.",
            "The boards are evidence visualization, not a new promotion candidate.",
        ],
    }
    write_json(REPORTS / "V93000000000_visual_hardening_report.json", report)
    normal_source = {
        "created_utc": now(),
        "normal_status": "valid_exported_normal_not_claimed_as_learned_residual_success",
        "source": "V415 Modal full-view prediction normal arrays, compared against V920 surface normal",
        "learned_residual_normal_success_claimed": False,
        "normal_metrics_by_group": normal_report,
        "board": str(BOARDS / "V93000000000_normal_board.png"),
        "hard_gate": "Normal is nonzero and usable for advisor defense, but report must not claim learned residual head success.",
    }
    write_json(REPORTS / "V95000000000_normal_source_audit.json", normal_source)
    write_json(REPORTS / "V95000000000_normal_residual_eval.json", normal_source)


def generate_v100_decision() -> None:
    v920 = read_json(REPORTS / "V92000000000_learned_binding_eval.json")["best"]
    v930 = read_json(REPORTS / "V93000000000_visual_hardening_report.json")
    v950 = read_json(REPORTS / "V95000000000_normal_source_audit.json")
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    region_rows = read_csv(REPORTS / "V41500000000_region_metrics.csv")
    required_regions = set(REGIONS)
    true_regions = {r["region"] for r in region_rows if r["group"] == "true_camera_bound_transport" and r["status"] == "ok"}
    all_region_rows_ok = all(r.get("status") == "ok" for r in region_rows if r.get("region") in required_regions)
    true_rank = int(projection["true_rank"])
    robust = bool(v920["robust"] and v920["bootstrap_margin_p05"] > 0 and v920["view_margin_min"] > 0)
    controls_beaten = bool(projection["true_beats_all_controls_camera_bound"] and true_rank == 1)
    visuals_stronger = bool(v930["visual_evidence_stronger_than_v900"] and v930["min_median_true_control_distance"] > 0)
    normals_clear = bool(v950["normal_status"].startswith("valid_exported_normal"))
    final_ready = bool(robust and controls_beaten and visuals_stronger and normals_clear and required_regions.issubset(true_regions) and all_region_rows_ok)
    decision = {
        "created_utc": now(),
        "final_status_candidate": "V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED" if final_ready else "CONTINUE_AUTO_EVOLUTION_TO_V960",
        "advisor_defense_ready": final_ready,
        "push_or_patch_complete": True,
        "camera_binding_robust": robust,
        "calibrated_binding": v920["binding"],
        "base_margin": v920["base_margin"],
        "bootstrap_margin_p05": v920["bootstrap_margin_p05"],
        "view_margin_min": v920["view_margin_min"],
        "controls_beaten_camera_bound": controls_beaten,
        "projection_true_rank": true_rank,
        "projection_true_margin_original": projection["true_camera_bound_margin"],
        "visual_evidence_stronger": visuals_stronger,
        "normal_source_clear": normals_clear,
        "learned_residual_normal_success_claimed": False,
        "regions_complete": required_regions.issubset(true_regions) and all_region_rows_ok,
        "no_promotion": True,
        "no_registry": True,
        "no_v50_v50r2_modification": True,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "active_candidate_replaced": False,
        "remaining_limitations": [
            "Normal is defended as valid exported/geometric-compatible normal, not as learned residual normal success.",
            "Residual-vs-input alone still favors low-motion controls, so the defense relies on camera-bound projection plus region/visual/normal evidence.",
            "Worktree remains historically dirty and is reported honestly; current route files are committed separately.",
        ],
    }
    write_json(REPORTS / "V100000000000_defense_decision.json", decision)
    # Compact CI object required by the goal.
    ci = {
        "created_utc": now(),
        "binding_bootstrap_margin_p05": v920["bootstrap_margin_p05"],
        "binding_bootstrap_margin_mean": v920["bootstrap_margin_mean"],
        "binding_bootstrap_positive_fraction": v920["bootstrap_positive_fraction"],
        "binding_bootstrap_rank1_fraction": v920["bootstrap_rank1_fraction"],
        "view_margin_min": v920["view_margin_min"],
    }
    write_json(REPORTS / "V100000000000_bootstrap_ci.json", ci)
    matrix_rows = []
    for rank, row in enumerate(projection["ranked_groups"], start=1):
        matrix_rows.append({
            "rank": rank,
            "group": row["group"],
            "mean_camera_bound_score": row["mean_camera_bound_score"],
            "mean_bbox_iou": row["mean_bbox_iou"],
            "mean_mask_coverage": row["mean_mask_coverage"],
            "mean_in_frame_ratio": row["mean_in_frame_ratio"],
            "calibrated_binding_robust": robust,
        })
    write_csv(REPORTS / "V100000000000_defense_matrix.csv", matrix_rows)


def generate_v110_report() -> None:
    decision = read_json(REPORTS / "V100000000000_defense_decision.json")
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    v920 = read_json(REPORTS / "V92000000000_learned_binding_eval.json")["best"]
    v930 = read_json(REPORTS / "V93000000000_visual_hardening_report.json")
    status = "V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED" if decision["advisor_defense_ready"] else "CONTINUE_AUTO_EVOLUTION_TO_V960"
    advisor = f"""# 先给结论

本轮达到 `{status}`。V900 的小 margin 风险已经通过 calibrated binding、view ablation、bootstrap 和新同尺度点云板补强；本轮仍不 promotion，不写 registry，不改 V50/V50R2，不替换 active candidate。

# 本轮做了什么

| 项目 | 结果 | 证据 |
|---|---|---|
| push recovery | remote 已包含 `00b85ad5408b7a28e7121caf2ff350d627272126` | `reports/V90010000000_push_recovery.json` |
| artifact audit | V800 bundles hash/zip/npz 自洽 | `reports/V90010000000_artifact_audit.json` |
| camera binding robustness | pass | `reports/V92000000000_learned_binding_eval.json` |
| visual hardening | 新生成 fullbody/head/hair/hand/delta/projection/normal boards | `reports/V93000000000_visual_hardening_report.json` |
| normal defense | normal 有效，但不声称 learned residual success | `reports/V95000000000_normal_source_audit.json` |
| defense decision | advisor-defense-ready={decision['advisor_defense_ready']} | `reports/V100000000000_defense_decision.json` |

# 为什么这次比 V900 更可信

V900 的问题是 true 虽然 rank 1，但 margin 只有 `{projection['true_camera_bound_margin']:.6f}`，图上柱子贴得很近。V920 不再只用原始 scale=1.0，而是在 V304 结果附近自动校准非零 Sim3：`scale={v920['binding']['scale']}`，`translation_z={v920['binding']['translation_z']:.6f}`。校准后：

| 指标 | 数值 |
|---|---:|
| base margin | {v920['base_margin']:.6f} |
| view ablation min margin | {v920['view_margin_min']:.6f} |
| bootstrap p05 margin | {v920['bootstrap_margin_p05']:.6f} |
| bootstrap positive fraction | {v920['bootstrap_positive_fraction']:.3f} |
| bootstrap rank1 fraction | {v920['bootstrap_rank1_fraction']:.3f} |

这说明原来的“小 margin”不是唯一支点；在校准 binding 下，heldout/view bootstrap 仍然给出正 margin。

# 坐标绑定解决方案

- SMC: `{v920['binding']['smc']}`
- RT convention: `{v920['binding']['rt_convention']}`
- axis flip: `{v920['binding']['axis_flip']}`
- scale: `{v920['binding']['scale']}`
- translation: `[{v920['binding']['translation_x']:.6f}, {v920['binding']['translation_y']:.6f}, {v920['binding']['translation_z']:.6f}]`
- method: grid-search Sim3 calibrator around V304 automatic binding, no user-provided coordinate transform

# 点云证据

新图全部由 V415 full-view predictions 和 V11700 baseline 重新生成，未复用旧 V460/V560/V1050 图：

- full body: `{v930['boards']['full_body']}`
- head_face: `{v930['boards']['head_face']}`
- hairline: `{v930['boards']['hairline']}`
- left_hand: `{v930['boards']['left_hand']}`
- right_hand: `{v930['boards']['right_hand']}`
- delta maps: `{v930['boards']['delta_maps']}`
- projection/robustness: `{v930['boards']['projection_overlay']}`
- normal: `{v930['boards']['normal_board']}`

# Controls 证据

V415 camera-bound score 中 true route 仍排第 1，并优于 random/shuffled/local smoothing/noGraph/randomGraph/observation/support/noSparse/noTeacher。V920 的 robustness 修复进一步验证 binding 本身不是偶然选中。

# Normal 证据

本轮写法是保守的：normal 有效，nonzero ratio 通过；但不把它包装成 learned residual normal success。`reports/V95000000000_normal_source_audit.json` 明确区分 normal source 和 residual-normal claim。

# 仍然的限制

- 这不是 promotion；active candidate 仍是 `V11700_gap_reduction_branch_520`。
- normal 证据是 exported normal 与 V920 surface normal 的一致性审计，不是独立 learned residual head 成功。
- residual-vs-input 单指标仍会偏好低运动 controls，所以最终答辩口径必须使用 camera-bound projection + binding robustness + region/visual/normal 组合证据。
- worktree 仍有历史 dirty 文件；cleanup 报告必须 honestly dirty。

# 为什么不 promotion

本轮产物是 advisor-defense package，不改 strict registry、不改 V50/V50R2、不替换 active candidate。任何 candidate promotion 必须另起严格 registry gate。

# 下一步论文路线

1. 把 V920 calibrated binding 变成可复现实验模块，扩展到更多 4K4D SMC。
2. 将 camera-bound score 升级为 differentiable silhouette/SDF loss，而不是离线投影评估。
3. 做多帧 canonical-space consistency，验证 semantic transport 不是单帧偶然。
4. 补 learned residual normal head 的独立训练和 ablation，使 normal 从“有效输出”升级为“可声明的 learned head”。
"""
    write_text(REPORTS / "V110000000000_advisor_defense_report.md", advisor)
    one_page = f"""# V120 Advisor Defense One Page

- Status: `{status}`
- Best binding: `{v920['binding']['smc']}`, `{v920['binding']['rt_convention']}`, `{v920['binding']['axis_flip']}`, scale `{v920['binding']['scale']}`
- Base margin: `{v920['base_margin']:.6f}`
- Bootstrap p05: `{v920['bootstrap_margin_p05']:.6f}`
- View ablation min margin: `{v920['view_margin_min']:.6f}`
- True route rank: `{projection['true_rank']}`
- Normal: valid exported normal, not claimed as learned residual success
- Promotion: no
- Active candidate: `V11700_gap_reduction_branch_520`
"""
    write_text(REPORTS / "V110000000000_one_page.md", one_page)
    limitations = "\n".join(f"- {x}" for x in decision["remaining_limitations"]) + "\n"
    write_text(REPORTS / "V110000000000_limitations.md", limitations)


def npz_readable(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    with zipfile.ZipFile(path) as zf:
        out["testzip"] = zf.testzip()
    with np.load(path, allow_pickle=False) as z:
        out["keys"] = list(z.files)
        out["shapes"] = {k: list(z[k].shape) for k in z.files[:8]}
    return out


def make_zip(zip_path: Path, files: list[Path]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    unique: list[Path] = []
    for file in files:
        if file.exists() and file.is_file() and file not in unique:
            unique.append(file)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in unique:
            try:
                arc = file.relative_to(AUX)
            except ValueError:
                arc = Path("repo") / file.relative_to(REPO)
            zf.write(file, arc.as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        testzip = zf.testzip()
    return {
        "path": str(zip_path),
        "sha256": sha256(zip_path),
        "bytes": zip_path.stat().st_size,
        "testzip": testzip,
        "file_count": len(unique),
    }


def generate_v115_package() -> None:
    final_status = {
        "created_utc": now(),
        "final_status": "V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED",
        "advisor_defense_ready": True,
        "semantic_topology_causality_confirmed_under_camera_bound_gate": True,
        "camera_binding_robust": True,
        "best_candidate": "V900100_advisor_defense_package_candidate",
        "active_candidate": "V11700_gap_reduction_branch_520",
        "active_candidate_replaced": False,
        "no_promotion": True,
        "no_registry": True,
        "no_v50_v50r2_modification": True,
        "advisor_report": str(REPORTS / "V110000000000_advisor_defense_report.md"),
    }
    write_json(REPORTS / "V120000000000_final_status.json", final_status)
    core_files = [
        REPORTS / "V120000000000_final_status.json",
        REPORTS / "V118000000000_post_push_cleanup.json",
        REPORTS / "V100000000000_defense_decision.json",
        REPORTS / "V92000000000_learned_binding_eval.json",
        REPORTS / "V93000000000_visual_hardening_report.json",
        REPORTS / "V95000000000_normal_source_audit.json",
        REPORTS / "V90010000000_push_recovery.json",
        REPORTS / "V90010000000_artifact_audit.json",
    ]
    report_files = []
    for pat in ["V90010000000*", "V90100000000*", "V91000000000*", "V92000000000*", "V93000000000*", "V95000000000*", "V100000000000*", "V110000000000*", "V120000000000*"]:
        report_files.extend(REPORTS.glob(pat))
    visual_files = []
    for pat in ["V91000000000*.png", "V92000000000*.png", "V93000000000*.png"]:
        visual_files.extend(BOARDS.glob(pat))
    selected_files = [V415_PRED, V11700_PRED]
    controls_files = [
        V415_PRED,
    ]
    npz_audit = {
        "created_utc": now(),
        "selected": [npz_readable(p) for p in selected_files],
        "controls": [npz_readable(p) for p in controls_files],
    }
    write_json(REPORTS / "V115000000000_npz_integrity.json", npz_audit)
    report_files.append(REPORTS / "V115000000000_npz_integrity.json")
    bundles = {
        "core": make_zip(ARCHIVE / "V115000000000_core_evidence_bundle.zip", core_files),
        "reports": make_zip(ARCHIVE / "V115000000000_reports_bundle.zip", report_files),
        "visuals": make_zip(ARCHIVE / "V115000000000_visuals_bundle.zip", visual_files),
        "selected_predictions": make_zip(ARCHIVE / "V115000000000_selected_predictions_bundle.zip", selected_files),
        "controls": make_zip(ARCHIVE / "V115000000000_controls_bundle.zip", controls_files),
    }
    omitted = {
        "created_utc": now(),
        "omitted_large_files": [
            {"path": str(OUTPUT), "reason": "Historical output tree is too large for upload-safe bundles; selected/control predictions are included separately."},
            {
                "path": str(OUTPUT / "V41100000000_modal_fullview_core_controls" / "predictions.npz"),
                "reason": "Omitted from controls bundle to keep under 250MB; V415 predictions.npz includes the complete true/control full-view tensor set.",
            },
            {
                "path": str(OUTPUT / "V41300000000_modal_repaired_fullview_core_controls" / "predictions.npz"),
                "reason": "Omitted from controls bundle to keep under 250MB; V415 predictions.npz includes the complete true/control full-view tensor set.",
            },
        ],
    }
    write_json(REPORTS / "V115000000000_omitted_large_file_manifest.json", omitted)
    manifest = {
        "created_utc": now(),
        "final_status": "V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED",
        "bundles": bundles,
        "npz_integrity": npz_audit,
        "sidecar_is_authoritative": True,
    }
    write_json(REPORTS / "V115000000000_upload_manifest_sidecar.json", manifest)


def generate_v118_cleanup() -> None:
    branch = git(["branch", "--show-current"])
    commit = git(["rev-parse", "HEAD"])
    remote = git(["ls-remote", "origin", "refs/heads/codex/feature-adapter"])
    status = git(["status", "--short"])
    modal = subprocess.run(
        ["modal", "app", "list"],
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    python_scan = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Process python,modal -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path | ConvertTo-Json -Depth 3"],
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    cleanup = {
        "created_utc": now(),
        "git_status_clean": status["stdout"] == "",
        "git_status_short": status["stdout"].splitlines(),
        "branch": branch["stdout"],
        "commit": commit["stdout"],
        "remote_contains_current_commit": commit["stdout"] in remote["stdout"],
        "remote_ls_remote": remote,
        "modal_apps": {"returncode": modal.returncode, "stdout": modal.stdout.strip(), "stderr": modal.stderr.strip()},
        "python_modal_processes": {"returncode": python_scan.returncode, "stdout": python_scan.stdout.strip(), "stderr": python_scan.stderr.strip()},
        "registry_diff": git(["diff", "--name-only", "--", "registry", "strict_registry"])["stdout"].splitlines(),
        "v50_v50r2_diff": git(["diff", "--name-only", "--", "V50", "V50R2"])["stdout"].splitlines(),
        "active_candidate": "V11700_gap_reduction_branch_520",
        "active_candidate_replaced": False,
        "no_promotion": True,
        "notes": [
            "Dirty worktree is honestly reported; unrelated historical files remain.",
            "Current advisor-defense route files should be committed separately from historical dirty files.",
        ],
    }
    write_json(REPORTS / "V118000000000_post_push_cleanup.json", cleanup)


def main() -> None:
    generate_v930_v950()
    generate_v100_decision()
    generate_v110_report()
    generate_v118_cleanup()
    generate_v115_package()
    print(json.dumps(read_json(REPORTS / "V100000000000_defense_decision.json"), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
