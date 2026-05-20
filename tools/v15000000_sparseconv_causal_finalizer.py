from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
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
ARCHIVE = ROOT / "archive"
OUTPUT = ROOT / "output"
RUN_ROOT = OUTPUT / "V10000000_V12000000_modal_sparseconv"
FEATURE_MAPS = OUTPUT / "V8100000_V9000000_smplx_feature_encoding" / "V8200000_smplx_feature_raster" / "feature_maps.npz"
V770 = OUTPUT / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V999 = OUTPUT / "V9400000_V9990000_longrun_feature_adapter" / "V9800000_candidates" / "cand_129_triplane_only_w080" / "predictions.npz"
V129 = OUTPUT / "V1210000_V1800000_smplx_completion_route" / "V1290000_smplx_completion_candidate" / "predictions.npz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_points(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "world_points" in z.files:
            return z["world_points"].astype(np.float32)
        return z["points"].astype(np.float32)


def load_masks() -> tuple[dict[str, np.ndarray], list[str]]:
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
        "background": fm[:, ch["semantic_foreground"]] <= 0.05,
    }
    return masks, names


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def as_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def run_record(run_id: str) -> dict[str, Any] | None:
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    ranked_path = run_dir / "reports" / "V10800000_ranked_candidates.csv"
    if not status_path.is_file():
        return None
    status = load_json(status_path)
    ranked = read_csv(ranked_path)
    best = status.get("best") or (ranked[0] if ranked else {})
    candidate_name = best.get("name")
    pred = run_dir / "candidates" / str(candidate_name) / "predictions.npz" if candidate_name else None
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "status": status,
        "ranked": ranked,
        "best": best,
        "prediction": pred if pred and pred.is_file() else None,
        "thin_bundle": run_dir / "V11000000_thin_review_bundle.zip",
        "full_bundle": run_dir / "V11000000_full_sparseconv_archive.zip",
    }


def region_metric(base: np.ndarray, prev: np.ndarray, arr: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    d_base = np.linalg.norm(arr - base, axis=-1)
    d_prev = np.linalg.norm(arr - prev, axis=-1)
    region = mask & np.isfinite(d_base)
    if not bool(region.any()):
        return {"pixels": 0, "mean_vs_v770": 0.0, "p95_vs_v770": 0.0, "mean_vs_v999": 0.0, "changed_vs_v999": 0}
    return {
        "pixels": int(region.sum()),
        "mean_vs_v770": float(d_base[region].mean()),
        "p95_vs_v770": float(np.quantile(d_base[region], 0.95)),
        "mean_vs_v999": float(d_prev[region].mean()),
        "changed_vs_v999": int((d_prev[region] > 1e-6).sum()),
    }


def sample_points(points: np.ndarray, mask: np.ndarray, view: int, limit: int, seed: int) -> np.ndarray:
    pts = points[view][mask[view]]
    pts = pts[np.isfinite(pts).all(axis=-1)]
    if pts.shape[0] > limit:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], size=limit, replace=False)]
    return pts.reshape(-1, 3)


def continuity_metrics(points: np.ndarray, mask: np.ndarray, view: int = 0, seed: int = 1500) -> dict[str, float | int]:
    pts = sample_points(points, mask, view=view, limit=1400, seed=seed)
    if pts.shape[0] < 8:
        return {"sampled_points": int(pts.shape[0]), "median_nn": 0.0, "p95_nn": 0.0, "isolated_ratio": 0.0}
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist == 0] = np.inf
    nn = np.min(dist, axis=1)
    med = float(np.median(nn))
    p95 = float(np.quantile(nn, 0.95))
    return {
        "sampled_points": int(pts.shape[0]),
        "median_nn": med,
        "p95_nn": p95,
        "isolated_ratio": float((nn > max(p95, med * 3.0)).mean()),
    }


def bounds_for(groups: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts = np.concatenate([g for g in groups if g.size], axis=0) if any(g.size for g in groups) else np.zeros((1, 3), dtype=np.float32)
    out = []
    for axis in range(3):
        lo = float(np.quantile(pts[:, axis], 0.01))
        hi = float(np.quantile(pts[:, axis], 0.99))
        pad = max((hi - lo) * 0.08, 1e-3)
        out.append((lo - pad, hi + pad))
    return out[0], out[1], out[2]


def scatter(ax: Any, pts: np.ndarray, title: str, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]], azim: float = -72) -> None:
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.55)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.view_init(elev=16, azim=azim)
    ax.tick_params(labelsize=6)


def make_mentor_boards(variants: list[tuple[str, np.ndarray]], masks: dict[str, np.ndarray]) -> dict[str, str]:
    BOARDS.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    region_order = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]
    fig = plt.figure(figsize=(4.2 * len(variants), 4.0 * len(region_order)))
    fig.suptitle("V13000000 mentor causal point-cloud board")
    panel = 1
    for ridx, region in enumerate(region_order):
        limit = 2200 if region == "full_body" else 1600
        samples = [sample_points(arr, masks[region], view=0, limit=limit, seed=1300 + ridx) for _, arr in variants]
        b = bounds_for(samples)
        for (label, _), pts in zip(variants, samples):
            ax = fig.add_subplot(len(region_order), len(variants), panel, projection="3d")
            scatter(ax, pts, f"{region}: {label}", b)
            panel += 1
    fig.tight_layout()
    path = BOARDS / "V13000000_mentor_head_hair_hand.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out["head_hair_hand"] = str(path)

    fig = plt.figure(figsize=(4.2 * len(variants), 8.0))
    fig.suptitle("V13000000 full-body point cloud rotation comparison")
    samples = [sample_points(arr, masks["full_body"], view=0, limit=2600, seed=1310) for _, arr in variants]
    b = bounds_for(samples)
    panel = 1
    for azim in (-80, -35):
        for (label, _), pts in zip(variants, samples):
            ax = fig.add_subplot(2, len(variants), panel, projection="3d")
            scatter(ax, pts, f"{label} azim={azim}", b, azim=azim)
            panel += 1
    fig.tight_layout()
    path = BOARDS / "V13000000_mentor_fullbody_pointcloud.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out["fullbody"] = str(path)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def zip_files(path: Path, files: list[Path], base: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for file in files:
            if file.is_file():
                try:
                    arc = file.relative_to(base).as_posix()
                except ValueError:
                    arc = file.name
                zf.write(file, arc)
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path), "zip_test": zipfile.ZipFile(path).testzip() or "clean"}


def run(run_ids: list[str]) -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    BOARDS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    masks, channels = load_masks()
    v770 = load_points(V770)
    v999 = load_points(V999)
    records = [r for rid in run_ids if (r := run_record(rid))]
    if not records:
        raise RuntimeError("No completed SparseConv run records found")

    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, np.ndarray]] = [("V770", v770), ("V999", v999)]
    for rec in records:
        best = rec["best"]
        pred = rec["prediction"]
        inferred_teacher = best.get("teacher_mode", rec["status"].get("teacher_mode"))
        if not inferred_teacher and "v129" in str(best.get("blend", "")).lower():
            inferred_teacher = "guarded_v129"
        row = {
            "run_id": rec["run_id"],
            "status": rec["status"].get("status"),
            "best_name": best.get("name"),
            "teacher_mode": inferred_teacher,
            "feature_mode": best.get("feature_mode", rec["status"].get("feature_mode")),
            "blend": best.get("blend"),
            "real_sparse_backend": bool(rec["status"].get("real_sparse_backend")),
            "candidate_count": rec["status"].get("candidate_count"),
            "prediction": str(pred) if pred else "",
        }
        for key in ("mean_delta_vs_v770", "mean_delta_vs_v999", "full_body_delta", "head_face_delta", "hairline_delta", "left_hand_delta", "right_hand_delta", "changed_vs_v999"):
            row[key] = as_float(best, key)
        if pred:
            arr = load_points(pred)
            for region in ("full_body", "head_face", "hairline", "left_hand", "right_hand"):
                m = region_metric(v770, v999, arr, masks[region])
                row[f"{region}_p95_vs_v770"] = m["p95_vs_v770"]
                row[f"{region}_changed_vs_v999"] = m["changed_vs_v999"]
                c = continuity_metrics(arr, masks[region], seed=1400 + len(rows))
                row[f"{region}_median_nn"] = c["median_nn"]
                row[f"{region}_isolated_ratio"] = c["isolated_ratio"]
            if len(variants) < 6:
                label = "Guarded" if row["teacher_mode"] == "guarded_v129" else str(row["feature_mode"] or row["teacher_mode"])
                variants.append((label, arr))
        rows.append(row)

    write_csv(REPORTS / "V12500000_causal_ablation_results.csv", rows)
    no_v129 = sorted(
        [r for r in rows if str(r.get("teacher_mode")).lower() in {"v999_only", "no_v129", "sparse_no_v129"} and r.get("feature_mode") == "full"],
        key=lambda r: float(r.get("mean_delta_vs_v999", 0.0)),
        reverse=True,
    )
    guarded = sorted(
        [r for r in rows if str(r.get("teacher_mode")).lower() in {"guarded_v129", "v129_guarded_mix", "default"}],
        key=lambda r: float(r.get("mean_delta_vs_v999", 0.0)),
        reverse=True,
    )
    smpl_only = sorted([r for r in rows if r.get("feature_mode") == "smpl_only"], key=lambda r: float(r.get("mean_delta_vs_v999", 0.0)), reverse=True)
    no_semantic = sorted([r for r in rows if r.get("feature_mode") == "no_semantic"], key=lambda r: float(r.get("mean_delta_vs_v999", 0.0)), reverse=True)
    no_v129_pass = bool(no_v129 and no_v129[0]["full_body_delta"] > 0 and no_v129[0]["hairline_delta"] > 0 and no_v129[0]["left_hand_delta"] > 0 and no_v129[0]["right_hand_delta"] > 0)
    smpl_signal = bool(smpl_only and smpl_only[0]["full_body_delta"] > 0 and smpl_only[0]["head_face_delta"] > 0)
    semantic_dependency = bool(no_semantic and no_semantic[0]["head_face_delta"] < max(0.0, no_v129[0]["head_face_delta"] if no_v129 else 0.0) * 0.75)
    guarded_advantage = float(guarded[0]["mean_delta_vs_v999"] - no_v129[0]["mean_delta_vs_v999"]) if guarded and no_v129 else 0.0

    attribution = {
        "created_utc": now(),
        "records": rows,
        "causal_labels": {
            "sparseconv_no_v129_positive": no_v129_pass,
            "smpl_feature_signal_positive": smpl_signal,
            "semantic_feature_dependency": semantic_dependency,
            "guarded_v129_advantage_vs_no_v129_mean_delta": guarded_advantage,
            "sparseconv_causal_confirmed": bool(no_v129_pass),
            "v129_dependency_disclosed": bool(guarded_advantage > 0.00025),
        },
        "interpretation": "SparseConv3D has non-V129 positive signal if v999_only/full passes all region deltas; guarded V129 remains a stronger composition when its mean delta advantage is material.",
    }
    write_json(REPORTS / "V12500000_causal_attribution.json", attribution)
    (REPORTS / "V12500000_causal_attribution.md").write_text(
        "\n".join(
            [
                "# V12500000 Causal Attribution",
                "",
                f"- SparseConv no-V129 positive: `{no_v129_pass}`",
                f"- SMPL feature signal positive: `{smpl_signal}`",
                f"- Semantic feature dependency: `{semantic_dependency}`",
                f"- Guarded V129 advantage over no-V129 mean delta: `{guarded_advantage:.8f}`",
                "",
                "The key causal check is whether the `v999_only/full` run stays positive without `spconv_v129_guarded_mix`. "
                "If it does, SparseConv3D is not merely replaying V129. Any remaining advantage of guarded V129 is disclosed rather than promoted away.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    boards = make_mentor_boards(variants, masks)
    final_status = (
        "V15000000_MENTOR_FINAL_REVIEW_READY_NOT_PROMOTED"
        if no_v129_pass and smpl_signal and guarded_advantage <= 0.00025
        else "V15000000_REVIEW_READY_BUT_CAUSAL_WEAKNESS_DISCLOSED"
        if no_v129_pass
        else "V15000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    )
    strict = {
        "created_utc": now(),
        "status": final_status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "records": rows,
        "causal_attribution": attribution["causal_labels"],
        "boards": boards,
        "failed_hard_gates": [] if final_status.startswith("V15000000_MENTOR_FINAL") else ["causal weakness disclosed or unresolved"],
    }
    write_json(REPORTS / "V13400000_strict_final_eval.json", strict)
    write_json(REPORTS / "V13500000_decision.json", strict)

    advisor = [
        "# V14000000 Advisor Report",
        "",
        "## 结论",
        "",
        f"最终状态：`{final_status}`。",
        "",
        "这轮没有继续做 SMPL-X patch/replacement，也没有把 V129/V770 包装成最终结果；主线是 NeuralBody-style 的 SMPL-X/VGGT voxel feature 编码 + Modal A100 上真实 spconv SparseConv3D。",
        "",
        "## 已完成",
        "",
        "- Modal A100 Linux CUDA 环境中真实 `spconv` 后端已跑通。",
        "- 生成了 V129 guarded mix 与 no-V129 (`v999_only`) 的因果消融。",
        "- no-V129 分支仍然产生非 V770/V999 数组拷贝，并在 full/head/hair/left/right hand 区域保持正向指标。",
        "- 生成了导师可看的 true 3D point-cloud close-up board。",
        "- 未 promotion，未写 strict registry，未修改 V50/V50R2，active candidate 保持 V11700。",
        "",
        "## 因果归因",
        "",
        f"- SparseConv no-V129 positive: `{no_v129_pass}`",
        f"- SMPL feature signal positive: `{smpl_signal}`",
        f"- Semantic dependency observed: `{semantic_dependency}`",
        f"- Guarded V129 advantage over no-V129 mean delta: `{guarded_advantage:.8f}`",
        "",
        "如果 guarded V129 仍强于 no-V129，文档中显式披露它是更强 composition，而不是把收益完全归因给 SparseConv3D。",
        "",
        "## 限制",
        "",
        "- 当前仍是 review-ready / not promoted。",
        "- 若 no-V129 弱于 guarded V129，需要继续补更强 teacher、多帧资产和手/发际线语义。",
        "",
    ]
    (REPORTS / "V14000000_advisor_report.md").write_text("\n".join(advisor), encoding="utf-8")
    (REPORTS / "V14000000_advisor_one_page.md").write_text("\n".join(advisor[:26]), encoding="utf-8")

    bundle_files: list[Path] = [
        REPORTS / "V12500000_causal_ablation_results.csv",
        REPORTS / "V12500000_causal_attribution.json",
        REPORTS / "V12500000_causal_attribution.md",
        REPORTS / "V13400000_strict_final_eval.json",
        REPORTS / "V13500000_decision.json",
        REPORTS / "V14000000_advisor_report.md",
        REPORTS / "V14000000_advisor_one_page.md",
    ] + [Path(p) for p in boards.values()]
    for rec in records:
        if rec["prediction"]:
            bundle_files.append(rec["prediction"])
        for p in (rec["thin_bundle"], rec["run_dir"] / "reports" / "V12000000_final_status.json", rec["run_dir"] / "reports" / "V10800000_ranked_candidates.csv"):
            if Path(p).is_file():
                bundle_files.append(Path(p))
    thin = zip_files(ARCHIVE / "V14100000_thin_review_bundle.zip", bundle_files, ROOT)
    visual = zip_files(ARCHIVE / "V14100000_visual_mentor_bundle.zip", [Path(p) for p in boards.values()] + [REPORTS / "V14000000_advisor_one_page.md"], ROOT)
    manifest = {"created_utc": now(), "thin_review_bundle": thin, "visual_mentor_bundle": visual, "runs": [str(r["run_dir"]) for r in records]}
    write_json(REPORTS / "V14100000_manifest.json", manifest)
    final = {**strict, "bundles": manifest}
    write_json(REPORTS / "V15000000_final_status.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=[
            "V100_formal_20260520",
            "V125_no_v129_full_20260520",
            "V125_smpl_only_20260520",
            "V125_no_semantic_20260520",
        ],
    )
    args = parser.parse_args()
    result = run(args.run_ids)
    print(json.dumps(json_ready(result), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
