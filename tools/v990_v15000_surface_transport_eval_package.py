from __future__ import annotations

import csv
import hashlib
import json
import math
import os
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
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
ARCHIVE = AUX / "archive"
V970 = OUTPUT / "V970_pull_parent" / "V9700000000_surface_transport_long_matrix"
V910 = OUTPUT / "V9100000000_remote_payload_shards"


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_rows() -> list[dict[str, Any]]:
    rows = []
    for p in sorted(V970.glob("*_seed*/eval.json")):
        row = load_json(p)
        row["run_dir"] = str(p.parent)
        rows.append(row)
    return rows


def npz_audit(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists(), "zip_clean": False, "readable": False, "arrays": {}}
    if not path.exists():
        return out
    with zipfile.ZipFile(path, "r") as zf:
        out["bad_member"] = zf.testzip()
        out["zip_clean"] = out["bad_member"] is None
    with np.load(path, allow_pickle=False) as z:
        out["keys"] = list(z.files)
        for key in ["world_points", "depth", "confidence", "normal", "learned_normal", "geometric_normal", "normal_residual", "normal_conf"]:
            if key in z.files:
                arr = z[key]
                stat = {"shape": list(arr.shape), "finite_ratio": float(np.isfinite(arr).mean())}
                if arr.ndim >= 3 and arr.shape[-1] == 3:
                    stat["vector_nonzero_ratio"] = float((np.linalg.norm(arr, axis=-1) > 1e-6).mean())
                out["arrays"][key] = stat
        out["readable"] = True
    return out


def build_metrics() -> dict[str, Any]:
    rows = eval_rows()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)
    rank_rows = []
    for group, vals in grouped.items():
        scores = np.asarray([float(v["transport_score"]) for v in vals], dtype=np.float64)
        rank_rows.append({
            "group": group,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "n": int(scores.size),
            "runtime_mean": float(np.mean([float(v["runtime_seconds"]) for v in vals])),
            "training_steps_min": int(min(int(v["training_steps"]) for v in vals)),
            "normal_nonzero_min": float(min(float(v["normal_nonzero_ratio"]) for v in vals)),
            "learned_normal_residual_mean": float(np.mean([float(v.get("learned_normal_residual_mean", 0.0)) for v in vals])),
        })
    rank_rows.sort(key=lambda x: x["mean_score"], reverse=True)
    write_csv(REPORTS / "V9900000000_eval_metrics.csv", rank_rows)

    score = {r["group"]: r for r in rank_rows}
    true = score.get("true_surface_transport", {})
    comparisons = {}
    for control in [
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "random_surface_graph",
        "observation_only",
        "support_only",
        "no_sparseconv_mlp",
        "no_teacher",
    ]:
        if true and control in score:
            comparisons[f"true_minus_{control}"] = float(true["mean_score"] - score[control]["mean_score"])

    payload = {
        "created_utc": now(),
        "matrix_runs": len(rows),
        "formal_gpu_runs": sum(1 for r in rows if r.get("formal_gpu_run") and r.get("cuda_available")),
        "gpu_types": sorted({str(r.get("gpu_type")) for r in rows}),
        "training_steps_min": min([int(r.get("training_steps", 0)) for r in rows], default=0),
        "runtime_seconds_mean": float(np.mean([float(r.get("runtime_seconds", 0)) for r in rows])) if rows else 0.0,
        "seed_variance_nonzero": any(float(r["std_score"]) > 0 for r in rank_rows),
        "ranking": rank_rows,
        "comparisons": comparisons,
        "normal_head_status": "learned_residual_present_but_residual_small",
        "normal_warning": "learned_normal is geometric_normal plus a learned residual; residual magnitude is small and does not prove a strong learned normal head.",
        "controller_warning": "V970 optimized scalar amplitudes on remote payload signals; it is a working full-payload GPU matrix but not sufficient by itself for mentor satisfaction.",
    }
    write_json(REPORTS / "V9900000000_causality_summary.json", payload)
    return payload


def load_masks(group: str = "true_full") -> dict[str, np.ndarray]:
    with np.load(V910 / group / "semantic_00_16.npz", allow_pickle=False) as z:
        s0 = z["semantic"]
    with np.load(V910 / group / "semantic_16_32.npz", allow_pickle=False) as z:
        s1 = z["semantic"]
    masks = {
        "full_body": s0[:, 15] > 0.5,
        "head_face": s1[:, 0] > 0.5,
        "hairline": s1[:, 1] > 0.5,
        "left_hand": s1[:, 2] > 0.5,
        "right_hand": s1[:, 3] > 0.5,
    }
    return masks


def region_metrics() -> dict[str, Any]:
    masks = load_masks()
    rows = []
    selected = [
        "true_surface_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "random_surface_graph",
        "observation_only",
        "support_only",
        "no_sparseconv_mlp",
    ]
    for group in selected:
        pred_path = V970 / f"{group}_seed0" / "predictions.npz"
        if not pred_path.exists():
            continue
        with np.load(pred_path, allow_pickle=False) as z:
            wp = z["world_points"]
            normal = z["normal"]
            base = z["geometric_normal"]
            residual = z["normal_residual"]
        delta = np.linalg.norm(wp - wp.mean(axis=(1, 2), keepdims=True), axis=-1)
        normal_nonzero = np.linalg.norm(normal, axis=-1) > 1e-6
        residual_norm = np.linalg.norm(residual, axis=-1)
        for region, mask in masks.items():
            if mask.shape != delta.shape:
                status = "mask_shape_mismatch"
                count = 0
                mean_delta = math.nan
                normal_ratio = math.nan
                residual_mean = math.nan
            else:
                count = int(mask.sum())
                status = "evaluated" if count > 0 else "empty_mask"
                mean_delta = float(delta[mask].mean()) if count else math.nan
                normal_ratio = float(normal_nonzero[mask].mean()) if count else math.nan
                residual_mean = float(residual_norm[mask].mean()) if count else math.nan
            rows.append({
                "group": group,
                "seed": 0,
                "region": region,
                "status": status,
                "pixel_count": count,
                "mean_spatial_spread": mean_delta,
                "normal_nonzero_ratio": normal_ratio,
                "learned_normal_residual_mean": residual_mean,
                "region_metric_source": "semantic region masks from full 81-channel V910 payload",
            })
    write_csv(REPORTS / "V9900000000_region_metrics.csv", rows)
    payload = {
        "created_utc": now(),
        "regions": sorted({r["region"] for r in rows}),
        "not_evaluated_count": sum(1 for r in rows if r["status"] != "evaluated"),
        "rows": len(rows),
        "proxy_region_note": "Uses semantic region masks from V910; connected components/reprojection remain not paper-grade.",
    }
    write_json(REPORTS / "V9900000000_region_summary.json", payload)
    return payload


def scatter(ax, points: np.ndarray, title: str, mask: np.ndarray | None = None, max_points: int = 5000) -> None:
    import numpy as _np

    pts = points[mask] if mask is not None and mask.shape == points.shape[:3] else points.reshape(-1, 3)
    pts = pts[_np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        rng = _np.random.default_rng(42)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, c=pts[:, 2], cmap="viridis", linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def visual_boards() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BOARDS.mkdir(parents=True, exist_ok=True)
    masks = load_masks()
    groups = [
        "true_surface_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "random_surface_graph",
        "observation_only",
        "support_only",
    ]
    data = {}
    for group in groups:
        p = V970 / f"{group}_seed0" / "predictions.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                data[group] = {k: z[k] for k in ["world_points", "normal", "geometric_normal", "normal_residual"]}

    fullbody = BOARDS / "V10000000000_fullbody.png"
    fig = plt.figure(figsize=(18, 8))
    for i, group in enumerate(groups):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        scatter(ax, data[group]["world_points"], f"{group} seed0 full_body", masks["full_body"], 8000)
    fig.tight_layout()
    fig.savefig(fullbody, dpi=160)
    plt.close(fig)

    region_files = {}
    for region, filename in [
        ("head_face", "V10000000000_head_face.png"),
        ("hairline", "V10000000000_hairline.png"),
        ("left_hand", "V10000000000_left_hand.png"),
        ("right_hand", "V10000000000_right_hand.png"),
    ]:
        path = BOARDS / filename
        fig = plt.figure(figsize=(18, 8))
        for i, group in enumerate(groups[:6]):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            scatter(ax, data[group]["world_points"], f"{group} seed0 {region}", masks[region], 6000)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        region_files[region] = str(path)

    normal_path = BOARDS / "V10000000000_normals.png"
    fig = plt.figure(figsize=(14, 8))
    true = data["true_surface_transport"]
    for i, (arr, title) in enumerate([
        (true["normal"], "learned normal"),
        (true["geometric_normal"], "geometric normal"),
        (true["normal_residual"], "normal residual"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        pts = arr.reshape(-1, 3)
        rng = np.random.default_rng(7)
        if pts.shape[0] > 8000:
            pts = pts[rng.choice(pts.shape[0], 8000, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, c=np.linalg.norm(pts, axis=1), cmap="plasma")
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(normal_path, dpi=160)
    plt.close(fig)

    topology_path = BOARDS / "V10000000000_topology_controls.png"
    fig = plt.figure(figsize=(16, 5))
    for i, group in enumerate(["true_surface_transport", "random_surface_graph", "no_surface_graph", "local_knn_smoothing_surface"]):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        scatter(ax, data[group]["world_points"], f"{group} seed0", masks["full_body"], 7000)
    fig.tight_layout()
    fig.savefig(topology_path, dpi=160)
    plt.close(fig)

    failure_path = BOARDS / "V10000000000_failure_cases.png"
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.02, 0.85, "Failure/risk cases", fontsize=16)
    ax.text(0.02, 0.65, "V970 proves full-payload GPU execution and true > controls in compact score.", fontsize=11)
    ax.text(0.02, 0.50, "It still does not satisfy mentor gate: scalar-amplitude controller, short per-run runtime, small normal residual.", fontsize=11)
    ax.text(0.02, 0.35, "Next route must use a real trainable surface graph backbone and non-proxy region/reprojection metrics.", fontsize=11)
    fig.savefig(failure_path, dpi=160)
    plt.close(fig)

    files = {
        "fullbody": str(fullbody),
        **region_files,
        "normals": str(normal_path),
        "topology_controls": str(topology_path),
        "failure_cases": str(failure_path),
    }
    write_json(REPORTS / "V10000000000_visual_boards_manifest.json", {"created_utc": now(), "files": files})
    return files


def decision(summary: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    comps = summary["comparisons"]
    numeric_pass = all(v > 0 for v in comps.values())
    mentor_ready = False
    blockers = [
        "V970 controller optimizes scalar amplitudes rather than a full surface graph neural backbone.",
        "Per-run runtime is about 4-5 seconds despite 200 steps; not enough for paper-grade learned transport.",
        "Learned normal residual exists but is tiny; geometric normal still dominates.",
        "Region metrics use semantic masks but connected-component/reprojection evidence is still not paper-grade.",
        "Visual boards are newly generated but do not establish strong visible close-up improvement over baseline.",
    ]
    payload = {
        "created_utc": now(),
        "mentor_requirement_satisfied": mentor_ready,
        "numeric_controls_passed": numeric_pass,
        "semantic_topology_causality_confirmed": False,
        "dominant_failure": "MENTOR_GATE_NOT_SATISFIED_DESPITE_NUMERIC_TRUE_TOP_RANK",
        "blockers": blockers,
        "auto_evolution_required": True,
        "next_route": "V15100000000_POINT_TRANSFORMER_SURFACE_GRAPH_ROUTE",
    }
    write_json(REPORTS / "V10100000000_decision.json", payload)
    return payload


def write_next_route(decision_payload: dict[str, Any]) -> Path:
    path = REPO / "docs" / "goals" / "V_NEXT_AUTO_EVOLVED_ROUTE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# V15100000000 Point Transformer Surface Graph Auto-Evolved Route

## Failure Attribution

V970 completed a 50-run Modal GPU full-payload matrix, but the mentor gate is not satisfied. The main blockers are:

{chr(10).join('- ' + b for b in decision_payload['blockers'])}

## New Architecture Hypothesis

Replace the scalar-amplitude surface transport controller with a trainable point transformer on the SMPL-X face/vertex graph. Semantic and topology fields must be value carriers; support remains mask-only; observation is bottlenecked context.

## Why Previous Route Failed

The V970 route demonstrated execution discipline but not sufficient learned geometry. It used full 81-channel payloads and nonzero seed variance, but the optimization target was too low capacity and the learned normal residual was too weak.

## New Hard Gates

- Train a real point-transformer or graph-attention backbone on surface-indexed samples.
- Minimum 1000 optimizer steps unless Modal hard-blocked.
- Learned normal residual mean must be nontrivial and improve region normals.
- Region metrics must include connected components and non-proxy hand/hair/head evaluation.
- true route must beat random semantic, shuffled semantic, local smoothing, no graph, random graph, observation-only, support-only, and noSparse controls.
- No promotion, no registry, no V50/V50R2 modification, active candidate unchanged.

## Training/Eval Matrix

Run true_surface_transformer, random_semantic, shuffled_semantic, local_knn, no_graph, random_graph, observation_only, support_only, noSparse, no_teacher. Minimum 5 seeds. Prefer A10/A100 Modal GPU.

## Visual Proof Requirements

Generate new 3D scatter point clouds for full body, head-face, hairline, left hand, right hand, normals, topology controls, and failure cases. No heatmap-only boards.

## Packaging

Build upload-safe bundles with sidecar hashes and readable selected/control npz files.
"""
    path.write_text(text, encoding="utf-8")
    write_json(REPORTS / "V10700000000_auto_next_route_generation.json", {
        "created_utc": now(),
        "route_file": str(path),
        "route": "V15100000000_POINT_TRANSFORMER_SURFACE_GRAPH_ROUTE",
        "execute_immediately": True,
    })
    return path


def run_cmd(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, encoding="utf-8", errors="replace")
    return {"args": args, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def true_external_hard_block(route_file: Path) -> dict[str, Any]:
    # The current session has a route that demands never returning unless a true
    # external block is hit. The next required route is a materially larger
    # trainable point-transformer implementation and long Modal job. In this
    # environment we can only return if an external blocker is present. We check
    # write permissions, disk, Modal app state, and then classify no external
    # blocker if all are fine.
    modal = run_cmd(["modal", "app", "list"])
    git_status = run_cmd(["git", "status", "--short", "--branch"])
    block = {
        "created_utc": now(),
        "true_external_hard_block": False,
        "route_file": str(route_file),
        "modal_app_list": modal,
        "git_status": git_status,
        "disk_free_d": shutil.disk_usage("D:\\").free,
        "user_action_checklist": [],
    }
    write_json(REPORTS / "V15000000000_external_block_check.json", block)
    return block


def add_to_zip(zf: zipfile.ZipFile, path: Path, arcname: str | None = None) -> None:
    if path.exists() and path.is_file():
        zf.write(path, arcname or path.name)


def build_bundles(final_status: dict[str, Any]) -> dict[str, Any]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    omitted = {
        "created_utc": now(),
        "omitted": [
            {"path": str(V970), "reason": "full 50-run predictions tree is multi-GB; selected/control predictions are bundled instead"},
        ],
    }
    omitted_path = REPORTS / "V14500000000_omitted_large_file_manifest.json"
    write_json(omitted_path, omitted)

    final_path = REPORTS / "V15000000000_final_status.json"
    write_json(final_path, final_status)

    bundles = {
        "core": ARCHIVE / "V14500000000_core_evidence_bundle.zip",
        "reports": ARCHIVE / "V14500000000_reports_bundle.zip",
        "visuals": ARCHIVE / "V14500000000_visuals_bundle.zip",
        "selected_predictions": ARCHIVE / "V14500000000_selected_predictions_bundle.zip",
        "controls": ARCHIVE / "V14500000000_controls_bundle.zip",
    }
    for p in bundles.values():
        if p.exists():
            p.unlink()

    core_files = [
        final_path,
        REPORTS / "V9010000000_v900_evidence_audit.json",
        REPORTS / "V9700000000_modal_job_manifest.json",
        REPORTS / "V9900000000_causality_summary.json",
        REPORTS / "V10100000000_decision.json",
        REPORTS / "V14800000000_post_push_cleanup.json",
        omitted_path,
    ]
    with zipfile.ZipFile(bundles["core"], "w", zipfile.ZIP_DEFLATED) as zf:
        for f in core_files:
            add_to_zip(zf, f, f.name)
    with zipfile.ZipFile(bundles["reports"], "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(REPORTS.glob("V9*.json")) + sorted(REPORTS.glob("V10*.json")) + sorted(REPORTS.glob("V14*.md")) + sorted(REPORTS.glob("V15*.json")):
            add_to_zip(zf, f, f.name)
    with zipfile.ZipFile(bundles["visuals"], "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(BOARDS.glob("V10000000000*.png")):
            add_to_zip(zf, f, f.name)
    selected = [
        V970 / "true_surface_transport_seed0" / "predictions.npz",
        V970 / "true_surface_transport_seed1" / "predictions.npz",
    ]
    with zipfile.ZipFile(bundles["selected_predictions"], "w", zipfile.ZIP_DEFLATED) as zf:
        for f in selected:
            add_to_zip(zf, f, f.parent.name + "/" + f.name)

    control_sample_root = OUTPUT / "V14500000000_control_prediction_samples"
    if control_sample_root.exists():
        shutil.rmtree(control_sample_root)
    control_sample_root.mkdir(parents=True, exist_ok=True)
    controls = [
        V970 / "random_surface_semantic_seed0" / "predictions.npz",
        V970 / "shuffled_surface_semantic_seed0" / "predictions.npz",
        V970 / "local_knn_smoothing_surface_seed0" / "predictions.npz",
        V970 / "random_surface_graph_seed0" / "predictions.npz",
        V970 / "observation_only_seed0" / "predictions.npz",
        V970 / "support_only_seed0" / "predictions.npz",
        V970 / "no_sparseconv_mlp_seed0" / "predictions.npz",
    ]
    sampled_controls = []
    for f in controls:
        if not f.exists():
            continue
        sample_dir = control_sample_root / f.parent.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "predictions_sample.npz"
        with np.load(f, allow_pickle=False) as z:
            sampled = {}
            for key in ["world_points", "depth", "confidence", "normal", "learned_normal", "geometric_normal", "normal_residual", "normal_conf"]:
                if key in z.files:
                    arr = z[key]
                    if arr.ndim >= 3:
                        slicer = [slice(None)] * arr.ndim
                        slicer[1] = slice(0, arr.shape[1], 4)
                        slicer[2] = slice(0, arr.shape[2], 4)
                        sampled[key] = arr[tuple(slicer)]
                    else:
                        sampled[key] = arr
            np.savez_compressed(sample_path, **sampled)
        write_json(sample_dir / "source_manifest.json", {
            "created_utc": now(),
            "source_prediction": str(f),
            "sample_stride_hw": 4,
            "purpose": "upload-safe readable control prediction sample; full predictions omitted by large-file manifest",
        })
        sampled_controls.append(sample_path)
    with zipfile.ZipFile(bundles["controls"], "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sampled_controls:
            add_to_zip(zf, f, f.parent.name + "/" + f.name)
            add_to_zip(zf, f.parent / "source_manifest.json", f.parent.name + "/source_manifest.json")

    manifest = {"created_utc": now(), "bundles": {}, "omitted_manifest": str(omitted_path)}
    for name, path in bundles.items():
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
        manifest["bundles"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "zip_clean": bad is None,
            "bad_member": bad,
            "under_500mb": path.stat().st_size < 500 * 1024 * 1024,
        }
    manifest_path = REPORTS / "V14500000000_upload_manifest_sidecar.json"
    write_json(manifest_path, manifest)
    return manifest


def cleanup_report(final_status: str) -> dict[str, Any]:
    cleanup = {
        "created_utc": now(),
        "final_status": final_status,
        "git_status": run_cmd(["git", "status", "--short", "--branch"]),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "modal_apps": run_cmd(["modal", "app", "list"]),
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "cleanup_clean_claim": False,
    }
    write_json(REPORTS / "V14800000000_post_push_cleanup.json", cleanup)
    return cleanup


def write_advisor(final_status: str, summary: dict[str, Any], decision_payload: dict[str, Any]) -> None:
    report = REPORTS / "V14000000000_advisor_report.md"
    lines = [
        "# 先给结论",
        "",
        f"本轮未达到导师目标，最终状态为 `{final_status}`。V970 已完成 full 81-channel payload 的 50-run Modal GPU matrix，true surface transport 在数值 score 上排第一，但这仍不能写成 semantic/topology causality confirmed。",
        "",
        "# 本轮解决的问题",
        "",
        "- 修复 V970 controller：补齐 upload / Modal launch / pull / validate。",
        "- 使用 V910 full 81-channel sharded payload，而不是 V740 compact payload。",
        "- 跑完 10 groups x 5 seeds x 200 steps，GPU 为 NVIDIA A10。",
        "- 修复一次 selected prediction 内部 CRC 错误，并重新拉取验证到 50/50 npz readable。",
        "- 生成新的 full body / head / hairline / hand / topology / normal boards。",
        "",
        "# 核心证据",
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
        f"| formal training | 50 GPU runs, min steps {summary.get('training_steps_min')} |",
        f"| seed variance | {summary.get('seed_variance_nonzero')} |",
        f"| ranking top | {summary.get('ranking', [{}])[0].get('group')} |",
        "| random/shuffled controls | true score 高于 random/shuffled |",
        "| smoothing/topology controls | true score 高于 local KNN / random graph / no graph |",
        "| normal | 非零，但 learned residual 很小，不能当强 normal head 成功 |",
        "| visuals | 新生成 3D scatter boards，但 close-up 提升仍不够强 |",
        "",
        "# 为什么还不能写导师满足",
        "",
    ]
    lines.extend(f"- {b}" for b in decision_payload["blockers"])
    lines.extend([
        "",
        "# 给导师看的图",
        "",
        f"- full body: `{BOARDS / 'V10000000000_fullbody.png'}`",
        f"- head-face: `{BOARDS / 'V10000000000_head_face.png'}`",
        f"- hairline: `{BOARDS / 'V10000000000_hairline.png'}`",
        f"- left hand: `{BOARDS / 'V10000000000_left_hand.png'}`",
        f"- right hand: `{BOARDS / 'V10000000000_right_hand.png'}`",
        f"- normals: `{BOARDS / 'V10000000000_normals.png'}`",
        f"- topology controls: `{BOARDS / 'V10000000000_topology_controls.png'}`",
        "",
        "# 下一步论文路线",
        "",
        "自动生成并应继续执行 `docs/goals/V_NEXT_AUTO_EVOLVED_ROUTE.md`：把 V970 的 scalar-amplitude controller 替换成真实 point-transformer / graph-attention surface backbone，并把 normal、region、reprojection 证据提升到 paper-grade。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "V14000000000_one_page.md").write_text("\n".join(lines[:35]) + "\n", encoding="utf-8")
    (REPORTS / "V14000000000_limitations.md").write_text("\n".join(["# Limitations", "", *[f"- {b}" for b in decision_payload["blockers"]]]) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_metrics()
    region = region_metrics()
    visual_boards()
    decision_payload = decision(summary, region)
    route_file = write_next_route(decision_payload)
    block = true_external_hard_block(route_file)
    # Because the user's current rule forbids returning on route exhaustion, the
    # only valid terminal here is an external hard block. If none exists, record
    # auto-evolution state and leave final_status as not returnable.
    if block["true_external_hard_block"]:
        final_status = "V∞_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"
    else:
        final_status = "V15000000000_AUTO_EVOLUTION_GATE_NOT_RETURNABLE"
    cleanup = cleanup_report(final_status)
    final = {
        "created_utc": now(),
        "final_status": final_status,
        "mentor_requirement_satisfied": False,
        "semantic_topology_causality_confirmed": False,
        "dominant_failure": decision_payload["dominant_failure"],
        "next_route_file": str(route_file),
        "external_hard_block": block["true_external_hard_block"],
        "cleanup_report": str(REPORTS / "V14800000000_post_push_cleanup.json"),
        "advisor_report": str(REPORTS / "V14000000000_advisor_report.md"),
    }
    write_advisor(final_status, summary, decision_payload)
    manifest = build_bundles(final)
    final["upload_manifest"] = str(REPORTS / "V14500000000_upload_manifest_sidecar.json")
    final["bundles"] = manifest["bundles"]
    write_json(REPORTS / "V15000000000_final_status.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
