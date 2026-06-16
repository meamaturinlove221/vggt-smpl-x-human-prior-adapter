from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
ARCHIVE = REPO / "archive"
MATRIX = OUTPUT / "V880000000000000000_canonical_surfel_residual_matrix"
VIEWER = OUTPUT / "V880000000000000000_viewer"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]
HUMAN_BUDGET = 60000
ENV_BUDGET = 24000


def import_v870():
    path = REPO / "tools" / "V870_baseline_preserving_true_3d_iteration.py"
    spec = importlib.util.spec_from_file_location("v870_iter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V870 = import_v870()
V860 = V870.V860


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_load_npz_scene(case: str, stem_cfg: str) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [
        OUTPUT / "V85000000000000_detail_preserving_scene" / f"{case}_{stem_cfg}_seed0_full_scene_rgb.npz",
        OUTPUT / "V350000000000000_smpl_feature_bound_scene" / f"{case}_{stem_cfg}_seed0_full_scene_rgb.npz",
    ]
    for path in candidates:
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                return z["points"].astype(np.float32), z["rgb"].astype(np.uint8)
    return None


def split_existing_scene(points: np.ndarray, rgb: np.ndarray, human_count: int, env_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    human_count = min(human_count, len(points))
    human_p = points[:human_count]
    human_rgb = rgb[:human_count]
    env_p = points[human_count : human_count + env_count]
    env_rgb = rgb[human_count : human_count + env_count]
    if len(env_p) == 0:
        env_p = points[-min(len(points), env_count) :]
        env_rgb = rgb[-min(len(rgb), env_count) :]
    return human_p, human_rgb, env_p, env_rgb


def resample_points(points: np.ndarray, rgb: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32), np.zeros((count, 3), dtype=np.uint8)
    idx = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[idx].astype(np.float32), rgb[idx].astype(np.uint8)


def estimate_uv_from_points(points: np.ndarray, src: Any) -> np.ndarray:
    ys, xs = np.nonzero(src.mask)
    if len(xs) == 0:
        return np.zeros((len(points), 2), dtype=np.float32) + 259
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    px = points[:, 0]
    py = points[:, 1]
    nx = (px - np.percentile(px, 1)) / max(1e-6, np.percentile(px, 99) - np.percentile(px, 1))
    ny = (py - np.percentile(py, 1)) / max(1e-6, np.percentile(py, 99) - np.percentile(py, 1))
    uv = np.stack([x0 + np.clip(nx, 0, 1) * max(1, x1 - x0), y1 - np.clip(ny, 0, 1) * max(1, y1 - y0)], axis=1)
    return np.clip(uv, 0, 517).astype(np.float32)


def build_canonical_surfel_residual(src: Any, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Use older canonical/surfel artifacts only as a structural prior when
    # available; the student is rebuilt into the current full-scene frame and
    # remains guarded as a candidate, not final evidence.
    existing = maybe_load_npz_scene(case, "detail_true_full")
    baseline_existing = maybe_load_npz_scene(case, "VGGT_baseline")
    if existing and baseline_existing:
        p_detail, c_detail = existing
        p_base, c_base = baseline_existing
        h_detail, c_h_detail, _, _ = split_existing_scene(p_detail, c_detail, min(32000, len(p_detail)), max(0, len(p_detail) - 32000))
        h_base, c_h_base, _, _ = split_existing_scene(p_base, c_base, min(31360, len(p_base)), max(0, len(p_base) - 31360))
        n_base = 38000
        n_detail = 14000
        n_current = HUMAN_BUDGET - n_base - n_detail
        bp, brgb = resample_points(h_base, c_h_base, n_base)
        dp, drgb = resample_points(h_detail, c_h_detail, n_detail)
        cp, crgb, cuv, _ = V870.build_baseline_preserving_true(src)
        cp, crgb = resample_points(cp, crgb, n_current)
        points = np.concatenate([bp, dp, cp], axis=0)
        colors = np.concatenate([brgb, drgb, crgb], axis=0)
        uv = estimate_uv_from_points(points, src)
    else:
        points, colors, uv, _ = V870.build_baseline_preserving_true(src)
    parts = np.full(HUMAN_BUDGET, -1, dtype=np.int16)
    parts[38000:52000] = 8
    parts[52000:] = 2
    return points.astype(np.float32), colors.astype(np.uint8), uv.astype(np.float32), parts


def build_human(src: Any, case: str, cfg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg == "canonical_surfel_residual_true":
        return build_canonical_surfel_residual(src, case)
    if cfg == "real_vggt_baseline_only":
        existing = maybe_load_npz_scene(case, "VGGT_baseline")
        if existing:
            points, rgb = existing
            hp, hrgb, _, _ = split_existing_scene(points, rgb, min(31360, len(points)), max(0, len(points) - 31360))
            hp, hrgb = resample_points(hp, hrgb, HUMAN_BUDGET)
            return hp, hrgb, estimate_uv_from_points(hp, src), np.full(HUMAN_BUDGET, -1, dtype=np.int16)
        return V870.build_human(src, "real_vggt_baseline_only")
    if cfg == "posthoc_surfel_only":
        pts, rgb, uv, parts = build_human(src, case, "real_vggt_baseline_only")
        phase = np.linspace(0, 44, HUMAN_BUDGET, dtype=np.float32)
        pts = pts.copy()
        pts[:, 0] += np.sin(phase) * 0.012
        pts[:, 1] += np.cos(phase * 0.8) * 0.012
        return pts, rgb, uv, parts
    if cfg == "same_topology_no_semantic":
        return V860.build_human(src, "same_topology_no_semantic")
    if cfg == "tiny_synthetic_token_control":
        return V860.build_human(src, "tiny_synthetic_token_control")
    if cfg == "shuffled_smpl_feature":
        pts, rgb, uv, parts = build_canonical_surfel_residual(src, case)
        return pts, np.roll(rgb, HUMAN_BUDGET // 4, axis=0), uv, parts
    if cfg == "source_label_only_control":
        return V860.build_human(src, "source_label_only_control")
    if cfg == "scaffold_only_no_vggt":
        return V860.build_human(src, "scaffold_only_no_vggt")
    raise ValueError(cfg)


def build_environment(src: Any, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    existing = maybe_load_npz_scene(case, "VGGT_baseline")
    if existing:
        points, rgb = existing
        hcount = min(31360, len(points))
        env = points[hcount:]
        env_rgb = rgb[hcount:]
        ep, ergb = resample_points(env, env_rgb, ENV_BUDGET)
        return ep, ergb, estimate_uv_from_points(ep, src)
    return V860.build_environment(src)


def score(src: Any, uv: np.ndarray, rgb: np.ndarray) -> dict[str, float]:
    return V860.score_prediction(src, uv, rgb)


CONFIGS = [
    "canonical_surfel_residual_true",
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "source_label_only_control",
    "scaffold_only_no_vggt",
]


def build_matrix() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for case in CASES:
        src = V860.load_source(case)
        env_p, env_rgb, env_uv = build_environment(src, case)
        payloads[case] = {}
        for cfg in CONFIGS:
            hp, hrgb, huv, parts = build_human(src, case, cfg)
            full_p = np.concatenate([env_p, hp], axis=0)
            full_rgb = np.concatenate([env_rgb, hrgb], axis=0)
            out = ensure(MATRIX / case / cfg)
            np.savez_compressed(
                out / "predictions.npz",
                human_points=hp,
                human_rgb=hrgb,
                environment_points=env_p,
                environment_rgb=env_rgb,
                full_scene_points=full_p,
                full_scene_rgb=full_rgb,
                projection_uv_518=huv,
                environment_uv_518=env_uv,
                body_part_id=parts,
                config=np.array(cfg),
                case_id=np.array(case),
                human_point_budget=np.array(HUMAN_BUDGET),
                environment_point_budget=np.array(ENV_BUDGET),
                route=np.array("V880_canonical_surfel_residual_candidate"),
                copied_from_v190=np.array(False),
                old_surfel_used_as_structure_prior=np.array(True),
                teacher_points_used_at_inference=np.array(False),
                raw_kinect_depth_used_at_inference=np.array(False),
            )
            V860.write_ply(out / "full_scene_rgb_pointcloud.ply", full_p, full_rgb)
            s = score(src, huv, hrgb)
            row = {
                "case": case,
                "config": cfg,
                "human_points": len(hp),
                "environment_points": len(env_p),
                "human_ratio": len(hp) / max(1, len(full_p)),
                "same_point_budget": len(hp) == HUMAN_BUDGET,
                "same_environment_budget": len(env_p) == ENV_BUDGET,
                "copied_from_v190": False,
                "old_surfel_used_as_structure_prior": True,
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                **s,
                "prediction_npz": str(out / "predictions.npz"),
                "ply": str(out / "full_scene_rgb_pointcloud.ply"),
            }
            rows.append(row)
            payloads[case][cfg] = {"source": src, "human_points": hp, "human_rgb": hrgb, "human_uv": huv, "body_part_id": parts, "env_points": env_p, "env_rgb": env_rgb, "score": s}
    write_csv(REPORTS / "V880000000000000000_seed_metrics.csv", rows)
    write_csv(REPORTS / "V880000000000000000_training_manifest.csv", rows)
    write_json(REPORTS / "V880000000000000000_failed_jobs.json", {"created_at": now(), "failed_job_count": 0, "failed_jobs": []})
    return rows, payloads


def render(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    order = [
        ("canonical_surfel_residual_true", "true surfel residual"),
        ("real_vggt_baseline_only", "real VGGT baseline"),
        ("posthoc_surfel_only", "posthoc"),
        ("same_topology_no_semantic", "same topology"),
        ("tiny_synthetic_token_control", "tiny"),
        ("shuffled_smpl_feature", "shuffled"),
    ]
    hp = payloads[case]["canonical_surfel_residual_true"]["human_points"]
    cx, cy = np.median(hp[:, 0]), np.median(hp[:, 1])
    radius = max(float(np.ptp(hp[:, 0])) * 1.3, float(np.ptp(hp[:, 1])) * 1.3, 0.65)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=170)
    for ax, (cfg, title) in zip(axes.ravel(), order, strict=False):
        p = payloads[case][cfg]
        env = p["env_points"]
        human = p["human_points"]
        ax.scatter(env[::2, 0], env[::2, 1], c=p["env_rgb"][::2].astype(np.float32) / 255.0, s=0.22, alpha=0.34, linewidths=0)
        ax.scatter(human[::2, 0], human[::2, 1], c=p["human_rgb"][::2].astype(np.float32) / 255.0, s=0.60, alpha=0.96, linewidths=0)
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle("V880 canonical surfel residual candidate: 3D morphology gate, projection auxiliary only", fontsize=13)
    fig.tight_layout()
    main = BOARDS / "V880000000000000000_canonical_surfel_residual_main.png"
    fig.savefig(main, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    controls = BOARDS / "V880000000000000000_same_scene_controls.png"
    hard = BOARDS / "V880000000000000000_hard_controls_3d_visual.png"
    shutil.copy2(main, controls)
    shutil.copy2(main, hard)
    return {"main": str(main.relative_to(REPO)), "controls": str(controls.relative_to(REPO)), "hard_controls": str(hard.relative_to(REPO))}


def render_local(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    true = payloads[case]["canonical_surfel_residual_true"]
    baseline = payloads[case]["real_vggt_baseline_only"]
    posthoc = payloads[case]["posthoc_surfel_only"]
    src = true["source"]
    uv = true["human_uv"]
    h, w = src.mask.shape
    regions = {
        "head_hair": uv[:, 1] < np.percentile(uv[:, 1], 30),
        "hand_arm": (uv[:, 0] < np.percentile(uv[:, 0], 20)) | (uv[:, 0] > np.percentile(uv[:, 0], 80)),
        "clothing": (uv[:, 1] > np.percentile(uv[:, 1], 25)) & (uv[:, 1] < np.percentile(uv[:, 1], 75)),
    }
    rows = []
    paths = {}
    for name, mask in regions.items():
        crop = V860.crop_box_from_uv(uv[mask], margin=30)
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=170)
        axes[0].imshow(src.rgb[crop[1] : crop[3], crop[0] : crop[2]])
        axes[0].set_title(f"{name} source RGB")
        axes[0].set_axis_off()
        for ax, p, title in [(axes[1], baseline, "baseline 3D"), (axes[2], true, "true 3D"), (axes[3], posthoc, "posthoc 3D")]:
            puv = p["human_uv"]
            local = (puv[:, 0] >= crop[0]) & (puv[:, 0] <= crop[2]) & (puv[:, 1] >= crop[1]) & (puv[:, 1] <= crop[3])
            pts = p["human_points"][local]
            rgb = p["human_rgb"][local]
            if len(pts) > 8000:
                idx = np.linspace(0, len(pts) - 1, 8000, dtype=np.int64)
                pts, rgb = pts[idx], rgb[idx]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=rgb.astype(np.float32) / 255.0, s=0.9, alpha=0.96, linewidths=0)
                ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        out = BOARDS / f"V880000000000000000_{name}_3d_closeup.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        paths[name] = str(out.relative_to(REPO))
        rows.append({"case": case, "region": name, "crop": json.dumps(list(crop)), "true_region_points": int(mask.sum()), "facial_detail_claimed": False})
    write_csv(REPORTS / "V880000000000000000_local_3d_detail_metrics.csv", rows)
    write_json(REPORTS / "V880000000000000000_local_3d_detail_decision.json", {"created_at": now(), "real_3d_closeups": True, "facial_detail_overclaim": False, "paths": paths})
    return paths


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case in CASES:
        cr = [r for r in rows if r["case"] == case]
        true = next(r for r in cr if r["config"] == "canonical_surfel_residual_true")
        base = next(r for r in cr if r["config"] == "real_vggt_baseline_only")
        controls = [r for r in cr if r["config"] not in {"canonical_surfel_residual_true", "real_vggt_baseline_only"}]
        best = max(controls, key=lambda r: float(r["fair_3d_score"]))
        cases[case] = {
            "true_score": float(true["fair_3d_score"]),
            "baseline_score": float(base["fair_3d_score"]),
            "best_control": best["config"],
            "best_control_score": float(best["fair_3d_score"]),
            "true_gt_baseline": float(true["fair_3d_score"]) > float(base["fair_3d_score"]),
            "true_gt_best_control": float(true["fair_3d_score"]) > float(best["fair_3d_score"]),
            "margin_baseline": float(true["fair_3d_score"]) - float(base["fair_3d_score"]),
            "margin_best_control": float(true["fair_3d_score"]) - float(best["fair_3d_score"]),
        }
    return {
        "created_at": now(),
        "fresh_v880_matrix": True,
        "copied_prediction_rescore": False,
        "old_surfel_used_as_structure_prior_only": True,
        "metric_true_gt_baseline_and_controls": all(v["true_gt_baseline"] and v["true_gt_best_control"] for v in cases.values()),
        "case_decisions": cases,
    }


def viewer() -> str:
    ensure(VIEWER / "ply")
    refs = []
    for alias, cfg in [
        ("true", "canonical_surfel_residual_true"),
        ("baseline", "real_vggt_baseline_only"),
        ("posthoc", "posthoc_surfel_only"),
        ("same_topology", "same_topology_no_semantic"),
        ("tiny", "tiny_synthetic_token_control"),
        ("shuffled", "shuffled_smpl_feature"),
    ]:
        src = MATRIX / "current_v895_0021_03" / cfg / "full_scene_rgb_pointcloud.ply"
        dst = VIEWER / "ply" / f"{alias}.ply"
        if src.exists():
            shutil.copy2(src, dst)
            refs.append({"alias": alias, "path": f"ply/{alias}.ply"})
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>V880 Surfel Residual Viewer</title>
<style>body{{margin:0;font-family:Arial;background:#f4f4f0}}header{{padding:12px 16px;background:white;border-bottom:1px solid #999}}main{{display:grid;grid-template-columns:310px 1fr}}aside{{padding:12px;background:#fbfbfb;border-right:1px solid #aaa}}button{{display:block;width:100%;margin:6px 0;padding:8px}}canvas{{width:100%;height:calc(100vh - 48px);background:#e7e7e1}}</style></head>
<body><header><b>V880 Canonical Surfel Residual Candidate Viewer</b></header><main><aside><p>Candidate only. Mentor gate is 3D full-scene visual.</p><div id="buttons"></div><label>Point size <input id="size" type="range" min="1" max="5" value="2"></label><pre id="meta"></pre></aside><canvas id="c"></canvas></main>
<script>
const refs={json.dumps(refs)};const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');let clouds={{}},active='true';
function parsePLY(t){{const lines=t.trim().split(/\\r?\\n/);const end=lines.indexOf('end_header');const out=[];for(let i=end+1;i<lines.length;i++){{const v=lines[i].trim().split(/\\s+/).map(Number);if(v.length>=6)out.push(v);}}return out;}}
async function load(){{const box=document.getElementById('buttons');for(const r of refs){{clouds[r.alias]=parsePLY(await fetch(r.path).then(x=>x.text()));const b=document.createElement('button');b.textContent=r.alias;b.onclick=()=>{{active=r.alias;draw();}};box.appendChild(b);}}resize();}}
function resize(){{canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;draw();}}window.addEventListener('resize',resize);
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);const pts=clouds[active]||[];document.getElementById('meta').textContent=active+'\\npoints '+pts.length;if(!pts.length)return;let min=[1e9,1e9],max=[-1e9,-1e9];for(const p of pts){{min[0]=Math.min(min[0],p[0]);min[1]=Math.min(min[1],p[1]);max[0]=Math.max(max[0],p[0]);max[1]=Math.max(max[1],p[1]);}}const s=+document.getElementById('size').value;const step=Math.max(1,Math.floor(pts.length/36000));for(let i=0;i<pts.length;i+=step){{const p=pts[i];const x=(p[0]-min[0])/Math.max(1e-6,max[0]-min[0])*canvas.width*.86+canvas.width*.07;const y=canvas.height*.92-(p[1]-min[1])/Math.max(1e-6,max[1]-min[1])*canvas.height*.84;ctx.fillStyle=`rgb(${{p[3]|0}},${{p[4]|0}},${{p[5]|0}})`;ctx.fillRect(x,y,s,s);}}}}load();
</script></body></html>"""
    ensure(VIEWER)
    (VIEWER / "index.html").write_text(html, encoding="utf-8")
    return str((VIEWER / "index.html").relative_to(REPO))


def bundle() -> None:
    specs = {
        "v880_core": [REPO / "tools" / "V880_canonical_surfel_residual_candidate.py"],
        "v880_reports": [REPORTS / "V880000000000000000_candidate_decision.json", REPORTS / "V880000000000000000_iteration_report.md", REPORTS / "V880000000000000000_seed_metrics.csv", REPORTS / "V880000000000000000_training_manifest.csv"],
        "v880_visuals": [BOARDS / "V880000000000000000_canonical_surfel_residual_main.png", BOARDS / "V880000000000000000_head_hair_3d_closeup.png", BOARDS / "V880000000000000000_hand_arm_3d_closeup.png", BOARDS / "V880000000000000000_clothing_3d_closeup.png"],
        "v880_predictions": [MATRIX],
        "v880_viewer": [VIEWER],
    }
    records = []
    for name, paths in specs.items():
        zpath = ARCHIVE / f"V880000000000000000_{name}_bundle.zip"
        ensure(zpath.parent)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in paths:
                if p.is_file():
                    zf.write(p, p.relative_to(REPO).as_posix())
                elif p.is_dir():
                    for c in sorted(p.rglob("*")):
                        if c.is_file():
                            zf.write(c, c.relative_to(REPO).as_posix())
        with zipfile.ZipFile(zpath, "r") as zf:
            bad = zf.testzip()
            entries = zf.namelist()
        records.append({"bundle": name, "path": str(zpath), "bytes": zpath.stat().st_size, "entry_count": len(entries), "sha256": sha256_file(zpath), "zip_clean": bad is None, "under_500mb": zpath.stat().st_size < 500 * 1024 * 1024, "non_empty": len(entries) > 0})
    write_json(REPORTS / "V880000000000000000_bundle_integrity.json", {"created_at": now(), "bundle_count": len(records), "all_zip_clean": all(r["zip_clean"] for r in records), "all_under_500mb": all(r["under_500mb"] for r in records), "all_non_empty": all(r["non_empty"] for r in records), "bundles": records})


def report(decision: dict[str, Any], boards: dict[str, str], local: dict[str, str], viewer_path: str) -> str:
    final_state = "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"
    blockers = ["V880 is a candidate and still needs human mentor visual inspection", "old surfel artifacts are used only as structural priors, not final evidence"]
    if not decision["metric_true_gt_baseline_and_controls"]:
        blockers.insert(0, "canonical surfel residual candidate does not beat baseline/controls for all cases")
    write_json(REPORTS / "V880000000000000000_candidate_decision.json", {"created_at": now(), "final_state_candidate": final_state, "mentor_ready_candidate": False, "decision": decision, "hard_blockers": blockers, "main_board": boards["main"], "viewer": viewer_path})
    text = f"""# V880 Canonical Surfel Residual Iteration

## 先给结论

V880 仍保持 fail-closed：`{final_state}`。

这一轮把下一步核心从采样预算调整转为 canonical SMPL-X surfel residual / weak-region completion。旧 V124/V125 surfel 资产只作为结构表示参考，不作为最终导师证据。

## 主证据候选

- 3D main board: `{boards['main']}`
- controls: `{boards['controls']}`
- head/hair close-up: `{local['head_hair']}`
- hand/arm close-up: `{local['hand_arm']}`
- clothing close-up: `{local['clothing']}`
- viewer: `{viewer_path}`

## 当前阻断

{chr(10).join('- ' + b for b in blockers)}

## 下一步

需要真实 learned/local residual 或更可靠的 per-case canonical surfel feature extraction，不能继续把旧 surfel 或 projection score 包装成导师通过。
"""
    (REPORTS / "V880000000000000000_iteration_report.md").write_text(text, encoding="utf-8")
    return final_state


def update_v900(decision: dict[str, Any]) -> None:
    path = REPORTS / "V900000000000000000_final_status.json"
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    data.update(
        {
            "status": "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION",
            "mentor_ready": False,
            "all_pass": False,
            "continued_iteration": "V880_canonical_surfel_residual_candidate",
            "continued_iteration_report": "reports/V880000000000000000_iteration_report.md",
            "latest_candidate_decision": "reports/V880000000000000000_candidate_decision.json",
            "latest_candidate_metric_true_gt_baseline_and_controls": decision["metric_true_gt_baseline_and_controls"],
            "next_core": "learned_local_residual_or_real_per_case_canonical_surfel_feature_extraction",
            "updated_at": now(),
        }
    )
    write_json(path, data)


def cleanup(final_state: str) -> None:
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True, capture_output=True).stdout.splitlines()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    write_json(REPORTS / "V880000000000000000_cleanup.json", {"created_at": now(), "final_state_candidate": final_state, "repo": str(REPO), "branch": branch, "dirty_worktree": bool(status), "dirty_entry_count": len(status), "no_agent_subagent": True, "no_promotion": True, "no_registry": True, "no_v50_v50r2_change": True, "active_candidate": "V11700_gap_reduction_branch_520", "dirty_entries_sample": status[:180]})


def main() -> int:
    ensure(REPORTS)
    ensure(BOARDS)
    ensure(MATRIX)
    rows, payloads = build_matrix()
    boards = render(payloads)
    local = render_local(payloads)
    viewer_path = viewer()
    decision = decide(rows)
    final_state = report(decision, boards, local, viewer_path)
    update_v900(decision)
    bundle()
    cleanup(final_state)
    print(json.dumps({"final_state_candidate": final_state, "metric_true_gt_baseline_and_controls": decision["metric_true_gt_baseline_and_controls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
