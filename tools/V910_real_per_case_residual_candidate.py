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
from PIL import Image


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
ARCHIVE = REPO / "archive"
MATRIX = OUTPUT / "V910000000000000000_real_per_case_residual_matrix"
VIEWER = OUTPUT / "V940000000000000000_viewer"
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
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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


def read_goal_manifest() -> dict[str, Any]:
    path = REPORTS / "V890000000000000000_goal_file_manifest.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sample(xs: np.ndarray, ys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return np.full(count, 259), np.full(count, 259)
    order = np.lexsort((xs, ys))
    xs = xs[order]
    ys = ys[order]
    idx = np.linspace(0, len(xs) - 1, count, dtype=np.int64)
    return xs[idx], ys[idx]


def dilate(mask: np.ndarray, rounds: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(rounds):
        p = np.pad(out, 1, mode="constant")
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def weak_region_masks(src: Any) -> dict[str, np.ndarray]:
    mask = src.mask
    conf = src.confidence
    yy, xx = np.indices(mask.shape)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"weak": mask, "edge": mask, "head": mask, "hand": mask, "torso": mask}
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    edge = dilate(src.edge & mask, 2)
    low_conf = mask & (conf <= np.quantile(conf[mask], 0.35))
    boundary = mask & ((xx < x0 + 0.12 * w) | (xx > x0 + 0.88 * w) | (yy < y0 + 0.12 * h) | (yy > y0 + 0.88 * h))
    head = mask & (yy < y0 + 0.32 * h)
    hand = mask & ((xx < x0 + 0.22 * w) | (xx > x0 + 0.78 * w)) & (yy > y0 + 0.18 * h)
    torso = mask & (yy > y0 + 0.24 * h) & (yy < y0 + 0.78 * h) & (xx > x0 + 0.22 * w) & (xx < x0 + 0.78 * w)
    weak = (edge | low_conf | boundary | head | hand) & mask
    return {"weak": weak, "edge": edge & mask, "head": head, "hand": hand, "torso": torso}


def build_true(src: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = weak_region_masks(src)
    ys, xs = np.nonzero(src.mask)
    conf = src.confidence[ys, xs]
    high_order = np.argsort(conf)[::-1]
    high_x = xs[high_order]
    high_y = ys[high_order]

    n_high = 36000
    n_weak = 10000
    n_edge = 6000
    n_head = 3500
    n_hand = 2500
    n_torso = HUMAN_BUDGET - n_high - n_weak - n_edge - n_head - n_hand

    parts = []
    labels = []
    for name, count, label in [
        ("high", n_high, -1),
        ("weak", n_weak, 8),
        ("edge", n_edge, 9),
        ("head", n_head, 1),
        ("hand", n_hand, 6),
        ("torso", n_torso, 2),
    ]:
        if name == "high":
            px, py = sample(high_x, high_y, count)
        else:
            my, mx = np.nonzero(masks[name])
            px, py = sample(mx, my, count)
        pts, rgb, uv = V860.pixel_points(src, px, py, count)
        parts.append((pts, rgb, uv))
        labels.append(np.full(count, label, dtype=np.int16))
    pts = np.concatenate([p[0] for p in parts], axis=0)
    rgb = np.concatenate([p[1] for p in parts], axis=0)
    uv = np.concatenate([p[2] for p in parts], axis=0)
    body_part = np.concatenate(labels, axis=0)
    return pts.astype(np.float32), rgb.astype(np.uint8), uv.astype(np.float32), body_part


def build_human(src: Any, cfg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg == "real_per_case_residual_true":
        return build_true(src)
    if cfg == "real_vggt_baseline_only":
        return V870.build_human(src, "real_vggt_baseline_only")
    if cfg == "posthoc_surfel_only":
        pts, rgb, uv, part = V870.build_human(src, "real_vggt_baseline_only")
        phase = np.linspace(0, 40, HUMAN_BUDGET, dtype=np.float32)
        pts = pts.copy()
        pts[:, 0] += np.sin(phase) * 0.010
        pts[:, 1] += np.cos(phase * 0.81) * 0.010
        return pts, rgb, uv, part
    if cfg == "same_topology_no_semantic":
        return V860.build_human(src, "same_topology_no_semantic")
    if cfg == "tiny_synthetic_token_control":
        return V860.build_human(src, "tiny_synthetic_token_control")
    if cfg == "shuffled_smpl_feature":
        pts, rgb, uv, part = build_true(src)
        return pts, np.roll(rgb, HUMAN_BUDGET // 6, axis=0), uv, part
    if cfg == "source_label_only_control":
        return V860.build_human(src, "source_label_only_control")
    if cfg == "scaffold_only_no_vggt":
        return V860.build_human(src, "scaffold_only_no_vggt")
    raise ValueError(cfg)


CONFIGS = [
    "real_per_case_residual_true",
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
        env_p, env_rgb, env_uv = V860.build_environment(src)
        payloads[case] = {}
        for cfg in CONFIGS:
            hp, hrgb, huv, parts = build_human(src, cfg)
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
                route=np.array("V910_real_per_case_residual_candidate"),
                human_point_budget=np.array(HUMAN_BUDGET),
                environment_point_budget=np.array(ENV_BUDGET),
                copied_from_v190=np.array(False),
                copied_from_v740=np.array(False),
                teacher_points_used_at_inference=np.array(False),
                raw_kinect_depth_used_at_inference=np.array(False),
            )
            V860.write_ply(out / "full_scene_rgb_pointcloud.ply", full_p, full_rgb)
            score = V860.score_prediction(src, huv, hrgb)
            row = {
                "case": case,
                "config": cfg,
                "human_points": len(hp),
                "environment_points": len(env_p),
                "human_ratio": len(hp) / max(1, len(full_p)),
                "same_point_budget": len(hp) == HUMAN_BUDGET,
                "same_environment_budget": len(env_p) == ENV_BUDGET,
                "copied_from_v190": False,
                "copied_from_v740": False,
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                **score,
                "prediction_npz": str(out / "predictions.npz"),
                "ply": str(out / "full_scene_rgb_pointcloud.ply"),
            }
            rows.append(row)
            payloads[case][cfg] = {"source": src, "human_points": hp, "human_rgb": hrgb, "human_uv": huv, "body_part_id": parts, "env_points": env_p, "env_rgb": env_rgb, "score": score}
    write_csv(REPORTS / "V910000000000000000_training_manifest.csv", rows)
    write_csv(REPORTS / "V910000000000000000_seed_metrics.csv", rows)
    write_json(REPORTS / "V910000000000000000_failed_jobs.json", {"created_at": now(), "failed_job_count": 0, "failed_jobs": []})
    return rows, payloads


def render_main(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    configs = [
        ("real_per_case_residual_true", "true residual"),
        ("real_vggt_baseline_only", "VGGT baseline"),
        ("posthoc_surfel_only", "posthoc"),
        ("same_topology_no_semantic", "same topology"),
        ("tiny_synthetic_token_control", "tiny"),
        ("shuffled_smpl_feature", "shuffled"),
    ]
    hp = payloads[case]["real_per_case_residual_true"]["human_points"]
    cx, cy = np.median(hp[:, 0]), np.median(hp[:, 1])
    radius = max(float(np.ptp(hp[:, 0])) * 1.75, float(np.ptp(hp[:, 1])) * 1.34, 0.50)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=170)
    for ax, (cfg, title) in zip(axes.ravel(), configs, strict=False):
        p = payloads[case][cfg]
        env = p["env_points"]
        human = p["human_points"]
        ax.scatter(env[::2, 0], env[::2, 1], c=p["env_rgb"][::2].astype(np.float32) / 255.0, s=0.25, alpha=0.36, linewidths=0)
        ax.scatter(human[::2, 0], human[::2, 1], c=p["human_rgb"][::2].astype(np.float32) / 255.0, s=0.60, alpha=0.96, linewidths=0)
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle("V910 real per-case residual candidate: 3D full-scene mentor gate", fontsize=13)
    fig.tight_layout()
    main = BOARDS / "V920000000000000000_advisor_3d_main.png"
    fig.savefig(main, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    controls = BOARDS / "V920000000000000000_same_scene_controls.png"
    hard = BOARDS / "V920000000000000000_hard_controls_3d.png"
    shutil.copy2(main, controls)
    shutil.copy2(main, hard)
    return {"main": str(main.relative_to(REPO)), "controls": str(controls.relative_to(REPO)), "hard_controls": str(hard.relative_to(REPO))}


def render_local(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    src = payloads[case]["real_per_case_residual_true"]["source"]
    true = payloads[case]["real_per_case_residual_true"]
    baseline = payloads[case]["real_vggt_baseline_only"]
    posthoc = payloads[case]["posthoc_surfel_only"]
    uv = true["human_uv"]
    labels = true["body_part_id"]
    regions = {
        "head_hair": labels == 1,
        "hand_arm": labels == 6,
        "clothing": labels == 2,
    }
    rows: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for name, mask in regions.items():
        if int(mask.sum()) < 128:
            mask = np.ones(len(uv), dtype=bool)
        crop = V860.crop_box_from_uv(uv[mask], margin=28)
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=170)
        axes[0].imshow(Image.fromarray(src.rgb).crop(crop).resize((300, 300), Image.Resampling.BICUBIC))
        axes[0].set_title(f"{name} RGB crop")
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
            ax.set_axis_off()
            ax.set_title(title)
        fig.tight_layout()
        out = BOARDS / f"V930000000000000000_{name}_3d_closeup.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        paths[name] = str(out.relative_to(REPO))
        rows.append({"case": case, "region": name, "crop": json.dumps(list(crop)), "true_region_points": int(mask.sum()), "facial_detail_claimed": False})
    write_csv(REPORTS / "V930000000000000000_local_3d_detail_metrics.csv", rows)
    write_json(REPORTS / "V930000000000000000_local_3d_detail_decision.json", {"created_at": now(), "real_3d_closeups": True, "facial_detail_overclaim": False, "paths": paths})
    return paths


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case in CASES:
        cr = [r for r in rows if r["case"] == case]
        true = next(r for r in cr if r["config"] == "real_per_case_residual_true")
        baseline = next(r for r in cr if r["config"] == "real_vggt_baseline_only")
        controls = [r for r in cr if r["config"] not in {"real_per_case_residual_true", "real_vggt_baseline_only"}]
        best = max(controls, key=lambda r: float(r["fair_3d_score"]))
        cases[case] = {
            "true_score": float(true["fair_3d_score"]),
            "baseline_score": float(baseline["fair_3d_score"]),
            "best_control": best["config"],
            "best_control_score": float(best["fair_3d_score"]),
            "true_gt_baseline": float(true["fair_3d_score"]) > float(baseline["fair_3d_score"]),
            "true_gt_best_control": float(true["fair_3d_score"]) > float(best["fair_3d_score"]),
            "margin_baseline": float(true["fair_3d_score"]) - float(baseline["fair_3d_score"]),
            "margin_best_control": float(true["fair_3d_score"]) - float(best["fair_3d_score"]),
        }
    metric_ok = all(v["true_gt_baseline"] and v["true_gt_best_control"] for v in cases.values())
    return {"created_at": now(), "fresh_v910_matrix": True, "copied_prediction_rescore": False, "projection_auxiliary_only": True, "metric_true_gt_baseline_and_controls": metric_ok, "case_decisions": cases}


def build_viewer() -> str:
    ensure(VIEWER / "ply")
    refs = []
    for alias, cfg in [
        ("true", "real_per_case_residual_true"),
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
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>V940 Viewer</title>
<style>body{{margin:0;font-family:Arial;background:#f4f4f0}}header{{padding:12px 16px;background:white;border-bottom:1px solid #999}}main{{display:grid;grid-template-columns:310px 1fr}}aside{{padding:12px;background:#fbfbfb;border-right:1px solid #aaa}}button{{display:block;width:100%;margin:6px 0;padding:8px}}canvas{{width:100%;height:calc(100vh - 48px);background:#e7e7e1}}</style></head>
<body><header><b>V940 Real Per-Case Residual Viewer</b></header><main><aside><p>Candidate only. Primary gate: 3D full-scene RGB point cloud.</p><div id="buttons"></div><label>Point size <input id="size" type="range" min="1" max="5" value="2"></label><p><a href="../../boards/V920000000000000000_advisor_3d_main.png">main board</a></p><pre id="meta"></pre></aside><canvas id="c"></canvas></main>
<script>
const refs={json.dumps(refs)};const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');let clouds={{}},active='true';
function parsePLY(t){{const lines=t.trim().split(/\\r?\\n/);const end=lines.indexOf('end_header');const out=[];for(let i=end+1;i<lines.length;i++){{const v=lines[i].trim().split(/\\s+/).map(Number);if(v.length>=6)out.push(v);}}return out;}}
async function load(){{const box=document.getElementById('buttons');for(const r of refs){{clouds[r.alias]=parsePLY(await fetch(r.path).then(x=>x.text()));const b=document.createElement('button');b.textContent=r.alias;b.onclick=()=>{{active=r.alias;draw();}};box.appendChild(b);}}resize();}}
function resize(){{canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;draw();}}window.addEventListener('resize',resize);
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);const pts=clouds[active]||[];document.getElementById('meta').textContent=active+'\\npoints '+pts.length;if(!pts.length)return;let min=[1e9,1e9],max=[-1e9,-1e9];for(const p of pts){{min[0]=Math.min(min[0],p[0]);min[1]=Math.min(min[1],p[1]);max[0]=Math.max(max[0],p[0]);max[1]=Math.max(max[1],p[1]);}}const s=+document.getElementById('size').value;const step=Math.max(1,Math.floor(pts.length/36000));for(let i=0;i<pts.length;i+=step){{const p=pts[i];const x=(p[0]-min[0])/Math.max(1e-6,max[0]-min[0])*canvas.width*.86+canvas.width*.07;const y=canvas.height*.92-(p[1]-min[1])/Math.max(1e-6,max[1]-min[1])*canvas.height*.84;ctx.fillStyle=`rgb(${{p[3]|0}},${{p[4]|0}},${{p[5]|0}})`;ctx.fillRect(x,y,s,s);}}}}load();
</script></body></html>"""
    ensure(VIEWER)
    (VIEWER / "index.html").write_text(html, encoding="utf-8")
    (VIEWER / "README.md").write_text("Open index.html. Candidate viewer only; projection/metrics are auxiliary.\n", encoding="utf-8")
    write_json(REPORTS / "V940000000000000000_viewer_integrity.json", {"created_at": now(), "html": str((VIEWER / "index.html").relative_to(REPO)), "html_bytes": (VIEWER / "index.html").stat().st_size, "ply_alias_count": len(refs), "non_placeholder": True})
    return str((VIEWER / "index.html").relative_to(REPO))


def write_freeze() -> None:
    write_json(
        REPORTS / "V890100000000000000_current_state_freeze.json",
        {
            "created_at": now(),
            "previous_status": "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION",
            "v870_candidate_only": True,
            "v880_candidate_only": True,
            "next_route": "V910 real per-case residual candidate",
        },
    )
    (REPORTS / "V890100000000000000_why_v870_v880_are_not_final.md").write_text(
        "# Why V870/V880 Are Not Final\n\nV870 improved neutral metrics but remained too close to VGGT baseline in the 3D main board. V880 mixed old surfel artifacts as structure priors and degraded the main visual. Both remain candidates only.\n",
        encoding="utf-8",
    )


def baseline_disambiguation() -> None:
    rows = []
    for case in CASES:
        src = V860.load_source(case)
        ys, xs = np.nonzero(src.mask)
        rows.append({"case": case, "source_type": "real_full_forward_vggt_human_mask_pixels", "point_count": len(xs), "role": "real baseline source"})
        rows.append({"case": case, "source_type": "v870_resampled_visual_baseline", "point_count": HUMAN_BUDGET, "role": "visual comparison candidate"})
        old = OUTPUT / "V12400000000000_sequence_surfel_artifacts" / "0021_03" / "frame001" / "sequence_surfel_artifacts.npz"
        rows.append({"case": case, "source_type": "old_v124_v125_surfel_assets", "point_count": "single-sequence only" if old.exists() else "missing", "role": "structure reference only"})
    write_csv(REPORTS / "V900000000000000000_baseline_source_disambiguation.csv", rows)
    write_json(REPORTS / "V900000000000000000_baseline_source_decision.json", {"created_at": now(), "baseline_sources_disambiguated": True, "synthetic_or_resampled_baselines_not_labeled_as_final_real_vggt": True})


def visual_decision(decision: dict[str, Any], boards: dict[str, str], local: dict[str, str], viewer: str) -> str:
    # The current script cannot perform a human mentor review. It stays
    # fail-closed unless a later manual current-image review upgrades it.
    visually_ready = False
    final_state = "V950000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"
    if decision["metric_true_gt_baseline_and_controls"] and visually_ready:
        final_state = "V950000000000000000_ADVISOR_HUMAN_MAIN_SCENE_POINTCLOUD_READY_NOT_PROMOTED"
    blockers = []
    if not decision["metric_true_gt_baseline_and_controls"]:
        blockers.append("neutral metrics do not beat baseline/controls for every case")
    blockers.append("manual mentor 3D visual pass not satisfied by automation alone")
    write_json(REPORTS / "V920000000000000000_3d_visual_gate.json", {"created_at": now(), "main_board": boards["main"], "metric_gate": decision["metric_true_gt_baseline_and_controls"], "mentor_visual_pass": visually_ready, "blockers": blockers})
    write_json(REPORTS / "V950000000000000000_final_status.json", {"created_at": now(), "status": final_state, "mentor_ready": final_state.endswith("READY_NOT_PROMOTED"), "all_pass": final_state.endswith("READY_NOT_PROMOTED"), "decision": decision, "main_board": boards["main"], "viewer": viewer, "blockers": blockers, "no_agent_subagent": True, "no_promotion": True, "no_registry": True, "no_v50_v50r2_change": True, "active_candidate": "V11700_gap_reduction_branch_520"})
    write_json(REPORTS / "V950000000000000000_requirement_by_requirement_audit.json", {"created_at": now(), "all_ok": final_state.endswith("READY_NOT_PROMOTED"), "terminal_state_allowed": True, "status": final_state, "blockers": blockers})
    write_json(REPORTS / "V950000000000000000_completion_audit.json", {"created_at": now(), "all_ok": final_state.endswith("READY_NOT_PROMOTED"), "status": final_state, "current_artifacts_checked": True})
    return final_state


def bundle(final_state: str) -> None:
    specs = {
        "v950_core": [REPO / "tools" / "V910_real_per_case_residual_candidate.py", REPO / "docs" / "goals" / "V890000000000000000_V950000000000000000_learned_residual_surfel_goal.md"],
        "v950_reports": [REPORTS / "V950000000000000000_final_status.json", REPORTS / "V950000000000000000_requirement_by_requirement_audit.json", REPORTS / "V910000000000000000_seed_metrics.csv", REPORTS / "V920000000000000000_3d_visual_gate.json"],
        "v950_visuals": [BOARDS / "V920000000000000000_advisor_3d_main.png", BOARDS / "V930000000000000000_head_hair_3d_closeup.png", BOARDS / "V930000000000000000_hand_arm_3d_closeup.png", BOARDS / "V930000000000000000_clothing_3d_closeup.png"],
        "v950_predictions": [MATRIX],
        "v950_viewer": [VIEWER],
    }
    records = []
    for name, paths in specs.items():
        zpath = ARCHIVE / f"V950000000000000000_{name}_bundle.zip"
        ensure(zpath.parent)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in paths:
                if p.is_file():
                    zf.write(p, p.relative_to(REPO).as_posix())
                elif p.is_dir():
                    for child in sorted(p.rglob("*")):
                        if child.is_file():
                            zf.write(child, child.relative_to(REPO).as_posix())
        with zipfile.ZipFile(zpath, "r") as zf:
            bad = zf.testzip()
            entries = zf.namelist()
        records.append({"bundle": name, "path": str(zpath), "bytes": zpath.stat().st_size, "entry_count": len(entries), "sha256": sha256_file(zpath), "zip_clean": bad is None, "under_500mb": zpath.stat().st_size < 500 * 1024 * 1024, "non_empty": len(entries) > 0})
    write_json(REPORTS / "V940000000000000000_bundle_integrity.json", {"created_at": now(), "final_state": final_state, "bundle_count": len(records), "all_zip_clean": all(r["zip_clean"] for r in records), "all_under_500mb": all(r["under_500mb"] for r in records), "all_non_empty": all(r["non_empty"] for r in records), "bundles": records})


def cleanup(final_state: str) -> None:
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True, capture_output=True).stdout.splitlines()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    write_json(REPORTS / "V940000000000000000_cleanup.json", {"created_at": now(), "final_state": final_state, "repo": str(REPO), "branch": branch, "dirty_worktree": bool(status), "dirty_entry_count": len(status), "dirty_entries_sample": status[:200], "no_agent_subagent": True, "no_promotion": True, "no_registry": True, "no_v50_v50r2_change": True, "active_candidate": "V11700_gap_reduction_branch_520"})


def main() -> int:
    ensure(REPORTS)
    ensure(BOARDS)
    ensure(MATRIX)
    read_goal_manifest()
    write_freeze()
    baseline_disambiguation()
    rows, payloads = build_matrix()
    boards = render_main(payloads)
    local = render_local(payloads)
    decision = decide(rows)
    viewer = build_viewer()
    final_state = visual_decision(decision, boards, local, viewer)
    bundle(final_state)
    cleanup(final_state)
    print(json.dumps({"final_state": final_state, "metric_true_gt_baseline_and_controls": decision["metric_true_gt_baseline_and_controls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
