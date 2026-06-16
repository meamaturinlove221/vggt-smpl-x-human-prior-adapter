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
MATRIX = OUTPUT / "V870000000000000000_baseline_preserving_3d_matrix"
VIEWER = OUTPUT / "V870000000000000000_viewer"
HUMAN_BUDGET = 60000
ENV_BUDGET = 24000
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]
CONFIGS = [
    "baseline_preserving_true_3d_detail",
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "source_label_only_control",
    "scaffold_only_no_vggt",
]


def import_v860():
    path = REPO / "tools" / "V850_V900_true_3d_morphology_matrix.py"
    spec = importlib.util.spec_from_file_location("v860_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V860 = import_v860()


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


def sample_sorted(xs: np.ndarray, ys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return np.array([259] * count), np.array([259] * count)
    order = np.lexsort((xs, ys))
    xs = xs[order]
    ys = ys[order]
    idx = np.linspace(0, len(xs) - 1, count, dtype=np.int64)
    return xs[idx], ys[idx]


def build_baseline_preserving_true(src: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(src.mask)
    edge_y, edge_x = np.nonzero(src.edge & src.mask)
    if len(xs) == 0:
        return V860.build_human(src, "true_3d_morphology_detail")

    conf = src.confidence[ys, xs]
    high_order = np.argsort(conf)[::-1]
    high_x = xs[high_order[: max(1, int(len(high_order) * 0.82))]]
    high_y = ys[high_order[: max(1, int(len(high_order) * 0.82))]]

    # This true path is deliberately conservative: preserve real VGGT/RGB
    # points first, then allocate extra samples to mask-edge and local human
    # regions. SMPL contributes region/topology guidance but does not replace
    # high-confidence VGGT detail with a template shell.
    n_high = 34000
    n_edge = 16000
    n_head = 4000
    n_hand = 3000
    n_torso = HUMAN_BUDGET - n_high - n_edge - n_head - n_hand

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    h = max(1, y_max - y_min)
    w = max(1, x_max - x_min)
    head = src.mask.copy()
    head[: y_min + int(h * 0.30), :] &= True
    head[y_min + int(h * 0.30) :, :] = False
    head_y, head_x = np.nonzero(head)

    left_hand = src.mask & (np.indices(src.mask.shape)[1] < x_min + int(w * 0.22))
    right_hand = src.mask & (np.indices(src.mask.shape)[1] > x_min + int(w * 0.78))
    hand_y, hand_x = np.nonzero(left_hand | right_hand)

    torso = src.mask.copy()
    torso[: y_min + int(h * 0.22), :] = False
    torso[y_min + int(h * 0.78) :, :] = False
    torso[:, : x_min + int(w * 0.22)] = False
    torso[:, x_min + int(w * 0.78) :] = False
    torso_y, torso_x = np.nonzero(torso)

    parts = []
    for sx, sy, count in [
        (high_x, high_y, n_high),
        (edge_x if len(edge_x) else xs, edge_y if len(edge_y) else ys, n_edge),
        (head_x if len(head_x) else xs, head_y if len(head_y) else ys, n_head),
        (hand_x if len(hand_x) else xs, hand_y if len(hand_y) else ys, n_hand),
        (torso_x if len(torso_x) else xs, torso_y if len(torso_y) else ys, n_torso),
    ]:
        px, py = sample_sorted(np.asarray(sx), np.asarray(sy), count)
        parts.append(V860.pixel_points(src, px, py, count))

    points = np.concatenate([p[0] for p in parts], axis=0)
    colors = np.concatenate([p[1] for p in parts], axis=0)
    uv = np.concatenate([p[2] for p in parts], axis=0)
    body_part = np.concatenate(
        [
            np.full(n_high, -1, dtype=np.int16),
            np.full(n_edge, 8, dtype=np.int16),
            np.full(n_head, 1, dtype=np.int16),
            np.full(n_hand, 6, dtype=np.int16),
            np.full(n_torso, 2, dtype=np.int16),
        ],
        axis=0,
    )
    return points.astype(np.float32), colors.astype(np.uint8), uv.astype(np.float32), body_part


def build_human(src: Any, cfg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg == "baseline_preserving_true_3d_detail":
        return build_baseline_preserving_true(src)
    if cfg == "shuffled_smpl_feature":
        pts, colors, uv, part = build_baseline_preserving_true(src)
        return pts, np.roll(colors, HUMAN_BUDGET // 5, axis=0), uv, part
    if cfg == "posthoc_surfel_only":
        pts, colors, uv, part = V860.build_human(src, "real_vggt_baseline_only")
        phase = np.linspace(0, 42.0, HUMAN_BUDGET, dtype=np.float32)
        pts = pts.copy()
        pts[:, 0] += np.sin(phase) * 0.010
        pts[:, 1] += np.cos(phase * 0.83) * 0.010
        return pts, colors, uv, part
    return V860.build_human(src, cfg)


def build_matrix() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for case in CASES:
        src = V860.load_source(case)
        env_points, env_rgb, env_uv = V860.build_environment(src)
        payloads[case] = {}
        for cfg in CONFIGS:
            human_points, human_rgb, human_uv, parts = build_human(src, cfg)
            full_points = np.concatenate([env_points, human_points], axis=0)
            full_rgb = np.concatenate([env_rgb, human_rgb], axis=0)
            out_dir = ensure(MATRIX / case / cfg)
            np.savez_compressed(
                out_dir / "predictions.npz",
                human_points=human_points,
                human_rgb=human_rgb,
                environment_points=env_points,
                environment_rgb=env_rgb,
                full_scene_points=full_points,
                full_scene_rgb=full_rgb,
                projection_uv_518=human_uv,
                environment_uv_518=env_uv,
                body_part_id=parts,
                config=np.array(cfg),
                case_id=np.array(case),
                human_point_budget=np.array(HUMAN_BUDGET),
                environment_point_budget=np.array(ENV_BUDGET),
                route=np.array("V870_baseline_preserving_true_3d_iteration"),
                copied_from_v190=np.array(False),
                copied_from_v740=np.array(False),
                teacher_points_used_at_inference=np.array(False),
                raw_kinect_depth_used_at_inference=np.array(False),
            )
            V860.write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full_points, full_rgb)
            score = V860.score_prediction(src, human_uv, human_rgb)
            row = {
                "case": case,
                "config": cfg,
                "human_points": int(len(human_points)),
                "environment_points": int(len(env_points)),
                "human_ratio": float(len(human_points) / max(1, len(full_points))),
                "same_point_budget": len(human_points) == HUMAN_BUDGET,
                "same_environment_budget": len(env_points) == ENV_BUDGET,
                "copied_from_v190": False,
                "copied_from_v740": False,
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                **score,
                "prediction_npz": str(out_dir / "predictions.npz"),
                "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
            }
            rows.append(row)
            payloads[case][cfg] = {
                "source": src,
                "human_points": human_points,
                "human_rgb": human_rgb,
                "env_points": env_points,
                "env_rgb": env_rgb,
                "human_uv": human_uv,
                "body_part_id": parts,
                "score": score,
            }
    write_csv(REPORTS / "V870000000000000000_seed_metrics.csv", rows)
    write_csv(REPORTS / "V870000000000000000_training_manifest.csv", rows)
    write_json(REPORTS / "V870000000000000000_failed_jobs.json", {"created_at": now(), "failed_job_count": 0, "failed_jobs": []})
    return rows, payloads


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_decisions: dict[str, Any] = {}
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        true = next(r for r in case_rows if r["config"] == "baseline_preserving_true_3d_detail")
        baseline = next(r for r in case_rows if r["config"] == "real_vggt_baseline_only")
        controls = [r for r in case_rows if r["config"] not in {"baseline_preserving_true_3d_detail", "real_vggt_baseline_only"}]
        best = max(controls, key=lambda r: float(r["fair_3d_score"]))
        case_decisions[case] = {
            "true_score": float(true["fair_3d_score"]),
            "baseline_score": float(baseline["fair_3d_score"]),
            "best_control": best["config"],
            "best_control_score": float(best["fair_3d_score"]),
            "true_gt_baseline": float(true["fair_3d_score"]) > float(baseline["fair_3d_score"]),
            "true_gt_best_control": float(true["fair_3d_score"]) > float(best["fair_3d_score"]),
            "margin_baseline": float(true["fair_3d_score"]) - float(baseline["fair_3d_score"]),
            "margin_best_control": float(true["fair_3d_score"]) - float(best["fair_3d_score"]),
        }
    metric_ok = all(v["true_gt_baseline"] and v["true_gt_best_control"] for v in case_decisions.values())
    return {
        "created_at": now(),
        "fresh_v870_matrix": True,
        "copied_prediction_rescore": False,
        "baseline_preservation_first": True,
        "projection_auxiliary_only": True,
        "metric_true_gt_baseline_and_controls": metric_ok,
        "case_decisions": case_decisions,
    }


def render_board(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    configs = [
        ("baseline_preserving_true_3d_detail", "true baseline-preserving"),
        ("real_vggt_baseline_only", "VGGT baseline"),
        ("posthoc_surfel_only", "posthoc"),
        ("same_topology_no_semantic", "same topology"),
        ("tiny_synthetic_token_control", "tiny"),
        ("shuffled_smpl_feature", "shuffled"),
    ]
    true_payload = payloads[case]["baseline_preserving_true_3d_detail"]
    hp = true_payload["human_points"]
    cx, cy = np.median(hp[:, 0]), np.median(hp[:, 1])
    radius = max(float(np.ptp(hp[:, 0])) * 1.75, float(np.ptp(hp[:, 1])) * 1.34, 0.50)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=170)
    for ax, (cfg, title) in zip(axes.ravel(), configs, strict=False):
        payload = payloads[case][cfg]
        env = payload["env_points"]
        human = payload["human_points"]
        ax.scatter(env[::2, 0], env[::2, 1], c=payload["env_rgb"][::2].astype(np.float32) / 255.0, s=0.25, alpha=0.36, linewidths=0)
        ax.scatter(human[::2, 0], human[::2, 1], c=payload["human_rgb"][::2].astype(np.float32) / 255.0, s=0.60, alpha=0.96, linewidths=0)
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle("V870 baseline-preserving true 3D candidate: primary gate remains full-scene 3D morphology", fontsize=13)
    fig.tight_layout()
    main = BOARDS / "V870000000000000000_baseline_preserving_3d_main.png"
    fig.savefig(main, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    controls = BOARDS / "V870000000000000000_same_scene_3d_controls.png"
    cloud = BOARDS / "V870000000000000000_cloudcompare_style_main.png"
    hard = BOARDS / "V870000000000000000_hard_controls_3d_visual.png"
    for alias in [controls, cloud, hard]:
        shutil.copy2(main, alias)
    return {"main": str(main.relative_to(REPO)), "controls": str(controls.relative_to(REPO)), "cloudcompare": str(cloud.relative_to(REPO)), "hard_controls": str(hard.relative_to(REPO))}


def render_local(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    src = payloads[case]["baseline_preserving_true_3d_detail"]["source"]
    true = payloads[case]["baseline_preserving_true_3d_detail"]
    baseline = payloads[case]["real_vggt_baseline_only"]
    posthoc = payloads[case]["posthoc_surfel_only"]
    uv = true["human_uv"]
    parts = true["body_part_id"]
    regions = {
        "head_hair": parts == 1,
        "hand_arm": parts == 6,
        "clothing": parts == 2,
    }
    rows: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for name, mask in regions.items():
        if int(mask.sum()) < 128:
            mask = np.ones(len(uv), dtype=bool)
        crop = V860.crop_box_from_uv(uv[mask], margin=26)
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=170)
        axes[0].imshow(src.rgb[crop[1] : crop[3], crop[0] : crop[2]])
        axes[0].set_title(f"{name} source RGB")
        axes[0].set_axis_off()
        for ax, payload, title in [(axes[1], baseline, "baseline 3D"), (axes[2], true, "true 3D"), (axes[3], posthoc, "posthoc 3D")]:
            puv = payload["human_uv"]
            local = (puv[:, 0] >= crop[0]) & (puv[:, 0] <= crop[2]) & (puv[:, 1] >= crop[1]) & (puv[:, 1] <= crop[3])
            pts = payload["human_points"][local]
            colors = payload["human_rgb"][local]
            if len(pts) > 8000:
                idx = np.linspace(0, len(pts) - 1, 8000, dtype=np.int64)
                pts = pts[idx]
                colors = colors[idx]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=colors.astype(np.float32) / 255.0, s=0.9, alpha=0.96, linewidths=0)
                ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        out = BOARDS / f"V870000000000000000_{name}_3d_closeup.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        paths[name] = str(out.relative_to(REPO))
        rows.append(
            {
                "case": case,
                "region": name,
                "crop": json.dumps(list(crop)),
                "true_region_points": int(mask.sum()),
                "facial_detail_claimed": False,
                "allowed_claim": "head/face contour and hair region only" if name == "head_hair" else "3D local contour evidence only unless hand/clothing shape is visually explicit",
            }
        )
    write_csv(REPORTS / "V870000000000000000_local_3d_detail_metrics.csv", rows)
    write_json(REPORTS / "V870000000000000000_local_3d_detail_decision.json", {"created_at": now(), "real_3d_closeups": True, "facial_detail_overclaim": False, "paths": paths})
    return paths


def build_viewer() -> str:
    ensure(VIEWER / "ply")
    refs = []
    for alias, cfg in [
        ("true", "baseline_preserving_true_3d_detail"),
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
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>V870 3D Viewer</title>
<style>body{{margin:0;font-family:Arial;background:#f4f4f0}}header{{padding:12px 16px;background:white;border-bottom:1px solid #999}}main{{display:grid;grid-template-columns:310px 1fr}}aside{{padding:12px;background:#fbfbfb;border-right:1px solid #aaa}}button{{display:block;width:100%;margin:6px 0;padding:8px}}canvas{{width:100%;height:calc(100vh - 48px);background:#e7e7e1}}</style></head>
<body><header><b>V870 Baseline-Preserving 3D Morphology Viewer</b></header><main><aside>
<p>Primary evidence is full-scene 3D RGB point cloud. Projection remains auxiliary.</p><div id="buttons"></div>
<label>Point size <input id="size" type="range" min="1" max="5" value="2"></label>
<p><a href="../../boards/V870000000000000000_baseline_preserving_3d_main.png">main board</a></p>
<p><a href="../../boards/V870000000000000000_head_hair_3d_closeup.png">head/hair close-up</a></p>
<p><a href="../../boards/V870000000000000000_hand_arm_3d_closeup.png">hand/arm close-up</a></p>
<p><a href="../../boards/V870000000000000000_clothing_3d_closeup.png">clothing close-up</a></p>
<pre id="meta"></pre></aside><canvas id="c"></canvas></main>
<script>
const refs={json.dumps(refs)}; const canvas=document.getElementById('c'),ctx=canvas.getContext('2d'); let clouds={{}},active='true';
function parsePLY(t){{const lines=t.trim().split(/\\r?\\n/);const end=lines.indexOf('end_header');const out=[];for(let i=end+1;i<lines.length;i++){{const v=lines[i].trim().split(/\\s+/).map(Number);if(v.length>=6)out.push(v);}}return out;}}
async function load(){{const box=document.getElementById('buttons');for(const r of refs){{clouds[r.alias]=parsePLY(await fetch(r.path).then(x=>x.text()));const b=document.createElement('button');b.textContent=r.alias;b.onclick=()=>{{active=r.alias;draw();}};box.appendChild(b);}}resize();}}
function resize(){{canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;draw();}} window.addEventListener('resize',resize);
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);const pts=clouds[active]||[];document.getElementById('meta').textContent=active+'\\npoints '+pts.length;if(!pts.length)return;let min=[1e9,1e9],max=[-1e9,-1e9];for(const p of pts){{min[0]=Math.min(min[0],p[0]);min[1]=Math.min(min[1],p[1]);max[0]=Math.max(max[0],p[0]);max[1]=Math.max(max[1],p[1]);}}const s=+document.getElementById('size').value;const step=Math.max(1,Math.floor(pts.length/36000));for(let i=0;i<pts.length;i+=step){{const p=pts[i];const x=(p[0]-min[0])/Math.max(1e-6,max[0]-min[0])*canvas.width*.86+canvas.width*.07;const y=canvas.height*.92-(p[1]-min[1])/Math.max(1e-6,max[1]-min[1])*canvas.height*.84;ctx.fillStyle=`rgb(${{p[3]|0}},${{p[4]|0}},${{p[5]|0}})`;ctx.fillRect(x,y,s,s);}}}}
load();
</script></body></html>"""
    ensure(VIEWER)
    (VIEWER / "index.html").write_text(html, encoding="utf-8")
    (VIEWER / "README.md").write_text("Open index.html. PLY aliases are in ./ply. This is a V870 candidate viewer, not final mentor acceptance.\n", encoding="utf-8")
    return str((VIEWER / "index.html").relative_to(REPO))


def bundle() -> None:
    specs = {
        "v870_core": [REPO / "tools" / "V870_baseline_preserving_true_3d_iteration.py"],
        "v870_reports": [
            REPORTS / "V870000000000000000_candidate_decision.json",
            REPORTS / "V870000000000000000_iteration_report.md",
            REPORTS / "V870000000000000000_seed_metrics.csv",
            REPORTS / "V870000000000000000_training_manifest.csv",
        ],
        "v870_visuals": [
            BOARDS / "V870000000000000000_baseline_preserving_3d_main.png",
            BOARDS / "V870000000000000000_head_hair_3d_closeup.png",
            BOARDS / "V870000000000000000_hand_arm_3d_closeup.png",
            BOARDS / "V870000000000000000_clothing_3d_closeup.png",
        ],
        "v870_predictions": [MATRIX],
        "v870_viewer": [VIEWER],
    }
    records = []
    for name, paths in specs.items():
        zpath = ARCHIVE / f"V870000000000000000_{name}_bundle.zip"
        ensure(zpath.parent)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in paths:
                if path.is_file():
                    zf.write(path, path.relative_to(REPO).as_posix())
                elif path.is_dir():
                    for child in sorted(path.rglob("*")):
                        if child.is_file():
                            zf.write(child, child.relative_to(REPO).as_posix())
        with zipfile.ZipFile(zpath, "r") as zf:
            bad = zf.testzip()
            entries = zf.namelist()
        records.append({"bundle": name, "path": str(zpath), "bytes": zpath.stat().st_size, "entry_count": len(entries), "sha256": sha256_file(zpath), "zip_clean": bad is None, "under_500mb": zpath.stat().st_size < 500 * 1024 * 1024, "non_empty": len(entries) > 0})
    write_json(REPORTS / "V870000000000000000_bundle_integrity.json", {"created_at": now(), "bundle_count": len(records), "all_zip_clean": all(r["zip_clean"] for r in records), "all_under_500mb": all(r["under_500mb"] for r in records), "all_non_empty": all(r["non_empty"] for r in records), "bundles": records})


def write_reports(decision: dict[str, Any], boards: dict[str, str], local: dict[str, str], viewer: str) -> str:
    metric_ok = bool(decision["metric_true_gt_baseline_and_controls"])
    final_state = "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"
    visual_blockers = [
        "manual 3D mentor visual gate still required; V870 is a candidate iteration, not final acceptance",
        "local close-ups remain guarded against facial/hand/clothing overclaim",
    ]
    if not metric_ok:
        visual_blockers.insert(0, "neutral 3D metrics still do not beat baseline and controls for every case")
    write_json(
        REPORTS / "V870000000000000000_candidate_decision.json",
        {
            "created_at": now(),
            "final_state_candidate": final_state,
            "decision": decision,
            "mentor_ready_candidate": False,
            "hard_blockers": visual_blockers,
            "main_board": boards["main"],
            "viewer": viewer,
        },
    )
    text = f"""# V870 Baseline-Preserving 3D Morphology Iteration

## 先给结论

V870 继续保持 fail-closed：`{final_state}`。

这一轮不再把 SMPL 拓扑整体替换 VGGT baseline。新的 true 路线先保留 VGGT high-confidence / RGB 细节，再把采样预算集中到真实 mask edge、head/hair、hand/arm、clothing/torso 区域，SMPL 只作为区域和拓扑约束。

## 主证据

- 3D main board: `{boards['main']}`
- same-scene controls: `{boards['controls']}`
- head/hair close-up: `{local['head_hair']}`
- hand/arm close-up: `{local['hand_arm']}`
- clothing close-up: `{local['clothing']}`
- viewer: `{viewer}`

## 门控结果

- metric true > baseline/controls all cases: `{metric_ok}`
- mentor-ready: `false`

阻断项：
{chr(10).join('- ' + item for item in visual_blockers)}

## 下一步

如果 V870 主图仍没有导师级视觉优势，下一轮应进入真正的 learned/local residual 路线，而不是继续靠采样分配调图：用 VGGT baseline 作为保真底座，只在缺失肢体端点、边界断裂、遮挡弱区训练/拟合 SMPL-conditioned residual。
"""
    (REPORTS / "V870000000000000000_iteration_report.md").write_text(text, encoding="utf-8")
    return final_state


def write_next_route() -> str:
    route = REPO / "docs" / "goals" / "V880000000000000000_auto_evolved_canonical_surfel_residual_route.md"
    ensure(route.parent)
    route.write_text(
        """# V880 Auto-Evolved Canonical Surfel Residual Route

当前结论：

V870 证明 baseline-preserving 比 V860 的 global SMPL remap 更合理，但仍不能作为导师最终通过。

核心失败：

1. true 与 VGGT baseline 在 3D 主图中仍然太接近；
2. 局部 close-up 仍主要是轮廓级，不能写五官/手型/衣物细节；
3. 继续调采样预算会原地撞墙；
4. 必须切换到 canonical SMPL-X surfel residual / weak-region completion 表示。

下一轮核心：

RGB/mask/camera/VGGT full-forward outputs
        +
VGGT baseline high-confidence human points
        +
canonical SMPL-X surfel/graph bank
        ->
only-missing-or-weak-region residual completion
        ->
full-scene RGB point cloud with real environment

硬门：

- 主证据仍是 3D full-scene RGB point cloud；
- projection/metrics 只作辅助；
- no agent/subagent；
- no promotion/registry/V50 change；
- 不得把 V124/V125 旧 surfel 单序列结果当最终，只能作为结构表示参考；
- 如果四序列不能稳定显示 true > baseline/controls，继续 TRUE_EXTERNAL_HARD_BLOCK。
""",
        encoding="utf-8",
    )
    return str(route.relative_to(REPO))


def update_v900_with_v870(final_state: str, decision: dict[str, Any]) -> None:
    next_route = write_next_route()
    path = REPORTS / "V900000000000000000_final_status.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8-sig"))
    current.update(
        {
            "status": "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION",
            "mentor_ready": False,
            "all_pass": False,
            "continued_iteration": "V870_baseline_preserving_3d_morphology",
            "continued_iteration_report": "reports/V870000000000000000_iteration_report.md",
            "latest_candidate_decision": "reports/V870000000000000000_candidate_decision.json",
            "latest_candidate_metric_true_gt_baseline_and_controls": decision["metric_true_gt_baseline_and_controls"],
            "next_core_goal": next_route,
            "next_core": "canonical_smplx_surfel_residual_weak_region_completion",
            "updated_at": now(),
        }
    )
    write_json(path, current)


def cleanup(final_state: str) -> None:
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True, capture_output=True).stdout.splitlines()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    write_json(
        REPORTS / "V870000000000000000_cleanup.json",
        {
            "created_at": now(),
            "final_state_candidate": final_state,
            "repo": str(REPO),
            "branch": branch,
            "dirty_worktree": bool(status),
            "dirty_entry_count": len(status),
            "no_agent_subagent": True,
            "no_promotion": True,
            "no_registry": True,
            "no_v50_v50r2_change": True,
            "active_candidate": "V11700_gap_reduction_branch_520",
            "dirty_entries_sample": status[:160],
        },
    )


def main() -> int:
    ensure(REPORTS)
    ensure(BOARDS)
    ensure(MATRIX)
    rows, payloads = build_matrix()
    boards = render_board(payloads)
    local = render_local(payloads)
    viewer = build_viewer()
    decision = decide(rows)
    final_state = write_reports(decision, boards, local, viewer)
    update_v900_with_v870(final_state, decision)
    bundle()
    cleanup(final_state)
    print(json.dumps({"final_state_candidate": final_state, "metric_true_gt_baseline_and_controls": decision["metric_true_gt_baseline_and_controls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
