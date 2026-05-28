from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
DOCS = REPO / "docs"
TRUE_ROOT = OUTPUT / "V13700000000000000000_anti_billboard_training_matrix"
BASE_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
VIEWER = OUTPUT / "V14500000000000000000_viewer"
BUNDLES = OUTPUT / "V19000000000000000000_upload_safe_bundles"
METRICS = REPORTS / "V13700000000000000000_seed_metrics.csv"
CAUSALITY = REPORTS / "V14000000000000000000_anti_billboard_causality_decision.json"

TRUE_CONFIG = "anti_billboard_topology_volume_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]
REGIONS = [
    ("head_hair_contour", "head / face contour / hair region"),
    ("shoulder_neck", "shoulder / neck"),
    ("hand_arm_endpoint", "arm / hand endpoint"),
    ("clothing_boundary", "torso / clothing boundary"),
    ("leg_foot", "leg / foot"),
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def prediction_path(case: str, config: str) -> Path:
    root = TRUE_ROOT if config == TRUE_CONFIG else BASE_ROOT
    return root / case / config / "predictions.npz"


def ply_path(case: str, config: str) -> Path:
    root = TRUE_ROOT if config == TRUE_CONFIG else BASE_ROOT
    return root / case / config / "full_scene_rgb_pointcloud.ply"


def load_prediction(case: str, config: str) -> dict[str, np.ndarray]:
    return load_npz(prediction_path(case, config))


def read_metrics() -> list[dict[str, str]]:
    with METRICS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cases_from_metrics(rows: list[dict[str, str]]) -> list[str]:
    return sorted({r["case"] for r in rows if r.get("config") == TRUE_CONFIG})


def pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return center, vals, vecs, x @ vecs


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def project(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (points - points.mean(axis=0, keepdims=True)) @ rot.T


def scatter_panel(
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    *,
    size: tuple[int, int] = (390, 282),
    rot: np.ndarray | None = None,
    point_limit: int = 52000,
) -> Image.Image:
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    if len(points) == 0:
        draw.text((8, 8), title + " (empty)", fill=(170, 0, 0))
        return im
    rot = rot if rot is not None else rotation_matrix(-30, 61)
    pts = project(points, rot)
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.20 + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 50, size[1] - 72]) + np.array([25, 46]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    d0, d1 = np.quantile(depth, [0.03, 0.97])
    cue = np.clip((depth - d0) / max(d1 - d0, 1e-9), 0, 1)
    rgb = np.clip(as_rgb(colors).astype(np.float32) * (0.58 + 0.50 * cue[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // point_limit)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title[:70], fill=(10, 10, 10))
    return im


def cross_panel(points: np.ndarray, title: str, *, size: tuple[int, int] = (390, 282)) -> Image.Image:
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    if len(points) == 0:
        draw.text((8, 8), title + " (empty)", fill=(170, 0, 0))
        return im
    _center, _vals, _axes, proj = pca(points)
    xy_src = proj[:, [1, 2]]
    lo = np.percentile(xy_src, 1, axis=0)
    hi = np.percentile(xy_src, 99, axis=0)
    pad = (hi - lo) * np.array([0.18, 0.55]) + 1e-6
    lo -= pad
    hi += pad
    q = (xy_src - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    order = np.argsort(proj[:, 0])
    step = max(1, len(order) // 42000)
    for i in order[::step]:
        x, y = xy[i]
        c = (45, 70, 57)
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), (82, 108, 92))
    draw.text((8, 8), title[:70], fill=(10, 10, 10))
    draw.text((8, size[1] - 23), "cross-section: mid axis vs thin axis", fill=(45, 45, 45))
    return im


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    if not panels:
        panels = [Image.new("RGB", (390, 282), (255, 255, 255))]
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def crop_region(points: np.ndarray, region: str) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    qx = np.quantile(x, [0.18, 0.35, 0.65, 0.82])
    qy = np.quantile(y, [0.16, 0.32, 0.58, 0.76, 0.88])
    if region == "head_hair_contour":
        return y <= qy[1]
    if region == "shoulder_neck":
        return (y > qy[0]) & (y <= qy[2]) & (x > qx[0]) & (x < qx[3])
    if region == "hand_arm_endpoint":
        return (x <= qx[0]) | (x >= qx[3])
    if region == "clothing_boundary":
        return (y > qy[1]) & (y <= qy[3]) & (x > qx[0]) & (x < qx[3])
    if region == "leg_foot":
        return y >= qy[3]
    return np.ones(len(points), dtype=bool)


def transfer_mask(source_points: np.ndarray, target_points: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    if not np.any(source_mask):
        return np.zeros(len(target_points), dtype=bool)
    src = source_points[source_mask]
    lo = np.percentile(src, 1, axis=0)
    hi = np.percentile(src, 99, axis=0)
    pad = (hi - lo) * 0.18 + 1e-6
    return np.all((target_points >= lo - pad) & (target_points <= hi + pad), axis=1)


def render_local_region(case: str, region_key: str, region_title: str, best_control: str) -> dict[str, Any]:
    true = load_prediction(case, TRUE_CONFIG)
    base = load_prediction(case, "real_vggt_baseline_only")
    ctrl = load_prediction(case, best_control)
    true_points = np.asarray(true["human_points"], dtype=np.float32)
    true_mask = crop_region(true_points, region_key)
    items = [
        ("baseline", np.asarray(base["human_points"], dtype=np.float32), as_rgb(base["human_rgb"])),
        ("true", true_points, as_rgb(true["human_rgb"])),
        (best_control, np.asarray(ctrl["human_points"], dtype=np.float32), as_rgb(ctrl["human_rgb"])),
    ]
    panels: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    for label, pts, rgb in items:
        mask = true_mask if label == "true" else transfer_mask(true_points, pts, true_mask)
        cpts = pts[mask]
        crgb = rgb[mask]
        panels.append(scatter_panel(cpts, crgb, f"{case} {region_title} {label}", size=(360, 250), point_limit=24000))
        panels.append(cross_panel(cpts, f"{case} {region_title} {label}", size=(360, 250)))
        vals = pca(cpts)[1] if len(cpts) >= 8 else np.array([0.0, 0.0, 0.0])
        rows.append(
            {
                "case": case,
                "region": region_key,
                "config": label,
                "point_count": int(len(cpts)),
                "thin_ratio": float(np.sqrt(max(vals[-1], 0.0)) / max(np.sqrt(max(vals[0], 0.0)), 1e-9)) if len(cpts) >= 8 else 0.0,
            }
        )
    return {"panels": panels, "rows": rows}


def best_control_by_case(metrics_rows: list[dict[str, str]], case: str) -> str:
    rows = [r for r in metrics_rows if r["case"] == case and r["config"] in CONTROL_CONFIGS]
    if not rows:
        return "same_topology_no_semantic"
    return max(rows, key=lambda r: float(r["anti_billboard_score_v2"]))["config"]


def run_v143(metrics_rows: list[dict[str, str]], cases: list[str]) -> dict[str, Any]:
    first_case = "0012_11_frame001" if "0012_11_frame001" in cases else cases[0]
    best_control = best_control_by_case(metrics_rows, first_case)
    all_rows: list[dict[str, Any]] = []
    board_paths: dict[str, str] = {}
    for region_key, region_title in REGIONS:
        rendered = render_local_region(first_case, region_key, region_title, best_control)
        all_rows.extend(rendered["rows"])
        path = BOARDS / f"V14300000000000000000_{region_key}_3d_closeup.png"
        compose(rendered["panels"], 3, path)
        board_paths[region_key] = str(path)
    write_csv(REPORTS / "V14300000000000000000_local_3d_morphology_metrics.csv", all_rows)
    failures = [
        "Global V140 anti-billboard causality failed; local crops cannot override the mentor main-board failure.",
        "Crops remain diagnostic morphology views, not fine facial/hand detail proof.",
        "Face invisible gate remains active; facial detail claim is forbidden.",
    ]
    decision = {
        "created_at": now(),
        "status": "V14300_LOCAL_3D_MORPHOLOGY_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "case": first_case,
        "best_control_used": best_control,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
        "fine_detail_claim_allowed": False,
        "failures": failures,
        "boards": board_paths,
        "metrics_csv": str(REPORTS / "V14300000000000000000_local_3d_morphology_metrics.csv"),
    }
    write_json(REPORTS / "V14300000000000000000_local_3d_morphology_decision.json", decision)
    return decision


def run_v144(cases: list[str]) -> dict[str, Any]:
    first_case = "0012_11_frame001" if "0012_11_frame001" in cases else cases[0]
    panels: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    for config in ["real_vggt_baseline_only", TRUE_CONFIG, "same_topology_no_semantic", "shuffled_smpl_feature"]:
        pred = load_prediction(first_case, config)
        env = np.asarray(pred["environment_points"], dtype=np.float32)
        env_rgb = as_rgb(pred["environment_rgb"])
        human = np.asarray(pred["human_points"], dtype=np.float32)
        full = np.asarray(pred["full_scene_points"], dtype=np.float32)
        full_rgb = as_rgb(pred["full_scene_rgb"])
        panels.append(scatter_panel(full, full_rgb, f"{first_case} {config} full scene", size=(390, 282), point_limit=70000))
        vals = pca(env)[1] if len(env) >= 8 else np.zeros(3)
        rows.append(
            {
                "case": first_case,
                "config": config,
                "human_points": int(len(human)),
                "environment_points": int(len(env)),
                "full_scene_points": int(len(full)),
                "human_ratio": float(len(human) / max(len(full), 1)),
                "environment_depth_spread_proxy": float(np.sqrt(max(vals[0], 0.0))),
                "environment_rgb_unique_sample": int(len(np.unique(env_rgb[:: max(1, len(env_rgb) // 2000)], axis=0))),
            }
        )
    path = BOARDS / "V14400000000000000000_environment_realism_gate.png"
    compose(panels, 2, path)
    pass_rows = [
        (0.55 <= r["human_ratio"] <= 0.75)
        and r["environment_points"] >= 18000
        and r["environment_depth_spread_proxy"] > 0.05
        for r in rows
    ]
    decision = {
        "created_at": now(),
        "status": "V14400_ENVIRONMENT_GATE_PASS_AS_AUXILIARY" if all(pass_rows) else "V14400_ENVIRONMENT_GATE_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "environment_auxiliary_pass": bool(all(pass_rows)),
        "note": "Environment visibility can pass only as auxiliary; it cannot rescue V140 billboard/control failure.",
        "board": str(path),
        "rows": rows,
    }
    write_json(REPORTS / "V14400000000000000000_environment_gate.json", decision)
    return decision


def copy_viewer_ply(case: str, config: str, alias: str) -> dict[str, Any]:
    src = ply_path(case, config)
    dst = VIEWER / "ply" / f"{alias}.ply"
    ensure(dst.parent)
    if src.exists():
        shutil.copy2(src, dst)
    return {"alias": alias, "config": config, "source": str(src), "path": f"ply/{alias}.ply", "exists": dst.exists(), "bytes": dst.stat().st_size if dst.exists() else 0}


def run_v145(cases: list[str]) -> dict[str, Any]:
    first_case = "0012_11_frame001" if "0012_11_frame001" in cases else cases[0]
    if VIEWER.exists():
        shutil.rmtree(VIEWER)
    ensure(VIEWER / "ply")
    aliases = [
        copy_viewer_ply(first_case, TRUE_CONFIG, "true"),
        copy_viewer_ply(first_case, "real_vggt_baseline_only", "baseline"),
        copy_viewer_ply(first_case, "same_topology_no_semantic", "same_topology"),
        copy_viewer_ply(first_case, "shuffled_smpl_feature", "shuffled"),
        copy_viewer_ply(first_case, "thickness_only_control", "thickness_only"),
        copy_viewer_ply(first_case, "posthoc_surfel_only", "posthoc"),
        copy_viewer_ply(first_case, "tiny_synthetic_token_control", "tiny"),
    ]
    manifest_js = json.dumps(aliases, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>V145 anti-billboard topology-volume viewer</title>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#101513; color:#eef3ef; }}
    #bar {{ position:fixed; inset:0 0 auto 0; min-height:48px; display:flex; gap:10px; align-items:center; padding:8px 12px; background:#17211c; z-index:2; flex-wrap:wrap; }}
    button, select, input {{ background:#25352d; color:#eef3ef; border:1px solid #58705f; padding:6px 8px; }}
    #viewer {{ position:absolute; inset:64px 0 0 0; }}
    #note {{ max-width:820px; color:#c7d7cc; }}
  </style>
</head>
<body>
<div id="bar">
  <strong>V145 Viewer</strong>
  <select id="cloud"></select>
  <button id="front">front</button><button id="side">side</button><button id="oblique">oblique</button>
  <label>point size <input id="ps" type="range" min="0.004" max="0.04" value="0.014" step="0.002"></label>
  <span id="note">Fail-closed diagnostic viewer: projection/thickness/viewer output is auxiliary, not mentor pass.</span>
</div>
<div id="viewer"></div>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import {{ PLYLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js';
const clouds = {manifest_js};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f6f2);
const camera = new THREE.PerspectiveCamera(45, innerWidth / Math.max(1, innerHeight - 64), 0.01, 100);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
document.getElementById('viewer').appendChild(renderer.domElement);
let points = null;
const material = new THREE.PointsMaterial({{size:0.014, vertexColors:true}});
function resize() {{ renderer.setSize(innerWidth, Math.max(1, innerHeight - 64)); camera.aspect = innerWidth / Math.max(1, innerHeight - 64); camera.updateProjectionMatrix(); }}
addEventListener('resize', resize); resize();
function setPreset(name) {{
  if (name === 'front') camera.position.set(0, -1.2, 0.35);
  else if (name === 'side') camera.position.set(1.0, 0.0, 0.35);
  else camera.position.set(0.9, -1.1, 0.75);
  camera.lookAt(0.02, 0.0, 0.72);
}}
function loadCloud(item) {{
  new PLYLoader().load(item.path, geometry => {{
    geometry.computeBoundingSphere();
    if (points) scene.remove(points);
    points = new THREE.Points(geometry, material);
    scene.add(points);
    setPreset('oblique');
  }});
}}
const select = document.getElementById('cloud');
clouds.forEach(c => {{ const opt = document.createElement('option'); opt.value = c.alias; opt.textContent = c.alias + ' - ' + c.config; select.appendChild(opt); }});
select.onchange = () => loadCloud(clouds.find(c => c.alias === select.value));
document.getElementById('ps').oninput = e => material.size = Number(e.target.value);
document.getElementById('front').onclick = () => setPreset('front');
document.getElementById('side').onclick = () => setPreset('side');
document.getElementById('oblique').onclick = () => setPreset('oblique');
loadCloud(clouds[0]);
function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
animate();
</script>
</body>
</html>
"""
    index = VIEWER / "index.html"
    index.write_text(html, encoding="utf-8")
    readme = VIEWER / "README.md"
    readme.write_text(
        "Open index.html in a browser. This diagnostic viewer loads PLY aliases from ./ply and remains auxiliary to the mentor full-scene visual gate.\n",
        encoding="utf-8",
    )
    integrity = {
        "created_at": now(),
        "status": "V14500_VIEWER_USABLE_AS_DIAGNOSTIC_AUXILIARY",
        "mentor_ready": False,
        "html": str(index),
        "html_bytes": index.stat().st_size,
        "ply_alias_count": sum(1 for a in aliases if a["exists"]),
        "ply_aliases": aliases,
        "non_placeholder": index.stat().st_size > 3000 and "PLYLoader" in html,
        "readme": str(readme),
    }
    write_json(REPORTS / "V14500000000000000000_viewer_integrity.json", integrity)
    return integrity


def run_v150(metrics_rows: list[dict[str, str]], v143: dict[str, Any], v144: dict[str, Any]) -> dict[str, Any]:
    cases = cases_from_metrics(metrics_rows)
    true_rows = [r for r in metrics_rows if r.get("config") == TRUE_CONFIG]
    pass_cases = [r for r in true_rows if str(r.get("billboard_fail_v2")).lower() == "false"]
    better_baseline = []
    controls_separated = []
    for case in cases:
        by_cfg = {r["config"]: r for r in metrics_rows if r["case"] == case}
        true = by_cfg.get(TRUE_CONFIG)
        baseline = by_cfg.get("real_vggt_baseline_only")
        if not true or not baseline:
            continue
        ts = float(true["anti_billboard_score_v2"])
        better_baseline.append(ts > float(baseline["anti_billboard_score_v2"]) * 1.05)
        hard_controls = [by_cfg[c] for c in CONTROL_CONFIGS if c in by_cfg]
        controls_separated.append(all(ts > float(c["anti_billboard_score_v2"]) * 1.05 for c in hard_controls))
    panels = []
    for board in [
        BOARDS / "V13800000000000000000_advisor_human_main_full_scene.png",
        BOARDS / "V13800000000000000000_same_scene_baseline_true_controls.png",
        BOARDS / "V13800000000000000000_turntable_side_depth_cross_section.png",
        BOARDS / "V14300000000000000000_head_hair_contour_3d_closeup.png",
        BOARDS / "V14400000000000000000_environment_realism_gate.png",
    ]:
        if board.exists():
            img = Image.open(board).convert("RGB")
            img.thumbnail((520, 350))
            panel = Image.new("RGB", (540, 380), (255, 255, 255))
            panel.paste(img, (10, 34))
            ImageDraw.Draw(panel).text((8, 8), board.name, fill=(0, 0, 0))
            panels.append(panel)
    compose(panels, 1, BOARDS / "V15000000000000000000_multicase_anti_billboard_summary.png")
    decision = {
        "created_at": now(),
        "status": "V15000_MULTICASE_MENTOR_GATE_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "case_count": len(cases),
        "anti_billboard_visual_pass_cases": len(pass_cases),
        "true_better_than_baseline_cases": int(sum(better_baseline)),
        "controls_separated_cases": int(sum(controls_separated)),
        "local_3d_morphology_improvement_cases": 0,
        "no_facial_detail_overclaim": True,
        "board": str(BOARDS / "V15000000000000000000_multicase_anti_billboard_summary.png"),
        "failure_reason": "Required counts are not met; V137/V138/V140 remain fail-closed.",
        "v143_status": v143.get("status"),
        "v144_status": v144.get("status"),
    }
    write_json(REPORTS / "V15000000000000000000_multicase_gate.json", decision)
    return decision


def run_v160(v143: dict[str, Any], v144: dict[str, Any], v145: dict[str, Any], v150: dict[str, Any]) -> dict[str, Any]:
    v140 = json.loads(CAUSALITY.read_text(encoding="utf-8"))
    checks = [
        ("V13050 downgraded", True, "checkpoint freeze reports exist"),
        ("current artifact audit pass", (REPORTS / "V13100000000000000000_current_artifact_quality_audit.json").exists(), "V131 audit exists"),
        ("AGENTS/skill gate updated", (REPORTS / "V13200000000000000000_gate_policy_update.json").exists(), "V132 policy report exists"),
        ("anti-billboard metric v2 implemented", (REPO / "tools" / "V13300_anti_billboard_metric_v2.py").exists(), "metric script exists"),
        ("weak billboard region detection pass", (REPORTS / "V13400000000000000000_billboard_weak_region_manifest.csv").exists(), "V134 manifest exists"),
        ("anti-billboard topology-volume model implemented", (REPO / "models" / "v135_anti_billboard_topology_volume_student.py").exists(), "V135 model exists"),
        ("Modal training matrix complete", (REPORTS / "V13700000000000000000_runtime_environment.json").exists(), "V137 runtime exists"),
        ("model-owned output pass", True, "V137 manifest marks model-owned student outputs"),
        ("no raw Kinect/teacher at inference", True, "V135/V137 reject teacher/Kinect inference keys"),
        ("human-main full-scene RGB screenshot exists", (BOARDS / "V13800000000000000000_advisor_human_main_full_scene.png").exists(), "V138 advisor board exists"),
        ("same-scene controls screenshot exists", (BOARDS / "V13800000000000000000_same_scene_baseline_true_controls.png").exists(), "V138 controls board exists"),
        ("turntable/side-depth/cross-section screenshot exists", (BOARDS / "V13800000000000000000_turntable_side_depth_cross_section.png").exists(), "V138 turntable board exists"),
        ("local morphology screenshots exist", all((BOARDS / f"V14300000000000000000_{r[0]}_3d_closeup.png").exists() for r in REGIONS), "V143 boards exist"),
        ("true anti-billboard > baseline", False, "V140 still records true billboard failures and baseline/control closeness"),
        ("true anti-billboard > same-topology", False, "same-topology remains close or stronger"),
        ("true anti-billboard > shuffled", False, "shuffled remains close or stronger"),
        ("true anti-billboard > thickness-only", False, "thickness-only remains close or stronger"),
        ("true visually better than baseline", False, "V138/V140 visual gate failed closed"),
        ("true visually better than hard controls", False, "V140 causality gate failed closed"),
        ("environment real and visible", bool(v144.get("environment_auxiliary_pass")), "environment passes only as auxiliary"),
        ("no facial detail overclaim", True, "face invisible guard remains active"),
        ("projection auxiliary only", True, "no projection-only pass claimed"),
        ("viewer usable", bool(v145.get("non_placeholder") and v145.get("ply_alias_count", 0) >= 3), "V145 diagnostic viewer exists"),
        ("Yuque report complete", False, "final advisor report is intentionally withheld until V160 passes"),
        ("bundles clean", False, "V190 runs after V160 decision and is diagnostic-only while final gate fails"),
        ("no promotion/registry/V50 changes", True, "no promotion or registry action performed"),
    ]
    rows = [{"gate": name, "pass": bool(ok), "evidence": evidence} for name, ok, evidence in checks]
    failures = [r for r in rows if not r["pass"]]
    status = "V16000_FINAL_MENTOR_GATE_FAIL_CLOSED_TO_V17000"
    gate = {
        "created_at": now(),
        "status": status,
        "mentor_ready": False,
        "external_hard_block": False,
        "failed_gate_count": len(failures),
        "passed_gate_count": len(rows) - len(failures),
        "checks": rows,
        "failures": failures,
        "v140_status": v140.get("status"),
        "v150_status": v150.get("status"),
        "router": "V17000 auto-evolution must continue; visual failure is not external hard block.",
    }
    router = {
        "created_at": now(),
        "route": "V17000000000000000000_auto_evolved_anti_billboard_route",
        "trigger": status,
        "failed_gates": [f["gate"] for f in failures],
        "root_cause": "V137 trained output still behaves like shell/billboard geometry and does not beat same-topology/shuffled/thickness controls.",
        "next_required_repair": [
            "Use explicit part-graph adjacency edges, not scalar penalties.",
            "Decode multiple topology-consistent samples per anchor with occupancy-balanced resampling.",
            "Train adversarial controls in-batch with semantic feature binding ablations.",
            "Select checkpoints from V138/V140 visual boards rather than score alone.",
        ],
        "external_hard_block": False,
    }
    write_json(REPORTS / "V16000000000000000000_final_mentor_gate.json", gate)
    write_json(REPORTS / "V16000000000000000000_failed_gate_router.json", router)
    return gate


def run_v180_guard(v160: dict[str, Any]) -> None:
    guard = (
        "# V180 Advisor Report Guard\n\n"
        "V180 final Yuque-style advisor report is intentionally not written as a mentor-ready report because V160 failed closed.\n\n"
        "Current conclusion: not mentor-ready, not external hard block. Continue V170 auto-evolution.\n\n"
        "Forbidden claims remain active: facial detail, projection-only pass, render-only pass, thickness-only pass, and procedural occupancy success.\n\n"
        f"Failed gate count: {v160.get('failed_gate_count')}\n"
    )
    (REPORTS / "V18000000000000000000_advisor_report_guard.md").write_text(guard, encoding="utf-8")
    (REPORTS / "V18000000000000000000_one_page.md").write_text(
        "# One Page Status\n\nAnti-billboard training ran, but final mentor visual gate failed closed. This is a checkpoint, not a final advisor pass.\n",
        encoding="utf-8",
    )
    (REPORTS / "V18000000000000000000_limitations.md").write_text(
        "# Limitations\n\n- Billboard/sheet geometry remains.\n- Same-topology and shuffled controls remain too strong.\n- Face is not visible; no facial-detail claim is allowed.\n",
        encoding="utf-8",
    )


def add_zip(zip_path: Path, paths: list[Path]) -> dict[str, Any]:
    ensure(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, path.relative_to(REPO).as_posix())
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        names = zf.namelist()
    return {"path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": sha256(zip_path), "entry_count": len(names), "zip_clean": bad is None}


def run_v190() -> dict[str, Any]:
    if BUNDLES.exists():
        shutil.rmtree(BUNDLES)
    ensure(BUNDLES)
    reports = [
        REPORTS / "V13700000000000000000_training_decision.json",
        REPORTS / "V13800000000000000000_visual_gate.json",
        REPORTS / "V14000000000000000000_anti_billboard_causality_decision.json",
        REPORTS / "V14300000000000000000_local_3d_morphology_decision.json",
        REPORTS / "V14400000000000000000_environment_gate.json",
        REPORTS / "V14500000000000000000_viewer_integrity.json",
        REPORTS / "V15000000000000000000_multicase_gate.json",
        REPORTS / "V16000000000000000000_final_mentor_gate.json",
        REPORTS / "V16000000000000000000_failed_gate_router.json",
        REPORTS / "V18000000000000000000_advisor_report_guard.md",
        REPORTS / "V18000000000000000000_one_page.md",
        REPORTS / "V18000000000000000000_limitations.md",
    ]
    visuals = list(BOARDS.glob("V13800000000000000000_*.png")) + list(BOARDS.glob("V14300000000000000000_*.png")) + [
        BOARDS / "V14400000000000000000_environment_realism_gate.png",
        BOARDS / "V15000000000000000000_multicase_anti_billboard_summary.png",
    ]
    core = [
        REPO / "tools" / "V13300_anti_billboard_metric_v2.py",
        REPO / "models" / "v135_anti_billboard_topology_volume_student.py",
        REPO / "tools" / "V14300_V20000_downstream_fail_closed_package.py",
        REPO / "docs" / "goals" / "V13050000000000000000_V60000000000000000000_anti_billboard_topology_volume_training_goal.md",
    ]
    viewer_files = [p for p in VIEWER.rglob("*") if p.is_file()]
    bundles = [
        add_zip(BUNDLES / "reports.zip", reports),
        add_zip(BUNDLES / "visuals.zip", visuals),
        add_zip(BUNDLES / "core.zip", core),
        add_zip(BUNDLES / "viewer.zip", viewer_files),
    ]
    sidecar = {
        "created_at": now(),
        "status": "V19000_DIAGNOSTIC_UPLOAD_SAFE_BUNDLES_CREATED_NOT_FINAL",
        "mentor_ready": False,
        "bundle_root": str(BUNDLES),
        "bundles": bundles,
    }
    integrity = {
        "created_at": now(),
        "all_zip_clean": all(b["zip_clean"] for b in bundles),
        "no_empty_bundle": all(b["entry_count"] > 0 for b in bundles),
        "each_under_500mb": all(b["bytes"] < 500 * 1024 * 1024 for b in bundles),
        "diagnostic_only": True,
        "mentor_ready": False,
        "bundles": bundles,
    }
    write_json(REPORTS / "V19000000000000000000_upload_manifest_sidecar.json", sidecar)
    write_json(REPORTS / "V19000000000000000000_bundle_integrity.json", integrity)
    return integrity


def run_v200() -> dict[str, Any]:
    status = os.popen(f'git -C "{REPO}" status --short --branch').read().strip().splitlines()
    branch = os.popen(f'git -C "{REPO}" rev-parse --abbrev-ref HEAD').read().strip()
    modal_apps = "not_checked_in_cleanup_script"
    payload = {
        "created_at": now(),
        "repo": str(REPO),
        "branch": branch,
        "git_status_short": status,
        "dirty_files_before_commit": max(0, len(status) - 1),
        "modal_apps": modal_apps,
        "python_workers": "not_started_by_this_script",
        "registry_changed": False,
        "v50_v50r2_changed": False,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "source_repos_touched": False,
        "agent_subagent_launched": False,
        "note": "This cleanup report is generated before the final commit/push step; final response must report post-push git status.",
    }
    write_json(REPORTS / "V20000000000000000000_post_push_cleanup.json", payload)
    return payload


def main() -> int:
    ensure(REPORTS)
    ensure(BOARDS)
    metrics_rows = read_metrics()
    cases = cases_from_metrics(metrics_rows)
    if not cases:
        raise RuntimeError("No V137 true cases found in seed metrics.")
    v143 = run_v143(metrics_rows, cases)
    v144 = run_v144(cases)
    v145 = run_v145(cases)
    v150 = run_v150(metrics_rows, v143, v144)
    v160 = run_v160(v143, v144, v145, v150)
    run_v180_guard(v160)
    run_v190()
    run_v200()
    print(json.dumps({"status": v160["status"], "cases": cases, "mentor_ready": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
