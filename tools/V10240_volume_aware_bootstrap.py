from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
GOALS = REPO / "docs" / "goals"
TOOLS = REPO / "tools"
ARCHIVE = REPO / "archive"

CASE_CONFIGS = [
    ("baseline", "real_vggt_baseline_only"),
    ("posthoc", "posthoc_surfel_only"),
    ("same_topology", "same_topology_no_semantic"),
    ("tiny", "tiny_synthetic_token_control"),
    ("shuffled", "shuffled_smpl_feature"),
]
TRAINING_MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
V10190_CANDIDATE = OUTPUT / "V10190000000000000000_thickness_aware_geometry_candidate"
V140_ROOT = OUTPUT / "V1400000000000000000_learned_residual_matrix"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
FINAL_ALLOWED = [
    "V30000000000000000000_VOLUME_AWARE_3D_MORPHOLOGY_MENTOR_READY_NOT_PROMOTED",
    "V30000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["name"])
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest_rows() -> list[dict[str, str]]:
    with TRAINING_MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    return center, vals, vecs, proj


def geometry_metrics(points: np.ndarray, *, prefix: str = "") -> dict[str, float | int]:
    pts = np.asarray(points, dtype=np.float64)
    _center, vals, _vecs, proj = pca_frame(pts)
    ranges = np.ptp(proj, axis=0)
    bbox = np.ptp(pts, axis=0)
    bins = np.floor((pts[:, :2] - pts[:, :2].min(axis=0)) / np.maximum(np.ptp(pts[:, :2], axis=0), 1e-6) * 16).astype(int)
    bins = np.clip(bins, 0, 15)
    occupied = len({(int(a), int(b)) for a, b in bins})
    return {
        prefix + "point_count": int(len(pts)),
        prefix + "bbox_x": float(bbox[0]),
        prefix + "bbox_y": float(bbox[1]),
        prefix + "bbox_z": float(bbox[2]),
        prefix + "z_range": float(bbox[2]),
        prefix + "side_thickness": float(min(bbox[0], bbox[1])),
        prefix + "front_back_thickness": float(bbox[2]),
        prefix + "pca_range_1": float(ranges[0]),
        prefix + "pca_range_2": float(ranges[1]),
        prefix + "pca_range_3": float(ranges[2]),
        prefix + "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        prefix + "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
        prefix + "z_iqr": float(np.quantile(pts[:, 2], 0.75) - np.quantile(pts[:, 2], 0.25)),
        prefix + "z_p95_p05": float(np.quantile(pts[:, 2], 0.95) - np.quantile(pts[:, 2], 0.05)),
        prefix + "xy_occupancy_bins_16": int(occupied),
    }


def local_region_metric(points: np.ndarray, mask: np.ndarray, name: str) -> dict[str, float | int]:
    if not np.any(mask):
        return {f"{name}_points": 0, f"{name}_thickness": 0.0, f"{name}_continuity": 0.0}
    pts = points[mask]
    met = geometry_metrics(pts)
    return {
        f"{name}_points": int(len(pts)),
        f"{name}_thickness": float(met["pca_thickness_ratio"]),
        f"{name}_z_range": float(met["z_range"]),
        f"{name}_continuity": float(met["xy_occupancy_bins_16"] / 256.0),
    }


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return rz @ rx @ ry


def project(points: np.ndarray, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pts = (points - np.mean(points, axis=0, keepdims=True)) @ rot.T
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    width, height = size
    xy = np.clip(q * np.array([width - 54, height - 82]) + np.array([27, 48]), 0, [width - 1, height - 1]).astype(np.int32)
    return xy, pts[:, 2]


def depth_tint(colors: np.ndarray, depth: np.ndarray, *, strength: float = 0.38) -> np.ndarray:
    rgb = as_rgb(colors).astype(np.float32)
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    shade = 0.66 + strength * d[:, None]
    cool = np.array([0.82, 0.92, 1.04], dtype=np.float32)
    warm = np.array([1.08, 1.00, 0.86], dtype=np.float32)
    tint = cool[None] * (1.0 - d[:, None]) + warm[None] * d[:, None]
    return np.clip(rgb * shade * tint, 0, 255).astype(np.uint8)


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, size: tuple[int, int] = (420, 330), point_step_cap: int = 70000) -> Image.Image:
    width, height = size
    im = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(im)
    xy, depth = project(points, rot, lo, hi, size)
    rgb = depth_tint(colors, depth)
    order = np.argsort(depth)
    step = max(1, len(order) // point_step_cap)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((10, 9), title, fill=(18, 18, 18))
    return im


def global_bounds(preds: list[tuple[str, np.ndarray, np.ndarray]], rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = []
    for _title, points, _rgb in preds:
        centered = points - np.mean(points, axis=0, keepdims=True)
        projected.append(centered @ rot.T)
    xy = np.concatenate([p[:, :2] for p in projected], axis=0)
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    pad = (hi - lo) * 0.16 + 1e-6
    return lo - pad, hi + pad


def compose_grid(panels: list[Image.Image], cols: int, path: Path) -> Path:
    ensure(path.parent)
    w, h = panels[0].size
    rows = int(math.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)
    return path


def prediction_for(row: dict[str, str], label: str, config: str) -> Path:
    if label == "candidate" and row["case"] == "0012_11_frame001":
        return V10190_CANDIDATE / row["case"] / "thickness_aware_geometry_candidate" / "predictions.npz"
    if label == "candidate":
        return Path(row["baseline_path"])
    if config == "real_vggt_baseline_only":
        return Path(row["baseline_path"])
    key = f"control_{config}_path"
    return Path(row[key])


def audit_path(path: Path, category: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "category": category,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256(path) if path.exists() and path.is_file() and path.stat().st_size < 64 * 1024 * 1024 else "",
    }
    if not path.exists():
        return row
    suffix = path.suffix.lower()
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                bad = z.testzip()
                row.update({"zip_clean": bad is None, "zip_entry_count": len(z.infolist())})
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                row.update({"npz_readable": True, "npz_keys": ";".join(z.files[:20])})
        elif suffix == ".png":
            with Image.open(path) as im:
                row.update({"png_open": True, "image_size": f"{im.width}x{im.height}"})
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            row.update({"json_readable": True})
        elif suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                row.update({"csv_readable": True, "csv_columns": len(header)})
        elif suffix in {".html", ".htm"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            row.update({"html_usable": "<html" in text.lower() and len(text) > 512, "html_size": len(text)})
        elif suffix == ".py":
            text = path.read_text(encoding="utf-8", errors="replace")
            row.update(
                {
                    "py_readable": True,
                    "mentions_raw_points_xy": "points[:, :2]" in text or "points[:,:2]" in text,
                    "raw_points_xy_main_risk": (
                        ("points[:, :2]" in text or "points[:,:2]" in text)
                        and "render_panel" in text
                        and "rotation_matrix" not in text
                        and "depth_tint" not in text
                    ),
                }
            )
        else:
            row.update({"readable": True})
    except Exception as exc:  # noqa: BLE001
        row.update({"read_error": f"{type(exc).__name__}: {exc}"})
    return row


def build_checkpoint_freeze(created_at: str) -> None:
    sources = {
        "v10230_state": REPORTS / "V10230000000000000000_current_route_state.json",
        "v10170_render_decision": REPORTS / "V10170000000000000000_flatness_and_depth_render_decision.json",
        "v10180_repair_decision": REPORTS / "V10180000000000000000_depth_cue_and_geometry_repair_decision.json",
        "v10190_candidate_decision": REPORTS / "V10190000000000000000_thickness_aware_geometry_candidate_decision.json",
        "v404_face_visibility": REPORTS / "V4040000000000000000_face_visibility_gate.json",
        "v10220_model_smoke": REPORTS / "V10220000000000000000_true_3d_geometry_model_smoke.json",
    }
    loaded = {name: load_json(path) for name, path in sources.items() if path.exists() and path.suffix == ".json"}
    freeze = {
        "created_at": created_at,
        "status": "V10240_V10230_DOWNGRADED_TO_CHECKPOINT_INTERNAL_VISUAL_HARD_BLOCK",
        "previous_state": loaded.get("v10230_state", {}).get("status"),
        "mentor_ready": False,
        "external_hard_block": False,
        "reason_summary": [
            "V10150/V10170 show render boards using raw XY-style projection are diagnostic only.",
            "V10180 shows rendering is not the only issue; geometry still lacks stable 3D volume advantage.",
            "V10190 increases thickness for one candidate but shuffled control is thicker, so thickness gain is not causal mentor success.",
            "V10220 is model smoke only and does not provide full-scene mentor visual pass.",
            "V404 proves facial-detail target is not applicable; only head/face contour and hair region can be claimed.",
        ],
        "sources": {name: str(path) for name, path in sources.items()},
        "final_allowed_states": FINAL_ALLOWED,
        "no_agent_rule": "No agent/subagent launch in this run.",
    }
    write_json(REPORTS / "V10240000000000000000_v10230_checkpoint_freeze.json", freeze)
    why = f"""# Why V10230 Is Not Final

Created: {created_at}

V10230 is frozen as a checkpoint / internal visual hard block. It is not mentor-ready and it is not a true external hard block.

Key findings:

- V10150/V10170 exposed a real render issue: mentor-style boards cannot use raw `points[:, :2]` projection as the main point-cloud figure.
- V10180 showed that render repair alone is insufficient; the candidate geometry was not meaningfully more 3D than VGGT baseline.
- V10190 increased thickness from the baseline level, but the shuffled control was thicker, so thickness gain alone cannot prove a causal VGGT-SMPL route.
- V10220 only smoked the canonical surfel/graph student path. It proves the path can train and rejects teacher keys, but does not prove mentor visual success.
- V404 proves the current source views do not show a readable face. Facial detail claims are forbidden; the allowed claim is `{ALLOWED_FACE_CLAIM}`.

Next route: volume-aware 3D morphology, with oblique/depth-cued rendering only as an audit aid, and a multi-layer SMPL-conditioned residual representation as the model path.
"""
    write_text(REPORTS / "V10240000000000000000_why_v10230_is_not_final.md", why)
    failure_register = """# V10240 Visual And Geometry Failure Register

| Failure | Evidence | Routing |
| --- | --- | --- |
| Raw XY/2D render board risk | V10170 render decision and audit board | Render standardizer required, but render-only pass is forbidden |
| Thin/sheet-like human geometry | V10180 and V10190 geometry audits | Build volume-aware representation and weak-volume detection |
| Thickness-only false positive | V10190 candidate vs shuffled control | Add volume causality gate and hard-control separation |
| Facial detail target invalid | V393/V404 face visibility gates | Forbid eyes/nose/mouth claims; optimize visible morphology |
| Model smoke not final | V10220 smoke | Continue to matrix training and mentor visual gate |
"""
    write_text(REPORTS / "V10240000000000000000_visual_and_geometry_failure_register.md", failure_register)


def artifact_audit(created_at: str, rows: list[dict[str, str]]) -> None:
    paths: list[tuple[Path, str]] = []
    for pattern in [
        "V1017*",
        "V1018*",
        "V1019*",
        "V1021*",
        "V1022*",
        "V1023*",
        "V393*",
        "V404*",
        "V405*",
        "V1024*",
    ]:
        for folder in [REPORTS, BOARDS, TOOLS, GOALS]:
            paths.extend((p, folder.name) for p in folder.glob(pattern))
    for row in rows:
        for key, value in row.items():
            if key.endswith("_path") or key in {"baseline_path", "graph_path", "visible_target_path"}:
                if value:
                    paths.append((Path(value), "training_input"))
    for p in ARCHIVE.glob("V*bundle.zip"):
        if any(token in p.name for token in ["V950", "V2800", "V2900", "V550", "V940"]):
            paths.append((p, "bundle"))
    seen: set[str] = set()
    audit_rows = []
    for path, category in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        audit_rows.append(audit_path(path, category))
    write_csv(REPORTS / "V10250000000000000000_current_artifact_index.csv", audit_rows)
    py_xy = [r["path"] for r in audit_rows if r.get("raw_points_xy_main_risk")]
    missing = [r["path"] for r in audit_rows if not r.get("exists")]
    quality = {
        "created_at": created_at,
        "status": "V10250_CURRENT_ARTIFACT_AUDIT_COMPLETE_FAIL_CLOSED_FOR_MAIN_VISUAL",
        "artifact_count": len(audit_rows),
        "missing_count": len(missing),
        "raw_points_xy_main_risk_count": len(py_xy),
        "raw_points_xy_main_risk_files": py_xy,
        "zip_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith(".zip")),
        "npz_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith(".npz")),
        "png_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith(".png")),
        "json_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith(".json")),
        "csv_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith(".csv")),
        "html_count": sum(1 for r in audit_rows if str(r["path"]).lower().endswith((".html", ".htm"))),
        "mentor_ready": False,
        "external_hard_block": False,
        "hard_gate_notes": [
            "Recovered evidence is current local file evidence, not final mentor success.",
            "Any raw points[:, :2] mentor main renderer is diagnostic only; rotated/depth-cued projections are allowed as render/audit views.",
            "Face-invisible cases forbid facial detail claims.",
            "Thickness metric pass cannot override shuffled/random controls.",
        ],
    }
    write_json(REPORTS / "V10250000000000000000_current_artifact_quality_audit.json", quality)
    write_text(
        REPORTS / "V10250000000000000000_current_route_decision.md",
        f"""# V10250 Current Route Decision

Created: {created_at}

The current artifacts are readable enough to continue locally: the V10210 training manifest points to four eligible cases with baseline, graph, visible target, and controls. This is not an external hard block.

Fail-closed findings:

- V10230 is incomplete and cannot be returned.
- Render repair is required but cannot be final.
- Thickness-only gains are not sufficient because shuffled/random controls can be thicker.
- Facial detail remains not applicable; only `{ALLOWED_FACE_CLAIM}` is allowed.

Next: run render standardization, volume metrics, weak-volume detection, volume-aware model smoke, and then prepare V10700 training execution.
""",
    )


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = load_npz(path)
    human = np.asarray(pred["human_points"], dtype=np.float32)
    human_rgb = as_rgb(pred["human_rgb"])
    env = np.asarray(pred.get("environment_points", np.empty((0, 3))), dtype=np.float32)
    env_rgb = as_rgb(pred.get("environment_rgb", np.empty((0, 3), dtype=np.uint8))) if len(env) else np.empty((0, 3), dtype=np.uint8)
    full = np.asarray(pred.get("full_scene_points", np.concatenate([human, env], axis=0)), dtype=np.float32)
    full_rgb = as_rgb(pred.get("full_scene_rgb", np.concatenate([human_rgb, env_rgb], axis=0))) if len(full) else human_rgb
    return human, human_rgb, full, full_rgb


def render_standard_boards(rows: list[dict[str, str]], created_at: str) -> None:
    row = next(r for r in rows if r["case"] == "0012_11_frame001")
    preds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, config in [("baseline", "real_vggt_baseline_only"), ("candidate", "candidate"), ("posthoc", "posthoc_surfel_only"), ("same topology", "same_topology_no_semantic"), ("tiny", "tiny_synthetic_token_control"), ("shuffled", "shuffled_smpl_feature")]:
        path = prediction_for(row, label, config)
        if path.exists():
            _human, _hrgb, full, full_rgb = load_prediction(path)
            preds.append((f"0012_11 {label}", full, full_rgb))
    oblique_rot = rotation_matrix(-34, 58, 0)
    side_rot = rotation_matrix(84, 52, 0)
    lo, hi = global_bounds(preds, oblique_rot)
    oblique_panels = [render_panel(points, rgb, title, oblique_rot, lo, hi) for title, points, rgb in preds]
    compose_grid(oblique_panels, 3, BOARDS / "V10260000000000000000_render_standard_oblique_depth.png")
    side_lo, side_hi = global_bounds(preds, side_rot)
    side_panels = [render_panel(points, rgb, title + " side-depth", side_rot, side_lo, side_hi) for title, points, rgb in preds[:6]]
    compose_grid(side_panels, 3, BOARDS / "V10260000000000000000_render_standard_same_scene_controls.png")
    first_title, first_points, first_rgb = preds[1] if len(preds) > 1 else preds[0]
    turn_rots = [
        ("front", rotation_matrix(0, 58, 0)),
        ("right", rotation_matrix(90, 58, 0)),
        ("back", rotation_matrix(180, 58, 0)),
        ("left", rotation_matrix(270, 58, 0)),
        ("oblique", oblique_rot),
        ("side-depth", side_rot),
    ]
    turn_panels = []
    for title, rot in turn_rots:
        tlo, thi = global_bounds([(first_title, first_points, first_rgb)], rot)
        turn_panels.append(render_panel(first_points, first_rgb, f"{first_title} {title}", rot, tlo, thi))
    compose_grid(turn_panels, 3, BOARDS / "V10260000000000000000_render_standard_turntable.png")
    decision = {
        "created_at": created_at,
        "status": "V10260_RENDER_STANDARDIZER_COMPLETE_AUXILIARY_ONLY",
        "outputs": {
            "oblique_depth": str(BOARDS / "V10260000000000000000_render_standard_oblique_depth.png"),
            "turntable": str(BOARDS / "V10260000000000000000_render_standard_turntable.png"),
            "same_scene_controls": str(BOARDS / "V10260000000000000000_render_standard_same_scene_controls.png"),
        },
        "uses_raw_points_xy_as_main": False,
        "render_only_mentor_ready": False,
        "mentor_ready": False,
        "external_hard_block": False,
    }
    write_json(REPORTS / "V10260000000000000000_render_standard_decision.json", decision)


def volume_metrics(rows: list[dict[str, str]], created_at: str) -> None:
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        case = row["case"]
        graph = load_npz(Path(row["graph_path"]))
        region_masks = {
            "head_hair": np.asarray(graph.get("head_hair_contour_mask", np.zeros(0)), dtype=bool),
            "shoulder_neck": np.asarray(graph.get("shoulder_neck_mask", np.zeros(0)), dtype=bool),
            "hand_arm": np.asarray(graph.get("hand_arm_endpoint_mask", np.zeros(0)), dtype=bool),
            "clothing": np.asarray(graph.get("clothing_torso_boundary_mask", np.zeros(0)), dtype=bool),
            "leg_foot": np.asarray(graph.get("leg_foot_morphology_mask", np.zeros(0)), dtype=bool),
        }
        configs = list(CASE_CONFIGS)
        if case == "0012_11_frame001":
            configs.insert(1, ("candidate", "candidate"))
        for label, config in configs:
            path = prediction_for(row, label, config)
            if not path.exists():
                metric_rows.append({"case": case, "label": label, "config": config, "exists": False})
                continue
            human, _hrgb, full, _frgb = load_prediction(path)
            out: dict[str, Any] = {"case": case, "label": label, "config": config, "exists": True, "path": str(path)}
            out.update(geometry_metrics(human, prefix="human_"))
            out.update(geometry_metrics(full, prefix="full_"))
            for region, mask in region_masks.items():
                if len(mask) == len(human):
                    out.update(local_region_metric(human, mask, region))
            metric_rows.append(out)
    write_csv(REPORTS / "V10300000000000000000_volume_geometry_metrics.csv", metric_rows)
    # Make a compact visual board from the first case metrics.
    img = Image.new("RGB", (1300, 520), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), "V10300 volume metrics: thickness is diagnostic; shuffled/random thicker than true fails closed", fill=(0, 0, 0))
    first = [r for r in metric_rows if r.get("case") == "0012_11_frame001" and r.get("exists")]
    max_v = max(float(r.get("human_pca_thickness_ratio", 0)) for r in first) if first else 1.0
    for i, r in enumerate(first):
        y = 62 + i * 52
        val = float(r.get("human_pca_thickness_ratio", 0))
        z = float(r.get("human_z_range", 0))
        w = int(700 * val / max(max_v, 1e-6))
        color = (70, 120, 190) if r["label"] == "candidate" else (100, 100, 100)
        if r["label"] == "shuffled":
            color = (190, 80, 80)
        draw.text((24, y), f"{r['label']}: thickness={val:.4f}, z={z:.4f}", fill=(0, 0, 0))
        draw.rectangle([360, y, 360 + w, y + 24], fill=color)
    ensure(BOARDS)
    img.save(BOARDS / "V10300000000000000000_volume_metric_visual_board.png")
    by_case: dict[str, dict[str, float]] = {}
    for r in metric_rows:
        if not r.get("exists"):
            continue
        by_case.setdefault(str(r["case"]), {})[str(r["label"])] = float(r.get("human_pca_thickness_ratio", 0))
    failures = []
    for case, vals in by_case.items():
        true_val = vals.get("candidate", vals.get("baseline", 0))
        for control in ["shuffled", "same_topology", "tiny", "posthoc"]:
            if vals.get(control, -1) >= true_val:
                failures.append({"case": case, "control": control, "control_thickness": vals.get(control), "true_or_candidate_thickness": true_val})
    decision = {
        "created_at": created_at,
        "status": "V10300_VOLUME_GEOMETRY_METRICS_COMPLETE_FAIL_CLOSED",
        "metrics_csv": str(REPORTS / "V10300000000000000000_volume_geometry_metrics.csv"),
        "board": str(BOARDS / "V10300000000000000000_volume_metric_visual_board.png"),
        "true_stronger_than_controls": len(failures) == 0,
        "failures": failures[:20],
        "mentor_ready": False,
        "external_hard_block": False,
        "note": "Thickness/volume metrics are auxiliary; visual gate is still required.",
    }
    write_json(REPORTS / "V10300000000000000000_volume_geometry_decision.json", decision)


def weak_volume_regions(rows: list[dict[str, str]], created_at: str) -> None:
    out_root = ensure(OUTPUT / "V10400000000000000000_weak_volume_regions")
    manifest: list[dict[str, Any]] = []
    preview_panels: list[Image.Image] = []
    for row in rows:
        case = row["case"]
        base = load_npz(Path(row["baseline_path"]))
        graph = load_npz(Path(row["graph_path"]))
        points = np.asarray(base["human_points"], dtype=np.float32)
        rgb = as_rgb(base["human_rgb"])
        weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
        no_change = np.asarray(graph["no_change_mask"], dtype=bool)
        local_masks = [
            np.asarray(graph.get("head_hair_contour_mask", np.zeros(len(points))), dtype=bool),
            np.asarray(graph.get("shoulder_neck_mask", np.zeros(len(points))), dtype=bool),
            np.asarray(graph.get("hand_arm_endpoint_mask", np.zeros(len(points))), dtype=bool),
            np.asarray(graph.get("clothing_torso_boundary_mask", np.zeros(len(points))), dtype=bool),
            np.asarray(graph.get("leg_foot_morphology_mask", np.zeros(len(points))), dtype=bool),
        ]
        visible_part = np.logical_or.reduce(local_masks)
        _center, _vals, vecs, proj = pca_frame(points)
        abs_thin_axis = np.abs(proj[:, 2])
        sheet_region = abs_thin_axis < np.quantile(abs_thin_axis, 0.25)
        weak_volume = (weak > 0.18) & visible_part & ~no_change
        multi_layer_missing = weak_volume & sheet_region
        out = out_root / case / "weak_volume_regions.npz"
        ensure(out.parent)
        np.savez_compressed(
            out,
            human_points=points,
            human_rgb=rgb,
            weak_region_score=weak,
            no_change_mask=no_change,
            visible_part_mask=visible_part,
            sheet_region_mask=sheet_region,
            weak_volume_region_mask=weak_volume,
            multi_layer_missing_mask=multi_layer_missing,
            pca_axes=vecs.astype(np.float32),
            facial_detail_target_applicable=np.array(False),
            face_detail_claim_allowed=np.array(False),
            allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        )
        manifest.append(
            {
                "case": case,
                "output": str(out),
                "human_points": len(points),
                "weak_volume_ratio": float(np.mean(weak_volume)),
                "multi_layer_missing_ratio": float(np.mean(multi_layer_missing)),
                "no_change_ratio": float(np.mean(no_change)),
                "face_detail_claim_allowed": False,
                "allowed_face_claim": ALLOWED_FACE_CLAIM,
            }
        )
        colors = rgb.copy()
        colors[weak_volume] = np.array([255, 70, 40], dtype=np.uint8)
        colors[multi_layer_missing] = np.array([255, 210, 20], dtype=np.uint8)
        full = np.concatenate([points, np.asarray(base["environment_points"], dtype=np.float32)], axis=0)
        full_rgb = np.concatenate([colors, as_rgb(base["environment_rgb"])], axis=0)
        rot = rotation_matrix(-34, 58, 0)
        lo, hi = global_bounds([(case, full, full_rgb)], rot)
        preview_panels.append(render_panel(full, full_rgb, f"{case} weak-volume", rot, lo, hi))
    write_csv(REPORTS / "V10400000000000000000_weak_volume_region_manifest.csv", manifest)
    compose_grid(preview_panels, 2, BOARDS / "V10400000000000000000_weak_volume_region_preview.png")


def model_and_loss_smoke(created_at: str, rows: list[dict[str, str]]) -> None:
    sys.path.insert(0, str(REPO))
    import torch
    from models.v105_volume_aware_visible_morphology_student import (
        VolumeAwareVisibleMorphologyConfig,
        VolumeAwareVisibleMorphologyStudent,
        smoke_test,
    )

    cfg = VolumeAwareVisibleMorphologyConfig()
    generic = smoke_test()
    row = rows[0]
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak_npz = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / row["case"] / "weak_volume_regions.npz")
    pts = np.asarray(base["human_points"], dtype=np.float32)
    rgb = np.asarray(base["human_rgb"], dtype=np.float32) / 255.0
    weak = np.asarray(weak_npz["weak_volume_region_mask"], dtype=bool).astype(np.float32)
    score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    idx = np.argsort(-(weak + score * 0.1))[:1024]
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    part = np.eye(8, dtype=np.float32)[np.clip(body[idx], 0, 7)]
    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, :8] = part
    smpl[:, 8] = score[idx]
    smpl[:, 9] = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)[idx]
    batch = {
        "anchor_xyz": torch.from_numpy(pts[idx][None]),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]),
        "confidence": torch.from_numpy(smpl[:, 9][None]),
        "weak_region": torch.from_numpy(np.maximum(weak[idx], score[idx] * 0.25)[None]),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim),
        "smpl_features": torch.from_numpy(smpl[None]),
        "vggt_token_context": torch.zeros(1, cfg.token_dim),
    }
    model = VolumeAwareVisibleMorphologyStudent(cfg)
    out = model(batch)
    residual_norm = torch.linalg.norm(out["residual_xyz"], dim=-1)
    weak_tensor = batch["weak_region"]
    no_change = torch.from_numpy(np.asarray(graph["no_change_mask"], dtype=bool)[idx][None])
    baseline_preservation_loss = (residual_norm * no_change.float()).mean()
    weak_region_residual_loss = -(residual_norm * weak_tensor).mean()
    thickness_field_loss = -out["thickness_field"].mean()
    shell_sep = (
        torch.linalg.norm(out["front_shell"] - out["back_shell"], dim=-1).mean()
        + torch.linalg.norm(out["front_shell"] - out["side_shell"], dim=-1).mean() * 0.25
    )
    limb_continuity_loss = out["normal"].diff(dim=1).abs().mean()
    rgb_loss = out["rgb_delta"].abs().mean()
    loss = (
        baseline_preservation_loss * 2.0
        + weak_region_residual_loss * 0.5
        + thickness_field_loss * 0.1
        - shell_sep * 0.02
        + limb_continuity_loss * 0.02
        + rgb_loss * 0.1
    )
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    forbidden_rejected = False
    try:
        bad = dict(batch)
        bad["teacher_points"] = batch["anchor_xyz"]
        model(bad)
    except ValueError:
        forbidden_rejected = True
    contract = {
        "created_at": created_at,
        "model": "models/v105_volume_aware_visible_morphology_student.py::VolumeAwareVisibleMorphologyStudent",
        "representation_layers": [
            "baseline_anchor_layer",
            "SMPL-X_bound_support_layer",
            "multi_layer_shell_layer",
            "weak_region_gated_residual_field",
            "real_environment_branch",
        ],
        "forbidden_inference_inputs": sorted(
            [
                "raw_kinect_depth",
                "kinect_depth",
                "teacher_points",
                "teacher_xyz",
                "v591_points",
                "v591_teacher",
                "dense_teacher",
            ]
        ),
        "mentor_ready_from_contract": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
    }
    write_json(REPORTS / "V10500000000000000000_model_contract.json", contract)
    write_text(
        REPORTS / "V10500000000000000000_architecture_diagram.md",
        """# V105 Volume-Aware Visible Morphology Architecture

```text
VGGT baseline anchors + RGB/confidence
        + SMPL-X surfel/graph local features
        + weak-volume region mask
        -> VolumeAwareVisibleMorphologyStudent
        -> residual xyz + front/back/side shell offsets
        -> real environment insertion
        -> human-main full-scene RGB point cloud
```

Projection, mask, RGB, and edge metrics are auxiliary. Mentor success still requires a human-main 3D board with visible environment and controls.
""",
    )
    smoke = {
        "created_at": created_at,
        "status": "V105_VOLUME_AWARE_MODEL_SMOKE_PASS_INTERNAL_ONLY"
        if generic["grad_norm_positive"] and forbidden_rejected and grad_norm > 0
        else "V105_VOLUME_AWARE_MODEL_SMOKE_FAIL_CLOSED",
        "generic_smoke": generic,
        "case": row["case"],
        "batch_points": int(len(idx)),
        "grad_norm_positive": grad_norm > 0,
        "grad_norm": grad_norm,
        "forbidden_teacher_key_rejected": forbidden_rejected,
        "student_shape": list(out["student_xyz"].shape),
        "thickness_field_shape": list(out["thickness_field"].shape),
        "mentor_ready": False,
        "external_hard_block": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
    }
    write_json(REPORTS / "V10500000000000000000_forward_smoke.json", smoke)
    loss_contract = f"""# V106 Volume Supervision Loss Contract

Created: {created_at}

Losses:

- baseline preservation loss: penalize residuals in VGGT high-confidence/no-change zones.
- weak-region residual loss: allow residual movement only where V104 weak-volume masks say baseline is missing, low-confidence, or single-layer.
- thickness field loss: encourages front/back/side shell separation, but cannot be final evidence alone.
- front/back shell separation loss: supervises multi-layer surface structure.
- limb continuity loss: smooths normals/residuals inside visible body-part regions.
- environment preservation loss: requires real VGGT environment points to remain inserted, not random sprinkle.
- projection auxiliary loss: optional mask/RGB/edge consistency only; it cannot rescue a failed 3D mentor board.
- control separation loss: same-budget baseline/posthoc/same-topology/tiny/shuffled controls must remain weaker than true.

Forbidden: raw Kinect or teacher points at inference. Face detail claims are forbidden for current back/side-back cases.
"""
    write_text(REPORTS / "V10600000000000000000_loss_contract.md", loss_contract)
    loss_smoke = {
        "created_at": created_at,
        "status": "V106_VOLUME_LOSS_SMOKE_PASS_INTERNAL_ONLY" if grad_norm > 0 else "V106_VOLUME_LOSS_SMOKE_FAIL_CLOSED",
        "case": row["case"],
        "loss": float(loss.detach()),
        "baseline_preservation_loss": float(baseline_preservation_loss.detach()),
        "weak_region_residual_loss": float(weak_region_residual_loss.detach()),
        "thickness_field_loss": float(thickness_field_loss.detach()),
        "shell_separation": float(shell_sep.detach()),
        "limb_continuity_loss": float(limb_continuity_loss.detach()),
        "rgb_loss": float(rgb_loss.detach()),
        "projection_auxiliary_only": True,
        "mentor_ready": False,
        "external_hard_block": False,
    }
    write_json(REPORTS / "V10600000000000000000_loss_smoke.json", loss_smoke)


def main() -> int:
    created_at = now()
    rows = read_manifest_rows()
    build_checkpoint_freeze(created_at)
    artifact_audit(created_at, rows)
    render_standard_boards(rows, created_at)
    volume_metrics(rows, created_at)
    weak_volume_regions(rows, created_at)
    model_and_loss_smoke(created_at, rows)
    print(
        json.dumps(
            {
                "created_at": created_at,
                "status": "V10240_TO_V10600_LOCAL_BOOTSTRAP_COMPLETE_INTERNAL_ONLY",
                "mentor_ready": False,
                "external_hard_block": False,
                "next": "Prepare and run V10700 volume-aware training matrix; do not return at local smoke.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
