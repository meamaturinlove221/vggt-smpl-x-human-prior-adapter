from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.v360_canonical_smplx_graph_volume_student import (  # noqa: E402
    CanonicalSMPLXGraphVolumeStudent,
    V360Config,
)
from tools.V505_teacher_copy_detector import detect as detect_teacher_copy  # noqa: E402
from tools.V514_v50r2_checkpoint_adjudicator import (  # noqa: E402
    TEACHER_BANK,
    load_observation_inputs,
    render_points,
    write_json,
    write_ply,
)


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
GOALS = ROOT / "docs" / "goals"
OUT = ROOT / "output" / "V5180000000000000000000_canonical_surfel_graph_repair"

DECISION = REPORTS / "V5180000000000000000000_canonical_surfel_graph_repair_decision.json"
METRICS = REPORTS / "V5180000000000000000000_canonical_surfel_graph_repair_metrics.csv"
COPY_JSON = REPORTS / "V5180000000000000000000_teacher_copy_check.json"
BOARD = BOARDS / "V5180000000000000000000_canonical_surfel_graph_repair_board.png"
CONTROLS_BOARD = BOARDS / "V5180000000000000000000_canonical_surfel_graph_controls.png"
ROUTE = GOALS / "V5180000000000000000000_auto_evolved_canonical_surfel_graph_repair_route.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["row"])
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(points: np.ndarray, rgb: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points).all(axis=-1)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return points[:0].astype(np.float32), rgb[:0].astype(np.float32)
    if idx.size > limit:
        idx = np.random.default_rng(seed).choice(idx, size=limit, replace=False)
    idx = np.sort(idx)
    return points[idx].astype(np.float32), rgb[idx].astype(np.float32)


def make_v360_batch(batch: dict[str, torch.Tensor], *, sample_limit: int, seed: int) -> dict[str, torch.Tensor]:
    pts = batch["vggt_world_points"].reshape(-1, 3).cpu().numpy()
    rgb = batch["vggt_rgb"].reshape(-1, 3).cpu().numpy()
    conf = batch["vggt_confidence"].reshape(-1, 1).cpu().numpy()
    pts, rgb = sample_rows(pts, rgb, sample_limit, seed)
    conf, _ = sample_rows(conf, np.zeros((conf.shape[0], 1), dtype=np.float32), sample_limit, seed)
    if conf.shape[0] != pts.shape[0]:
        conf = np.ones((pts.shape[0], 1), dtype=np.float32)
    weak = np.clip(conf.astype(np.float32), 0.0, 1.0)
    return {
        "visible_points": torch.from_numpy(pts[None]).float(),
        "visible_rgb": torch.from_numpy(rgb[None]).float(),
        "visible_confidence": torch.from_numpy(weak[None]).float(),
        "weak_score": torch.from_numpy(weak[None]).float(),
        "proposal_score": torch.from_numpy(weak[None]).float(),
    }


def surfel_directions() -> np.ndarray:
    dirs = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.7071, 0.7071, 0.0),
        (-0.7071, 0.7071, 0.0),
        (0.7071, -0.7071, 0.0),
        (-0.7071, -0.7071, 0.0),
        (0.7071, 0.0, 0.7071),
        (-0.7071, 0.0, 0.7071),
        (0.7071, 0.0, -0.7071),
        (-0.7071, 0.0, -0.7071),
        (0.0, 0.7071, 0.7071),
        (0.0, -0.7071, 0.7071),
        (0.0, 0.7071, -0.7071),
        (0.0, -0.7071, -0.7071),
    ]
    arr = np.asarray(dirs, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1.0e-6)


def flatten_v360(out: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    nodes = out["scene_nodes"].detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    radius = out["cross_section_radius"].detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    radius = np.maximum(radius, np.array([0.003, 0.004, 0.018], dtype=np.float32))
    node_rgb = out["volume_rgb"].detach().cpu().numpy().reshape(nodes.shape[0], -1, 3)[:, 0].astype(np.float32)
    flat_occ = out["occupancy"].detach().cpu().numpy().reshape(nodes.shape[0], -1).astype(np.float32)
    node_occ = flat_occ.mean(axis=1)
    flat_part = out["body_part_id"].detach().cpu().numpy().reshape(nodes.shape[0], -1).astype(np.int32)
    node_part = flat_part[:, 0]
    dirs = surfel_directions()
    points = nodes[:, None, :] + dirs[None, :, :] * radius[:, None, :]
    rgb = np.repeat(node_rgb[:, None, :], dirs.shape[0], axis=1)
    occ = np.repeat(node_occ[:, None], dirs.shape[0], axis=1)
    part = np.repeat(node_part[:, None], dirs.shape[0], axis=1)
    points = points.reshape(-1, 3).astype(np.float32)
    rgb = rgb.reshape(-1, 3).astype(np.float32)
    occ = occ.reshape(-1).astype(np.float32)
    part = part.reshape(-1).astype(np.int32)
    keep = np.isfinite(points).all(axis=-1) & (occ >= np.percentile(occ, 10))
    return {
        "points": points[keep],
        "rgb": rgb[keep],
        "occupancy": occ[keep],
        "body_part_id": part[keep],
    }


def select_environment(env_points: np.ndarray, env_rgb: np.ndarray, human: np.ndarray, max_env: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(human, axis=0)
    span = np.percentile(human, 95, axis=0) - np.percentile(human, 5, axis=0)
    span = np.maximum(span, np.array([0.08, 0.08, 0.08], dtype=np.float32))
    rel = np.abs(env_points - center)
    near = (rel[:, 0] < span[0] * 4.5) & (rel[:, 1] < span[1] * 5.0) & (rel[:, 2] < span[2] * 3.0)
    floor = env_points[:, 1] < np.percentile(human[:, 1], 12)
    idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1) & (near | floor))
    if idx.size == 0:
        idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1))
    if idx.size > max_env:
        idx = np.random.default_rng(seed).choice(idx, size=max_env, replace=False)
    return env_points[idx].astype(np.float32), env_rgb[idx].astype(np.float32)


def compose_scene(pred: dict[str, np.ndarray], env_points: np.ndarray, env_rgb: np.ndarray, max_env: int, seed: int) -> dict[str, np.ndarray]:
    env_p, env_r = select_environment(env_points, env_rgb, pred["points"], max_env, seed)
    full_points = np.concatenate([env_p, pred["points"]], axis=0).astype(np.float32)
    full_rgb = np.concatenate([env_r, pred["rgb"]], axis=0).astype(np.float32)
    is_human = np.concatenate([np.zeros(env_p.shape[0], dtype=np.uint8), np.ones(pred["points"].shape[0], dtype=np.uint8)])
    return {
        "full_points": full_points,
        "full_rgb": full_rgb,
        "human_points": pred["points"],
        "human_rgb": pred["rgb"],
        "environment_points": env_p,
        "environment_rgb": env_r,
        "is_human": is_human,
        "body_part_id": pred["body_part_id"],
    }


def scene_metrics(scene: dict[str, np.ndarray]) -> dict[str, Any]:
    human = scene["human_points"]
    env = scene["environment_points"]
    span = np.percentile(human, 95, axis=0) - np.percentile(human, 5, axis=0)
    thickness_ratio = float(min(span) / max(max(span), 1.0e-6))
    part_ids = scene["body_part_id"]
    return {
        "human_point_count": int(human.shape[0]),
        "environment_point_count": int(env.shape[0]),
        "environment_ratio": float(env.shape[0] / max(human.shape[0] + env.shape[0], 1)),
        "human_span_x": float(span[0]),
        "human_span_y": float(span[1]),
        "human_span_z": float(span[2]),
        "human_thickness_ratio": thickness_ratio,
        "body_part_count": int(len(np.unique(part_ids))),
    }


def make_board(true_scene: dict[str, np.ndarray], controls: dict[str, dict[str, np.ndarray]], row: dict[str, Any]) -> None:
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V518 canonical SMPL-X surfel/graph representation repair smoke", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Representation switch after V517 blob failure. Not promoted; V50R2 is not used at inference.", fill=(140, 0, 0), font=font)
    panels = [
        ("true full-scene XZ", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 340), (0, 2))),
        ("true full-scene XY", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 340), (0, 1))),
        ("true full-scene YZ", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 340), (1, 2))),
        ("true human ROI XZ", render_points(true_scene["human_points"], true_scene["human_rgb"], (580, 340), (0, 2))),
        ("true human ROI XY", render_points(true_scene["human_points"], true_scene["human_rgb"], (580, 340), (0, 1))),
        ("no graph control ROI XY", render_points(controls["no_smpl_graph"]["human_points"], controls["no_smpl_graph"]["human_rgb"], (580, 340), (0, 1))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 400
        board.paste(image, (x, y + 24))
        draw.rectangle([x, y + 24, x + 580, y + 364], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 895
    for line in [
        f"status: {row['status']}",
        f"human_points: {row['human_point_count']} | environment_points: {row['environment_point_count']} | env_ratio: {row['environment_ratio']:.3f}",
        f"body_part_count: {row['body_part_count']} | thickness_ratio: {row['human_thickness_ratio']:.4f}",
        f"no_teacher_copy: {row['no_teacher_copy']} | model_owned: True | mentor_ready: False",
        "decision: representation switch executed; next route must train/adjudicate against controls before V509/V512.",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 30
    board.save(BOARD)


def make_controls_board(scenes: dict[str, dict[str, np.ndarray]]) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1020), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V518 controls: canonical graph true vs no-SMPL-graph / weak-semantic / VGGT-visible baseline", fill=(0, 0, 0), font=font)
    order = [
        ("true canonical graph", "true"),
        ("no SMPL graph", "no_smpl_graph"),
        ("weak semantic", "weak_semantic"),
        ("VGGT visible baseline", "vggt_visible_baseline"),
    ]
    for i, (label, key) in enumerate(order):
        x = 18 + (i % 2) * 890
        y = 60 + (i // 2) * 450
        scene = scenes[key]
        image = render_points(scene["full_points"], scene["full_rgb"], (860, 390), (0, 2))
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 860, y + 416], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    draw.text((18, 960), "Not pass evidence: controls are generated for adjudication; visual/control separation still requires a trained V519 route.", fill=(140, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def write_route_file() -> None:
    ROUTE.parent.mkdir(parents=True, exist_ok=True)
    ROUTE.write_text(
        """# V518 Auto-Evolved Canonical Surfel/Graph Repair Route

Trigger:
- V517 produced a full-scene model-owned candidate but V512 failed closed because the human remained blob-like and below the V50R2 visual floor.

Route:
- switch from free-point clarity composition to canonical SMPL-X graph/surfel representation;
- keep VGGT observation as scene/frame/RGB context;
- keep V50R2 as visual floor / teacher / reference only;
- generate same-scene controls: no SMPL graph, weak semantic, VGGT visible baseline;
- no promotion, no registry, no V50/V50R2 modification.

Next required step:
- train/adjudicate V519 with VGGT feature sampling on surfels, local body-part heads, anti-blob/anti-sheet losses, and strict controls before V509/V512 can pass.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-human-samples", type=int, default=12000)
    parser.add_argument("--max-env-points", type=int, default=9000)
    parser.add_argument("--nodes-per-part", type=int, default=192)
    parser.add_argument("--seed", type=int, default=518)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    batch, meta = load_observation_inputs(max_human_per_view=4096, max_env_per_view=4096, seed=args.seed)
    v360_batch = make_v360_batch(batch, sample_limit=args.max_human_samples, seed=args.seed)
    model = CanonicalSMPLXGraphVolumeStudent(V360Config(nodes_per_part=args.nodes_per_part, samples_per_node=9))
    model.eval()
    with torch.no_grad():
        pred_true = flatten_v360(model(v360_batch, semantic_scale=1.0))
        pred_no_graph = flatten_v360(model(v360_batch, semantic_scale=0.0))
        pred_weak = flatten_v360(model(v360_batch, semantic_scale=0.35))

    raw_points = batch["vggt_world_points"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    raw_rgb = batch["vggt_rgb"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    base_points, base_rgb = sample_rows(raw_points, raw_rgb, pred_true["points"].shape[0], args.seed + 20)
    pred_base = {
        "points": base_points,
        "rgb": base_rgb,
        "occupancy": np.ones(base_points.shape[0], dtype=np.float32),
        "body_part_id": np.zeros(base_points.shape[0], dtype=np.int32),
    }

    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    scenes = {
        "true": compose_scene(pred_true, env_points, env_rgb, args.max_env_points, args.seed),
        "no_smpl_graph": compose_scene(pred_no_graph, env_points, env_rgb, args.max_env_points, args.seed + 1),
        "weak_semantic": compose_scene(pred_weak, env_points, env_rgb, args.max_env_points, args.seed + 2),
        "vggt_visible_baseline": compose_scene(pred_base, env_points, env_rgb, args.max_env_points, args.seed + 3),
    }

    candidate_npz = OUT / "v518_canonical_surfel_graph_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        full_scene_points=scenes["true"]["full_points"],
        full_scene_rgb=scenes["true"]["full_rgb"],
        human_points=scenes["true"]["human_points"],
        human_rgb=scenes["true"]["human_rgb"],
        environment_points=scenes["true"]["environment_points"],
        environment_rgb=scenes["true"]["environment_rgb"],
        is_human=scenes["true"]["is_human"],
        body_part_id=scenes["true"]["body_part_id"],
        model_owned_student_output=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v518_canonical_surfel_graph_repair_smoke"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V518_teacher_copy_check", "created_at": now(), **copy_result})

    for key, scene in scenes.items():
        write_ply(OUT / f"v518_{key}_full_scene_rgb.ply", scene["full_points"], scene["full_rgb"])
        write_ply(OUT / f"v518_{key}_human_only_rgb.ply", scene["human_points"], scene["human_rgb"])

    metrics = scene_metrics(scenes["true"])
    no_teacher_copy = not bool(copy_result.get("leak_detected", False))
    row = {
        "status": "V518_CANONICAL_SURFEL_GRAPH_REPRESENTATION_REPAIR_SMOKE_COMPLETE_NOT_PROMOTED",
        "human_point_count": metrics["human_point_count"],
        "environment_point_count": metrics["environment_point_count"],
        "environment_ratio": metrics["environment_ratio"],
        "human_span_x": metrics["human_span_x"],
        "human_span_y": metrics["human_span_y"],
        "human_span_z": metrics["human_span_z"],
        "human_thickness_ratio": metrics["human_thickness_ratio"],
        "body_part_count": metrics["body_part_count"],
        "no_teacher_copy": no_teacher_copy,
        "model_owned_student_output": True,
        "mentor_ready": False,
        "accepted_for_v509": False,
    }
    write_csv(METRICS, [row])
    make_board(scenes["true"], {"no_smpl_graph": scenes["no_smpl_graph"]}, row)
    make_controls_board(scenes)
    write_route_file()

    payload = {
        "task": "V518_canonical_surfel_graph_repair_smoke",
        "status": row["status"],
        "created_at": now(),
        "repo": str(ROOT),
        "route_file": str(ROUTE),
        "candidate_npz": str(candidate_npz),
        "true_full_scene_ply": str(OUT / "v518_true_full_scene_rgb.ply"),
        "controls": {
            key: str(OUT / f"v518_{key}_full_scene_rgb.ply")
            for key in ["no_smpl_graph", "weak_semantic", "vggt_visible_baseline"]
        },
        "boards": {
            "main": str(BOARD),
            "controls": str(CONTROLS_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "canonical_smplx_graph_representation": True,
            "v50r2_used_at_inference": False,
            "v50r2_used_for_visual_floor_reference_only": True,
            "kinect_depth_used_at_inference": False,
        },
        "gates": {
            "representation_switch_executed": True,
            "canonical_graph_body_parts_present": metrics["body_part_count"] >= 8,
            "partial_environment_visible": 0.08 <= metrics["environment_ratio"] <= 0.45,
            "no_teacher_copy": no_teacher_copy,
            "same_scene_controls_generated": True,
            "accepted_for_v509": False,
            "manual_visual_gate_pass": False,
            "mentor_ready": False,
            "not_promoted": True,
            "auto_evolve_required": True,
        },
        "metrics": row,
        "decision": (
            "V518 executes the required representation switch after V517 visual failure. "
            "It is a smoke/repair candidate only: train and adjudicate V519 before any V509/V512 pass claim."
        ),
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
            "main_board": sha256(BOARD),
            "controls_board": sha256(CONTROLS_BOARD),
            "route_file": sha256(ROUTE),
        },
    }
    write_json(DECISION, payload)
    print(json.dumps({"status": row["status"], "decision": str(DECISION), "board": str(BOARD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
