from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUT_ROOT = OUTPUT / "V19200000000000000000_upright_pose_frame_layout"
TRUE_CONFIG = "upright_pose_frame_true"
V190_CONFIG = "pose_frame_occupancy_true"
V187_CONFIG = "visible_anchor_canonical_surfel_true"
V186_CONFIG = "part_coverage_canonical_surfel_true"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from models.v184_canonical_surfel_graph_occupancy_student import CanonicalSurfelGraphOccupancyStudent  # noqa: E402
from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    compose,
    cross_panel,
    load_npz,
    read_manifest,
    render_panel,
    rotation_matrix,
    select_device,
    write_ply,
)
from tools.V18000_adjacency_aware_collision_metric import adjacency_collision_metric_v4  # noqa: E402
from tools.V18700_visible_anchor_canonical_surfel_training import build_batch  # noqa: E402
from tools.V19000_pose_frame_occupancy_repair import decode_pose_frame, pose_frame_losses  # noqa: E402


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
        writer = csv.DictWriter(f, fieldnames=fields or ["case"])
        writer.writeheader()
        writer.writerows(rows)


def unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else fallback.astype(np.float32)


def body_frame(points: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pts = np.asarray(points, dtype=np.float32)
    part = np.asarray(body).astype(int)
    center = np.median(pts, axis=0).astype(np.float32)

    def part_center(ids: list[int]) -> np.ndarray:
        mask = np.isin(part, ids)
        if int(mask.sum()) < 16:
            return center
        return np.median(pts[mask], axis=0).astype(np.float32)

    upper = part_center([0, 1])
    lower = part_center([4, 5, 6, 7])
    y_axis = unit(upper - lower, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    if float(np.dot(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))) < 0:
        y_axis = -y_axis
    left = part_center([2, 4, 6])
    right = part_center([3, 5, 7])
    x_axis = right - left
    x_axis = x_axis - float(np.dot(x_axis, y_axis)) * y_axis
    if float(np.linalg.norm(x_axis)) <= 1e-8:
        cov = (pts - center).T @ (pts - center) / max(1, len(pts) - 1)
        vals, vecs = np.linalg.eigh(cov)
        x_axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
        x_axis = x_axis - float(np.dot(x_axis, y_axis)) * y_axis
    x_axis = unit(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    if float(np.dot(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))) < 0:
        x_axis = -x_axis
    z_axis = unit(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    x_axis = unit(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    info = {
        "axis_y_dot_world_y": float(np.dot(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))),
        "axis_x_dot_world_x": float(np.dot(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))),
        "axis_z_dot_world_z": float(np.dot(z_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))),
    }
    return center, basis, info


def to_local(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return ((np.asarray(points, dtype=np.float32) - center[None]) @ basis).astype(np.float32)


def vec_to_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(vectors, dtype=np.float32) @ basis).astype(np.float32)


def to_world(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float32) @ basis.T + center[None]).astype(np.float32)


def localize_batch(row: dict[str, str], max_points: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    batch, target, aux = build_batch(row, max_points=max_points, device=device)
    center, basis, frame_info = body_frame(aux["surfel_xyz"], aux["body"])
    idx = aux["idx"]
    local_all = to_local(aux["surfel_xyz"], center, basis)
    local_sel = local_all[idx]
    batch["surfel_xyz"] = torch.from_numpy(local_sel[None]).to(device)
    batch["surfel_normal"] = torch.from_numpy(vec_to_local(batch["surfel_normal"][0].detach().cpu().numpy(), basis)[None]).to(device)
    batch["surfel_tangent"] = torch.from_numpy(vec_to_local(batch["surfel_tangent"][0].detach().cpu().numpy(), basis)[None]).to(device)
    aux = dict(aux)
    aux["surfel_xyz"] = local_all
    aux["world_surfel_xyz"] = aux.get("world_surfel_xyz", None) if "world_surfel_xyz" in aux else None
    aux["frame_center"] = center
    aux["frame_basis"] = basis
    aux["frame_info"] = frame_info
    aux["environment_points_local"] = to_local(aux["environment_points"], center, basis)
    return batch, target, aux


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = CanonicalSurfelGraphOccupancyStudent().to(device)
    batch, target, aux = localize_batch(row, max_points=max_points, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = pose_frame_losses(out, target, batch)
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "pose_width": float(losses["pose_width"].detach().cpu()),
                    "normal_align": float(losses["normal_align"].detach().cpu()),
                    "completeness": float(losses["completeness"].detach().cpu()),
                    "part_occupancy": float(losses["part_occupancy"].detach().cpu()),
                    "floating": float(losses["floating"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human_local, human_rgb, body = decode_pose_frame(out, batch, target, aux)
    center = aux["frame_center"]
    basis = aux["frame_basis"]
    human_world = to_world(human_local, center, basis)
    env_world = aux["environment_points"]
    env_rgb = aux["environment_rgb"]
    full_world = np.concatenate([human_world, env_world], axis=0)
    full_rgb = np.concatenate([human_rgb, env_rgb], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / TRUE_CONFIG)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human_world,
        human_rgb=human_rgb,
        environment_points=env_world,
        environment_rgb=env_rgb,
        full_scene_points=full_world,
        full_scene_rgb=full_rgb,
        render_frame_human_points=human_local,
        render_frame_environment_points=aux["environment_points_local"],
        frame_center=center,
        frame_basis=basis,
        body_part_id=body,
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(row["case"]),
        upright_pose_frame_layout=np.array(True),
        feature_bank_fallback_used=np.array(bool(aux["feature_bank_fallback_used"])),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full_world, full_rgb)
    metric, _pairs = adjacency_collision_metric_v4(human_world, body)
    return {
        "case": row["case"],
        "config": TRUE_CONFIG,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "feature_bank_fallback_used": bool(aux["feature_bank_fallback_used"]),
        **aux["frame_info"],
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config == V190_CONFIG:
        return OUTPUT / "V19000000000000000000_pose_frame_occupancy_repair" / case / config / "predictions.npz"
    if config == V187_CONFIG:
        return OUTPUT / "V18700000000000000000_visible_anchor_canonical_surfel_training" / case / config / "predictions.npz"
    if config == V186_CONFIG:
        return OUTPUT / "V18600000000000000000_part_coverage_canonical_surfel_training" / case / config / "predictions.npz"
    return OUTPUT / "V1400000000000000000_learned_residual_matrix" / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    render = np.asarray(pred["render_frame_human_points"], dtype=np.float32) if "render_frame_human_points" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body, render


def render_upright_panel(case: str, config: str, title: str) -> Image.Image | None:
    path = config_path(case, config)
    if not path.exists():
        return None
    pts, rgb, _body, render = load_config(case, config)
    # True V192 includes a training render frame. Other configs stay in their
    # own world frame so this board is diagnostic rather than success evidence.
    return render_panel(render if render is not None else pts, rgb, title, rot=rotation_matrix(0, 0))


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V190_CONFIG, V187_CONFIG, V186_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
    for row in rows:
        case = row["case"]
        case_scores: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            path = config_path(case, cfg)
            if not path.exists():
                continue
            pts, _rgb, body, _render = load_config(case, cfg)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            score_rows.append({"case": case, "config": cfg, **metric})
            case_scores[cfg] = metric
        true = case_scores.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_upright_pose_frame_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in [V190_CONFIG, V187_CONFIG, V186_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
            if cfg in case_scores and float(case_scores[cfg]["combined_topology_volume_score_v4"]) >= true_score * 0.96:
                failures.append(
                    {
                        "case": case,
                        "reason": "control_or_prior_close_or_better_v4",
                        "control": cfg,
                        "true_score": true_score,
                        "control_score": float(case_scores[cfg]["combined_topology_volume_score_v4"]),
                    }
                )
    write_csv(REPORTS / "V19200000000000000000_upright_pose_frame_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in configs:
        panel = render_upright_panel(first, cfg, f"{first} {cfg.replace('_', ' ')}") if first else None
        if panel is not None:
            panels.append(panel)
    if panels:
        compose(panels, 3, BOARDS / "V19200000000000000000_upright_pose_frame_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body, render = load_config(first, TRUE_CONFIG)
        plot_pts = render if render is not None else pts
        turn = [
            render_panel(plot_pts, rgb, f"{first} upright {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        turn.append(cross_panel(plot_pts, f"{first} upright cross-section"))
        compose(turn, 3, BOARDS / "V19200000000000000000_upright_pose_frame_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V19200_UPRIGHT_POSE_FRAME_FAIL_CLOSED_CONTINUE" if failures else "V19200_UPRIGHT_POSE_FRAME_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V19200000000000000000_upright_pose_frame_scores.csv"),
        "board": str(BOARDS / "V19200000000000000000_upright_pose_frame_board.png"),
        "turntable_cross_section": str(BOARDS / "V19200000000000000000_upright_pose_frame_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V192 trains and renders in a body-local upright frame, then transforms model-owned points back to world coordinates. It remains fail-closed unless full-scene mentor visuals and same-scene controls pass.",
    }
    write_json(REPORTS / "V19200000000000000000_upright_pose_frame_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V19200_STEPS", "40"))
    max_points = int(os.environ.get("V19200_MAX_POINTS", "8192"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V19200000000000000000_upright_pose_frame_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V19200000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V19200_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
