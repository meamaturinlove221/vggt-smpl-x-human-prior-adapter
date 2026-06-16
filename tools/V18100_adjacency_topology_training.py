from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
TARGET_ROOT = OUTPUT / "V17100000000000000000_part_graph_cross_section_targets"
OUT_ROOT = OUTPUT / "V18100000000000000000_adjacency_topology_training"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
V173_ROOT = OUTPUT / "V17300000000000000000_multishell_topology_decoder_training"
V179_ROOT = OUTPUT / "V17900000000000000000_collision_aware_topology_repair"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
TRUE_CONFIG = "adjacency_topology_true"
V173_CONFIG = "multishell_topology_decoder_true"
V179_CONFIG = "collision_aware_topology_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]

from models.v135_anti_billboard_topology_volume_student import (  # noqa: E402
    AntiBillboardTopologyVolumeConfig,
    AntiBillboardTopologyVolumeStudent,
)
from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    build_batch,
    balanced_multishell_decode,
    compose,
    cross_panel,
    hard_control_scores,
    load_npz,
    read_manifest,
    render_panel,
    repo_path,
    rotation_matrix,
    select_device,
    training_losses,
    write_ply,
)
from tools.V18000_adjacency_aware_collision_metric import (  # noqa: E402
    ADJACENT_EDGES,
    DISTANT_PAIR_WEIGHTS,
    adjacency_collision_metric_v4,
)


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


def pairwise_adjacency_losses(points: torch.Tensor, body_part: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable adjacency-aware topology loss on a selected point set."""
    parts = [int(x) for x in torch.unique(body_part).detach().cpu().tolist()]
    centers: dict[int, torch.Tensor] = {}
    spreads: dict[int, torch.Tensor] = {}
    for part in parts:
        mask = body_part == part
        if int(mask.sum()) < 20:
            continue
        p = points[mask]
        centers[part] = p.mean(dim=0)
        spreads[part] = torch.sqrt(torch.var(p, dim=0, unbiased=False) + 1e-6)
    valid_contact_terms: list[torch.Tensor] = []
    valid_overmerge_terms: list[torch.Tensor] = []
    invalid_terms: list[torch.Tensor] = []
    for a, ca in centers.items():
        for b, cb in centers.items():
            if b <= a:
                continue
            pair = tuple(sorted((a, b)))
            dist = torch.norm(ca - cb)
            scale = 0.5 * (torch.norm(spreads[a]) + torch.norm(spreads[b])) + 1e-6
            normalized = dist / scale
            if pair in ADJACENT_EDGES:
                # Adjacent parts should be close enough to stay connected, but
                # not so merged that topology becomes a single textured sheet.
                valid_contact_terms.append(torch.relu(normalized - 2.15))
                valid_overmerge_terms.append(torch.relu(0.65 - normalized))
            elif pair in DISTANT_PAIR_WEIGHTS:
                weight = float(DISTANT_PAIR_WEIGHTS[pair])
                invalid_terms.append(torch.relu(2.55 - normalized) * weight)
    device = points.device
    zero = torch.zeros((), device=device)
    return {
        "valid_contact_loss": torch.stack(valid_contact_terms).mean() if valid_contact_terms else zero,
        "valid_overmerge_loss": torch.stack(valid_overmerge_terms).mean() if valid_overmerge_terms else zero,
        "invalid_pair_separation_loss": torch.stack(invalid_terms).mean() if invalid_terms else zero,
    }


def adjacency_training_losses(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    control_scores: dict[str, float],
    body_part: torch.Tensor,
) -> dict[str, torch.Tensor]:
    losses = training_losses(out, targets, control_scores)
    shells = torch.stack(
        [out["student_xyz"], out["front_shell"], out["back_shell"], out["left_shell"], out["right_shell"]],
        dim=2,
    )[0]
    shell_points = shells.reshape(-1, 3)
    shell_body = body_part[0].repeat_interleave(shells.shape[1])
    pair_losses = pairwise_adjacency_losses(shell_points, shell_body)
    losses.update(pair_losses)
    losses["total"] = (
        losses["total"]
        + 0.55 * pair_losses["invalid_pair_separation_loss"]
        + 0.32 * pair_losses["valid_contact_loss"]
        + 0.22 * pair_losses["valid_overmerge_loss"]
    )
    return losses


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = AntiBillboardTopologyVolumeStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    control_scores = hard_control_scores(row["case"])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = adjacency_training_losses(out, targets, control_scores, batch["body_part_id"])
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "invalid_pair": float(losses["invalid_pair_separation_loss"].detach().cpu()),
                    "valid_contact": float(losses["valid_contact_loss"].detach().cpu()),
                    "valid_overmerge": float(losses["valid_overmerge_loss"].detach().cpu()),
                    "shell_width": float(losses["shell_width"].detach().cpu()),
                    "preserve": float(losses["preserve"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = balanced_multishell_decode(out, aux, targets)
    full = np.concatenate([human, aux["environment_points"]], axis=0)
    full_rgb = np.concatenate([human_rgb, aux["environment_rgb"]], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / TRUE_CONFIG)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=human_rgb,
        environment_points=aux["environment_points"],
        environment_rgb=aux["environment_rgb"],
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        body_part_id=body,
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(row["case"]),
        adjacency_aware_topology_training=np.array(True),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric, _pairs = adjacency_collision_metric_v4(human, body)
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
        **{f"control_{k}_score": v for k, v in control_scores.items()},
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config == V173_CONFIG:
        return V173_ROOT / case / config / "predictions.npz"
    if config == V179_CONFIG:
        return V179_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], manifest: list[dict[str, Any]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V173_CONFIG, V179_CONFIG, *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_scores: dict[str, dict[str, Any]] = {}
        for config in configs:
            path = config_path(case, config)
            if not path.exists():
                continue
            pts, _rgb, body = load_config(case, config)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            score_rows.append({"case": case, "config": config, **metric})
            case_scores[config] = metric
        true = case_scores.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_adjacency_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        if float(true["invalid_overlap_penalty"]) > 0.58:
            failures.append({"case": case, "reason": "invalid_distant_overlap_too_high", "penalty": float(true["invalid_overlap_penalty"])})
        for config in [V173_CONFIG, V179_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
            if config not in case_scores:
                continue
            score = float(case_scores[config]["combined_topology_volume_score_v4"])
            if score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_or_prior_close_or_better_v4", "control": config, "true_score": true_score, "control_score": score})
    write_csv(REPORTS / "V18100000000000000000_adjacency_training_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for config in [TRUE_CONFIG, V173_CONFIG, V179_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature"]:
        path = config_path(first, config)
        if path.exists():
            pts, rgb, _body = load_config(first, config)
            panels.append(render_panel(pts, rgb, f"{first} {config.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V18100000000000000000_adjacency_training_board.png")
    turn: list[Image.Image] = []
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        for label, rot in [
            ("front", rotation_matrix(0, 0)),
            ("back", rotation_matrix(180, 0)),
            ("left", rotation_matrix(-90, 0)),
            ("right", rotation_matrix(90, 0)),
            ("oblique", rotation_matrix(-30, 61)),
        ]:
            turn.append(render_panel(pts, rgb, f"{first} adjacency true {label}", rot=rot))
        turn.append(cross_panel(pts, f"{first} adjacency true cross-section"))
        compose(turn, 3, BOARDS / "V18100000000000000000_adjacency_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V18100_ADJACENCY_TOPOLOGY_TRAINING_FAIL_CLOSED_CONTINUE" if failures else "V18100_ADJACENCY_TOPOLOGY_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "cases": [r["case"] for r in rows],
        "failures": failures,
        "training_manifest": str(REPORTS / "V18100000000000000000_training_manifest.csv"),
        "score_csv": str(REPORTS / "V18100000000000000000_adjacency_training_scores.csv"),
        "board": str(BOARDS / "V18100000000000000000_adjacency_training_board.png"),
        "turntable_cross_section": str(BOARDS / "V18100000000000000000_adjacency_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V181 trains the multishell student with adjacency-aware topology losses from V180. It remains fail-closed unless it beats V173, controls, and the mentor visual gate.",
    }
    write_json(REPORTS / "V18100000000000000000_training_decision.json", decision)
    route = f"""# V181 Adjacency Topology Training Route

Created: {created_at}

Status: `{decision["status"]}`

This route is not mentor-ready. It uses V180 adjacency-aware collision analysis as a training loss, but the mentor gate remains the original full-scene RGB point-cloud visual comparison.

## Result

- Mentor ready: `false`
- External hard block: `false`
- Failure count: `{len(failures)}`

## Interpretation

If V181 improves invalid distant overlap but still fails, the next repair must move the adjacency/topology signal into the model representation itself: part-pair exclusion heads, per-part occupancy fields, and semantic graph decoding. Do not return to viewer, thickness-only, shell-offset, or procedural occupancy fixes.
"""
    (REPORTS / "V18100000000000000000_route_state.md").write_text(route, encoding="utf-8")


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V18100_STEPS", "40"))
    max_points = int(os.environ.get("V18100_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V18100000000000000000_training_manifest.csv", manifest)
    compare(rows, manifest, created_at)
    write_json(
        REPORTS / "V18100000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V18100_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
