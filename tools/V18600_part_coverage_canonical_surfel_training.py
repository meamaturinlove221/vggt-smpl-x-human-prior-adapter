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
import torch.nn.functional as F
from PIL import Image


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
OUT_ROOT = OUTPUT / "V18600000000000000000_part_coverage_canonical_surfel_training"
SMPL_BANK = OUTPUT / "V9500000000000000_smpl_feature_bank_v4"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
V173_ROOT = OUTPUT / "V17300000000000000000_multishell_topology_decoder_training"
V183_ROOT = OUTPUT / "V18300000000000000000_part_pair_exclusion_training"
TRUE_CONFIG = "part_coverage_canonical_surfel_true"
V185_CONFIG = "canonical_surfel_graph_true"
V173_CONFIG = "multishell_topology_decoder_true"
V183_CONFIG = "part_pair_exclusion_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
PART_COUNT = 8

from models.v184_canonical_surfel_graph_occupancy_student import (  # noqa: E402
    CanonicalSurfelGraphOccupancyConfig,
    CanonicalSurfelGraphOccupancyStudent,
)
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


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    if p.exists():
        return p
    marker = "vggt-canonical-surfel-adapter/"
    text = str(value).replace("\\", "/")
    if marker in text:
        mapped = REPO / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    return p


def pad_feature(arr: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] >= width:
        return arr[:, :width]
    out = np.zeros((arr.shape[0], width), dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def nearest_body_mask(source: np.ndarray, target: np.ndarray, max_points: int = 32000) -> np.ndarray:
    src = source.astype(np.float32)
    tgt = target.astype(np.float32)
    if len(tgt) > max_points:
        rng = np.random.default_rng(185)
        tgt = tgt[rng.choice(len(tgt), max_points, replace=False)]
    # Chunked nearest-neighbor distance without sklearn dependency.
    nearest = np.full(len(src), np.inf, dtype=np.float32)
    for start in range(0, len(tgt), 2048):
        chunk = tgt[start : start + 2048]
        d2 = ((src[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        nearest = np.minimum(nearest, d2.min(axis=1))
    thresh = float(np.quantile(nearest, 0.72))
    return nearest <= max(thresh, 1e-8)


def part_balanced_indices(priority: np.ndarray, body: np.ndarray, max_points: int) -> np.ndarray:
    """Select surfels without dropping small body parts.

    V185 selected by global priority and produced only 4-5 visible parts. This
    keeps the same model-owned inference path while preventing the decoder
    from silently removing head/arm/foot support before training starts.
    """
    priority = np.asarray(priority, dtype=np.float32)
    body = np.asarray(body).astype(np.int64)
    selected: list[np.ndarray] = []
    used = np.zeros(len(priority), dtype=bool)
    quota = max(256, int(max_points * 0.085))
    for part in range(PART_COUNT):
        ids = np.flatnonzero(body == part)
        if len(ids) == 0:
            continue
        take = min(len(ids), quota)
        order = ids[np.argsort(-priority[ids])[:take]]
        selected.append(order)
        used[order] = True
    remaining = max_points - int(used.sum())
    if remaining > 0:
        rest = np.flatnonzero(~used)
        if len(rest):
            order = rest[np.argsort(-priority[rest])[:remaining]]
            selected.append(order)
            used[order] = True
    idx = np.flatnonzero(used)
    if len(idx) > max_points:
        idx = idx[np.argsort(-priority[idx])[:max_points]]
    return idx.astype(np.int64)


def build_batch(row: dict[str, str], max_points: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    case = row["case"]
    cfg = CanonicalSurfelGraphOccupancyConfig()
    bank = load_npz(SMPL_BANK / case / "smpl_feature_bank_v4.npz")
    graph = load_npz(repo_path(row["graph_path"]))
    baseline = load_npz(repo_path(row["baseline_path"]))
    visible = load_npz(repo_path(row["visible_target_path"]))
    surfel_xyz = np.asarray(bank["posed_world_xyz"], dtype=np.float32)
    if "world_points" in bank:
        surfel_xyz = 0.5 * surfel_xyz + 0.5 * np.asarray(bank["world_points"], dtype=np.float32)
    rgb = as_rgb(bank["rgb"]).astype(np.float32) / 255.0
    conf = np.asarray(bank["confidence"], dtype=np.float32)
    normal = np.asarray(bank["local_normal"], dtype=np.float32)
    tangent = np.asarray(bank["local_tangent"], dtype=np.float32)
    body = np.asarray(bank["body_part_id"], dtype=np.int64)
    visibility = np.asarray(bank["visibility"], dtype=np.float32)
    visible_points = np.asarray(visible["world_points"] if "world_points" in visible else visible["points"], dtype=np.float32)
    occupancy = nearest_body_mask(surfel_xyz, visible_points)
    weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    # Bind graph weak score to surfels by index where possible; both are body aligned.
    if len(weak) != len(surfel_xyz):
        weak = np.interp(np.linspace(0, len(weak) - 1, len(surfel_xyz)), np.arange(len(weak)), weak).astype(np.float32)
    priority = occupancy.astype(np.float32) * 7.0 + visibility * 2.5 + conf * 1.5 + weak * 1.2
    idx = part_balanced_indices(priority, body, min(max_points, len(priority)))
    surfel_features = np.concatenate(
        [
            pad_feature(bank.get("joint_relative_coordinates", np.zeros((len(surfel_xyz), 1), dtype=np.float32)), 6),
            pad_feature(bank.get("skinning_weights", np.zeros((len(surfel_xyz), 1), dtype=np.float32)), 8),
            pad_feature(bank.get("voxel_coords_32", np.zeros((len(surfel_xyz), 3), dtype=np.float32)) / 32.0, 3),
            conf[:, None],
            visibility[:, None],
            weak[:, None],
        ],
        axis=1,
    )
    vggt_features = np.concatenate([rgb, conf[:, None], visibility[:, None], weak[:, None]], axis=1)
    batch = {
        "surfel_xyz": torch.from_numpy(surfel_xyz[idx][None]).to(device),
        "surfel_rgb": torch.from_numpy(rgb[idx][None]).to(device),
        "surfel_normal": torch.from_numpy(normal[idx][None]).to(device),
        "surfel_tangent": torch.from_numpy(tangent[idx][None]).to(device),
        "vggt_confidence": torch.from_numpy(conf[idx][None]).to(device),
        "surfel_features": torch.from_numpy(pad_feature(surfel_features, cfg.surfel_feature_dim)[idx][None]).to(device),
        "vggt_features": torch.from_numpy(pad_feature(vggt_features, cfg.vggt_feature_dim)[idx][None]).to(device),
        "vggt_token_context": torch.zeros(1, cfg.token_dim, device=device),
        "body_part_id": torch.from_numpy(body[idx][None]).to(device),
    }
    target = {
        "occupancy": torch.from_numpy(occupancy[idx].astype(np.float32)[None, :, None]).to(device),
        "visibility": torch.from_numpy(visibility[idx].astype(np.float32)[None, :, None]).to(device),
        "body_part": torch.from_numpy(body[idx][None]).to(device),
        "weak": torch.from_numpy(np.maximum(weak[idx], occupancy[idx].astype(np.float32))[None, :, None].astype(np.float32)).to(device),
        "part_present": torch.ones(1, PART_COUNT, device=device),
    }
    aux = {
        "case": case,
        "idx": idx,
        "surfel_xyz": surfel_xyz,
        "rgb": as_rgb(bank["rgb"]),
        "conf": conf,
        "visibility": visibility,
        "body": body,
        "occupancy": occupancy,
        "priority": priority,
        "selected_part_count": int(len(np.unique(body[idx]))),
        "source_part_count": int(len(np.unique(body))),
        "environment_points": np.asarray(baseline["environment_points"], dtype=np.float32),
        "environment_rgb": as_rgb(baseline["environment_rgb"]),
    }
    return batch, target, aux


def graph_losses(out: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    occ_loss = F.binary_cross_entropy_with_logits(out["occupancy_logits"], target["occupancy"])
    vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logits"], target["visibility"].clamp(0, 1))
    part_target = F.one_hot(target["body_part"].long().clamp(0, 7), num_classes=8).float()
    part_loss = F.binary_cross_entropy_with_logits(out["part_graph_logits"], part_target)
    exclusion_target = 1.0 - part_target
    exclusion_loss = F.binary_cross_entropy_with_logits(out["part_exclusion_logits"], exclusion_target)
    thickness = out["thickness"].mean()
    thickness_loss = F.relu(0.020 - thickness)
    residual_loss = (out["residual_xyz"].square() * (1.0 - target["weak"])).mean()
    # Encourage occupied surfels to maintain front/back separation.
    shell_sep = (out["front_shell"] - out["back_shell"]).norm(dim=-1, keepdim=True)
    shell_loss = (F.relu(0.050 - shell_sep) * target["occupancy"]).mean()
    body = target["body_part"].long().clamp(0, PART_COUNT - 1)
    occ = out["occupancy"]
    coverage_terms: list[Tensor] = []
    for part in range(PART_COUNT):
        mask = body == part
        if bool(mask.any()):
            part_occ = occ[mask.unsqueeze(-1)]
            coverage_terms.append(F.relu(0.38 - part_occ.mean()))
    coverage_loss = torch.stack(coverage_terms).mean() if coverage_terms else occ.mean() * 0.0
    target_density = target["occupancy"].clamp(0, 1)
    weak_density = torch.maximum(target_density, target["weak"].clamp(0, 1))
    occupancy_completeness_loss = F.relu(0.44 - (occ * weak_density).mean())
    total = (
        occ_loss
        + 0.45 * vis_loss
        + 0.24 * part_loss
        + 0.18 * exclusion_loss
        + 0.32 * thickness_loss
        + 0.18 * residual_loss
        + 0.36 * shell_loss
        + 0.42 * coverage_loss
        + 0.24 * occupancy_completeness_loss
    )
    return {
        "total": total,
        "occupancy": occ_loss,
        "visibility": vis_loss,
        "part": part_loss,
        "exclusion": exclusion_loss,
        "thickness": thickness_loss,
        "residual": residual_loss,
        "shell": shell_loss,
        "coverage": coverage_loss,
        "occupancy_completeness": occupancy_completeness_loss,
    }


def decode(out: dict[str, torch.Tensor], aux: dict[str, np.ndarray], target: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ = out["occupancy"][0, :, 0].detach().cpu().numpy()
    xyz = out["student_xyz"][0].detach().cpu().numpy()
    front = out["front_shell"][0].detach().cpu().numpy()
    back = out["back_shell"][0].detach().cpu().numpy()
    idx = aux["idx"]
    body_sel = aux["body"][idx]
    rgb_sel = aux["rgb"][idx]
    keep = np.zeros_like(occ, dtype=bool)
    per_part_quota = max(128, int(np.ceil(60000 / (PART_COUNT * 3))))
    for part in range(PART_COUNT):
        ids = np.flatnonzero(body_sel == part)
        if len(ids) == 0:
            continue
        part_occ = occ[ids]
        local_thresh = min(0.40, max(0.24, float(np.quantile(part_occ, 0.46))))
        chosen = ids[part_occ >= local_thresh]
        if len(chosen) < min(per_part_quota, len(ids)):
            chosen = ids[np.argsort(-part_occ)[: min(per_part_quota, len(ids))]]
        keep[chosen] = True
    # Preserve strong globally occupied surfels as well, without letting them
    # erase smaller semantic parts.
    keep |= occ > max(0.43, float(np.quantile(occ, 0.30)))
    shells = np.stack([xyz, front, back], axis=1)
    points = shells[keep].reshape(-1, 3)
    colors = np.repeat(rgb_sel[keep][:, None, :], 3, axis=1).reshape(-1, 3)
    body = np.repeat(body_sel[keep], 3)
    if len(points) == 0:
        order = np.argsort(-aux["priority"])[:60000]
        return aux["surfel_xyz"][order].astype(np.float32), aux["rgb"][order], aux["body"][order].astype(np.int16)
    target_n = 60000
    if len(points) < target_n:
        chunks_p: list[np.ndarray] = []
        chunks_c: list[np.ndarray] = []
        chunks_b: list[np.ndarray] = []
        per_part_out = max(256, target_n // PART_COUNT)
        for part in range(PART_COUNT):
            ids = np.flatnonzero(body == part)
            if len(ids) == 0:
                continue
            take = min(per_part_out, len(ids))
            order = np.linspace(0, len(ids) - 1, take).astype(int)
            chosen = ids[order]
            chunks_p.append(points[chosen])
            chunks_c.append(colors[chosen])
            chunks_b.append(body[chosen])
        if chunks_p:
            points = np.concatenate(chunks_p, axis=0)
            colors = np.concatenate(chunks_c, axis=0)
            body = np.concatenate(chunks_b, axis=0)
        reps = int(np.ceil(target_n / max(len(points), 1)))
        points = np.tile(points, (reps, 1))[:target_n]
        colors = np.tile(colors, (reps, 1))[:target_n]
        body = np.tile(body, reps)[:target_n]
    elif len(points) > target_n:
        order = np.linspace(0, len(points) - 1, target_n).astype(int)
        points, colors, body = points[order], colors[order], body[order]
    return points.astype(np.float32), as_rgb(colors), body.astype(np.int16)


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = CanonicalSurfelGraphOccupancyStudent().to(device)
    batch, target, aux = build_batch(row, max_points=max_points, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = graph_losses(out, target)
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "occupancy": float(losses["occupancy"].detach().cpu()),
                    "visibility": float(losses["visibility"].detach().cpu()),
                    "part": float(losses["part"].detach().cpu()),
                    "shell": float(losses["shell"].detach().cpu()),
                    "coverage": float(losses["coverage"].detach().cpu()),
                    "occupancy_completeness": float(losses["occupancy_completeness"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = decode(out, aux, target)
    env, env_rgb = aux["environment_points"], aux["environment_rgb"]
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([human_rgb, env_rgb], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / TRUE_CONFIG)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=human_rgb,
        environment_points=env,
        environment_rgb=env_rgb,
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
        part_coverage_canonical_surfel_occupancy=np.array(True),
        source_part_count=np.array(aux["source_part_count"]),
        selected_part_count=np.array(aux["selected_part_count"]),
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
        "source_part_count": aux["source_part_count"],
        "selected_part_count": aux["selected_part_count"],
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config == V185_CONFIG:
        return OUTPUT / "V18500000000000000000_canonical_surfel_graph_training" / case / config / "predictions.npz"
    if config == V173_CONFIG:
        return V173_ROOT / case / config / "predictions.npz"
    if config == V183_CONFIG:
        return V183_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V185_CONFIG, V173_CONFIG, V183_CONFIG, *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_scores: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            path = config_path(case, cfg)
            if not path.exists():
                continue
            pts, _rgb, body = load_config(case, cfg)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            score_rows.append({"case": case, "config": cfg, **metric})
            case_scores[cfg] = metric
        true = case_scores.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_canonical_surfel_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in [V185_CONFIG, V173_CONFIG, V183_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
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
    write_csv(REPORTS / "V18600000000000000000_part_coverage_canonical_surfel_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in [TRUE_CONFIG, V185_CONFIG, V173_CONFIG, V183_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature"]:
        path = config_path(first, cfg)
        if path.exists():
            pts, rgb, _body = load_config(first, cfg)
            panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V18600000000000000000_part_coverage_canonical_surfel_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        turn = [
            render_panel(pts, rgb, f"{first} canonical {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        turn.append(cross_panel(pts, f"{first} canonical cross-section"))
        compose(turn, 3, BOARDS / "V18600000000000000000_part_coverage_canonical_surfel_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V18600_PART_COVERAGE_CANONICAL_SURFEL_FAIL_CLOSED_CONTINUE" if failures else "V18600_PART_COVERAGE_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "cases": [r["case"] for r in rows],
        "failures": failures,
        "score_csv": str(REPORTS / "V18600000000000000000_part_coverage_canonical_surfel_scores.csv"),
        "board": str(BOARDS / "V18600000000000000000_part_coverage_canonical_surfel_board.png"),
        "turntable_cross_section": str(BOARDS / "V18600000000000000000_part_coverage_canonical_surfel_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "v185_diagnosis": "V185 reduced some distant overlap but decoded only partial semantic coverage (part_presence around 0.5).",
        "summary": "V186 adds part-balanced surfel selection, coverage loss, and per-part decode quotas. It remains fail-closed unless full-scene mentor visuals and hard controls pass.",
    }
    write_json(REPORTS / "V18600000000000000000_training_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V18600_STEPS", "40"))
    max_points = int(os.environ.get("V18600_MAX_POINTS", "8192"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V18600000000000000000_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V18600000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V18600_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
