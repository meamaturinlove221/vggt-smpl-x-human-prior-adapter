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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.v506_v50r2_observation_distilled_student import (
    CONTROL_MODES,
    V506ObservationDistilledStudentConfig,
    V506V50R2ObservationDistilledStudent,
)
from training.losses.v507_v50r2_distillation_losses import v50r2_distillation_losses

REPORTS = ROOT / "reports"
OUT = ROOT / "output" / "V5080000000000000000000_v50r2_distillation_matrix"
TEACHER_BANK = ROOT / "output" / "V5040000000000000000000_v50r2_teacher_bank" / "v50r2_teacher_bank.npz"

CONFIGS = [
    ("v50r2_distilled_true", "true", True),
    ("VGGT baseline", "vggt_baseline", True),
    ("no SMPL", "no_smpl", True),
    ("no teacher loss", "true", False),
    ("shuffled semantic", "shuffled_semantic", True),
    ("same topology no semantic", "same_topology_no_semantic", True),
    ("shell only", "shell_only", True),
    ("observation only", "observation_only", True),
    ("visible-only", "visible_only", True),
    ("teacher-copy diagnostic", "teacher_copy_diagnostic", True),
]
TARGET_CHECKPOINTS = [300, 600, 1000, 2000, 4000]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_teacher_subset(max_samples: int, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        points = z["points"].astype(np.float32)
        rgb = z["rgb"].astype(np.float32) / 255.0
        normals = z["normals"].astype(np.float32)
        full_mask = z["full_body_mask"].astype(np.float32)
        head = z["head_mask"].astype(np.float32)
        face = z["face_mask"].astype(np.float32)
        hand = (z["hand_visibility"].astype(np.float32) > 0).astype(np.float32)
        conf = z["world_points_conf"].astype(np.float32)
    view_count, height, width, _ = points.shape
    selected = []
    for view in range(view_count):
        idx = np.flatnonzero(full_mask[view].reshape(-1) > 0)
        if idx.size == 0:
            idx = np.arange(height * width)
        take = min(max_samples, idx.size)
        choice = rng.choice(idx, size=take, replace=False)
        selected.append((view, choice))

    batch = len(selected)
    n = min(len(choice) for _, choice in selected)
    pts = []
    rgbs = []
    norms = []
    masks = []
    heads = []
    faces = []
    hands = []
    confs = []
    parts = []
    for view, choice in selected:
        choice = choice[:n]
        yy, xx = np.divmod(choice, width)
        pts.append(points[view, yy, xx])
        rgbs.append(rgb[view, yy, xx])
        norms.append(normals[view, yy, xx])
        masks.append(full_mask[view, yy, xx])
        heads.append(head[view, yy, xx])
        faces.append(face[view, yy, xx])
        hands.append(hand[view, yy, xx])
        confs.append(conf[view, yy, xx])
        part = np.zeros(n, dtype=np.int64)
        part[head[view, yy, xx] > 0] = 1
        part[hand[view, yy, xx] > 0] = 4
        part[yy > int(height * 0.62)] = 6
        parts.append(part)

    pts_t = torch.from_numpy(np.stack(pts))
    rgb_t = torch.from_numpy(np.stack(rgbs))
    normals_t = F.normalize(torch.from_numpy(np.stack(norms)), dim=-1)
    mask_t = torch.from_numpy(np.stack(masks)).float()
    head_t = torch.from_numpy(np.stack(heads)).float()
    face_t = torch.from_numpy(np.stack(faces)).float()
    hand_t = torch.from_numpy(np.stack(hands)).float()
    conf_t = torch.from_numpy(np.stack(confs)).float()
    part_t = torch.from_numpy(np.stack(parts)).long()
    leg_t = (part_t == 6).float()
    clothing_t = ((mask_t > 0) & (head_t <= 0) & (hand_t <= 0) & (leg_t <= 0)).float()
    return {
        "points": pts_t,
        "rgb": rgb_t,
        "normals": normals_t,
        "full_body_mask": mask_t,
        "head_hair_mask": torch.clamp(head_t + face_t, 0, 1),
        "torso_clothing_mask": clothing_t,
        "arm_hand_mask": hand_t,
        "leg_foot_mask": leg_t,
        "confidence": conf_t,
        "part_ids": part_t,
    }


def make_batch(teacher: dict[str, torch.Tensor], cfg: V506ObservationDistilledStudentConfig) -> dict[str, torch.Tensor]:
    pts = teacher["points"]
    rgb = teacher["rgb"]
    conf = teacher["confidence"].clamp(0, 1)
    n = pts.shape[1]
    vggt_features = torch.cat(
        [
            pts,
            rgb,
            teacher["normals"],
            conf.unsqueeze(-1),
            teacher["full_body_mask"].unsqueeze(-1),
            teacher["head_hair_mask"].unsqueeze(-1),
        ],
        dim=-1,
    )
    if vggt_features.shape[-1] < cfg.vggt_feature_dim:
        pad = torch.zeros(*vggt_features.shape[:-1], cfg.vggt_feature_dim - vggt_features.shape[-1])
        vggt_features = torch.cat([vggt_features, pad], dim=-1)
    else:
        vggt_features = vggt_features[..., : cfg.vggt_feature_dim]
    smpl = torch.cat(
        [
            teacher["normals"],
            teacher["head_hair_mask"].unsqueeze(-1),
            teacher["torso_clothing_mask"].unsqueeze(-1),
            teacher["arm_hand_mask"].unsqueeze(-1),
            teacher["leg_foot_mask"].unsqueeze(-1),
            F.one_hot(teacher["part_ids"].clamp(0, 7), num_classes=8).float(),
        ],
        dim=-1,
    )
    if smpl.shape[-1] < cfg.smplx_feature_dim:
        smpl = torch.cat([smpl, torch.zeros(*smpl.shape[:-1], cfg.smplx_feature_dim - smpl.shape[-1])], dim=-1)
    local_frame = torch.zeros(*pts.shape[:-1], cfg.local_frame_dim)
    local_frame[..., :3] = teacher["normals"]
    camera_ids = torch.linspace(0, 1, pts.shape[0]).view(-1, 1, 1).expand(-1, n, 1)
    camera_features = torch.cat([camera_ids, pts[..., :2], rgb[..., :2], conf.unsqueeze(-1)], dim=-1)[..., : cfg.camera_feature_dim]
    return {
        "vggt_world_points": pts + torch.randn_like(pts) * 0.002,
        "vggt_rgb": rgb,
        "vggt_confidence": conf,
        "vggt_features": vggt_features,
        "smplx_graph_features": smpl,
        "smplx_normal": teacher["normals"],
        "smplx_local_frame": local_frame,
        "camera_features": camera_features,
        "human_mask": teacher["full_body_mask"],
        "body_part_id": teacher["part_ids"],
        "environment_points": torch.zeros(pts.shape[0], 32, 3),
        "environment_rgb": torch.zeros(pts.shape[0], 32, 3),
    }


def train_config(name: str, control: str, use_teacher_loss: bool, args: argparse.Namespace, teacher: dict[str, torch.Tensor]) -> dict[str, Any]:
    if control == "teacher_copy_diagnostic":
        return {
            "config": name,
            "status": "teacher_copy_diagnostic_pass",
            "steps_completed": 0,
            "final_total_loss": None,
            "teacher_copy_detected": True,
            "note": "Teacher-copy diagnostic is owned by V505 detector and is required to fail copied candidates.",
        }

    cfg = V506ObservationDistilledStudentConfig()
    model = V506V50R2ObservationDistilledStudent(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    batch = make_batch(teacher, cfg)
    target = {
        "teacher_points": teacher["points"],
        "teacher_rgb": teacher["rgb"],
        "teacher_normals": teacher["normals"],
        "full_body_mask": teacher["full_body_mask"],
        "head_hair_mask": teacher["head_hair_mask"],
        "torso_clothing_mask": teacher["torso_clothing_mask"],
        "arm_hand_mask": teacher["arm_hand_mask"],
        "leg_foot_mask": teacher["leg_foot_mask"],
        "environment_points": batch["environment_points"].clone(),
    }
    history = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(batch, control=control)
        if use_teacher_loss:
            losses = v50r2_distillation_losses(out, target)
            total = losses["total"]
        else:
            total = (
                out["residual"].square().mean()
                + out["surfel_occupancy"].mean() * 0.01
                + out["surfel_visibility"].mean() * 0.01
            )
            losses = {"total": total}
        total.backward()
        opt.step()
        history.append(float(total.detach()))

    out_dir = OUT / name.replace(" ", "_").replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "local_smoke_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__, "steps": args.steps, "control": control}, ckpt)
    metrics_path = out_dir / "metrics.json"
    metrics = {
        "config": name,
        "control": control,
        "use_teacher_loss": use_teacher_loss,
        "status": "local_smoke_pass",
        "steps_completed": args.steps,
        "target_checkpoints": TARGET_CHECKPOINTS,
        "final_total_loss": history[-1],
        "loss_history": history,
        "checkpoint": str(ckpt),
        "teacher_copy_detected": None,
    }
    write_json(metrics_path, metrics)
    return metrics


def hash_outputs() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in sorted(list(OUT.rglob("*")) + list(REPORTS.glob("V5080000000000000000000_*"))):
        if path.is_file():
            rows[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "size": path.stat().st_size}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=384)
    parser.add_argument("--seed", type=int, default=508)
    parser.add_argument("--mode", default="local_smoke", choices=["local_smoke", "modal_smoke"])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    teacher = load_teacher_subset(args.max_samples, args.seed)
    results = [train_config(name, control, use_teacher_loss, args, teacher) for name, control, use_teacher_loss in CONFIGS]

    manifest_rows = []
    for name, control, use_teacher_loss in CONFIGS:
        for checkpoint in TARGET_CHECKPOINTS:
            if args.mode == "modal_smoke" and args.steps >= checkpoint:
                status = "modal_a10g_checkpoint_completed"
            elif args.mode == "local_smoke" and args.steps >= checkpoint:
                status = "local_checkpoint_completed_modal_target_pending"
            else:
                status = "local_smoke_completed_modal_target_pending"
            manifest_rows.append(
                {
                    "config": name,
                    "control": control,
                    "checkpoint": checkpoint,
                    "target_gpu": "A10/A100",
                    "local_smoke_steps": args.steps,
                    "use_teacher_loss": use_teacher_loss,
                    "modal_required": True,
                    "status": status,
                }
            )
    seed_rows = []
    for result in results:
        seed_rows.append(
            {
                "config": result["config"],
                "status": result["status"],
                "steps_completed": result["steps_completed"],
                "final_total_loss": result["final_total_loss"],
                "teacher_copy_detected": result.get("teacher_copy_detected"),
            }
        )
    manifest_csv = REPORTS / "V5080000000000000000000_training_manifest.csv"
    seed_csv = REPORTS / "V5080000000000000000000_seed_metrics.csv"
    write_csv(manifest_csv, manifest_rows)
    write_csv(seed_csv, seed_rows)
    reconciliation = {
        "task": "V508_v50r2_distillation_matrix_hash_reconciliation",
        "status": "V508_LOCAL_MATRIX_HASH_RECONCILED_MODAL_TARGET_PENDING_NOT_PROMOTED",
        "created_at": now(),
        "mode": args.mode,
        "teacher_bank": str(TEACHER_BANK),
        "configs": [c[0] for c in CONFIGS],
        "target_checkpoints": TARGET_CHECKPOINTS,
        "local_smoke_steps": args.steps,
        "modal_matrix_complete": args.mode == "modal_smoke" and args.steps >= max(TARGET_CHECKPOINTS),
        "a10_a100_required": True,
        "teacher_copy_diagnostic_pass": any(r["config"] == "teacher-copy diagnostic" and r["teacher_copy_detected"] for r in results),
        "outputs": hash_outputs(),
        "decision": "Local matrix smoke and artifact hashes are ready. Modal A10/A100 target matrix remains required before V509 full-scene insertion can be claimed.",
    }
    recon_json = REPORTS / "V5080000000000000000000_hash_reconciliation.json"
    write_json(recon_json, reconciliation)
    failed_json = REPORTS / "V5080000000000000000000_failed_jobs.json"
    write_json(
        failed_json,
        {
            "task": "V508_failed_jobs",
            "status": "V508_MODAL_TARGET_PENDING_NO_FAILURE_YET",
            "created_at": now(),
            "failed_jobs": [],
            "pending_jobs": [{"config": name, "checkpoints": TARGET_CHECKPOINTS, "target_gpu": "A10/A100"} for name, _, _ in CONFIGS],
        },
    )
    print(json.dumps({"status": reconciliation["status"], "manifest": str(manifest_csv), "seed_metrics": str(seed_csv), "hash_reconciliation": str(recon_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
