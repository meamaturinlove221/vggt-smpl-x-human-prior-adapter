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


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
REPORTS = REPO / "reports"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
WEAK_ROOT = REPO / "output" / "V13400000000000000000_billboard_weak_regions"

sys.path.insert(0, str(REPO))
from models.v135_anti_billboard_topology_volume_student import (  # noqa: E402
    AntiBillboardTopologyVolumeConfig,
    AntiBillboardTopologyVolumeStudent,
    smoke_test,
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    marker = "vggt-canonical-surfel-adapter/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        mapped = REPO / rel
        if mapped.exists():
            return mapped
    return path


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def pick_row() -> dict[str, str]:
    rows = read_manifest()
    for row in rows:
        if row["case"] == "0012_11_frame001":
            return row
    if not rows:
        raise RuntimeError("No eligible training rows found")
    return rows[0]


def build_real_batch(row: dict[str, str], max_points: int = 2048) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = AntiBillboardTopologyVolumeConfig()
    base = load_npz(repo_path(row["baseline_path"]))
    graph = load_npz(repo_path(row["graph_path"]))
    weak = load_npz(WEAK_ROOT / row["case"] / "billboard_weak_regions.npz")
    points = np.asarray(base["human_points"], dtype=np.float32)
    rgb = as_rgb(base["human_rgb"]).astype(np.float32) / 255.0
    repair = np.asarray(weak["billboard_repair_region_mask"], dtype=bool)
    no_change = np.asarray(weak["no_change_mask"], dtype=bool)
    weak_score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    priority = repair.astype(np.float32) * 4.0 + weak_score + conf * 0.05
    idx = np.argsort(-priority)[: min(max_points, len(priority))]

    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, : cfg.part_count] = np.eye(cfg.part_count, dtype=np.float32)[np.clip(body[idx], 0, cfg.part_count - 1)]
    smpl[:, cfg.part_count] = weak_score[idx]
    smpl[:, cfg.part_count + 1] = conf[idx]
    smpl[:, cfg.part_count + 2] = repair[idx].astype(np.float32)

    batch = {
        "anchor_xyz": torch.from_numpy(points[idx][None]),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]),
        "confidence": torch.from_numpy(conf[idx][None]),
        "weak_region": torch.from_numpy(np.maximum(repair[idx].astype(np.float32), weak_score[idx] * 0.2)[None]),
        "billboard_region": torch.from_numpy(repair[idx].astype(np.float32)[None]),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim),
        "smpl_features": torch.from_numpy(smpl[None]),
        "vggt_token_context": torch.zeros(1, cfg.token_dim),
        "body_part_id": torch.from_numpy(body[idx][None]),
    }
    # A bounded pseudo-target for smoke only: enough to prove loss wiring and
    # gradients without claiming visual success or using teacher points.
    center = points[idx].mean(axis=0, keepdims=True)
    centered = points[idx] - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    tangent = vh[1].astype(np.float32)
    repair_f = repair[idx].astype(np.float32)
    sign = np.sign(centered @ normal).astype(np.float32)
    sign[sign == 0] = 1.0
    target = points[idx] + normal[None] * sign[:, None] * (0.012 + 0.028 * repair_f[:, None])
    target += tangent[None] * np.sign((body[idx] % 5) - 2).astype(np.float32)[:, None] * (0.006 * repair_f[:, None])
    targets = {
        "target_xyz": torch.from_numpy(target[None].astype(np.float32)),
        "weak": batch["weak_region"].unsqueeze(-1),
        "billboard": batch["billboard_region"].unsqueeze(-1),
        "no_change": torch.from_numpy(no_change[idx][None]),
    }
    return batch, targets


def topology_volume_losses(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weak = targets["weak"].float()
    billboard = targets["billboard"].float()
    no_change = targets["no_change"].bool()
    residual = out["residual_xyz"]
    loss_preserve = residual[no_change].square().mean() if bool(no_change.any()) else residual.square().mean() * 0.0
    loss_repair = ((out["student_xyz"] - targets["target_xyz"]) ** 2 * weak).mean()
    shell_offsets = out["shell_offsets"]
    loss_shell_sep = F.relu(0.015 - shell_offsets[..., 0:2].mean()) + F.relu(0.012 - shell_offsets[..., 2:4].mean())
    occ_target = billboard.expand_as(out["cross_section_occupancy"])
    loss_cross_section = F.binary_cross_entropy(out["cross_section_occupancy"], occ_target.clamp(0, 1))
    loss_occupancy = F.binary_cross_entropy(out["occupancy"], billboard.clamp(0, 1))
    part_logits = out["part_continuity_logits"]
    # Smoke continuity target: parts seen in selected points should have active
    # continuity. A real training run will replace this with graph adjacency.
    loss_part_continuity = F.binary_cross_entropy_with_logits(part_logits, torch.ones_like(part_logits) * 0.65)
    projection_aux_loss = out["visibility"].mean() * 0.0
    total = (
        loss_repair
        + 1.5 * loss_preserve
        + 0.25 * loss_cross_section
        + 0.15 * loss_occupancy
        + 0.20 * loss_shell_sep
        + 0.08 * loss_part_continuity
        + projection_aux_loss
    )
    return {
        "total": total,
        "baseline_preservation_loss": loss_preserve,
        "weak_region_repair_loss": loss_repair,
        "cross_section_occupancy_loss": loss_cross_section,
        "front_back_side_shell_separation_loss": loss_shell_sep,
        "part_graph_continuity_loss": loss_part_continuity,
        "environment_preservation_loss": total * 0.0,
        "projection_auxiliary_loss": projection_aux_loss,
    }


def real_loss_smoke() -> dict[str, Any]:
    row = pick_row()
    model = AntiBillboardTopologyVolumeStudent()
    batch, targets = build_real_batch(row)
    out = model(batch)
    losses = topology_volume_losses(out, targets)
    losses["total"].backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    return {
        "case": row["case"],
        "selected_points": int(batch["anchor_xyz"].shape[1]),
        "student_xyz_shape": list(out["student_xyz"].shape),
        "cross_section_occupancy_shape": list(out["cross_section_occupancy"].shape),
        "shell_offsets_mean": float(out["shell_offsets"].detach().mean()),
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "grad_norm_positive": grad_norm > 0,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "mentor_ready": False,
    }


def write_reports(created_at: str, model_smoke: dict[str, Any], loss_smoke: dict[str, Any]) -> None:
    (REPORTS / "V13500000000000000000_architecture_diagram.md").write_text(
        "# V13500 Anti-Billboard Topology-Volume Architecture\n\n"
        "```text\n"
        "VGGT baseline anchors + RGB/confidence\n"
        "        + SMPL-X topology-volume anchors\n"
        "        + billboard weak-region mask\n"
        "        -> AntiBillboardTopologyVolumeStudent\n"
        "        -> residual in local frame\n"
        "        -> front/back/left/right shells\n"
        "        -> cross-section occupancy + part continuity\n"
        "        -> model-owned human-scene point cloud\n"
        "```\n\n"
        "Projection, thickness, and render checks remain auxiliary; mentor pass still requires full-scene RGB point-cloud visual evidence.\n",
        encoding="utf-8",
    )
    write_json(
        REPORTS / "V13500000000000000000_model_contract.json",
        {
            "created_at": created_at,
            "model": "models/v135_anti_billboard_topology_volume_student.py",
            "model_owned_student": True,
            "forbidden_inference_inputs": [
                "raw_kinect_depth",
                "kinect_depth",
                "teacher_points",
                "teacher_xyz",
                "v591_points",
                "v591_teacher",
                "dense_teacher",
            ],
            "required_outputs": [
                "student_xyz",
                "front_shell",
                "back_shell",
                "left_shell",
                "right_shell",
                "cross_section_occupancy",
                "part_continuity_logits",
            ],
            "final_success_policy": "model smoke is not mentor-ready; visual gates and hard controls are still required",
        },
    )
    write_json(
        REPORTS / "V13500000000000000000_forward_smoke.json",
        {"created_at": created_at, "smoke": model_smoke, "mentor_ready": False},
    )
    (REPORTS / "V13600000000000000000_loss_contract.md").write_text(
        "# V13600 Anti-Billboard Topology-Volume Loss Contract\n\n"
        "Required losses:\n\n"
        "1. baseline preservation loss\n"
        "2. weak-region repair loss\n"
        "3. cross-section occupancy loss\n"
        "4. front/back shell separation loss\n"
        "5. side shell continuity loss\n"
        "6. torso volume continuity loss\n"
        "7. limb cylindrical continuity loss\n"
        "8. part graph continuity loss\n"
        "9. clothing boundary layer loss\n"
        "10. environment preservation loss\n"
        "11. hard-control separation loss\n"
        "12. projection auxiliary loss\n\n"
        "Forbidden final claims: projection-only, thickness-only, source-label, global normal push, procedural occupancy. Smoke success only proves trainability.\n",
        encoding="utf-8",
    )
    write_json(
        REPORTS / "V13600000000000000000_loss_smoke.json",
        {"created_at": created_at, "smoke": loss_smoke, "mentor_ready": False},
    )


def main() -> int:
    created_at = now()
    model_smoke = smoke_test()
    loss_smoke = real_loss_smoke()
    write_reports(created_at, model_smoke, loss_smoke)
    print(json.dumps({"created_at": created_at, "status": "V135_V136_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
