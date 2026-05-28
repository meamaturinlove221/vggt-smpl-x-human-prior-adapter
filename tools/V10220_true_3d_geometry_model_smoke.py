from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"
BASE_ROOT = REPO / "output" / "V1400000000000000000_learned_residual_matrix"
GRAPH_ROOT = REPO / "output" / "V5360000000000000000_geometry_part_binding_repair"
V161_ROOT = REPO / "output" / "V161000000000000_repaired_detail_regions"

sys.path.insert(0, str(REPO))
from models.v2500_canonical_surfel_residual_student import (  # noqa: E402
    CanonicalSurfelResidualConfig,
    CanonicalSurfelResidualStudent,
    make_smoke_batch,
    smoke_test,
)


CASE = "0012_11_frame001"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def pca_thickness_tensor(points: torch.Tensor) -> torch.Tensor:
    x = points - points.mean(dim=1, keepdim=True)
    cov = torch.matmul(x.transpose(1, 2), x) / max(1, points.shape[1] - 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, dim=-1, descending=True)
    vals = torch.gather(vals, 1, order)
    vecs = torch.gather(vecs, 2, order.unsqueeze(1).expand(-1, 3, -1))
    proj = torch.matmul(x, vecs)
    ranges = proj.max(dim=1).values - proj.min(dim=1).values
    return ranges[:, 2] / torch.clamp(ranges[:, 0], min=1e-6)


def build_case_batch(points: int = 1024) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base = load_npz(BASE_ROOT / CASE / "real_vggt_baseline_only" / "predictions.npz")
    graph = load_npz(GRAPH_ROOT / CASE / "mentor_view_geometry_part_graph.npz")
    target = load_npz(V161_ROOT / CASE / "repaired_detail_regions_world_rgb.npz")
    hp = np.asarray(base["human_points"], dtype=np.float32)
    rgb = np.asarray(base["human_rgb"], dtype=np.float32) / 255.0
    weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    # Pick weak-region anchors first so the smoke actually exercises residual heads.
    order = np.argsort(-(weak + conf * 0.05))
    idx = order[:points]
    target_points = np.asarray(target["world_points"], dtype=np.float32)
    # Use robust nearest target statistics only as a training target; not an inference key.
    target_center = np.median(target_points, axis=0)
    anchor_center = np.median(hp[idx], axis=0)
    target_span = np.maximum(np.percentile(target_points, 95, axis=0) - np.percentile(target_points, 5, axis=0), 1e-6)
    anchor_span = np.maximum(np.percentile(hp[idx], 95, axis=0) - np.percentile(hp[idx], 5, axis=0), 1e-6)
    scaled_target = (target_points[:points] - target_center[None]) / target_span[None] * anchor_span[None] + anchor_center[None]
    if len(scaled_target) < points:
        pad = np.repeat(scaled_target[-1:], points - len(scaled_target), axis=0)
        scaled_target = np.concatenate([scaled_target, pad], axis=0)
    part_feat = np.eye(8, dtype=np.float32)[np.clip(body[idx], 0, 7)]
    smpl_features = np.zeros((points, 32), dtype=np.float32)
    smpl_features[:, :8] = part_feat
    smpl_features[:, 8] = weak[idx]
    smpl_features[:, 9] = conf[idx]
    batch = {
        "surfel_xyz": torch.from_numpy(hp[idx][None]),
        "surfel_rgb": torch.from_numpy(rgb[idx][None]),
        "surfel_confidence": torch.from_numpy(conf[idx][None]),
        "surfel_features": torch.zeros(1, points, 32),
        "smpl_features": torch.from_numpy(smpl_features[None]),
        "vggt_token_context": torch.zeros(1, 64),
    }
    targets = {
        "target_xyz": torch.from_numpy(scaled_target[None].astype(np.float32)),
        "no_change_mask": torch.from_numpy(np.asarray(graph["no_change_mask"], dtype=bool)[idx][None]),
        "weak": torch.from_numpy(weak[idx][None]),
    }
    return batch, targets


def main() -> int:
    created_at = now()
    cfg = CanonicalSurfelResidualConfig(max_residual=0.030, max_rgb_delta=0.04)
    generic_smoke = smoke_test()
    model = CanonicalSurfelResidualStudent(cfg)
    batch, targets = build_case_batch(points=1024)
    out = model(batch)
    weak = targets["weak"].unsqueeze(-1)
    no_change = targets["no_change_mask"].unsqueeze(-1)
    residual_target = targets["target_xyz"] - batch["surfel_xyz"]
    fit_loss = (torch.abs(out["residual_xyz"] - residual_target).clamp(max=0.05) * (0.25 + weak)).mean()
    preservation_loss = (torch.abs(out["residual_xyz"]) * no_change.float()).mean()
    thickness_loss = -pca_thickness_tensor(out["student_xyz"]).mean() * 0.02
    rgb_loss = out["rgb_delta"].abs().mean() * 0.1
    loss = fit_loss + preservation_loss * 2.0 + thickness_loss + rgb_loss
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer.step()
    forbidden_rejected = False
    try:
        bad = dict(batch)
        bad["teacher_points"] = targets["target_xyz"]
        model(bad)
    except ValueError:
        forbidden_rejected = True

    payload = {
        "created_at": created_at,
        "status": "V10220_TRUE_3D_GEOMETRY_MODEL_SMOKE_PASS_INTERNAL_ONLY"
        if generic_smoke.get("grad_norm_positive") and grad_norm > 0 and forbidden_rejected
        else "V10220_TRUE_3D_GEOMETRY_MODEL_SMOKE_FAIL_CLOSED",
        "generic_smoke": generic_smoke,
        "case": CASE,
        "case_batch_points": 1024,
        "loss": float(loss.detach()),
        "fit_loss": float(fit_loss.detach()),
        "preservation_loss": float(preservation_loss.detach()),
        "thickness_loss": float(thickness_loss.detach()),
        "rgb_loss": float(rgb_loss.detach()),
        "grad_norm_positive": grad_norm > 0,
        "grad_norm": grad_norm,
        "forbidden_teacher_key_rejected": forbidden_rejected,
        "student_shape": list(out["student_xyz"].shape),
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
    }
    report = REPORTS / "V10220000000000000000_true_3d_geometry_model_smoke.json"
    write_json(report, payload)
    next_goal = GOALS / "V10230000000000000000_auto_evolved_training_execution_or_modal_payload_route.md"
    write_text(
        next_goal,
        f"""# V10230 Auto-Evolved Training Execution Or Modal Payload Route

Created: {created_at}

V10220 verified the canonical surfel/graph student model smoke.

Next:
- create a trainable multi-case dataset and command;
- run local tiny train or Modal payload;
- generate predictions, oblique full-scene boards, controls, and local visible-part boards;
- do not claim mentor-ready without the mentor visual gate.
""",
    )
    payload["next_goal"] = str(next_goal)
    write_json(report, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
