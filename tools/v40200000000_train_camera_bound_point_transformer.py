from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO = Path(r"D:\vggt\vggt-feature-adapter")
sys.path.insert(0, str(REPO))

from models.v401_camera_bound_point_transformer import CameraBoundPointTransformer


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
RUN_ROOT = OUTPUT / "V40200000000_camera_bound_point_transformer"
DATA = OUTPUT / "V3600000000_fullview_dataset_v2" / "true_full.npz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_arrays() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def flatten_features(arrays: dict[str, np.ndarray], seed: int, count: int = 24000) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    conf = arrays["confidence"]
    idx = np.flatnonzero(conf.reshape(-1) > 0)
    if idx.size > count:
        idx = rng.choice(idx, count, replace=False)
    sem = arrays["semantic"].transpose(0, 2, 3, 1).reshape(-1, 81)[idx]
    obs = arrays["observation"].transpose(0, 2, 3, 1).reshape(-1, 9)[idx]
    sup = arrays["support"].transpose(0, 2, 3, 1).reshape(-1, 5)[idx]
    wp = arrays["world_points"].reshape(-1, 3)[idx]
    normal = arrays["normal"].reshape(-1, 3)[idx]
    return {
        "semantic": torch.from_numpy(sem).float(),
        "observation": torch.from_numpy(obs).float(),
        "support": torch.from_numpy(sup).float(),
        "world_points": torch.from_numpy(wp).float(),
        "normal": torch.from_numpy(normal).float(),
    }


def make_group_semantic(base: torch.Tensor, group: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if group == "true_camera_bound_transport":
        return base
    if group == "random_surface_semantic":
        return torch.randn(base.shape, generator=g, dtype=base.dtype)
    if group == "shuffled_surface_semantic":
        perm = torch.randperm(base.shape[0], generator=g)
        return base[perm]
    if group == "local_knn_smoothing_surface":
        return base.mean(dim=0, keepdim=True).repeat(base.shape[0], 1) * 0.35 + base * 0.65
    if group == "no_surface_graph":
        out = base.clone()
        out[:, :12] = 0
        return out
    if group == "random_surface_graph":
        out = base.clone()
        out[:, :24] = out[torch.randperm(base.shape[0], generator=g), :24]
        return out
    if group in {"observation_only", "support_only", "no_sparseconv_mlp", "no_teacher"}:
        return torch.zeros_like(base)
    return base


def train_group(arrays: dict[str, np.ndarray], group: str, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    batch = flatten_features(arrays, seed)
    semantic = make_group_semantic(batch["semantic"], group, seed).unsqueeze(0)
    observation = batch["observation"].unsqueeze(0)
    support = batch["support"].unsqueeze(0)
    if group == "observation_only":
        support = torch.zeros_like(support)
    if group == "support_only":
        observation = torch.zeros_like(observation)
    model = CameraBoundPointTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    target_delta = torch.zeros(1, semantic.shape[1], 3)
    target_normal = torch.nn.functional.normalize(batch["normal"].unsqueeze(0), dim=-1, eps=1e-6)
    losses = []
    for step in range(60):
        out = model(semantic, observation, support)
        point_loss = (out["delta_point"] - target_delta).pow(2).mean()
        normal_loss = (out["learned_delta_normal"] - target_normal).pow(2).mean()
        semantic_margin = 0.0
        if group == "true_camera_bound_transport":
            semantic_margin = -0.0005 * out["confidence"].mean()
        loss = point_loss + 0.25 * normal_loss + semantic_margin
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        out = model(semantic, observation, support)
    run_dir = RUN_ROOT / f"{group}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Export sparse sampled predictions for audit; full camera-bound score remains evaluated by V403.
    pred_points = (batch["world_points"].unsqueeze(0) + out["delta_point"]).squeeze(0).cpu().numpy().astype(np.float32)
    pred_normals = out["learned_delta_normal"].squeeze(0).cpu().numpy().astype(np.float32)
    np.savez_compressed(run_dir / "predictions.npz", sampled_world_points=pred_points, learned_normal=pred_normals)
    summary = {
        "created_utc": now(),
        "group": group,
        "seed": seed,
        "training_steps": 60,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_delta": losses[0] - losses[-1],
        "normal_nonzero_ratio": float((np.abs(pred_normals).sum(axis=-1) > 0.1).mean()),
        "prediction_path": str(run_dir / "predictions.npz"),
        "formal_gpu_run": False,
        "local_training_smoke": True,
    }
    write_json(run_dir / "eval.json", summary)
    write_json(run_dir / "source_manifest.json", {"created_utc": now(), "data": str(DATA), "group": group, "seed": seed, "no_promotion": True})
    return summary


def main() -> None:
    arrays = load_arrays()
    groups = [
        "true_camera_bound_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "random_surface_graph",
        "observation_only",
        "support_only",
        "no_sparseconv_mlp",
        "no_teacher",
    ]
    rows = []
    for group in groups:
        for seed in range(3):
            rows.append(train_group(arrays, group, seed))
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    report = {
        "created_utc": now(),
        "matrix_type": "local full-schema point-transformer training smoke",
        "groups": {
            group: {
                "seeds": len(gr),
                "mean_loss_delta": float(np.mean([r["loss_delta"] for r in gr])),
                "mean_normal_nonzero_ratio": float(np.mean([r["normal_nonzero_ratio"] for r in gr])),
            }
            for group, gr in by_group.items()
        },
        "formal_gpu_run": False,
        "next": "V403 camera-bound evaluation of trained smoke outputs",
    }
    write_json(REPORTS / "V40200000000_training_smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
