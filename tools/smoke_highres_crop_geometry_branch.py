from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt.models.highres_crop_geometry import HighResCropGeometryBranch


def main() -> None:
    torch.manual_seed(7)
    feature_dim = 16
    branch = HighResCropGeometryBranch(feature_dim=feature_dim, hidden_dim=32)
    world = torch.zeros(1, 2, 16, 16, 3)
    depth = torch.zeros(1, 2, 16, 16, 1)
    normal = torch.zeros(1, 2, 16, 16, 3)
    normal[..., 2] = 1.0
    features = torch.randn(1, 24, feature_dim)
    indices = torch.zeros(1, 24, 3, dtype=torch.long)
    indices[0, :, 0] = torch.arange(24) % 2
    indices[0, :, 1] = torch.arange(24) % 12 + 2
    indices[0, :, 2] = (torch.arange(24) * 3) % 12 + 2
    out0 = branch(world, depth, normal, features, indices)
    identity_l2 = torch.linalg.norm(out0["world_points"] - world, dim=-1).mean()
    target = torch.full_like(out0["crop_delta_point"], 4.0e-4)
    optim = torch.optim.AdamW(branch.parameters(), lr=5.0e-2)
    trace = []
    for step in range(12):
        optim.zero_grad()
        out = branch(world, depth, normal, features, indices)
        loss = torch.mean((out["crop_delta_point"] - target) ** 2)
        loss.backward()
        grad_norm = 0.0
        for p in branch.parameters():
            if p.grad is not None:
                grad_norm += float(torch.linalg.norm(p.grad).item())
        optim.step()
        trace.append({"step": step, "loss": float(loss.item()), "grad_norm": grad_norm})
    out1 = branch(world, depth, normal, features, indices)
    changed = torch.linalg.norm(out1["world_points"] - world, dim=-1) > 1e-7
    allowed = torch.zeros_like(changed)
    allowed[0, indices[0, :, 0], indices[0, :, 1], indices[0, :, 2]] = True
    report = {
        "identity_l2": float(identity_l2.item()),
        "loss_start": trace[0]["loss"],
        "loss_end": trace[-1]["loss"],
        "grad_nonzero": any(t["grad_norm"] > 0 for t in trace),
        "changed_pixels": int(changed.sum().item()),
        "outside_changed_pixels": int((changed & ~allowed).sum().item()),
        "target_indices": int(allowed.sum().item()),
        "trace": trace,
        "pass": bool(identity_l2.item() == 0.0 and trace[-1]["loss"] < trace[0]["loss"] and int((changed & ~allowed).sum().item()) == 0),
    }
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "V540_live_highres_crop_geometry_smoke.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
