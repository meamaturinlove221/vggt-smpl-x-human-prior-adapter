from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


WORKTREE = Path(r"D:\vggt\vggt-feature-adapter")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from vggt.models.vggt import VGGT


ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "output" / "V54000000_vggt_token_training" / "tiny_vggt_forward_smoke"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(54000000)
    model = VGGT(
        img_size=28,
        patch_size=14,
        patch_embed="conv",
        embed_dim=32,
        enable_camera=True,
        enable_point=False,
        enable_depth=False,
        enable_normal=False,
        enable_track=False,
        enable_human_prior_fusion=True,
        human_prior_in_chans=4,
        human_prior_hidden_dim=16,
        human_prior_scales=(1,),
        enable_human_prior_summary=False,
    )
    for name, param in model.named_parameters():
        param.requires_grad_("sparse_prior_adapter" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    images = torch.rand(1, 2, 3, 28, 28)
    sparse_prior_tokens = torch.randn(1, 2, 4, 32)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2)
    curves = []
    initial_gate = float(model.aggregator.sparse_prior_adapter.gamma.detach().abs().mean().item())
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        pred = model(images, sparse_prior_tokens=sparse_prior_tokens)
        pose = pred["pose_enc"]
        loss = -(pose.mean() + 0.1 * pose.std())
        loss.backward()
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.detach().norm().item())
        opt.step()
        curves.append(
            {
                "step": step,
                "loss": float(loss.detach().item()),
                "grad_norm": grad_norm,
                "gate_abs_mean": float(model.aggregator.sparse_prior_adapter.gamma.detach().abs().mean().item()),
            }
        )
    final_gate = float(model.aggregator.sparse_prior_adapter.gamma.detach().abs().mean().item())
    ckpt = OUTPUT / "checkpoint.pt"
    torch.save({"sparse_prior_adapter": model.aggregator.sparse_prior_adapter.state_dict(), "curves": curves}, ckpt)
    payload = {
        "created_utc": now(),
        "status": "V54000000_TINY_VGGT_FORWARD_TOKEN_TRAINING_SMOKE_PASS",
        "production_vggt_model_forward_integrated": True,
        "production_vggt_checkpoint_loaded": False,
        "forward_arg": "sparse_prior_tokens",
        "trainable_parameters": trainable,
        "initial_gate_abs_mean": initial_gate,
        "final_gate_abs_mean": final_gate,
        "gate_changed": bool(final_gate > initial_gate),
        "curves": curves,
        "checkpoint": str(ckpt),
        "scope": "Real VGGT class forward/training smoke with tiny dimensions and camera head; not a full pretrained VGGT checkpoint training run.",
    }
    (REPORTS / "V54000000_token_training_eval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
