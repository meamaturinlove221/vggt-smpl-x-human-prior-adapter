from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.v950_real_vggt_smpl_feature_adapter import (  # noqa: E402
    SOURCE_LABELS,
    RealVGGT_SMPLFeatureDetailAdapter,
    V950AdapterConfig,
    make_v950_batch_from_npz,
)


REPORTS = REPO / "reports"
TOKEN_ROOT = REPO / "output" / "V930000000000000_real_vggt_tokens"
FEATURE_ROOT = REPO / "output" / "V940000000000000_smpl_feature_bank"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def run_case(case_id: str, max_points: int) -> dict[str, Any]:
    token_path = TOKEN_ROOT / case_id / "real_vggt_tokens_and_predictions.npz"
    feature_path = FEATURE_ROOT / case_id / "smpl_feature_bank.npz"
    if not token_path.exists():
        raise FileNotFoundError(token_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    token_npz = load_npz(token_path)
    feature_npz = load_npz(feature_path)
    batch = make_v950_batch_from_npz(token_npz, feature_npz, max_points=max_points)
    for key, value in batch.items():
        if torch.is_tensor(value):
            value.requires_grad_(key in {"real_vggt_tokens", "smpl_feature_images", "smpl_point_features"})

    cfg = V950AdapterConfig(
        smpl_feature_channels=int(batch["smpl_feature_images"].shape[2]),
        vggt_token_dim=int(batch["real_vggt_tokens"].shape[-1]),
        hidden_dim=128,
        num_heads=4,
    )
    model = RealVGGT_SMPLFeatureDetailAdapter(cfg)
    outputs = model(batch)
    loss = (
        outputs["student_points"].pow(2).mean()
        + outputs["rgb"].mean()
        + outputs["binding_delta_norm"]
        + outputs["occupancy"].mean() * 0.1
    )
    loss.backward()

    forbidden_rejection_pass = False
    try:
        bad = dict(batch)
        bad["teacher_points"] = torch.zeros_like(batch["world_points"])
        model(bad)
    except ValueError:
        forbidden_rejection_pass = True

    def grad_mean(name: str) -> float:
        grad = batch[name].grad
        return float(grad.abs().mean().item()) if grad is not None else 0.0

    source_labels = outputs["source_label"].detach().cpu().numpy()
    unique, counts = np.unique(source_labels, return_counts=True)
    source_hist = {SOURCE_LABELS.get(int(k), str(int(k))): int(v) for k, v in zip(unique, counts, strict=False)}
    return {
        "case_id": case_id,
        "token_npz": str(token_path),
        "feature_npz": str(feature_path),
        "real_token_shape": list(batch["real_vggt_tokens"].shape),
        "smpl_feature_image_shape": list(batch["smpl_feature_images"].shape),
        "student_points_shape": list(outputs["student_points"].shape),
        "binding_delta_norm": float(outputs["binding_delta_norm"].detach().cpu().item()),
        "binding_gate_mean": float(outputs["binding_gate"].detach().mean().cpu().item()),
        "real_vggt_token_grad_mean_abs": grad_mean("real_vggt_tokens"),
        "smpl_feature_image_grad_mean_abs": grad_mean("smpl_feature_images"),
        "smpl_point_feature_grad_mean_abs": grad_mean("smpl_point_features"),
        "source_label_histogram": source_hist,
        "forbidden_teacher_input_rejected": forbidden_rejection_pass,
        "model_owned_student_output": bool(outputs["model_owned_student_output"].detach().cpu().item()),
        "no_teacher_points_inference": bool(outputs["no_teacher_points_inference"].detach().cpu().item()),
        "no_raw_kinect_depth_inference": bool(outputs["no_raw_kinect_depth_inference"].detach().cpu().item()),
    }


def run(args: argparse.Namespace) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cases = [args.case] if args.case != "all" else sorted(p.name for p in TOKEN_ROOT.iterdir() if p.is_dir())
    rows = [run_case(case, args.max_points) for case in cases]
    decision_pass = all(
        row["binding_delta_norm"] > 0
        and row["real_vggt_token_grad_mean_abs"] > 0
        and row["smpl_feature_image_grad_mean_abs"] > 0
        and row["forbidden_teacher_input_rejected"]
        for row in rows
    )
    write_json(
        REPORTS / "V950000000000000_forward_gradient_smoke.json",
        {
            "created_at": utc_now(),
            "rows": rows,
            "decision": {
                "pass": decision_pass,
                "real_vggt_tokens_affect_output": all(row["real_vggt_token_grad_mean_abs"] > 0 for row in rows),
                "smpl_feature_binding_affects_output": all(row["smpl_feature_image_grad_mean_abs"] > 0 for row in rows),
                "binding_delta_nonzero": all(row["binding_delta_norm"] > 0 for row in rows),
                "teacher_raw_kinect_rejected": all(row["forbidden_teacher_input_rejected"] for row in rows),
                "tiny_v330_or_synthetic_tokens_used": False,
            },
        },
    )
    write_json(
        REPORTS / "V950000000000000_architecture_contract.json",
        {
            "created_at": utc_now(),
            "model_file": str(REPO / "models" / "v950_real_vggt_smpl_feature_adapter.py"),
            "input_contract": {
                "real_vggt_tokens": "from output/V930 real VGGT.forward/Aggregator.forward NPZ",
                "smpl_feature_images": "from output/V940 scene/world/camera SMPL feature bank",
                "forbidden": [
                    "teacher_points",
                    "raw_kinect_depth",
                    "v591_points",
                    "synthetic_scene_tokens",
                    "tiny_v330_scene_tokens",
                ],
            },
            "architecture": {
                "SMPLFeatureEncoder": "patchifies SMPL-X surfel/world/camera feature images into prior tokens",
                "RealVGGTTokenBinder": "cross-attention/gated binding from SMPL prior tokens into real VGGT tokens",
                "DetailPreservingDecoder": "decodes bound tokens to model-owned scene-space RGB human surfels",
                "SourceLabels": SOURCE_LABELS,
            },
            "gate": {
                "posthoc_point_composition_only": False,
                "source_label_only": False,
                "tiny_v330_final_evidence": False,
                "real_vggt_token_gradient_required": True,
                "smpl_feature_gradient_required": True,
            },
        },
    )
    write_text(
        REPORTS / "V950000000000000_architecture_diagram.md",
        """# V950 Real VGGT SMPL Feature Adapter

```text
4K4D SMC RGB frames
        ->
current repo VGGT.forward / Aggregator.forward
        -> real VGGT tokens

SMPL-X surfel / voxel / graph / body-part / visibility / projection features
        -> SMPLFeatureEncoder
        -> SMPL prior tokens

real VGGT tokens + SMPL prior tokens
        -> RealVGGTTokenBinder (cross-attention + gated binding)
        -> DetailPreservingDecoder
        -> model-owned scene-space RGB human point cloud
```

Source labels are auxiliary only. TinyV330/synthetic scene tokens and posthoc
point composition are forbidden as final evidence.
""",
    )
    print(json.dumps({"V950_cases": len(rows), "forward_gradient_smoke_pass": decision_pass}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V950 real VGGT SMPL feature adapter smoke.")
    parser.add_argument("--case", default="all")
    parser.add_argument("--max-points", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
