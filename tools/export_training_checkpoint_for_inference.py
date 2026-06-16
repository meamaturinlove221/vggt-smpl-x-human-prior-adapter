from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a training checkpoint into the lightweight inference_model.pt payload used by VGGT inference."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt/checkpoint_*.pt/inference_model.pt")
    parser.add_argument("--output", required=True, help="Output inference_model.pt path")
    return parser.parse_args()


def _extract_model_state_dict(payload):
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _infer_model_kwargs_from_state_dict(state_dict: dict) -> dict:
    camera_token = state_dict.get("aggregator.camera_token")
    embed_dim = int(camera_token.shape[-1]) if camera_token is not None else 1024
    prior_conv = state_dict.get("aggregator.human_prior_patch_embeds.0.0.weight")
    legacy_proj = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_ln = state_dict.get("aggregator.human_prior_summary_proj.0.weight")
    legacy_summary = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    scale_count = len({
        key.split(".")[2]
        for key in state_dict
        if key.startswith("aggregator.human_prior_patch_embeds.") and key.endswith(".0.weight")
    })
    if scale_count <= 0:
        scale_count = 1
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": embed_dim,
        "enable_camera": any(key.startswith("camera_head.") for key in state_dict),
        "enable_point": any(key.startswith("point_head.") for key in state_dict),
        "enable_depth": any(key.startswith("depth_head.") for key in state_dict),
        "enable_normal": any(key.startswith("normal_head.") for key in state_dict),
        "enable_track": any(key.startswith("track_head.") for key in state_dict),
        "human_prior_in_chans": int(prior_conv.shape[1]) if prior_conv is not None else (int(legacy_proj.shape[1]) if legacy_proj is not None else 17),
        "human_prior_summary_in_dim": int(summary_ln.shape[0]) if summary_ln is not None else (int(legacy_summary.shape[1]) if legacy_summary is not None else 12),
        "human_prior_hidden_dim": int(prior_conv.shape[0]) if prior_conv is not None else (int(legacy_proj.shape[0]) if legacy_proj is not None else 128),
        "human_prior_scales": list(range(1, scale_count + 1)) if scale_count != 3 else [1, 2, 4],
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(payload)
    model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
    if not isinstance(model_kwargs, dict):
        model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state_dict,
            "model_kwargs": model_kwargs,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
        output_path,
    )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "output": str(output_path.resolve()),
        "num_tensors": len(state_dict),
        "model_kwargs": model_kwargs,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
