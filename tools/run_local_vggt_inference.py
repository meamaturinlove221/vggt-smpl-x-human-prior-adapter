from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT inference locally on an exported scene directory.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--hf-repo", default="facebook/VGGT-1B")
    parser.add_argument("--image-mode", default="pad")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
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


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _load_model(args: argparse.Namespace, device: str) -> tuple[VGGT, dict]:
    if args.checkpoint.strip():
        checkpoint_path = Path(args.checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_model_state_dict(payload)
        model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
        if not isinstance(model_kwargs, dict):
            model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
        model = VGGT(**model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
        load_summary = {
            "checkpoint": str(checkpoint_path.resolve()),
            "model_kwargs": model_kwargs,
        }
    else:
        model = VGGT.from_pretrained(args.hf_repo)
        load_summary = {"hf_repo": args.hf_repo}
    model = model.to(device)
    model.eval()
    return model, load_summary


def _write_preview_png(array: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        preview = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(arr[finite], 2))
        hi = float(np.percentile(arr[finite], 98))
        if hi <= lo:
            hi = lo + 1e-6
        preview = (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(output_path)


def _write_normal_preview_png(array: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr).all(axis=-1, keepdims=True)
    arr = np.where(finite, arr, 0.0)
    arr = arr / np.clip(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-6, None)
    preview = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    Image.fromarray((preview * 255.0).astype(np.uint8)).save(output_path)


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    image_dir = scene_dir / "images"
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    start = time.time()
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=args.image_mode).to(device)
    prior_maps = None
    prior_summary_tokens = None
    prior_maps_path = scene_dir / "prior_maps.npz"
    if prior_maps_path.is_file():
        with np.load(prior_maps_path, allow_pickle=False) as payload:
            prior_maps = torch.from_numpy(np.array(payload["prior_maps"])).to(device=device, dtype=torch.float32)
            if "prior_summary_tokens" in payload.files:
                prior_summary_tokens = torch.from_numpy(np.array(payload["prior_summary_tokens"])).to(
                    device=device,
                    dtype=torch.float32,
                )

    model, load_summary = _load_model(args, device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
            predictions = model(
                images,
                human_prior_feature_maps=prior_maps,
                human_prior_summary_tokens=prior_summary_tokens,
            )

    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    arrays = {
        "pose_enc": _to_numpy(pose_enc.squeeze(0)),
        "extrinsic": _to_numpy(extrinsic.squeeze(0)),
        "intrinsic": _to_numpy(intrinsic.squeeze(0)),
        "depth": _to_numpy(predictions["depth"].squeeze(0)),
        "depth_conf": _to_numpy(predictions["depth_conf"].squeeze(0)),
        "world_points": _to_numpy(predictions["world_points"].squeeze(0)),
        "world_points_conf": _to_numpy(predictions["world_points_conf"].squeeze(0)),
    }
    if "normal" in predictions:
        arrays["normal"] = _to_numpy(predictions["normal"].squeeze(0))
    if "normal_conf" in predictions:
        arrays["normal_conf"] = _to_numpy(predictions["normal_conf"].squeeze(0))
    np.savez_compressed(output_dir / "predictions.npz", **arrays)

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        _write_preview_png(arrays["depth"][idx, ..., 0], preview_dir / f"{stem}_depth.png")
        _write_preview_png(arrays["depth_conf"][idx], preview_dir / f"{stem}_depth_conf.png")
        _write_preview_png(arrays["world_points_conf"][idx], preview_dir / f"{stem}_point_conf.png")
        if "normal" in arrays:
            _write_normal_preview_png(arrays["normal"][idx], preview_dir / f"{stem}_normal.png")
        if "normal_conf" in arrays:
            _write_preview_png(arrays["normal_conf"][idx], preview_dir / f"{stem}_normal_conf.png")

    manifest_path = scene_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    summary = {
        "task": "run_local_vggt_inference",
        "scene_dir": str(scene_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "image_mode": args.image_mode,
        "image_names": [path.name for path in image_paths],
        "num_images": len(image_paths),
        "device": device,
        "dtype": str(dtype),
        "input_tensor_shape": list(images.shape),
        "prior_tensor_shape": list(prior_maps.shape) if prior_maps is not None else None,
        "prior_summary_tensor_shape": list(prior_summary_tokens.shape) if prior_summary_tokens is not None else None,
        "load_summary": load_summary,
        "output_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "elapsed_seconds": round(time.time() - start, 3),
        "scene_manifest": manifest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
