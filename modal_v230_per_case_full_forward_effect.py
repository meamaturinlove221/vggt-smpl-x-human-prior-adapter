from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = os.environ.get("VGGT_MODAL_V230_APP_NAME", "vggt-v230-per-case-full-forward-effect")
VOLUME_NAME = os.environ.get("VGGT_MODAL_V230_VOLUME", "vggt-v230-per-case-full-forward-effect-output")
REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REMOTE_ROOT = Path("/workspace")
REMOTE_VOLUME_ROOT = Path("/v230_volume")
REMOTE_OUT = REMOTE_VOLUME_ROOT / "out"
REMOTE_CACHE = REMOTE_VOLUME_ROOT / "cache"
MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "torchvision==0.18.1", "numpy==1.26.4", "Pillow", "einops", "huggingface_hub")
    .add_local_dir(str(REPO / "vggt"), remote_path="/workspace/vggt")
    .add_local_dir(
        str(REPO / "output" / "V1840000000000000_fullres_or_multiscale_tokens"),
        remote_path="/workspace/output/V1840000000000000_fullres_or_multiscale_tokens",
    )
    .add_local_dir(
        str(REPO / "output" / "V1850000000000000_smpl_feature_bank_v3"),
        remote_path="/workspace/output/V1850000000000000_smpl_feature_bank_v3",
    )
)


@app.function(image=image, gpu=os.environ.get("VGGT_MODAL_V230_GPU", "A10G"), timeout=60 * 60 * 4, volumes={str(REMOTE_VOLUME_ROOT): volume})
def run_case(case_id: str, max_prior_tokens: int = 1369) -> dict:
    import hashlib
    import json
    import sys
    import urllib.request
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(REMOTE_ROOT))
    from vggt.models.vggt import VGGT  # noqa: E402

    def now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def ensure_checkpoint() -> Path:
        REMOTE_CACHE.mkdir(parents=True, exist_ok=True)
        ckpt = REMOTE_CACHE / "facebook_VGGT-1B_model.pt"
        if not ckpt.exists() or ckpt.stat().st_size < 4_000_000_000:
            tmp = ckpt.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(ckpt)
        return ckpt

    def load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((518, 518), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = REMOTE_OUT / "V23000000000000000_per_case_full_forward_effect" / case_id
    out_root.mkdir(parents=True, exist_ok=True)
    image_path = REMOTE_ROOT / "output" / "V1840000000000000_fullres_or_multiscale_tokens" / case_id / "full_body_518.png"
    feature_path = REMOTE_ROOT / "output" / "V1850000000000000_smpl_feature_bank_v3" / case_id / "smpl_feature_bank_v3.npz"
    image_tensor = load_image(image_path).unsqueeze(0).unsqueeze(0).to(device)
    ckpt = ensure_checkpoint()
    model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False, enable_human_prior_fusion=True, enable_human_prior_summary=False)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model = model.to(device).eval()
    for adapter in [model.aggregator.sparse_prior_adapter, model.aggregator.input_prior_adapter]:
        if adapter is not None:
            torch.nn.init.constant_(adapter.gamma, 0.05)
    patch_count = (518 // 14) * (518 // 14)
    prior_count = min(max_prior_tokens, patch_count)
    with np.load(feature_path, allow_pickle=False) as z:
        xyz = np.asarray(z["world_points"], dtype=np.float32)
        rgb = np.asarray(z["rgb"], dtype=np.float32) / 255.0
        part = np.asarray(z["body_part_id"], dtype=np.float32)
        conf = np.asarray(z["confidence"], dtype=np.float32)
    idx = np.linspace(0, max(0, xyz.shape[0] - 1), prior_count).astype(np.int64)
    base = np.concatenate([xyz[idx], rgb[idx], part[idx, None] / max(1.0, float(part.max())), conf[idx, None]], axis=1)
    base_t = torch.from_numpy(base).to(device)
    # Deterministic case-specific projection from SMPL/VGGT features into 1024-dim prior tokens.
    seed = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % 1000003
    gen = torch.Generator(device=device).manual_seed(seed)
    proj = torch.randn(base_t.shape[1], 1024, generator=gen, device=device)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
    sparse_patch_tokens = (base_t @ proj).unsqueeze(0).unsqueeze(0)
    if prior_count < patch_count:
        sparse_patch_tokens = torch.cat([sparse_patch_tokens, torch.zeros(1, 1, patch_count - prior_count, 1024, device=device)], dim=2)
    sparse_patch_tokens.requires_grad_(True)
    with torch.no_grad():
        no_prior = model(image_tensor)
    with torch.enable_grad():
        with_prior = model(image_tensor, sparse_prior_tokens=sparse_patch_tokens)
        point_effect = (with_prior["world_points"] - no_prior["world_points"].detach()).abs().mean()
        depth_effect = (with_prior["depth"] - no_prior["depth"].detach()).abs().mean()
        conf_effect = (with_prior["world_points_conf"] - no_prior["world_points_conf"].detach()).abs().mean()
        effect = point_effect + depth_effect + 0.1 * conf_effect
        effect.backward()
    grad_mean = float(sparse_patch_tokens.grad.abs().mean().detach().cpu())
    payload = {
        "created_at": now(),
        "case_id": case_id,
        "route": "per_case_full_vggt_forward_effect",
        "full_vggt_forward_executed": True,
        "checkpoint_sha256": sha256(ckpt),
        "model_load_missing_count": len(missing),
        "model_load_unexpected_count": len(unexpected),
        "unexpected_all_track_head": all(str(k).startswith("track_head.") for k in unexpected),
        "camera_output_present": "pose_enc" in with_prior,
        "depth_output_present": "depth" in with_prior and "depth_conf" in with_prior,
        "point_output_present": "world_points" in with_prior and "world_points_conf" in with_prior,
        "sparse_prior_tokens_shape": list(sparse_patch_tokens.shape),
        "smpl_prior_token_injection_attempted": True,
        "sparse_prior_grad_mean": grad_mean,
        "point_effect_l1": float(point_effect.detach().cpu()),
        "depth_effect_l1": float(depth_effect.detach().cpu()),
        "confidence_effect_l1": float(conf_effect.detach().cpu()),
        "output_effect_l1": float(effect.detach().cpu()),
        "projection_seed": seed,
        "world_points_shape": list(with_prior["world_points"].shape),
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "synthetic_scene_tokens_used": False,
        "posthoc_point_composition_final": False,
    }
    np.savez_compressed(
        out_root / "full_forward_outputs.npz",
        world_points=with_prior["world_points"].detach().cpu().numpy().astype(np.float16),
        world_points_conf=with_prior["world_points_conf"].detach().cpu().numpy().astype(np.float16),
        depth=with_prior["depth"].detach().cpu().numpy().astype(np.float16),
        depth_conf=with_prior["depth_conf"].detach().cpu().numpy().astype(np.float16),
        pose_enc=with_prior["pose_enc"].detach().cpu().numpy().astype(np.float16),
        sparse_prior_grad_mean=np.asarray([grad_mean], dtype=np.float32),
        output_effect_l1=np.asarray([payload["output_effect_l1"]], dtype=np.float32),
    )
    (out_root / "trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume.commit()
    return payload


@app.local_entrypoint()
def main(case: str = "all", max_prior_tokens: int = 1369) -> None:
    selected = CASES if case == "all" else [case]
    for result in run_case.map(selected, kwargs={"max_prior_tokens": max_prior_tokens}):
        print(result)
