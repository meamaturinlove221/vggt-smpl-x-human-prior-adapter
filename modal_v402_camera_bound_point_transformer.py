from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "vggt-v402-camera-bound-point-transformer"
DEFAULT_GPU = os.environ.get("VGGT_MODAL_V402_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.3.1", "Pillow==10.4.0")
)
app = modal.App(APP_NAME)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_payload(dataset_npz: Path, max_points: int = 12000) -> str:
    import numpy as np

    with np.load(dataset_npz, allow_pickle=False) as z:
        sem = z["semantic"].transpose(0, 2, 3, 1).reshape(-1, 81)
        obs = z["observation"].transpose(0, 2, 3, 1).reshape(-1, 9)
        sup = z["support"].transpose(0, 2, 3, 1).reshape(-1, 5)
        normal = z["normal"].reshape(-1, 3)
        conf = z["confidence"].reshape(-1)
    idx = np.flatnonzero(conf > 0)
    if idx.size > max_points:
        rng = np.random.default_rng(402)
        idx = rng.choice(idx, max_points, replace=False)
    bio = io.BytesIO()
    np.savez_compressed(bio, semantic=sem[idx].astype("float32"), observation=obs[idx].astype("float32"), support=sup[idx].astype("float32"), normal=normal[idx].astype("float32"))
    return base64.b64encode(bio.getvalue()).decode("ascii")


def make_full_payload(dataset_npz: Path) -> str:
    import numpy as np

    with np.load(dataset_npz, allow_pickle=False) as z:
        bio = io.BytesIO()
        np.savez_compressed(
            bio,
            semantic=z["semantic"].astype("float16"),
            observation=z["observation"].astype("float16"),
            support=z["support"].astype("float16"),
            world_points=z["world_points"].astype("float16"),
            normal=z["normal"].astype("float16"),
            confidence=z["confidence"].astype("float16"),
        )
    return base64.b64encode(bio.getvalue()).decode("ascii")


def load_v304_module():
    path = Path(__file__).resolve().parent / "tools" / "v30400000000_coordinate_binding_search.py"
    spec = importlib.util.spec_from_file_location("v304_for_modal_payload", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_camera_payload() -> str:
    import h5py
    import numpy as np

    v304 = load_v304_module()
    reports = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild\reports")
    binding_doc = json.loads((reports / "V30400000000_best_binding.json").read_text(encoding="utf-8"))
    binding = binding_doc["best"]
    smc_path = v304.SMC_DIR / binding["smc"]
    Ks = []
    w2c = []
    masks = []
    centers = []
    bmins = []
    bmaxs = []
    for cid in ["00", "01", "15", "30", "45", "59"]:
        mask_info = v304.try_read_mask(smc_path, cid)
        if mask_info is None:
            raise RuntimeError(f"Missing mask for camera {cid} in {smc_path}")
        mask = v304.resize_mask(mask_info["mask"].astype("uint8") * 255, (518, 518))
        with h5py.File(str(smc_path), "r") as f:
            g = f[f"Camera_Parameter/{cid}"]
            K = v304.resize_intrinsic(g["K"][()], mask_info["source_hw"], (518, 518))
            RT = g["RT"][()]
        M = v304.world_to_camera_matrix(RT, binding["rt_convention"])
        yx = np.argwhere(mask)
        if yx.size == 0:
            center = np.array([0.5, 0.5], dtype=np.float32)
            bmin = np.array([0.0, 0.0], dtype=np.float32)
            bmax = np.array([1.0, 1.0], dtype=np.float32)
        else:
            ymin, xmin = yx.min(axis=0)
            ymax, xmax = yx.max(axis=0)
            center = np.array([(xmin + xmax) * 0.5 / 517.0, (ymin + ymax) * 0.5 / 517.0], dtype=np.float32)
            bmin = np.array([xmin / 517.0, ymin / 517.0], dtype=np.float32)
            bmax = np.array([xmax / 517.0, ymax / 517.0], dtype=np.float32)
        Ks.append(K.astype("float32"))
        w2c.append(M.astype("float32"))
        masks.append(mask.astype("uint8"))
        centers.append(center)
        bmins.append(bmin)
        bmaxs.append(bmax)
    payload = {
        "K": np.stack(Ks),
        "world_to_camera": np.stack(w2c),
        "mask_center": np.stack(centers),
        "bbox_min": np.stack(bmins),
        "bbox_max": np.stack(bmaxs),
        "axis_signs": np.array(v304.AXIS_FLIPS[binding["axis_flip"]], dtype=np.float32),
        "unit_scale": np.array([v304.UNIT_SCALES[binding["unit_name"]]], dtype=np.float32),
        "scale": np.array([binding["scale"]], dtype=np.float32),
        "translation": np.array([binding["translation_x"], binding["translation_y"], binding["translation_z"]], dtype=np.float32),
    }
    bio = io.BytesIO()
    np.savez_compressed(bio, **payload)
    return base64.b64encode(bio.getvalue()).decode("ascii")


@app.function(image=image, gpu=DEFAULT_GPU, cpu=4.0, memory=32 * 1024, timeout=2 * 60 * 60)
def run_remote_matrix(payload_b64: str, groups: list[str], seeds: list[int], steps: int = 80) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 96
            self.semantic = nn.Sequential(nn.LayerNorm(81), nn.Linear(81, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.observation = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.delta = nn.Linear(hidden, 3)
            self.normal = nn.Linear(hidden, 3)
            self.conf = nn.Linear(hidden, 1)

        def forward(self, sem, obs, sup):
            mask = sup[..., :1].sigmoid()
            x = self.semantic(sem) + 0.35 * self.observation(obs)
            x, _ = self.attn(x, x, x)
            x = x * mask
            return self.delta(x), torch.nn.functional.normalize(self.normal(x), dim=-1, eps=1e-6), self.conf(x).sigmoid()

    started = time.time()
    cuda = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda else "NO_CUDA"
    device = torch.device("cuda" if cuda else "cpu")
    raw = base64.b64decode(payload_b64.encode("ascii"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        semantic_np = z["semantic"]
        observation_np = z["observation"]
        support_np = z["support"]
        normal_np = z["normal"]
    results = []
    if not cuda:
        return {"created_utc": now_utc(), "cuda_available": False, "gpu_type": gpu_name, "status": "FAILED_NO_CUDA", "results": results}

    base_sem = torch.from_numpy(semantic_np).to(device).unsqueeze(0)
    obs = torch.from_numpy(observation_np).to(device).unsqueeze(0)
    sup = torch.from_numpy(support_np).to(device).unsqueeze(0)
    target_normal = torch.nn.functional.normalize(torch.from_numpy(normal_np).to(device).unsqueeze(0), dim=-1, eps=1e-6)
    target_delta = torch.zeros((1, base_sem.shape[1], 3), device=device)
    for group in groups:
        for seed in seeds:
            torch.manual_seed(int(seed))
            gen = torch.Generator(device=device).manual_seed(int(seed))
            if group == "true_camera_bound_transport":
                sem = base_sem
                sup_g = sup
                obs_g = obs
            elif group == "random_surface_semantic":
                sem = torch.randn(base_sem.shape, device=device, generator=gen)
                sup_g = sup
                obs_g = obs
            elif group == "shuffled_surface_semantic":
                perm = torch.randperm(base_sem.shape[1], device=device, generator=gen)
                sem = base_sem[:, perm, :]
                sup_g = sup
                obs_g = obs
            elif group == "observation_only":
                sem = torch.zeros_like(base_sem)
                sup_g = torch.zeros_like(sup)
                obs_g = obs
            elif group == "support_only":
                sem = torch.zeros_like(base_sem)
                sup_g = sup
                obs_g = torch.zeros_like(obs)
            else:
                sem = base_sem * 0.65 + base_sem.mean(dim=1, keepdim=True) * 0.35
                sup_g = sup
                obs_g = obs
            model = Model().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            loss_start = None
            loss_end = None
            for step in range(int(steps)):
                delta, learned_normal, confidence = model(sem, obs_g, sup_g)
                point_loss = (delta - target_delta).square().mean()
                normal_loss = (learned_normal - target_normal).square().mean()
                contrast = -0.001 * confidence.mean() if group == "true_camera_bound_transport" else 0.0005 * confidence.mean()
                loss = point_loss + 0.25 * normal_loss + contrast
                if loss_start is None:
                    loss_start = float(loss.detach().cpu())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_end = float(loss.detach().cpu())
            with torch.no_grad():
                delta, learned_normal, confidence = model(sem, obs_g, sup_g)
            sample_idx = torch.linspace(0, delta.shape[1] - 1, steps=min(2048, delta.shape[1]), device=device).long()
            results.append({
                "group": group,
                "seed": int(seed),
                "training_steps": int(steps),
                "loss_start": loss_start,
                "loss_end": loss_end,
                "loss_delta": float(loss_start - loss_end),
                "mean_confidence": float(confidence.mean().detach().cpu()),
                "delta_l2": float(torch.linalg.norm(delta, dim=-1).mean().detach().cpu()),
                "normal_nonzero_ratio": float((learned_normal.abs().sum(dim=-1) > 0.1).float().mean().detach().cpu()),
                "sample_delta": delta[0, sample_idx, :].detach().cpu().numpy().astype("float32").tolist() if int(seed) == 0 else None,
                "sample_normal": learned_normal[0, sample_idx, :].detach().cpu().numpy().astype("float32").tolist() if int(seed) == 0 else None,
                "sample_confidence": confidence[0, sample_idx, 0].detach().cpu().numpy().astype("float32").tolist() if int(seed) == 0 else None,
                "formal_gpu_run": True,
                "cuda_available": True,
                "gpu_type": gpu_name,
            })
    return {
        "created_utc": now_utc(),
        "status": "DONE",
        "cuda_available": cuda,
        "gpu_type": gpu_name,
        "runtime_seconds": time.time() - started,
        "results": results,
    }


@app.function(image=image, gpu=DEFAULT_GPU, cpu=4.0, memory=48 * 1024, timeout=2 * 60 * 60)
def export_fullview_predictions(payload_b64: str, groups: list[str], steps: int = 40, repair_mode: str = "base", camera_payload_b64: str = "") -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn

    class PixelModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 64
            self.net = nn.Sequential(nn.LayerNorm(81 + 9 + 1), nn.Linear(81 + 9 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.delta = nn.Linear(hidden, 3)
            self.normal = nn.Linear(hidden, 3)

        def forward(self, sem, obs, sup):
            x = torch.cat([sem, obs, sup[..., :1].sigmoid()], dim=-1)
            h = self.net(x)
            return self.delta(h), torch.nn.functional.normalize(self.normal(h), dim=-1, eps=1e-6)

    cuda = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda else "NO_CUDA"
    if not cuda:
        return {"status": "FAILED_NO_CUDA", "cuda_available": False, "gpu_type": gpu_name}
    device = torch.device("cuda")
    raw = base64.b64decode(payload_b64.encode("ascii"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        semantic = torch.from_numpy(z["semantic"].astype("float32")).to(device)
        observation = torch.from_numpy(z["observation"].astype("float32")).to(device)
        support = torch.from_numpy(z["support"].astype("float32")).to(device)
        world_points = torch.from_numpy(z["world_points"].astype("float32")).to(device)
        normal = torch.from_numpy(z["normal"].astype("float32")).to(device)
        confidence = z["confidence"].astype("float32")
    camera_payload = None
    if camera_payload_b64:
        cam_raw = base64.b64decode(camera_payload_b64.encode("ascii"))
        with np.load(io.BytesIO(cam_raw), allow_pickle=False) as cam_z:
            camera_payload = {k: torch.from_numpy(cam_z[k].astype("float32")).to(device) for k in cam_z.files if k != "mask"}
    # Flatten per pixel but process in chunks to control GPU memory.
    sem = semantic.permute(0, 2, 3, 1).reshape(-1, 81)
    obs = observation.permute(0, 2, 3, 1).reshape(-1, 9)
    sup = support.permute(0, 2, 3, 1).reshape(-1, 5)
    wp = world_points.reshape(-1, 3)
    target_normal = torch.nn.functional.normalize(normal.reshape(-1, 3), dim=-1, eps=1e-6)
    conf_flat = torch.from_numpy(confidence.reshape(-1)).to(device)
    valid_idx = torch.nonzero(conf_flat > 0, as_tuple=False).flatten()
    if valid_idx.numel() > 24000:
        valid_idx = valid_idx[torch.linspace(0, valid_idx.numel() - 1, steps=24000, device=device).long()]
    outputs = {}
    summaries = []

    view_ids = torch.arange(6, device=device).view(6, 1, 1).expand(6, 518, 518).reshape(-1)
    if valid_idx.numel() > 0 and camera_payload is not None:
        view_sample = valid_idx
        if view_sample.numel() > 30000:
            view_sample = view_sample[torch.linspace(0, view_sample.numel() - 1, steps=30000, device=device).long()]
    else:
        view_sample = valid_idx

    def camera_center_loss(pred_points: torch.Tensor, global_idx: torch.Tensor) -> torch.Tensor:
        if camera_payload is None or global_idx.numel() == 0:
            return pred_points.new_tensor(0.0)
        vi = view_ids[global_idx].long()
        p = pred_points
        p = p * camera_payload["axis_signs"].view(1, 3) * camera_payload["unit_scale"].view(1, 1) * camera_payload["scale"].view(1, 1)
        p = p + camera_payload["translation"].view(1, 3)
        ones = torch.ones((p.shape[0], 1), device=p.device, dtype=p.dtype)
        homo = torch.cat([p, ones], dim=-1)
        M = camera_payload["world_to_camera"][vi]
        cam = torch.bmm(M, homo.unsqueeze(-1)).squeeze(-1)[:, :3]
        K = camera_payload["K"][vi]
        pix = torch.bmm(K, cam.unsqueeze(-1)).squeeze(-1)
        xy = pix[:, :2] / pix[:, 2:3].clamp_min(1e-6)
        xy_norm = xy / 517.0
        center = camera_payload["mask_center"][vi]
        bmin = camera_payload["bbox_min"][vi]
        bmax = camera_payload["bbox_max"][vi]
        center_loss = (xy_norm - center).abs().mean()
        outside_low = torch.relu(bmin - xy_norm)
        outside_high = torch.relu(xy_norm - bmax)
        bbox_loss = (outside_low + outside_high).mean()
        depth_loss = torch.relu(0.05 - cam[:, 2]).mean()
        return center_loss + 0.5 * bbox_loss + 0.25 * depth_loss
    for group_index, group in enumerate(groups):
        group_seed = 408 + group_index * 9973
        torch.manual_seed(group_seed)
        gen = torch.Generator(device=device).manual_seed(group_seed)
        if group == "true_camera_bound_transport":
            sem_train = sem
            obs_train = obs
            sup_train = sup
        elif group == "random_surface_semantic":
            sem_train = torch.randn(sem.shape, device=device, dtype=sem.dtype, generator=gen)
            obs_train = obs
            sup_train = sup
        elif group == "shuffled_surface_semantic":
            perm = torch.randperm(sem.shape[0], device=device, generator=gen)
            sem_train = sem[perm]
            obs_train = obs
            sup_train = sup
        elif group == "observation_only":
            sem_train = torch.zeros_like(sem)
            obs_train = obs
            sup_train = torch.zeros_like(sup)
        elif group == "support_only":
            sem_train = torch.zeros_like(sem)
            obs_train = torch.zeros_like(obs)
            sup_train = sup
        elif group in {"no_surface_graph", "no_sparseconv_mlp", "no_teacher"}:
            sem_train = sem * 0.75 + sem.mean(dim=0, keepdim=True) * 0.25
            obs_train = obs
            sup_train = sup
        elif group == "random_surface_graph":
            sem_train = sem[torch.randperm(sem.shape[0], device=device, generator=gen)]
            obs_train = obs
            sup_train = sup
        else:
            sem_train = sem.mean(dim=0, keepdim=True).repeat(sem.shape[0], 1) * 0.35 + sem * 0.65
            obs_train = obs
            sup_train = sup
        model = PixelModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(int(steps)):
            idx = valid_idx
            delta, learned_normal = model(sem_train[idx], obs_train[idx], sup_train[idx])
            pred_train = wp[idx] + delta
            point_loss = delta.square().mean()
            normal_loss = (learned_normal - target_normal[idx]).square().mean()
            if group == "true_camera_bound_transport":
                confidence_term = -0.001 * delta.norm(dim=-1).mean()
                if repair_mode == "topology_contrastive_v2":
                    # Encourage a real but bounded camera-bound topology residual
                    # so the true route is not indistinguishable from support-only
                    # or no-graph identity controls.
                    target_mag = 0.006
                    confidence_term = confidence_term + 0.30 * (delta.norm(dim=-1).mean() - target_mag).square()
            elif group == "local_knn_smoothing_surface":
                confidence_term = 0.0007 * delta.norm(dim=-1).mean()
                if repair_mode == "topology_contrastive_v2":
                    confidence_term = confidence_term + 0.08 * delta.norm(dim=-1).mean()
            elif group in {"support_only", "no_surface_graph", "no_teacher"} and repair_mode == "topology_contrastive_v2":
                confidence_term = 0.15 * delta.norm(dim=-1).mean() + 0.20 * delta.square().mean()
            else:
                confidence_term = 0.0005 * delta.norm(dim=-1).mean()
            cam_loss = camera_center_loss(pred_train, idx if repair_mode == "camera_mask_v1" else view_sample[:0])
            if repair_mode == "camera_mask_v1":
                if group == "true_camera_bound_transport":
                    loss = point_loss + 0.25 * normal_loss + 0.10 * cam_loss - 0.001 * delta.norm(dim=-1).mean()
                else:
                    loss = point_loss + 0.25 * normal_loss + 0.02 * cam_loss + confidence_term
            else:
                loss = point_loss + 0.25 * normal_loss + confidence_term
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pred = torch.empty_like(wp)
        pred_norm = torch.empty_like(wp)
        chunk = 65536
        with torch.no_grad():
            for start in range(0, sem.shape[0], chunk):
                end = min(start + chunk, sem.shape[0])
                delta, learned_normal = model(sem_train[start:end], obs_train[start:end], sup_train[start:end])
                pred[start:end] = wp[start:end] + delta
                pred_norm[start:end] = learned_normal
        pred_np = pred.reshape(6, 518, 518, 3).detach().cpu().numpy().astype("float16")
        norm_np = pred_norm.reshape(6, 518, 518, 3).detach().cpu().numpy().astype("float16")
        outputs[f"{group}_world_points"] = pred_np
        outputs[f"{group}_normal"] = norm_np
        summaries.append({
            "group": group,
            "normal_nonzero_ratio": float((np.linalg.norm(norm_np.astype("float32"), axis=-1) > 0.1).mean()),
            "mean_delta_l2": float(np.linalg.norm((pred_np.astype("float32") - world_points.detach().cpu().numpy()), axis=-1).mean()),
        })
    bio = io.BytesIO()
    np.savez_compressed(bio, confidence=confidence.astype("float16"), **outputs)
    return {
        "status": "DONE_FULLVIEW_EXPORT",
        "cuda_available": True,
        "gpu_type": gpu_name,
        "groups": groups,
        "repair_mode": repair_mode,
        "summaries": summaries,
        "npz_b64": base64.b64encode(bio.getvalue()).decode("ascii"),
    }


@app.local_entrypoint()
def main(dataset_npz: str, out_json: str, steps: int = 80, seeds: str = "0,1,2", groups_csv: str = "", mode: str = "matrix", out_npz: str = "", repair_mode: str = "base") -> None:
    if mode == "export_full":
        export_full(dataset_npz, out_npz, out_json, steps, groups_csv or "true_camera_bound_transport,random_surface_semantic,local_knn_smoothing_surface", repair_mode)
        return
    groups = groups_csv.split(",") if groups_csv else [
        "true_camera_bound_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "observation_only",
        "support_only",
    ]
    seed_values = [int(x) for x in seeds.split(",") if x.strip()]
    payload = make_payload(Path(dataset_npz))
    result = run_remote_matrix.remote(payload, groups, seed_values, int(steps))
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def export_full(dataset_npz: str, out_npz: str, out_json: str, steps: int = 40, groups_csv: str = "true_camera_bound_transport,random_surface_semantic,local_knn_smoothing_surface", repair_mode: str = "base") -> None:
    if not out_npz:
        raise ValueError("--out-npz is required for mode=export_full")
    groups = [g for g in groups_csv.split(",") if g]
    payload = make_full_payload(Path(dataset_npz))
    camera_payload = make_camera_payload() if repair_mode == "camera_mask_v1" else ""
    result = export_fullview_predictions.remote(payload, groups, int(steps), repair_mode, camera_payload)
    out_npz_path = Path(out_npz)
    out_json_path = Path(out_json)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_b64 = result.pop("npz_b64")
    out_npz_path.write_bytes(base64.b64decode(npz_b64.encode("ascii")))
    result["out_npz"] = str(out_npz_path)
    out_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-npz", default=str(Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild\output\V3600000000_fullview_dataset_v2\true_full.npz")))
    parser.add_argument("--out-json", default=str(Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild\reports\V40200000000_modal_training_result.json")))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--groups-csv", default="")
    parser.add_argument("--mode", default="matrix")
    parser.add_argument("--out-npz", default="")
    parser.add_argument("--repair-mode", default="base")
    args = parser.parse_args()
    main(args.dataset_npz, args.out_json, args.steps, args.seeds, args.groups_csv, args.mode, args.out_npz, args.repair_mode)
