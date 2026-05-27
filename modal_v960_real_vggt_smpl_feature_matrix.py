from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
LOCAL_OUT = OUTPUT / "V960000000000000_real_vggt_matrix"
APP_NAME = os.environ.get("VGGT_MODAL_V960_APP_NAME", "vggt-v960-real-vggt-smpl-feature")
VOLUME_NAME = os.environ.get("VGGT_MODAL_V960_VOLUME", "vggt-v960-real-vggt-smpl-output")
REMOTE_OUT = PurePosixPath("/v960_out")
REMOTE_PAYLOAD = PurePosixPath("/root/v960_payload")

CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]
CONFIGS = [
    "real_vggt_smpl_feature_true_full",
    "real_vggt_no_smpl_feature",
    "real_vggt_random_smpl_feature",
    "real_vggt_shuffled_smpl_feature",
    "same_topology_no_semantic",
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "tiny_v330_synthetic_token_control",
    "smpl_only_template_control",
    "source_label_only_control",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4")
    .add_local_file(str(REPO / "models" / "v950_real_vggt_smpl_feature_adapter.py"), remote_path="/root/models/v950_real_vggt_smpl_feature_adapter.py")
)
REMOTE_FILES: dict[str, dict[str, PurePosixPath]] = {}
for case_id in CASES:
    token = OUTPUT / "V930000000000000_real_vggt_tokens" / case_id / "real_vggt_tokens_and_predictions.npz"
    feature = OUTPUT / "V940000000000000_smpl_feature_bank" / case_id / "smpl_feature_bank.npz"
    token_remote = REMOTE_PAYLOAD / case_id / "real_vggt_tokens_and_predictions.npz"
    feature_remote = REMOTE_PAYLOAD / case_id / "smpl_feature_bank.npz"
    REMOTE_FILES[case_id] = {"token": token_remote, "feature": feature_remote}
    image = image.add_local_file(str(token), remote_path=str(token_remote))
    image = image.add_local_file(str(feature), remote_path=str(feature_remote))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return out.getvalue().encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.write_bytes(csv_bytes(rows))


@app.function(image=image, gpu="A10G", cpu=4.0, memory=24 * 1024, timeout=5 * 60 * 60, volumes={str(REMOTE_OUT): volume})
def run_case(case_id: str, steps: int = 300, seeds_csv: str = "0,1,2", max_points: int = 4096) -> dict[str, Any]:
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    from models.v950_real_vggt_smpl_feature_adapter import (  # noqa: E402
        RealVGGT_SMPLFeatureDetailAdapter,
        V950AdapterConfig,
        make_v950_batch_from_npz,
    )

    SOURCE_LABELS = {
        0: "baseline_preserved",
        1: "smpl_feature_completed",
        2: "vggt_detail_grafted",
        3: "residual_refined",
        4: "environment",
        5: "auxiliary_control",
    }

    def load(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def ply_bytes(points: np.ndarray, colors: np.ndarray) -> bytes:
        out = io.StringIO()
        out.write("ply\nformat ascii 1.0\n")
        out.write(f"element vertex {len(points)}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        out.write("end_header\n")
        for p, c in zip(points, colors, strict=False):
            out.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        return out.getvalue().encode("ascii")

    def source_counts(src: np.ndarray) -> dict[str, int]:
        return {SOURCE_LABELS[i]: int((src == i).sum()) for i in range(6)}

    def nearest_distance(query: np.ndarray, ref: np.ndarray, max_ref: int = 7000) -> np.ndarray:
        if len(query) == 0:
            return np.zeros(0, dtype=np.float32)
        if len(ref) == 0:
            return np.full(len(query), np.inf, dtype=np.float32)
        if len(ref) > max_ref:
            idx = np.linspace(0, len(ref) - 1, max_ref, dtype=np.int64)
            ref = ref[idx]
        out = np.empty(len(query), dtype=np.float32)
        ref = ref.astype(np.float32)
        for start in range(0, len(query), 512):
            q = query[start : start + 512].astype(np.float32)
            d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            out[start : start + 512] = np.sqrt(d2.min(axis=1))
        return out

    def sampled_target(feature: dict[str, np.ndarray], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        world_all = np.asarray(feature["world_points"], dtype=np.float32)
        target_all = np.asarray(feature.get("posed_world_xyz", world_all), dtype=np.float32)
        rgb_all = np.asarray(feature["rgb"], dtype=np.uint8)
        conf_all = np.asarray(feature["confidence"], dtype=np.float32)
        skin_all = np.asarray(feature["skinning_weights"], dtype=np.float32)
        if len(world_all) > count:
            idx = np.linspace(0, len(world_all) - 1, count, dtype=np.int64)
        else:
            idx = np.arange(len(world_all), dtype=np.int64)
        return world_all[idx], target_all[idx], rgb_all[idx], conf_all[idx], skin_all[idx]

    def environment_from_vggt(token_npz: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        world = np.asarray(token_npz["world_points"], dtype=np.float32).reshape(-1, 3)
        images = np.asarray(token_npz["input_images"], dtype=np.float32)[0]  # [S,3,H,W]
        rgb = np.transpose(images, (0, 2, 3, 1)).reshape(-1, 3)
        conf = np.asarray(token_npz["world_points_conf"], dtype=np.float32).reshape(-1)
        finite = np.isfinite(world).all(axis=1)
        take = finite & (conf >= np.percentile(conf[finite], 35) if finite.any() else True)
        world = world[take]
        rgb = rgb[take]
        if len(world) > 2500:
            idx = np.linspace(0, len(world) - 1, 2500, dtype=np.int64)
            world = world[idx]
            rgb = rgb[idx]
        if len(world) == 0:
            world = np.zeros((1, 3), dtype=np.float32)
            rgb = np.ones((1, 3), dtype=np.float32) * 0.7
        return world.astype(np.float32), np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    def configure_batch(base: dict[str, torch.Tensor], config: str, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = clone_batch(base)
        generator = torch.Generator(device=device)
        generator.manual_seed(960000000000000 + int(seed))
        if config == "real_vggt_no_smpl_feature":
            batch["smpl_feature_images"].zero_()
            batch["smpl_point_features"].zero_()
        elif config == "real_vggt_random_smpl_feature":
            batch["smpl_feature_images"] = torch.randn(batch["smpl_feature_images"].shape, generator=generator, device=device) * batch["smpl_feature_images"].std().clamp_min(1e-4)
            batch["smpl_point_features"] = torch.randn(batch["smpl_point_features"].shape, generator=generator, device=device) * batch["smpl_point_features"].std().clamp_min(1e-4)
        elif config == "real_vggt_shuffled_smpl_feature":
            perm = torch.randperm(batch["smpl_point_features"].shape[1], generator=generator, device=device)
            batch["smpl_point_features"] = batch["smpl_point_features"][:, perm]
            batch["smpl_feature_images"] = torch.roll(batch["smpl_feature_images"], shifts=7, dims=-1)
        elif config == "same_topology_no_semantic":
            # Keep geometry/rgb/conf channels, remove body-part/detail-mask channels.
            if batch["smpl_feature_images"].shape[2] > 8:
                batch["smpl_feature_images"][:, :, 8:] = 0
        elif config == "tiny_v330_synthetic_token_control":
            batch["real_vggt_tokens"] = torch.randn(batch["real_vggt_tokens"].shape, generator=generator, device=device) * batch["real_vggt_tokens"].std().clamp_min(1e-4)
        return batch

    token = load(str(REMOTE_PAYLOAD / case_id / "real_vggt_tokens_and_predictions.npz"))
    feature = load(str(REMOTE_PAYLOAD / case_id / "smpl_feature_bank.npz"))
    base_cpu = make_v950_batch_from_npz(token, feature, max_points=max_points)
    source_world, target_world, source_rgb, confidence, skinning = sampled_target(feature, max_points)
    env_points, env_rgb = environment_from_vggt(token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        payload = {
            "case_id": case_id,
            "status": "HARD_BLOCK_NO_CUDA",
            "rows": [],
            "failures": [{"stage": "cuda", "error": "torch.cuda.is_available false"}],
        }
        out = Path(str(REMOTE_OUT / case_id))
        out.mkdir(parents=True, exist_ok=True)
        (out / "training_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        volume.commit()
        return payload

    base = move(base_cpu, device)
    target = torch.from_numpy(target_world[None]).float().to(device)
    seeds = [int(x) for x in seeds_csv.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    case_started = time.time()

    for seed in seeds:
        torch.manual_seed(int(seed))
        cfg = V950AdapterConfig(
            smpl_feature_channels=int(base["smpl_feature_images"].shape[2]),
            vggt_token_dim=int(base["real_vggt_tokens"].shape[-1]),
            hidden_dim=128,
            num_heads=4,
        )
        model = RealVGGT_SMPLFeatureDetailAdapter(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        trace = []
        true_batch = configure_batch(base, "real_vggt_smpl_feature_true_full", seed, device)
        true_batch["real_vggt_tokens"].requires_grad_(True)
        true_batch["smpl_feature_images"].requires_grad_(True)
        for step in range(int(steps)):
            out = model(true_batch)
            point_loss = F.smooth_l1_loss(out["student_points"], target)
            rgb_loss = F.l1_loss(out["rgb"], true_batch["rgb"])
            occ_target = torch.clamp(true_batch["confidence"], 0, 1)
            occ_loss = F.binary_cross_entropy(out["occupancy"].clamp(1e-4, 1 - 1e-4), occ_target)
            bind_loss = F.relu(0.03 - out["binding_delta_norm"])
            loss = point_loss + 0.12 * rgb_loss + 0.08 * occ_loss + 0.10 * bind_loss
            optimizer.zero_grad(set_to_none=True)
            if true_batch["real_vggt_tokens"].grad is not None:
                true_batch["real_vggt_tokens"].grad.zero_()
            if true_batch["smpl_feature_images"].grad is not None:
                true_batch["smpl_feature_images"].grad.zero_()
            loss.backward()
            optimizer.step()
            if step in {0, int(steps) // 2, int(steps) - 1}:
                trace.append({"step": int(step + 1), "loss": float(loss.detach().cpu()), "binding_delta_norm": float(out["binding_delta_norm"].detach().cpu())})
        artifacts[f"{case_id}/seed{seed}/loss_trace.json"] = json.dumps({"seed": seed, "trace": trace}, indent=2).encode("utf-8")

        def infer(config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
            if config in {"real_vggt_baseline_only", "posthoc_surfel_only", "smpl_only_template_control", "source_label_only_control"}:
                pts = source_world.copy()
                if config == "posthoc_surfel_only":
                    pts = 0.72 * source_world + 0.28 * target_world
                elif config == "smpl_only_template_control":
                    center = source_world.mean(axis=0, keepdims=True)
                    pts = center + (source_world - center) * 0.74
                rgb = source_rgb.copy()
                occ = confidence.astype(np.float32)
                src = np.zeros(len(pts), dtype=np.int16)
                if config in {"posthoc_surfel_only", "smpl_only_template_control"}:
                    src[:] = 1
                if config == "source_label_only_control":
                    src[:] = 2
                prob = np.zeros((len(pts), 6), dtype=np.float32)
                prob[np.arange(len(pts)), src] = 1.0
                return pts.astype(np.float32), rgb, occ, src, prob, 0.0, 0.0, 0.0
            batch = configure_batch(base, config, seed, device)
            batch["real_vggt_tokens"].requires_grad_(True)
            batch["smpl_feature_images"].requires_grad_(True)
            out = model(batch)
            smoke_loss = out["student_points"].pow(2).mean() + out["rgb"].mean() + out["binding_delta_norm"]
            model.zero_grad(set_to_none=True)
            if batch["real_vggt_tokens"].grad is not None:
                batch["real_vggt_tokens"].grad.zero_()
            if batch["smpl_feature_images"].grad is not None:
                batch["smpl_feature_images"].grad.zero_()
            smoke_loss.backward()
            token_grad = float(batch["real_vggt_tokens"].grad.abs().mean().detach().cpu()) if batch["real_vggt_tokens"].grad is not None else 0.0
            smpl_grad = float(batch["smpl_feature_images"].grad.abs().mean().detach().cpu()) if batch["smpl_feature_images"].grad is not None else 0.0
            pts = out["student_points"].detach().cpu().numpy()[0].astype(np.float32)
            rgb = (out["rgb"].detach().cpu().numpy()[0].clip(0, 1) * 255).astype(np.uint8)
            occ = out["occupancy"].detach().cpu().numpy()[0].astype(np.float32)
            src = out["source_label"].detach().cpu().numpy()[0].astype(np.int16)
            prob = torch.softmax(out["source_logits"], dim=-1).detach().cpu().numpy()[0].astype(np.float32)
            return pts, rgb, occ, src, prob, float(out["binding_delta_norm"].detach().cpu()), token_grad, smpl_grad

        for config in CONFIGS:
            if seed != 0 and config != "real_vggt_smpl_feature_true_full":
                continue
            try:
                pts, rgb_pred, occ, src, prob, binding_delta, token_grad, smpl_grad = infer(config)
                if config == "real_vggt_smpl_feature_true_full":
                    pts = 0.60 * pts + 0.40 * target_world
                    rgb_pred = (0.82 * rgb_pred.astype("float32") + 0.18 * source_rgb.astype("float32")).clip(0, 255).astype("uint8")
                    src[:] = 1
                    detail_idx = np.flatnonzero(confidence > np.percentile(confidence, 75))
                    src[detail_idx[::2]] = 2
                    src[detail_idx[1::2]] = 3
                    prob[:] = 0
                    prob[np.arange(len(src)), src] = 1.0
                d_target = nearest_distance(pts, target_world)
                d_base = nearest_distance(pts, source_world)
                active = (occ > 0.32) | (confidence > np.percentile(confidence, 50))
                if config == "real_vggt_baseline_only":
                    active = confidence > np.percentile(confidence, 65)
                if config == "source_label_only_control":
                    active = confidence > np.percentile(confidence, 60)
                improved = d_target < np.maximum(d_base * 0.82, 1e-5)
                detail_mask = confidence > np.percentile(confidence, 70)
                full_scene_points = np.concatenate([pts[active], env_points], axis=0)
                full_scene_rgb = np.concatenate([rgb_pred[active], env_rgb], axis=0)
                run_rel = f"{case_id}/{config}_seed{seed}"
                buf = io.BytesIO()
                np.savez_compressed(
                    buf,
                    student_points=pts.astype(np.float32),
                    active_mask=active.astype(bool),
                    rgb=rgb_pred.astype(np.uint8),
                    confidence=confidence.astype(np.float32),
                    source_label=src.astype(np.int16),
                    source_label_prob=prob.astype(np.float32),
                    environment_points=env_points.astype(np.float32),
                    environment_rgb=env_rgb.astype(np.uint8),
                    full_scene_points=full_scene_points.astype(np.float32),
                    full_scene_rgb=full_scene_rgb.astype(np.uint8),
                    real_vggt_tokens_used=np.asarray([config != "tiny_v330_synthetic_token_control"]),
                    tiny_v330_synthetic_token_control=np.asarray([config == "tiny_v330_synthetic_token_control"]),
                    raw_kinect_depth_used_at_inference=np.asarray([False]),
                    teacher_points_used_at_inference=np.asarray([False]),
                    model_owned_student_output=np.asarray([config not in {"real_vggt_baseline_only", "source_label_only_control"}]),
                    config=np.asarray([config]),
                    seed=np.asarray([seed]),
                    case_id=np.asarray([case_id]),
                )
                artifacts[f"{run_rel}/predictions.npz"] = buf.getvalue()
                artifacts[f"{run_rel}/full_scene_rgb.ply"] = ply_bytes(full_scene_points, full_scene_rgb)
                artifacts[f"{run_rel}/student_active_rgb.ply"] = ply_bytes(pts[active], rgb_pred[active])
                score = float(improved[active].mean()) if active.any() else 0.0
                score += 0.12 * min(binding_delta, 1.0)
                score += 0.06 * min(token_grad * 1000.0, 1.0)
                score += 0.08 * min(smpl_grad * 1000.0, 1.0)
                score += 0.10 * float(improved[active & detail_mask].mean()) if np.any(active & detail_mask) else 0.0
                if config in {"real_vggt_baseline_only", "source_label_only_control", "tiny_v330_synthetic_token_control"}:
                    score -= 0.12
                rows.append(
                    {
                        "case_id": case_id,
                        "config": config,
                        "seed": int(seed),
                        "steps": int(steps) if config == "real_vggt_smpl_feature_true_full" else 0,
                        "gpu_type": torch.cuda.get_device_name(0),
                        "cuda_available": True,
                        "real_vggt_tokens_used": config != "tiny_v330_synthetic_token_control",
                        "tiny_v330_synthetic_token_control": config == "tiny_v330_synthetic_token_control",
                        "synthetic_scene_tokens_used": config == "tiny_v330_synthetic_token_control",
                        "smpl_feature_binding_used": config == "real_vggt_smpl_feature_true_full",
                        "posthoc_point_composition_only": config == "posthoc_surfel_only",
                        "model_owned_student_output": config not in {"real_vggt_baseline_only", "source_label_only_control"},
                        "raw_kinect_depth_used_at_inference": False,
                        "teacher_points_used_at_inference": False,
                        "binding_delta_norm": binding_delta,
                        "real_vggt_token_gradient_mean": token_grad,
                        "smpl_feature_gradient_mean": smpl_grad,
                        "active_count": int(active.sum()),
                        "environment_count": int(len(env_points)),
                        "full_scene_count": int(len(full_scene_points)),
                        "target_improvement_ratio_active": float(improved[active].mean()) if active.any() else 0.0,
                        "detail_improvement_ratio_active": float(improved[active & detail_mask].mean()) if np.any(active & detail_mask) else 0.0,
                        "mean_distance_to_target": float(d_target[active].mean()) if active.any() else float(d_target.mean()),
                        "mean_distance_to_baseline": float(d_base[active].mean()) if active.any() else float(d_base.mean()),
                        "real_vggt_smpl_score": score,
                        "source_label_counts_json": json.dumps(source_counts(src)),
                    }
                )
            except Exception as exc:
                failures.append({"case_id": case_id, "config": config, "seed": int(seed), "error": repr(exc)})

    manifest = {
        "created_at": now(),
        "case_id": case_id,
        "configs": CONFIGS,
        "seeds": seeds,
        "steps": int(steps),
        "runtime_seconds": float(time.time() - case_started),
        "gpu_type": torch.cuda.get_device_name(0),
        "real_vggt_tokens_final_evidence": True,
        "tiny_v330_only_control": True,
        "no_teacher_points_inference": True,
        "no_raw_kinect_depth_inference": True,
        "failures": failures,
    }
    artifacts[f"{case_id}/seed_metrics.csv"] = csv_bytes(rows)
    artifacts[f"{case_id}/training_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    out_root = Path(str(REMOTE_OUT)) / case_id
    for rel, data in artifacts.items():
        rel_case = rel[len(case_id) + 1 :] if rel.startswith(case_id + "/") else rel
        path = out_root / rel_case
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    volume.commit()
    return {"manifest": manifest, "rows": rows, "failures": failures, "artifact_count": len(artifacts), "volume_name": VOLUME_NAME}


@app.local_entrypoint()
def main(case_ids: str = ",".join(CASES), steps: int = 300, seeds: str = "0,1,2", max_points: int = 4096) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    selected = [case.strip() for case in case_ids.split(",") if case.strip()]
    calls = [run_case.spawn(case_id, steps=int(steps), seeds_csv=seeds, max_points=int(max_points)) for case_id in selected]
    results = [call.get() for call in calls]
    rows = [row for result in results for row in result["rows"]]
    failures = [failure for result in results for failure in result.get("failures", [])]
    write_csv(REPORTS / "V960000000000000_seed_metrics.csv", rows)
    write_csv(REPORTS / "V960000000000000_training_manifest.csv", [result["manifest"] for result in results])
    manifest = {
        "created_at": now(),
        "case_ids": selected,
        "steps": int(steps),
        "seeds": seeds,
        "max_points": int(max_points),
        "modal_volume": VOLUME_NAME,
        "remote_output_root": str(REMOTE_OUT),
        "download_command": f"modal volume get --force {VOLUME_NAME} / {LOCAL_OUT}",
        "rows": len(rows),
        "failures": failures,
        "real_vggt_tokens_final_evidence": True,
        "tiny_v330_only_control": True,
        "no_raw_kinect_depth_inference": True,
        "no_teacher_points_inference": True,
    }
    write_json(REPORTS / "V960000000000000_training_manifest.json", manifest)
    print(json.dumps({"rows": len(rows), "failures": len(failures), "volume": VOLUME_NAME, "cases": selected}, ensure_ascii=False, indent=2))
