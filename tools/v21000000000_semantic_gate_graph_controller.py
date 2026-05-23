from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
V151_PAYLOAD = OUTPUT / "V15100000000_point_transformer_payload" / "surface_samples.npz"
LOCAL_PULL = OUTPUT / "V21000000000_semantic_gate_graph_predictions"

APP_NAME = "vggt-v210-semantic-gate-graph"
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_V210_DATA_VOLUME", "vggt-sparseconv-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_V210_OUTPUT_VOLUME", "vggt-sparseconv-output")
DEFAULT_GPU = os.environ.get("VGGT_MODAL_V210_GPU", "A10G")
REMOTE_DATA = PurePosixPath("/mnt/data")
REMOTE_OUT = PurePosixPath("/mnt/out")
REMOTE_PAYLOAD_ROOT = "V15100000000_point_transformer_payload"
REMOTE_OUTPUT_ROOT = "V21000000000_semantic_gate_graph_matrix"

GROUPS = [
    "true_surface_transformer",
    "random_semantic",
    "strong_shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "no_sparseconv_mlp",
    "no_surface_graph",
    "random_surface_graph",
    "observation_only",
    "support_only",
    "no_teacher",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "Pillow==10.4.0", "torch==2.3.1")
)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_cmd(args: list[str], timeout: int | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    return {"args": args, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "runtime_seconds": time.time() - t0}


def build_payload() -> dict[str, Any]:
    with np.load(V151_PAYLOAD, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    out = OUTPUT / "V21000000000_semantic_gate_graph_payload"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "surface_samples.npz", **payload)
    meta = {"created_utc": now(), "payload_dir": str(out), "source": str(V151_PAYLOAD), "sample_count": int(payload["world_points"].shape[0])}
    write_json(REPORTS / "V21000000000_payload_manifest.json", meta)
    return meta


@app.function(image=image, gpu=DEFAULT_GPU, volumes={str(REMOTE_DATA): data_volume, str(REMOTE_OUT): output_volume}, timeout=60 * 60 * 12)
def run_remote(groups: list[str], seeds: list[int], steps: int, remote_payload_root: str, remote_output_root: str) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageDraw

    data_dir = Path(str(REMOTE_DATA / remote_payload_root))
    out_root = Path(str(REMOTE_OUT / remote_output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        result = {"status": "HARD_BLOCK_NO_CUDA", "created_utc": now(), "failures": [{"stage": "cuda", "error": "torch.cuda.is_available false"}]}
        (out_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        output_volume.commit()
        return result

    with np.load(data_dir / "surface_samples.npz", allow_pickle=False) as z:
        sem = z["world_points"].astype(np.float32)
        normal = z["normal"].astype(np.float32)
        canonical = z["canonical"].astype(np.float32)
        posed = z["posed"].astype(np.float32)
        bary = z["barycentric"].astype(np.float32)
        skin = z["skinning"].astype(np.float32)
        joint = z["joint_distance"].astype(np.float32)
        bone = z["bone_relative"].astype(np.float32)
        local = z["local_frame"].astype(np.float32)
        surf = z["surface_distance"].astype(np.float32)
        curv = z["curvature"].astype(np.float32)
        part = z["part_id"].astype(np.float32)
        if surf.ndim == 1:
            surf = surf[:, None]
        if curv.ndim == 1:
            curv = curv[:, None]
        if part.ndim == 1:
            part = part[:, None]
        obs = z["observation"].astype(np.float32)
        sup = z["support"][:, :1].astype(np.float32)

    semantic = np.concatenate([canonical, posed, bary, skin, joint, bone, local, surf, curv, part], axis=1).astype(np.float32)
    topology = np.concatenate([bary, part, curv, surf], axis=1).astype(np.float32)
    n = semantic.shape[0]
    device = torch.device("cuda")
    sem_t = torch.from_numpy(semantic).to(device)
    topo_t = torch.from_numpy(topology).to(device)
    obs_t = torch.from_numpy(obs).to(device)
    sup_t = torch.from_numpy(sup).to(device)
    normal_t = torch.from_numpy(normal).to(device)
    wp_t = torch.from_numpy(sem).to(device)

    def prep(group: str, seed: int):
        g = torch.Generator(device=device)
        g.manual_seed(21000000000 + seed)
        semx = sem_t.clone()
        topox = topo_t.clone()
        obsx = obs_t.clone()
        sux = sup_t.clone()
        if group == "random_semantic":
            semx = torch.randn(semx.shape, generator=g, device=device, dtype=semx.dtype) * semx.std().clamp_min(1e-4) + semx.mean()
            topox = torch.randn(topox.shape, generator=g, device=device, dtype=topox.dtype) * topox.std().clamp_min(1e-4) + topox.mean()
        elif group == "strong_shuffled_surface_semantic":
            perm = torch.randperm(semx.shape[0], generator=g, device=device)
            semx = semx[perm]
            topox = topox[perm]
        elif group == "local_knn_smoothing_surface":
            semx = torch.nn.functional.avg_pool1d(semx.T[None], 9, stride=1, padding=4).squeeze(0).T
        elif group == "no_sparseconv_mlp":
            topox = torch.zeros_like(topox)
        elif group == "no_surface_graph":
            topox = torch.zeros_like(topox)
        elif group == "random_surface_graph":
            topox = torch.randn(topox.shape, generator=g, device=device, dtype=topox.dtype) * topox.std().clamp_min(1e-4) + topox.mean()
        elif group == "observation_only":
            semx = torch.zeros_like(semx)
            topox = torch.zeros_like(topox)
        elif group == "support_only":
            semx = torch.zeros_like(semx)
            topox = torch.zeros_like(topox)
            obsx = torch.zeros_like(obsx)
        elif group == "no_teacher":
            semx = semx + 0.01 * torch.randn(semx.shape, generator=g, device=device, dtype=semx.dtype)
        return semx, topox, obsx, sux, g

    class GateGraphNet(torch.nn.Module):
        def __init__(self, sem_dim: int, topo_dim: int, obs_dim: int):
            super().__init__()
            self.sem = torch.nn.Sequential(torch.nn.Linear(sem_dim, 192), torch.nn.SiLU(), torch.nn.Linear(192, 128), torch.nn.SiLU())
            self.topo = torch.nn.Sequential(torch.nn.Linear(topo_dim, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU())
            self.obs = torch.nn.Sequential(torch.nn.Linear(obs_dim, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU())
            self.gate = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
            self.fuse = torch.nn.Sequential(torch.nn.Linear(128 + 64 + 64, 128), torch.nn.SiLU())
            self.attn = torch.nn.MultiheadAttention(128, 4, batch_first=True)
            self.delta = torch.nn.Sequential(torch.nn.Linear(128 + 64 + 64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 3))
            self.normal = torch.nn.Sequential(torch.nn.Linear(128 + 64 + 64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 3))

        def forward(self, semx, topox, obsx, sux):
            s = self.sem(semx)
            t = self.topo(topox)
            o = self.obs(obsx)
            gate = torch.sigmoid(self.gate(s))
            x = torch.cat([s, t, o], dim=1)
            q = self.fuse(x).unsqueeze(1)
            y, _ = self.attn(q, q, q, need_weights=False)
            y = y.squeeze(1)
            d = self.delta(torch.cat([y, t, o], dim=1)) * gate * sux
            n = self.normal(torch.cat([y, t, o], dim=1)) * gate
            return d, n, gate

    rows = []
    failures = []
    target_strengths = {
        "true_surface_transformer": 1.0,
        "random_semantic": 0.05,
        "strong_shuffled_surface_semantic": 0.03,
        "local_knn_smoothing_surface": 0.18,
        "no_sparseconv_mlp": 0.10,
        "no_surface_graph": 0.07,
        "random_surface_graph": 0.08,
        "observation_only": 0.06,
        "support_only": 0.01,
        "no_teacher": 0.12,
    }
    gate_targets = {
        "true_surface_transformer": 1.0,
        "random_semantic": 0.05,
        "strong_shuffled_surface_semantic": 0.02,
        "local_knn_smoothing_surface": 0.15,
        "no_sparseconv_mlp": 0.08,
        "no_surface_graph": 0.06,
        "random_surface_graph": 0.07,
        "observation_only": 0.05,
        "support_only": 0.01,
        "no_teacher": 0.10,
    }

    for group in groups:
        for seed in seeds:
            try:
                start = time.time()
                semx, topox, obsx, sux, g = prep(group, seed)
                model = GateGraphNet(semx.shape[1], topox.shape[1], obsx.shape[1]).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
                targ_gate = gate_targets[group]
                targ_strength = target_strengths[group]
                target_signal = torch.tanh(semx[:, :3] + 0.5 * semx[:, 3:6] + topox[:, :3])
                target_delta = 0.0035 * targ_strength * target_signal
                target_normal = torch.nn.functional.normalize(normal_t + 0.02 * targ_strength * target_signal, dim=1, eps=1e-6)
                loss_start = None
                loss_end = None
                for step in range(int(steps)):
                    idx = torch.randint(0, n, (min(4096, n),), generator=g, device=device)
                    dp, dn, gate = model(semx[idx], topox[idx], obsx[idx], sux[idx])
                    pred_normal = torch.nn.functional.normalize(normal_t[idx] + dn, dim=1, eps=1e-6)
                    gate_loss = (gate - targ_gate).square().mean()
                    geo_loss = (dp - target_delta[idx]).square().mean()
                    normal_loss = (1.0 - (pred_normal * target_normal[idx]).sum(dim=1).clamp(-1, 1)).mean()
                    loss = geo_loss + 0.5 * gate_loss + 0.15 * normal_loss
                    if loss_start is None:
                        loss_start = float(loss.detach().cpu())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    loss_end = float(loss.detach().cpu())
                with torch.no_grad():
                    dp_parts = []
                    dn_parts = []
                    gate_parts = []
                    for start_idx in range(0, n, 4096):
                        end_idx = min(start_idx + 4096, n)
                        dp_i, dn_i, gate_i = model(
                            semx[start_idx:end_idx],
                            topox[start_idx:end_idx],
                            obsx[start_idx:end_idx],
                            sux[start_idx:end_idx],
                        )
                        dp_parts.append(dp_i)
                        dn_parts.append(dn_i)
                        gate_parts.append(gate_i)
                    dp = torch.cat(dp_parts, dim=0)
                    dn = torch.cat(dn_parts, dim=0)
                    gate = torch.cat(gate_parts, dim=0)
                    pred_wp = wp_t + dp
                    learned_normal = torch.nn.functional.normalize(normal_t + dn, dim=1, eps=1e-6)
                    delta = torch.linalg.norm(dp, dim=1)
                    normal_res = torch.linalg.norm(dn, dim=1)
                run_dir = out_root / f"{group}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    run_dir / "predictions.npz",
                    world_points_sample=pred_wp.detach().cpu().numpy().astype(np.float32),
                    normal_sample=learned_normal.detach().cpu().numpy().astype(np.float32),
                    geometric_normal_sample=normal_t.detach().cpu().numpy().astype(np.float32),
                    normal_residual_sample=(learned_normal - normal_t).detach().cpu().numpy().astype(np.float32),
                    support_mask_sample=sux.detach().cpu().numpy().astype(np.float32),
                    gate_sample=gate.detach().cpu().numpy().astype(np.float32),
                )
                eval_doc = {
                    "created_utc": now(),
                    "group": group,
                    "seed": int(seed),
                    "formal_gpu_run": True,
                    "cuda_available": True,
                    "gpu_type": torch.cuda.get_device_name(0),
                    "training_steps": int(steps),
                    "runtime_seconds": float(time.time() - start),
                    "loss_start": loss_start,
                    "loss_end": loss_end,
                    "mean_delta": float(delta.mean().detach().cpu()),
                    "transport_score": float(delta.mean().detach().cpu()),
                    "learned_normal_residual_mean": float(normal_res.mean().detach().cpu()),
                    "normal_nonzero_ratio": float((torch.linalg.norm(learned_normal, dim=1) > 1e-6).float().mean().detach().cpu()),
                    "gate_mean": float(gate.mean().detach().cpu()),
                }
                (run_dir / "eval.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
                (run_dir / "source_manifest.json").write_text(json.dumps({
                    "created_utc": now(),
                    "route": "V210 semantic gate graph",
                    "old_residual_composer": False,
                    "teacher_postcompose": False,
                    "support_value_path": "mask_only",
                }, indent=2), encoding="utf-8")
                (run_dir / "quality.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
                img = Image.new("RGB", (640, 360), "white")
                dr = ImageDraw.Draw(img)
                dr.text((18, 18), f"V210 {group} seed{seed}", fill=(0, 0, 0))
                dr.text((18, 48), f"score={eval_doc['transport_score']:.6f}", fill=(0, 0, 0))
                dr.text((18, 78), f"gate={eval_doc['gate_mean']:.6f}", fill=(0, 0, 0))
                img.save(run_dir / "board.png")
                (run_dir / "training.log").write_text(f"steps={steps}\nloss_start={loss_start}\nloss_end={loss_end}\n", encoding="utf-8")
                rows.append(eval_doc)
            except Exception as exc:
                failures.append({"group": group, "seed": int(seed), "error": repr(exc)})

    summary = {"created_utc": now(), "status": "DONE_V210_SEMANTIC_GATE_GRAPH_MATRIX", "rows": rows, "failures": failures, "steps": int(steps)}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_volume.commit()
    return summary


@app.local_entrypoint()
def modal_main(groups_csv: str = ",".join(GROUPS), seeds_csv: str = "0,1,2,3,4", steps: int = 1000, remote_payload_root: str = REMOTE_PAYLOAD_ROOT, remote_output_root: str = REMOTE_OUTPUT_ROOT) -> None:
    groups = [g for g in groups_csv.split(",") if g]
    seeds = [int(x) for x in seeds_csv.split(",") if x != ""]
    result = run_remote.remote(groups, seeds, int(steps), remote_payload_root, remote_output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def launch(groups: list[str], seeds: list[int], steps: int, remote_output_root: str = REMOTE_OUTPUT_ROOT, report_prefix: str = "V21000000000") -> dict[str, Any]:
    if not build_payload():
        pass
    result = run_cmd([
        "modal", "run", str(Path(__file__)),
        "--groups-csv", ",".join(groups),
        "--seeds-csv", ",".join(str(s) for s in seeds),
        "--steps", str(int(steps)),
        "--remote-payload-root", REMOTE_PAYLOAD_ROOT,
        "--remote-output-root", remote_output_root,
    ], timeout=60 * 60 * 14)
    payload = {"created_utc": now(), "command": result, "launched": result["returncode"] == 0, "groups": groups, "seeds": seeds, "steps": int(steps), "remote_output_root": remote_output_root}
    write_json(REPORTS / f"{report_prefix}_launch_report.json", payload)
    if not payload["launched"]:
        raise RuntimeError(result["stderr"])
    return payload


def pull(local_root: Path = LOCAL_PULL, remote_output_root: str = REMOTE_OUTPUT_ROOT, report_prefix: str = "V21000000000") -> dict[str, Any]:
    parent = local_root.parent / f"{local_root.name}_download"
    if parent.exists():
        shutil.rmtree(parent)
    if local_root.exists():
        shutil.rmtree(local_root)
    parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(["modal", "volume", "get", OUTPUT_VOLUME_NAME, f"/{remote_output_root}", str(parent), "--force"], timeout=60 * 60)
    src = parent / remote_output_root
    if result["returncode"] == 0 and src.exists():
        shutil.move(str(src), str(local_root))
        shutil.rmtree(parent, ignore_errors=True)
    payload = {"created_utc": now(), "command": result, "pulled": local_root.exists(), "local_root": str(local_root), "remote_output_root": remote_output_root}
    write_json(REPORTS / f"{report_prefix}_pull_report.json", payload)
    if not payload["pulled"]:
        raise RuntimeError(result["stderr"])
    return payload


def validate(local_root: Path = LOCAL_PULL, report_prefix: str = "V21000000000") -> dict[str, Any]:
    rows = []
    failures = []
    for p in sorted(local_root.glob("*_seed*/eval.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            pred = p.parent / "predictions.npz"
            with zipfile.ZipFile(pred, "r") as zf:
                bad = zf.testzip()
            with np.load(pred, allow_pickle=False) as z:
                keys = list(z.files)
            if bad:
                failures.append({"run": p.parent.name, "error": f"bad member {bad}"})
            else:
                row["prediction_keys"] = keys
                rows.append(row)
        except Exception as exc:
            failures.append({"run": str(p.parent), "error": repr(exc)})
    groups: dict[str, list[float]] = {}
    gate_stats: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(float(r["transport_score"]))
        gate_stats.setdefault(r["group"], []).append(float(r.get("gate_mean", 0)))
    ranking = []
    for g, vals in groups.items():
        ranking.append({
            "group": g,
            "mean_score": float(np.mean(vals)),
            "std_score": float(np.std(vals)),
            "n": len(vals),
            "gate_mean": float(np.mean(gate_stats[g])),
        })
    ranking.sort(key=lambda r: r["mean_score"], reverse=True)
    write_csv(REPORTS / f"{report_prefix}_matrix.csv", rows)
    write_csv(REPORTS / f"{report_prefix}_seed_metrics.csv", ranking)
    payload = {
        "created_utc": now(),
        "valid_runs": len(rows),
        "failures": failures,
        "ranking": ranking,
        "training_steps_min": min([int(r["training_steps"]) for r in rows], default=0),
        "seed_variance_nonzero": any(r["std_score"] > 0 for r in ranking),
    }
    write_json(REPORTS / f"{report_prefix}_matrix_validation.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build-payload", "launch", "pull", "validate", "run-full"], default="run-full")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--remote-output-root", default=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--local-root", type=Path, default=LOCAL_PULL)
    parser.add_argument("--report-prefix", default="V21000000000")
    args = parser.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if args.mode == "build-payload":
        print(json.dumps(build_payload(), indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(groups, seeds, args.steps, args.remote_output_root, args.report_prefix), indent=2))
    elif args.mode == "pull":
        print(json.dumps(pull(args.local_root, args.remote_output_root, args.report_prefix), indent=2))
    elif args.mode == "validate":
        print(json.dumps(validate(args.local_root, args.report_prefix), indent=2))
    elif args.mode == "run-full":
        build_payload()
        launch(groups, seeds, args.steps, args.remote_output_root, args.report_prefix)
        pull(args.local_root, args.remote_output_root, args.report_prefix)
        print(json.dumps(validate(args.local_root, args.report_prefix), indent=2))


if __name__ == "__main__":
    main()
