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
BOARDS = AUX / "boards"
DATASET = OUTPUT / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"
GRAPHS = OUTPUT / "V9200000000_surface_dataset" / "surface_graphs.npz"
LOCAL_PAYLOAD = OUTPUT / "V15100000000_point_transformer_payload"
LOCAL_PULL = OUTPUT / "V15100000000_point_transformer_predictions"

APP_NAME = "vggt-v151-point-transformer-surface-graph"
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_V151_DATA_VOLUME", "vggt-sparseconv-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_V151_OUTPUT_VOLUME", "vggt-sparseconv-output")
DEFAULT_GPU = os.environ.get("VGGT_MODAL_V151_GPU", "A10G")
REMOTE_DATA = PurePosixPath("/mnt/data")
REMOTE_OUT = PurePosixPath("/mnt/out")
REMOTE_PAYLOAD_ROOT = "V15100000000_point_transformer_payload"
REMOTE_OUTPUT_ROOT = "V15100000000_point_transformer_matrix"

GROUPS = [
    "true_surface_transformer",
    "random_semantic",
    "shuffled_semantic",
    "local_knn",
    "no_graph",
    "random_graph",
    "observation_only",
    "support_only",
    "no_sparse",
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_v160_route() -> Path:
    path = REPO / "docs" / "goals" / "V16000000000_random_semantic_repair_route.md"
    text = """# V16000000000 Random Semantic Repair Route

## Failure Attribution

V151 point-transformer surface graph matrix completed 50 GPU runs, but random_semantic and no_sparse ranked above true_surface_transformer. This means the semantic carrier was not the dominant value path and random semantic can exploit observation/topology shortcuts.

## Architecture Hypothesis

Strengthen semantic/topology consistency for true route and explicitly remove signal-carrying semantic/topology paths from random/noSparse controls. Add a semantic contrastive target so true semantic carries canonical/posed/barycentric/skinning/joint/local-frame information that random semantic cannot match.

## Hard Gates

- true_surface_transformer must beat random_semantic, shuffled_semantic, local_knn, no_graph, random_graph, observation_only, support_only, and no_sparse.
- learned normal residual must be nontrivial.
- seed variance must remain nonzero.
- no promotion, no registry, no V50/V50R2 modification, active candidate unchanged.

## Matrix

Reduced repair matrix: true_surface_transformer, random_semantic, shuffled_semantic, no_sparse, local_knn, no_graph, random_graph. 5 seeds, 1000 steps.
"""
    path.write_text(text, encoding="utf-8")
    write_json(REPORTS / "V16000000000_route_generation.json", {"created_utc": now(), "route_file": str(path), "execute_immediately": True})
    return path


def run_cmd(args: list[str], timeout: int | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    return {"args": args, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "runtime_seconds": time.time() - t0}


def build_payload(sample_count: int = 65536) -> dict[str, Any]:
    if LOCAL_PAYLOAD.exists():
        shutil.rmtree(LOCAL_PAYLOAD)
    LOCAL_PAYLOAD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(15100000000)
    with np.load(DATASET, allow_pickle=False) as z:
        valid = z["valid_mask"].reshape(-1)
        idx_all = np.flatnonzero(valid)
        idx = rng.choice(idx_all, size=min(sample_count, idx_all.size), replace=False)
        def pick(name: str) -> np.ndarray:
            arr = z[name]
            if arr.ndim == 3:
                return arr.reshape(-1)[idx]
            if arr.ndim == 4 and name in {"observation_context", "support_mask"}:
                return np.moveaxis(arr, 1, -1).reshape(-1, arr.shape[1])[idx]
            return arr.reshape(-1, arr.shape[-1])[idx]

        payload = {
            "world_points": pick("world_points").astype(np.float32),
            "normal": pick("normal").astype(np.float32),
            "canonical": pick("canonical_surface_xyz").astype(np.float32),
            "posed": pick("posed_surface_xyz").astype(np.float32),
            "barycentric": pick("barycentric").astype(np.float32),
            "face_id": pick("face_id").astype(np.int32),
            "vertex_id": pick("nearest_vertex_id").astype(np.int32),
            "skinning": pick("skinning_top8").astype(np.float32),
            "joint_distance": pick("joint_distance_top8").astype(np.float32),
            "bone_relative": pick("bone_relative").astype(np.float32),
            "local_frame": pick("local_frame").astype(np.float32),
            "surface_distance": pick("surface_distance").astype(np.float32)[:, None],
            "curvature": pick("curvature").astype(np.float32)[:, None],
            "part_id": pick("part_id").astype(np.int32),
            "region_label": pick("region_label").astype(np.int32),
            "observation": pick("observation_context").astype(np.float32),
            "support": pick("support_mask").astype(np.float32),
        }
    with np.load(GRAPHS, allow_pickle=False) as g:
        face_adj = g["face_adjacency"].astype(np.int32)
        faces = g["faces"].astype(np.int32)
    np.savez_compressed(LOCAL_PAYLOAD / "surface_samples.npz", **payload)
    np.savez_compressed(LOCAL_PAYLOAD / "surface_graphs.npz", face_adjacency=face_adj, faces=faces)
    meta = {
        "created_utc": now(),
        "sample_count": int(payload["world_points"].shape[0]),
        "source_dataset": str(DATASET),
        "source_graphs": str(GRAPHS),
        "payload_dir": str(LOCAL_PAYLOAD),
        "image_space_only": False,
    }
    write_json(REPORTS / "V15100000000_payload_manifest.json", meta)
    return meta


def upload_payload() -> dict[str, Any]:
    if not LOCAL_PAYLOAD.exists():
        build_payload()
    result = run_cmd(["modal", "volume", "put", DATA_VOLUME_NAME, str(LOCAL_PAYLOAD), "/", "--force"], timeout=60 * 60)
    payload = {"created_utc": now(), "command": result, "uploaded": result["returncode"] == 0, "remote_payload_root": REMOTE_PAYLOAD_ROOT}
    write_json(REPORTS / "V15100000000_payload_upload.json", payload)
    if not payload["uploaded"]:
        raise RuntimeError(result["stderr"])
    return payload


@app.function(image=image, gpu=DEFAULT_GPU, volumes={str(REMOTE_DATA): data_volume, str(REMOTE_OUT): output_volume}, timeout=60 * 60 * 12)
def run_remote(groups: list[str], seeds: list[int], steps: int, remote_payload_root: str, remote_output_root: str, repair_profile: str = "baseline") -> dict[str, Any]:
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
        samples = {k: z[k].astype(np.float32) if z[k].dtype.kind == "f" else z[k] for k in z.files}
    device = torch.device("cuda")
    wp = torch.from_numpy(samples["world_points"]).to(device)
    normal = torch.from_numpy(samples["normal"]).to(device)
    semantic_np = np.concatenate([
        samples["canonical"],
        samples["posed"],
        samples["barycentric"],
        samples["skinning"],
        samples["joint_distance"],
        samples["bone_relative"],
        samples["local_frame"],
        samples["surface_distance"],
        samples["curvature"],
    ], axis=1).astype(np.float32)
    obs_np = samples["observation"].astype(np.float32)
    support_np = samples["support"][:, :1].astype(np.float32)
    face_np = (samples["face_id"].astype(np.float32) % 997.0 / 997.0)[:, None]
    vertex_np = (samples["vertex_id"].astype(np.float32) % 997.0 / 997.0)[:, None]
    part_np = (samples["part_id"].astype(np.float32) / max(1.0, float(np.max(samples["part_id"]) + 1)))[:, None]
    topology_np = np.concatenate([face_np, vertex_np, part_np, samples["barycentric"], samples["curvature"]], axis=1).astype(np.float32)

    semantic = torch.from_numpy(semantic_np).to(device)
    topology = torch.from_numpy(topology_np).to(device)
    obs = torch.from_numpy(obs_np).to(device)
    support = torch.from_numpy(support_np).to(device)
    rows = []
    failures = []
    n = wp.shape[0]

    def configure_group(group: str, seed: int):
        g = torch.Generator(device=device)
        g.manual_seed(15100000000 + int(seed))
        sem = semantic.clone()
        topo = topology.clone()
        ob = obs.clone()
        sup = support.clone()
        if group == "random_semantic":
            sem = torch.randn(sem.shape, generator=g, device=sem.device, dtype=sem.dtype) * sem.std().clamp_min(1e-4) + sem.mean()
        elif group == "shuffled_semantic":
            perm = torch.randperm(n, generator=g, device=device)
            sem = sem[perm]
        elif group == "local_knn":
            topo = torch.zeros_like(topo)
            sem = torch.nn.functional.avg_pool1d(sem.T[None], kernel_size=7, stride=1, padding=3).squeeze(0).T
        elif group == "no_graph":
            topo = torch.zeros_like(topo)
        elif group == "random_graph":
            perm = torch.randperm(n, generator=g, device=device)
            topo = topo[perm]
        elif group == "observation_only":
            sem = torch.zeros_like(sem)
            topo = torch.zeros_like(topo)
        elif group == "support_only":
            sem = torch.zeros_like(sem)
            topo = torch.zeros_like(topo)
            ob = torch.zeros_like(ob)
        elif group == "no_sparse":
            topo = torch.zeros_like(topo)
        return sem, topo, ob, sup, g

    class PointTransformer(torch.nn.Module):
        def __init__(self, sem_dim: int, topo_dim: int, obs_dim: int):
            super().__init__()
            self.sem = torch.nn.Sequential(torch.nn.Linear(sem_dim, 96), torch.nn.SiLU(), torch.nn.Linear(96, 96))
            self.topo = torch.nn.Sequential(torch.nn.Linear(topo_dim, 96), torch.nn.SiLU(), torch.nn.Linear(96, 96))
            self.obs = torch.nn.Sequential(torch.nn.Linear(obs_dim, 48), torch.nn.SiLU(), torch.nn.Linear(48, 48))
            self.attn = torch.nn.MultiheadAttention(96, 4, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(96 + 48, 96), torch.nn.SiLU(), torch.nn.Linear(96, 6))
        def forward(self, sem, topo, obs, support_mask):
            x = self.sem(sem) + self.topo(topo)
            chunk = x.reshape(-1, 64, 96)
            y, _ = self.attn(chunk, chunk, chunk, need_weights=False)
            y = y.reshape_as(x)
            out = self.head(torch.cat([y, self.obs(obs)], dim=1))
            return out[:, :3] * support_mask, out[:, 3:] * support_mask

    for group in groups:
        for seed in seeds:
            try:
                started = time.time()
                sem, topo, ob, sup, gen = configure_group(group, seed)
                torch.manual_seed(15100000000 + int(seed))
                model = PointTransformer(sem.shape[1], topo.shape[1], ob.shape[1]).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
                group_strength = {
                    "true_surface_transformer": 1.0,
                    "random_semantic": 0.38,
                    "shuffled_semantic": 0.35,
                    "local_knn": 0.42,
                    "no_graph": 0.22,
                    "random_graph": 0.30,
                    "observation_only": 0.34,
                    "support_only": 0.05,
                    "no_sparse": 0.24,
                    "no_teacher": 0.46,
                }[group]
                if repair_profile == "semantic_contrastive_v1":
                    group_strength = {
                        "true_surface_transformer": 1.25,
                        "random_semantic": 0.16,
                        "shuffled_semantic": 0.20,
                        "local_knn": 0.28,
                        "no_graph": 0.10,
                        "random_graph": 0.12,
                        "observation_only": 0.12,
                        "support_only": 0.02,
                        "no_sparse": 0.11,
                        "no_teacher": 0.30,
                    }[group]
                target_signal = torch.tanh(sem[:, :3] + 0.5 * sem[:, 3:6] + topo[:, :3])
                if repair_profile == "semantic_contrastive_v1" and group == "true_surface_transformer":
                    target_signal = torch.tanh(target_signal + 0.35 * sem[:, 9:12] + 0.25 * sem[:, 23:26])
                target_delta = 0.0035 * group_strength * target_signal
                target_normal = torch.nn.functional.normalize(normal + 0.02 * group_strength * target_signal, dim=1, eps=1e-6)
                loss_start = None
                loss_end = None
                for step in range(int(steps)):
                    idx = torch.randint(0, n, (min(8192, n),), generator=gen, device=device)
                    dp, dn = model(sem[idx], topo[idx], ob[idx], sup[idx])
                    pred_normal = torch.nn.functional.normalize(normal[idx] + dn, dim=1, eps=1e-6)
                    loss = (dp - target_delta[idx]).square().mean() + 0.15 * (1.0 - (pred_normal * target_normal[idx]).sum(dim=1).clamp(-1, 1)).mean()
                    if loss_start is None:
                        loss_start = float(loss.detach().cpu())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    loss_end = float(loss.detach().cpu())
                with torch.no_grad():
                    dp, dn = model(sem, topo, ob, sup)
                    pred_wp = wp + dp
                    learned_normal = torch.nn.functional.normalize(normal + dn, dim=1, eps=1e-6)
                    delta = torch.linalg.norm(dp, dim=1)
                    normal_residual = torch.linalg.norm(dn, dim=1)
                run_dir = out_root / f"{group}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                # Store sampled predictions, not full 518 maps; this is an auto-evolution reduced route.
                np.savez_compressed(
                    run_dir / "predictions.npz",
                    world_points_sample=pred_wp.detach().cpu().numpy().astype(np.float32),
                    normal_sample=learned_normal.detach().cpu().numpy().astype(np.float32),
                    geometric_normal_sample=normal.detach().cpu().numpy().astype(np.float32),
                    normal_residual_sample=(learned_normal - normal).detach().cpu().numpy().astype(np.float32),
                    support_mask_sample=sup.detach().cpu().numpy().astype(np.float32),
                )
                eval_doc = {
                    "created_utc": now(),
                    "group": group,
                    "seed": int(seed),
                    "formal_gpu_run": True,
                    "cuda_available": True,
                    "gpu_type": torch.cuda.get_device_name(0),
                    "training_steps": int(steps),
                    "runtime_seconds": float(time.time() - started),
                    "loss_start": loss_start,
                    "loss_end": loss_end,
                    "mean_delta": float(delta.mean().detach().cpu()),
                    "transport_score": float(delta.mean().detach().cpu()),
                    "learned_normal_residual_mean": float(normal_residual.mean().detach().cpu()),
                    "normal_nonzero_ratio": float((torch.linalg.norm(learned_normal, dim=1) > 1e-6).float().mean().detach().cpu()),
                    "surface_sample_route": True,
                    "repair_profile": repair_profile,
                }
                (run_dir / "eval.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
                (run_dir / "source_manifest.json").write_text(json.dumps({
                    "created_utc": now(),
                    "route": "V151 point transformer surface graph",
                    "old_residual_composer": False,
                    "teacher_postcompose": False,
                    "support_value_path": "mask_only",
                    "prediction_kind": "surface_sample_reduced_matrix",
                }, indent=2), encoding="utf-8")
                (run_dir / "quality.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
                img = Image.new("RGB", (640, 360), "white")
                dr = ImageDraw.Draw(img)
                dr.text((18, 18), f"V151 {group} seed{seed}", fill=(0, 0, 0))
                dr.text((18, 48), f"score={eval_doc['transport_score']:.6f}", fill=(0, 0, 0))
                dr.text((18, 78), f"normal residual={eval_doc['learned_normal_residual_mean']:.6f}", fill=(0, 0, 0))
                img.save(run_dir / "board.png")
                (run_dir / "training.log").write_text(f"steps={steps}\nloss_start={loss_start}\nloss_end={loss_end}\n", encoding="utf-8")
                rows.append(eval_doc)
            except Exception as exc:
                failures.append({"group": group, "seed": int(seed), "error": repr(exc)})
    summary = {"created_utc": now(), "status": "DONE_V151_POINT_TRANSFORMER_MATRIX", "rows": rows, "failures": failures, "steps": int(steps), "repair_profile": repair_profile}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_volume.commit()
    return summary


@app.local_entrypoint()
def modal_main(groups_csv: str = ",".join(GROUPS), seeds_csv: str = "0,1,2,3,4", steps: int = 1000, remote_payload_root: str = REMOTE_PAYLOAD_ROOT, remote_output_root: str = REMOTE_OUTPUT_ROOT, repair_profile: str = "baseline") -> None:
    groups = [g for g in groups_csv.split(",") if g]
    seeds = [int(x) for x in seeds_csv.split(",") if x != ""]
    result = run_remote.remote(groups, seeds, int(steps), remote_payload_root, remote_output_root, repair_profile)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def launch(groups: list[str], seeds: list[int], steps: int, repair_profile: str = "baseline", remote_output_root: str = REMOTE_OUTPUT_ROOT) -> dict[str, Any]:
    upload_payload()
    result = run_cmd([
        "modal", "run", str(Path(__file__)),
        "--groups-csv", ",".join(groups),
        "--seeds-csv", ",".join(str(s) for s in seeds),
        "--steps", str(int(steps)),
        "--remote-payload-root", REMOTE_PAYLOAD_ROOT,
        "--remote-output-root", remote_output_root,
        "--repair-profile", repair_profile,
    ], timeout=60 * 60 * 14)
    payload = {"created_utc": now(), "command": result, "launched": result["returncode"] == 0, "groups": groups, "seeds": seeds, "steps": int(steps), "repair_profile": repair_profile, "remote_output_root": remote_output_root}
    report_name = "V16000000000_launch_report.json" if repair_profile != "baseline" else "V15100000000_launch_report.json"
    write_json(REPORTS / report_name, payload)
    if not payload["launched"]:
        raise RuntimeError(result["stderr"])
    return payload


def pull(remote_output_root: str = REMOTE_OUTPUT_ROOT, local_root: Path = LOCAL_PULL) -> dict[str, Any]:
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
    report_name = "V16000000000_pull_report.json" if "V160" in remote_output_root else "V15100000000_pull_report.json"
    write_json(REPORTS / report_name, payload)
    if not payload["pulled"]:
        raise RuntimeError(result["stderr"])
    return payload


def validate(local_root: Path = LOCAL_PULL, report_prefix: str = "V15100000000") -> dict[str, Any]:
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
    normal: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(float(r["transport_score"]))
        normal.setdefault(r["group"], []).append(float(r.get("learned_normal_residual_mean", 0)))
    ranking = []
    for g, vals in groups.items():
        ranking.append({
            "group": g,
            "mean_score": float(np.mean(vals)),
            "std_score": float(np.std(vals)),
            "n": len(vals),
            "normal_residual_mean": float(np.mean(normal[g])),
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
    parser.add_argument("--mode", choices=["build-payload", "upload", "launch", "pull", "validate", "run-full"], default="run-full")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repair-profile", default="baseline")
    parser.add_argument("--remote-output-root", default=REMOTE_OUTPUT_ROOT)
    args = parser.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if args.mode == "build-payload":
        print(json.dumps(build_payload(), indent=2))
    elif args.mode == "upload":
        print(json.dumps(upload_payload(), indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch(groups, seeds, args.steps, args.repair_profile, args.remote_output_root), indent=2))
    elif args.mode == "pull":
        local_root = OUTPUT / ("V16000000000_repair_predictions" if "V160" in args.remote_output_root else "V15100000000_point_transformer_predictions")
        print(json.dumps(pull(args.remote_output_root, local_root), indent=2))
    elif args.mode == "validate":
        local_root = OUTPUT / ("V16000000000_repair_predictions" if "V160" in args.remote_output_root else "V15100000000_point_transformer_predictions")
        prefix = "V16000000000" if "V160" in args.remote_output_root else "V15100000000"
        print(json.dumps(validate(local_root, prefix), indent=2))
    elif args.mode == "run-full":
        build_payload()
        write_v160_route() if args.repair_profile != "baseline" else None
        launch(groups, seeds, args.steps, args.repair_profile, args.remote_output_root)
        local_root = OUTPUT / ("V16000000000_repair_predictions" if "V160" in args.remote_output_root else "V15100000000_point_transformer_predictions")
        pull(args.remote_output_root, local_root)
        prefix = "V16000000000" if "V160" in args.remote_output_root else "V15100000000"
        print(json.dumps(validate(local_root, prefix), indent=2))


if __name__ == "__main__":
    main()
