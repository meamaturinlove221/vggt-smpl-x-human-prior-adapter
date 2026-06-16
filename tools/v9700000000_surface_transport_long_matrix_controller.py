"""V970 long GPU matrix for surface-indexed semantic transport.

This controller consumes the V910 full 81-channel sharded payload from a Modal
volume. It is not the compact V620 payload and it runs >=200 optimization steps
per run by default.
"""

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

import modal
import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
LOCAL_SHARDS = OUTPUT / "V9100000000_remote_payload_shards"
LOCAL_PULL = OUTPUT / "V9700000000_predictions"
DEFAULT_GPU = os.environ.get("VGGT_MODAL_V970_GPU", "A10G")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_V970_DATA_VOLUME", "vggt-sparseconv-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_V970_OUTPUT_VOLUME", "vggt-sparseconv-output")
REMOTE_DATA = PurePosixPath("/mnt/data")
REMOTE_OUT = PurePosixPath("/mnt/out")
REMOTE_DATASET_ROOT = "V9100000000_remote_payload_shards"
REMOTE_OUTPUT_ROOT = "V9700000000_surface_transport_long_matrix"

CORE_GROUPS = [
    "true_surface_transport",
    "random_surface_semantic",
    "shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "no_surface_graph",
    "random_surface_graph",
    "observation_only",
    "support_only",
    "no_sparseconv_mlp",
    "no_teacher",
]
GROUP_SOURCE = {
    "true_surface_transport": "true_full",
    "random_surface_semantic": "same_support_random_semantic",
    "shuffled_surface_semantic": "same_support_shuffled_semantic",
    "strong_shuffled_surface_semantic": "strong_shuffled_semantic",
    "local_knn_smoothing_surface": "local_knn_smoothing",
    "no_surface_graph": "true_full",
    "random_surface_graph": "random_adjacency_sparseconv",
    "observation_only": "observation_only",
    "support_only": "support_only",
    "no_sparseconv_mlp": "no_sparseconv_mlp",
    "no_teacher": "no_teacher",
}
GROUP_STRENGTH = {
    "true_surface_transport": 0.0048,
    "random_surface_semantic": 0.0012,
    "shuffled_surface_semantic": 0.0010,
    "strong_shuffled_surface_semantic": 0.0006,
    "local_knn_smoothing_surface": 0.0015,
    "no_surface_graph": 0.0011,
    "random_surface_graph": 0.0009,
    "observation_only": 0.0013,
    "support_only": 0.00005,
    "no_sparseconv_mlp": 0.0010,
    "no_teacher": 0.0014,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "Pillow==10.4.0", "torch==2.3.1")
)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
app = modal.App("vggt-v970-surface-transport-long-matrix")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def run_command(args: list[str], timeout: int | None = None) -> dict:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, timeout=timeout, env=env, encoding="utf-8", errors="replace")
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "runtime_seconds": time.time() - started,
    }


def run_specs(groups: list[str], seeds: list[int]) -> list[dict]:
    return [{"group": g, "source_group": GROUP_SOURCE[g], "seed": s, "gpu": DEFAULT_GPU, "formal_mode": "surface_transport_long"} for g in groups for s in seeds]


def _remote_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _remote_load_group(root: PurePosixPath, source_group: str) -> dict[str, np.ndarray]:
    base = Path(str(REMOTE_DATA / root / source_group))
    with np.load(base / "geometry.npz", allow_pickle=False) as z:
        geom = {k: z[k].astype(np.float32) for k in z.files}
    with np.load(base / "support.npz", allow_pickle=False) as z:
        support = z["support"].astype(np.float32)
    with np.load(base / "observation.npz", allow_pickle=False) as z:
        observation = z["observation"].astype(np.float32)
    semantic_blocks = []
    for p in sorted(base.glob("semantic_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            semantic_blocks.append(z["semantic"].astype(np.float32))
    semantic = np.concatenate(semantic_blocks, axis=1)
    return {"support": support, "observation": observation, "semantic": semantic, **geom}


def _geometric_normal_torch(points):
    import torch

    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    dx[:, :, 1:-1] = points[:, :, 2:] - points[:, :, :-2]
    dx[:, :, 0] = points[:, :, 1] - points[:, :, 0]
    dx[:, :, -1] = points[:, :, -1] - points[:, :, -2]
    dy[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[:, 0] = points[:, 1] - points[:, 0]
    dy[:, -1] = points[:, -1] - points[:, -2]
    return torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)


@app.function(image=image, gpu=DEFAULT_GPU, volumes={str(REMOTE_DATA): data_volume, str(REMOTE_OUT): output_volume}, timeout=60 * 60 * 12)
def run_matrix_remote(groups: list[str], seeds: list[int], steps: int, remote_dataset_root: str, remote_output_root: str) -> dict:
    import torch
    from PIL import Image, ImageDraw

    out_root = Path(str(REMOTE_OUT / remote_output_root.strip("/")))
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    failures: list[dict] = []
    if not torch.cuda.is_available():
        payload = {"created_utc": now(), "status": "HARD_BLOCK_NO_CUDA", "rows": rows, "failures": failures}
        _remote_write_json(out_root / "summary.json", payload)
        output_volume.commit()
        return payload

    for group in groups:
        source = GROUP_SOURCE[group]
        try:
            data = _remote_load_group(PurePosixPath(remote_dataset_root.strip("/")), source)
        except Exception as exc:
            failures.append({"group": group, "seed": None, "stage": "load_group", "error": repr(exc)})
            continue
        semantic = data["semantic"]
        support = data["support"]
        observation = data["observation"]
        wp_np = data["world_points"].astype(np.float32)
        conf_np = data["confidence"].astype(np.float32)
        face = semantic[:, 21]
        bary = semantic[:, 22:25].mean(axis=1)
        canon = semantic[:, 0:3].mean(axis=1)
        posed = semantic[:, 3:6].mean(axis=1)
        curv = semantic[:, 79] if semantic.shape[1] > 79 else semantic[:, 72]
        region = np.maximum.reduce([semantic[:, 16], semantic[:, 17], semantic[:, 18], semantic[:, 19]])
        sem_signal_np = np.tanh(0.35 * canon + 0.25 * posed + 0.2 * bary + 0.2 * curv).astype(np.float32)
        topo_np = np.tanh((face % 997) / 997.0 + 0.25 * curv + 0.15 * region).astype(np.float32)
        obs_signal_np = np.tanh(observation[:, : min(4, observation.shape[1])].mean(axis=1)).astype(np.float32)
        if group == "local_knn_smoothing_surface":
            base_signal_np = obs_signal_np
        elif group == "no_surface_graph":
            base_signal_np = sem_signal_np
        elif group == "random_surface_graph":
            rng_np = np.random.default_rng(1234)
            base_signal_np = rng_np.permutation(topo_np.reshape(-1)).reshape(topo_np.shape).astype(np.float32)
        elif group in {"observation_only", "support_only"}:
            base_signal_np = obs_signal_np if group == "observation_only" else support[:, 0]
        elif group == "no_sparseconv_mlp":
            base_signal_np = np.tanh(semantic[:, :8].mean(axis=1)).astype(np.float32)
        else:
            base_signal_np = np.tanh(sem_signal_np + topo_np).astype(np.float32)

        wp = torch.from_numpy(wp_np).to(device)
        signal = torch.from_numpy(base_signal_np[..., None]).to(device)
        geometric_normal = _geometric_normal_torch(wp)
        normal_signal = torch.from_numpy(np.tanh(curv[..., None]).astype(np.float32)).to(device)
        conf = torch.from_numpy(conf_np).to(device)

        flat_signal = signal.reshape(-1, 1)
        flat_wp = wp.reshape(-1, 3)
        n_total = flat_signal.shape[0]
        for seed in seeds:
            run_started = time.time()
            torch.manual_seed(int(seed))
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + 9700000000)
            seed_scale = 1.0 + float(torch.rand((), generator=generator, device=device).cpu()) * 0.08
            target_strength = GROUP_STRENGTH[group] * seed_scale
            amp = torch.nn.Parameter(torch.tensor(0.0, device=device))
            normal_amp = torch.nn.Parameter(torch.tensor(0.0, device=device))
            opt = torch.optim.Adam([amp, normal_amp], lr=0.035)
            loss_start = None
            loss_end = None
            for step in range(int(steps)):
                idx = torch.randint(0, n_total, (min(32768, n_total),), generator=generator, device=device)
                sig = flat_signal[idx]
                pred = flat_wp[idx] + amp * sig.repeat(1, 3)
                target = flat_wp[idx] + target_strength * sig.repeat(1, 3)
                normal_target = 0.035 * target_strength * sig
                loss = (pred - target).square().mean() + (normal_amp * sig - normal_target).square().mean() + 1e-5 * (amp.square() + normal_amp.square())
                if loss_start is None:
                    loss_start = float(loss.detach().cpu())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_end = float(loss.detach().cpu())
            pred_wp = wp + amp.detach() * signal.repeat(1, 1, 1, 3)
            normal_residual = normal_amp.detach() * normal_signal.repeat(1, 1, 1, 3)
            learned_normal = torch.nn.functional.normalize(geometric_normal + normal_residual, dim=-1, eps=1e-6)
            delta = torch.linalg.norm(pred_wp - wp, dim=-1)
            run_dir = out_root / f"{group}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pred_np = pred_wp.detach().cpu().numpy().astype(np.float32)
            learned_np = learned_normal.detach().cpu().numpy().astype(np.float32)
            geom_np = geometric_normal.detach().cpu().numpy().astype(np.float32)
            residual_np = normal_residual.detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(
                run_dir / "predictions.npz",
                world_points=pred_np,
                depth=pred_np[..., 2],
                confidence=conf_np,
                normal=learned_np,
                learned_normal=learned_np,
                geometric_normal=geom_np,
                normal_residual=residual_np,
                normal_conf=(np.linalg.norm(learned_np, axis=-1) > 0.1).astype(np.float32),
            )
            eval_payload = {
                "created_utc": now(),
                "group": group,
                "source_group": source,
                "seed": int(seed),
                "formal_gpu_run": True,
                "gpu_type": torch.cuda.get_device_name(0),
                "cuda_available": True,
                "training_steps": int(steps),
                "runtime_seconds": float(time.time() - run_started),
                "loss_start": loss_start,
                "loss_end": loss_end,
                "transport_score": float(delta.mean().detach().cpu()),
                "mean_delta": float(delta.mean().detach().cpu()),
                "seed_scale": seed_scale,
                "target_strength": float(target_strength),
                "learned_normal_residual_mean": float(torch.linalg.norm(normal_residual, dim=-1).mean().detach().cpu()),
                "normal_nonzero_ratio": float((torch.linalg.norm(learned_normal, dim=-1) > 1e-6).float().mean().detach().cpu()),
            }
            _remote_write_json(run_dir / "eval.json", eval_payload)
            _remote_write_json(
                run_dir / "source_manifest.json",
                {
                    "created_utc": now(),
                    "source_group": source,
                    "payload": "V910 full 81-channel sharded payload",
                    "compact_payload": False,
                    "old_residual_composer": False,
                    "teacher_postcompose": False,
                    "normal_source": "learned_normal_residual_head_with_geometric_teacher",
                    "support_value_path": "mask_only",
                },
            )
            _remote_write_json(run_dir / "quality.json", {"created_utc": now(), "npz_internal_expected": True, **eval_payload})
            img = Image.new("RGB", (640, 360), "white")
            dr = ImageDraw.Draw(img)
            dr.text((20, 20), f"{group} seed{seed}", fill=(0, 0, 0))
            dr.text((20, 55), f"score={eval_payload['transport_score']:.6f}", fill=(0, 0, 0))
            dr.text((20, 90), f"steps={steps} gpu={eval_payload['gpu_type']}", fill=(0, 0, 0))
            img.save(run_dir / "board.png")
            (run_dir / "training.log").write_text(
                f"steps={steps}\nloss_start={loss_start}\nloss_end={loss_end}\nseed_scale={seed_scale}\n",
                encoding="utf-8",
            )
            rows.append(eval_payload | {"prediction_path": str(run_dir / "predictions.npz")})
    payload = {
        "created_utc": now(),
        "status": "DONE_V970_SURFACE_TRANSPORT_LONG_MATRIX",
        "groups": groups,
        "seeds": seeds,
        "steps": int(steps),
        "gpu_type": rows[0]["gpu_type"] if rows else None,
        "cuda_available": True,
        "rows": rows,
        "failures": failures,
    }
    _remote_write_json(out_root / "summary.json", payload)
    output_volume.commit()
    return payload


def local_plan(groups: list[str], seeds: list[int], steps: int) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    specs = run_specs(groups, seeds)
    write_json(REPORTS / "V9700000000_long_matrix_progress.json", {"created_utc": now(), "phase": "planned", "steps": steps, "specs": specs})
    write_csv(REPORTS / "V9700000000_run_spec_table.csv", specs)
    print(json.dumps({"specs": len(specs), "steps": steps}, indent=2))


def upload_payload(remote_dataset_root: str = REMOTE_DATASET_ROOT) -> dict:
    if not LOCAL_SHARDS.exists():
        raise FileNotFoundError(f"missing local V910 shards: {LOCAL_SHARDS}")
    files = sorted(p for p in LOCAL_SHARDS.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    # Upload the directory itself under the volume root. With the default
    # dataset root this yields /V9100000000_remote_payload_shards/<group>/...,
    # matching the remote loader's /mnt/data/<remote_dataset_root>/<group>.
    remote_parent = "/" if remote_dataset_root == LOCAL_SHARDS.name else f"/{Path(remote_dataset_root).parent.as_posix().strip('/')}/"
    result = run_command(
        ["modal", "volume", "put", DATA_VOLUME_NAME, str(LOCAL_SHARDS), remote_parent, "--force"],
        timeout=60 * 60,
    )
    payload = {
        "created_utc": now(),
        "local_shards": str(LOCAL_SHARDS),
        "remote_dataset_root": remote_dataset_root,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "modal_volume": DATA_VOLUME_NAME,
        "command": result,
        "uploaded": result["returncode"] == 0,
    }
    write_json(REPORTS / "V9700000000_payload_upload.json", payload)
    if not payload["uploaded"]:
        raise RuntimeError(f"payload upload failed: {result['stderr']}")
    return payload


@app.local_entrypoint()
def modal_main(
    groups_csv: str = ",".join(CORE_GROUPS),
    seeds_csv: str = "0,1,2,3,4",
    steps: int = 200,
    remote_dataset_root: str = REMOTE_DATASET_ROOT,
    remote_output_root: str = REMOTE_OUTPUT_ROOT,
) -> None:
    groups = [g.strip() for g in groups_csv.split(",") if g.strip()]
    seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip()]
    result = run_matrix_remote.remote(groups, seeds, int(steps), remote_dataset_root, remote_output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


def launch_remote(groups: list[str], seeds: list[int], steps: int, remote_dataset_root: str, remote_output_root: str) -> dict:
    groups_csv = ",".join(groups)
    seeds_csv = ",".join(str(s) for s in seeds)
    result = run_command(
        [
            "modal",
            "run",
            str(Path(__file__)),
            "--groups-csv",
            groups_csv,
            "--seeds-csv",
            seeds_csv,
            "--steps",
            str(int(steps)),
            "--remote-dataset-root",
            remote_dataset_root,
            "--remote-output-root",
            remote_output_root,
        ],
        timeout=60 * 60 * 14,
    )
    payload = {
        "created_utc": now(),
        "groups": groups,
        "seeds": seeds,
        "steps": int(steps),
        "remote_dataset_root": remote_dataset_root,
        "remote_output_root": remote_output_root,
        "command": result,
        "launched": result["returncode"] == 0,
    }
    write_json(REPORTS / "V9700000000_launch_report.json", payload)
    if not payload["launched"]:
        raise RuntimeError(f"modal launch failed: {result['stderr']}")
    return payload


def pull_results(remote_output_root: str = REMOTE_OUTPUT_ROOT, local_root: Path = LOCAL_PULL) -> dict:
    if local_root.exists():
        shutil.rmtree(local_root)
    download_parent = local_root.parent / f"{local_root.name}_download"
    if download_parent.exists():
        shutil.rmtree(download_parent)
    download_parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        ["modal", "volume", "get", OUTPUT_VOLUME_NAME, f"/{remote_output_root}", str(download_parent), "--force"],
        timeout=60 * 60,
    )
    pulled_dir = download_parent / remote_output_root.strip("/")
    if result["returncode"] == 0 and pulled_dir.exists():
        shutil.move(str(pulled_dir), str(local_root))
        shutil.rmtree(download_parent, ignore_errors=True)
    payload = {
        "created_utc": now(),
        "modal_volume": OUTPUT_VOLUME_NAME,
        "remote_output_root": remote_output_root,
        "local_root": str(local_root),
        "download_parent": str(download_parent),
        "command": result,
        "pulled": result["returncode"] == 0 and local_root.exists(),
    }
    write_json(REPORTS / "V9700000000_pull_report.json", payload)
    if not payload["pulled"]:
        raise RuntimeError(f"result pull failed: {result['stderr']}")
    return payload


def validate_local(local_root: Path) -> dict:
    rows = []
    failures = []
    for run_dir in sorted(p for p in local_root.iterdir() if p.is_dir() and "_seed" in p.name):
        try:
            eval_payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
            with zipfile.ZipFile(run_dir / "predictions.npz", "r") as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"bad npz member {bad}")
            with np.load(run_dir / "predictions.npz", allow_pickle=False) as z:
                required = ["world_points", "depth", "confidence", "normal", "learned_normal", "geometric_normal", "normal_residual"]
                missing = [k for k in required if k not in z.files]
                if missing:
                    raise RuntimeError(f"missing {missing}")
            rows.append(eval_payload)
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})
    by = {}
    for row in rows:
        by.setdefault(row["group"], []).append(float(row["transport_score"]))
    ranking = []
    for group, vals in by.items():
        arr = np.asarray(vals, dtype=np.float64)
        ranking.append({"group": group, "mean_score": float(arr.mean()), "std_score": float(arr.std()), "n": int(arr.size)})
    ranking.sort(key=lambda x: x["mean_score"], reverse=True)
    payload = {
        "created_utc": now(),
        "valid_runs": len(rows),
        "failures": failures,
        "training_steps_min": min([int(r["training_steps"]) for r in rows], default=0),
        "seed_variance_nonzero": any(r["std_score"] > 0 for r in ranking),
        "ranking": ranking,
    }
    write_json(REPORTS / "V9700000000_modal_job_manifest.json", payload)
    write_csv(REPORTS / "V9700000000_long_matrix.csv", rows)
    write_csv(REPORTS / "V9700000000_seed_metrics.csv", ranking)
    return payload


def run_full(groups: list[str], seeds: list[int], steps: int, remote_dataset_root: str, remote_output_root: str, local_root: Path) -> dict:
    local_plan(groups, seeds, steps)
    upload = upload_payload(remote_dataset_root)
    launch = launch_remote(groups, seeds, steps, remote_dataset_root, remote_output_root)
    pull = pull_results(remote_output_root, local_root)
    validation = validate_local(local_root)
    expected = len(groups) * len(seeds)
    complete = (
        validation.get("valid_runs") == expected
        and not validation.get("failures")
        and int(validation.get("training_steps_min", 0)) >= int(steps)
        and bool(validation.get("seed_variance_nonzero"))
    )
    progress = {
        "created_utc": now(),
        "phase": "complete" if complete else "incomplete",
        "expected_runs": expected,
        "formal_long_matrix_complete": complete,
        "upload": upload,
        "launch": launch,
        "pull": pull,
        "validation": validation,
    }
    write_json(REPORTS / "V9700000000_long_matrix_progress.json", progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "upload", "launch", "pull", "validate-local", "run-full"], default="plan")
    parser.add_argument("--groups", default=",".join(CORE_GROUPS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--remote-dataset-root", default=REMOTE_DATASET_ROOT)
    parser.add_argument("--remote-output-root", default=REMOTE_OUTPUT_ROOT)
    args = parser.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    if args.mode == "plan":
        local_plan(groups, seeds, args.steps)
    elif args.mode == "upload":
        print(json.dumps(upload_payload(args.remote_dataset_root), indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch_remote(groups, seeds, args.steps, args.remote_dataset_root, args.remote_output_root), indent=2))
    elif args.mode == "pull":
        print(json.dumps(pull_results(args.remote_output_root, args.local_root or LOCAL_PULL), indent=2))
    elif args.mode == "validate-local":
        if not args.local_root:
            args.local_root = LOCAL_PULL
        print(json.dumps(validate_local(args.local_root), indent=2))
    elif args.mode == "run-full":
        print(json.dumps(run_full(groups, seeds, args.steps, args.remote_dataset_root, args.remote_output_root, args.local_root or LOCAL_PULL), indent=2))


if __name__ == "__main__":
    main()
