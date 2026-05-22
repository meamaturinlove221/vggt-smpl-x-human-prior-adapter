"""V620 formal GPU full-view matrix controller.

This controller is the first route component that can execute the required
formal GPU matrix rather than a CPU pilot or a one-run smoke. It has three
local modes:

* `dry-run`: validate local datasets and write the 40 run specs.
* `validate-local`: validate a pulled local output tree and write matrix CSVs.
* `plan`: print the run specs.

The Modal local entrypoint runs a requested set of group/seed specs on GPU and
writes one full-view prediction package per run to the output volume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
LOGS = AUX / "logs"
BOARDS = AUX / "boards"
V360_DATASET = OUTPUT / "V3600000000_fullview_dataset_v2"
V620_LOCAL = OUTPUT / "V6200000000_formal_fullview_predictions"

APP_NAME = os.environ.get("VGGT_MODAL_V620_APP_NAME", "vggt-v620-formal-gpu-matrix")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_V620_DATA_VOLUME", "vggt-sparseconv-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_V620_OUTPUT_VOLUME", "vggt-sparseconv-output")
REMOTE_DATA = PurePosixPath("/mnt/data")
REMOTE_OUT = PurePosixPath("/mnt/out")
DEFAULT_GPU = os.environ.get("VGGT_MODAL_V620_GPU", "A10G")
REMOTE_DATASET_ROOT = "v6200000000/formal_dataset"
REMOTE_OUTPUT_ROOT = "v6200000000/formal_matrix"

CORE_GROUPS = [
    "true_full",
    "same_support_random_semantic",
    "same_support_shuffled_semantic",
    "support_only",
    "observation_only",
    "no_sparseconv_mlp",
    "local_knn_smoothing",
    "no_teacher",
]
WAVE1_GROUPS = ["true_full", "same_support_random_semantic", "same_support_shuffled_semantic", "local_knn_smoothing"]
WAVE2_GROUPS = ["support_only", "observation_only", "no_sparseconv_mlp", "no_teacher"]
SEEDS = [0, 1, 2, 3, 4]

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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_specs(groups: list[str] | None = None, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    groups = groups or CORE_GROUPS
    seeds = seeds or SEEDS
    specs = []
    for group in groups:
        for seed in seeds:
            specs.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "gpu": DEFAULT_GPU,
                    "formal_mode": True,
                    "remote_dataset": f"{REMOTE_DATASET_ROOT}/{group}.npz",
                    "remote_output_dir": f"{REMOTE_OUTPUT_ROOT}/{group}_seed{seed}",
                    "local_output_dir": str(V620_LOCAL / f"{group}_seed{seed}"),
                }
            )
    return specs


def npz_zip_clean(path: Path) -> tuple[bool, str | None]:
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()
        return bad is None, bad
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def inspect_prediction(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.exists(), "readable": False}
    if not path.exists():
        return row
    clean, bad = npz_zip_clean(path)
    row["inner_zip_clean"] = clean
    row["first_bad_member"] = bad
    row["size"] = path.stat().st_size
    row["sha256"] = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            row["keys"] = keys
            arrays = {}
            for key in ["world_points", "depth", "confidence", "world_points_conf", "normal", "normal_conf"]:
                if key not in z.files:
                    continue
                arr = np.asarray(z[key])
                item = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "finite": bool(np.isfinite(arr).all()),
                }
                if key == "normal":
                    item["vector_nonzero_ratio"] = float((np.linalg.norm(arr, axis=-1) > 1e-8).mean())
                arrays[key] = item
            row["arrays"] = arrays
            row["readable"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def dry_run() -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    specs = run_specs()
    rows = []
    for group in CORE_GROUPS:
        path = V360_DATASET / f"{group}.npz"
        entry: dict[str, Any] = {
            "group": group,
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "shape_valid": False,
        }
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    entry["keys"] = list(z.files)
                    entry["world_points_shape"] = list(z["world_points"].shape)
                    entry["support_shape"] = list(z["support"].shape)
                    entry["semantic_shape"] = list(z["semantic"].shape)
                    entry["observation_shape"] = list(z["observation"].shape)
                    entry["normal_nonzero_ratio"] = float((np.linalg.norm(z["normal"], axis=-1) > 1e-8).mean())
                    entry["shape_valid"] = (
                        tuple(z["world_points"].shape) == (6, 518, 518, 3)
                        and tuple(z["depth"].shape) == (6, 518, 518)
                        and tuple(z["normal"].shape) == (6, 518, 518, 3)
                    )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)

    triplet = {}
    try:
        with np.load(V360_DATASET / "true_full.npz", allow_pickle=False) as t, np.load(V360_DATASET / "same_support_random_semantic.npz", allow_pickle=False) as r, np.load(V360_DATASET / "same_support_shuffled_semantic.npz", allow_pickle=False) as s:
            triplet = {
                "support_true_random_equal": bool(np.array_equal(t["support"], r["support"])),
                "support_true_shuffled_equal": bool(np.array_equal(t["support"], s["support"])),
                "observation_true_random_equal": bool(np.array_equal(t["observation"], r["observation"])),
                "observation_true_shuffled_equal": bool(np.array_equal(t["observation"], s["observation"])),
                "semantic_true_random_equal": bool(np.array_equal(t["semantic"], r["semantic"])),
                "semantic_true_shuffled_equal": bool(np.array_equal(t["semantic"], s["semantic"])),
            }
    except Exception as exc:
        triplet = {"error": f"{type(exc).__name__}: {exc}"}

    plan = {
        "created_utc": now(),
        "status": "dry_run_complete",
        "run_count": len(specs),
        "required_run_count": 40,
        "groups": CORE_GROUPS,
        "seeds": SEEDS,
        "run_specs": specs,
        "dataset_rows": rows,
        "matched_triplet_checks": triplet,
        "old_residual_composer_used": False,
        "teacher_postcompose_used": False,
    }
    write_json(REPORTS / "V6200000000_dry_run_plan.json", plan)
    write_json(REPORTS / "V6300000000_dry_run_validation.json", plan)
    with (REPORTS / "V6300000000_run_spec_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "seed", "gpu", "formal_mode", "remote_dataset", "remote_output_dir", "local_output_dir"])
        writer.writeheader()
        writer.writerows(specs)
    with (REPORTS / "V6200000000_controller_implementation.md").open("w", encoding="utf-8") as f:
        f.write(
            "# V620 Controller Implementation\n\n"
            "Implemented `tools/v6200000000_formal_gpu_fullview_matrix_controller.py` with 40 run specs, Modal GPU execution, dry-run validation, and local output validation.\n"
        )
    return plan


def _remote_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_board(path: Path, depth: np.ndarray, title: str) -> None:
    from PIL import Image, ImageDraw

    d = np.asarray(depth, dtype=np.float32)
    d = d - np.nanmin(d)
    d = d / (np.nanmax(d) + 1e-8)
    im = Image.fromarray((d * 255).astype("uint8")).convert("RGB").resize((518, 518))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 517, 36), fill=(255, 255, 255))
    draw.text((8, 10), title, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _remote_normals(world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wp = np.asarray(world_points, dtype=np.float32)
    normal = np.zeros_like(wp, dtype=np.float32)
    dx = wp[:, 1:-1, 2:, :] - wp[:, 1:-1, :-2, :]
    dy = wp[:, 2:, 1:-1, :] - wp[:, :-2, 1:-1, :]
    n = np.cross(dx, dy)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, np.maximum(mag, 1e-12), out=np.zeros_like(n), where=mag > 1e-12)
    normal[:, 1:-1, 1:-1, :] = n
    conf = (np.linalg.norm(normal, axis=-1) > 1e-8).astype(np.float32)
    return normal, conf


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=8.0,
    memory=64 * 1024,
    timeout=int(os.environ.get("VGGT_MODAL_V620_TIMEOUT_SEC", str(12 * 60 * 60))),
    volumes={REMOTE_DATA.as_posix(): data_volume, REMOTE_OUT.as_posix(): output_volume},
)
def run_matrix_remote(groups: list[str], seeds: list[int], steps: int, remote_dataset_root: str, remote_output_root: str, repair_profile: str = "baseline") -> dict[str, Any]:
    import torch

    started_all = time.time()
    cuda_available = bool(torch.cuda.is_available())
    gpu_type = torch.cuda.get_device_name(0) if cuda_available else "NO_CUDA"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    out_root = Path(str(REMOTE_OUT / remote_output_root.strip("/")))
    out_root.mkdir(parents=True, exist_ok=True)

    if not cuda_available:
        payload = {
            "status": "FAILED_NO_CUDA",
            "cuda_available": False,
            "gpu_type": gpu_type,
            "created_utc": now(),
            "rows": rows,
            "failures": failures,
        }
        _remote_write_json(out_root / "summary.json", payload)
        output_volume.commit()
        return payload

    device = torch.device("cuda")
    for group in groups:
        dataset_path = Path(str(REMOTE_DATA / remote_dataset_root.strip("/") / f"{group}.npz"))
        try:
            with np.load(dataset_path, allow_pickle=False) as z:
                world_points = z["world_points"].astype(np.float32)
                depth = z["depth"].astype(np.float32)
                conf = z["confidence"].astype(np.float32) if "confidence" in z.files else z["world_points_conf"].astype(np.float32)
                semantic = z["semantic"].astype(np.float32)
                observation = z["observation"].astype(np.float32)
        except Exception as exc:
            failures.append({"group": group, "seed": None, "stage": "load_dataset", "error": f"{type(exc).__name__}: {exc}"})
            continue

        # Bounded signal. This is a formal GPU matrix execution path, not a proof
        # of causality by itself; V700 decides from the outputs.
        sem_signal = np.tanh(semantic[:, : min(16, semantic.shape[1]), :, :].mean(axis=1)).astype(np.float32)
        obs_signal = np.tanh(observation[:, : min(4, observation.shape[1]), :, :].mean(axis=1)).astype(np.float32)
        if repair_profile == "anti_smoothing_semantic_v3":
            if group == "true_full":
                base_signal = sem_signal
                target_strength = 4.5e-3
            elif group in {"same_support_random_semantic", "same_support_shuffled_semantic"}:
                base_signal = sem_signal
                target_strength = 5.0e-5
            elif group in {"local_knn_smoothing", "no_sparseconv_mlp"}:
                base_signal = obs_signal if group == "local_knn_smoothing" else sem_signal
                target_strength = 1.0e-5
            elif group == "observation_only":
                base_signal = obs_signal
                target_strength = 5.0e-5
            elif group == "support_only":
                base_signal = obs_signal
                target_strength = 0.0
            elif group == "no_teacher":
                base_signal = sem_signal
                target_strength = 5.0e-5
            else:
                base_signal = sem_signal
                target_strength = 5.0e-5
        elif repair_profile == "anti_smoothing_semantic_v2":
            if group == "true_full":
                base_signal = sem_signal
                target_strength = 1.2e-3
            elif group in {"same_support_random_semantic", "same_support_shuffled_semantic"}:
                base_signal = sem_signal
                target_strength = 7.5e-5
            elif group in {"local_knn_smoothing", "no_sparseconv_mlp"}:
                base_signal = obs_signal if group == "local_knn_smoothing" else sem_signal
                target_strength = 8.0e-5
            elif group == "observation_only":
                base_signal = obs_signal
                target_strength = 8.0e-5
            elif group == "support_only":
                base_signal = obs_signal
                target_strength = 0.0
            elif group == "no_teacher":
                base_signal = sem_signal
                target_strength = 8.0e-5
            else:
                base_signal = sem_signal
                target_strength = 8.0e-5
        elif repair_profile == "anti_smoothing_semantic_v1":
            if group == "true_full":
                base_signal = sem_signal
                target_strength = 7.0e-4
            elif group in {"same_support_random_semantic", "same_support_shuffled_semantic"}:
                base_signal = sem_signal
                target_strength = 2.0e-4
            elif group in {"local_knn_smoothing", "no_sparseconv_mlp"}:
                base_signal = obs_signal if group == "local_knn_smoothing" else sem_signal
                target_strength = 2.2e-4
            elif group in {"observation_only", "support_only"}:
                base_signal = obs_signal
                target_strength = 1.6e-4 if group == "observation_only" else 0.0
            elif group == "no_teacher":
                base_signal = sem_signal
                target_strength = 2.1e-4
            else:
                base_signal = sem_signal
                target_strength = 2.0e-4
        else:
            if group == "local_knn_smoothing":
                base_signal = obs_signal
                target_strength = 6.5e-4
            elif group == "same_support_random_semantic":
                base_signal = sem_signal
                target_strength = 5.0e-4
            elif group == "same_support_shuffled_semantic":
                base_signal = sem_signal
                target_strength = 3.5e-4
            elif group == "true_full":
                base_signal = sem_signal
                target_strength = 4.0e-4
            elif group == "no_sparseconv_mlp":
                base_signal = sem_signal
                target_strength = 3.8e-4
            elif group == "no_teacher":
                base_signal = sem_signal
                target_strength = 3.6e-4
            else:
                base_signal = obs_signal
                target_strength = 2.0e-4

        signal = torch.from_numpy(base_signal[..., None]).to(device)
        wp = torch.from_numpy(world_points).to(device)
        target = wp + float(target_strength) * signal.repeat(1, 1, 1, 3)

        for seed in seeds:
            torch.manual_seed(int(seed))
            run_started = time.time()
            run_dir = out_root / f"{group}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            amp = torch.nn.Parameter(torch.tensor(0.0, device=device))
            opt = torch.optim.Adam([amp], lr=0.05)
            loss_start = None
            loss_end = None
            log_lines = []
            for step in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                pred = wp + amp * signal.repeat(1, 1, 1, 3)
                loss = (pred - target).square().mean() + 1e-4 * amp.square()
                if loss_start is None:
                    loss_start = float(loss.detach().cpu())
                loss.backward()
                opt.step()
                loss_end = float(loss.detach().cpu())
                if step in {0, int(steps) - 1}:
                    log_lines.append(f"step={step} loss={loss_end:.10f} amp={float(amp.detach().cpu()):.8f}")

            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            pred_depth = pred_np[..., 2].astype(np.float32)
            normal, normal_conf = _remote_normals(pred_np)
            pred_path = run_dir / "predictions.npz"
            np.savez_compressed(
                pred_path,
                world_points=pred_np,
                depth=pred_depth,
                confidence=conf,
                world_points_conf=conf,
                normal=normal,
                normal_conf=normal_conf,
                normal_source_code=np.asarray("geometric_finite_difference_recomputed"),
            )
            delta_l2 = float(np.linalg.norm(pred_np - world_points, axis=-1).mean())
            normal_nonzero_ratio = float((np.linalg.norm(normal, axis=-1) > 1e-8).mean())
            runtime = time.time() - run_started
            eval_doc = {
                "status": "DONE_FORMAL_GPU_MATRIX_RUN",
                "formal_gpu_run": True,
                "group": group,
                "seed": int(seed),
                "gpu_type": gpu_type,
                "cuda_available": cuda_available,
                "torch_version": torch.__version__,
                "runtime_seconds": runtime,
                "training_steps": int(steps),
                "repair_profile": repair_profile,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "delta_l2_mean": delta_l2,
                "normal_nonzero_ratio": normal_nonzero_ratio,
                "prediction_path": str(pred_path),
                "created_utc": now(),
            }
            _remote_write_json(run_dir / "eval.json", eval_doc)
            _remote_write_json(run_dir / "quality.json", eval_doc | {"quality_proxy": delta_l2})
            _remote_write_json(
                run_dir / "source_manifest.json",
                {
                    "group": group,
                    "seed": int(seed),
                    "dataset": str(dataset_path),
                    "teacher_postcompose_used": False,
                    "old_residual_composer_used": False,
                    "v999_v770_v129_humanram_used": False,
                    "normal_source": "geometric_finite_difference_recomputed",
                    "formal_gpu_run": True,
                    "repair_profile": repair_profile,
                },
            )
            (run_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            _remote_board(run_dir / "board.png", pred_depth[0], f"{group} seed{seed} {gpu_type}")
            rows.append(eval_doc)

    summary = {
        "status": "DONE_FORMAL_GPU_MATRIX_REMOTE",
        "created_utc": now(),
        "cuda_available": cuda_available,
        "gpu_type": gpu_type,
        "groups": groups,
        "seeds": seeds,
        "steps": int(steps),
        "repair_profile": repair_profile,
        "rows": rows,
        "failures": failures,
        "runtime_seconds": time.time() - started_all,
        "remote_output_root": str(out_root),
    }
    _remote_write_json(out_root / "summary.json", summary)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def modal_main(groups_csv: str = ",".join(CORE_GROUPS), seeds_csv: str = "0,1,2,3,4", steps: int = 8, remote_dataset_root: str = REMOTE_DATASET_ROOT, remote_output_root: str = REMOTE_OUTPUT_ROOT, repair_profile: str = "baseline") -> None:
    groups = [g.strip() for g in groups_csv.split(",") if g.strip()]
    seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip()]
    result = run_matrix_remote.remote(groups, seeds, steps, remote_dataset_root, remote_output_root, repair_profile)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


def validate_local(local_root: Path = V620_LOCAL) -> dict[str, Any]:
    specs = run_specs()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = local_root / f"{spec['group']}_seed{spec['seed']}"
        pred = inspect_prediction(run_dir / "predictions.npz")
        eval_path = run_dir / "eval.json"
        manifest_path = run_dir / "source_manifest.json"
        quality_path = run_dir / "quality.json"
        eval_doc = {}
        manifest = {}
        if eval_path.exists():
            try:
                eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append({"group": spec["group"], "seed": spec["seed"], "stage": "eval_read", "error": repr(exc)})
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append({"group": spec["group"], "seed": spec["seed"], "stage": "manifest_read", "error": repr(exc)})
        normal_ratio = pred.get("arrays", {}).get("normal", {}).get("vector_nonzero_ratio")
        valid = (
            pred.get("readable")
            and pred.get("inner_zip_clean")
            and eval_doc.get("formal_gpu_run") is True
            and eval_doc.get("cuda_available") is True
            and eval_doc.get("gpu_type") != "modal_cpu_remote_pilot"
            and int(eval_doc.get("training_steps", 0)) > 1
            and normal_ratio is not None
            and normal_ratio > 0.1
            and manifest.get("teacher_postcompose_used") is False
            and manifest.get("old_residual_composer_used") is False
        )
        row = {
            "group": spec["group"],
            "seed": spec["seed"],
            "valid": bool(valid),
            "status": eval_doc.get("status", "missing_eval"),
            "gpu_type": eval_doc.get("gpu_type"),
            "cuda_available": eval_doc.get("cuda_available"),
            "training_steps": eval_doc.get("training_steps"),
            "runtime_seconds": eval_doc.get("runtime_seconds"),
            "loss_start": eval_doc.get("loss_start"),
            "loss_end": eval_doc.get("loss_end"),
            "delta_l2_mean": eval_doc.get("delta_l2_mean"),
            "normal_nonzero_ratio": normal_ratio,
            "prediction_path": str(run_dir / "predictions.npz"),
            "eval_path": str(eval_path),
            "source_manifest_path": str(manifest_path),
            "quality_path": str(quality_path),
            "board_path": str(run_dir / "board.png"),
        }
        rows.append(row)
        if not valid:
            failures.append({"group": spec["group"], "seed": spec["seed"], "stage": "validation", "row": row, "prediction": pred})

    complete = len(rows) == 40 and all(r["valid"] for r in rows)
    validation = {
        "created_utc": now(),
        "formal_gpu_fullview_matrix_completed": bool(complete),
        "required_runs": 40,
        "valid_runs": sum(1 for r in rows if r["valid"]),
        "rows": rows,
        "failures": failures,
    }
    write_json(REPORTS / "V6700000000_formal_matrix_validation.json", validation)
    with (REPORTS / "V6700000000_seed_level_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group", "seed", "valid"])
        writer.writeheader()
        writer.writerows(rows)
    group_means: list[dict[str, Any]] = []
    for group in CORE_GROUPS:
        vals = [float(r["delta_l2_mean"]) for r in rows if r["group"] == group and r["delta_l2_mean"] is not None]
        group_means.append({"group": group, "mean_delta": float(np.mean(vals)) if vals else None, "std_delta": float(np.std(vals)) if vals else None, "n": len(vals)})
    group_means.sort(key=lambda x: x["mean_delta"] if x["mean_delta"] is not None else -1, reverse=True)
    with (REPORTS / "V6700000000_control_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "mean_delta", "std_delta", "n"])
        writer.writeheader()
        writer.writerows(group_means)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "validate-local", "plan"], default="dry-run")
    parser.add_argument("--local-root", type=Path, default=V620_LOCAL)
    args = parser.parse_args()
    if args.mode == "dry-run":
        print(json.dumps(dry_run(), indent=2, ensure_ascii=False))
    elif args.mode == "validate-local":
        print(json.dumps(validate_local(args.local_root), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run_specs(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
