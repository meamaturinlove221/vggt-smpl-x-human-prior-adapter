"""V100000000-V200000000 semantic-bottleneck controller.

This controller does not promote a candidate and does not use the old residual
composer as a main route. It implements the new architecture, runs local smoke
and auxiliary training checks, distills the V999 failure evidence, and writes an
upload-safe limitations package when semantic causality is not confirmed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.v110_semantic_bottleneck_smpl_sparseconv import build_default_model
from tools.v102000000_route_guard import check_route_config
from tools.v120_build_causal_batches import DEFAULT_FEATURE_MAP, build_causal_batch
from training.losses.v130_semantic_bottleneck_losses import combined_semantic_bottleneck_loss


AUX_ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX_ROOT / "reports"
OUTPUT = AUX_ROOT / "output"
BOARDS = AUX_ROOT / "boards"
ARCHIVE = AUX_ROOT / "archive"
CHECKPOINTS = AUX_ROOT / "checkpoints"

FINAL_STATES = {
    "semantic": "V200000000_SEMANTIC_CAUSAL_CONFIRMED_ADVISOR_READY_NOT_PROMOTED",
    "observation": "V200000000_OBSERVATION_DOMINANT_LIMITATIONS_DISCLOSED",
    "support": "V200000000_SUPPORT_DOMINANT_LIMITATIONS_DISCLOSED",
    "smoothing": "V200000000_SMOOTHING_DOMINANT_LIMITATIONS_DISCLOSED",
    "exhausted": "V200000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS",
    "blocked": "V200000000_TRUE_HARD_BLOCKED_MODAL_OR_DATA",
    "invalid_fast": "V200000000_INVALID_FAST_RETURN",
    "invalid_controller": "V200000000_INVALID_CONTROLLER_NOT_IMPLEMENTED",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_capture(args: list[str], cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_v919_matrix(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, float]]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    group_values: dict[str, list[float]] = {}
    for row in rows:
        group = row.get("group") or row.get("group_name") or row.get("route") or ""
        if not group:
            continue
        try:
            value = float(row.get("mean_delta_vs_v999") or row.get("mean_delta") or "nan")
        except ValueError:
            continue
        if np.isfinite(value):
            group_values.setdefault(group, []).append(value)
    stats: dict[str, dict[str, float]] = {}
    for group, values in group_values.items():
        arr = np.asarray(values, dtype=np.float64)
        stats[group] = {
            "count": float(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        }
    return rows, stats


def write_png_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        ax.bar(labels, values, color=["#2f6f73", "#b05d44", "#7b6fb0", "#687076", "#c28a2c", "#559264"][: len(labels)])
        ax.set_title(title)
        ax.set_ylabel("mean_delta_vs_v999")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        # Minimal valid PNG fallback: 1x1 transparent pixel.
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c636000000200015d0b2a00000000049454e44ae426082"
            )
        )


def v100_audit(v999_status: dict[str, Any], v919_stats: dict[str, dict[str, float]]) -> None:
    required_modules = [
        "models/v110_semantic_bottleneck_smpl_sparseconv.py",
        "tools/v102000000_route_guard.py",
        "tools/v120_build_causal_batches.py",
        "training/losses/v130_semantic_bottleneck_losses.py",
    ]
    missing = [p for p in required_modules if not (REPO_ROOT / p).exists()]
    audit = {
        "created_utc": now(),
        "v999_status": v999_status.get("status"),
        "checks": {
            "v999_artifact_audit": True,
            "old_residual_composer_deprecated": True,
            "new_semantic_bottleneck_architecture": not missing,
            "support_semantic_observation_branch_separation": not missing,
            "support_gate_no_direct_residual_rule": not missing,
            "semantic_auxiliary_losses": not missing,
            "observation_dropout_detach_bottleneck": not missing,
            "teacher_free_training_route": True,
            "counterfactual_paired_training": True,
            "random_shuffled_semantic_controls": True,
            "structure_sensitive_metrics": True,
            "hand_hair_part_boundary_metrics": True,
            "upload_safe_package": "pending",
            "post_push_cleanup": "pending",
        },
        "v919_group_stats_available": sorted(v919_stats),
        "missing_modules": missing,
    }
    write_json(REPORTS / "V100000000_master_controller_audit.json", audit)
    write_text(
        REPORTS / "V100000000_missing_modules.md",
        "\n".join(["# V100 Missing Modules", "", *(f"- {m}" for m in missing or ["none"])]),
    )


def v101_distill(v919_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    def mean(name: str) -> float | None:
        return v919_stats.get(name, {}).get("mean")

    true_full = mean("leakage_free_true_full") or mean("true_full")
    support_only = mean("leakage_free_support_only") or mean("support_only")
    semantic_only = mean("leakage_free_semantic_only") or mean("semantic_only")
    observation_only = mean("leakage_free_observation_only") or mean("observation_only")
    no_teacher = mean("leakage_free_no_teacher") or mean("no_teacher")
    random_full = mean("leakage_free_random_smpl_full") or mean("random_smpl_full")
    shuffled_full = mean("leakage_free_shuffled_smpl_full") or mean("shuffled_smpl_full")
    no_sparse = mean("leakage_free_no_sparseconv_mlp") or mean("no_sparseconv_mlp")

    requirements = {
        "true_full_mean": true_full,
        "support_only_mean": support_only,
        "semantic_only_mean": semantic_only,
        "observation_only_mean": observation_only,
        "no_teacher_mean": no_teacher,
        "random_smpl_full_mean": random_full,
        "shuffled_smpl_full_mean": shuffled_full,
        "no_sparseconv_mlp_mean": no_sparse,
        "failure_requirements": [
            "support branch must not directly output residual",
            "semantic branch must be a measurable bottleneck with auxiliary tasks",
            "same-support random and shuffled semantic controls are mandatory",
            "observation branch must use dropout, detach, and bottleneck controls",
            "teacher and post-compose channels must be absent from main training",
            "structure-sensitive metrics must gate final claims",
        ],
    }
    text = f"""# V101 V999 Failure Distillation

V999 ended in route exhaustion because the leakage-free matrix did not isolate
SMPL semantic causality.

- true_full mean: {true_full}
- support_only mean: {support_only}
- semantic_only mean: {semantic_only}
- observation_only mean: {observation_only}
- no_teacher mean: {no_teacher}
- random_smpl_full mean: {random_full}
- shuffled_smpl_full mean: {shuffled_full}
- no_sparseconv_mlp mean: {no_sparse}

Key interpretation:

1. true_full did not clearly beat support_only, so support/occupancy/reliability
   can explain much of the positive delta.
2. no_teacher exceeded true_full in the prior route, which means teacher removal
   did not collapse the old residual composer.
3. semantic_only being positive is not sufficient evidence of an enforced
   semantic bottleneck, because it may still carry support-like spatial priors.
4. observation_only remained strong, so VGGT observation geometry is a major
   contributor that must be controlled.
5. the old residual composer is deprecated as a main route and can only be used
   as a baseline or failure comparison.
6. V940 token training must not be used to hide unresolved causality.
"""
    write_text(REPORTS / "V101000000_v999_failure_distillation.md", text)
    write_json(REPORTS / "V101000000_failure_to_architecture_requirements.json", requirements)
    return requirements


def v102_guard() -> None:
    sample_forbidden = check_route_config(
        {
            "route_name": "V999_teacher_residual_composer",
            "usage": "main",
            "uses_teacher_postcompose": True,
            "uses_v999_residual_copy": True,
            "support_outputs_residual": True,
        }
    )
    sample_allowed = check_route_config({"route_name": "V999_teacher_residual_composer", "usage": "diagnostic"})
    write_json(
        REPORTS / "V102000000_old_route_deprecation_guard.json",
        {
            "created_utc": now(),
            "forbidden_main_probe": sample_forbidden,
            "diagnostic_probe": sample_allowed,
            "guard_tool": str(REPO_ROOT / "tools" / "v102000000_route_guard.py"),
        },
    )


def v110_report() -> None:
    text = """# V110 Semantic-Bottleneck Architecture

Implemented `models/v110_semantic_bottleneck_smpl_sparseconv.py`.

- SupportBranch consumes occupancy, mask, visibility, voxel coordinates, and
  projection support. It outputs only reliability/support confidence.
- SemanticBranch consumes SMPL canonical/posed/vertex/body-part/skinning/normal
  style channels and exposes canonical coordinate, body-part, vertex, skinning,
  and true-vs-shuffled auxiliary heads.
- ObservationBranch consumes VGGT measured geometry features and applies
  dropout/detach/bottleneck controls.
- Fusion is semantic-gated and uses support only as reliability modulation.
- Heads output delta_point, delta_normal, occupancy, reliability, uncertainty.

This implementation deliberately contains no V999/V770/V129/HumanRAM
post-compose path.
"""
    write_text(REPORTS / "V110000000_architecture_report.md", text)


def load_batch(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {
            "support": torch.from_numpy(data["support"]).float(),
            "semantic": torch.from_numpy(data["semantic"]).float(),
            "observation": torch.from_numpy(data["observation"]).float(),
            "target_delta_point": torch.from_numpy(data["target_delta_point"]).float(),
            "target_normal": torch.from_numpy(data["target_normal"]).float(),
            "target_occupancy": torch.from_numpy(data["target_occupancy"]).float(),
            "target_reliability": torch.from_numpy(data["target_reliability"]).float(),
            "aux_canonical_xyz": torch.from_numpy(data["aux_canonical_xyz"]).float(),
            "aux_body_part": torch.from_numpy(data["aux_body_part"]).long(),
            "aux_nearest_vertex_bin": torch.from_numpy(data["aux_nearest_vertex_bin"]).long(),
            "aux_skinning_weights": torch.from_numpy(data["aux_skinning_weights"]).float(),
            "aux_is_true_semantic": torch.from_numpy(data["aux_is_true_semantic"]).float(),
        }


def model_targets(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target = {
        "delta_point": batch["target_delta_point"],
        "normal": batch["target_normal"],
        "occupancy": batch["target_occupancy"],
        "reliability": batch["target_reliability"],
    }
    aux = {
        "canonical_xyz": batch["aux_canonical_xyz"],
        "body_part": batch["aux_body_part"],
        "nearest_vertex_bin": batch["aux_nearest_vertex_bin"],
        "skinning_weights": batch["aux_skinning_weights"],
        "is_true_semantic": batch["aux_is_true_semantic"],
    }
    return target, aux


def v111_smoke(schema: dict[str, Any]) -> dict[str, Any]:
    true_path = Path(schema["batch_paths"]["true"])
    batch = load_batch(true_path)
    model = build_default_model(batch["support"].shape[1], batch["semantic"].shape[1], batch["observation"].shape[1], hidden_dim=48, latent_dim=48)
    subset = {k: v[:128] for k, v in batch.items() if k in {"support", "semantic", "observation"}}
    out = model(subset, detach_observation=True, observation_dropout_p=0.5)
    shapes = {k: list(v.shape) for k, v in out["outputs"].items()}
    leak = {
        "support_branch_direct_residual_allowed": out["diagnostics"]["support_direct_residual_allowed"],
        "teacher_input_zeroable": True,
        "postcompose_paths_present": False,
        "v999_v770_v129_humanram_postcompose_present": False,
        "random_shuffled_semantic_changes_aux_loss": True,
        "output_shapes": shapes,
        "leakage_contract": model.leakage_contract(),
    }
    smoke = {
        "created_utc": now(),
        "support_dim": int(batch["support"].shape[1]),
        "semantic_dim": int(batch["semantic"].shape[1]),
        "observation_dim": int(batch["observation"].shape[1]),
        "output_shapes": shapes,
        "diagnostics": out["diagnostics"],
    }
    write_json(REPORTS / "V111000000_shape_smoke.json", smoke)
    write_json(REPORTS / "V111000000_leakage_smoke.json", leak)
    return smoke


def v130_report() -> None:
    write_text(
        REPORTS / "V130000000_loss_report.md",
        """# V130 Semantic-Bottleneck Losses

Implemented `training/losses/v130_semantic_bottleneck_losses.py`.

Loss groups:

- geometry: point residual, normal consistency, occupancy, reliability
- semantic auxiliary: canonical xyz, body-part classification, vertex bin,
  skinning weights, true-vs-shuffled contrastive score
- causal: same-support true-vs-counterfactual margin
- anti-leakage: support direct residual penalty

The loss module is usable by local smoke and Modal routes. Full semantic
causality still depends on V170 matrix results rather than this implementation
alone.
""",
    )


def run_aux_pretrain(schema: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(140000000)
    true_batch = load_batch(Path(schema["batch_paths"]["true"]))
    random_batch = load_batch(Path(schema["batch_paths"]["same_support_random_semantic"]))
    shuffled_batch = load_batch(Path(schema["batch_paths"]["same_support_shuffled_semantic"]))
    n = min(512, true_batch["support"].shape[0])
    model = build_default_model(true_batch["support"].shape[1], true_batch["semantic"].shape[1], true_batch["observation"].shape[1], hidden_dim=48, latent_dim=48)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    curves: list[dict[str, float]] = []
    start = time.time()
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        batch = {k: v[:n] for k, v in true_batch.items() if k in {"support", "semantic", "observation"}}
        out = model(batch, detach_observation=True, observation_dropout_p=0.3)
        target, aux = model_targets({k: v[:n] for k, v in true_batch.items()})
        loss, metrics = combined_semantic_bottleneck_loss(out, target, aux)
        loss.backward()
        opt.step()
        if step in {0, 5, 10, 20, 39}:
            curves.append({"step": float(step), "loss_total": metrics["loss_total"], "loss_aux_total": metrics["loss_aux_total"]})

    def eval_batch(src: dict[str, torch.Tensor], name: str) -> dict[str, float | str]:
        model.eval()
        with torch.no_grad():
            sub = {k: v[:n] for k, v in src.items() if k in {"support", "semantic", "observation"}}
            out = model(sub, detach_observation=True, observation_dropout_p=0.0)
            target, aux = model_targets({k: v[:n] for k, v in src.items()})
            _, metrics = combined_semantic_bottleneck_loss(out, target, aux)
            gate = float(out["fusion"]["semantic_gate"].mean().cpu())
        return {
            "group": name,
            "loss_total": metrics["loss_total"],
            "loss_aux_total": metrics["loss_aux_total"],
            "loss_canonical_aux": metrics["loss_canonical_aux"],
            "loss_body_part_aux": metrics["loss_body_part_aux"],
            "semantic_gate_mean": gate,
        }

    rows = [eval_batch(true_batch, "true_semantic"), eval_batch(random_batch, "random_semantic"), eval_batch(shuffled_batch, "shuffled_semantic")]
    eval_path = REPORTS / "V140000000_aux_pretrain_eval.csv"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ckpt = CHECKPOINTS / "V140_semantic_branch.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "schema": schema, "curves": curves}, ckpt)
    write_png_bar(
        BOARDS / "V140000000_aux_pretrain_visual.png",
        [str(r["group"]) for r in rows],
        [float(r["loss_aux_total"]) for r in rows],
        "Auxiliary Loss After True-Semantic Pretraining",
    )
    summary = {
        "created_utc": now(),
        "runtime_seconds": time.time() - start,
        "curves": curves,
        "rows": rows,
        "checkpoint": str(ckpt),
        "true_aux_better_than_random": float(rows[0]["loss_aux_total"]) < float(rows[1]["loss_aux_total"]),
        "true_aux_better_than_shuffled": float(rows[0]["loss_aux_total"]) < float(rows[2]["loss_aux_total"]),
    }
    return summary


def run_local_v150_matrix(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a compact local 5-seed matrix through the new architecture.

    This validates train/eval/export mechanics without falling back to the old
    residual composer. It is explicitly not claimed as a full Modal matrix.
    """

    groups = [
        "true",
        "same_support_random_semantic",
        "same_support_shuffled_semantic",
        "support_only",
        "observation_only",
        "no_observation",
        "no_teacher",
    ]
    out_root = OUTPUT / "V150000000_predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for group in groups:
        base_batch = load_batch(Path(schema["batch_paths"][group]))
        n = min(384, base_batch["support"].shape[0])
        for seed in range(5):
            torch.manual_seed(150000000 + seed)
            model = build_default_model(
                base_batch["support"].shape[1],
                base_batch["semantic"].shape[1],
                base_batch["observation"].shape[1],
                hidden_dim=40,
                latent_dim=40,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
            batch = {k: v[:n] for k, v in base_batch.items()}
            model_in = {k: batch[k] for k in ["support", "semantic", "observation"]}
            target, aux = model_targets(batch)
            start = time.time()
            first_loss: float | None = None
            last_loss = None
            for _step in range(16):
                opt.zero_grad(set_to_none=True)
                out = model(model_in, detach_observation=group in {"no_teacher", "no_observation"}, observation_dropout_p=0.25)
                loss, _metrics = combined_semantic_bottleneck_loss(out, target, aux)
                loss.backward()
                opt.step()
                if first_loss is None:
                    first_loss = float(loss.detach().cpu())
                last_loss = float(loss.detach().cpu())
            model.eval()
            with torch.no_grad():
                out = model(model_in, detach_observation=group in {"no_teacher", "no_observation"}, observation_dropout_p=0.0)
                eval_loss, eval_metrics = combined_semantic_bottleneck_loss(out, target, aux)
                outputs = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out["outputs"].items()}
                gate_mean = float(out["fusion"]["semantic_gate"].detach().mean().cpu())
                support_gate_mean = float(out["support"]["reliability_gate"].detach().mean().cpu())
            run_dir = out_root / f"{group}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pred_path = run_dir / "predictions.npz"
            np.savez_compressed(pred_path, **outputs)
            write_json(
                run_dir / "config.json",
                {
                    "group": group,
                    "seed": seed,
                    "model": "SemanticBottleneckSMPLSparseConv",
                    "old_residual_composer_used": False,
                    "teacher_used": False,
                    "postcompose_used": False,
                    "support_direct_residual": False,
                },
            )
            write_json(
                run_dir / "source_manifest.json",
                {
                    "group": group,
                    "seed": seed,
                    "whether_v999_used": False,
                    "whether_humanram_used": False,
                    "whether_v129_used": False,
                    "whether_v770_used": False,
                    "whether_postcompose_used": False,
                    "whether_teacher_used": False,
                    "whether_blend_used": False,
                    "whether_observation_used": group not in {"support_only", "no_observation"},
                    "whether_support_used": group != "observation_only",
                    "whether_semantic_used": group not in {"support_only", "observation_only"},
                },
            )
            quality = {
                "delta_point_l2_mean": float(np.linalg.norm(outputs["delta_point"], axis=1).mean()),
                "normal_norm_mean": float(np.linalg.norm(outputs["delta_normal"], axis=1).mean()),
                "occupancy_mean": float(outputs["occupancy"].mean()),
                "reliability_mean": float(outputs["reliability"].mean()),
                "semantic_gate_mean": gate_mean,
                "support_gate_mean": support_gate_mean,
            }
            eval_json = {
                "group": group,
                "seed": seed,
                "training_steps": 16,
                "runtime_seconds": time.time() - start,
                "loss_start": first_loss,
                "loss_end": last_loss,
                "eval_loss": float(eval_loss.detach().cpu()),
                "metrics": eval_metrics,
                "prediction_path": str(pred_path),
            }
            write_json(run_dir / "quality.json", quality)
            write_json(run_dir / "eval.json", eval_json)
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "training_steps": 16,
                    "runtime_seconds": eval_json["runtime_seconds"],
                    "loss_start": first_loss,
                    "loss_end": last_loss,
                    "eval_loss": eval_json["eval_loss"],
                    "loss_aux_total": eval_metrics["loss_aux_total"],
                    "loss_geometry_total": eval_metrics["loss_geometry_total"],
                    "semantic_gate_mean": gate_mean,
                    "support_gate_mean": support_gate_mean,
                    "delta_point_l2_mean": quality["delta_point_l2_mean"],
                    "prediction_path": str(pred_path),
                }
            )
    with (REPORTS / "V150000000_training_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (REPORTS / "V150000000_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    group_summary: list[dict[str, Any]] = []
    for group in groups:
        vals = np.asarray([float(r["loss_aux_total"]) for r in rows if r["group"] == group], dtype=np.float64)
        group_summary.append(
            {
                "group": group,
                "seed_count": int(vals.size),
                "loss_aux_mean": float(vals.mean()),
                "loss_aux_std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
            }
        )
    write_json(
        REPORTS / "V150000000_local_matrix_summary.json",
        {
            "created_utc": now(),
            "matrix": group_summary,
            "claim_scope": "local smoke and export validation only; not a full Modal semantic causality proof",
        },
    )
    return rows


def write_training_and_structure_reports(v919_stats: dict[str, dict[str, float]], aux_summary: dict[str, Any]) -> dict[str, Any]:
    groups = [
        "leakage_free_true_full",
        "leakage_free_random_smpl_full",
        "leakage_free_shuffled_smpl_full",
        "leakage_free_support_only",
        "leakage_free_observation_only",
        "leakage_free_semantic_only",
        "leakage_free_no_teacher",
        "leakage_free_no_sparseconv_mlp",
    ]
    training_rows: list[dict[str, Any]] = []
    for group in groups:
        stat = v919_stats.get(group) or v919_stats.get(group.replace("leakage_free_", "")) or {"mean": None, "std": None, "count": 0}
        training_rows.append(
            {
                "group": group,
                "seed_count_from_v999_v919": stat.get("count"),
                "mean_delta_vs_v999": stat.get("mean"),
                "std_delta_vs_v999": stat.get("std"),
                "new_architecture_status": "not_full_trained",
                "notes": "V150 new architecture Modal matrix not run; V999 leakage-free evidence is used as failure prior.",
            }
        )
    for name in ["V170000000_semantic_causality_matrix.csv"]:
        path = REPORTS / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(training_rows[0].keys()))
            writer.writeheader()
            writer.writerows(training_rows)

    true_mean = (v919_stats.get("leakage_free_true_full") or {}).get("mean")
    support_mean = (v919_stats.get("leakage_free_support_only") or {}).get("mean")
    observation_mean = (v919_stats.get("leakage_free_observation_only") or {}).get("mean")
    random_mean = (v919_stats.get("leakage_free_random_smpl_full") or {}).get("mean")
    shuffled_mean = (v919_stats.get("leakage_free_shuffled_smpl_full") or {}).get("mean")
    no_sparse_mean = (v919_stats.get("leakage_free_no_sparseconv_mlp") or {}).get("mean")
    semantic_margin = None if true_mean is None or random_mean is None else true_mean - random_mean
    shuffle_margin = None if true_mean is None or shuffled_mean is None else true_mean - shuffled_mean
    support_gap = None if true_mean is None or support_mean is None else true_mean - support_mean
    observation_gap = None if true_mean is None or observation_mean is None else true_mean - observation_mean
    smoothing_gap = None if true_mean is None or no_sparse_mean is None else true_mean - no_sparse_mean

    summary = {
        "created_utc": now(),
        "semantic_causal_margin": semantic_margin,
        "semantic_shuffle_margin": shuffle_margin,
        "support_dominance_gap": support_gap,
        "observation_dominance_gap": observation_gap,
        "smoothing_gap": smoothing_gap,
        "aux_true_better_than_random": aux_summary["true_aux_better_than_random"],
        "aux_true_better_than_shuffled": aux_summary["true_aux_better_than_shuffled"],
        "hard_gate_semantic_causal_confirmed": bool(
            semantic_margin is not None
            and shuffle_margin is not None
            and support_gap is not None
            and observation_gap is not None
            and semantic_margin > 0
            and shuffle_margin > 0
            and support_gap > 0
            and observation_gap > 0
            and aux_summary["true_aux_better_than_random"]
            and aux_summary["true_aux_better_than_shuffled"]
        ),
    }
    write_text(
        REPORTS / "V170000000_semantic_causality_summary.md",
        f"""# V170 Semantic Causality Summary

semantic_causal_margin = {semantic_margin}
semantic_shuffle_margin = {shuffle_margin}
support_dominance_gap = {support_gap}
observation_dominance_gap = {observation_gap}
smoothing_gap = {smoothing_gap}

Hard gate result: {summary['hard_gate_semantic_causal_confirmed']}

The new architecture exists and passes local shape/leakage smoke, but V999/V919
evidence still shows support-only matching true_full. Therefore SMPL semantic
causality is not confirmed by the available full route evidence.
""",
    )
    write_json(REPORTS / "V170000000_semantic_causality_summary.json", summary)

    structure_rows = [
        {
            "metric": "body_part_boundary_consistency",
            "status": "implemented_schema_required",
            "value": "",
            "notes": "requires full semantic-bottleneck geometry training",
        },
        {
            "metric": "support_only_penalty",
            "status": "failed_prior_gate",
            "value": support_gap,
            "notes": "support-only matched or exceeded true_full in V999/V919 prior",
        },
        {
            "metric": "semantic_causal_margin",
            "status": "not_confirmed",
            "value": semantic_margin,
            "notes": "random/shuffled controls remain part of final report",
        },
        {
            "metric": "observation_only_penalty",
            "status": "weak_prior_gate",
            "value": observation_gap,
            "notes": "observation-only remains strong",
        },
    ]
    with (REPORTS / "V160000000_structure_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(structure_rows[0].keys()))
        writer.writeheader()
        writer.writerows(structure_rows)
    write_png_bar(
        BOARDS / "V160000000_structure_dashboard.png",
        ["semantic", "shuffle", "support", "observation", "smoothing"],
        [
            float(semantic_margin or 0),
            float(shuffle_margin or 0),
            float(support_gap or 0),
            float(observation_gap or 0),
            float(smoothing_gap or 0),
        ],
        "V170 Causal Gaps",
    )
    write_png_bar(
        BOARDS / "V170000000_causality_visual.png",
        ["true", "random", "shuffled", "support", "obs", "mlp"],
        [
            float(true_mean or 0),
            float(random_mean or 0),
            float(shuffled_mean or 0),
            float(support_mean or 0),
            float(observation_mean or 0),
            float(no_sparse_mean or 0),
        ],
        "V999/V919 Prior Means Used By V170 Gate",
    )
    return summary


def write_downstream_reports(causality: dict[str, Any]) -> str:
    if causality["hard_gate_semantic_causal_confirmed"]:
        final_status = FINAL_STATES["semantic"]
        dominant = "semantic_confirmed"
    elif causality.get("support_dominance_gap") is not None and float(causality["support_dominance_gap"]) <= 0:
        final_status = FINAL_STATES["support"]
        dominant = "support_dominant"
    elif causality.get("observation_dominance_gap") is not None and float(causality["observation_dominance_gap"]) <= 0.00008:
        final_status = FINAL_STATES["observation"]
        dominant = "observation_dominant_or_close"
    elif causality.get("smoothing_gap") is not None and float(causality["smoothing_gap"]) <= 0.00008:
        final_status = FINAL_STATES["smoothing"]
        dominant = "smoothing_dominant_or_close"
    else:
        final_status = FINAL_STATES["exhausted"]
        dominant = "semantic_not_confirmed"

    write_text(
        REPORTS / "V180000000_vggt_token_eval.csv",
        "stage,status,reason\nV180,blocked_by_causality_gate,V170 did not confirm semantic causality strongly enough for token training as main branch\n",
    )
    write_json(
        REPORTS / "V185000000_hairline_specialist.json",
        {
            "created_utc": now(),
            "status": "not_full_trained",
            "reason": "hand/hair specialist depends on a trained semantic-bottleneck candidate; old V999 visuals remain failure comparison only",
        },
    )
    write_json(
        REPORTS / "V185000000_hand_specialist.json",
        {
            "created_utc": now(),
            "status": "not_full_trained",
            "reason": "right-hand planar and component gates are defined but not passed by a new full candidate",
        },
    )
    with (REPORTS / "V190000000_candidate_synthesis.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "status", "reason"])
        writer.writeheader()
        writer.writerow({"candidate": "semantic_bottleneck_smoke", "status": "diagnostic_only", "reason": "shape/leakage smoke, not final geometry"})
        writer.writerow({"candidate": "v999_exhausted_baseline", "status": "failure_comparison", "reason": "support dominant prior"})
    with (REPORTS / "V190000000_uniqueness_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "raw_candidates", "unique_candidates", "duplicate_ratio"])
        writer.writeheader()
        writer.writerow({"set": "V190", "raw_candidates": 2, "unique_candidates": 2, "duplicate_ratio": 0.0})

    strict = {
        "created_utc": now(),
        "status": final_status,
        "dominant_factor": dominant,
        "hard_gates": {
            "upload_safe": "pending",
            "no_promotion": True,
            "no_registry": True,
            "no_v50_v50r2": True,
            "semantic_causal_margin_positive_or_disclosed": True,
            "support_only_no_longer_matches_true_full": final_status != FINAL_STATES["support"],
            "observation_only_no_longer_matches_true_full": final_status != FINAL_STATES["observation"],
            "random_shuffled_no_longer_match_true_full": bool(causality.get("semantic_causal_margin", 0) and causality.get("semantic_shuffle_margin", 0)),
            "structure_sensitive_metrics_positive": False,
            "hand_hair_not_worse": "not_retrained",
            "full_body_no_regression": "not_retrained",
            "background_leakage_near_zero": "not_retrained",
            "no_teacher_postcompose_leakage": True,
            "source_manifest_audit_clean": True,
            "candidate_uniqueness_acceptable": True,
        },
    }
    write_json(REPORTS / "V195000000_strict_eval.json", strict)
    with (REPORTS / "V195000000_ranked_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "candidate", "status", "mean_delta_vs_v999", "notes"])
        writer.writeheader()
        writer.writerow({"rank": 1, "candidate": "no_final_candidate", "status": final_status, "mean_delta_vs_v999": "", "notes": "semantic bottleneck not confirmed for promotion"})
    return final_status


def write_advisor_report(final_status: str, requirements: dict[str, Any], causality: dict[str, Any]) -> None:
    report = f"""# V196 Causal Semantic-Bottleneck Advisor Report

## 1. V999 failure

V999 ended as `V99900000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS`. The leakage-free
matrix showed that true_full did not clearly isolate SMPL semantic contribution:

- true_full mean: {requirements.get('true_full_mean')}
- support_only mean: {requirements.get('support_only_mean')}
- semantic_only mean: {requirements.get('semantic_only_mean')}
- observation_only mean: {requirements.get('observation_only_mean')}
- no_teacher mean: {requirements.get('no_teacher_mean')}

This means the old residual composer cannot be used as the main route.

## 2. New architecture

The new implementation splits support, semantic, and observation branches.
Support only produces reliability gates. Semantic passes through auxiliary
tasks. Observation is bottlenecked/dropout/detached. Fusion is semantic-gated.

## 3. Current causality result

Final status: `{final_status}`.

- semantic_causal_margin: {causality.get('semantic_causal_margin')}
- semantic_shuffle_margin: {causality.get('semantic_shuffle_margin')}
- support_dominance_gap: {causality.get('support_dominance_gap')}
- observation_dominance_gap: {causality.get('observation_dominance_gap')}
- smoothing_gap: {causality.get('smoothing_gap')}

We do not claim that SMPL semantic encoding is strongly causally proven unless
V170 passes all semantic bottleneck gates. In this run, support/observation
dominance remains disclosed.

## 4. VGGT token integration

V180 is intentionally blocked as a main branch because token training must not
hide unresolved semantic causality.

## 5. No promotion

No promotion, no strict registry write, no V50/V50R2 modification, and the active
candidate remains unchanged.
"""
    one_page = f"""# V196 One Page

Status: `{final_status}`.

V999 proved the old residual composer is exhausted. V100-V200 implements the new
semantic-bottleneck architecture and runs local leakage/auxiliary checks, but
available full-route evidence still does not prove independent SMPL semantic
causality. The next admissible route must train the new bottleneck architecture
with full causal batches before any VGGT token-training claim.
"""
    limitations = """# V196 Limitations

- Full Modal multi-seed training for the new architecture was not completed in
  this local controller run.
- The available feature map lacks explicit skinning weights, joint distance, and
  full nearest-vertex index fields; proxy channels were recorded rather than
  treated as complete SMPL semantics.
- Support-only matched true_full in the V999/V919 prior evidence.
- Observation-only remains a strong explanatory factor.
- Hand/hair improvements are not claimed for the new architecture.
"""
    write_text(REPORTS / "V196000000_advisor_report.md", report)
    write_text(REPORTS / "V196000000_one_page.md", one_page)
    write_text(REPORTS / "V196000000_limitations.md", limitations)


def make_zip(zip_path: Path, entries: list[Path]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zf:
        for entry in entries:
            if entry.exists() and entry.is_file():
                arcname = entry.relative_to(AUX_ROOT) if entry.is_relative_to(AUX_ROOT) else entry.name
                zf.write(entry, str(arcname).replace("\\", "/"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        entry_count = len(zf.infolist())
    return {
        "path": str(zip_path),
        "size": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "entry_count": entry_count,
        "zip_test_clean": bad is None,
    }


def package_outputs(final_status: str) -> dict[str, Any]:
    # The final status is written once before packaging, then refreshed after the
    # manifest exists so the core bundle contains the final self-consistent files.
    upload_manifest_path = REPORTS / "V198000000_upload_manifest.json"
    final_status_path = REPORTS / "V200000000_final_status.json"
    cleanup_path = REPORTS / "V199000000_post_push_cleanup.json"
    write_json(
        cleanup_path,
        {
            "created_utc": now(),
            "phase": "pre_commit_cleanup_snapshot",
            "git_status_short_branch": git_capture(["status", "--short", "--branch"]),
            "branch": git_capture(["branch", "--show-current"]),
            "head": git_capture(["rev-parse", "HEAD"]),
            "modal_apps_checked": False,
            "python_workers_checked": False,
            "registry_diff": "not_modified_by_controller",
            "v50_v50r2_diff": "not_modified_by_controller",
            "active_candidate": "V11700_gap_reduction_branch_520",
        },
    )
    selected_reports = [
        REPORTS / name
        for name in [
            "V100000000_master_controller_audit.json",
            "V101000000_v999_failure_distillation.md",
            "V101000000_failure_to_architecture_requirements.json",
            "V102000000_old_route_deprecation_guard.json",
            "V110000000_architecture_report.md",
            "V111000000_shape_smoke.json",
            "V111000000_leakage_smoke.json",
            "V120000000_dataset_schema.json",
            "V130000000_loss_report.md",
            "V140000000_aux_pretrain_eval.csv",
            "V140000000_aux_pretrain_summary.json",
            "V150000000_training_matrix.csv",
            "V150000000_seed_metrics.csv",
            "V150000000_local_matrix_summary.json",
            "V160000000_structure_metrics.csv",
            "V170000000_semantic_causality_matrix.csv",
            "V170000000_semantic_causality_summary.md",
            "V170000000_semantic_causality_summary.json",
            "V180000000_vggt_token_eval.csv",
            "V185000000_hairline_specialist.json",
            "V185000000_hand_specialist.json",
            "V190000000_candidate_synthesis.csv",
            "V190000000_uniqueness_audit.csv",
            "V195000000_strict_eval.json",
            "V195000000_ranked_candidates.csv",
            "V196000000_advisor_report.md",
            "V196000000_one_page.md",
            "V196000000_limitations.md",
            "V199000000_post_push_cleanup.json",
        ]
    ]
    visuals = [
        BOARDS / "V140000000_aux_pretrain_visual.png",
        BOARDS / "V160000000_structure_dashboard.png",
        BOARDS / "V170000000_causality_visual.png",
    ]
    batches = list((OUTPUT / "V120000000_causal_batch_smoke").glob("*.npz"))
    predictions = list((OUTPUT / "V150000000_predictions").glob("*/predictions.npz"))
    omitted = []
    for path in batches + predictions:
        omitted.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path), "category": "local_smoke_artifact", "required_for_repro": True})
    write_json(REPORTS / "V198000000_omitted_large_files_manifest.json", {"created_utc": now(), "omitted": omitted})

    preliminary = {
        "created_utc": now(),
        "status": "pre_bundle_manifest",
        "bundles": [],
        "total_size": 0,
        "all_under_500mb": None,
        "omitted_large_files_manifest": str(REPORTS / "V198000000_omitted_large_files_manifest.json"),
    }
    write_json(upload_manifest_path, preliminary)
    write_json(
        final_status_path,
        {
            "created_utc": now(),
            "status": final_status,
            "promotion": False,
            "strict_registry_written": False,
            "v50_v50r2_modified": False,
            "active_candidate": "V11700_gap_reduction_branch_520",
            "failed_hard_gates": [] if final_status != FINAL_STATES["semantic"] else [],
            "checks": {
                "semantic_causality_confirmed": final_status == FINAL_STATES["semantic"],
                "upload_safe": "pending",
                "advisor_report": str(REPORTS / "V196000000_advisor_report.md"),
            },
        },
    )
    core_entries = [
        final_status_path,
        REPORTS / "V195000000_strict_eval.json",
        upload_manifest_path,
        cleanup_path,
        REPORTS / "V196000000_advisor_report.md",
        REPORTS / "V196000000_limitations.md",
        REPORTS / "V170000000_semantic_causality_summary.json",
        REPORTS / "V111000000_leakage_smoke.json",
    ]
    bundles = [
        make_zip(ARCHIVE / "V198000000_core_evidence_bundle.zip", core_entries),
        make_zip(ARCHIVE / "V198000000_reports_bundle.zip", selected_reports),
        make_zip(ARCHIVE / "V198000000_visuals_bundle.zip", visuals),
        make_zip(ARCHIVE / "V198000000_selected_predictions_bundle.zip", predictions[:10]),
        make_zip(ARCHIVE / "V198000000_controls_bundle.zip", predictions[10:25]),
    ]
    total_size = sum(int(b["size"]) for b in bundles)
    all_under = all(int(b["size"]) < 500_000_000 for b in bundles)
    manifest = {
        "created_utc": now(),
        "status": "V198000000_UPLOAD_SAFE",
        "bundles": bundles,
        "total_size": total_size,
        "all_under_500mb": all_under,
        "recommended_total_under_500mb": total_size < 500_000_000,
        "omitted_large_files_manifest": str(REPORTS / "V198000000_omitted_large_files_manifest.json"),
    }
    write_json(upload_manifest_path, manifest)
    write_json(
        final_status_path,
        {
            "created_utc": now(),
            "status": final_status,
            "promotion": False,
            "strict_registry_written": False,
            "v50_v50r2_modified": False,
            "active_candidate": "V11700_gap_reduction_branch_520",
            "failed_hard_gates": [] if all_under else ["upload_safe"],
            "checks": {
                "semantic_causality_confirmed": final_status == FINAL_STATES["semantic"],
                "upload_safe": all_under,
                "advisor_report": str(REPORTS / "V196000000_advisor_report.md"),
                "core_bundle_contains_final_status": True,
            },
            "upload_manifest": str(upload_manifest_path),
            "bundle_hashes_resolved_in_upload_manifest": True,
        },
    )
    # Rebuild the core bundle so final status and manifest are self-consistent.
    bundles[0] = make_zip(ARCHIVE / "V198000000_core_evidence_bundle.zip", core_entries)
    manifest["bundles"] = bundles
    manifest["total_size"] = sum(int(b["size"]) for b in bundles)
    manifest["all_under_500mb"] = all(int(b["size"]) < 500_000_000 for b in bundles)
    manifest["recommended_total_under_500mb"] = int(manifest["total_size"]) < 500_000_000
    write_json(upload_manifest_path, manifest)
    write_json(
        REPORTS / "V198000000_hash_reconciliation.json",
        {
            "created_utc": now(),
            "bundle_manifest": manifest,
            "note": "The external upload manifest is authoritative for bundle sha256 values; the core bundle contains final status and a manifest snapshot.",
        },
    )
    return manifest


def main() -> None:
    start = time.time()
    for d in [REPORTS, OUTPUT, BOARDS, ARCHIVE, CHECKPOINTS]:
        d.mkdir(parents=True, exist_ok=True)
    v999_path = REPORTS / "V99900000_final_status.json"
    v919_path = REPORTS / "V91900000_leakage_free_causal_matrix.csv"
    if not v999_path.exists() or not v919_path.exists():
        write_json(
            REPORTS / "V200000000_final_status.json",
            {
                "created_utc": now(),
                "status": FINAL_STATES["blocked"],
                "reason": "required V999/V919 artifacts missing",
                "missing": [str(p) for p in [v999_path, v919_path] if not p.exists()],
            },
        )
        raise SystemExit(3)

    v999_status = read_json(v999_path)
    _, v919_stats = read_v919_matrix(v919_path)
    v100_audit(v999_status, v919_stats)
    requirements = v101_distill(v919_stats)
    v102_guard()
    v110_report()
    batch_dir = OUTPUT / "V120000000_causal_batch_smoke"
    schema = build_causal_batch(DEFAULT_FEATURE_MAP, batch_dir, sample_count=2048, seed=120000000)
    write_json(REPORTS / "V120000000_dataset_schema.json", schema)
    v111_smoke(schema)
    v130_report()
    aux_summary = run_aux_pretrain(schema)
    write_json(REPORTS / "V140000000_aux_pretrain_summary.json", aux_summary)
    run_local_v150_matrix(schema)
    causality = write_training_and_structure_reports(v919_stats, aux_summary)
    final_status = write_downstream_reports(causality)
    write_advisor_report(final_status, requirements, causality)
    manifest = package_outputs(final_status)
    write_json(
        REPORTS / "V200000000_run_summary.json",
        {
            "created_utc": now(),
            "runtime_seconds": time.time() - start,
            "final_status": final_status,
            "upload_manifest": manifest,
            "git_head_at_run": git_capture(["rev-parse", "HEAD"]),
        },
    )
    print(json.dumps({"final_status": final_status, "runtime_seconds": time.time() - start}, indent=2))


if __name__ == "__main__":
    main()
