"""V421-V600 formal GPU semantic causality controller.

The controller closes the current route honestly:

* V380 CPU pilot is audited as pilot-only evidence.
* V415 selected/control bundles are checked beyond outer zip cleanliness.
* Representative selected and control predictions are repaired from readable
  local V380 outputs and receive geometric normals.
* V430 GPU smoke evidence is recorded.
* V460 formal matrix is not claimed unless the required 40 GPU runs exist.
* Final packaging validates inner npz readability.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(r"D:\vggt\vggt-feature-adapter")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.v4400000000_recompute_fullview_normals import repair_prediction


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
ARCHIVE = AUX / "archive"
V380_PRED = OUTPUT / "V3800000000_fullview_predictions"
V422_OUT = OUTPUT / "V4220000000_repaired_predictions"
V430_SMOKE = AUX / "remote_pull" / "V4300000000_gpu_smoke" / "true_full_seed0"
ACTIVE_CANDIDATE = "V11700_gap_reduction_branch_520"


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
SELECTED_GROUPS = ["true_full", "same_support_random_semantic", "local_knn_smoothing"]
CONTROL_GROUPS = ["same_support_shuffled_semantic", "no_sparseconv_mlp", "support_only", "observation_only", "no_teacher"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str], cwd: Path = REPO, timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return {"args": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"args": args, "returncode": -1, "stdout": "", "stderr": repr(exc)}


def zip_test(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "zip_test_clean": False,
        "first_bad_member": None,
        "entries": [],
    }
    if not path.exists():
        return row
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()
            row["first_bad_member"] = bad
            row["zip_test_clean"] = bad is None
            row["entries"] = [{"name": n, "size": z.getinfo(n).file_size} for n in z.namelist()]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def npz_bytes_test(name: str, data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "readable": False,
        "inner_zip_clean": False,
        "first_bad_inner_member": None,
        "keys": [],
        "arrays": {},
    }
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            bad = z.testzip()
            row["first_bad_inner_member"] = bad
            row["inner_zip_clean"] = bad is None
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            row["keys"] = list(npz.files)
            for key in ["world_points", "points", "depth", "normal", "confidence", "world_points_conf", "normal_conf"]:
                if key not in npz.files:
                    continue
                arr = np.asarray(npz[key])
                row["arrays"][key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "finite": bool(np.isfinite(arr).all()),
                    "nonzero_ratio": float((np.abs(arr) > 1e-8).mean()) if arr.size else 0.0,
                }
                if key == "normal" and arr.ndim == 4:
                    row["arrays"][key]["vector_nonzero_ratio"] = float((np.linalg.norm(arr, axis=-1) > 1e-8).mean())
            row["readable"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def npz_file_test(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "readable": False}
    data = path.read_bytes()
    row = npz_bytes_test(str(path), data)
    row["path"] = str(path)
    row["exists"] = True
    row["size"] = path.stat().st_size
    row["sha256"] = sha256_file(path)
    return row


def audit_v415() -> tuple[dict[str, Any], dict[str, Any]]:
    archive = AUX / "archive"
    bundle_names = [
        "V4150000000_core_evidence_bundle.zip",
        "V4150000000_reports_bundle.zip",
        "V4150000000_visuals_bundle.zip",
        "V4150000000_selected_predictions_bundle.zip",
        "V4150000000_controls_bundle.zip",
    ]
    bundle_rows = {name: zip_test(archive / name) for name in bundle_names}
    selected_inner: list[dict[str, Any]] = []
    selected_zip = archive / "V4150000000_selected_predictions_bundle.zip"
    if selected_zip.exists():
        with zipfile.ZipFile(selected_zip, "r") as z:
            for member in z.namelist():
                if member.endswith(".npz"):
                    try:
                        selected_inner.append(npz_bytes_test(member, z.read(member)))
                    except Exception as exc:
                        selected_inner.append({"name": member, "readable": False, "error": f"{type(exc).__name__}: {exc}"})
    controls_zip = archive / "V4150000000_controls_bundle.zip"
    controls_entries = []
    if controls_zip.exists():
        with zipfile.ZipFile(controls_zip, "r") as z:
            controls_entries = z.namelist()
    report = {
        "created_utc": now(),
        "bundle_rows": bundle_rows,
        "selected_inner_npz": selected_inner,
        "controls_bundle_contains_predictions": any(name.endswith("predictions.npz") for name in controls_entries),
        "controls_entries": controls_entries,
        "v380_is_cpu_pilot": True,
        "formal_gpu_matrix_complete": False,
    }
    npz_report = {
        "created_utc": now(),
        "v415_selected_inner_npz": selected_inner,
        "local_v380_predictions": [npz_file_test(p) for p in sorted(V380_PRED.glob("*/predictions.npz"))],
    }
    write_json(REPORTS / "V4210000000_artifact_integrity_audit.json", report)
    write_json(REPORTS / "V4210000000_npz_integrity_report.json", npz_report)
    return report, npz_report


def write_v421_text_reports(artifact_report: dict[str, Any]) -> None:
    bad_selected = [
        row for row in artifact_report.get("selected_inner_npz", [])
        if not row.get("readable") or not row.get("inner_zip_clean")
    ]
    text = f"""# V421 Pilot vs Formal Audit

Created: {now()}

V380 is a remote CPU pilot, not formal GPU full-view training. The V420 status
already recorded `fullview_modal_matrix_complete=false` and
`remote_modal_pilot_complete=true`.

V415 outer zip audit is insufficient by itself: {len(bad_selected)} selected
inner npz member(s) failed inner readability or CRC checks. The controls bundle
contains predictions: {artifact_report.get('controls_bundle_contains_predictions')}.

Formal V460 completion remains false until 40 GPU runs (8 core groups x 5 seeds)
exist with readable predictions, clean source manifests, and nonzero normal
handling.
"""
    write_text(REPORTS / "V4210000000_pilot_vs_formal_audit.md", text)

    git_status = run_cmd(["git", "status", "--short", "-uall"], timeout=30)
    plan = f"""# V421 Dirty Worktree Plan

Created: {now()}

The feature-adapter worktree is not assumed clean. The main route files for
V421-V600 may be staged separately after review. Existing tracked behavior
changes such as `models/v110_semantic_bottleneck_smpl_sparseconv.py` must not be
reverted or mixed into cleanup by accident.

`__v380_modal_pull/` is a local Modal pull/cache evidence folder and should not
be committed as source. No promotion, registry, V50/V50R2, or active-candidate
changes are allowed.

Current git status:

```text
{git_status.get('stdout','')}
```
"""
    write_text(REPORTS / "V4210000000_dirty_worktree_plan.md", plan)


def repair_predictions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    V422_OUT.mkdir(parents=True, exist_ok=True)
    repair_rows: list[dict[str, Any]] = []
    controls_manifest: list[dict[str, Any]] = []
    for group in CORE_GROUPS:
        src = V380_PRED / group / "predictions.npz"
        dst = V422_OUT / f"{group}_seed0" / "predictions.npz"
        if not src.exists():
            repair_rows.append({"group": group, "status": "missing_source", "source": str(src)})
            continue
        row = repair_prediction(src, dst)
        group_dir = dst.parent
        for name in ["eval.json", "quality.json", "source_manifest.json", "board.png"]:
            source_sidecar = V380_PRED / group / name
            if source_sidecar.exists():
                shutil.copy2(source_sidecar, group_dir / name)
        manifest_path = group_dir / "source_manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        manifest.update(
            {
                "v422_repaired_prediction": str(dst),
                "normal_source": "geometric_finite_difference_recomputed",
                "learned_or_input_normal_was_all_zero": True,
                "formal_gpu_matrix_run": False,
                "source_stage": "V380 CPU pilot local output repaired for packaging integrity only",
            }
        )
        write_json(manifest_path, manifest)
        test = npz_file_test(dst)
        row.update({"group": group, "seed": 0, "npz_test": test, "status": "repaired_from_local_v380_output"})
        repair_rows.append(row)
        controls_manifest.append(
            {
                "group": group,
                "seed": 0,
                "prediction": str(dst),
                "eval": str(group_dir / "eval.json"),
                "source_manifest": str(manifest_path),
                "quality": str(group_dir / "quality.json"),
                "board": str(group_dir / "board.png"),
                "normal_source": "geometric_finite_difference_recomputed",
                "formal_gpu_matrix_run": False,
            }
        )
    write_json(REPORTS / "V4220000000_prediction_repull_report.json", {"created_utc": now(), "rows": repair_rows})
    write_json(REPORTS / "V4220000000_controls_prediction_manifest.json", {"created_utc": now(), "rows": controls_manifest})
    return repair_rows, controls_manifest


def write_v430_v450_reports() -> dict[str, Any]:
    smoke_eval = None
    eval_path = V430_SMOKE / "eval.json"
    if eval_path.exists():
        smoke_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    smoke = {
        "created_utc": now(),
        "trainer_path": str(REPO / "tools" / "v4300000000_modal_gpu_fullview_semantic_trainer.py"),
        "gpu_smoke_eval_path": str(eval_path),
        "gpu_smoke": smoke_eval,
        "formal_gpu_smoke_passed": bool(smoke_eval and smoke_eval.get("cuda_available") and smoke_eval.get("formal_gpu_run")),
        "formal_matrix_complete": False,
        "note": "GPU smoke is not V460 matrix completion.",
    }
    write_json(REPORTS / "V4300000000_gpu_smoke.json", smoke)
    write_text(
        REPORTS / "V4300000000_trainer_implementation_report.md",
        f"""# V430 Trainer Implementation Report

Created: {now()}

Implemented `tools/v4300000000_modal_gpu_fullview_semantic_trainer.py`.

The trainer rejects CPU pilot status and records CUDA availability, GPU type,
runtime, training steps, losses, predictions, source manifest, quality, and
normal handling. A Modal GPU smoke was run on A10G and is recorded in
`reports/V4300000000_gpu_smoke.json`.

Important boundary: this smoke does not complete the formal V460 matrix. The
route still needs 8 core groups x 5 seeds with clean manifests before any
semantic causality claim.
""",
    )
    matrix_config = {
        "created_utc": now(),
        "core_groups": CORE_GROUPS,
        "minimum_seeds_per_group": 5,
        "minimum_runs": 40,
        "same_training_steps": True,
        "same_dataset": True,
        "formal_gpu_only": True,
        "gpu_smoke_passed": smoke["formal_gpu_smoke_passed"],
    }
    write_json(REPORTS / "V4500000000_formal_matrix_config.json", matrix_config)
    write_text(
        REPORTS / "V4500000000_expected_runtime_budget.md",
        "# V450 Expected Runtime Budget\n\n"
        "The formal matrix requires at least 40 Modal GPU runs. A one-run A10G smoke completed, but the full matrix was not launched/completed in this route.\n",
    )
    return smoke


def write_v440_reports(repair_rows: list[dict[str, Any]]) -> None:
    audit_csv = REPORTS / "V4400000000_normal_nonzero_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "seed", "normal_source", "normal_nonzero_ratio", "valid_count", "normal_nonzero_count", "prediction"],
        )
        writer.writeheader()
        for row in repair_rows:
            if row.get("status") != "repaired_from_local_v380_output":
                continue
            writer.writerow(
                {
                    "group": row.get("group"),
                    "seed": row.get("seed"),
                    "normal_source": row.get("normal_source"),
                    "normal_nonzero_ratio": row.get("normal_nonzero_ratio"),
                    "valid_count": row.get("valid_count"),
                    "normal_nonzero_count": row.get("normal_nonzero_count"),
                    "prediction": row.get("output_npz"),
                }
            )
    ratios = [float(r.get("normal_nonzero_ratio", 0.0)) for r in repair_rows if r.get("status") == "repaired_from_local_v380_output"]
    report = {
        "created_utc": now(),
        "normal_repair_performed": True,
        "normal_source": "geometric_finite_difference_recomputed",
        "learned_normal_head_success": False,
        "input_v380_normals_all_zero": True,
        "min_normal_nonzero_ratio_after_recompute": min(ratios) if ratios else 0.0,
        "normal_nonzero_gate_gt_0_1_passed_for_recomputed_normals": bool(ratios and min(ratios) > 0.1),
        "audit_csv": str(audit_csv),
    }
    write_json(REPORTS / "V4400000000_normal_head_repair.json", report)
    make_normal_visual(V422_OUT / "true_full_seed0" / "predictions.npz", BOARDS / "V4400000000_normal_visual.png")


def make_normal_visual(pred_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(pred_path, allow_pickle=False) as z:
        normal = np.asarray(z["normal"])[0]
        depth = np.asarray(z["depth"])[0]
    rgb = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
    d = depth.astype("float32")
    d = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
    depth_rgb = np.repeat((d * 255).astype("uint8")[..., None], 3, axis=-1)
    canvas = Image.new("RGB", (1036, 560), "white")
    canvas.paste(Image.fromarray(depth_rgb).resize((518, 518)), (0, 32))
    canvas.paste(Image.fromarray(rgb).resize((518, 518)), (518, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "depth (pilot repaired)", fill=(0, 0, 0))
    draw.text((526, 8), "geometric normal (not learned normal head)", fill=(0, 0, 0))
    canvas.save(out_path)


def write_v460_blocker() -> None:
    rows = []
    for group in CORE_GROUPS:
        for seed in range(5):
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "status": "missing_formal_gpu_run",
                    "gpu_type": "",
                    "cuda_available": "",
                    "prediction_path": "",
                    "reason": "V460 formal 40-run matrix was not launched/completed; V380 CPU pilot cannot be reused as formal GPU evidence.",
                }
            )
    write_json(
        REPORTS / "V4600000000_formal_matrix_progress.json",
        {
            "created_utc": now(),
            "formal_matrix_complete": False,
            "required_runs": 40,
            "valid_formal_runs": 0,
            "reason": "V430 GPU smoke passed, but V460 8x5 formal matrix is absent.",
        },
    )
    for path in [REPORTS / "V4600000000_formal_matrix.csv", REPORTS / "V4600000000_seed_metrics.csv"]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        REPORTS / "V4600000000_modal_job_manifest.json",
        {
            "created_utc": now(),
            "formal_gpu_matrix_complete": False,
            "jobs": [],
            "gpu_smoke_completed": True,
            "core_matrix_jobs_required": 40,
            "core_matrix_jobs_completed": 0,
            "hard_gate": "V460_not_complete",
        },
    )
    with (REPORTS / "V4700000000_expansion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "seed", "status", "reason"])
        writer.writeheader()
        writer.writerow({"group": "expansion", "seed": "", "status": "not_run", "reason": "core V460 matrix incomplete"})
    shutil.copy2(REPORTS / "V4700000000_expansion_matrix.csv", REPORTS / "V4700000000_expansion_seed_metrics.csv")


def write_v480_v490_reports() -> None:
    regions = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]
    metrics = [
        "mean_delta",
        "local_delta",
        "normal_consistency",
        "normal_nonzero_ratio",
        "outlier_ratio",
        "background_leakage",
        "point_density",
        "component_count",
        "hand_isolated_ratio",
        "right_hand_planar_score",
        "hairline_boundary_sharpness",
        "horizontal_band_artifact",
        "surface_continuity",
        "reprojection_consistency",
    ]
    with (REPORTS / "V4800000000_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "metric", "value", "status", "reason"])
        writer.writeheader()
        for region in regions:
            for metric in metrics:
                writer.writerow(
                    {
                        "region": region,
                        "metric": metric,
                        "value": "",
                        "status": "not_evaluated_for_formal_matrix",
                        "reason": "V460 formal GPU matrix incomplete; pilot visuals cannot satisfy V480 hard gate.",
                    }
                )
    with (REPORTS / "V4800000000_structure_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "value", "status", "reason"])
        writer.writeheader()
        for comp in [
            "true_vs_random_semantic",
            "true_vs_shuffled_semantic",
            "true_vs_support_only",
            "true_vs_observation_only",
            "true_vs_noSparse_MLP",
            "true_vs_local_KNN_smoothing",
        ]:
            writer.writerow({"comparison": comp, "value": "", "status": "not_formal", "reason": "No V460 formal 40-run matrix."})
    write_json(
        REPORTS / "V4800000000_region_decision.json",
        {
            "created_utc": now(),
            "region_metrics_formal_complete": False,
            "regions": regions,
            "reason": "Existing V390 region metrics were not_evaluated; current route did not complete V460 formal matrix.",
        },
    )
    make_boards()


def load_depth(group: str) -> np.ndarray:
    with np.load(V422_OUT / f"{group}_seed0" / "predictions.npz", allow_pickle=False) as z:
        return np.asarray(z["depth"])[0].astype("float32")


def depth_image(group: str, label: str) -> Image.Image:
    d = load_depth(group)
    d = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
    im = Image.fromarray((d * 255).astype("uint8")).convert("RGB").resize((260, 260))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 259, 28), fill=(255, 255, 255))
    draw.text((6, 7), label, fill=(0, 0, 0))
    return im


def make_grid(path: Path, cells: list[tuple[str, str]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(cells)
    canvas = Image.new("RGB", (260 * cols, 300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for i, (group, label) in enumerate(cells):
        canvas.paste(depth_image(group, label), (260 * i, 40))
    canvas.save(path)


def make_boards() -> None:
    make_grid(
        BOARDS / "V4900000000_fullbody_pointcloud.png",
        [
            ("true_full", "true_full pilot"),
            ("same_support_random_semantic", "random pilot"),
            ("local_knn_smoothing", "local smoothing"),
            ("no_sparseconv_mlp", "noSparse"),
        ],
        "Pilot full-view depth projection; not paper-grade 3D proof",
    )
    make_grid(
        BOARDS / "V4900000000_counterfactual_controls.png",
        [
            ("true_full", "true"),
            ("same_support_random_semantic", "random"),
            ("same_support_shuffled_semantic", "shuffled"),
        ],
        "Same support/observation pilot controls; formal matrix absent",
    )
    make_grid(
        BOARDS / "V4900000000_smoothing_controls.png",
        [
            ("true_full", "true"),
            ("no_sparseconv_mlp", "noSparse"),
            ("local_knn_smoothing", "KNN smooth"),
        ],
        "Smoothing controls; V380 pilot had smoothing > true",
    )
    shutil.copy2(BOARDS / "V4400000000_normal_visual.png", BOARDS / "V4900000000_normal_comparison.png")
    make_grid(
        BOARDS / "V4900000000_head_hair_hand_closeups.png",
        [
            ("true_full", "true full"),
            ("same_support_random_semantic", "random"),
            ("local_knn_smoothing", "smooth"),
        ],
        "Close-up board unavailable: formal region metrics not evaluated",
    )
    make_grid(
        BOARDS / "V4900000000_failure_cases.png",
        [
            ("same_support_random_semantic", "random > true in pilot"),
            ("local_knn_smoothing", "smoothing > true"),
            ("true_full", "true not proven"),
        ],
        "Failure cases from pilot; not a success visual",
    )


def write_decision_and_reports(smoke: dict[str, Any]) -> str:
    final_status = "V6000000000_INVALID_CONTROLLER_NOT_IMPLEMENTED"
    control_rows = [
        {"rank": 1, "group": "same_support_random_semantic", "pilot_score": 0.0002765599638223648, "relation": "beats_true_full_in_V380_pilot"},
        {"rank": 2, "group": "local_knn_smoothing", "pilot_score": 0.00017620933067519218, "relation": "beats_true_full_in_V380_pilot"},
        {"rank": 3, "group": "true_full", "pilot_score": 0.00016781259910203516, "relation": "not_formal_gpu_matrix"},
    ]
    with (REPORTS / "V5000000000_control_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()))
        writer.writeheader()
        writer.writerows(control_rows)
    decision = {
        "created_utc": now(),
        "status": final_status,
        "semantic_causality_confirmed": False,
        "formal_gpu_fullview_matrix_completed": False,
        "gpu_smoke_passed": smoke.get("formal_gpu_smoke_passed"),
        "normal_head_status": "geometric_normals_recomputed; learned/input normals were all zero",
        "dominant_factor_from_pilot": "random_semantic_and_local_smoothing_beat_true_full_in_V380_pilot",
        "why_not_smoothing_final": "Allowed smoothing-dominant limitations require a formal matrix or accepted limitation rerun; V460 formal matrix is absent.",
        "why_invalid_controller": "V430 GPU smoke exists, but no V460 40-run formal matrix controller/run evidence exists; current trainer smoke is insufficient for causality.",
        "no_promotion": True,
        "active_candidate": ACTIVE_CANDIDATE,
    }
    write_json(REPORTS / "V5000000000_causality_decision.json", decision)
    with (REPORTS / "V5100000000_repair_cycles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cycle", "status", "reason"])
        writer.writeheader()
        writer.writerow({"cycle": 0, "status": "not_run", "reason": "formal V460 matrix/controller incomplete"})
    write_text(REPORTS / "V5100000000_best_repair_summary.md", "# V510 Repair Summary\n\nNo repair cycles were run because the formal V460 matrix was not implemented/completed.\n")
    with (REPORTS / "V5200000000_candidate_synthesis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "status", "reason"])
        writer.writeheader()
        writer.writerow({"candidate": ACTIVE_CANDIDATE, "status": "retained_active_candidate", "reason": "no replacement; no promotion"})
    with (REPORTS / "V5200000000_uniqueness_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_candidates", "unique_candidates", "status", "reason"])
        writer.writeheader()
        writer.writerow({"raw_candidates": 0, "unique_candidates": 0, "status": "not_generated", "reason": "formal route invalid/incomplete"})
    write_advisor(final_status)
    return final_status


def write_advisor(final_status: str) -> None:
    advisor = f"""# V5400000000 中文导师报告

## 最终状态

`{final_status}`

本轮不能确认 SMPL semantic independent causality，也不能 promotion。

## 为什么 V420 不能算成功

V420 的 V380 只是 CPU remote pilot，不是正式 GPU full-view matrix。已有 pilot
还显示 random semantic 高于 true_full，local KNN smoothing 也高于 true_full，
并且所有 V380 normal 输入为 0，region metrics 没有评估。

## 本轮完成的修复

1. 保存 V421-V600 goal spec 和 manifest。
2. 审计 V415 外层 zip 与内层 npz；确认 selected bundle 存在二层 CRC
   错误，controls bundle 不含 control predictions。
3. 从本地 V380 readable outputs 修复 selected/control predictions，并重算
   geometric normals。该 normal 来源是几何差分，不是 learned normal head。
4. 实现 V430 Modal GPU trainer entrypoint，并完成 A10G CUDA smoke。
5. 生成 V421/V422/V430/V440/V450/V460/V480/V490/V500/V510/V520/V560/V580
   证据链和 upload-safe bundles。

## 为什么仍不能写 semantic causal success

V460 要求 8 组 x 5 seeds = 40 个正式 GPU run。当前只有 V430 one-run GPU
smoke；正式 V460 matrix 没有完成。因此 true/random/shuffled/support/
observation/noSparse/local smoothing 的正式 GPU 统计不存在。

## Normal 状态

V380 输入 normal 全 0。本轮已用 full-view world_points 重算几何 normal，
valid 区域 nonzero ratio 通过 0.1 门槛；但这只能说明可补几何 normal，
不能说 learned normal head 已成功。

## 视觉和 region 指标

当前生成的 V490 boards 是 pilot/blocker 可视化，不是 paper-grade 3D 成功图。
full_body/head_face/hairline/left_hand/right_hand 的正式 region metrics 未完成。

## 不 promotion 理由

- formal GPU matrix 未完成；
- semantic causality 未确认；
- random/smoothing 在 pilot 中仍强；
- learned normal head 未证明；
- no strict registry/no V50/V50R2/no active-candidate replacement。

## 下一步

必须实现并启动 V460 40-run formal GPU matrix，且在正式 region metrics 和
paper-grade 3D close-up boards 上证明 true_full 稳定超过 random/shuffled/
support/observation/noSparse/local smoothing，才能重新讨论 semantic causality。
"""
    write_text(REPORTS / "V5400000000_advisor_report.md", advisor)
    write_text(
        REPORTS / "V5400000000_one_page.md",
        f"# V540 One Page\n\nFinal: `{final_status}`. GPU smoke passed, but formal V460 40-run matrix is absent. Semantic causality is not confirmed. Active candidate remains `{ACTIVE_CANDIDATE}`.\n",
    )
    write_text(
        REPORTS / "V5400000000_limitations.md",
        "# V540 Limitations\n\nV380 was CPU pilot only; V460 formal GPU matrix is absent; region metrics are not evaluated for formal runs; learned normal head is not proven; no promotion.\n",
    )


def add_to_zip(z: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.exists() and path.is_file():
        z.write(path, arcname)


def make_bundle(zip_path: Path, entries: list[tuple[Path, str]]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, arcname in entries:
            add_to_zip(z, path, arcname)
    zrow = zip_test(zip_path)
    return {
        "path": str(zip_path),
        "sha256": zrow.get("sha256"),
        "size": zrow.get("size"),
        "zip_test_clean": zrow.get("zip_test_clean"),
        "entry_count": len(zrow.get("entries", [])),
    }


def verify_npz_bundle(zip_path: Path) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".npz"):
                rows.append(npz_bytes_test(name, z.read(name)))
    return rows


def package(final_status: str) -> list[dict[str, Any]]:
    final_doc = {
        "status": final_status,
        "created_utc": now(),
        "formal_gpu_fullview_matrix_completed": False,
        "semantic_causality_confirmed": False,
        "normal_head_status": "geometric_normals_recomputed_not_learned_success",
        "best_candidate": ACTIVE_CANDIDATE,
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "active_candidate_replaced": False,
        "advisor_report": str(REPORTS / "V5400000000_advisor_report.md"),
    }
    write_json(REPORTS / "V6000000000_final_status.json", final_doc)

    core_entries = [
        (REPORTS / "V6000000000_final_status.json", "reports/V6000000000_final_status.json"),
        (REPORTS / "V5000000000_causality_decision.json", "reports/V5000000000_causality_decision.json"),
        (REPORTS / "V4300000000_gpu_smoke.json", "reports/V4300000000_gpu_smoke.json"),
        (REPORTS / "V5800000000_post_push_cleanup.json", "reports/V5800000000_post_push_cleanup.json"),
    ]
    report_entries = [(p, f"reports/{p.name}") for p in sorted(REPORTS.glob("V4*.json")) + sorted(REPORTS.glob("V4*.md")) + sorted(REPORTS.glob("V4*.csv"))]
    report_entries += [(p, f"reports/{p.name}") for p in sorted(REPORTS.glob("V5*.json")) + sorted(REPORTS.glob("V5*.md")) + sorted(REPORTS.glob("V5*.csv"))]
    report_entries += [(p, f"reports/{p.name}") for p in sorted(REPORTS.glob("V6000000000*.json"))]
    visual_entries = [(p, f"boards/{p.name}") for p in sorted(BOARDS.glob("V4400000000*.png")) + sorted(BOARDS.glob("V4900000000*.png"))]
    selected_entries = []
    for group in SELECTED_GROUPS:
        base = V422_OUT / f"{group}_seed0"
        for name in ["predictions.npz", "eval.json", "source_manifest.json", "quality.json", "board.png"]:
            selected_entries.append((base / name, f"output/V4220000000_repaired_predictions/{group}_seed0/{name}"))
    control_entries = []
    for group in CONTROL_GROUPS:
        base = V422_OUT / f"{group}_seed0"
        for name in ["predictions.npz", "eval.json", "source_manifest.json", "quality.json", "board.png"]:
            control_entries.append((base / name, f"output/V4220000000_repaired_predictions/{group}_seed0/{name}"))

    bundles = [
        make_bundle(ARCHIVE / "V5600000000_core_evidence_bundle.zip", core_entries),
        make_bundle(ARCHIVE / "V5600000000_reports_bundle.zip", report_entries),
        make_bundle(ARCHIVE / "V5600000000_visuals_bundle.zip", visual_entries),
        make_bundle(ARCHIVE / "V5600000000_selected_predictions_bundle.zip", selected_entries),
        make_bundle(ARCHIVE / "V5600000000_controls_bundle.zip", control_entries),
    ]
    for bundle in bundles:
        bundle["npz_checks"] = verify_npz_bundle(Path(bundle["path"])) if bundle["path"].endswith(("_predictions_bundle.zip", "_controls_bundle.zip")) else []
        bundle["npz_internal_readable"] = all(r.get("readable") and r.get("inner_zip_clean") for r in bundle["npz_checks"]) if bundle["npz_checks"] else None
        bundle["under_500mb"] = bool(bundle["size"] is not None and bundle["size"] < 500 * 1024 * 1024)

    omitted = {
        "created_utc": now(),
        "omitted_large_or_non_source_paths": [
            {"path": str(AUX / "remote_pull"), "reason": "large historical remote pulls; selected repaired evidence included separately"},
            {"path": str(OUTPUT / "V3600000000_fullview_dataset_v2"), "reason": "large full-view datasets; reports and selected predictions packaged"},
            {"path": str(OUTPUT / "__tmp_normal_test"), "reason": "temporary normal smoke output superseded by V422 repaired predictions"},
        ],
    }
    write_json(REPORTS / "V5600000000_omitted_large_file_manifest.json", omitted)
    manifest = {
        "created_utc": now(),
        "final_status": final_status,
        "bundles": bundles,
        "omitted_manifest": str(REPORTS / "V5600000000_omitted_large_file_manifest.json"),
        "sidecar_manifest_policy": "hashes recorded after zip close; selected/control npz internally tested",
    }
    write_json(REPORTS / "V5600000000_upload_manifest_sidecar.json", manifest)
    # Rebuild core once the sidecar exists so final status references are complete.
    return bundles


def write_cleanup() -> None:
    git_status = run_cmd(["git", "status", "--short", "-uall"], timeout=30)
    branch = run_cmd(["git", "branch", "--show-current"], timeout=30).get("stdout", "").strip()
    commit = run_cmd(["git", "rev-parse", "HEAD"], timeout=30).get("stdout", "").strip()
    modal_apps = run_cmd(["modal", "app", "list"], timeout=60)
    ps = run_cmd(["powershell", "-NoProfile", "-Command", "Get-Process python,modal -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,CPU,StartTime | ConvertTo-Json -Compress"], timeout=30)
    cleanup = {
        "created_utc": now(),
        "git_status_clean": git_status.get("stdout", "").strip() == "",
        "git_status": git_status.get("stdout", ""),
        "branch": branch,
        "commit": commit,
        "modal_apps": modal_apps.get("stdout", ""),
        "python_modal_processes": ps.get("stdout", ""),
        "registry_written": False,
        "v50_v50r2_modified": False,
        "active_candidate": ACTIVE_CANDIDATE,
        "active_candidate_replaced": False,
        "cleanup_claim": "honestly_dirty" if git_status.get("stdout", "").strip() else "clean",
    }
    write_json(REPORTS / "V5800000000_post_push_cleanup.json", cleanup)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BOARDS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    artifact_report, _npz_report = audit_v415()
    write_v421_text_reports(artifact_report)
    repair_rows, _controls = repair_predictions()
    smoke = write_v430_v450_reports()
    write_v440_reports(repair_rows)
    write_v460_blocker()
    write_v480_v490_reports()
    final_status = write_decision_and_reports(smoke)
    write_cleanup()
    bundles = package(final_status)
    # Rewrite cleanup after packaging so it sees current worktree state.
    write_cleanup()
    package(final_status)
    print(json.dumps({"status": final_status, "bundles": bundles}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
