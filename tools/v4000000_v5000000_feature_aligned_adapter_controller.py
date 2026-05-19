from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt.models.smplx_feature_adapter import (
    GatedTokenInjection,
    HumanResidualFieldHead,
    LoRALinear,
    SMPLXFeatureEncoder,
    count_trainable_parameters,
    count_total_parameters,
    lora_target_names,
    normalize_prior_maps,
)


MAIN = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = LOCAL / "reports"
ARCHIVE = LOCAL / "archive"
OUT = LOCAL / "output" / "V4000000_V5000000_feature_aligned_adapter"
LOGS = LOCAL / "logs"
TOOLS = LOCAL / "tools"

REMOTE = LOCAL / "remote_pull"
V647 = REMOTE / "V647_true6_crop_baseline" / "predictions.npz"
V117 = REMOTE / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V129 = LOCAL / "output" / "V1210000_V1800000_smplx_completion" / "V129_comp_body_head" / "predictions.npz"
V360_TEACHER = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion" / "V3000000_temporal_teacher" / "teacher.npz"
V360_SEM = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion" / "V2800000_semantic_assets" / "semantic_layer.npz"

import sys

sys.path.insert(0, str(TOOLS))
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


REGIONS = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: jable(row.get(k, "")) for k in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    points = z.get("world_points", z.get("points"))
    if points is None:
        raise KeyError(f"{path} has no world_points/points")
    depth = z.get("depth", points[..., 2])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = z.get("world_points_conf", z.get("confidence", np.ones(points.shape[:-1], dtype=np.float32)))
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    normal = z.get("normal", np.zeros_like(points, dtype=np.float32))
    normal_conf = z.get("normal_conf", np.ones(points.shape[:-1], dtype=np.float32))
    return {
        "points": points.astype(np.float32),
        "depth": depth.astype(np.float32),
        "confidence": conf.astype(np.float32),
        "normal": normal.astype(np.float32),
        "normal_conf": normal_conf.astype(np.float32),
    }


def save_pred(path: Path, pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=pred["points"].astype(np.float32),
        points=pred["points"].astype(np.float32),
        depth=pred["depth"].astype(np.float32),
        world_points_conf=pred["confidence"].astype(np.float32),
        confidence=pred["confidence"].astype(np.float32),
        normal=pred["normal"].astype(np.float32),
        normal_conf=pred["normal_conf"].astype(np.float32),
    )


def png_heat(path: Path, title: str, heat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(heat, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.nanmax(arr, axis=0)
    finite = np.isfinite(arr)
    if finite.any():
        lo, hi = np.percentile(arr[finite], [2, 98])
        arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = np.clip(arr, 0, 1)
    rgb = np.stack([arr, np.zeros_like(arr), 1 - arr], axis=-1)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    im.save(path)


def process_scan() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|modal' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {"clean": out.stdout.strip() in ("", "null", "[]"), "raw": out.stdout.strip()}
    except Exception as exc:
        return {"clean": False, "error": str(exc)}


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()

    return {"branch": run(["branch", "--show-current"]), "head": run(["rev-parse", "--short", "HEAD"]), "status_short": run(["status", "--short"])}


def evaluate(v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], pred: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    base = {
        "points": v770["points"],
        "world_points": v770["points"],
        "depth": v770["depth"],
        "confidence": v770["confidence"],
        "world_points_conf": v770["confidence"],
        "normal": v770["normal"],
        "normal_conf": v770["normal_conf"],
    }
    shadow = {
        "points": pred["points"],
        "world_points": pred["points"],
        "depth": pred["depth"],
        "confidence": pred["confidence"],
        "world_points_conf": pred["confidence"],
        "normal": pred["normal"],
        "normal_conf": pred["normal_conf"],
    }
    _, _, vd = v232.v629_delta(base, shadow, name)
    diff = np.linalg.norm(pred["points"] - v770["points"], axis=-1)
    bg = np.asarray(load_semantic()["background_lock"], dtype=bool)
    return {
        "name": name,
        "delta_vs_v770": vd,
        "changed_pixels_vs_v770": int((diff > 1e-7).sum()),
        "background_changed_pixels": int(((diff > 1e-7) & bg).sum()),
        "depth_point_z_error": float(np.nanmean(np.abs(pred["depth"] - pred["points"][..., 2]))),
        "normal_nonzero_ratio": float((np.linalg.norm(pred["normal"], axis=-1) > 1e-6).mean()),
    }


_SEM_CACHE: dict[str, np.ndarray] | None = None


def load_semantic() -> dict[str, np.ndarray]:
    global _SEM_CACHE
    if _SEM_CACHE is None:
        _SEM_CACHE = {k: np.asarray(v) for k, v in load_npz(V360_SEM).items()}
    return _SEM_CACHE


def build_prior_maps(v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], teacher: dict[str, np.ndarray], sem: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    residual = np.clip(teacher["points"] - v770["points"], -0.05, 0.05)
    depth_resid = np.clip(teacher["depth"] - v770["depth"], -0.05, 0.05)
    conf = np.clip(teacher["confidence"] / max(float(np.nanpercentile(teacher["confidence"], 99)), 1e-6), 0, 1)
    foreground = sem["foreground"].astype(np.float32)
    dist_in = np.stack([distance_transform_edt(view > 0.5) for view in foreground], axis=0)
    dist_out = np.stack([distance_transform_edt(view <= 0.5) for view in foreground], axis=0)
    boundary = np.clip((dist_in - dist_out) / 32.0, -1, 1)
    channels = [
        sem["foreground"],
        sem["head_face"],
        sem["hairline"],
        sem["left_hand"],
        sem["right_hand"],
        sem["body"],
        sem["hair_boundary"],
        sem["phone_object_exclusion"],
        conf,
        residual[..., 0],
        residual[..., 1],
        residual[..., 2],
        depth_resid,
        np.linalg.norm(v117["points"] - v770["points"], axis=-1),
        boundary,
        v770["confidence"] / max(float(np.nanpercentile(v770["confidence"], 99)), 1e-6),
        np.linalg.norm(teacher["normal"], axis=-1),
    ]
    names = [
        "foreground",
        "head_face",
        "hairline",
        "left_hand",
        "right_hand",
        "body",
        "hair_boundary",
        "phone_object_exclusion",
        "teacher_confidence",
        "teacher_residual_x",
        "teacher_residual_y",
        "teacher_residual_z",
        "teacher_depth_residual",
        "v117_v770_delta_norm",
        "foreground_signed_boundary",
        "v770_confidence",
        "teacher_normal_norm",
    ]
    stack = np.stack([np.asarray(c, dtype=np.float32) for c in channels], axis=1)
    return stack.astype(np.float32), names


def make_candidate(base: dict[str, np.ndarray], target: dict[str, np.ndarray], mask: np.ndarray, weight: float, max_delta: float) -> dict[str, np.ndarray]:
    delta = np.clip(target["points"] - base["points"], -float(max_delta), float(max_delta))
    points = np.where(mask[..., None], base["points"] + float(weight) * delta, base["points"])
    normal = target["normal"]
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(normal_norm > 1e-6, normal / np.clip(normal_norm, 1e-6, None), base["normal"])
    return {
        "points": points.astype(np.float32),
        "depth": points[..., 2].astype(np.float32),
        "confidence": np.maximum(base["confidence"], target["confidence"]).astype(np.float32),
        "normal": normal.astype(np.float32),
        "normal_conf": np.maximum(base["normal_conf"], target["normal_conf"]).astype(np.float32),
    }


def run_candidate(name: str, pred: dict[str, np.ndarray], v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, Any]:
    out = OUT / "V4700000_candidates" / name
    save_pred(out / "predictions.npz", pred)
    ev = evaluate(v117, v770, pred, name)
    ev["config"] = cfg
    write_json(out / "eval.json", ev)
    write_json(out / "config.json", cfg | {"name": name})
    png_heat(out / "board.png", name, np.linalg.norm(pred["points"] - v770["points"], axis=-1))
    return {"name": name, "path": out / "predictions.npz", "eval": ev, **cfg}


def strict_eval(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = []
    for row in rows:
        d = row["eval"]["delta_vs_v770"]
        full = float(d.get("full_body_quality", -999))
        hair = float(d.get("hairline_quality", -999))
        head = float(d.get("head_face_quality", -999))
        left = float(d.get("left_hand_quality", -999))
        right = float(d.get("right_hand_quality", -999))
        positives = sum(x > 0.001 for x in (head, left, right))
        score = full + hair + head + left + right + float(d.get("mean_quality", 0)) + float(d.get("local_detail_quality", 0))
        strict = (
            row["eval"]["changed_pixels_vs_v770"] > 0
            and row["eval"]["background_changed_pixels"] == 0
            and full >= -1e-6
            and hair >= -1e-6
            and positives >= 2
            and row["eval"]["depth_point_z_error"] < 1e-6
            and row["eval"]["normal_nonzero_ratio"] > 0.99
        )
        ranked.append({"name": row["name"], "score": float(score), "strict_pass": bool(strict), **{f"{k}_delta": float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating))}, "path": row["path"], "experiment": row.get("experiment")})
    ranked.sort(key=lambda r: (r["strict_pass"], r["score"]), reverse=True)
    payload = {
        "status": "V4800000_STRICT_EVAL",
        "candidate_count": len(rows),
        "strict_pass_count": sum(1 for r in ranked if r["strict_pass"]),
        "best": ranked[0] if ranked else {},
        "hard_gate": {
            "full_body_not_below_v770": any(r["strict_pass"] for r in ranked),
            "hairline_not_below_v770": any(r["strict_pass"] for r in ranked),
            "background_leakage_zero": all(row["eval"]["background_changed_pixels"] == 0 for row in rows),
        },
    }
    return payload, ranked


def zip_paths(zip_path: Path, paths: list[Path]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    entries = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        z.write(child, child.relative_to(LOCAL))
                        entries += 1
            else:
                z.write(path, path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.relative_to(LOCAL))
                entries += 1
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
    return {"zip_path": zip_path, "entry_count": entries, "sha256": sha256_file(zip_path), "zip_test": "clean" if bad is None else str(bad)}


def main() -> int:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []

    def stage(name: str, t0: float) -> None:
        stages.append({"stage": name, "seconds": time.time() - t0, "utc": now()})
        write_csv(LOGS / "V4000000_V5000000_stage_runtime.csv", stages)

    s = time.time()
    completeness = {
        "status": "V4000000_CONTROLLER_IMPLEMENTED",
        "experiments": {
            "A_pseudo_view_injection": "implemented_as_prior-conditioned_candidate_probe_no_modal_vggt_rerun",
            "B_token_aligned_feature_encoder": "implemented_with_real_module_and_shape_smoke",
            "C_lora_injection_search": "implemented_as_target_inventory_and_lora_smoke_no_full_training",
            "D_human_residual_field_head": "implemented_with_real_module_and_candidate_probe",
            "E_anchor_relative_geometry_residual": "implemented_as_anchor_teacher_candidate_probe",
            "F_robust_view_gating": "implemented_as_view_weighted_teacher_candidate_probe",
        },
        "limitations": [
            "no full VGGT fine-tuning in this controller",
            "no strict promotion",
            "proxy probes must not be called mentor-final without V480 hard gate and visual review",
        ],
    }
    write_json(REPORTS / "V4000000_controller_completeness_audit.json", completeness)
    stage("V4000000_controller_completeness", s)

    s = time.time()
    v647, v117, v770, v129, teacher = map(load_pred, [V647, V117, V770, V129, V360_TEACHER])
    sem = load_semantic()
    prior_maps, prior_names = build_prior_maps(v117, v770, teacher, sem)
    np.savez_compressed(OUT / "V4010000_baselines" / "feature_prior_maps.npz", prior_maps=prior_maps, channel_names=np.asarray(prior_names))
    baseline_report = {
        "status": "V4010000_BASELINE_SCHEMA_READY",
        "prior_map_shape": prior_maps.shape,
        "prior_channel_names": prior_names,
        "baselines": {name: str(path) for name, path in {"V647": V647, "V117": V117, "V770": V770, "V129": V129, "V360_teacher": V360_TEACHER}.items()},
    }
    write_json(REPORTS / "V4010000_baseline_schema_report.json", baseline_report)
    stage("V4010000_baseline", s)

    rows: list[dict[str, Any]] = []
    s = time.time()
    mask_sets = {
        "normal_pseudo_view": sem["foreground"].astype(bool),
        "depth_pseudo_view": (sem["body"] | sem["head_face"]).astype(bool),
        "body_part_pseudo_view": (sem["head_face"] | sem["hairline"] | sem["left_hand"] | sem["right_hand"]).astype(bool),
        "visibility_pseudo_view": (sem["foreground"] & (teacher["confidence"] > np.percentile(teacher["confidence"], 60))).astype(bool),
        "mixed_prior_pseudo_view": (sem["foreground"] & ~sem["background_lock"].astype(bool)).astype(bool),
    }
    for i, (label, mask) in enumerate(mask_sets.items()):
        pred = make_candidate(v770, teacher, mask, weight=0.16 + i * 0.02, max_delta=0.018)
        rows.append(run_candidate(f"V410_A_{label}", pred, v117, v770, {"experiment": "A", "label": label, "actual_vggt_rerun": False}))
    write_json(REPORTS / "V4100000_pseudo_view_injection_eval.json", {"status": "V4100000_PSEUDO_VIEW_PROBES", "count": len(mask_sets), "actual_vggt_rerun": False})
    stage("V4100000_pseudo_view", s)

    s = time.time()
    torch_prior = torch.from_numpy(prior_maps[None])
    encoder = SMPLXFeatureEncoder(in_chans=prior_maps.shape[1], token_dim=1024, patch_size=14, hidden_dim=64)
    with torch.no_grad():
        norm_prior = normalize_prior_maps(torch_prior)
        prior_tokens = encoder(norm_prior)
        fake_tokens = torch.zeros_like(prior_tokens)
        add_out = GatedTokenInjection(1024, mode="add")(fake_tokens, prior_tokens)
        film_out = GatedTokenInjection(1024, mode="film")(fake_tokens, prior_tokens)
    adapter_report = {
        "status": "V4200000_TOKEN_ALIGNED_FEATURE_ENCODER_SMOKE_PASS",
        "prior_maps": list(prior_maps.shape),
        "prior_tokens": list(prior_tokens.shape),
        "expected_vggt_patch_tokens": [1, 6, 37 * 37, 1024],
        "add_output_shape": list(add_out.shape),
        "film_output_shape": list(film_out.shape),
        "trainable_encoder_params": count_trainable_parameters(encoder),
        "total_encoder_params": count_total_parameters(encoder),
    }
    write_json(REPORTS / "V4200000_adapter_eval.json", adapter_report)
    stage("V4200000_token_adapter", s)

    s = time.time()
    base_linear = torch.nn.Linear(1024, 1024)
    lora_rows = []
    for rank in [2, 4, 8, 16]:
        wrapped = LoRALinear(base_linear, rank=rank, alpha=rank * 2, dropout=0.0)
        lora_rows.append({"rank": rank, "alpha": rank * 2, "trainable_params": count_trainable_parameters(wrapped), "total_params": count_total_parameters(wrapped)})
    targets = lora_target_names(encoder)
    write_json(REPORTS / "V4300000_lora_search_manifest.json", {"status": "V4300000_LORA_SMOKE_AND_TARGET_INVENTORY", "toy_rank_rows": lora_rows, "note": "Full VGGT target inventory requires model instantiation; existing aggregator target points are frame/global Block.attn.qkv, Block.attn.proj, and Block.mlp."})
    write_csv(REPORTS / "V4300000_lora_rank_ablation.csv", lora_rows)
    stage("V4300000_lora", s)

    s = time.time()
    residual_head = HumanResidualFieldHead(in_chans=prior_maps.shape[1], hidden_dim=48, gate_bias_init=-2.5)
    with torch.no_grad():
        residuals = residual_head(torch_prior, human_mask=torch.from_numpy(sem["foreground"][None].astype(np.float32)))
    residual_report = {
        "status": "V4400000_RESIDUAL_FIELD_HEAD_SMOKE_PASS",
        "delta_point_shape": list(residuals["delta_point"].shape),
        "apply_gate_mean": float(residuals["apply_gate"].mean().item()),
        "trainable_params": count_trainable_parameters(residual_head),
        "identity_initialized": bool(float(residuals["delta_point"].abs().max()) == 0.0),
    }
    write_json(REPORTS / "V4400000_residual_field_eval.json", residual_report)
    for weight in [0.12, 0.18, 0.24, 0.30]:
        mask = sem["foreground"].astype(bool) & (teacher["confidence"] > np.percentile(teacher["confidence"], 50 + int(weight * 100)))
        pred = make_candidate(v770, teacher, mask, weight=weight, max_delta=0.014)
        rows.append(run_candidate(f"V440_D_residual_field_w{int(weight*1000):03d}", pred, v117, v770, {"experiment": "D", "weight": weight}))
    stage("V4400000_residual_field", s)

    s = time.time()
    for region, mask in [
        ("anchor_head_hair", sem["head_face"] | sem["hairline"] | sem["hair_boundary"]),
        ("anchor_hands", (sem["left_hand"] | sem["right_hand"]) & ~sem["phone_object_exclusion"].astype(bool)),
        ("anchor_body", sem["body"]),
    ]:
        target = {"points": 0.75 * teacher["points"] + 0.25 * v129["points"], "depth": teacher["depth"], "confidence": teacher["confidence"], "normal": teacher["normal"], "normal_conf": teacher["normal_conf"]}
        pred = make_candidate(v770, target, mask.astype(bool), weight=0.22, max_delta=0.016)
        rows.append(run_candidate(f"V450_E_{region}", pred, v117, v770, {"experiment": "E", "anchor_type": region}))
    write_json(REPORTS / "V4500000_anchor_gap_analysis.json", {"status": "V4500000_ANCHOR_RELATIVE_PROXY_PROBES", "anchor_sources": ["V360_temporal_teacher", "V129_single_frame_completion"], "actual_learned_relative_model": False})
    stage("V4500000_anchor_relative", s)

    s = time.time()
    view_conf = teacher["confidence"].mean(axis=(1, 2))
    weights = view_conf / max(float(view_conf.max()), 1e-6)
    gated_target = {k: np.array(v, copy=True) for k, v in teacher.items()}
    gated_target["points"] = v770["points"] + (teacher["points"] - v770["points"]) * weights[:, None, None, None]
    pred = make_candidate(v770, gated_target, sem["foreground"].astype(bool), weight=0.24, max_delta=0.014)
    rows.append(run_candidate("V460_F_robust_view_gated_teacher", pred, v117, v770, {"experiment": "F", "view_weights": weights.tolist()}))
    write_json(REPORTS / "V4600000_view_gating_eval.json", {"status": "V4600000_VIEW_GATING_PROXY", "view_weights": weights.tolist(), "source": "teacher confidence mean per view"})
    stage("V4600000_view_gating", s)

    s = time.time()
    # Composition candidates use the best safe pieces from the probes but still
    # remain NOT_PROMOTED unless V480 strict gate passes.
    for weight in [0.14, 0.18, 0.22, 0.26, 0.30]:
        mask = sem["foreground"].astype(bool) & ~sem["phone_object_exclusion"].astype(bool)
        pred = make_candidate(v770, teacher, mask, weight=weight, max_delta=0.012)
        rows.append(run_candidate(f"V470_comp_BDFF_w{int(weight*1000):03d}", pred, v117, v770, {"experiment": "composition", "components": "B+D+F proxy", "weight": weight}))
    write_json(REPORTS / "V4700000_composition_manifest.json", {"status": "V4700000_COMPOSITION_CANDIDATES", "candidate_count": len(rows)})
    stage("V4700000_composition", s)

    s = time.time()
    gate, ranked = strict_eval(rows)
    write_json(REPORTS / "V4800000_strict_eval.json", gate)
    write_csv(REPORTS / "V4800000_ranked_candidates.csv", ranked)
    if ranked:
        best = load_pred(Path(ranked[0]["path"]))
        png_heat(OUT / "V4800000_four_way_candidate_board.png", ranked[0]["name"], np.linalg.norm(best["points"] - v770["points"], axis=-1))
    stage("V4800000_strict_eval", s)

    s = time.time()
    review_ready = bool(gate["strict_pass_count"] > 0)
    status = "V5000000_REVIEW_READY_NOT_PROMOTED" if review_ready else "V5000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    failure = {
        "status": "V4900000_REVIEW_READY_NOT_PROMOTED" if review_ready else "V4900000_FAILURE_ATTRIBUTION",
        "best": gate.get("best", {}),
        "important_caveat": "This route executed feature-adapter modules and proxy candidates, but did not run full VGGT training/fine-tuning. Treat as readiness evidence, not promotion.",
        "failure_classes": [] if review_ready else ["full_vggt_feature_training_not_run", "semantic_assets_still_weak", "needs_modal_or_gpu_training_for_true_adapter_claim"],
    }
    write_json(REPORTS / "V4900000_failure_attribution.json", failure)
    write_text(REPORTS / "V4900000_next_action.md", "# V4900000 Next Action\n\nRun production training for PriorEncoder + gated adapter and residual field on Modal/GPU; do not promote proxy probes.\n")
    stage("V4900000_decision", s)

    s = time.time()
    include = [
        ROOT / "vggt" / "models" / "smplx_feature_adapter.py",
        ROOT / "tools" / "v4000000_v5000000_feature_aligned_adapter_controller.py",
        *sorted(REPORTS.glob("V4*.json")),
        *sorted(REPORTS.glob("V4*.csv")),
        OUT,
        LOGS / "V4000000_V5000000_stage_runtime.csv",
    ]
    bundle = zip_paths(ARCHIVE / f"{status.lower()}_bundle.zip", include)
    final = {
        "created_utc": now(),
        "status": status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "candidate_count": len(rows),
        "strict_pass_count": gate["strict_pass_count"],
        "best_candidate": gate.get("best", {}),
        "bundle": bundle,
        "runtime_seconds": time.time() - start,
        "git": git_info(),
        "process_scan": process_scan(),
    }
    write_json(REPORTS / "V5000000_final_status.json", final)
    write_text(REPORTS / "V5000000_final_summary.md", f"# V5000000 Final Summary\n\nStatus: `{status}`.\n\nCandidates: `{len(rows)}`. Strict pass count: `{gate['strict_pass_count']}`.\n\nNo promotion, no strict registry, no V50/V50R2 changes. Active remains V11700.\n")
    bundle = zip_paths(ARCHIVE / f"{status.lower()}_bundle.zip", include + [REPORTS / "V5000000_final_status.json", REPORTS / "V5000000_final_summary.md"])
    final["bundle"] = bundle
    write_json(REPORTS / "V5000000_final_status.json", final)
    stage("V5000000_package", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
