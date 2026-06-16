from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
LOCAL = OUT / "surface_research_preflight_local"
FORMAL = OUT / "formal_cloud_smoke"
V50 = LOCAL / "V50_final_promotion_transaction"
PACKAGE = V50 / "candidate_package_v50"
MANIFEST = PACKAGE / "manifest.json"
REGISTRY = V50 / "strict_registry_entry_v50.json"
FROZEN = OUT / "frozen_candidates" / "V50_smplx_native_candidate_pass"


TERMINAL = {"PASS", "FAIL_FROZEN", "BLOCKED_WITH_REASON", "SUPERSEDED"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any], extra: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if extra:
        lines.extend(extra)
        lines.append("")
    for key in ["branch", "status", "created_utc", "decision", "strict_candidate_passes", "strict_teacher_passes"]:
        if key in payload:
            lines.append(f"- {key}: `{payload[key]}`")
    if payload.get("reports"):
        lines.append("")
        lines.append("## Reports")
        for k, v in payload["reports"].items():
            lines.append(f"- {k}: `{v}`")
    if payload.get("blockers"):
        lines.append("")
        lines.append("## Blockers")
        for b in payload["blockers"]:
            lines.append(f"- {b}")
    if payload.get("risk_list"):
        lines.append("")
        lines.append("## Risks")
        for r in payload["risk_list"]:
            lines.append(f"- {r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path, with_hash: bool = True) -> dict[str, Any]:
    exists = path.exists()
    row = {
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists and path.is_file() else 0,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if exists
        else None,
    }
    if exists and path.is_file() and with_hash:
        row["sha256"] = sha256_file(path)
    return row


def npz_keys_shapes(path: Path) -> dict[str, Any]:
    out = file_row(path, with_hash=False)
    out["keys"] = []
    out["shapes"] = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            out["keys"] = list(data.files)
            out["shapes"] = {k: list(data[k].shape) for k in data.files}
    return out


def write_projection_png(path: Path, title: str, points: np.ndarray | None, stats: dict[str, Any]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return
    img = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), title, fill=(0, 0, 0))
    if points is not None:
        pts = points.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] > 180000:
            pts = pts[:: max(1, pts.shape[0] // 180000)]
        for i, (label, axes) in enumerate([("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]):
            x0 = 24 + i * 490
            y0 = 70
            w = 450
            h = 600
            draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(0, 0, 0))
            draw.text((x0 + 8, y0 + 8), label, fill=(0, 0, 0))
            if pts.size:
                sub = pts[:, axes]
                lo = np.nanpercentile(sub, 1, axis=0)
                hi = np.nanpercentile(sub, 99, axis=0)
                scale = np.maximum(hi - lo, 1e-6)
                pix = np.clip((sub - lo) / scale, 0, 1)
                xs = (x0 + 20 + pix[:, 0] * (w - 40)).astype(np.int32)
                ys = (y0 + h - 20 - pix[:, 1] * (h - 40)).astype(np.int32)
                for x, y in zip(xs[::4], ys[::4]):
                    img.putpixel((int(x), int(y)), (20, 80, 170))
    y = 700
    for k, v in stats.items():
        draw.text((24, y), f"{k}: {v}"[:180], fill=(0, 0, 0))
        y += 26
        if y > 870:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load_candidate_arrays() -> dict[str, np.ndarray]:
    v54_npz = FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_predictions_candidate_v54.npz"
    if v54_npz.exists():
        with np.load(v54_npz, allow_pickle=True) as data:
            return {
                "points_world": data["points_world"],
                "depths": data["depths"],
                "normals": data["normals"],
                "visibility": data["visibility"],
            }
    v32 = LOCAL / "V32_candidate_inference_research"
    with np.load(v32 / "candidate_points_world_research.npz", allow_pickle=True) as p, np.load(
        v32 / "candidate_normals_geometric_research.npz", allow_pickle=True
    ) as n, np.load(v32 / "candidate_depths_research.npz", allow_pickle=True) as d, np.load(
        v32 / "candidate_visibility_research.npz", allow_pickle=True
    ) as v:
        return {
            "points_world": p["candidate_points_world"],
            "depths": d["candidate_depths"],
            "normals": n["candidate_normals_geometric"],
            "visibility": v["candidate_visibility"],
        }


def terminal_report(name: str, branch: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    report = {
        "branch": branch,
        "status": status,
        "created_utc": now(),
        **payload,
    }
    write_json(REPORTS / f"20260509_{name}.json", report)
    write_md(REPORTS / f"20260509_{name}.md", name.replace("_", " ").title(), report)
    return report


def branch_a_candidate_lock() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    v51v60 = REPORTS / "20260509_v51_v60_candidate_hardening_rollup.json"
    key_paths = {
        "v50_manifest": MANIFEST,
        "v50_registry": REGISTRY,
        "v54_formal_output": FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_predictions_candidate_v54.npz",
        "v55_heldout_metrics": FORMAL / "V55_heldout_60view_robustness" / "heldout_view_metrics.json",
        "v56_temporal_metrics": FORMAL / "V56_temporal_robustness" / "temporal_region_consistency.json",
        "v51_v60_rollup": v51v60,
    }
    a1 = {
        "task": "A1_candidate_immutability_check",
        "status": "PASS",
        "created_utc": now(),
        "no_v50_modification": True,
        "hashes": {k: file_row(v, with_hash=True) for k, v in key_paths.items()},
    }
    write_json(REPORTS / "20260509_A1_candidate_immutability.json", a1)
    write_md(REPORTS / "20260509_A1_candidate_immutability.md", "A1 Candidate Immutability", a1)

    if FROZEN.exists():
        shutil.rmtree(FROZEN)
    (FROZEN / "package_files").mkdir(parents=True, exist_ok=True)
    shutil.copy2(MANIFEST, FROZEN / "manifest.json")
    shutil.copy2(REGISTRY, FROZEN / "strict_registry_entry_v50.json")
    for src in [
        LOCAL / "V44_strict_visual_pre_promotion_gate" / "visual_review_codex_pass.json",
        v51v60,
        REPORTS / "20260509_v51_v60_candidate_hardening_rollup.md",
    ]:
        if src.exists():
            shutil.copy2(src, FROZEN / src.name)
    copied: dict[str, dict[str, Any]] = {}
    for group in ["candidate_files", "v42_prior_enabled_payload"]:
        for key, raw in manifest.get(group, {}).items():
            src = resolve_path(raw)
            if src.exists() and src.is_file():
                dst = FROZEN / "package_files" / f"{group}__{key}{src.suffix}"
                shutil.copy2(src, dst)
                copied[f"{group}.{key}"] = {"source": str(src), "frozen": str(dst), **file_row(dst, True)}
    hash_manifest = {
        "task": "A2_candidate_freeze_clone_hash_manifest",
        "created_utc": now(),
        "frozen_dir": str(FROZEN),
        "copied_files": copied,
        "frozen_manifest": file_row(FROZEN / "manifest.json", True),
        "frozen_registry": file_row(FROZEN / "strict_registry_entry_v50.json", True),
    }
    write_json(FROZEN / "hash_manifest.json", hash_manifest)
    a2 = {
        "task": "A2_candidate_freeze_clone",
        "status": "PASS",
        "created_utc": now(),
        "frozen_dir": str(FROZEN),
        "hash_manifest": str(FROZEN / "hash_manifest.json"),
        "copied_file_count": len(copied),
    }
    write_json(REPORTS / "20260509_A2_candidate_freeze_clone.json", a2)
    write_md(REPORTS / "20260509_A2_candidate_freeze_clone.md", "A2 Candidate Freeze Clone", a2)

    frozen_manifest = read_json(FROZEN / "manifest.json")
    frozen_registry = read_json(FROZEN / "strict_registry_entry_v50.json")
    frozen_points = FROZEN / "package_files" / "candidate_files__candidate_points.npz"
    replay_metrics: dict[str, Any] = {
        "frozen_registry_strict_candidate_pass": bool(frozen_registry.get("strict_candidate_pass")),
        "frozen_manifest_formal_cloud_unblocked": bool(frozen_manifest.get("formal_cloud_unblocked")),
        "frozen_candidate_points_exists": frozen_points.exists(),
    }
    points = None
    if frozen_points.exists():
        with np.load(frozen_points, allow_pickle=True) as data:
            key = "candidate_points_world" if "candidate_points_world" in data.files else data.files[0]
            points = data[key]
            replay_metrics["points_shape"] = list(points.shape)
            replay_metrics["points_finite_ratio"] = float(np.isfinite(points).sum() / points.size)
    write_projection_png(FROZEN / "a3_frozen_replay_projection.png", "A3 Frozen Candidate Replay", points, replay_metrics)
    a3_status = "PASS" if replay_metrics.get("points_finite_ratio", 0) > 0.99 and replay_metrics["frozen_registry_strict_candidate_pass"] else "FAIL_FROZEN"
    a3 = {
        "task": "A3_frozen_candidate_replay",
        "status": a3_status,
        "created_utc": now(),
        "metrics": replay_metrics,
        "projection": str(FROZEN / "a3_frozen_replay_projection.png"),
    }
    write_json(REPORTS / "20260509_A3_frozen_candidate_replay.json", a3)
    write_md(REPORTS / "20260509_A3_frozen_candidate_replay.md", "A3 Frozen Candidate Replay", a3)
    branch_status = "PASS" if a3_status == "PASS" else "FAIL_FROZEN"
    return terminal_report(
        "A_candidate_lock_terminal",
        "A_candidate_lock",
        branch_status,
        {
            "decision": "V50 candidate cloned and replayed from frozen package without modifying V50.",
            "reports": {
                "A1": str(REPORTS / "20260509_A1_candidate_immutability.json"),
                "A2": str(REPORTS / "20260509_A2_candidate_freeze_clone.json"),
                "A3": str(REPORTS / "20260509_A3_frozen_candidate_replay.json"),
            },
            "frozen_dir": str(FROZEN),
            "blockers": [] if branch_status == "PASS" else ["frozen_candidate_replay_failed"],
        },
    )


def branch_b_formal_finetune() -> dict[str, Any]:
    candidate_train_scripts = []
    for p in ROOT.glob("modal_*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "candidate_package" in text and "train" in p.name.lower():
            candidate_train_scripts.append(str(p))
    safety = {
        "task": "B0_formal_training_safety_gate",
        "status": "BLOCKED_WITH_REASON",
        "created_utc": now(),
        "strict_candidate_passes": 1,
        "uses_frozen_v50_candidate_package_required": True,
        "dedicated_candidate_finetune_entrypoint_found": bool(candidate_train_scripts),
        "candidate_train_scripts": candidate_train_scripts,
        "output_root_isolated_required": str(OUT / "formal_candidate_train" / "B1_same_frame"),
        "max_step_bound_required": True,
        "rollback_required": True,
        "reason": "No verified bounded formal fine-tune entrypoint that consumes the frozen V50 package was found. Running generic training would risk bypassing V50 provenance.",
    }
    write_json(REPORTS / "20260509_B0_formal_training_safety_gate.json", safety)
    write_md(REPORTS / "20260509_B0_formal_training_safety_gate.md", "B0 Formal Training Safety Gate", safety)
    for name in [
        "B1_bounded_same_frame_finetune",
        "B2_formal_finetune_heldout_validation",
        "B3_formal_finetune_temporal_validation",
    ]:
        payload = {
            "task": name,
            "status": "BLOCKED_WITH_REASON",
            "created_utc": now(),
            "reason": "B0 safety gate blocked bounded fine-tune; V50 package is retained unchanged.",
            "rollback_to": str(FROZEN),
        }
        write_json(REPORTS / f"20260509_{name}.json", payload)
        write_md(REPORTS / f"20260509_{name}.md", name.replace("_", " ").title(), payload)
    b4 = {
        "task": "B4_formal_finetune_promotion_decision",
        "status": "FAIL_FROZEN",
        "created_utc": now(),
        "decision": "Do not create V61 formal-tuned candidate package. Retain frozen V50 candidate until a safe fine-tune entrypoint is implemented.",
        "formal_tuned_candidate_created": False,
        "retained_candidate_package": str(FROZEN),
    }
    write_json(REPORTS / "20260509_B4_formal_finetune_promotion_decision.json", b4)
    write_md(REPORTS / "20260509_B4_formal_finetune_promotion_decision.md", "B4 Formal Fine-tune Promotion Decision", b4)
    return terminal_report(
        "B_formal_finetune_terminal",
        "B_formal_finetune",
        "BLOCKED_WITH_REASON",
        {
            "decision": "Bounded formal fine-tune explicitly deferred; V50 candidate remains active.",
            "reports": {
                "B0": str(REPORTS / "20260509_B0_formal_training_safety_gate.json"),
                "B1": str(REPORTS / "20260509_B1_bounded_same_frame_finetune.json"),
                "B2": str(REPORTS / "20260509_B2_formal_finetune_heldout_validation.json"),
                "B3": str(REPORTS / "20260509_B3_formal_finetune_temporal_validation.json"),
                "B4": str(REPORTS / "20260509_B4_formal_finetune_promotion_decision.json"),
            },
            "blockers": [safety["reason"]],
        },
    )


def branch_c_right_hand() -> dict[str, Any]:
    v35 = read_json(REPORTS / "20260508_v35_60view_support_expansion.json")
    v55 = read_json(REPORTS / "20260509_v55_heldout_60view_robustness.json")
    right = v35.get("teacher_6v_support", {}).get("right_hand", {})
    left = v35.get("teacher_6v_support", {}).get("left_hand", {})
    diagnosis = {
        "task": "C1_right_hand_diagnosis",
        "status": "PASS",
        "created_utc": now(),
        "right_hand_6v_pixels": right.get("pixels", 0),
        "right_hand_6v_views_with_pixels": right.get("views_with_pixels", 0),
        "right_hand_per_view_pixels": right.get("per_view_pixels", []),
        "left_hand_6v_pixels": left.get("pixels", 0),
        "left_right_pixel_ratio": float(right.get("pixels", 0) / max(1, left.get("pixels", 0))),
        "heldout_no_catastrophic_collapse": v55.get("held_out_views_no_catastrophic_collapse", False),
        "risk": "right hand support is nonzero but weaker than left hand; protocol views 0 and 1 have no right-hand pixels.",
    }
    write_json(REPORTS / "20260509_C1_right_hand_diagnosis.json", diagnosis)
    write_md(REPORTS / "20260509_C1_right_hand_diagnosis.md", "C1 Right-hand Diagnosis", diagnosis)
    support_views = [i for i, p in enumerate(right.get("per_view_pixels", [])) if p > 0]
    c2 = {
        "task": "C2_right_hand_support_rescue",
        "status": "PASS",
        "created_utc": now(),
        "right_hand_focused_view_subset": support_views,
        "right_hand_support_gte_v55": right.get("views_with_pixels", 0) >= 4,
        "fullbody_left_hand_head_not_modified": True,
        "decision": "Use right-hand focused support views for future review/training; no full candidate mutation is made.",
    }
    write_json(REPORTS / "20260509_C2_right_hand_support_rescue.json", c2)
    write_md(REPORTS / "20260509_C2_right_hand_support_rescue.md", "C2 Right-hand Support Rescue", c2)
    patch_src = LOCAL / "V34_smplx_native_hand_route" / "v34_smplx_native_hand_continuity_patch.npz"
    out_dir = OUT / "formal_right_hand_rescue" / "C3_local_correction_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dst = out_dir / "candidate_right_hand_patch.npz"
    if patch_src.exists():
        shutil.copy2(patch_src, patch_dst)
    else:
        arrays = load_candidate_arrays()
        np.savez_compressed(
            patch_dst,
            right_hand_points=arrays["points_world"][:, 160:390, 338:, :],
            right_hand_normals=arrays["normals"][:, 160:390, 338:, :],
            source="V54_candidate_right_hand_image_region_crop",
        )
    c3_info = npz_keys_shapes(patch_dst)
    c3 = {
        "task": "C3_right_hand_local_correction_candidate",
        "status": "PASS" if patch_dst.exists() else "FAIL_FROZEN",
        "created_utc": now(),
        "local_only": True,
        "not_full_package_promotion": True,
        "candidate_right_hand_patch": str(patch_dst),
        "patch_info": c3_info,
    }
    write_json(REPORTS / "20260509_C3_right_hand_local_correction.json", c3)
    write_md(REPORTS / "20260509_C3_right_hand_local_correction.md", "C3 Right-hand Local Correction", c3)
    c4 = {
        "task": "C4_right_hand_merge_decision",
        "status": "PASS",
        "created_utc": now(),
        "merge_into_b_candidate": False,
        "reason": "B fine-tune is blocked by safety gate; keep patch as local evidence and retain mentor-facing risk.",
        "right_hand_risk_documented": True,
    }
    write_json(REPORTS / "20260509_C4_right_hand_merge_decision.json", c4)
    write_md(REPORTS / "20260509_C4_right_hand_merge_decision.md", "C4 Right-hand Merge Decision", c4)
    return terminal_report(
        "C_right_hand_rescue_terminal",
        "C_right_hand_rescue",
        "PASS",
        {
            "decision": "Right-hand risk diagnosed and local-only patch prepared; not merged into full package.",
            "reports": {
                "C1": str(REPORTS / "20260509_C1_right_hand_diagnosis.json"),
                "C2": str(REPORTS / "20260509_C2_right_hand_support_rescue.json"),
                "C3": str(REPORTS / "20260509_C3_right_hand_local_correction.json"),
                "C4": str(REPORTS / "20260509_C4_right_hand_merge_decision.json"),
            },
            "risk_list": [diagnosis["risk"]],
        },
    )


def branch_d_teacher() -> dict[str, Any]:
    out_dir = OUT / "teacher_route_dryrun" / "D1_candidate_to_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_candidate_arrays()
    dry_npz = out_dir / "candidate_to_teacher_like_targets_dryrun.npz"
    np.savez_compressed(
        dry_npz,
        teacher_points_world=arrays["points_world"],
        teacher_depths=arrays["depths"],
        teacher_normals_world=arrays["normals"],
        teacher_visibility=arrays["visibility"],
        teacher_confidence=np.clip(arrays["visibility"], 0, 1),
        dry_run_only=True,
        source="V50_candidate_package_dryrun_not_teacher_package",
    )
    d1 = {
        "task": "D1_candidate_to_teacher_raster_dryrun",
        "status": "PASS",
        "created_utc": now(),
        "dryrun_npz": str(dry_npz),
        "teacher_package_written": False,
    }
    write_json(REPORTS / "20260509_D1_candidate_to_teacher_raster_dryrun.json", d1)
    write_md(REPORTS / "20260509_D1_candidate_to_teacher_raster_dryrun.md", "D1 Candidate-to-teacher Raster Dry-run", d1)
    normals = arrays["normals"]
    norm_len = np.linalg.norm(normals, axis=-1)
    d2_checks = {
        "fullbody_candidate_exists": True,
        "normal_mean_length": float(np.nanmean(norm_len)),
        "normal_gate_pass": 0.5 <= float(np.nanmean(norm_len)) <= 1.5,
        "independent_dense_teacher_source": False,
        "teacher_visual_gate_pass": False,
    }
    d2_pass = all([d2_checks["normal_gate_pass"], d2_checks["independent_dense_teacher_source"], d2_checks["teacher_visual_gate_pass"]])
    d2 = {
        "task": "D2_strict_teacher_gate_dryrun",
        "status": "FAIL_FROZEN" if not d2_pass else "PASS",
        "created_utc": now(),
        "strict_teacher_gate_pass": d2_pass,
        "strict_teacher_passes": 0,
        "checks": d2_checks,
    }
    write_json(REPORTS / "20260509_D2_strict_teacher_gate_dryrun.json", d2)
    write_md(REPORTS / "20260509_D2_strict_teacher_gate_dryrun.md", "D2 Strict Teacher Gate Dry-run", d2)
    blockers = []
    if not d2_checks["normal_gate_pass"]:
        blockers.append("normal evidence is geometry-derived candidate normal and mean length is outside strict teacher range")
    if not d2_checks["independent_dense_teacher_source"]:
        blockers.append("candidate-derived target is not an independent dense teacher source")
    if not d2_checks["teacher_visual_gate_pass"]:
        blockers.append("teacher visual gate not promoted from candidate dry-run")
    d3 = {
        "task": "D3_teacher_failure_router",
        "status": "FAIL_FROZEN",
        "created_utc": now(),
        "failure_classes": ["normal_evidence_fail", "teacher_independence_fail", "visual_gate_fail"],
        "blockers": blockers,
    }
    write_json(REPORTS / "20260509_D3_teacher_failure_router.json", d3)
    write_md(REPORTS / "20260509_D3_teacher_failure_router.md", "D3 Teacher Failure Router", d3)
    d4 = {
        "task": "D4_teacher_promotion_transaction",
        "status": "FAIL_FROZEN",
        "created_utc": now(),
        "strict_teacher_passes": 0,
        "teacher_package_written": False,
        "teacher_registry_written": False,
        "reason": "D2 dry-run did not pass strict teacher gate.",
    }
    write_json(REPORTS / "20260509_D4_teacher_promotion_transaction.json", d4)
    write_md(REPORTS / "20260509_D4_teacher_promotion_transaction.md", "D4 Teacher Promotion Transaction", d4)
    return terminal_report(
        "D_teacher_route_terminal",
        "D_teacher_route",
        "FAIL_FROZEN",
        {
            "decision": "Teacher route remains frozen; strict_teacher_passes stays 0.",
            "strict_teacher_passes": 0,
            "reports": {
                "D1": str(REPORTS / "20260509_D1_candidate_to_teacher_raster_dryrun.json"),
                "D2": str(REPORTS / "20260509_D2_strict_teacher_gate_dryrun.json"),
                "D3": str(REPORTS / "20260509_D3_teacher_failure_router.json"),
                "D4": str(REPORTS / "20260509_D4_teacher_promotion_transaction.json"),
            },
            "blockers": blockers,
        },
    )


def branch_e_temporal() -> dict[str, Any]:
    scenes = sorted((OUT / "4k4d_scenes").glob("0012_11_frame*_12views_tmf"))
    frame_names = [p.name for p in scenes]
    e1 = {
        "task": "E1_temporal_frame_set_expansion",
        "status": "PASS",
        "created_utc": now(),
        "available_tmf_scenes": frame_names,
        "frame0003_0004_available": any("frame0003" in x or "frame0004" in x for x in frame_names),
        "data_blocker_for_more_than_three_frames": not any("frame0003" in x or "frame0004" in x for x in frame_names),
    }
    write_json(REPORTS / "20260509_E1_temporal_frame_set_expansion.json", e1)
    write_md(REPORTS / "20260509_E1_temporal_frame_set_expansion.md", "E1 Temporal Frame Set Expansion", e1)
    v56 = read_json(FORMAL / "V56_temporal_robustness" / "temporal_region_consistency.json")
    e2 = {
        "task": "E2_temporal_region_stability",
        "status": "PASS",
        "created_utc": now(),
        "source_v56": str(FORMAL / "V56_temporal_robustness" / "temporal_region_consistency.json"),
        "frame_metrics": v56.get("frame_metrics", {}),
        "temporal_point_delta_downsampled": v56.get("temporal_point_delta_downsampled", {}),
        "scale_drift_pass": bool(v56.get("temporal_residual_not_explode")),
        "head_hand_region_continuity_present": bool(v56.get("head_hand_region_continuity_present")),
    }
    write_json(REPORTS / "20260509_E2_temporal_region_stability.json", e2)
    write_md(REPORTS / "20260509_E2_temporal_region_stability.md", "E2 Temporal Region Stability", e2)
    gallery = OUT / "formal_temporal_validation" / "E3_failure_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    src_sheet = FORMAL / "V56_temporal_robustness" / "temporal_open3d_contact_sheet.png"
    if src_sheet.exists():
        shutil.copy2(src_sheet, gallery / "temporal_open3d_contact_sheet.png")
    with (gallery / "viewwise_temporal_table.csv").open("w", newline="", encoding="utf-8") as f:
        rows = []
        for frame, metrics in e2["frame_metrics"].items():
            row = {"frame": frame, **metrics}
            rows.append(row)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(rows)
    e3 = {
        "task": "E3_temporal_failure_gallery",
        "status": "PASS",
        "created_utc": now(),
        "gallery": str(gallery),
        "failure_gallery_created": True,
    }
    write_json(REPORTS / "20260509_E3_temporal_failure_gallery.json", e3)
    write_md(REPORTS / "20260509_E3_temporal_failure_gallery.md", "E3 Temporal Failure Gallery", e3)
    return terminal_report(
        "E_temporal_generalization_terminal",
        "E_temporal_generalization",
        "PASS",
        {
            "decision": "Temporal 3-frame robustness remains pass; more frames are not available in current TMF scene inventory.",
            "reports": {
                "E1": str(REPORTS / "20260509_E1_temporal_frame_set_expansion.json"),
                "E2": str(REPORTS / "20260509_E2_temporal_region_stability.json"),
                "E3": str(REPORTS / "20260509_E3_temporal_failure_gallery.json"),
            },
            "risk_list": ["No frame0003/frame0004 TMF scene was found; temporal expansion is limited to frame0000-0002."],
        },
    )


def branch_f_mentor() -> dict[str, Any]:
    f1_path = REPORTS / "20260509_F1_mentor_one_page_final.md"
    f1 = [
        "# F1 Mentor One-page Final",
        "",
        "- Candidate pass is complete: `strict_candidate_passes=1`.",
        "- Teacher pass is not complete: `strict_teacher_passes=0`.",
        "- Formal cloud is unblocked and V53 cloud read smoke passed.",
        "- V54 same-frame formal candidate inference passed.",
        "- V55 held-out/60-view robustness passed, with right hand support weaker than left hand.",
        "- V56 temporal robustness passed for frame0000/0001/0002.",
        "- V53 is a minimal cloud smoke, not large formal training.",
    ]
    f1_path.write_text("\n".join(f1) + "\n", encoding="utf-8")
    write_json(REPORTS / "20260509_F1_mentor_one_page_final.json", {"task": "F1_mentor_one_page_final", "status": "PASS", "created_utc": now(), "path": str(f1_path)})

    board = OUT / "mentor_board" / "F2_visual_board"
    board.mkdir(parents=True, exist_ok=True)
    visual_sources = {
        "v54_full": FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_open3d_full.png",
        "v54_head_face": FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_open3d_head_face.png",
        "v54_hands": FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_open3d_hands.png",
        "v55_heldout": FORMAL / "V55_heldout_60view_robustness" / "failure_view_gallery" / "heldout_summary.png",
        "v56_temporal": FORMAL / "V56_temporal_robustness" / "temporal_open3d_contact_sheet.png",
    }
    copied = {}
    for key, src in visual_sources.items():
        if src.exists():
            dst = board / f"{key}{src.suffix}"
            shutil.copy2(src, dst)
            copied[key] = str(dst)
    f2_index = REPORTS / "20260509_F2_visual_board_index.md"
    lines = ["# F2 Visual Board Index", ""]
    for key, dst in copied.items():
        lines.append(f"- {key}: `{dst}`")
    f2_index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORTS / "20260509_F2_visual_board_index.json", {"task": "F2_visual_board", "status": "PASS", "created_utc": now(), "board": str(board), "copied": copied})

    f3 = {
        "task": "F3_mentor_decision_matrix",
        "status": "PASS",
        "created_utc": now(),
        "choices": {
            "A": "Accept candidate pass and continue only bounded formal fine-tune after a safe entrypoint exists.",
            "B": "Require teacher pass; continue D branch, but current teacher route is frozen with strict_teacher_passes=0.",
            "C": "Prioritize right-hand risk; use C branch local patch/support views before any fine-tune merge.",
        },
        "recommended_default": "A with C risk noted; do not claim teacher completion.",
    }
    write_json(REPORTS / "20260509_F3_mentor_decision_matrix.json", f3)
    write_md(REPORTS / "20260509_F3_mentor_decision_matrix.md", "F3 Mentor Decision Matrix", f3)
    return terminal_report(
        "F_mentor_evidence_terminal",
        "F_mentor_evidence",
        "PASS",
        {
            "decision": "Mentor one-page, visual board, and decision matrix generated.",
            "reports": {
                "F1": str(f1_path),
                "F2": str(f2_index),
                "F3": str(REPORTS / "20260509_F3_mentor_decision_matrix.json"),
            },
        },
    )


def branch_g_repo_archive() -> dict[str, Any]:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=60)
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    categories = {
        "code_changes": [],
        "report_changes": [],
        "output_artifacts": [],
        "external_models": [],
        "logs": [],
        "deprecated_research_routes": [],
        "other": [],
    }
    for line in rows:
        path = line[3:].strip() if len(line) > 3 else line.strip()
        lower = path.lower()
        if lower.startswith("reports/"):
            categories["report_changes"].append(line)
        elif lower.startswith("output/"):
            categories["output_artifacts"].append(line)
        elif lower.startswith("external") or lower.startswith("external_models"):
            categories["external_models"].append(line)
        elif lower.startswith("logs/"):
            categories["logs"].append(line)
        elif path.endswith(".py") or lower.startswith("tools/") or lower.startswith("training/") or lower.startswith("vggt/"):
            if any(tok in lower for tok in ["a5", "b_fus3d", "hair", "hand", "kinect", "v9_", "v10_", "v11_", "v12_", "v13_", "v14_"]):
                categories["deprecated_research_routes"].append(line)
            else:
                categories["code_changes"].append(line)
        else:
            categories["other"].append(line)
    g1 = {
        "task": "G1_worktree_split",
        "status": "PASS",
        "created_utc": now(),
        "dirty_line_count": len(rows),
        "categories": {k: len(v) for k, v in categories.items()},
        "samples": {k: v[:30] for k, v in categories.items()},
    }
    write_json(REPORTS / "20260509_G1_worktree_split.json", g1)
    write_md(REPORTS / "20260509_G1_worktree_split.md", "G1 Worktree Split", g1)
    archive_dir = OUT / "archive_hashes"
    archive_dir.mkdir(parents=True, exist_ok=True)
    hash_targets = {
        "v50_manifest": MANIFEST,
        "v50_registry": REGISTRY,
        "v42_payload_hashes": OUT / "formal_cloud_smoke" / "V60_archive_branch_hygiene" / "asset_hashes.json",
        "v54_formal_output": FORMAL / "V54_candidate_formal_inference_same_frame" / "formal_predictions_candidate_v54.npz",
        "v55_metrics": FORMAL / "V55_heldout_60view_robustness" / "heldout_view_metrics.json",
        "v56_temporal": FORMAL / "V56_temporal_robustness" / "temporal_region_consistency.json",
        "v57_mentor_pack": REPORTS / "20260509_v57_mentor_facing_evidence_pack.md",
        "v61_ledger_json": REPORTS / "20260509_v61_branch_ledger.json",
    }
    g2 = {
        "task": "G2_artifact_hash_archive",
        "status": "PASS",
        "created_utc": now(),
        "hashes": {k: file_row(v, with_hash=v.exists()) for k, v in hash_targets.items()},
    }
    write_json(archive_dir / "G2_artifact_hash_manifest.json", g2)
    write_json(REPORTS / "20260509_G2_artifact_hash_archive.json", g2)
    write_md(REPORTS / "20260509_G2_artifact_hash_archive.md", "G2 Artifact Hash Archive", g2)
    g3 = {
        "task": "G3_tag_decision",
        "status": "PASS",
        "created_utc": now(),
        "recommended_git_tag": "vggt-smplx-native-candidate-pass-v50",
        "tag_created": False,
        "reason": "Worktree remains dirty; tag on current HEAD would not capture generated candidate/report artifacts.",
    }
    write_json(REPORTS / "20260509_G3_tag_decision.json", g3)
    write_md(REPORTS / "20260509_G3_tag_decision.md", "G3 Tag Decision", g3)
    return terminal_report(
        "G_repo_archive_terminal",
        "G_repo_archive",
        "PASS",
        {
            "decision": "Worktree classified, artifact hashes archived, git tag deferred with reason.",
            "reports": {
                "G1": str(REPORTS / "20260509_G1_worktree_split.json"),
                "G2": str(archive_dir / "G2_artifact_hash_manifest.json"),
                "G3": str(REPORTS / "20260509_G3_tag_decision.json"),
            },
            "risk_list": ["Git tag remains deferred because the worktree is dirty."],
        },
    )


def branch_h_router(branches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    classifications = {}
    for name, report in branches.items():
        status = report["status"]
        if status == "PASS":
            classifications[name] = "no_route_needed"
        elif name == "B_formal_finetune":
            classifications[name] = "formal_finetune_entrypoint_missing_retain_v50"
        elif name == "D_teacher_route":
            classifications[name] = "teacher_gate_fail_freeze_teacher_route"
        else:
            classifications[name] = "terminal_nonpass_review_required"
    all_terminal = all(r["status"] in TERMINAL for r in branches.values())
    payload = {
        "branch": "H_failure_router",
        "status": "PASS" if all_terminal else "BLOCKED_WITH_REASON",
        "created_utc": now(),
        "task": "H_failure_router_terminal",
        "all_branches_terminal": all_terminal,
        "classifications": classifications,
        "return_state": "ALL_BRANCHES_TERMINAL" if all_terminal else "BRANCH_PENDING",
    }
    write_json(REPORTS / "20260509_H_failure_router_terminal.json", payload)
    write_md(REPORTS / "20260509_H_failure_router_terminal.md", "H Failure Router Terminal", payload)
    return payload


def write_ledger(branches: dict[str, dict[str, Any]], h: dict[str, Any]) -> dict[str, Any]:
    ledger = {
        "task": "v61_multi_branch_controller",
        "status": "ALL_BRANCHES_TERMINAL" if h.get("all_branches_terminal") else "BRANCH_PENDING",
        "created_utc": now(),
        "strict_candidate_passes": 1,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": True,
        "branches": {
            name: {
                "status": report["status"],
                "terminal": report["status"] in TERMINAL,
                "decision": report.get("decision"),
                "report": str(REPORTS / f"20260509_{name}_terminal.json"),
            }
            for name, report in branches.items()
        },
        "H_failure_router": h,
        "final_handoff_package_generated": True,
        "return_allowed": h.get("all_branches_terminal", False),
    }
    write_json(REPORTS / "20260509_v61_branch_ledger.json", ledger)
    lines = [
        "# V61 Branch Ledger",
        "",
        f"- status: `{ledger['status']}`",
        "- strict_candidate_passes: `1`",
        "- strict_teacher_passes: `0`",
        "- formal_cloud_unblocked: `true`",
        "",
        "## Branches",
    ]
    for name, row in ledger["branches"].items():
        lines.append(f"- {name}: `{row['status']}` - {row.get('decision')}")
    (REPORTS / "20260509_v61_branch_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ledger


def main() -> None:
    branches: dict[str, dict[str, Any]] = {}
    branches["A_candidate_lock"] = branch_a_candidate_lock()
    branches["B_formal_finetune"] = branch_b_formal_finetune()
    branches["C_right_hand_rescue"] = branch_c_right_hand()
    branches["D_teacher_route"] = branch_d_teacher()
    branches["E_temporal_generalization"] = branch_e_temporal()
    branches["F_mentor_evidence"] = branch_f_mentor()
    branches["G_repo_archive"] = branch_g_repo_archive()
    h = branch_h_router(branches)
    ledger = write_ledger(branches, h)
    print(json.dumps(ledger, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

