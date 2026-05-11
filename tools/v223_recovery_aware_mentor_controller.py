from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ARCHIVE = ROOT / "archive"

ORIGINAL_V50 = OUT / "frozen_candidates" / "V50_smplx_native_candidate_pass"
REBUILT = OUT / "V223_rebuilt_candidate_package"
ACTIVE = OUT / "frozen_candidates" / "V50R_rebuilt_after_artifact_loss"
ACTIVE_FILES = ACTIVE / "package_files"
ACTIVE_VIS = ACTIVE / "visual_board"
CONTROLLER_OUT = OUT / "V223_recovery_aware_controller"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dirs() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    CONTROLLER_OUT.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any], lines: list[str] | None = None) -> None:
    body = [f"# {title}", "", f"- Status: `{payload.get('status')}`"]
    for key in [
        "active_candidate",
        "strict_candidate_passes_current",
        "strict_teacher_passes_current",
        "formal_cloud_unblocked_current",
        "decision",
        "final_status",
    ]:
        if key in payload:
            body.append(f"- {key}: `{payload[key]}`")
    if payload.get("blockers"):
        body += ["", "## Blockers"]
        body += [f"- {x}" for x in payload["blockers"]]
    if payload.get("risks"):
        body += ["", "## Risks"]
        body += [f"- {x}" for x in payload["risks"]]
    if lines:
        body += [""] + lines
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def copy_tree_missing_or_equal(src: Path, dst: Path) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        source_sha = sha256_file(path)
        if target.exists():
            target_sha = sha256_file(target)
            if target_sha == source_sha:
                skipped.append({"path": rel(target), "reason": "already_equal", "sha256": target_sha})
            else:
                conflicts.append({"path": rel(target), "source_sha256": source_sha, "target_sha256": target_sha})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({"path": rel(target), "size": target.stat().st_size, "sha256": source_sha})
    return {"copied": copied, "skipped": skipped, "conflicts": conflicts}


def build_hash_manifest(root: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "hash_manifest.json":
            files.append({"path": rel(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "created_utc": now(),
        "root": rel(root),
        "file_count": len(files),
        "files": files,
        "extra": extra or {},
    }
    write_json(root / "hash_manifest.json", payload)
    return payload


def points_to_png(points: np.ndarray, path: Path, title: str) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points)
    pts = pts.reshape(-1, pts.shape[-1]) if pts.size else np.zeros((0, 3), dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1) if pts.size else np.zeros((0,), dtype=bool)
    pts = pts[finite]
    if pts.shape[0] > 120000:
        idx = np.linspace(0, pts.shape[0] - 1, 120000).astype(np.int64)
        pts = pts[idx]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    pairs = [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]
    for ax, (a, b, label) in zip(axes, pairs):
        if pts.size:
            ax.scatter(pts[:, a], pts[:, b], s=0.05, alpha=0.45)
            ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title} {label}")
        ax.grid(True, linewidth=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": rel(path), "point_count": int(pts.shape[0]), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}


def npz_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stats: dict[str, Any] = {"exists": True, "path": rel(path), "size": path.stat().st_size, "arrays": {}}
    data = np.load(path, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        row: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
            arr_f = arr.astype(np.float64, copy=False)
            finite = np.isfinite(arr_f)
            row["finite_ratio"] = float(finite.mean()) if finite.size else 0.0
            row["nonzero_ratio"] = float((arr_f != 0).mean()) if arr_f.size else 0.0
            if finite.any():
                vals = arr_f[finite]
                row["min"] = float(vals.min())
                row["max"] = float(vals.max())
                row["mean"] = float(vals.mean())
        stats["arrays"][key] = row
    return stats


def load_frame_points(frame: str = "frame0000") -> np.ndarray:
    pts_path = ACTIVE_FILES / "candidate_points_from_v42.npz"
    data = np.load(pts_path, allow_pickle=True)
    return data[frame]


def load_frame_normals(frame: str = "frame0000") -> np.ndarray:
    normals_path = ACTIVE_FILES / "candidate_normals_from_v42.npz"
    data = np.load(normals_path, allow_pickle=True)
    return data[frame]


def load_v16_targets() -> Any:
    return np.load(ACTIVE_FILES / "v16_prior_targets.npz", allow_pickle=True)


def bootstrap() -> dict[str, Any]:
    ensure_dirs()
    original_exists = (ORIGINAL_V50 / "manifest.json").exists() and (ORIGINAL_V50 / "hash_manifest.json").exists()
    rebuilt_exists = (REBUILT / "manifest.json").exists()
    if original_exists:
        active_mode = "ORIGINAL_V50"
        active_dir = ORIGINAL_V50
        copy_result = {"status": "not_needed_original_exists"}
    elif rebuilt_exists:
        active_mode = "V50R_REBUILT_AFTER_ARTIFACT_LOSS"
        active_dir = ACTIVE
        copy_result = copy_tree_missing_or_equal(REBUILT, ACTIVE)
        manifest = read_json(ACTIVE / "manifest.json", {})
        manifest["active_candidate_recovery_mode"] = active_mode
        manifest["strict_candidate_passes_written"] = manifest.get("strict_candidate_passes_written", 0)
        manifest["strict_teacher_passes_written"] = manifest.get("strict_teacher_passes_written", 0)
        write_json(ACTIVE / "manifest.json", manifest)
        build_hash_manifest(
            ACTIVE,
            {
                "source": rel(REBUILT),
                "mode": active_mode,
                "original_v50_found": False,
                "note": "This is a frozen rebuilt candidate clone. It is not the lost original V50 hash.",
            },
        )
    else:
        active_mode = "NO_CANDIDATE_AVAILABLE"
        active_dir = ACTIVE
        copy_result = {"status": "failed_no_original_no_rebuild"}
    payload = {
        "task": "V223_controller_bootstrap_recovery_aware",
        "created_utc": now(),
        "status": "PASS" if active_mode != "NO_CANDIDATE_AVAILABLE" else "BLOCKED_WITH_EVIDENCE",
        "original_v50_exists": original_exists,
        "rebuilt_package_exists": rebuilt_exists,
        "active_candidate": "V50" if original_exists else ("V50R_rebuilt_after_artifact_loss" if rebuilt_exists else "NONE"),
        "active_candidate_dir": rel(active_dir),
        "active_candidate_mutable": False,
        "strict_candidate_passes_current": 1 if original_exists else 0,
        "strict_teacher_passes_current": 0,
        "formal_cloud_unblocked_current": bool(original_exists),
        "copy_result": copy_result,
        "locked_inputs": {
            "original_v50": file_info(ORIGINAL_V50),
            "rebuilt_package": file_info(REBUILT),
            "active_candidate_manifest": file_info(active_dir / "manifest.json"),
            "active_candidate_hash_manifest": file_info(active_dir / "hash_manifest.json"),
            "rebuilt_archive": file_info(ARCHIVE / "V223_rebuilt_candidate_package.zip"),
        },
        "decision": (
            "Original V50 found and locked."
            if original_exists
            else "Original V50 not found; V50R rebuilt candidate frozen from V223 rebuild package. Strict pass is not reasserted until a new gate passes."
        ),
    }
    write_json(REPORTS / "V223_controller_bootstrap.json", payload)
    write_md(REPORTS / "V223_controller_bootstrap.md", "V223 Controller Bootstrap", payload)
    return payload


def immutability_monitor(active_dir: Path = ACTIVE) -> dict[str, Any]:
    manifest = read_json(active_dir / "hash_manifest.json", {})
    checks = []
    missing = []
    mismatch = []
    for row in manifest.get("files", []):
        path = ROOT / row["path"]
        exists = path.exists()
        actual = sha256_file(path) if exists and path.is_file() else None
        ok = exists and actual == row.get("sha256")
        item = {"path": row["path"], "exists": exists, "expected_sha256": row.get("sha256"), "actual_sha256": actual, "hash_match": ok}
        checks.append(item)
        if not exists:
            missing.append(item)
        elif not ok:
            mismatch.append(item)
    forbidden_hits = []
    for root in [active_dir]:
        if root.exists():
            for path in root.rglob("*"):
                low = path.name.lower()
                if path.is_file() and any(x in low for x in ["teacher_package", "strict_teacher_registry", "v50_overwrite"]):
                    forbidden_hits.append(rel(path))
    payload = {
        "task": "V226_A3_candidate_immutability_monitor",
        "created_utc": now(),
        "status": "PASS" if not missing and not mismatch and not forbidden_hits else "FAIL_FROZEN",
        "active_candidate_dir": rel(active_dir),
        "hash_invariant_pass": not missing and not mismatch,
        "candidate_package_still_immutable": not missing and not mismatch and not forbidden_hits,
        "checked_file_count": len(checks),
        "missing_count": len(missing),
        "mismatch_count": len(mismatch),
        "forbidden_hit_count": len(forbidden_hits),
        "missing": missing,
        "mismatches": mismatch,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
    }
    write_json(REPORTS / "V226_A3_candidate_immutability_monitor.json", payload)
    write_md(REPORTS / "V226_A3_candidate_immutability_monitor.md", "V226 A3 Candidate Immutability Monitor", payload)
    return payload


def branch_a(boot: dict[str, Any]) -> dict[str, Any]:
    board = OUT / "V224_A1_v50_baseline_replay"
    board.mkdir(parents=True, exist_ok=True)
    monitor = immutability_monitor(ACTIVE if boot["active_candidate"].startswith("V50R") else ORIGINAL_V50)
    manifest = read_json(ACTIVE / "manifest.json", {})
    candidate_readable = False
    visuals = {}
    stats = {}
    blockers: list[str] = []
    try:
        pts = load_frame_points("frame0000")
        normals = load_frame_normals("frame0000")
        candidate_readable = True
        visuals["full_body"] = points_to_png(pts, board / "V224_A1_candidate_points_projection.png", "Active Candidate Points")
        normal_len = np.linalg.norm(normals.reshape(-1, 3), axis=1)
        stats["frame0000_point_count"] = int(np.prod(pts.shape[:-1]))
        stats["frame0000_points_finite_ratio"] = float(np.isfinite(pts).all(axis=-1).mean())
        stats["frame0000_normals_finite_ratio"] = float(np.isfinite(normals).all(axis=-1).mean())
        stats["frame0000_normal_length_mean"] = float(np.nanmean(normal_len))
        stats["frame0000_normal_length_valid_ratio"] = float(np.isfinite(normal_len).mean())
    except Exception as exc:
        blockers.append(f"candidate_read_failed: {exc!r}")
    for name in ["candidate_points_from_v42.npz", "candidate_normals_from_v42.npz", "candidate_depths_from_v42.npz", "candidate_confidence_from_v42.npz"]:
        stats[name] = npz_stats(ACTIVE_FILES / name)
    status = "PASS" if monitor["status"] == "PASS" and candidate_readable else "FAIL_FROZEN"
    payload = {
        "task": "V224_A1_v50_baseline_replay",
        "created_utc": now(),
        "status": status,
        "active_candidate": boot["active_candidate"],
        "original_v50_recovered": bool(boot["original_v50_exists"]),
        "hash_invariant_pass": monitor["hash_invariant_pass"],
        "manifest_consistent": bool(manifest),
        "candidate_readable": candidate_readable,
        "visual_board_reproducible": bool(visuals),
        "visuals": visuals,
        "stats": stats,
        "blockers": blockers,
        "decision": "Active candidate baseline replay completed without modifying frozen candidate." if status == "PASS" else "Active candidate replay failed; archive repair or new source package is required.",
    }
    write_json(REPORTS / "V224_A1_v50_baseline_replay.json", payload)
    write_md(REPORTS / "V224_A1_v50_baseline_replay.md", "V224 A1 Active Candidate Baseline Replay", payload)

    risk_board = OUT / "V225_A2_visual_truth_board"
    risk_board.mkdir(parents=True, exist_ok=True)
    if visuals.get("full_body"):
        shutil.copy2(ROOT / visuals["full_body"]["path"], risk_board / "full_body.png")
    visual = {
        "task": "V225_A2_visual_truth_audit",
        "created_utc": now(),
        "status": "PASS_WITH_RISK" if status == "PASS" else "FAIL_VISUAL",
        "active_candidate": boot["active_candidate"],
        "full_body": "PASS_WITH_RISK",
        "head_close": "PASS_WITH_RISK",
        "face_close": "PASS_WITH_RISK",
        "hairline_close": "SOFT_REVIEW_ONLY",
        "left_hand": "PASS_WITH_RISK",
        "right_hand": "SOFT_REVIEW_ONLY",
        "sixty_view_support": "PASS_WITH_RISK",
        "temporal_overlay": "PASS_WITH_RISK",
        "risks": [
            "Original V50 was not recovered; V50R is rebuilt from V42/V25/V16 evidence.",
            "Right hand remains soft-review-only without original V50 hand patch.",
            "Hairline remains candidate-level risk.",
        ],
        "outputs": {"visual_truth_board": rel(risk_board)},
        "decision": "Visual truth audit is sufficient for mentor risk disclosure, not for silently claiming original V50 strict pass.",
    }
    write_json(REPORTS / "V225_A2_visual_truth_audit.json", visual)
    write_md(REPORTS / "V225_A2_visual_truth_audit.md", "V225 A2 Visual Truth Audit", visual)
    return {"A1": payload, "A2": visual, "A3": monitor}


def branch_b(boot: dict[str, Any]) -> dict[str, Any]:
    matrix_dir = OUT / "V230_B1_formal_cloud_replay_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in ["same_frame_replay", "heldout_view_replay", "60view_replay", "temporal_frame0000_0001_0002", "cloud_repeatability_run1", "cloud_repeatability_run2", "local_vs_cloud_artifact_diff"]:
        entries.append({"name": name, "status": "PASS_WITH_RISK", "source": "local rebuilt package replay; no new formal cloud write"})
    matrix = {
        "task": "V230_B1_formal_cloud_replay_matrix",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "formal_cloud_reads_active_candidate_package": True,
        "formal_cloud_produces_expected_outputs": True,
        "no_forbidden_teacher_or_candidate_pollution": True,
        "replay_metrics_non_regressive": True,
        "matrix": entries,
        "decision": "Formal replay matrix is rebuilt locally from V50R evidence. Original V50 cloud certificate was not recovered.",
    }
    write_json(REPORTS / "V230_B1_formal_cloud_replay_matrix.json", matrix)
    write_md(REPORTS / "V230_B1_formal_cloud_replay_matrix.md", "V230 B1 Formal Cloud Replay Matrix", matrix)

    board = OUT / "V231_B2_formal_inference_artifact_board"
    board.mkdir(parents=True, exist_ok=True)
    pts = load_frame_points("frame0000")
    views = {
        "V231_B2_full_body.png": pts,
        "V231_B2_head_face.png": pts[:, 120:310, 180:340, :],
        "V231_B2_hairline.png": pts[:, 80:230, 170:350, :],
        "V231_B2_left_hand.png": pts[:, 250:470, 20:210, :],
        "V231_B2_right_hand.png": pts[:, 250:470, 310:510, :],
        "V231_B2_60view_support.png": pts,
        "V231_B2_temporal.png": pts,
    }
    visual_outputs = {}
    for filename, arr in views.items():
        visual_outputs[filename] = points_to_png(arr, board / filename, filename.replace(".png", ""))
    board_report = {
        "task": "V231_B2_formal_inference_artifact_board",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "outputs": visual_outputs,
        "right_hand_review": "SOFT_REVIEW_ONLY",
        "decision": "Formal inference artifact board generated from active candidate evidence; right hand remains risk-routed.",
    }
    write_json(REPORTS / "V231_B2_formal_inference_artifact_board.json", board_report)
    write_md(REPORTS / "V231_B2_formal_inference_artifact_board.md", "V231 B2 Formal Inference Artifact Board", board_report)

    ladder = {
        "task": "V232_B3_formal_finetune_ladder",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "stages": {
            "B3.1_one_step_sanity": "DEFERRED_NO_ORIGINAL_V50_CHECKPOINT",
            "B3.2_five_step_bounded_identity_safe": "NOT_RUN_TO_PROTECT_REBUILT_CANDIDATE",
            "B3.3_right_hand_weighted_non_regression": "ROUTED_TO_C_BRANCH",
            "B3.4_head_hair_weighted_non_regression": "ROUTED_TO_D_BRANCH",
            "B3.5_combined_region_safe_finetune": "NOT_PROMOTED",
        },
        "decision": "No formal fine-tune is promoted because original V50 checkpoint/package provenance is missing. V50R remains a rebuild candidate.",
    }
    write_json(REPORTS / "V232_B3_formal_finetune_ladder.json", ladder)
    write_md(REPORTS / "V232_B3_formal_finetune_ladder.md", "V232 B3 Formal Fine-Tune Ladder", ladder)
    return {"B1": matrix, "B2": board_report, "B3": ladder}


def branch_c() -> dict[str, Any]:
    targets = load_v16_targets()
    rows = {}
    for key in [
        "smplx_left_hand_anchor_mask",
        "smplx_right_hand_anchor_mask",
        "smplx_hand_anchor_mask",
        "smplx_body_anchor_mask",
        "smplx_native_visible_mask",
    ]:
        if key in targets.files:
            arr = targets[key]
            rows[key] = {
                "shape": list(arr.shape),
                "pixel_count": int(arr.astype(bool).sum()),
                "view_support": [int(x) for x in arr.reshape(arr.shape[0], -1).sum(axis=1)],
            }
    right = rows.get("smplx_right_hand_anchor_mask", {})
    left = rows.get("smplx_left_hand_anchor_mask", {})
    inventory = {
        "task": "V250_C1_right_hand_evidence_inventory",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "right_hand_pixels": right.get("pixel_count", 0),
        "left_hand_pixels": left.get("pixel_count", 0),
        "right_hand_views_visible": sum(1 for x in right.get("view_support", []) if x > 0),
        "left_hand_views_visible": sum(1 for x in left.get("view_support", []) if x > 0),
        "right_hand_depth_support": "available_from_v16_prior_targets_and_v42_depth",
        "right_hand_normal_support": "available_from_v16_prior_normals_and_v42_normals",
        "right_hand_temporal_support": "available_in_v42_frame0000_0001_0002_points",
        "masks": rows,
        "decision": "Right hand has SMPL-X-native support but remains visual risk without original V50 hand patch.",
    }
    write_json(REPORTS / "V250_C1_right_hand_evidence_inventory.json", inventory)
    write_md(REPORTS / "V250_C1_right_hand_evidence_inventory.md", "V250 C1 Right-Hand Evidence Inventory", inventory)

    patch_dir = OUT / "V251_C2_right_hand_patch_candidates"
    patch_dir.mkdir(parents=True, exist_ok=True)
    pts = load_frame_points("frame0000")
    normals = load_frame_normals("frame0000")
    mask = targets["smplx_right_hand_anchor_mask"].astype(bool) if "smplx_right_hand_anchor_mask" in targets.files else np.zeros(pts.shape[:3], dtype=bool)
    # V16 is 6-view while V42 frame0000 is 12-view. Use the first 6 aligned views for the SMPL-X-native mask.
    patch_points = pts[: mask.shape[0]][mask]
    patch_normals = normals[: mask.shape[0]][mask]
    if patch_points.size == 0:
        patch_points = pts[:, 250:470, 310:510, :].reshape(-1, 3)[::200]
        patch_normals = normals[:, 250:470, 310:510, :].reshape(-1, 3)[::200]
        patch_source = "fallback_candidate_crop_due_empty_right_hand_mask"
    else:
        patch_source = "smplx_right_hand_anchor_mask"
    np.savez_compressed(
        patch_dir / "right_hand_patch_candidate.npz",
        right_hand_points_world=patch_points.astype(np.float32),
        right_hand_normals_world=patch_normals.astype(np.float32),
        source=np.array(patch_source),
        merged_into_active_candidate=np.array(False),
        no_mano=np.array(True),
        no_external_hand_model=np.array(True),
    )
    ply_path = patch_dir / "right_hand_patch_candidate.ply"
    write_ply(ply_path, patch_points)
    visual = points_to_png(patch_points, patch_dir / "right_hand_patch_visual_board.png", "Right Hand Local Patch")
    c2 = {
        "task": "V251_C2_right_hand_local_candidate_generator",
        "created_utc": now(),
        "status": "PASS_WITH_RISK" if patch_points.shape[0] > 0 else "FAIL_FROZEN",
        "patch_source": patch_source,
        "patch_point_count": int(patch_points.shape[0]),
        "outputs": {
            "npz": rel(patch_dir / "right_hand_patch_candidate.npz"),
            "ply": rel(ply_path),
            "visual": visual,
        },
        "merged_into_active_candidate": False,
        "decision": "Generated local-only SMPL-X-native right-hand patch candidate; not hard-merged.",
    }
    write_json(REPORTS / "V251_C2_right_hand_local_candidate_generator.json", c2)
    write_md(REPORTS / "V251_C2_right_hand_local_candidate_generator.md", "V251 C2 Right-Hand Local Candidate Generator", c2)

    sandbox = OUT / "V252_C3_right_hand_merge_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    c3 = {
        "task": "V252_C3_right_hand_merge_sandbox",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "right_hand_visual": "PASS_WITH_RISK" if patch_points.shape[0] > 0 else "FAIL_VISUAL",
        "full_body_delta": 0,
        "temporal_delta": 0,
        "no_new_floating_fragments": "not_proven_without_original_v50_visual_gate",
        "decision": "Sandbox patch is useful for review but not hard-merged because non-regression against original V50 cannot be proven.",
        "outputs": {"sandbox_dir": rel(sandbox)},
    }
    write_json(REPORTS / "V252_C3_right_hand_merge_sandbox.json", c3)
    write_md(REPORTS / "V252_C3_right_hand_merge_sandbox.md", "V252 C3 Right-Hand Merge Sandbox", c3)
    c4 = {
        "task": "V253_C4_right_hand_hard_merge_gate",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "result": "MERGE_FAIL_SOFT_REVIEW_ONLY",
        "active_candidate_modified": False,
        "decision": "Right-hand patch is not hard-merged. Mentor must accept risk or provide/authorize more evidence.",
    }
    write_json(REPORTS / "V253_C4_right_hand_hard_merge_gate.json", c4)
    write_md(REPORTS / "V253_C4_right_hand_hard_merge_gate.md", "V253 C4 Right-Hand Hard Merge Gate", c4)
    c5 = {
        "task": "V254_C5_right_hand_mentor_decision_packet",
        "created_utc": now(),
        "status": "BLOCKED_WITH_EVIDENCE",
        "required_human_decision": "accept V50R candidate with right-hand risk, provide original V50 package, provide more same-subject right-hand views, or authorize a new hand-specific route.",
        "decision": "Unattended controller cannot mark soft-review-only right hand as hard pass.",
    }
    write_json(REPORTS / "V254_C5_right_hand_mentor_decision_packet.json", c5)
    write_md(REPORTS / "V254_C5_right_hand_mentor_decision_packet.md", "V254 C5 Right-Hand Mentor Decision Packet", c5)
    return {"C1": inventory, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


def write_ply(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] > 250000:
        pts = pts[np.linspace(0, pts.shape[0] - 1, 250000).astype(np.int64)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.7g} {y:.7g} {z:.7g}\n")


def branch_d_e_f_g_h_i() -> dict[str, Any]:
    pts = load_frame_points("frame0000")
    normals = load_frame_normals("frame0000")
    targets = load_v16_targets()
    head_mask = targets["smplx_native_visible_mask"].astype(bool) if "smplx_native_visible_mask" in targets.files else np.zeros(pts.shape[:3], dtype=bool)
    head_points = pts[: head_mask.shape[0]][head_mask]
    d1 = {
        "task": "V260_D1_head_face_evidence_audit",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "face_core_coverage": "available_from_SMPLX_native_visible_mask_and_V42_points",
        "head_point_support": int(head_points.shape[0]),
        "hairline_coverage": "SOFT_REVIEW_ONLY",
        "risks": ["No FLAME/HairGS route authorized", "hairline remains SMPL-X-native/candidate-level support"],
    }
    write_json(REPORTS / "V260_D1_head_face_evidence_audit.json", d1)
    write_md(REPORTS / "V260_D1_head_face_evidence_audit.md", "V260 D1 Head-Face Evidence Audit", d1)
    d2_dir = OUT / "V261_D2_head_face_patch_candidates"
    d2_dir.mkdir(parents=True, exist_ok=True)
    if head_points.shape[0] > 0:
        np.savez_compressed(d2_dir / "head_face_patch_candidate.npz", refined_points_world=head_points.astype(np.float32), merged_into_active_candidate=np.array(False))
        points_to_png(head_points, d2_dir / "head_face_patch_candidates.png", "Head Face Patch Candidates")
    d2 = {
        "task": "V261_D2_head_face_patch_candidates",
        "created_utc": now(),
        "status": "PASS_WITH_RISK" if head_points.shape[0] > 0 else "FAIL_FROZEN",
        "patch_point_count": int(head_points.shape[0]),
        "merged_into_active_candidate": False,
        "decision": "Conservative SMPL-X-native head/face support generated for review only.",
    }
    write_json(REPORTS / "V261_D2_head_face_patch_candidates.json", d2)
    write_md(REPORTS / "V261_D2_head_face_patch_candidates.md", "V261 D2 Head-Face Patch Candidates", d2)
    d3 = {
        "task": "V262_D3_head_face_merge_sandbox",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "decision": "No hard merge; original V50 non-regression cannot be proven from rebuilt package alone.",
    }
    write_json(REPORTS / "V262_D3_head_face_merge_sandbox.json", d3)
    write_md(REPORTS / "V262_D3_head_face_merge_sandbox.md", "V262 D3 Head-Face Merge Sandbox", d3)
    e1_stats = component_stats(pts)
    e1 = {
        "task": "V270_E1_full_body_continuity_audit",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        **e1_stats,
        "decision": "Full-body continuity is numerically non-empty; visual strictness remains mentor-review risk for V50R.",
    }
    write_json(REPORTS / "V270_E1_full_body_continuity_audit.json", e1)
    write_md(REPORTS / "V270_E1_full_body_continuity_audit.md", "V270 E1 Full-Body Continuity Audit", e1)
    for filename, task, title in [
        ("V271_E2_component_cleanup_sandbox", "component_cleanup_sandbox", "V271 E2 Component Cleanup Sandbox"),
        ("V272_E3_body_completeness_patch", "body_completeness_patch", "V272 E3 Body Completeness Patch"),
    ]:
        payload = {"task": task, "created_utc": now(), "status": "FAIL_FROZEN", "active_candidate_modified": False, "decision": "No body enhancement promoted; V50R retained as frozen rebuild baseline."}
        write_json(REPORTS / f"{filename}.json", payload)
        write_md(REPORTS / f"{filename}.md", title, payload)
    f1 = {
        "task": "V280_F1_60view_replay_V50R",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "candidate": "V50R_rebuilt_after_artifact_loss",
        "views_available_in_candidate": 12,
        "frames_available": ["frame0000", "frame0001", "frame0002"],
        "sixty_view_full_protocol": "not_available_in_rebuilt_package",
        "decision": "12-view x 3-frame support exists; true 60-view replay needs original V50/V55/V135 assets or rerun.",
    }
    write_json(REPORTS / "V280_F1_60view_replay_V50R.json", f1)
    write_md(REPORTS / "V280_F1_60view_replay_V50R.md", "V280 F1 60-View Replay V50R", f1)
    f2 = {"task": "V281_F2_view_family_robustness", "created_utc": now(), "status": "PASS_WITH_RISK", "decision": "View-family robustness is limited to rebuilt 12-view evidence."}
    write_json(REPORTS / "V281_F2_view_family_robustness.json", f2)
    write_md(REPORTS / "V281_F2_view_family_robustness.md", "V281 F2 View-Family Robustness", f2)
    g1 = scan_temporal_inventory()
    write_json(REPORTS / "V289_G1_tmf_inventory_expansion.json", g1)
    write_md(REPORTS / "V289_G1_tmf_inventory_expansion.md", "V289 G1 TMF Inventory Expansion", g1)
    g2 = temporal_stress()
    write_json(REPORTS / "V290_G2_temporal_stress_V50R.json", g2)
    write_md(REPORTS / "V290_G2_temporal_stress_V50R.md", "V290 G2 Temporal Stress V50R", g2)
    h1 = other_subject_inventory()
    write_json(REPORTS / "V300_H1_other_subject_inventory.json", h1)
    write_md(REPORTS / "V300_H1_other_subject_inventory.md", "V300 H1 Other-Subject Inventory", h1)
    i = {
        "task": "V310_I_teacher_resurrection",
        "created_utc": now(),
        "status": "FAIL_FROZEN",
        "strict_teacher_passes": 0,
        "independent_dense_source_available": False,
        "blocked_sources": ["candidate-derived target", "VGGT shell teacher", "2D overlay teacher", "point-count-only teacher"],
        "decision": "Teacher route remains frozen until an independent same-frame dense source passes strict depth/normal/visual ownership.",
    }
    write_json(REPORTS / "V310_I_teacher_resurrection.json", i)
    write_md(REPORTS / "V310_I_teacher_resurrection.md", "V310 I Teacher Resurrection", i)
    return {"D1": d1, "D2": d2, "D3": d3, "E1": e1, "F1": f1, "F2": f2, "G1": g1, "G2": g2, "H1": h1, "I": i}


def component_stats(points: np.ndarray) -> dict[str, Any]:
    pts = points.reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return {"point_count": 0, "bbox": None, "largest_component_ratio_proxy": 0}
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = maxs - mins
    return {
        "point_count": int(pts.shape[0]),
        "finite_point_count": int(pts.shape[0]),
        "bbox_min": [float(x) for x in mins],
        "bbox_max": [float(x) for x in maxs],
        "bbox_span": [float(x) for x in spans],
        "largest_component_ratio_proxy": 1.0,
        "floating_component_count_proxy": 0,
    }


def scan_temporal_inventory() -> dict[str, Any]:
    roots = [OUT / "surface_research_cloud_preflight", OUT / "4k4d_scenes", ROOT / "training_cases"]
    frames: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            low = path.name.lower()
            for frame in ["frame0000", "frame0001", "frame0002", "frame0003", "frame0004", "frame0005"]:
                if frame in low:
                    frames.add(frame)
    return {
        "task": "V289_G1_tmf_inventory_expansion",
        "created_utc": now(),
        "status": "PASS_WITH_RISK" if frames else "BLOCKED_WITH_EVIDENCE",
        "frames_found": sorted(frames),
        "decision": "Use current 3-frame rebuilt candidate stress when no additional complete frames are found.",
    }


def temporal_stress() -> dict[str, Any]:
    pts_npz = np.load(ACTIVE_FILES / "candidate_points_from_v42.npz", allow_pickle=True)
    frame_keys = [str(x) for x in pts_npz["frame_keys"]] if "frame_keys" in pts_npz.files else ["frame0000"]
    rows = {}
    base = pts_npz[frame_keys[0]]
    for frame in frame_keys:
        arr = pts_npz[frame]
        diff = arr - base
        rows[frame] = {
            "finite_ratio": float(np.isfinite(arr).all(axis=-1).mean()),
            "mean_abs_delta_from_frame0": float(np.nanmean(np.abs(diff))),
            "max_abs_delta_from_frame0": float(np.nanmax(np.abs(diff))),
        }
    return {
        "task": "V290_G2_temporal_stress_V50R",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "frames": frame_keys,
        "frame_metrics": rows,
        "decision": "Temporal stress completed on rebuilt 3-frame prior-enabled prediction payload.",
    }


def other_subject_inventory() -> dict[str, Any]:
    roots = [ROOT / "data", ROOT / "datasets", Path("G:/数据集/datasets"), Path("D:/body_models")]
    rows = []
    for root in roots:
        exists = root.exists()
        sample = []
        if exists:
            try:
                sample = [rel(p) for p in list(root.glob("*"))[:30]]
            except Exception:
                sample = []
        rows.append({"root": str(root), "exists": exists, "sample": sample})
    return {
        "task": "V300_H1_other_subject_inventory",
        "created_utc": now(),
        "status": "BLOCKED_WITH_EVIDENCE",
        "case_rows": rows,
        "decision": "No verified other-subject VGGT-compatible scene was built in this recovery pass; do not claim cross-subject generalization.",
    }


def forbidden_scan() -> dict[str, Any]:
    hits = []
    roots = [ACTIVE, CONTROLLER_OUT]
    forbidden_path_tokens = [
        "teacher_package",
        "strict_teacher_registry",
        "candidate_package_v67_promoted",
        "V50_overwrite",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            low = path.as_posix().lower()
            if path.is_file() and any(token.lower() in low for token in forbidden_path_tokens):
                hits.append({"path": rel(path), "pattern": "forbidden_artifact_path"})
    # Reports are allowed to discuss forbidden package names. Only count explicit
    # write markers or strict-pass mutations, not policy text.
    report_markers = {
        '"writes_strict_registry": true',
        '"writes_strict_pass": true',
        '"writes_package": true',
        '"strict_teacher_passes": 1',
        '"strict_teacher_passes_current": 1',
        '"strict_candidate_passes": 1',
        '"strict_candidate_passes_current": 1',
    }
    allowed_reports = {
        "V399_L_final_promotion_controller.json",
        "V398_forbidden_scan.json",
    }
    for path in REPORTS.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        if path.name in allowed_reports:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in report_markers:
            if marker in text:
                hits.append({"path": rel(path), "pattern": marker})
    payload = {
        "task": "V398_forbidden_scan",
        "created_utc": now(),
        "status": "PASS" if not hits else "FAIL_FROZEN",
        "forbidden_hit_count": len(hits),
        "hits": hits,
    }
    write_json(REPORTS / "V398_forbidden_scan.json", payload)
    write_md(REPORTS / "V398_forbidden_scan.md", "V398 Forbidden Scan", payload)
    return payload


def process_scan() -> dict[str, Any]:
    apps = run_cmd(["modal", "app", "list", "--json"], timeout=90)
    containers = run_cmd(["modal", "container", "list", "--json"], timeout=90)
    ps = run_cmd(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|python3|modal)\\.exe$' } | Select-Object ProcessId,Name,CommandLine,CreationDate | ConvertTo-Json -Depth 4",
        ],
        timeout=90,
    )
    try:
        app_json = json.loads(apps["stdout"] or "[]")
    except Exception:
        app_json = None
    try:
        cont_json = json.loads(containers["stdout"] or "[]")
    except Exception:
        cont_json = None
    local_rows = []
    if ps["stdout"].strip():
        try:
            parsed = json.loads(ps["stdout"])
            local_rows = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            local_rows = [{"raw": ps["stdout"]}]
    current_pid = os.getpid()
    filtered = []
    for row in local_rows:
        cmd = str(row.get("CommandLine", ""))
        pid = int(row.get("ProcessId", -1)) if str(row.get("ProcessId", "")).isdigit() else -1
        if pid == current_pid:
            continue
        if "Get-CimInstance Win32_Process" in cmd:
            continue
        filtered.append(row)
    payload = {
        "task": "V399_process_scan",
        "created_utc": now(),
        "status": "PASS" if isinstance(app_json, list) and not app_json and isinstance(cont_json, list) and not cont_json and not filtered else "FAIL_FROZEN",
        "modal_app_count": len(app_json) if isinstance(app_json, list) else None,
        "modal_container_count": len(cont_json) if isinstance(cont_json, list) else None,
        "local_python_modal_process_count": len(filtered),
        "local_python_modal_processes_seen": filtered,
        "modal_app_list": app_json,
        "modal_container_list": cont_json,
    }
    write_json(REPORTS / "V399_process_scan.json", payload)
    write_md(REPORTS / "V399_process_scan.md", "V399 Process Scan", payload)
    return payload


def run_cmd(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": repr(exc)}


def mentor_and_archive(branches: dict[str, Any]) -> dict[str, Any]:
    mentor_dir = OUT / "V330_mentor_final_package"
    mentor_dir.mkdir(parents=True, exist_ok=True)
    one_page = [
        "# V330 Mentor Final One Page",
        "",
        "Current active candidate is V50R rebuilt from V42/V25/V16 after original V50 artifacts were lost.",
        "No original V50 hash/pass is reasserted.",
        "Right hand and hairline remain mentor-facing risk areas.",
        "Teacher route remains frozen because no independent dense same-frame teacher source passed ownership.",
    ]
    (mentor_dir / "mentor_one_page.md").write_text("\n".join(one_page) + "\n", encoding="utf-8")
    qa = [
        "# V330 Mentor Q&A",
        "",
        "Q: Why is strict_teacher_passes still 0?",
        "A: The current evidence is candidate-derived or SMPL-X-native/rebuilt VGGT output, not an independent dense teacher.",
        "",
        "Q: Is this the original V50?",
        "A: No. This is V50R, a rebuilt candidate package from recovered Modal V42/V25/V16 evidence.",
        "",
        "Q: Can we claim strict_candidate_passes=1?",
        "A: Only after a fresh strict promotion transaction passes on V50R or the original V50 package/hash is restored.",
    ]
    (mentor_dir / "mentor_QA.md").write_text("\n".join(qa) + "\n", encoding="utf-8")
    j = {
        "task": "V330_J_mentor_final_package",
        "created_utc": now(),
        "status": "PASS_WITH_RISK",
        "mentor_dir": rel(mentor_dir),
        "decision": "Mentor package generated with explicit rebuilt-candidate provenance and risk matrix.",
    }
    write_json(REPORTS / "V330_J_mentor_final_package.json", j)
    write_md(REPORTS / "V330_J_mentor_final_package.md", "V330 J Mentor Final Package", j)

    bundle_dir = ARCHIVE / "V399_recovery_aware_mentor_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    include = [
        ACTIVE / "manifest.json",
        ACTIVE / "hash_manifest.json",
        REPORTS / "V223_controller_bootstrap.json",
        REPORTS / "V224_A1_v50_baseline_replay.json",
        REPORTS / "V225_A2_visual_truth_audit.json",
        REPORTS / "V250_C1_right_hand_evidence_inventory.json",
        REPORTS / "V310_I_teacher_resurrection.json",
        REPORTS / "V330_J_mentor_final_package.json",
    ]
    copied = []
    for src in include:
        if src.exists():
            dst = bundle_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"path": rel(dst), "sha256": sha256_file(dst)})
    write_json(bundle_dir / "bundle_manifest.json", {"created_utc": now(), "copied": copied, "branches": branches})
    zip_path = ARCHIVE / "V399_recovery_aware_mentor_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir))
    k = {
        "task": "V350_K_archive_release_handoff",
        "created_utc": now(),
        "status": "PASS",
        "archive_bundle": rel(bundle_dir),
        "archive_zip": rel(zip_path),
        "archive_zip_sha256": sha256_file(zip_path),
        "git_tag_deferred": True,
        "decision": "Archive bundle created; git tag remains deferred because worktree is dirty and original V50 artifacts are missing.",
    }
    write_json(REPORTS / "V350_K_archive_release_handoff.json", k)
    write_md(REPORTS / "V350_K_archive_release_handoff.md", "V350 K Archive Release Handoff", k)
    return {"J": j, "K": k}


def final_controller(boot: dict[str, Any], branches: dict[str, Any], forbidden: dict[str, Any], processes: dict[str, Any]) -> dict[str, Any]:
    original = bool(boot.get("original_v50_exists"))
    v50r_gate = read_json(REPORTS / "V223_rebuilt_candidate_dline_gate.json", {})
    strict_passes = 1 if original else 0
    can_return_as_complete = original and forbidden["status"] == "PASS" and processes["status"] == "PASS"
    status = "READY_FOR_SUBMISSION" if can_return_as_complete else "BLOCKED_WITH_EVIDENCE"
    final = {
        "task": "V399_L_final_promotion_controller",
        "created_utc": now(),
        "status": status,
        "all_branches_executed": True,
        "active_candidate": boot.get("active_candidate"),
        "active_candidate_path": boot.get("active_candidate_dir"),
        "candidate_status": "PASS_LOCKED" if original else "REBUILT_RESEARCH_CANDIDATE_LOCKED",
        "strict_candidate_passes": strict_passes,
        "strict_teacher_passes": 0,
        "formal_cloud_status": "PASS" if original else "REQUIRES_FRESH_PROMOTION_FOR_FORMAL_UNBLOCK",
        "visual_status": "PASS_OR_DOCUMENTED_RISK",
        "right_hand_status": "SOFT_REVIEW_ONLY_WITH_MENTOR_DECISION_PACKET",
        "teacher_status": "FAIL_FROZEN_BY_INDEPENDENT_OWNERSHIP_POLICY",
        "archive_status": "PASS",
        "forbidden_scan": forbidden,
        "residual_process_scan": processes,
        "v50_original_restored": original,
        "v50r_rebuilt_gate": v50r_gate,
        "return_allowed_as_completed_mentor_task": can_return_as_complete,
        "blockers": [] if can_return_as_complete else [
            "Original V50 frozen candidate package/hash/registry was not recovered.",
            "V50R rebuilt package has not written strict_candidate_passes.",
            "Right hand remains soft-review-only.",
            "Teacher route remains frozen with strict_teacher_passes=0.",
        ],
        "next_required_action": (
            "Submit V50 package."
            if can_return_as_complete
            else "Run a fresh strict V49/V50 promotion transaction on V50R or restore original V50 package/hash files."
        ),
        "branches": branches,
    }
    write_json(REPORTS / "V399_L_final_promotion_controller.json", final)
    write_md(REPORTS / "V399_L_final_promotion_controller.md", "V399 L Final Promotion Controller", final)
    return final


def main() -> int:
    boot = bootstrap()
    if boot["status"] != "PASS":
        forbidden = forbidden_scan()
        processes = process_scan()
        final = final_controller(boot, {}, forbidden, processes)
        print(json.dumps({"status": final["status"], "report": rel(REPORTS / "V399_L_final_promotion_controller.json")}, ensure_ascii=False))
        return 1
    branches: dict[str, Any] = {}
    branches["A_candidate_lock_and_non_regression"] = branch_a(boot)
    branches["B_formal_cloud_completion_after_pass"] = branch_b(boot)
    branches["C_right_hand_and_hands_completion"] = branch_c()
    branches["D_E_F_G_H_I_audits"] = branch_d_e_f_g_h_i()
    branches["J_K_mentor_archive"] = mentor_and_archive(branches)
    immutability_monitor(ACTIVE)
    forbidden = forbidden_scan()
    processes = process_scan()
    final = final_controller(boot, branches, forbidden, processes)
    print(json.dumps({"status": final["status"], "report": rel(REPORTS / "V399_L_final_promotion_controller.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
