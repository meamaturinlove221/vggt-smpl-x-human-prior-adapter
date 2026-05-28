from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
V940 = OUTPUT / "V940000000000000_smpl_feature_bank"
V950 = OUTPUT / "V9500000000000000_smpl_feature_bank_v4"
V140 = OUTPUT / "V1400000000000000000_learned_residual_matrix"
V161 = OUTPUT / "V161000000000000_repaired_detail_regions"
V536 = OUTPUT / "V5360000000000000000_geometry_part_binding_repair"
CASES = ["0012_11_frame001", "0013_01_frame001", "0021_03_frame001", "current_v895_0021_03"]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npz(path: Path, **arrays: Any) -> None:
    ensure(path.parent)
    np.savez_compressed(path, **arrays)


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["case"])
        writer.writeheader()
        writer.writerows(rows)


def nearest_index_and_dist(source: np.ndarray, target: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    best_dist = np.full(len(src), np.inf, dtype=np.float32)
    best_idx = np.zeros(len(src), dtype=np.int64)
    for start in range(0, len(tgt), chunk_size):
        chunk = tgt[start : start + chunk_size]
        d2 = ((src[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        local_idx = np.argmin(d2, axis=1)
        local_dist = d2[np.arange(len(src)), local_idx]
        take = local_dist < best_dist
        best_dist[take] = local_dist[take]
        best_idx[take] = start + local_idx[take]
    return best_idx.astype(np.int32), np.sqrt(np.maximum(best_dist, 0.0)).astype(np.float32)


def restore_v950(case: str) -> dict[str, Any]:
    src = V940 / case / "smpl_feature_bank.npz"
    dst = V950 / case / "smpl_feature_bank_v4.npz"
    data = load_npz(src)
    arrays = dict(data)
    arrays["restored_from_v940_feature_bank"] = np.array(True)
    arrays["v188_asset_restoration"] = np.array(True)
    save_npz(dst, **arrays)
    return {
        "case": case,
        "asset": "V950_smpl_feature_bank_v4",
        "source": str(src),
        "path": str(dst),
        "exists": dst.exists(),
        "size": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "keys": ";".join(load_npz(dst).keys()),
        "restored_from": "V940 smpl_feature_bank.npz",
    }


def mask_or_false(data: dict[str, np.ndarray], key: str, n: int) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=bool)
    return np.zeros(n, dtype=bool)


def restore_v536(case: str) -> dict[str, Any]:
    baseline_path = V140 / case / "real_vggt_baseline_only" / "predictions.npz"
    target_path = V161 / case / "repaired_detail_regions_world_rgb.npz"
    dst = V536 / case / "mentor_view_geometry_part_graph.npz"
    base = load_npz(baseline_path)
    target = load_npz(target_path)
    mentor = np.asarray(base["human_points"], dtype=np.float32)
    mentor_rgb = np.asarray(base["human_rgb"], dtype=np.uint8)
    target_points = np.asarray(target["world_points"], dtype=np.float32)
    target_body = np.asarray(target["body_part_id"], dtype=np.int16)
    nn_idx, nn_dist = nearest_index_and_dist(mentor, target_points)
    geom_body = target_body[nn_idx].astype(np.int16)
    raw_body = np.asarray(base["body_part_id"], dtype=np.int16)
    conf = np.asarray(target["confidence"], dtype=np.float32)[nn_idx]
    dist_norm = nn_dist / max(float(np.percentile(nn_dist, 95)), 1e-6)
    weak = np.clip(0.62 * dist_norm + 0.38 * (1.0 - conf), 0.0, 1.0).astype(np.float32)
    no_change = (nn_dist <= np.percentile(nn_dist, 45)) & (conf >= np.percentile(conf, 45))

    def remap(mask_key: str) -> np.ndarray:
        return mask_or_false(target, mask_key, len(target_points))[nn_idx]

    thresholds = {
        "restored_from": "V140 baseline + V161 repaired detail regions",
        "nn_distance_p50": float(np.percentile(nn_dist, 50)),
        "nn_distance_p95": float(np.percentile(nn_dist, 95)),
        "weak_mean": float(np.mean(weak)),
    }
    save_npz(
        dst,
        case_id=np.array(case),
        mentor_points=mentor,
        mentor_rgb=mentor_rgb,
        environment_points=np.asarray(base["environment_points"], dtype=np.float32),
        environment_rgb=np.asarray(base["environment_rgb"], dtype=np.uint8),
        raw_body_part_id=raw_body,
        geometry_body_part_id=geom_body,
        mentor_to_smpl_index=nn_idx,
        mentor_to_smpl_norm_distance=dist_norm.astype(np.float32),
        mentor_smpl_confidence=conf.astype(np.float32),
        mentor_weak_region_score=weak,
        no_change_mask=no_change.astype(bool),
        head_hair_contour_mask=remap("mask_head_hair") | remap("mask_face_head_silhouette"),
        shoulder_neck_mask=remap("mask_shoulder_neck"),
        hand_arm_endpoint_mask=remap("mask_arms_hands"),
        clothing_torso_boundary_mask=remap("mask_torso_clothing_boundary"),
        leg_foot_morphology_mask=remap("mask_feet_leg_boundary"),
        geometry_thresholds_json=np.array(json.dumps(thresholds, ensure_ascii=False)),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        restored_from_v140_v161=np.array(True),
        v188_asset_restoration=np.array(True),
    )
    return {
        "case": case,
        "asset": "V536_geometry_part_binding_graph",
        "source": f"{baseline_path};{target_path}",
        "path": str(dst),
        "exists": dst.exists(),
        "size": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "keys": ";".join(load_npz(dst).keys()),
        "restored_from": "V140 baseline + V161 target",
        "weak_mean": float(np.mean(weak)),
        "no_change_ratio": float(np.mean(no_change)),
    }


def audit_existing(case: str, asset: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"case": case, "asset": asset, "path": str(path), "exists": False}
    data = load_npz(path)
    return {
        "case": case,
        "asset": asset,
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "keys": ";".join(data.keys()),
    }


def main() -> int:
    created_at = now()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in CASES:
        required_sources = [
            V940 / case / "smpl_feature_bank.npz",
            V161 / case / "repaired_detail_regions_world_rgb.npz",
            V140 / case / "real_vggt_baseline_only" / "predictions.npz",
        ]
        missing = [str(p) for p in required_sources if not p.exists()]
        if missing:
            failures.append({"case": case, "reason": "missing_restore_source", "missing": missing})
            continue
        rows.append(restore_v950(case))
        rows.append(restore_v536(case))
        rows.append(audit_existing(case, "V161_repaired_detail_regions", V161 / case / "repaired_detail_regions_world_rgb.npz"))
        rows.append(audit_existing(case, "V140_real_vggt_baseline_only", V140 / case / "real_vggt_baseline_only" / "predictions.npz"))

    four_case_ok = len({r["case"] for r in rows if r.get("exists") and r["asset"] == "V950_smpl_feature_bank_v4"}) == 4 and len(
        {r["case"] for r in rows if r.get("exists") and r["asset"] == "V536_geometry_part_binding_graph"}
    ) == 4
    write_csv(REPORTS / "V18800000000000000000_restored_asset_manifest.csv", rows)
    write_json(
        REPORTS / "V18800000000000000000_asset_restoration_decision.json",
        {
            "created_at": created_at,
            "status": "V18800_ASSETS_RESTORED_FOR_NON_FALLBACK_V187" if four_case_ok and not failures else "V18800_ASSET_RESTORATION_INCOMPLETE",
            "mentor_ready": False,
            "external_hard_block": False,
            "four_case_v950_v536_ok": four_case_ok,
            "failures": failures,
            "manifest": str(REPORTS / "V18800000000000000000_restored_asset_manifest.csv"),
            "note": "Restored assets are training inputs only. They do not prove mentor readiness.",
        },
    )
    print(json.dumps({"status": "V18800_DONE", "four_case_v950_v536_ok": four_case_ok, "failures": failures}, indent=2))
    return 0 if four_case_ok and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
