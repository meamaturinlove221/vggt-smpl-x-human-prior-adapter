from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"

BASE_ROOT = OUTPUT / "V1400000000000000000_learned_residual_matrix"
GRAPH_ROOT = OUTPUT / "V5360000000000000000_geometry_part_binding_repair"
V161_ROOT = OUTPUT / "V161000000000000_repaired_detail_regions"
PAYLOAD_ROOT = OUTPUT / "V10210000000000000000_true_3d_geometry_training_payload"

CASES = [
    "0012_11_frame001",
    "0013_01_frame001",
    "0021_03_frame001",
    "current_v895_0021_03",
]
CONTROL_CONFIGS = [
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
]
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
REGION_KEYS = [
    "head_hair_contour_mask",
    "shoulder_neck_mask",
    "hand_arm_endpoint_mask",
    "clothing_torso_boundary_mask",
    "leg_foot_morphology_mask",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


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


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def base_path(case: str) -> Path:
    return BASE_ROOT / case / "real_vggt_baseline_only" / "predictions.npz"


def graph_path(case: str) -> Path:
    return GRAPH_ROOT / case / "mentor_view_geometry_part_graph.npz"


def target_path(case: str) -> Path:
    return V161_ROOT / case / "repaired_detail_regions_world_rgb.npz"


def control_path(case: str, config: str) -> Path:
    return BASE_ROOT / case / config / "predictions.npz"


def pca_metrics(points: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    ranges = np.ptp(proj, axis=0)
    bbox = np.ptp(pts, axis=0)
    return {
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "pca_range_1": float(ranges[0]),
        "pca_range_2": float(ranges[1]),
        "pca_range_3": float(ranges[2]),
        "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
    }


def summarize_case(case: str) -> dict[str, Any]:
    required = {
        "baseline": base_path(case),
        "graph": graph_path(case),
        "visible_target": target_path(case),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    row: dict[str, Any] = {
        "case": case,
        "baseline_path": str(required["baseline"]),
        "graph_path": str(required["graph"]),
        "visible_target_path": str(required["visible_target"]),
        "missing_required": ";".join(missing),
        "eligible_for_training_payload": not missing,
    }
    for config in CONTROL_CONFIGS:
        path = control_path(case, config)
        row[f"control_{config}_exists"] = path.exists()
        row[f"control_{config}_path"] = str(path)
    if missing:
        return row
    base = load_npz(required["baseline"])
    graph = load_npz(required["graph"])
    target = load_npz(required["visible_target"])
    hp = np.asarray(base["human_points"], dtype=np.float32)
    row.update({f"baseline_{k}": v for k, v in pca_metrics(hp).items()})
    row["human_points"] = int(len(hp))
    row["environment_points"] = int(len(base["environment_points"]))
    row["human_ratio"] = float(len(hp) / max(1, len(hp) + len(base["environment_points"])))
    weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    row["weak_region_mean"] = float(np.mean(weak))
    row["weak_region_gt_018"] = float(np.mean(weak > 0.18))
    row["no_change_ratio"] = float(np.mean(np.asarray(graph["no_change_mask"], dtype=bool)))
    for key in REGION_KEYS:
        row[key + "_ratio"] = float(np.mean(np.asarray(graph[key], dtype=bool))) if key in graph else 0.0
    if "world_points" in target:
        row["visible_target_points"] = int(len(target["world_points"]))
        row.update({f"visible_target_{k}": v for k, v in pca_metrics(np.asarray(target["world_points"], dtype=np.float32)).items()})
    elif "human_points" in target:
        row["visible_target_points"] = int(len(target["human_points"]))
        row.update({f"visible_target_{k}": v for k, v in pca_metrics(np.asarray(target["human_points"], dtype=np.float32)).items()})
    row["teacher_points_used_at_inference"] = False
    row["raw_kinect_depth_used_at_inference"] = False
    row["facial_detail_target_applicable"] = False
    row["face_detail_claim_allowed"] = False
    row["allowed_face_claim"] = ALLOWED_FACE_CLAIM
    return row


def main() -> int:
    created_at = now()
    ensure(PAYLOAD_ROOT)
    rows = [summarize_case(case) for case in CASES]
    eligible = [r["case"] for r in rows if r["eligible_for_training_payload"]]
    asset_csv = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
    write_csv(asset_csv, rows)

    loss_contract = REPORTS / "V10210000000000000000_true_3d_geometry_loss_contract.md"
    write_text(
        loss_contract,
        f"""# V10210 True 3D Geometry Loss Contract

Created: {created_at}

Purpose: continue the V950100 visual-supervised residual route after V10200 failed closed.

Primary mentor gate remains a full-scene RGB point cloud with human as subject and partial real environment. Projection, metrics, source labels, and render-only repairs are auxiliary.

Training inputs:
- VGGT baseline human/environment points and RGB.
- V536 canonical/graph visible body binding.
- V161 visible target points as supervision target only.
- Same-scene hard controls for separation.

Forbidden inference inputs:
- raw Kinect depth
- teacher points
- dense V591/Kinect fusion

Loss terms:
1. `weak_region_residual_l1`: fit residuals only in visible weak regions.
2. `baseline_preservation_l1`: preserve high-confidence/no-change VGGT baseline zones.
3. `thickness_side_loss`: increase valid side-view thickness only when topology remains coherent.
4. `limb_continuity_loss`: keep hand/arm, leg/foot, shoulder/neck regions connected rather than noisy.
5. `part_topology_loss`: respect graph/body-part neighborhoods.
6. `environment_preservation_loss`: keep real VGGT environment unchanged and visible.
7. `control_separation_loss`: separate true from posthoc, same-topology, tiny, shuffled controls in 3D morphology, not by source labels.
8. `projection_aux_loss`: optional auxiliary only; cannot rescue 3D visual failure.

Face detail policy:
- facial detail target applicable: false
- face detail claim allowed: false
- allowed claim: {ALLOWED_FACE_CLAIM}
""",
    )

    worker_spec = PAYLOAD_ROOT / "V10210_worker_spec.json"
    write_json(
        worker_spec,
        {
            "created_at": created_at,
            "route": "V10210_true_3d_geometry_training",
            "eligible_cases": eligible,
            "case_count": len(eligible),
            "model": "models/v2500_canonical_surfel_residual_student.py::CanonicalSurfelResidualStudent",
            "required_outputs": [
                "student predictions.npz per case/config",
                "full_scene_rgb_pointcloud.ply",
                "oblique depth-cued mentor board",
                "same-scene controls board",
                "local 3D visible-part board",
                "hard-control separation report",
            ],
            "final_claim_policy": "No mentor-ready claim from payload or smoke. Must pass full mentor visual gate.",
            "no_agent_rule": "main-thread/no-agent unless user explicitly reauthorizes in the current turn",
            "teacher_points_used_at_inference": False,
            "raw_kinect_depth_used_at_inference": False,
            "facial_detail_target_applicable": False,
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
        },
    )

    next_goal = GOALS / "V10220000000000000000_auto_evolved_true_3d_geometry_smoke_route.md"
    write_text(
        next_goal,
        f"""# V10220 Auto-Evolved True 3D Geometry Smoke Route

Created: {created_at}

V10210 created a training payload for true 3D geometry repair.

Next:
- run local model smoke against `CanonicalSurfelResidualStudent`;
- verify forbidden teacher/Kinect inference rejection;
- build a tiny case batch from the V10210 asset manifest;
- do not claim mentor-ready from smoke.
""",
    )

    decision = {
        "created_at": created_at,
        "status": "V10210_TRUE_3D_GEOMETRY_TRAINING_PAYLOAD_READY_INTERNAL_ONLY" if eligible else "V10210_NO_ELIGIBLE_CASES_FAIL_CLOSED",
        "asset_manifest": str(asset_csv),
        "loss_contract": str(loss_contract),
        "worker_spec": str(worker_spec),
        "eligible_cases": eligible,
        "eligible_case_count": len(eligible),
        "next_goal": str(next_goal),
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "next_action": "Run V10220 local model smoke and payload batch check; then prepare train/gate route.",
    }
    write_json(REPORTS / "V10210000000000000000_true_3d_geometry_training_payload_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
