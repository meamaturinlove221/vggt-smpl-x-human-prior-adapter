from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
V173_ROOT = OUTPUT / "V17300000000000000000_multishell_topology_decoder_training"
V177_ROOT = OUTPUT / "V17700000000000000000_front_back_occupancy_training"
V179_ROOT = OUTPUT / "V17900000000000000000_collision_aware_topology_repair"

V173_CONFIG = "multishell_topology_decoder_true"
V177_CONFIG = "front_back_occupancy_true"
V179_CONFIG = "collision_aware_topology_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]
EVAL_CONFIGS = [V173_CONFIG, V177_CONFIG, V179_CONFIG, *CONTROL_CONFIGS]

PART_NAMES = {
    0: "head_hair",
    1: "torso_shoulder",
    2: "left_arm_endpoint",
    3: "right_arm_endpoint",
    4: "left_leg",
    5: "right_leg",
    6: "left_foot_endpoint",
    7: "right_foot_endpoint",
}

# Edges that are allowed to touch because they are real body topology.
ADJACENT_EDGES = {
    tuple(sorted(edge))
    for edge in [
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (4, 6),
        (5, 7),
    ]
}

# Semantically distant overlaps that should remain separated in 3D.
DISTANT_PAIR_WEIGHTS = {
    tuple(sorted(pair)): weight
    for pair, weight in {
        (0, 4): 1.20,
        (0, 5): 1.20,
        (0, 6): 1.35,
        (0, 7): 1.35,
        (1, 6): 1.10,
        (1, 7): 1.10,
        (2, 3): 0.90,
        (2, 4): 1.00,
        (2, 5): 1.00,
        (2, 6): 1.25,
        (2, 7): 1.10,
        (3, 4): 1.00,
        (3, 5): 1.00,
        (3, 6): 1.10,
        (3, 7): 1.25,
        (4, 5): 0.85,
        (4, 7): 0.95,
        (5, 6): 0.95,
        (6, 7): 0.80,
    }.items()
}

from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2, pca_frame  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prediction_path(case: str, config: str) -> Path:
    if config == V173_CONFIG:
        return V173_ROOT / case / config / "predictions.npz"
    if config == V177_CONFIG:
        return V177_ROOT / case / config / "predictions.npz"
    if config == V179_CONFIG:
        return V179_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def bbox_overlap_ratio(
    center_a: np.ndarray,
    extent_a: np.ndarray,
    center_b: np.ndarray,
    extent_b: np.ndarray,
) -> tuple[float, float, float]:
    half_a = np.maximum(extent_a * 0.5, 1e-9)
    half_b = np.maximum(extent_b * 0.5, 1e-9)
    low = np.maximum(center_a - half_a, center_b - half_b)
    high = np.minimum(center_a + half_a, center_b + half_b)
    overlap = np.maximum(0.0, high - low)
    min_extent = np.maximum(np.minimum(extent_a, extent_b), 1e-9)
    per_axis = np.clip(overlap / min_extent, 0.0, 1.0)
    mean_overlap = float(np.mean(per_axis))
    volume_overlap = float(np.prod(per_axis) ** (1.0 / 3.0))
    return mean_overlap, volume_overlap, float(per_axis[2])


def part_stats(points: np.ndarray, body_part: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    _center, _vals, _axes, proj = pca_frame(points)
    centers: dict[int, np.ndarray] = {}
    extents: dict[int, np.ndarray] = {}
    for part in sorted(int(x) for x in np.unique(body_part)):
        mask = body_part == part
        if int(mask.sum()) < 50:
            continue
        part_proj = proj[mask]
        centers[part] = np.median(part_proj, axis=0)
        extents[part] = np.maximum(np.ptp(part_proj, axis=0), 1e-6)
    return centers, extents, proj, np.maximum(np.ptp(proj, axis=0), 1e-9)


def adjacency_collision_metric_v4(points: np.ndarray, body_part: np.ndarray | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geom = anti_billboard_metric_v2(points, body_part)
    if body_part is None or len(body_part) != len(points):
        return (
            {
                **geom,
                "adjacency_collision_score_v4": 0.0,
                "adjacency_collision_fail_v4": True,
                "part_presence_score": 0.0,
                "valid_contact_score": 0.0,
                "invalid_overlap_penalty": 1.0,
                "adjacent_overmerge_penalty": 1.0,
                "distant_separation_score": 0.0,
                "semantic_order_score": 0.0,
                "combined_topology_volume_score_v4": 0.0,
                "combined_fail_v4": True,
            },
            [],
        )

    pts = np.asarray(points, dtype=np.float64)
    body = np.asarray(body_part).astype(int)
    centers, extents, proj, ranges = part_stats(pts, body)
    part_count = len(centers)
    part_presence = min(part_count / 8.0, 1.0)
    valid_contacts: list[float] = []
    adjacent_overmerge: list[float] = []
    invalid_penalties: list[float] = []
    distant_separation: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    scale = float(np.linalg.norm(ranges[:2]) + 1e-9)

    for a, b in itertools.combinations(sorted(centers), 2):
        pair = tuple(sorted((a, b)))
        ca, cb = centers[a], centers[b]
        ea, eb = extents[a], extents[b]
        mean_overlap, volume_overlap, thin_overlap = bbox_overlap_ratio(ca, ea, cb, eb)
        center_dist = float(np.linalg.norm((ca - cb) / ranges))
        center_dist_xy = float(np.linalg.norm(ca[:2] - cb[:2]) / scale)
        is_adjacent = pair in ADJACENT_EDGES
        distant_weight = DISTANT_PAIR_WEIGHTS.get(pair, 0.65 if not is_adjacent else 0.0)
        if is_adjacent:
            # Adjacent parts should touch, but not collapse into the same bbox.
            contact = float(np.exp(-center_dist_xy / 0.22) * np.clip(mean_overlap / 0.30, 0.0, 1.0))
            overmerge = float(max(0.0, volume_overlap - 0.72) / 0.28)
            valid_contacts.append(contact)
            adjacent_overmerge.append(np.clip(overmerge, 0.0, 1.0))
            invalid_penalty = 0.0
            separation = contact
        else:
            invalid_penalty = float(np.clip((volume_overlap - 0.16) / 0.58, 0.0, 1.0) * distant_weight)
            # A pair may overlap in one axis, but should have a visible center margin
            # somewhere in the part-local frame.
            separation = float(np.clip(center_dist / 0.72, 0.0, 1.0))
            invalid_penalties.append(min(invalid_penalty, 1.0))
            distant_separation.append(separation)
        pair_rows.append(
            {
                "part_a": a,
                "part_b": b,
                "part_a_name": PART_NAMES.get(a, str(a)),
                "part_b_name": PART_NAMES.get(b, str(b)),
                "is_adjacent": is_adjacent,
                "mean_bbox_overlap": mean_overlap,
                "volume_bbox_overlap": volume_overlap,
                "thin_axis_overlap": thin_overlap,
                "normalized_center_distance": center_dist,
                "normalized_center_distance_xy": center_dist_xy,
                "invalid_overlap_penalty": min(invalid_penalty, 1.0),
                "pair_separation_or_contact_score": separation,
            }
        )

    valid_contact_score = float(np.mean(valid_contacts)) if valid_contacts else 0.0
    adjacent_overmerge_penalty = float(np.mean(adjacent_overmerge)) if adjacent_overmerge else 1.0
    invalid_overlap_penalty = float(np.mean(invalid_penalties)) if invalid_penalties else 1.0
    distant_separation_score = float(np.mean(distant_separation)) if distant_separation else 0.0

    expected_order = [0, 1, 4, 6]
    order_scores = []
    for a, b in zip(expected_order, expected_order[1:]):
        if a in centers and b in centers:
            order_scores.append(float(centers[b][0] > centers[a][0]))
    semantic_order_score = float(np.mean(order_scores)) if order_scores else 0.0

    adjacency_collision_score = (
        0.24 * part_presence
        + 0.22 * valid_contact_score
        + 0.22 * max(0.0, 1.0 - invalid_overlap_penalty)
        + 0.14 * max(0.0, 1.0 - adjacent_overmerge_penalty)
        + 0.10 * distant_separation_score
        + 0.08 * semantic_order_score
    )
    combined = 0.54 * float(geom["anti_billboard_score_v2"]) + 0.46 * adjacency_collision_score
    adj_fail = bool(
        adjacency_collision_score < 0.58
        or part_presence < 0.95
        or valid_contact_score < 0.18
        or invalid_overlap_penalty > 0.58
        or adjacent_overmerge_penalty > 0.64
    )
    combined_fail = bool(geom["billboard_fail_v2"] or adj_fail or combined < 0.62)
    return (
        {
            **geom,
            "adjacency_collision_score_v4": float(adjacency_collision_score),
            "adjacency_collision_fail_v4": adj_fail,
            "part_presence_score": float(part_presence),
            "valid_contact_score": float(valid_contact_score),
            "invalid_overlap_penalty": float(invalid_overlap_penalty),
            "adjacent_overmerge_penalty": float(adjacent_overmerge_penalty),
            "distant_separation_score": float(distant_separation_score),
            "semantic_order_score": float(semantic_order_score),
            "combined_topology_volume_score_v4": float(combined),
            "combined_fail_v4": combined_fail,
        },
        pair_rows,
    )


def render_board(rows: list[dict[str, Any]]) -> None:
    cases = sorted({r["case"] for r in rows})
    configs = [V173_CONFIG, V179_CONFIG, V177_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
    width = 1320
    row_h = 22
    height = 70 + len(cases) * (34 + row_h * len(configs))
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((18, 16), "V180 adjacency-aware collision metric v4: valid body adjacency vs distant part overlap; diagnostic only", fill=(0, 0, 0))
    y = 52
    for case in cases:
        draw.text((18, y), case, fill=(0, 0, 0))
        y += 24
        by_cfg = {r["config"]: r for r in rows if r["case"] == case}
        for cfg in configs:
            if cfg not in by_cfg:
                continue
            row = by_cfg[cfg]
            draw.text((34, y), cfg[:34], fill=(35, 35, 35))
            metrics = [
                ("geom", float(row["anti_billboard_score_v2"]), (79, 119, 92)),
                ("adj", float(row["adjacency_collision_score_v4"]), (92, 96, 160)),
                ("comb", float(row["combined_topology_volume_score_v4"]), (184, 105, 58)),
                ("badOverlap", float(row["invalid_overlap_penalty"]), (152, 65, 65)),
            ]
            x = 330
            for label, val, color in metrics:
                draw.text((x, y), label, fill=(35, 35, 35))
                draw.rectangle((x + 78, y + 3, x + 78 + int(165 * np.clip(val, 0, 1)), y + 14), fill=color)
                draw.text((x + 250, y), f"{val:.3f}", fill=(35, 35, 35))
                x += 245
            y += row_h
        y += 10
    ensure(BOARDS)
    im.save(BOARDS / "V18000000000000000000_adjacency_collision_metric_v4_board.png")


def write_route(decision: dict[str, Any]) -> None:
    route = f"""# V180 Adjacency-Aware Collision Route

Created: {decision["created_at"]}

## Decision

Status: `{decision["status"]}`

V180 is diagnostic only. It does not make the route mentor-ready because the hard mentor visual gate still requires a human-main full-scene RGB point cloud that visibly beats VGGT baseline and hard controls.

## What Changed

- V178 treated part bbox overlap too coarsely.
- V179 tried direct part separation and harmed at least one case.
- V180 separates valid adjacent contact from invalid distant part overlap.
- Adjacent edges, such as head-torso, torso-arm, torso-leg, and leg-foot, are allowed to touch.
- Semantically distant pairs, such as head-foot, arm-leg, left-right endpoints, and torso-foot, receive the main collision penalty.

## Current Result

- Mentor ready: `false`
- External hard block: `false`
- Failure count: `{len(decision["failures"])}`

## Next Route

If V180 still fails, the next route is not a viewer/thickness repair. It should train an adjacency-aware topology loss:

1. Preserve V173 multi-shell output as the current best source candidate.
2. Add valid-edge contact loss for body topology adjacency.
3. Add invalid-pair separation loss for semantically distant part overlaps.
4. Add cross-part occupancy exclusion for head/foot, arm/leg, left/right endpoints, and torso/foot.
5. Keep projection as auxiliary only.
6. Fail closed unless V138-style full-scene and same-scene controls visibly pass.
"""
    (REPORTS / "V18000000000000000000_adjacency_collision_route.md").write_text(route, encoding="utf-8")


def main() -> int:
    created_at = now()
    cases = sorted(p.parent.parent.name for p in V173_ROOT.glob(f"*/{V173_CONFIG}/predictions.npz"))
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        case_rows: dict[str, dict[str, Any]] = {}
        for cfg in EVAL_CONFIGS:
            path = prediction_path(case, cfg)
            if not path.exists():
                continue
            pred = load_npz(path)
            body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
            metric, pairs = adjacency_collision_metric_v4(np.asarray(pred["human_points"], dtype=np.float32), body)
            row = {"case": case, "config": cfg, **metric}
            rows.append(row)
            case_rows[cfg] = row
            for pair in pairs:
                pair_rows.append({"case": case, "config": cfg, **pair})
        true = case_rows.get(V173_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_v173_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "v173_true_combined_fail_v4", "true_score": true_score})
        if float(true["invalid_overlap_penalty"]) > 0.58:
            failures.append({"case": case, "reason": "invalid_distant_overlap_too_high", "penalty": float(true["invalid_overlap_penalty"])})
        for cfg in ["real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", V177_CONFIG, V179_CONFIG]:
            if cfg not in case_rows:
                continue
            control_score = float(case_rows[cfg]["combined_topology_volume_score_v4"])
            if control_score >= true_score * 0.96:
                failures.append(
                    {
                        "case": case,
                        "reason": "control_or_diagnostic_close_or_better_v4",
                        "control": cfg,
                        "true_score": true_score,
                        "control_score": control_score,
                    }
                )
    write_csv(REPORTS / "V18000000000000000000_adjacency_collision_metric_v4_scores.csv", rows)
    write_csv(REPORTS / "V18000000000000000000_adjacency_collision_pair_audit.csv", pair_rows)
    render_board(rows)
    decision = {
        "created_at": created_at,
        "status": "V18000_ADJACENCY_AWARE_COLLISION_METRIC_FAIL_CLOSED_CONTINUE" if failures else "V18000_ADJACENCY_AWARE_COLLISION_PRECHECK_PASS_REQUIRES_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V18000000000000000000_adjacency_collision_metric_v4_scores.csv"),
        "pair_audit_csv": str(REPORTS / "V18000000000000000000_adjacency_collision_pair_audit.csv"),
        "board": str(BOARDS / "V18000000000000000000_adjacency_collision_metric_v4_board.png"),
        "summary": "V180 makes collision analysis adjacency-aware. It distinguishes valid neighboring body contact from invalid distant part overlap, but remains a diagnostic metric and cannot replace mentor visual evidence.",
    }
    write_json(REPORTS / "V18000000000000000000_adjacency_collision_decision.json", decision)
    write_route(decision)
    print(json.dumps({"status": "V18000_DONE", "failure_count": len(failures), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
