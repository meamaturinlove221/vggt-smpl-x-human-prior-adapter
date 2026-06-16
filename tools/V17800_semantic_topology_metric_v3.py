from __future__ import annotations

import csv
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
TRUE_CONFIG = "multishell_topology_decoder_true"
V177_CONFIG = "front_back_occupancy_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]

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
    if config == TRUE_CONFIG:
        return V173_ROOT / case / config / "predictions.npz"
    if config == V177_CONFIG:
        return V177_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def semantic_topology_metric(points: np.ndarray, body_part: np.ndarray | None) -> dict[str, Any]:
    if body_part is None or len(body_part) != len(points):
        return {
            "semantic_topology_score_v3": 0.0,
            "semantic_topology_fail_v3": True,
            "part_count": 0,
            "part_order_score": 0.0,
            "adjacency_score": 0.0,
            "endpoint_score": 0.0,
            "collision_penalty": 1.0,
        }
    pts = np.asarray(points, dtype=np.float64)
    body = np.asarray(body_part).astype(int)
    _center, _vals, axes, proj = pca_frame(pts)
    main = proj[:, 0]
    thin = proj[:, 2]
    unique = sorted(int(x) for x in np.unique(body))
    centers: dict[int, np.ndarray] = {}
    extents: dict[int, np.ndarray] = {}
    for p in unique:
        m = body == p
        if int(m.sum()) < 50:
            continue
        centers[p] = np.median(proj[m], axis=0)
        extents[p] = np.ptp(proj[m], axis=0)
    part_count = len(centers)
    # Validity is intentionally weakly supervised: it does not assume front-face
    # visibility, but it requires parts to form a plausible chain instead of a
    # dense semantic scramble.
    expected_edges = [(0, 1), (1, 4), (1, 5), (4, 6), (5, 7), (1, 2), (1, 3)]
    edge_scores = []
    for a, b in expected_edges:
        if a not in centers or b not in centers:
            continue
        dist = float(np.linalg.norm(centers[a][:2] - centers[b][:2]))
        scale = float(np.median([np.linalg.norm(v[:2]) for v in extents.values()]) + 1e-6)
        edge_scores.append(float(np.exp(-dist / max(scale * 1.15, 1e-6))))
    adjacency_score = float(np.mean(edge_scores)) if edge_scores else 0.0
    # Part ordering should not collapse all parts to the same main-axis bin.
    order_values = np.array([centers[p][0] for p in sorted(centers)])
    order_spread = float(np.ptp(order_values) / max(np.ptp(main), 1e-9)) if len(order_values) >= 2 else 0.0
    part_order_score = float(np.clip(order_spread / 0.42, 0.0, 1.0))
    # Endpoint parts should have visible extension without becoming giant sheets.
    endpoint_parts = [p for p in [2, 3, 6, 7] if p in extents]
    endpoint_vals = []
    for p in endpoint_parts:
        e = extents[p]
        endpoint_vals.append(float(np.clip(e[0] / max(np.ptp(main), 1e-9) / 0.22, 0, 1)) * float(np.clip(e[2] / max(np.ptp(thin), 1e-9) / 0.18, 0, 1)))
    endpoint_score = float(np.mean(endpoint_vals)) if endpoint_vals else 0.0
    # Penalize heavy overlap between semantically distant parts in PCA bounding boxes.
    collisions = []
    distant = [(0, 6), (0, 7), (2, 6), (3, 7), (2, 3), (6, 7)]
    for a, b in distant:
        if a not in centers or b not in centers:
            continue
        ca, cb = centers[a], centers[b]
        ea, eb = extents[a] * 0.5, extents[b] * 0.5
        overlap = np.maximum(0.0, np.minimum(ca + ea, cb + eb) - np.maximum(ca - ea, cb - eb))
        denom = np.maximum(np.minimum(ea * 2, eb * 2), 1e-9)
        collisions.append(float(np.mean(overlap / denom)))
    collision_penalty = float(np.mean(collisions)) if collisions else 0.35
    score = (
        0.30 * min(part_count / 8.0, 1.0)
        + 0.25 * adjacency_score
        + 0.20 * part_order_score
        + 0.15 * endpoint_score
        + 0.10 * max(0.0, 1.0 - collision_penalty)
    )
    fail = bool(score < 0.48 or part_count < 6 or collision_penalty > 0.62)
    return {
        "semantic_topology_score_v3": float(score),
        "semantic_topology_fail_v3": fail,
        "part_count": int(part_count),
        "part_order_score": float(part_order_score),
        "adjacency_score": float(adjacency_score),
        "endpoint_score": float(endpoint_score),
        "collision_penalty": float(collision_penalty),
    }


def combined_metric(points: np.ndarray, body_part: np.ndarray | None) -> dict[str, Any]:
    geom = anti_billboard_metric_v2(points, body_part)
    sem = semantic_topology_metric(points, body_part)
    score = 0.62 * float(geom["anti_billboard_score_v2"]) + 0.38 * float(sem["semantic_topology_score_v3"])
    fail = bool(geom["billboard_fail_v2"] or sem["semantic_topology_fail_v3"] or score < 0.57)
    return {**geom, **sem, "combined_topology_volume_score_v3": float(score), "combined_fail_v3": fail}


def render_bar(rows: list[dict[str, Any]]) -> None:
    cases = sorted({r["case"] for r in rows})
    configs = [TRUE_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", V177_CONFIG]
    size = (1200, 260 + 180 * len(cases))
    im = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((20, 16), "V178 semantic topology metric v3: geometry + semantic validity; diagnostic, not final mentor pass", fill=(0, 0, 0))
    y = 54
    for case in cases:
        draw.text((20, y), case, fill=(0, 0, 0))
        y += 24
        case_rows = {r["config"]: r for r in rows if r["case"] == case}
        for cfg in configs:
            if cfg not in case_rows:
                continue
            r = case_rows[cfg]
            geom = float(r["anti_billboard_score_v2"])
            sem = float(r["semantic_topology_score_v3"])
            comb = float(r["combined_topology_volume_score_v3"])
            x0 = 270
            draw.text((35, y), cfg[:28], fill=(30, 30, 30))
            for label, val, color, off in [
                ("geom", geom, (78, 119, 92), 0),
                ("sem", sem, (118, 92, 153), 310),
                ("comb", comb, (184, 105, 58), 620),
            ]:
                draw.text((x0 + off, y), label, fill=(40, 40, 40))
                draw.rectangle((x0 + off + 48, y + 2, x0 + off + 48 + int(220 * val), y + 14), fill=color)
                draw.text((x0 + off + 274, y), f"{val:.3f}", fill=(40, 40, 40))
            y += 21
        y += 18
    ensure(BOARDS)
    im.save(BOARDS / "V17800000000000000000_semantic_topology_metric_v3_board.png")


def main() -> int:
    created_at = now()
    cases = sorted({p.parent.parent.name for p in V173_ROOT.glob(f"*/{TRUE_CONFIG}/predictions.npz")})
    configs = [TRUE_CONFIG, V177_CONFIG, *CONTROL_CONFIGS]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        case_rows: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            path = prediction_path(case, cfg)
            if not path.exists():
                continue
            pred = load_npz(path)
            body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
            met = combined_metric(np.asarray(pred["human_points"], dtype=np.float32), body)
            row = {"case": case, "config": cfg, **met}
            rows.append(row)
            case_rows[cfg] = row
        true = case_rows.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_v173_true"})
            continue
        ts = float(true["combined_topology_volume_score_v3"])
        if bool(true["combined_fail_v3"]):
            failures.append({"case": case, "reason": "true_combined_fail_v3", "true_score": ts})
        for cfg in ["real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
            if cfg not in case_rows:
                continue
            cs = float(case_rows[cfg]["combined_topology_volume_score_v3"])
            if cs >= ts * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better_v3", "control": cfg, "true_score": ts, "control_score": cs})
    write_csv(REPORTS / "V17800000000000000000_semantic_topology_metric_v3_scores.csv", rows)
    render_bar(rows)
    write_json(
        REPORTS / "V17800000000000000000_semantic_topology_metric_v3_decision.json",
        {
            "created_at": created_at,
            "status": "V17800_SEMANTIC_TOPOLOGY_METRIC_V3_FAIL_CLOSED_CONTINUE" if failures else "V17800_SEMANTIC_TOPOLOGY_METRIC_V3_PRECHECK_PASS_REQUIRES_VISUAL",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "score_csv": str(REPORTS / "V17800000000000000000_semantic_topology_metric_v3_scores.csv"),
            "board": str(BOARDS / "V17800000000000000000_semantic_topology_metric_v3_board.png"),
            "summary": "Metric v3 adds semantic topology validity to avoid rewarding semantically invalid same-topology/shuffled geometry. It remains auxiliary and cannot replace mentor visual gates.",
        },
    )
    print(json.dumps({"status": "V17800_DONE", "failure_count": len(failures), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
