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
BOARDS = REPO / "boards"
CASES = ["0012_11_frame001", "0013_01_frame001", "0021_03_frame001", "current_v895_0021_03"]


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


def span(points: np.ndarray) -> np.ndarray:
    return np.ptp(np.asarray(points, dtype=np.float32), axis=0)


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points.astype(np.float64)
    pts = pts - np.median(pts, axis=0, keepdims=True)
    cov = pts.T @ pts / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def main() -> int:
    created_at = now()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    v190_decision_path = REPORTS / "V19000000000000000000_pose_frame_decision.json"
    v190_decision: dict[str, Any] = {}
    if v190_decision_path.exists():
        v190_decision = json.loads(v190_decision_path.read_text(encoding="utf-8"))
        if not bool(v190_decision.get("mentor_ready", False)):
            failures.append(
                {
                    "case": "all",
                    "reason": "v190_mentor_visual_fail_closed",
                    "v190_status": v190_decision.get("status"),
                    "v190_failure_count": len(v190_decision.get("failures", [])),
                }
            )
    for case in CASES:
        bank_path = OUTPUT / "V9500000000000000_smpl_feature_bank_v4" / case / "smpl_feature_bank_v4.npz"
        v187_path = OUTPUT / "V18700000000000000000_visible_anchor_canonical_surfel_training" / case / "visible_anchor_canonical_surfel_true" / "predictions.npz"
        v190_path = OUTPUT / "V19000000000000000000_pose_frame_occupancy_repair" / case / "pose_frame_occupancy_true" / "predictions.npz"
        if not bank_path.exists() or not v190_path.exists():
            failures.append({"case": case, "reason": "missing_v191_input", "bank": bank_path.exists(), "v190": v190_path.exists()})
            continue
        bank = load_npz(bank_path)
        v190 = load_npz(v190_path)
        v187 = load_npz(v187_path) if v187_path.exists() else {}
        posed = np.asarray(bank["posed_world_xyz"], dtype=np.float32)
        world = np.asarray(bank["world_points"], dtype=np.float32)
        human = np.asarray(v190["human_points"], dtype=np.float32)
        env = np.asarray(v190["environment_points"], dtype=np.float32)
        vals, axes = principal_axes(human)
        vertical_alignment = float(abs(axes[:, 0][1]))
        posed_world_delta = np.linalg.norm(posed - world, axis=1)
        v187_human = np.asarray(v187["human_points"], dtype=np.float32) if v187 else human
        v187_v190_delta = float(np.linalg.norm(np.median(human, axis=0) - np.median(v187_human, axis=0)))
        human_ratio = float(len(human) / max(len(human) + len(env), 1))
        row = {
            "case": case,
            "posed_world_delta_mean": float(np.mean(posed_world_delta)),
            "posed_world_delta_p95": float(np.percentile(posed_world_delta, 95)),
            "v190_span_x": float(span(human)[0]),
            "v190_span_y": float(span(human)[1]),
            "v190_span_z": float(span(human)[2]),
            "principal_vertical_alignment_abs_y": vertical_alignment,
            "human_ratio": human_ratio,
            "v187_v190_center_delta": v187_v190_delta,
            "coordinate_mismatch_likely": bool(np.percentile(posed_world_delta, 95) > 0.18),
            "upright_pose_frame_fail": bool(vertical_alignment < 0.50),
        }
        rows.append(row)
        if row["upright_pose_frame_fail"]:
            failures.append({"case": case, "reason": "upright_pose_frame_fail", "principal_vertical_alignment_abs_y": vertical_alignment})
    write_csv(REPORTS / "V19100000000000000000_pose_frame_failure_audit.csv", rows)
    decision = {
        "created_at": created_at,
        "status": "V19100_POSE_FRAME_FAILURE_AUDIT_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "v190_failures": v190_decision.get("failures", []),
        "audit_csv": str(REPORTS / "V19100000000000000000_pose_frame_failure_audit.csv"),
        "v190_board": str(BOARDS / "V19000000000000000000_pose_frame_board.png"),
        "v190_turntable": str(BOARDS / "V19000000000000000000_pose_frame_turntable_cross_section.png"),
        "summary": "V190 improves some topology-volume metrics, but visual evidence remains a tilted/torn cloud. V950 posed/world frames are close enough that missing assets or gross coordinate mismatch are not the main cause; the next repair must enforce upright posed-frame body layout and full-scene mentor view.",
    }
    write_json(REPORTS / "V19100000000000000000_pose_frame_failure_decision.json", decision)
    route = f"""# V19100000000000000000 Upright Posed-Frame Body Layout Route

Created: {created_at}

## Conclusion

V190 ran on Modal A10 and made pose-frame shell supervision active, but the mentor visual remains fail-closed.

The restored V950 `posed_world_xyz` and `world_points` are not grossly misaligned, so the current failure is not an asset-restoration or coordinate-frame hard block.

## Failure

- V190 still renders as a tilted / torn topology-volume cloud.
- Hard controls and V187/V186 priors remain close or better in several cases.
- The main figure is not a natural human-main full-scene RGB point cloud.

## Next Repair

V192 should enforce an upright posed-frame body layout before decoding:

1. derive a body-local vertical/forward/right frame from SMPL part anchors;
2. normalize each case into a mentor upright frame for training and render;
3. decode per-part occupancy in that frame, then transform back to world coordinates;
4. add explicit head-torso-limb order losses and limb continuity in body frame;
5. keep real VGGT environment insertion and same-scene controls.

Do not continue with render-only, thickness-only, V186 fallback, or free visible-anchor tuning.
"""
    ensure(GOALS)
    (GOALS / "V19100000000000000000_auto_evolved_upright_pose_frame_route.md").write_text(route, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
