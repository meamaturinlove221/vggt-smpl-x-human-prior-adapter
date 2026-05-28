from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"
BOARDS = REPO / "boards"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created_at = now()
    v192_decision_path = REPORTS / "V19200000000000000000_upright_pose_frame_decision.json"
    v192_manifest_path = REPORTS / "V19200000000000000000_upright_pose_frame_training_manifest.csv"
    v192_scores_path = REPORTS / "V19200000000000000000_upright_pose_frame_scores.csv"
    missing = [str(p) for p in [v192_decision_path, v192_manifest_path, v192_scores_path] if not p.exists()]
    if missing:
        write_json(
            REPORTS / "V19300000000000000000_visible_surface_preservation_decision.json",
            {
                "created_at": created_at,
                "status": "V19300_MISSING_V192_INPUTS",
                "mentor_ready": False,
                "external_hard_block": False,
                "missing": missing,
            },
        )
        return 1
    decision = read_json(v192_decision_path)
    manifest = read_csv(v192_manifest_path)
    scores = read_csv(v192_scores_path)
    true_rows = [r for r in scores if r.get("config") == "upright_pose_frame_true"]
    active_completeness = []
    for row in manifest:
        hist = json.loads(row.get("history_json") or "[]")
        final = hist[-1] if hist else {}
        active_completeness.append(float(final.get("completeness", 0.0)))
    baseline_rows = [r for r in scores if r.get("config") == "real_vggt_baseline_only"]
    true_fail = [r for r in true_rows if r.get("combined_fail_v4", "").lower() == "true"]
    result = {
        "created_at": created_at,
        "status": "V19300_VISIBLE_SURFACE_PRESERVATION_ROUTE_REQUIRED",
        "mentor_ready": False,
        "external_hard_block": False,
        "v192_status": decision.get("status"),
        "v192_failure_count": len(decision.get("failures", [])),
        "v192_true_combined_fail_count": len(true_fail),
        "v192_final_completeness_values": active_completeness,
        "baseline_rows_present": len(baseline_rows),
        "route_decision": "Stop replacing the visible human surface with free shell clouds. Preserve the VGGT visible RGB surface as the front layer and train topology-volume infill only for hidden/back/weak regions, with a mask that forbids degrading source-visible pixels.",
        "board": str(BOARDS / "V19200000000000000000_upright_pose_frame_board.png"),
        "turntable": str(BOARDS / "V19200000000000000000_upright_pose_frame_turntable_cross_section.png"),
        "summary": "V192 still fails because all learned shell routes replace a coherent visible baseline with torn topology-volume clouds. The next route must be visible-surface-preserving infill, not whole-body shell replacement.",
    }
    write_json(REPORTS / "V19300000000000000000_visible_surface_preservation_decision.json", result)
    route = f"""# V19300000000000000000 Visible Surface Preservation + Topology Infill Route

Created: {created_at}

## Conclusion

V192 ran on Modal A10 and used an upright body-local frame, but it remains fail-closed.

The key visual fact is that the VGGT baseline visible surface is more coherent than the learned shell outputs. V187/V190/V192 keep replacing too much of the visible human surface, producing torn point-cloud shells.

## New Route

V194 must preserve the source-visible VGGT RGB surface as the front visible layer and train topology-volume infill only in weak / hidden / back / disconnected regions.

Required changes:

1. Build a no-degrade visible-surface mask from V950 confidence, source visibility, and V536 no-change regions.
2. Keep VGGT baseline RGB points in visible high-confidence areas.
3. Decode SMPL-conditioned topology infill only for back shell, side shell, limb continuity, and weak regions.
4. Blend in infill without replacing the coherent visible front surface.
5. Compare against baseline, V186, V187, V190, V192, same-topology, shuffled, and thickness-only controls.
6. Mentor main evidence remains full-scene RGB point cloud with partial real environment.

Forbidden:

- whole-body shell replacement;
- render-only pass;
- metric-only pass;
- contour-only local detail;
- visual failure as external hard block.
"""
    ensure(GOALS)
    (GOALS / "V19300000000000000000_auto_evolved_visible_surface_preservation_route.md").write_text(route, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
