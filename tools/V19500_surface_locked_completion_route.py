from __future__ import annotations

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


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created_at = now()
    v194 = read_json(REPORTS / "V19400000000000000000_visible_surface_infill_decision.json")
    decision = {
        "created_at": created_at,
        "status": "V19500_SURFACE_LOCKED_COMPLETION_ROUTE_REQUIRED",
        "mentor_ready": False,
        "external_hard_block": False,
        "v194_status": v194.get("status"),
        "v194_failure_count": len(v194.get("failures", [])),
        "v194_board": str(BOARDS / "V19400000000000000000_visible_surface_infill_board.png"),
        "v194_turntable": str(BOARDS / "V19400000000000000000_visible_surface_infill_turntable_cross_section.png"),
        "route_decision": "V194 preserved the visible body but added noisy infill. Continue with surface-locked sparse topology completion: small, connected, part-local infill bands tied to baseline surface and SMPL adjacency, not large free clouds.",
        "summary": "Visible-surface preservation improves mentor readability but the infill is noisy and metrics/controls fail. The next repair must constrain infill count, distance, and part continuity instead of adding broad shell clouds.",
    }
    write_json(REPORTS / "V19500000000000000000_surface_locked_completion_decision.json", decision)
    route = f"""# V19500000000000000000 Surface-Locked Sparse Topology Completion Route

Created: {created_at}

## Conclusion

V194 is the first route that keeps the coherent visible VGGT human surface, but it is not mentor-ready.

The main figure is more human-readable than full shell replacement, yet the infill is a noisy cloud and hard controls / topology metrics still fail.

## Next Route

V196 should perform surface-locked sparse topology completion:

1. Keep 70-85% of visible VGGT baseline points unchanged.
2. Add only 8k-18k infill points, not broad 60k replacement clouds.
3. Infill must be connected to nearby baseline surface and SMPL body-part adjacency.
4. Penalize floating infill and points farther than a local radius from the visible surface.
5. Use part-specific infill bands for shoulder/neck, clothing boundary, arm endpoint, leg/foot, and back/side shell.
6. Generate mentor full-scene RGB board and same-scene controls with real environment.

Forbidden:

- whole-body shell replacement;
- free infill clouds;
- metric-only or render-only pass;
- claiming face details.
"""
    ensure(GOALS)
    (GOALS / "V19500000000000000000_auto_evolved_surface_locked_completion_route.md").write_text(route, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
