from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPO = Path(r"D:\vggt\vggt-feature-adapter")
REPORTS = AUX / "reports"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    decision = json.loads((REPORTS / "V37000000000_visual_first_eval.json").read_text(encoding="utf-8"))
    payload = {
        "created_utc": now(),
        "route_file": str(REPO / "docs" / "goals" / "V40000000000_auto_evolved_route.md"),
        "trigger": decision,
        "auto_evolution_executed": True,
        "next_route": "V401 full-resolution camera-bound point-transformer/differentiable-renderer implementation",
        "not_final": True,
        "reason": "Mentor-ready gate not satisfied; route exhausted is not a return state.",
    }
    write_json(REPORTS / "V40000000000_auto_evolved_route_generation.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
