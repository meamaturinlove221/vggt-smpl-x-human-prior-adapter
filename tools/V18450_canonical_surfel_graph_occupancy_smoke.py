from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
REPORTS = REPO / "reports"

from models.v184_canonical_surfel_graph_occupancy_student import smoke_test  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    created_at = now()
    smoke = smoke_test()
    payload = {
        "created_at": created_at,
        "status": "V18450_CANONICAL_SURFEL_GRAPH_OCCUPANCY_SMOKE_READY_FOR_TRAINING" if smoke.get("grad_norm_positive") and smoke.get("forbidden_teacher_points_rejected") else "V18450_SMOKE_FAIL_REPAIR_REQUIRED",
        "mentor_ready": False,
        "external_hard_block": False,
        "smoke": smoke,
        "model": "models/v184_canonical_surfel_graph_occupancy_student.py",
        "summary": "Canonical surfel/graph occupancy architecture is scaffolded after point-anchor shell decoders failed. This is only a smoke/contract check, not mentor evidence.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "V18450000000000000000_canonical_surfel_graph_occupancy_smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
