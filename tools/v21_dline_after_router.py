from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_JSON = REPORTS / "20260508_v21_dline_after_router.json"
DEFAULT_MD = REPORTS / "20260508_v21_dline_after_router.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def status_for(path: Path) -> str:
    return str(read_json(path).get("status") or "DONE_FAIL_ROUTED")


def main() -> int:
    parser = argparse.ArgumentParser(description="V21 D-line after-router.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    inputs = {
        "v16_router": REPORTS / "20260508_v16_dline_failure_router.json",
        "v20": REPORTS / "20260508_v20_final_promotion_transaction.json",
        "v17": REPORTS / "20260508_v17_smplx_residual_surface_optimizer.json",
        "v18": REPORTS / "20260508_v18_residual_teacher_distillation_case.json",
        "v19": REPORTS / "20260508_v19_temporal_canonical_residual_teacher.json",
    }
    statuses = {name: status_for(path) for name, path in inputs.items()}
    routes = []
    if "negative" in statuses["v16_router"].lower() or "routed" in statuses["v16_router"].lower():
        routes.append({"branch": "V22", "reason": "V16 microfit weak or D-line routed"})
    if "head" in read_json(inputs["v17"]).get("decision", "").lower() or "research" in statuses["v17"].lower():
        routes.append({"branch": "V23", "reason": "V17 needs full region repair/audit before teacher use"})
    if "research" in statuses["v18"].lower():
        routes.append({"branch": "V24", "reason": "V18 teacher is research-only and must be regenerated/audited from V23"})
    if "predictions_missing" in statuses["v19"]:
        routes.append({"branch": "V25", "reason": "V19 adjacent predictions missing"})
    if "fail_closed" in statuses["v20"]:
        routes.append({"branch": "V28_HOLD", "reason": "Do not rerun promotion until V22-V27 complete"})
    summary = {
        "task": "v21_dline_after_router",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DONE_FAIL_ROUTED",
        "source_statuses": statuses,
        "routes": routes,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_package_write": True,
        "no_registry_write": True,
    }
    write_json(args.output_json, summary)
    lines = ["# V21 D-Line After Router", "", "Status: `DONE_FAIL_ROUTED`", "", "## Routes", ""]
    lines.extend([f"- {r['branch']}: {r['reason']}" for r in routes] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE_FAIL_ROUTED", "json": str(args.output_json), "route_count": len(routes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
