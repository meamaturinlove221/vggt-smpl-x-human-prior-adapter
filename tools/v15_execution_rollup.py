from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_source(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": data.get("status"),
        "decision": data.get("decision"),
        "strict_candidate_passes": data.get("strict_candidate_passes"),
        "strict_teacher_passes": data.get("strict_teacher_passes"),
        "blocker_count": len(data.get("blockers", []) or []),
    }


def artifact_path(path: Path) -> str:
    return str(path.resolve())


def merge_blockers(*items: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, list):
            continue
        for value in item:
            text = str(value)
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write V15 U/D execution rollup without candidate package writes.")
    parser.add_argument("--source-graph", type=Path, default=REPORTS / "20260508_v15_source_graph.json")
    parser.add_argument("--router-queue", type=Path, default=REPORTS / "20260508_v15_dline_router_queue.json")
    parser.add_argument("--legality-precheck", type=Path, default=REPORTS / "20260508_v15_unified_region_legality_precheck.json")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_execution_rollup.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_execution_rollup.md")
    args = parser.parse_args()

    source_graph = read_summary(args.source_graph)
    router = read_summary(args.router_queue)
    legality = read_summary(args.legality_precheck)
    strict_candidate_passes = max(
        int_value(source_graph.get("strict_candidate_passes")),
        int_value(router.get("strict_candidate_passes")),
        int_value(legality.get("strict_candidate_passes")),
    )
    strict_teacher_passes = max(
        int_value(source_graph.get("strict_teacher_passes")),
        int_value(router.get("strict_teacher_passes")),
        int_value(legality.get("strict_teacher_passes")),
    )
    legal = legality.get("status") == "v15_unified_regions_promotion_eligible"
    return_allowed = bool(router.get("return_allowed"))
    candidate_package_built = bool(legality.get("candidate_package_built"))
    formal_cloud_unblocked = bool(router.get("formal_cloud_unblocked")) and legal and return_allowed

    status = "v15_terminal_state_no_strict_pass"
    if legal and return_allowed:
        status = "v15_ready_for_separate_dline_package_transaction"
    if strict_candidate_passes > 0 or strict_teacher_passes > 0 or candidate_package_built:
        status = "v15_inconsistent_strict_pass_claim_detected"

    blockers = merge_blockers(source_graph.get("blockers"), router.get("blockers"), legality.get("blockers"))
    if not legality:
        blockers.append("V15 unified region legality precheck has not been run.")
    if not legal:
        blockers.append("V15 unified legality precheck did not mark all required regions promotion eligible.")
    if candidate_package_built:
        blockers.append("Legality report claims a candidate package was built; U/D ownership forbids that in blocked state.")
    if strict_candidate_passes > 0 or strict_teacher_passes > 0:
        blockers.append("A strict pass count is nonzero; verify D-line transaction before any return.")

    next_work_queue = router.get("next_work_queue", [])
    pending_queue = [item for item in next_work_queue if isinstance(item, dict) and item.get("status") != "complete"]

    summary = {
        "task": "v15_execution_rollup",
        "created_utc": utc_now(),
        "status": status,
        "strict_candidate_passes": 0 if status != "v15_inconsistent_strict_pass_claim_detected" else strict_candidate_passes,
        "strict_teacher_passes": 0 if status != "v15_inconsistent_strict_pass_claim_detected" else strict_teacher_passes,
        "formal_cloud_unblocked": formal_cloud_unblocked,
        "candidate_package_built": candidate_package_built,
        "candidate_package_path": legality.get("candidate_package_path"),
        "registry_entry_path": legality.get("registry_entry_path"),
        "promotion_status": router.get("dline_status"),
        "source_graph_status": source_graph.get("status"),
        "router_status": router.get("status"),
        "legality_status": legality.get("status"),
        "promotion_eligible_count": source_graph.get("promotion_eligible_count", 0),
        "promotion_eligible_regions": source_graph.get("promotion_eligible_regions", []),
        "missing_required_regions": legality.get("missing_required_regions", []),
        "reason_classes": router.get("reason_classes", []),
        "pending_queue_count": len(pending_queue),
        "next_work_queue": next_work_queue,
        "artifacts": {
            "source_graph": artifact_path(args.source_graph),
            "router_queue": artifact_path(args.router_queue),
            "legality_precheck": artifact_path(args.legality_precheck),
        },
        "sources": {
            "source_graph": compact_source(source_graph),
            "router_queue": compact_source(router),
            "legality_precheck": compact_source(legality),
        },
        "blockers": blockers,
        "decision": (
            "V15 U/D rollup is ready for a separate D-line package transaction."
            if status == "v15_ready_for_separate_dline_package_transaction"
            else "V15 remains blocked. No candidate package, registry entry, or strict pass was written."
        ),
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V15 Execution Rollup", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=True))
    return 0 if status == "v15_ready_for_separate_dline_package_transaction" else 2


if __name__ == "__main__":
    raise SystemExit(main())
