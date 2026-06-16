from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


REQUIRED_REGIONS = ("body", "head", "face", "hair", "left_hand", "right_hand")
HAND_ALIASES = {"hand": ("left_hand", "right_hand")}
STRICT_PASS_STATUSES = {"promotion_written", "strict_pass_written", "strict_teacher_pass_written"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_regions(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = [str(value)]
    regions: set[str] = set()
    for item in raw:
        key = item.strip()
        if not key or key == "none":
            continue
        aliases = HAND_ALIASES.get(key)
        if aliases:
            regions.update(aliases)
        else:
            regions.add(key)
    return regions


def eligible_region_sources(source_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_region = {region: [] for region in REQUIRED_REGIONS}
    sources = source_graph.get("sources", [])
    if not isinstance(sources, list):
        return by_region
    for source in sources:
        if not isinstance(source, dict) or not bool(source.get("promotion_eligible")):
            continue
        for region in normalize_regions(source.get("ownership_region")):
            if region in by_region:
                by_region[region].append(
                    {
                        "name": source.get("name"),
                        "source_type": source.get("source_type"),
                        "status": source.get("status"),
                        "path": source.get("path"),
                        "reason": source.get("reason"),
                    }
                )
    return by_region


def forbidden_source_findings(source_graph: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for source in source_graph.get("sources", []):
        if not isinstance(source, dict):
            continue
        if not bool(source.get("promotion_eligible")):
            continue
        source_type = str(source.get("source_type", "")).lower()
        reason = str(source.get("reason", "")).lower()
        status = str(source.get("status", "")).lower()
        if "forbidden" in source_type or "proxy" in reason or "diagnostic" in source_type or "diagnostic" in status:
            findings.append(
                {
                    "name": source.get("name"),
                    "source_type": source.get("source_type"),
                    "status": source.get("status"),
                    "reason": source.get("reason"),
                    "path": source.get("path"),
                }
            )
    return findings


def strict_pass_claims(router: dict[str, Any]) -> list[str]:
    claims: list[str] = []
    status = str(router.get("status", ""))
    dline_status = str(router.get("dline_status", ""))
    if status in STRICT_PASS_STATUSES:
        claims.append(f"router.status={status}")
    if dline_status in STRICT_PASS_STATUSES:
        claims.append(f"router.dline_status={dline_status}")
    for key in ("strict_candidate_passes", "strict_teacher_passes"):
        try:
            count = int(router.get(key, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            claims.append(f"{key}={count}")
    return claims


def build_summary(source_graph: dict[str, Any], router: dict[str, Any]) -> dict[str, Any]:
    by_region = eligible_region_sources(source_graph)
    present = sorted(region for region, entries in by_region.items() if entries)
    missing = [region for region in REQUIRED_REGIONS if not by_region[region]]
    forbidden_findings = forbidden_source_findings(source_graph)
    pass_claims = strict_pass_claims(router)
    promotion_eligible_regions = sorted(normalize_regions(source_graph.get("promotion_eligible_regions")))

    gates = {
        "source_graph_present": bool(source_graph),
        "router_queue_present": bool(router),
        "all_required_regions_promotion_eligible": not missing,
        "no_forbidden_or_diagnostic_eligible_sources": not forbidden_findings,
        "no_strict_pass_claim_in_router": not pass_claims,
        "router_return_allowed": bool(router.get("return_allowed")) if router else False,
    }
    legal = bool(
        gates["source_graph_present"]
        and gates["router_queue_present"]
        and gates["all_required_regions_promotion_eligible"]
        and gates["no_forbidden_or_diagnostic_eligible_sources"]
        and not pass_claims
        and bool(router.get("return_allowed"))
    )
    blockers: list[str] = []
    if not gates["source_graph_present"]:
        blockers.append("V15 source graph report is missing or unreadable.")
    if not gates["router_queue_present"]:
        blockers.append("V15 D-line router queue report is missing or unreadable.")
    blockers.extend([f"Missing promotion-eligible ownership region: {region}." for region in missing])
    if forbidden_findings:
        blockers.append("Promotion-eligible source set contains forbidden/proxy/diagnostic sources.")
    blockers.extend([f"Router contains strict pass claim while legality is being checked: {claim}." for claim in pass_claims])
    if router and not bool(router.get("return_allowed")):
        blockers.append("D-line router return_allowed is false.")

    return {
        "task": "v15_unified_region_legality_precheck",
        "created_utc": utc_now(),
        "status": "v15_unified_regions_promotion_eligible" if legal else "v15_unified_regions_promotion_blocked",
        "candidate_package_built": False,
        "candidate_package_path": None,
        "registry_entry_path": None,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "required_regions": list(REQUIRED_REGIONS),
        "promotion_eligible_regions_from_source_graph": promotion_eligible_regions,
        "present_required_regions": present,
        "missing_required_regions": missing,
        "eligible_region_sources": by_region,
        "forbidden_source_findings": forbidden_findings,
        "router_status": router.get("status"),
        "router_dline_status": router.get("dline_status"),
        "router_return_allowed": bool(router.get("return_allowed")) if router else False,
        "router_reason_classes": router.get("reason_classes", []),
        "gates": gates,
        "blockers": blockers,
        "decision": (
            "All required V15 ownership regions are promotion eligible; a separate D-line controlled package transaction may proceed."
            if legal
            else "Unified V15 candidate package must not be built because required ownership regions are not promotion eligible."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed V15 unified region legality precheck.")
    parser.add_argument("--source-graph", type=Path, default=REPORTS / "20260508_v15_source_graph.json")
    parser.add_argument("--router-queue", type=Path, default=REPORTS / "20260508_v15_dline_router_queue.json")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_unified_region_legality_precheck.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_unified_region_legality_precheck.md")
    args = parser.parse_args()

    source_graph = read_summary(args.source_graph)
    router = read_summary(args.router_queue)
    summary = build_summary(source_graph, router)
    write_json(args.output_json, summary)
    write_report(args.output_md, "V15 Unified Region Legality Precheck", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=True))
    return 0 if summary["status"] == "v15_unified_regions_promotion_eligible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
