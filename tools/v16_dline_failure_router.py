from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v16_dline_failure_router.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v16_dline_failure_router.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def bool_value(value: Any) -> bool:
    return bool(value)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def source_status(data: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "file": file_row(path),
        "task": data.get("task"),
        "status": data.get("status"),
        "decision": data.get("decision"),
        "strict_candidate_passes": data.get("strict_candidate_passes"),
        "strict_teacher_passes": data.get("strict_teacher_passes"),
        "blocker_count": len(data.get("blockers", []) or []),
    }


def route(id_: str, title: str, owner: str, action: str, reason: str, inputs: list[str]) -> dict[str, Any]:
    return {
        "id": id_,
        "title": title,
        "owner": owner,
        "status": "pending",
        "action": action,
        "reason": reason,
        "required_inputs": inputs,
    }


def gate(id_: str, title: str, passed: bool, failure_class: str, evidence: list[str], route_id: str | None) -> dict[str, Any]:
    return {
        "id": id_,
        "title": title,
        "passed": bool(passed),
        "failure_class": None if passed else failure_class,
        "route_id": None if passed else route_id,
        "evidence": evidence,
    }


def build_routes() -> dict[str, dict[str, Any]]:
    routes = [
        route(
            "route_asset_reacquire",
            "Reacquire strict ownership assets",
            "agent_h16_r16_assets",
            "Resolve licensed MANO/SMPL-X/FLAME or equivalent hand/hair topology assets, then rerun hand/hair readiness.",
            "Hand/hair ownership is still false, so later fusion cannot legally promote a unified surface.",
            ["reports/20260508_v15_hair_hand_readiness.json", "reports/20260508_v15_required_licensed_assets.json"],
        ),
        route(
            "route_true_vggt_train_probe",
            "Run true VGGT training probe",
            "agent_t16_train",
            "Run a bounded actual VGGT training/overfit entrypoint using the V15 native prior case, not the raw softsurfel optimizer.",
            "V15 negative result was raw softsurfel-only and does not answer whether VGGT can learn from the native prior.",
            ["training/config/4k4d_smplx_native_prior.yaml", "output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15"],
        ),
        route(
            "route_teacher_protocol_repair",
            "Repair strict teacher protocol",
            "agent_k16_g16_teacher",
            "Repair K/G/T strict teacher sources before any region backend claims body/head/face ownership.",
            "D-line still lacks a strict K/G/T teacher source.",
            ["reports/20260508_v14_dline_promotion_report.json", "reports/20260508_v15_source_graph.json"],
        ),
        route(
            "route_metric_objective_repair",
            "Repair optimization objective",
            "agent_m16_metrics",
            "Investigate why scalar loss dropped while IoU and target recall worsened; add objective or evaluation guards before rerun.",
            "The observed V15 local optimization had negative self-deltas on geometry metrics.",
            ["reports/20260508_v15_smplx_fusion_effect_audit.json", "reports/20260508_v16_v15_autopsy.json"],
        ),
        route(
            "route_unified_legality_hold",
            "Hold unified candidate legality",
            "agent_u16_legality",
            "Keep unified merge blocked until body/head/face/hair/left_hand/right_hand all become promotion eligible.",
            "Source graph has zero promotion-eligible regions and missing required ownership regions.",
            ["reports/20260508_v15_source_graph.json", "reports/20260508_v15_execution_rollup.json"],
        ),
        route(
            "route_dline_rerun_after_fixes",
            "Rerun D-line after branch fixes",
            "agent_d16",
            "Rerun this V16 router only after assets, true VGGT train evidence, strict teacher protocol, and legality sources change.",
            "D-line remains the strict judge; current router must not write pass/package/registry state.",
            ["reports/20260508_v16_dline_failure_router.json"],
        ),
    ]
    return {item["id"]: item for item in routes}


def main() -> int:
    parser = argparse.ArgumentParser(description="V16 D-line failure router for routed SMPL-X path.")
    parser.add_argument("--autopsy", type=Path, default=REPORTS / "20260508_v16_v15_autopsy.json")
    parser.add_argument("--dline", type=Path, default=REPORTS / "20260508_v14_dline_promotion_report.json")
    parser.add_argument("--v15-router", type=Path, default=REPORTS / "20260508_v15_dline_router_queue.json")
    parser.add_argument("--source-graph", type=Path, default=REPORTS / "20260508_v15_source_graph.json")
    parser.add_argument("--v15-rollup", type=Path, default=REPORTS / "20260508_v15_execution_rollup.json")
    parser.add_argument("--fusion-audit", type=Path, default=REPORTS / "20260508_v15_smplx_fusion_effect_audit.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    autopsy = read_json(args.autopsy)
    dline = read_json(args.dline)
    v15_router = read_json(args.v15_router)
    source_graph = read_json(args.source_graph)
    v15_rollup = read_json(args.v15_rollup)
    fusion_audit = read_json(args.fusion_audit)

    dline_gates = dline.get("gate_results", {}) if isinstance(dline.get("gate_results"), dict) else {}
    routes = build_routes()
    strict_candidate = max(
        int_value(dline.get("strict_candidate_passes")),
        int_value(v15_router.get("strict_candidate_passes")),
        int_value(v15_rollup.get("strict_candidate_passes")),
    )
    strict_teacher = max(
        int_value(dline.get("strict_teacher_passes")),
        int_value(v15_router.get("strict_teacher_passes")),
        int_value(v15_rollup.get("strict_teacher_passes")),
    )
    promotion_eligible_regions = set(source_graph.get("promotion_eligible_regions", []) or [])
    required_regions = {"body", "head", "face", "hair", "left_hand", "right_hand"}
    missing_regions = sorted(required_regions - promotion_eligible_regions)
    autopsy_raw_only = bool(autopsy.get("v15_negative_is_raw_softsurfel_only"))
    v15_true_training_negative = bool(autopsy.get("v15_negative_is_true_vggt_training"))
    self_deltas = fusion_audit.get("comparison", {}).get("self_deltas", {})
    self_positive = bool(fusion_audit.get("comparison", {}).get("self_positive"))
    hand_ready = bool_value(dline_gates.get("hand_ownership_ready"))
    hair_ready = bool_value(dline_gates.get("hair_ownership_ready"))
    k_pass = bool_value(dline_gates.get("k14_teacher_precheck_pass"))
    g_pass = bool_value(dline_gates.get("g14_teacher_precheck_pass"))
    t_ready = bool_value(dline_gates.get("t14_canonical_teacher_ready"))
    f_ready = bool_value(dline_gates.get("f14_body_head_face_ready"))

    gates = [
        gate(
            "D0_asset_gate",
            "Licensed and topology ownership assets",
            hand_ready and hair_ready,
            "ASSET_OWNERSHIP_FAIL",
            [
                f"hand_ownership_ready={hand_ready}",
                f"hair_ownership_ready={hair_ready}",
                "V15 reports say hand/hair ownership remains blocked.",
            ],
            "route_asset_reacquire",
        ),
        gate(
            "D1_v15_autopsy_gate",
            "V15 result interpretation",
            autopsy_raw_only and not v15_true_training_negative,
            "V15_NEGATIVE_NOT_TRUE_VGGT_TRAINING",
            [
                f"autopsy_result_class={autopsy.get('result_class')}",
                "V15 command routed to raw softsurfel optimizer, not a VGGT training entrypoint.",
            ],
            "route_true_vggt_train_probe",
        ),
        gate(
            "D2_teacher_protocol_gate",
            "Strict K/G/T teacher protocol",
            k_pass and g_pass and t_ready,
            "STRICT_TEACHER_PROTOCOL_FAIL",
            [
                f"k14_teacher_precheck_pass={k_pass}",
                f"g14_teacher_precheck_pass={g_pass}",
                f"t14_canonical_teacher_ready={t_ready}",
            ],
            "route_teacher_protocol_repair",
        ),
        gate(
            "D3_body_head_face_gate",
            "Body/head/face ownership backend",
            f_ready,
            "BODY_HEAD_FACE_TEACHER_FAIL",
            [
                f"f14_body_head_face_ready={f_ready}",
                "Fus3D/native prep remains research-only without strict teacher/ownership evidence.",
            ],
            "route_teacher_protocol_repair",
        ),
        gate(
            "D4_metric_effect_gate",
            "Local metric effect",
            self_positive,
            "NEGATIVE_OR_MIXED_METRIC_EFFECT",
            [
                f"self_positive={self_positive}",
                f"self_deltas={json_ready(self_deltas)}",
                "V15 scalar loss dropped but IoU/target recall self-deltas were negative.",
            ],
            "route_metric_objective_repair",
        ),
        gate(
            "D5_unified_region_legality_gate",
            "Unified required region legality",
            not missing_regions,
            "UNIFIED_REGION_ILLEGAL",
            [
                f"promotion_eligible_regions={sorted(promotion_eligible_regions)}",
                f"missing_required_regions={missing_regions}",
            ],
            "route_unified_legality_hold",
        ),
        gate(
            "D6_no_promotion_write_gate",
            "No pass/package/registry write in blocked state",
            strict_candidate == 0 and strict_teacher == 0 and not bool(v15_rollup.get("candidate_package_built")),
            "PROMOTION_WRITE_IN_BLOCKED_STATE",
            [
                f"strict_candidate_passes={strict_candidate}",
                f"strict_teacher_passes={strict_teacher}",
                f"candidate_package_built={bool(v15_rollup.get('candidate_package_built'))}",
            ],
            "route_dline_rerun_after_fixes",
        ),
        gate(
            "D7_final_strict_gate",
            "Final strict return gate",
            False,
            "FINAL_STRICT_GATE_BLOCKED",
            [
                "This V16 router is failure routing only.",
                "It intentionally does not write strict registry/package/pass.",
                "Return requires separate D-line transaction after D0-D6 pass.",
            ],
            "route_dline_rerun_after_fixes",
        ),
    ]

    failed_gates = [item for item in gates if not item["passed"]]
    failure_classes = sorted({str(item["failure_class"]) for item in failed_gates if item.get("failure_class")})
    route_ids = unique([str(item["route_id"]) for item in failed_gates if item.get("route_id")])
    next_branch_routing = [routes[route_id] for route_id in route_ids if route_id in routes]

    status = "v16_dline_failure_routed_no_strict_write"
    blockers = unique(
        [
            *(str(item) for item in dline.get("blockers", []) or []),
            *(str(item) for item in v15_router.get("blockers", []) or []),
            *(str(item) for item in v15_rollup.get("blockers", []) or []),
            *(str(item) for item in autopsy.get("blockers", []) or []),
        ]
    )
    decision = (
        "V16 D-line stays blocked and routes the failure. The highest-priority branch is true VGGT training evidence because V15's negative was raw-softsurfel-only; assets and strict teacher/region legality remain parallel blockers."
        if autopsy_raw_only
        else "V16 D-line stays blocked. Autopsy was inconclusive, so manual review must precede any training route."
    )

    summary = {
        "task": "v16_dline_failure_router",
        "created_utc": utc_now(),
        "status": status,
        "formal_cloud_unblocked": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "candidate_package_built": False,
        "registry_entry_path": None,
        "candidate_package_path": None,
        "teacher_package_path": None,
        "writes_strict_registry": False,
        "writes_package": False,
        "writes_strict_pass": False,
        "gates": gates,
        "failed_gate_count": len(failed_gates),
        "failure_classes": failure_classes,
        "next_branch_routing": next_branch_routing,
        "route_ids": route_ids,
        "autopsy_summary": {
            "status": autopsy.get("status"),
            "result_class": autopsy.get("result_class"),
            "v15_negative_is_raw_softsurfel_only": autopsy_raw_only,
            "v15_negative_is_true_vggt_training": v15_true_training_negative,
            "decision": autopsy.get("decision"),
        },
        "source_status": {
            "autopsy": source_status(autopsy, args.autopsy),
            "dline": source_status(dline, args.dline),
            "v15_router": source_status(v15_router, args.v15_router),
            "source_graph": source_status(source_graph, args.source_graph),
            "v15_rollup": source_status(v15_rollup, args.v15_rollup),
            "fusion_audit": source_status(fusion_audit, args.fusion_audit),
        },
        "blockers": blockers,
        "decision": decision,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "failed_gate_count": len(failed_gates), "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 2


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 D-Line Failure Router",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Gates",
        "",
    ]
    for gate_row in summary["gates"]:
        state = "PASS" if gate_row["passed"] else "FAIL"
        lines.append(f"- {gate_row['id']}: `{state}` ({gate_row.get('failure_class') or 'ok'})")
    lines.extend(
        [
            "",
            "## Failure Classes",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in summary["failure_classes"])
    lines.extend(
        [
            "",
            "## Next Branch Routing",
            "",
        ]
    )
    for item in summary["next_branch_routing"]:
        lines.append(f"- `{item['id']}` -> `{item['owner']}`: {item['action']}")
    lines.extend(
        [
            "",
            "## Autopsy",
            "",
            f"- result_class: `{summary['autopsy_summary'].get('result_class')}`",
            f"- raw_softsurfel_only: `{summary['autopsy_summary'].get('v15_negative_is_raw_softsurfel_only')}`",
            f"- true_vggt_training_negative: `{summary['autopsy_summary'].get('v15_negative_is_true_vggt_training')}`",
            "",
            "## Non-Promotion Guard",
            "",
            "- strict_candidate_passes: `0`",
            "- strict_teacher_passes: `0`",
            "- candidate_package_built: `False`",
            "- writes_strict_registry: `False`",
            "- writes_package: `False`",
            "- writes_strict_pass: `False`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = summary.get("blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
