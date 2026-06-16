from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_JSON = REPORTS / "20260508_v21_completion_contract.json"
DEFAULT_MD = REPORTS / "20260508_v21_completion_contract.md"
VALID = {"DONE_PASS", "DONE_FAIL_ROUTED", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"}


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


def main() -> int:
    parser = argparse.ArgumentParser(description="V21 completion contract.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    source_paths = {
        "v16_microfit": REPORTS / "20260508_v16_vggt_smplx_microfit_runner.json",
        "v17_residual": REPORTS / "20260508_v17_smplx_residual_surface_optimizer.json",
        "v18_distill": REPORTS / "20260508_v18_residual_teacher_distillation_case.json",
        "v19_temporal": REPORTS / "20260508_v19_temporal_canonical_residual_teacher.json",
        "v20_promotion": REPORTS / "20260508_v20_final_promotion_transaction.json",
    }
    source_statuses: dict[str, Any] = {}
    routed: list[dict[str, Any]] = []
    for name, path in source_paths.items():
        data = read_json(path)
        source_statuses[name] = {
            "path": str(path),
            "exists": path.exists(),
            "upstream_status": str(data.get("status") or ""),
            "blockers": [str(x) for x in (data.get("blockers") or [])],
        }

    if "negative" in source_statuses["v16_microfit"]["upstream_status"]:
        routed.append({
            "id": "V22_TRUE_MICROFIT",
            "owner": "agent_v22",
            "trigger": "V16_MICROFIT_WEAK",
            "action": "Run M2/M3 research microfit with controls and larger view/resolution support.",
            "required_inputs": ["reports/20260508_v16_vggt_smplx_microfit_runner.json"],
            "acceptance": ["DONE_PASS or DONE_FAIL_ROUTED with real/zero/shuffle/random/prior-dropout controls."],
        })
    if "research" in source_statuses["v17_residual"]["upstream_status"]:
        routed.append({
            "id": "V23_REGION_REPAIR",
            "owner": "agent_v23",
            "trigger": "V17_HEAD_FACE_EMPTY_OR_RESEARCH_ONLY",
            "action": "Repair residual evidence masks so body/head/face/left_hand/right_hand are nonempty.",
            "required_inputs": ["reports/20260508_v17_smplx_residual_surface_optimizer.json"],
            "acceptance": ["body/head/face/left_hand/right_hand nonempty or hard-impossible evidence."],
        })
    if "research" in source_statuses["v18_distill"]["upstream_status"]:
        routed.append({
            "id": "V24_TEACHER_V2",
            "owner": "agent_v24",
            "trigger": "V18_RESEARCH_ONLY",
            "action": "Regenerate residual teacher targets from V23 and audit 6v/12v/60v support.",
            "required_inputs": ["reports/20260508_v18_residual_teacher_distillation_case.json"],
            "acceptance": ["teacher v2 research audit complete."],
        })
    if "predictions_missing" in source_statuses["v19_temporal"]["upstream_status"]:
        routed.append({
            "id": "V25_RESEARCH_PREDICTIONS",
            "owner": "agent_v25",
            "trigger": "V19_PREDICTIONS_MISSING",
            "action": "Generate research-only frame0000/0001/0002 outputs without predictions.npz formal path.",
            "required_inputs": ["output/4k4d_scenes/0012_11_frame0000_12views_tmf"],
            "acceptance": ["3 frames have research_depths/points/normals/confidence or hard-impossible evidence."],
        })
    if "fail_closed" in source_statuses["v20_promotion"]["upstream_status"]:
        routed.append({
            "id": "V28_HOLD",
            "owner": "agent_v28",
            "trigger": "V20_EARLY_FAIL_CLOSED",
            "action": "Hold final promotion until V22-V27 complete.",
            "required_inputs": ["reports/20260508_v20_final_promotion_transaction.json"],
            "acceptance": ["V28 runs only after V22-V27 evidence exists."],
        })

    summary = {
        "task": "v21_completion_contract",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DONE_FAIL_ROUTED" if routed else "DONE_PASS",
        "allowed_terminal_statuses": sorted(VALID),
        "forbidden_terminal_terms": ["pending", "not run", "missing predictions", "needs rerun", "blocked, returning"],
        "rules": {
            "dline_fail": "route_to_next_branch",
            "branch_fail": "route_to_next_technical_branch",
            "v20_or_v28": "promotion_only_after_prerequisites",
            "external_routes": "MANO_FLAME_HairGS_HaMeR_WiLoR_HGGT_frozen",
            "formal_writes": "forbidden_until_v28_strict_gate_pass",
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "non_promotion_guard": {
            "observed": {
                "strict_candidate_passes": 0,
                "strict_teacher_passes": 0,
                "formal_cloud_unblocked": False,
                "writes_package": False,
                "writes_strict_pass": False,
                "writes_strict_registry": False,
                "candidate_package_path": None,
                "teacher_package_path": None,
                "registry_entry_path": None,
            }
        },
        "source_statuses": source_statuses,
        "routed_work_items": routed,
        "research_only": True,
    }
    write_json(args.output_json, summary)
    lines = [
        "# V21 Completion Contract",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Allowed terminal states: `DONE_PASS`, `DONE_FAIL_ROUTED`, `DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE`.",
        "",
        "D-line fail routes to the next branch; it is not a final return condition. Formal package/registry/pass writes remain forbidden until V28 strict gate passes.",
        "",
        "## Routed Work Items",
        "",
    ]
    lines.extend([f"- `{item['id']}` -> `{item['owner']}`: {item['action']}" for item in routed] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(args.output_json), "routed": len(routed)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
