from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/V9_unified_surface_precheck"
DEFAULT_STATUS_JSON = REPO_ROOT / "reports/20260507_v9_unified_surface_precheck_status.json"
DEFAULT_STATUS_MD = REPO_ROOT / "reports/20260507_v9_unified_surface_precheck_status.md"

DEFAULT_REALITY = REPO_ROOT / "output/surface_research_cloud_preflight/V9_reality_audit/summary.json"
DEFAULT_ASSET_VERIFY = REPO_ROOT / "reports/20260507_v9_cloud_asset_verification_status.json"
DEFAULT_A5X2 = REPO_ROOT / "output/surface_research_cloud_preflight/V9_A5X2_true_backend_precheck/summary.json"
DEFAULT_HAND_HAIR = REPO_ROOT / "reports/20260507_v9_hand_hair_real_module_status.json"

FORBIDDEN_OUTPUT_TOKENS = ("predictions", "teacher_export", "candidate_export", "strict_pass", "strict_gate_registry")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.expanduser().is_file():
        return {}
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing V9 unified precheck output path containing {token!r}: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V9 unified research surface candidate-precheck decision gate.")
    parser.add_argument("--reality-audit", type=Path, default=DEFAULT_REALITY)
    parser.add_argument("--asset-verification", type=Path, default=DEFAULT_ASSET_VERIFY)
    parser.add_argument("--a5x2-precheck", type=Path, default=DEFAULT_A5X2)
    parser.add_argument("--hand-hair-precheck", type=Path, default=DEFAULT_HAND_HAIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_MD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    blockers = summary["blockers"]
    lines = [
        "# V9 Unified Surface Precheck",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only precheck. It does not create a candidate package, teacher package, predictions export, registry entry, or strict pass.",
        "",
        "## Inputs",
        "",
        f"- reality_audit: `{summary['inputs']['reality_audit']}`",
        f"- asset_verification: `{summary['inputs']['asset_verification']}`",
        f"- a5x2_precheck: `{summary['inputs']['a5x2_precheck']}`",
        f"- hand_hair_precheck: `{summary['inputs']['hand_hair_precheck']}`",
        "",
        "## Component Readiness",
        "",
        "| Component | Ready | Verdict |",
        "| --- | ---: | --- |",
    ]
    for name, item in summary["component_readiness"].items():
        lines.append(f"| `{name}` | `{item['ready']}` | `{item['verdict']}` |")
    lines += [
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines += [f"- {item}" for item in blockers]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Decision",
        "",
        summary["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = safe_output_dir(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    reality = load_json(args.reality_audit)
    asset_verify = load_json(args.asset_verification)
    a5x2 = load_json(args.a5x2_precheck)
    hand_hair = load_json(args.hand_hair_precheck)

    asset_ready = bool(asset_verify.get("status") == "v9_cloud_assets_verified")
    a5x2_ready = bool(
        a5x2.get("status", "").startswith("ready")
        or a5x2.get("true_backend_available") is True
        or a5x2.get("teacher_intake_precheck_ready") is True
    )
    hand_ready = bool(nested_get(hand_hair, "lanes", "hand11", "ready", default=False))
    hair_ready = bool(nested_get(hand_hair, "lanes", "hair4", "ready", default=False))
    body_ready = False
    if isinstance(reality, dict):
        cloud_a = reality.get("cloud_a") or reality.get("targets", {}).get("cloud_a") or {}
        verdict = json.dumps(cloud_a, ensure_ascii=False).lower()
        body_ready = "procedural" not in verdict and "fallback" not in verdict and bool(cloud_a)

    component_readiness = {
        "cloud_asset_staging": {
            "ready": asset_ready,
            "verdict": asset_verify.get("status", "missing_asset_verification"),
        },
        "a5x2_true_dense_backend": {
            "ready": a5x2_ready,
            "verdict": a5x2.get("verdict") or a5x2.get("status", "missing_a5x2_precheck"),
        },
        "b_fus3d3_body_head_face": {
            "ready": body_ready,
            "verdict": "blocked_cloud_a_v8_used_procedural_fallback; rerun B-Fus3D3 only after staged assets are wired into training",
        },
        "b_hand11_real_decoder": {
            "ready": hand_ready,
            "verdict": nested_get(hand_hair, "lanes", "hand11", "verdict", default=hand_hair.get("hand_verdict", "missing_hand11_precheck")),
        },
        "b_hair4_real_topology": {
            "ready": hair_ready,
            "verdict": nested_get(hand_hair, "lanes", "hair4", "verdict", default=hand_hair.get("hair_verdict", "missing_hair4_precheck")),
        },
    }
    blockers = []
    if not asset_ready:
        blockers.append("Cloud-A real asset staging is not verified locally/remotely.")
    if not a5x2_ready:
        blockers.append("A5-X2 true external dense backend is unavailable; V8 A5-X synthetic smoke cannot be used as weak dense anchor.")
    if not body_ready:
        blockers.append("B-Fus3D3 body/head/face is not trained on verified real assets yet; V8 Cloud-A used procedural fallback.")
    if not hand_ready:
        blockers.append("B-hand11 real VGGT+hand-token decoder is unavailable; B-hand10 proxy cannot satisfy hand ownership.")
    if not hair_ready:
        blockers.append("B-hair4 real HairGS topology backend is unavailable; B-hair3 proxy cannot satisfy hairline/head-top ownership.")

    can_merge = asset_ready and a5x2_ready and body_ready and hand_ready and hair_ready
    summary = {
        "task": "v9_unified_surface_candidate_precheck",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "blocked_no_unified_surface_candidate_precheck" if not can_merge else "ready_for_research_only_unified_surface_precheck",
        "contract": {
            "research_only": True,
            "no_export": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_write": True,
            "formal_cloud_unblocked": False,
        },
        "inputs": {
            "reality_audit": str(args.reality_audit.expanduser().resolve()),
            "asset_verification": str(args.asset_verification.expanduser().resolve()),
            "a5x2_precheck": str(args.a5x2_precheck.expanduser().resolve()),
            "hand_hair_precheck": str(args.hand_hair_precheck.expanduser().resolve()),
        },
        "component_readiness": component_readiness,
        "can_merge_unified_surface": bool(can_merge),
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "blockers": blockers,
        "decision": (
            "UNIFIED_RESEARCH_SURFACE_PRECHECK_READY: all real components are available for a bounded research-only merge."
            if can_merge
            else "UNIFIED_SURFACE_PRECHECK_BLOCKED_FAIL_FAST: do not merge proxy/synthetic V8 artifacts into a candidate. Next valid action is wiring staged assets into B-Fus3D3 and installing/running true A5-X2/hand11/hair4 backends."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "report.md", summary)
    write_json(args.status_json.expanduser().resolve(), summary)
    write_markdown(args.status_report.expanduser().resolve(), summary)
    print(json.dumps({"status": summary["status"], "can_merge_unified_surface": can_merge, "blockers": blockers}, ensure_ascii=False))
    return 0 if can_merge else 2


if __name__ == "__main__":
    raise SystemExit(main())
