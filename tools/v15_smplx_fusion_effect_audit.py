from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
WORKER_C_ROOT = LOCAL_ROOT / "V15_SMPLX_native_worker_C"

DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v15_smplx_fusion_effect_audit.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v15_smplx_fusion_effect_audit.md"
DEFAULT_AUDIT_DIR = WORKER_C_ROOT / "fusion_effect_audit"
DEFAULT_RUNNER = REPORTS / "20260508_v15_smplx_native_overfit_runner.json"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "strict_gate_registry",
    "strict_pass",
    "formal_candidate",
    "candidate_gate",
)

RESEARCH_CONTRACT = {
    "research_only": True,
    "formal_cloud_unblocked": False,
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "no_predictions_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def safe_v15_research_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v15_smplx_native_worker_c" not in lower:
        raise ValueError(f"Refusing non-Worker-C research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def number_at(data: dict[str, Any], paths: list[tuple[str, ...]]) -> float | None:
    for path in paths:
        cur: Any = data
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            try:
                value = float(cur)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"present": False}
    attempt_summary = data.get("attempt", {}).get("summary", {}) if isinstance(data.get("attempt"), dict) else {}
    source = attempt_summary if isinstance(attempt_summary, dict) and attempt_summary.get("present") else data
    metrics = {
        "present": True,
        "status": data.get("status") or source.get("status"),
        "summary_path": source.get("summary_path") or data.get("_summary_path"),
        "mean_iou": number_at(source, [("mean_iou",), ("final_mean_iou",), ("mask_iou",), ("metrics", "mean_iou"), ("metrics", "final_mean_iou")]),
        "target_recall": number_at(source, [("target_recall",), ("mean_target_recall",), ("metrics", "target_recall"), ("metrics", "mean_target_recall")]),
        "rgb_residual": number_at(source, [("rgb_residual",), ("mean_rgb_residual",), ("metrics", "rgb_residual")]),
        "loss": number_at(source, [("final_loss",), ("best_loss",), ("loss",), ("metrics", "final_loss")]),
    }
    if isinstance(data.get("comparison"), dict):
        comparison = data["comparison"]
        for family in ("anchored_plus_free", "fused", "native_fused"):
            if isinstance(comparison.get(family), dict):
                family_row = comparison[family]
                for key in ("mean_iou", "mean_target_recall", "mean_rgb_residual"):
                    if key in family_row:
                        metrics[key if key != "mean_target_recall" else "target_recall"] = family_row[key]
                break
    return metrics


def metric_delta(fused: dict[str, Any], baseline: dict[str, Any], key: str, higher_is_better: bool) -> dict[str, Any]:
    f_val = fused.get(key)
    b_val = baseline.get(key)
    if f_val is None or b_val is None:
        return {"metric": key, "available": False}
    delta = float(f_val) - float(b_val)
    improved = delta > 0 if higher_is_better else delta < 0
    return {
        "metric": key,
        "available": True,
        "baseline": float(b_val),
        "fused": float(f_val),
        "delta": delta,
        "improved": bool(improved),
        "higher_is_better": higher_is_better,
    }


def discover_default_sources() -> dict[str, Path]:
    return {
        "runner": DEFAULT_RUNNER,
        "native_hair4": LOCAL_ROOT / "B_Hair4_native_4k4d_smplx_hair_topology" / "summary.json",
        "gs0_anchor": LOCAL_ROOT / "B_GS0_smplx_anchored_free_gaussian_smoke" / "b_gs0_summary.json",
        "fus3d_dispatch": REPORTS / "20260508_v15_fus3d_region_backend_dispatch.json",
        "hair_hand_readiness": REPORTS / "20260508_v15_hair_hand_readiness.json",
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V15 SMPL-X Fusion Effect Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only Worker C audit. It reads existing summaries and does not write predictions, teacher/candidate packages, registries, strict pass state, or formal cloud jobs.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(summary["comparison"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Sources",
        "",
    ]
    for label, row in summary["sources"].items():
        lines.append(f"- {label}: exists=`{row['file']['exists']}` status=`{row.get('status')}`")
    lines.extend(
        [
            "",
            "## Research Gate / D-Line",
            "",
            f"- research_gate_result: `{summary['research_gate_result']}`",
            f"- dline_allowed: `{summary['dline_allowed']}`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 Worker C SMPL-X fusion effect audit.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--fused-summary", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    audit_dir = safe_v15_research_dir(args.audit_dir)
    default_sources = discover_default_sources()
    source_paths = dict(default_sources)
    for idx, path in enumerate(args.source):
        source_paths[f"extra_{idx}"] = path

    sources: dict[str, dict[str, Any]] = {}
    for label, path in source_paths.items():
        data = read_json(path)
        sources[label] = {
            "file": file_row(path),
            "status": data.get("status"),
            "task": data.get("task"),
            "metrics": extract_metrics(data),
            "blocker_count": len(data.get("blockers", []) or []),
        }

    baseline_path = args.baseline_summary or default_sources["gs0_anchor"]
    fused_path = args.fused_summary
    baseline_data = read_json(baseline_path)
    fused_data = read_json(fused_path)
    baseline_metrics = extract_metrics(baseline_data)
    fused_metrics = extract_metrics(fused_data)

    deltas = [
        metric_delta(fused_metrics, baseline_metrics, "mean_iou", True),
        metric_delta(fused_metrics, baseline_metrics, "target_recall", True),
        metric_delta(fused_metrics, baseline_metrics, "rgb_residual", False),
        metric_delta(fused_metrics, baseline_metrics, "loss", False),
    ]
    available = [row for row in deltas if row.get("available")]
    improved = [row for row in available if row.get("improved")]

    blockers: list[str] = []
    if not baseline_metrics.get("present"):
        blockers.append(f"Baseline/non-fused summary is missing: {baseline_path}.")
    if not fused_metrics.get("present"):
        blockers.append(f"Fused/native runner summary is missing: {fused_path}.")
    if not available:
        blockers.append("No comparable numeric fusion metric was available from Worker A/B or Worker C outputs.")
    if available and not improved:
        blockers.append("Comparable metrics do not show a positive fusion effect.")
    hand_hair = read_json(default_sources["hair_hand_readiness"])
    if bool(hand_hair.get("hand_ownership_ready")) is False:
        blockers.append("Hand ownership remains false; positive fusion metrics would still not unblock D-line.")
    if bool(hand_hair.get("hair_ownership_ready")) is False:
        blockers.append("Hair ownership remains false; positive fusion metrics would still not unblock D-line.")

    if available and improved:
        status = "v15_smplx_fusion_effect_observed_research_only"
        decision = "At least one comparable metric improved, but this audit is research-only and writes no strict pass."
    elif available:
        status = "v15_smplx_fusion_effect_not_demonstrated"
        decision = "Comparable metrics were found, but they do not demonstrate a reliable positive fusion effect."
    else:
        status = "v15_smplx_fusion_effect_audit_blocked_missing_comparables"
        decision = "Worker C could not audit fusion effect because required baseline/fused comparable summaries are missing."

    comparison = {
        "audit_dir": audit_dir,
        "baseline_summary": file_row(baseline_path),
        "fused_summary": file_row(fused_path),
        "baseline_metrics": baseline_metrics,
        "fused_metrics": fused_metrics,
        "metric_deltas": deltas,
        "available_metric_count": len(available),
        "improved_metric_count": len(improved),
    }
    summary = {
        "task": "v15_smplx_fusion_effect_audit",
        "created_utc": utc_now(),
        "status": status,
        **RESEARCH_CONTRACT,
        "sources": sources,
        "comparison": comparison,
        "fusion_effect_observed": bool(available and improved),
        "research_gate_result": "local_research_only_no_formal_cloud",
        "dline_allowed": False,
        "decision": decision,
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
