from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V9_hand_hair_real_module_preflight"
LANE_OUTPUT = OUTPUT_ROOT / "hand11_real_token_decoder"
REPORT_JSON = REPO_ROOT / "reports/20260507_v9_hand_hair_real_module_status.json"
REPORT_MD = REPO_ROOT / "reports/20260507_v9_hand_hair_real_module_status.md"

CODE_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".task", ".npz", ".json"}
STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "preflight_only": True,
    "fail_fast": True,
    "no_train": True,
    "no_cloud": True,
    "no_predictions_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V9 fail-fast preflight for a real B-hand11 VGGT-token/HGGT hand decoder. "
            "It writes diagnostic reports only and exits non-zero when the real module assets are absent."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=LANE_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"available": False, "path": rel(path), "error": str(exc)}
    if isinstance(payload, dict):
        payload["available"] = True
        payload["path"] = rel(path)
        return payload
    return {"available": False, "path": rel(path), "error": "JSON payload is not an object"}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def grep_lines(path: Path, patterns: dict[str, str]) -> dict[str, Any]:
    text = read_text(path)
    rows: dict[str, Any] = {"path": rel(path), "exists": path.is_file(), "matches": {}}
    if not text:
        return rows
    lines = text.splitlines()
    for name, pattern in patterns.items():
        found = None
        rx = re.compile(pattern, re.IGNORECASE)
        for idx, line in enumerate(lines, start=1):
            if rx.search(line):
                found = {"line": idx, "text": line.strip()[:240]}
                break
        rows["matches"][name] = found
    return rows


def iter_files(roots: list[Path], suffixes: set[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in suffixes:
                out.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = {part.lower() for part in path.parts}
            if ".git" in parts or "__pycache__" in parts or ".venv313" in parts or "logs" in parts:
                continue
            if path.suffix.lower() in suffixes:
                out.append(path)
    return sorted(out)


def scan_code_hits(roots: list[Path], patterns: dict[str, str], *, max_hits: int = 80) -> dict[str, list[dict[str, Any]]]:
    rows = {name: [] for name in patterns}
    files = iter_files(roots, CODE_SUFFIXES)
    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in patterns.items()}
    for path in files:
        text = read_text(path)
        if not text:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, rx in compiled.items():
                if len(rows[name]) >= max_hits:
                    continue
                if rx.search(line):
                    rows[name].append({"path": rel(path), "line": line_no, "text": line.strip()[:220]})
    return rows


def scan_named_assets(keywords: list[str], *, max_hits: int = 80) -> list[dict[str, Any]]:
    roots = [
        REPO_ROOT / "external_models",
        REPO_ROOT / "tools",
        REPO_ROOT / "vggt",
        REPO_ROOT / "training",
        REPO_ROOT / "output/surface_research_preflight_local",
        REPO_ROOT / "output/surface_research_cloud_preflight",
    ]
    out: list[dict[str, Any]] = []
    for path in iter_files(roots, MODEL_SUFFIXES | {".py"}):
        lower = path.name.lower()
        if not any(keyword.lower() in lower for keyword in keywords):
            continue
        out.append({"path": rel(path), "suffix": path.suffix.lower(), "bytes": path.stat().st_size})
        if len(out) >= max_hits:
            break
    return out


def existing_path(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "is_file": path.is_file(), "is_dir": path.is_dir()}


def inspect_v8_hand10() -> dict[str, Any]:
    local_summary = load_json(REPO_ROOT / "reports/20260507_v8_cloud_c_b_hand10_status.json")
    cloud_summary = load_json(
        REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_C/b_hand10_hggt_style_hand_decoder/summary.json"
    )
    smoke_source = grep_lines(
        REPO_ROOT / "tools/b_hand10_hggt_style_hand_decoder_smoke.py",
        {
            "synthetic_token_proxy": r"synthetic token margin proxy",
            "bounded_linear_proxy": r"bounded linear proxy",
            "procedural_target": r"def hand_target",
            "procedural_controls": r"def control_points",
            "writes_hggt_named_ply": r"hggt_style_points",
        },
    )
    genealogy = cloud_summary.get("artifact_genealogy") or local_summary.get("artifact_genealogy") or {}
    return {
        "local_report_summary": {
            "available": local_summary.get("available", False),
            "status": local_summary.get("status"),
            "success": local_summary.get("success"),
            "artifact_genealogy": genealogy,
        },
        "cloud_output_summary": {
            "available": cloud_summary.get("available", False),
            "status": cloud_summary.get("status"),
            "success": cloud_summary.get("success"),
            "artifact_genealogy": cloud_summary.get("artifact_genealogy"),
            "outputs": cloud_summary.get("outputs"),
        },
        "source_evidence": smoke_source,
        "verdict": (
            "proxy"
            if "synthetic" in json.dumps(genealogy, ensure_ascii=False).lower()
            or "bounded linear proxy" in json.dumps(genealogy, ensure_ascii=False).lower()
            else "unclear"
        ),
    }


def inspect_backbone_token_support() -> dict[str, Any]:
    aggregator = grep_lines(
        REPO_ROOT / "vggt/models/aggregator.py",
        {
            "patch_embed_tokens": r"patch_tokens = self\.patch_embed",
            "special_token_concat": r"torch\.cat\(\[camera_token, register_token, patch_tokens\]",
            "returns_intermediate_tokens": r"return output_list, self\.patch_start_idx",
        },
    )
    vggt_forward = grep_lines(
        REPO_ROOT / "vggt/models/vggt.py",
        {
            "aggregator_call": r"aggregated_tokens_list, patch_start_idx = self\.aggregator",
            "point_head": r"self\.point_head = DPTHead",
            "depth_head": r"self\.depth_head = DPTHead",
            "normal_head": r"self\.normal_head = DPTHead",
        },
    )
    token_cache_dirs = sorted(
        [
            rel(path)
            for path in (REPO_ROOT / "output/surface_research_preflight_local").glob("B_Fus3D0_token_cache_extract*")
            if path.is_dir()
        ]
    )
    return {
        "general_vggt_tokens_available": bool(
            aggregator["matches"].get("returns_intermediate_tokens")
            and vggt_forward["matches"].get("aggregator_call")
        ),
        "aggregator_evidence": aggregator,
        "vggt_forward_evidence": vggt_forward,
        "existing_general_token_cache_dirs": token_cache_dirs[:12],
        "note": "General VGGT backbone token outputs exist; this is not the same as a hand-specific HGGT decoder.",
    }


def inspect_hand_module_assets() -> dict[str, Any]:
    hits = scan_code_hits(
        [REPO_ROOT / "vggt", REPO_ROOT / "training", REPO_ROOT / "tools"],
        {
            "hggt_name": r"\bhggt\b",
            "hand_token": r"hand[_ -]?token|token[_ -]?hand",
            "hand_decoder": r"hand.*decoder|decoder.*hand",
            "cross_attention": r"cross[_ -]?attention",
            "mano": r"\bmano\b",
            "learned_module": r"class\s+\w*(hand|hggt)\w*\(.*(nn\.Module|Module)",
            "checkpoint_load": r"load_state_dict|torch\.load|checkpoint|ckpt",
        },
        max_hits=50,
    )
    named_assets = scan_named_assets(["hggt", "hand", "mano", "decoder", "token"], max_hits=120)
    vggt_specific_hits = [
        row
        for rows in hits.values()
        for row in rows
        if row["path"].replace("\\", "/").startswith("vggt/")
    ]
    real_module_candidates = [
        row
        for row in named_assets
        if row["path"].replace("\\", "/").startswith(("vggt/", "external_models/"))
        and not row["path"].endswith("hand_landmarker.task")
    ]
    return {
        "code_hits": hits,
        "named_assets": named_assets,
        "vggt_package_hand_hggt_hits": vggt_specific_hits,
        "real_module_candidates_in_vggt_or_external_models": real_module_candidates,
        "external_models": [
            {"path": rel(path), "bytes": path.stat().st_size}
            for path in sorted((REPO_ROOT / "external_models").glob("*"))
            if path.is_file()
        ],
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    backbone = inspect_backbone_token_support()
    b_hand10 = inspect_v8_hand10()
    module_assets = inspect_hand_module_assets()
    required_assets = {
        "real_vggt_token_readout": {
            "present": bool(backbone["general_vggt_tokens_available"]),
            "sufficient": False,
            "reason": "Backbone tokens exist generally, but no B-hand11 decoder consumes them as a real hand module.",
        },
        "hggt_source_module": {
            "present": bool(module_assets["vggt_package_hand_hggt_hits"]),
            "sufficient": False,
            "reason": "No hand/HGGT nn.Module is present under vggt/.",
        },
        "hand_specific_learned_decoder_checkpoint": {
            "present": bool(module_assets["real_module_candidates_in_vggt_or_external_models"]),
            "sufficient": False,
            "reason": "No HGGT/hand decoder checkpoint or package asset was found; only MediaPipe hand_landmarker.task is present externally.",
        },
        "mano_plus_residual_decoder": {
            "present": False,
            "sufficient": False,
            "reason": "Search found no real MANO-base residual hand decoder implementation.",
        },
        "real_hand_roi_train_or_eval_protocol": {
            "present": True,
            "sufficient": False,
            "reason": "Hand ROI evidence/proxy smokes exist, but B-hand10 generated synthetic target/control points.",
        },
    }
    missing = [name for name, row in required_assets.items() if not row["sufficient"]]
    summary = {
        "task": "b_hand11_real_token_decoder_preflight",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "blocked_missing_real_hand_token_decoder",
        "contract": CONTRACT,
        "strict_facts": STRICT_FACTS,
        "output_dir": rel(output_dir),
        "backbone_token_support": backbone,
        "b_hand10_proxy_evidence": b_hand10,
        "module_asset_scan": module_assets,
        "required_assets": required_assets,
        "missing_or_insufficient": missing,
        "verdict": (
            "FAIL_FAST: B-hand10 is a synthetic/proxy HGGT-style smoke. "
            "A true B-hand11 VGGT-token/HGGT hand decoder asset is absent."
        ),
        "next_required": [
            "Add a real hand decoder nn.Module that consumes VGGT aggregated tokens or token-cache tensors.",
            "Add explicit left/right hand tokens and camera-aware cross-attention from hand ROI tokens.",
            "Add MANO weak base plus learned residual/finger decoder, or document the non-MANO replacement.",
            "Provide a checkpoint/train config and real 4K4D hand ROI train/eval controls.",
        ],
    }
    return summary


def write_lane_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-hand11 Real Token Decoder Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a fail-fast preflight report only. It writes no predictions, teacher, candidate, registry, strict pass, export, train job, or cloud job.",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## What Exists",
        "",
        f"- General VGGT backbone token output: `{summary['backbone_token_support']['general_vggt_tokens_available']}`",
        f"- Existing token-cache dirs sampled: `{len(summary['backbone_token_support']['existing_general_token_cache_dirs'])}`",
        f"- B-hand10 status: `{summary['b_hand10_proxy_evidence']['cloud_output_summary'].get('status')}`",
        f"- B-hand10 proxy verdict: `{summary['b_hand10_proxy_evidence']['verdict']}`",
        "",
        "## Missing Or Insufficient",
        "",
    ]
    for item in summary["missing_or_insufficient"]:
        row = summary["required_assets"][item]
        lines.append(f"- `{item}`: {row['reason']}")
    lines.extend(
        [
            "",
            "## Source Proof",
            "",
            "```json",
            json.dumps(
                {
                    "b_hand10_source_evidence": summary["b_hand10_proxy_evidence"]["source_evidence"],
                    "backbone_token_support": summary["backbone_token_support"],
                    "external_models": summary["module_asset_scan"]["external_models"],
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )[:30000],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_lane_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def write_aggregate(report_json: Path, report_md: Path, hand_summary: dict[str, Any]) -> None:
    hair_summary = load_lane_summary(OUTPUT_ROOT / "hair4_real_topology/summary.json")
    lanes = {"hand11": hand_summary}
    if hair_summary:
        lanes["hair4"] = hair_summary
    aggregate = {
        "task": "v9_hand_hair_real_module_preflight",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "blocked_missing_real_modules",
        "contract": CONTRACT,
        "strict_facts": STRICT_FACTS,
        "lanes": lanes,
        "overall_verdict": (
            "FAIL_FAST: true hand/hair real-module assets are absent or insufficient. "
            "Do not promote B-hand10/B-hair3 proxy smokes to candidate/export/pass."
        ),
    }
    write_json(report_json, aggregate)
    lines = [
        "# V9 Hand/Hair Real-Module Preflight",
        "",
        f"Status: `{aggregate['status']}`",
        "",
        aggregate["overall_verdict"],
        "",
        "## Lane Verdicts",
        "",
    ]
    for name, lane in lanes.items():
        lines.extend(
            [
                f"- `{name}`: `{lane.get('status')}`",
                f"  - {lane.get('verdict')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Strict Truth",
            "",
            "```json",
            json.dumps(STRICT_FACTS, indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Outputs",
            "",
            f"- `{rel(OUTPUT_ROOT)}`",
            f"- `{rel(report_json)}`",
            "",
        ]
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    write_json(output_dir / "summary.json", summary)
    write_lane_report(output_dir / "report.md", summary)
    write_aggregate(args.report_json, args.report_md, summary)
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"]}, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
