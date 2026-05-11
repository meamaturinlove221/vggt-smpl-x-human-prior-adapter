from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V9_hand_hair_real_module_preflight"
LANE_OUTPUT = OUTPUT_ROOT / "hair4_real_topology"
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
            "V9 fail-fast preflight for a real B-hair4 HairGS-style topology module. "
            "It writes diagnostic reports only and exits non-zero when the real topology assets are absent."
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


def scan_named_assets(keywords: list[str], *, max_hits: int = 100) -> list[dict[str, Any]]:
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


def inspect_v8_hair3() -> dict[str, Any]:
    local_summary = load_json(REPO_ROOT / "reports/20260507_v8_cloud_d_b_hair3_status.json")
    cloud_summary = load_json(REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_D/b_hair3_hairgs_topology/summary.json")
    smoke_source = grep_lines(
        REPO_ROOT / "tools/b_hair3_hairgs_topology_smoke.py",
        {
            "synthetic_topology": r"synthetic topology smoke",
            "image_first_proxy": r"image-first boundary proxy",
            "synthetic_token_proxy": r"synthetic residual control proxy",
            "procedural_target": r"def hair_target",
            "procedural_controls": r"def control\(",
            "writes_hairgs_named_ply": r"hairgs_strand_points",
        },
    )
    genealogy = cloud_summary.get("artifact_genealogy") or local_summary.get("artifact_genealogy") or {}
    return {
        "local_report_summary": {
            "available": local_summary.get("available", False),
            "status": local_summary.get("status"),
            "success": local_summary.get("success"),
            "artifact_genealogy": genealogy,
            "comparison": local_summary.get("comparison"),
        },
        "cloud_output_summary": {
            "available": cloud_summary.get("available", False),
            "status": cloud_summary.get("status"),
            "success": cloud_summary.get("success"),
            "artifact_genealogy": cloud_summary.get("artifact_genealogy"),
            "comparison": cloud_summary.get("comparison"),
            "outputs": cloud_summary.get("outputs"),
        },
        "source_evidence": smoke_source,
        "verdict": (
            "proxy"
            if "synthetic" in json.dumps(genealogy, ensure_ascii=False).lower()
            or "proxy" in json.dumps(genealogy, ensure_ascii=False).lower()
            else "unclear"
        ),
    }


def inspect_hair_module_assets() -> dict[str, Any]:
    hits = scan_code_hits(
        [REPO_ROOT / "vggt", REPO_ROOT / "training", REPO_ROOT / "tools"],
        {
            "hairgs_name": r"hair\s*gs|hairgs",
            "hair_topology": r"hair.*topology|topology.*hair",
            "strand": r"\bstrand\b",
            "gaussian_strand": r"gaussian.*strand|strand.*gaussian",
            "segment_merge": r"segment.*merg|merg.*segment",
            "differentiable_projection": r"differentiable.*projection|projection.*loss",
            "learned_module": r"class\s+\w*(hair|strand|topology)\w*\(.*(nn\.Module|Module)",
            "checkpoint_load": r"load_state_dict|torch\.load|checkpoint|ckpt",
        },
        max_hits=60,
    )
    named_assets = scan_named_assets(["hairgs", "hair", "strand", "topology", "gaussian"], max_hits=140)
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
        and not row["path"].endswith("face_landmarker.task")
    ]
    return {
        "code_hits": hits,
        "named_assets": named_assets,
        "vggt_package_hair_hairgs_hits": vggt_specific_hits,
        "real_module_candidates_in_vggt_or_external_models": real_module_candidates,
        "external_models": [
            {"path": rel(path), "bytes": path.stat().st_size}
            for path in sorted((REPO_ROOT / "external_models").glob("*"))
            if path.is_file()
        ],
    }


def inspect_prior_hair_smokes() -> dict[str, Any]:
    hair2 = load_json(REPO_ROOT / "reports/20260507_b_hair2_image_first_status.json")
    hair1 = load_json(REPO_ROOT / "reports/20260507_b_hair1_backend_status.json")
    hair2_source = grep_lines(
        REPO_ROOT / "tools/b_hair2_image_first_hair_surface_backend.py",
        {
            "research_only": r"research_only",
            "token_residual": r"token_residual",
            "strand_chain": r"strand_chain",
            "decision_fail": r"FAIL: B-hair2",
            "no_checkpoint": r"no_checkpoint_write",
        },
    )
    return {
        "b_hair1_summary": {
            "available": hair1.get("available", False),
            "status": hair1.get("status"),
            "success": hair1.get("success"),
            "decision": hair1.get("decision"),
        },
        "b_hair2_summary": {
            "available": hair2.get("available", False),
            "status": hair2.get("status"),
            "success": hair2.get("success"),
            "decision": hair2.get("decision"),
            "artifact_genealogy": hair2.get("artifact_genealogy"),
        },
        "b_hair2_source_evidence": hair2_source,
        "verdict": "research_proxy_or_control_smoke",
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    b_hair3 = inspect_v8_hair3()
    module_assets = inspect_hair_module_assets()
    prior_smokes = inspect_prior_hair_smokes()
    required_assets = {
        "hairgs_source_module": {
            "present": bool(module_assets["vggt_package_hair_hairgs_hits"]),
            "sufficient": False,
            "reason": "No HairGS/HairGS-style topology nn.Module is present under vggt/.",
        },
        "learned_root_or_strand_topology_network": {
            "present": False,
            "sufficient": False,
            "reason": "Search found procedural root/strand-chain smokes, not a learned root proposal/topology network.",
        },
        "strand_or_gaussian_parameters_checkpoint": {
            "present": bool(module_assets["real_module_candidates_in_vggt_or_external_models"]),
            "sufficient": False,
            "reason": "No HairGS topology checkpoint, Gaussian-strand parameter asset, or package asset was found.",
        },
        "differentiable_multiview_projection_loss": {
            "present": bool(module_assets["code_hits"].get("differentiable_projection")),
            "sufficient": False,
            "reason": "Projection/contact-sheet render helpers exist, but no trainable differentiable projection loss for hair topology was found.",
        },
        "real_hair_head_segmentation_and_merging": {
            "present": False,
            "sufficient": False,
            "reason": "No real hair/head segmentation plus segment-merging asset exists beyond mask/image boundary proxy scoring.",
        },
    }
    missing = [name for name, row in required_assets.items() if not row["sufficient"]]
    summary = {
        "task": "b_hair4_real_topology_preflight",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "blocked_missing_real_hairgs_topology_module",
        "contract": CONTRACT,
        "strict_facts": STRICT_FACTS,
        "output_dir": rel(output_dir),
        "b_hair3_proxy_evidence": b_hair3,
        "prior_hair_smoke_evidence": prior_smokes,
        "module_asset_scan": module_assets,
        "required_assets": required_assets,
        "missing_or_insufficient": missing,
        "verdict": (
            "FAIL_FAST: B-hair3 is a synthetic/proxy HairGS-style topology smoke. "
            "A true B-hair4 HairGS/topology module asset is absent."
        ),
        "next_required": [
            "Add a real HairGS-style topology module or imported backend package.",
            "Provide learned root/strand or Gaussian-chain parameters and a checkpoint/train config.",
            "Consume real multiview RGB, masks, cameras, and VGGT tokens rather than synthetic residual controls.",
            "Add differentiable projection/topology losses and explicit hair-vs-head segment merging.",
        ],
    }
    return summary


def write_lane_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-hair4 Real Topology Preflight",
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
        f"- B-hair3 status: `{summary['b_hair3_proxy_evidence']['cloud_output_summary'].get('status')}`",
        f"- B-hair3 proxy verdict: `{summary['b_hair3_proxy_evidence']['verdict']}`",
        f"- B-hair2 status: `{summary['prior_hair_smoke_evidence']['b_hair2_summary'].get('status')}`",
        f"- External model files: `{len(summary['module_asset_scan']['external_models'])}`",
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
                    "b_hair3_source_evidence": summary["b_hair3_proxy_evidence"]["source_evidence"],
                    "prior_hair_smoke_evidence": summary["prior_hair_smoke_evidence"],
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


def write_aggregate(report_json: Path, report_md: Path, hair_summary: dict[str, Any]) -> None:
    hand_summary = load_lane_summary(OUTPUT_ROOT / "hand11_real_token_decoder/summary.json")
    lanes: dict[str, Any] = {}
    if hand_summary:
        lanes["hand11"] = hand_summary
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
