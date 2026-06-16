from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output/surface_research_cloud_preflight/V9_reality_audit"
DEFAULT_REPORT_MD = REPO_ROOT / "reports/20260507_v9_reality_audit.md"
DEFAULT_REPORT_JSON = REPO_ROOT / "reports/20260507_v9_reality_audit.json"

V8_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight"

TARGETS = {
    "a5x": {
        "label": "A5-X external dense backend",
        "question": "real external backend or synthetic/proxy",
        "output_dir": V8_ROOT / "Cloud_B/a5x_external_dense_teacher_intake",
        "summary": V8_ROOT / "Cloud_B/a5x_external_dense_teacher_intake/summary.json",
        "status_json": REPO_ROOT / "reports/20260507_v8_cloud_b_a5x_external_dense_intake_status.json",
        "status_md": REPO_ROOT / "reports/20260507_v8_cloud_b_a5x_external_dense_intake_status.md",
        "script": REPO_ROOT / "tools/a5x_external_dense_teacher_intake_smoke.py",
        "modal_wrapper": REPO_ROOT / "modal_v8_research_cloud.py",
    },
    "b_hand10": {
        "label": "B-hand10 HGGT-style hand decoder",
        "question": "real VGGT+hand token decoder or proxy",
        "output_dir": V8_ROOT / "Cloud_C/b_hand10_hggt_style_hand_decoder",
        "summary": V8_ROOT / "Cloud_C/b_hand10_hggt_style_hand_decoder/summary.json",
        "status_json": REPO_ROOT / "reports/20260507_v8_cloud_c_b_hand10_status.json",
        "status_md": REPO_ROOT / "reports/20260507_v8_cloud_c_b_hand10_status.md",
        "script": REPO_ROOT / "tools/b_hand10_hggt_style_hand_decoder_smoke.py",
        "modal_wrapper": REPO_ROOT / "modal_v8_research_cloud.py",
    },
    "b_hair3": {
        "label": "B-hair3 HairGS topology",
        "question": "true HairGS topology or proxy",
        "output_dir": V8_ROOT / "Cloud_D/b_hair3_hairgs_topology",
        "summary": V8_ROOT / "Cloud_D/b_hair3_hairgs_topology/summary.json",
        "status_json": REPO_ROOT / "reports/20260507_v8_cloud_d_b_hair3_status.json",
        "status_md": REPO_ROOT / "reports/20260507_v8_cloud_d_b_hair3_status.md",
        "script": REPO_ROOT / "tools/b_hair3_hairgs_topology_smoke.py",
        "modal_wrapper": REPO_ROOT / "modal_v8_research_cloud.py",
    },
    "cloud_a": {
        "label": "Cloud-A B-Fus3D2 dataset train",
        "question": "procedural fallback or real assets",
        "output_dir": V8_ROOT / "Cloud_A/b_fus3d2_human_dataset_train",
        "summary": V8_ROOT / "Cloud_A/b_fus3d2_human_dataset_train/summary.json",
        "status_json": REPO_ROOT / "reports/20260507_v8_cloud_a_b_fus3d2_status.json",
        "status_md": REPO_ROOT / "reports/20260507_v8_cloud_a_b_fus3d2_status.md",
        "script": REPO_ROOT / "tools/b_fus3d2_human_dataset_train.py",
        "dataset_script": REPO_ROOT / "training/data/datasets/human_surface_sdf_dataset.py",
        "modal_wrapper": REPO_ROOT / "modal_v8_research_cloud.py",
    },
}

CONTEXT_FILES = {
    "manifest": V8_ROOT / "v8_research_cloud_manifest.json",
    "launch_status": REPO_ROOT / "reports/20260507_v8_research_cloud_launch_agent_status.json",
    "gate_status": REPO_ROOT / "reports/20260507_v8_research_cloud_gate_status.json",
    "artifact_referee": REPO_ROOT / "reports/20260507_v8_research_cloud_artifact_referee.json",
    "smoke_utils": REPO_ROOT / "tools/v8_research_smoke_utils.py",
}

FORBIDDEN_PROMOTION_NAMES = (
    "predictions.npz",
    "teacher_export",
    "candidate_export",
    "strict_pass",
    "formal_candidate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V9 fail-fast reality audit for V8 research-cloud artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def json_at(payload: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return payload
    cur = payload
    for raw_part in pointer.strip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def json_evidence(
    source: str,
    path: Path,
    pointer: str,
    payload: dict[str, Any],
    reason: str,
    expected: Any | None = None,
) -> dict[str, Any]:
    value = json_at(payload, pointer)
    item = {
        "kind": "json_field",
        "source": source,
        "path": str(path),
        "relative_path": rel(path),
        "json_pointer": pointer,
        "value": json_ready(value),
        "reason": reason,
    }
    if expected is not None:
        item["expected_for_real_claim"] = expected
    return item


def source_evidence(path: Path, pattern: str, reason: str, flags: int = re.IGNORECASE) -> list[dict[str, Any]]:
    text = read_text(path)
    evidence: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line, flags=flags):
            evidence.append(
                {
                    "kind": "source_line",
                    "path": str(path),
                    "relative_path": rel(path),
                    "line": line_no,
                    "text": line.strip(),
                    "reason": reason,
                }
            )
    return evidence


def first_source_evidence(path: Path, pattern: str, reason: str) -> dict[str, Any]:
    matches = source_evidence(path, pattern, reason)
    if matches:
        return matches[0]
    return {
        "kind": "source_line_missing",
        "path": str(path),
        "relative_path": rel(path),
        "pattern": pattern,
        "reason": reason,
    }


def path_evidence(path: Path, reason: str, *, must_exist: bool = True) -> dict[str, Any]:
    return {
        "kind": "path",
        "path": str(path),
        "relative_path": rel(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "expected_exists": must_exist,
        "reason": reason,
    }


def source_hits(path: Path, patterns: dict[str, str]) -> dict[str, bool]:
    lower = read_text(path).lower()
    return {name: bool(re.search(pattern, lower)) for name, pattern in patterns.items()}


def list_files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "relative_path": rel(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return files


def forbidden_outputs(root: Path) -> list[str]:
    hits = []
    if not root.is_dir():
        return hits
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if any(token in lowered for token in FORBIDDEN_PROMOTION_NAMES):
            hits.append(str(path))
    return sorted(hits)


def common_flags(summary: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    flag_names = (
        "research_only",
        "no_export",
        "no_predictions_write",
        "no_registry_write",
        "no_teacher_export",
        "no_candidate_export",
        "no_strict_pass_write",
        "not_teacher",
        "not_candidate",
        "pass",
        "strict_teacher_passes",
        "strict_candidate_passes",
        "teacher_export",
        "candidate_export",
        "predictions_export",
        "formal_cloud_train_infer_export",
    )
    return {
        "declared_flags": {name: summary.get(name) for name in flag_names if name in summary},
        "unexpected_promotion_outputs": forbidden_outputs(target["output_dir"]),
        "artifact_count": len(list_files(target["output_dir"])),
    }


def audit_a5x() -> dict[str, Any]:
    target = TARGETS["a5x"]
    summary = load_json(target["summary"])
    status = load_json(target["status_json"])
    methods = summary.get("methods", {})
    method_rows = {}
    for name, row in methods.items():
        method_rows[name] = {
            "ply": row.get("ply"),
            "known_camera_aligned_claim": row.get("known_camera_aligned"),
            "can_reproject_original_6_view_protocol_claim": row.get("can_reproject_original_6_view_protocol"),
            "mean_depth_residual": row.get("mean_depth_residual"),
            "calls_real_external_backend": False,
            "input_image_count": 0,
            "loads_real_checkpoint_or_backend_model": False,
            "uses_original_4k4d_frame_pixels": False,
            "synthetic_or_proxy_point_generator": True,
        }

    evidence = [
        path_evidence(target["summary"], "Cloud-B downloaded summary used for A5-X verdict."),
        path_evidence(target["status_json"], "Report-level V8 status used for A5-X cross-check."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/source", summary, "Declares the A5-X artifact source as synthetic."),
        json_evidence("summary", target["summary"], "/weak_teacher_pool_only", summary, "Declares weak-teacher pool only, not a real exported teacher."),
        json_evidence("summary", target["summary"], "/strict_teacher_ready", summary, "Strict teacher readiness is explicitly false."),
        json_evidence("summary", target["summary"], "/no_teacher_export", summary, "Teacher export is explicitly blocked."),
        json_evidence("summary", target["summary"], "/best_method", summary, "Best external-looking method is a label inside the synthetic smoke."),
        json_evidence("status_report", target["status_json"], "/artifact_genealogy/source", status, "Status report repeats the synthetic source flag."),
        first_source_evidence(target["script"], r"make_human_points", "A5-X point clouds start from synthetic human points."),
        first_source_evidence(target["script"], r"rng\.normal", "A5-X method clouds are perturbations/noise around the synthetic target."),
        first_source_evidence(target["script"], r"synthetic external dense intake", "A5-X script labels its own genealogy synthetic."),
        first_source_evidence(target["modal_wrapper"], r"a5x_external_dense_teacher_intake_smoke\.py", "Cloud-B launcher routes to the smoke script, not a real backend binary."),
    ]
    return {
        "target": target["label"],
        "question": target["question"],
        "answer": "synthetic/proxy, not a real external backend run",
        "fail_fast_verdict": "FAIL_FAST_SYNTHETIC_PROXY",
        "promotion_allowed": False,
        "summary_path": str(target["summary"]),
        "status_json_path": str(target["status_json"]),
        "script_path": str(target["script"]),
        "output_dir": str(target["output_dir"]),
        "best_method": summary.get("best_method"),
        "method_count": len(method_rows),
        "methods": method_rows,
        "source_hits": source_hits(
            target["script"],
            {
                "make_human_points": r"make_human_points",
                "rng_normal_point_generator": r"rng\.normal",
                "synthetic_source_string": r"synthetic external dense intake",
                "subprocess_backend_invocation": r"subprocess|os\.system",
                "checkpoint_load": r"checkpoint|ckpt|load_state_dict",
            },
        ),
        **common_flags(summary, target),
        "evidence": evidence,
        "next_required": "Run a real MUSt3R/MASt3R/2DGS/NeuS2 backend on real frames and export verifiable backend logs/checkpoints before teacher-intake claims.",
    }


def audit_hand10() -> dict[str, Any]:
    target = TARGETS["b_hand10"]
    summary = load_json(target["summary"])
    status = load_json(target["status_json"])
    sides = summary.get("sides", {})
    real_side_flags = {
        side: {
            "real_ply": data.get("real", {}).get("ply"),
            "real_not_mano_only_flag": data.get("real", {}).get("not_mano_only"),
            "real_not_procedural_tube_flag": data.get("real", {}).get("not_procedural_tube"),
            "mask_only_not_procedural_tube_flag": data.get("mask_only", {}).get("not_procedural_tube"),
            "zero_not_procedural_tube_flag": data.get("zero", {}).get("not_procedural_tube"),
        }
        for side, data in sides.items()
    }

    evidence = [
        path_evidence(target["summary"], "Cloud-C downloaded summary used for B-hand10 verdict."),
        path_evidence(target["status_json"], "Report-level V8 status used for B-hand10 cross-check."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/uses_vggt_tokens", summary, "Declares VGGT-token use as synthetic proxy, not real tokens."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/learned_decoder", summary, "Declares decoder as bounded proxy smoke."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/uses_mano_base", summary, "Declares only weak MANO base, not a full MANO+residual backend."),
        json_evidence("summary", target["summary"], "/no_teacher_export", summary, "Teacher export is explicitly blocked."),
        json_evidence("summary", target["summary"], "/strict_teacher_passes", summary, "No strict teacher pass exists."),
        json_evidence("status_report", target["status_json"], "/artifact_genealogy/uses_vggt_tokens", status, "Status report repeats synthetic token proxy flag."),
        first_source_evidence(target["script"], r"def hand_target", "The hand point target is generated procedurally in the smoke script."),
        first_source_evidence(target["script"], r"tube", "Finger geometry is built as procedural tube-like point sets."),
        first_source_evidence(target["script"], r"synthetic token margin proxy", "Script genealogy explicitly says VGGT tokens are synthetic proxy."),
        first_source_evidence(target["script"], r"bounded linear proxy", "Script genealogy explicitly says learned decoder is a proxy."),
        first_source_evidence(target["modal_wrapper"], r"b_hand10_hggt_style_hand_decoder_smoke\.py", "Cloud-C launcher routes to the smoke script."),
    ]
    return {
        "target": target["label"],
        "question": target["question"],
        "answer": "proxy, not a real VGGT+hand token decoder",
        "fail_fast_verdict": "FAIL_FAST_PROXY_DECODER",
        "promotion_allowed": False,
        "summary_path": str(target["summary"]),
        "status_json_path": str(target["status_json"]),
        "script_path": str(target["script"]),
        "output_dir": str(target["output_dir"]),
        "real_decoder_checks": {
            "reads_real_vggt_aggregator_tokens": False,
            "has_left_right_learnable_hand_tokens": False,
            "has_camera_aware_cross_attention": False,
            "has_real_hand_roi_from_dataset": False,
            "has_mano_base_plus_residual_decoder": False,
            "has_exportable_predictions": False,
        },
        "side_flags": real_side_flags,
        "source_hits": source_hits(
            target["script"],
            {
                "synthetic_token_margin_proxy": r"synthetic token margin proxy",
                "bounded_linear_proxy": r"bounded linear proxy",
                "procedural_tube": r"tube",
                "cross_attention": r"cross[-_ ]?attention",
                "real_vggt_token_loader": r"aggregator|token_cache|vggt",
            },
        ),
        **common_flags(summary, target),
        "evidence": evidence,
        "next_required": "Implement and run a real VGGT-token hand decoder with left/right hand tokens, camera-aware cross-attention, real hand ROI inputs, and an export-blocked referee until it passes.",
    }


def audit_hair3() -> dict[str, Any]:
    target = TARGETS["b_hair3"]
    summary = load_json(target["summary"])
    status = load_json(target["status_json"])
    comparison = summary.get("comparison", {})
    real_metrics = summary.get("controls_metrics", {}).get("real_token", {})
    evidence = [
        path_evidence(target["summary"], "Cloud-D downloaded summary used for B-hair3 verdict."),
        path_evidence(target["status_json"], "Report-level V8 status used for B-hair3 cross-check."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/source", summary, "Declares HairGS-style artifact source as synthetic topology smoke."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/root_proposal", summary, "Root proposal is an image-first boundary proxy."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/segment_merging", summary, "Segment merging is only a metric, not a real topology algorithm."),
        json_evidence("summary", target["summary"], "/artifact_genealogy/uses_vggt_tokens", summary, "VGGT-token use is synthetic residual control proxy."),
        json_evidence("summary", target["summary"], "/comparison/real_minus_image_only_topology", summary, "Topology gain over image-only is tiny and not real HairGS evidence."),
        json_evidence("summary", target["summary"], "/no_teacher_export", summary, "Teacher export is explicitly blocked."),
        json_evidence("status_report", target["status_json"], "/artifact_genealogy/uses_vggt_tokens", status, "Status report repeats synthetic token proxy flag."),
        first_source_evidence(target["script"], r"def hair_target", "Hair root/strand target is generated procedurally in the smoke script."),
        first_source_evidence(target["script"], r"np\.random\.default_rng", "Hair target/control generation is RNG driven."),
        first_source_evidence(target["script"], r"synthetic topology smoke", "Script genealogy explicitly labels the topology smoke synthetic."),
        first_source_evidence(target["script"], r"synthetic residual control proxy", "Script genealogy explicitly labels token use as proxy."),
        first_source_evidence(target["modal_wrapper"], r"b_hair3_hairgs_topology_smoke\.py", "Cloud-D launcher routes to the smoke script."),
    ]
    return {
        "target": target["label"],
        "question": target["question"],
        "answer": "proxy, not true HairGS topology",
        "fail_fast_verdict": "FAIL_FAST_PROXY_TOPOLOGY",
        "promotion_allowed": False,
        "summary_path": str(target["summary"]),
        "status_json_path": str(target["status_json"]),
        "script_path": str(target["script"]),
        "output_dir": str(target["output_dir"]),
        "hairgs_checks": {
            "uses_multiview_rgb": False,
            "uses_real_vggt_tokens": False,
            "has_root_proposal_network": False,
            "optimizes_gaussian_strand_parameters": False,
            "has_segment_merging_algorithm": False,
            "has_differentiable_projection_loss": False,
            "has_multiview_consistency_loss": False,
            "has_hair_vs_head_segmentation_inputs": False,
            "has_exportable_predictions": False,
        },
        "real_token_metrics": real_metrics,
        "comparison": comparison,
        "source_hits": source_hits(
            target["script"],
            {
                "synthetic_topology_source": r"synthetic topology smoke",
                "image_first_boundary_proxy": r"image-first boundary proxy",
                "synthetic_residual_control_proxy": r"synthetic residual control proxy",
                "projection_loss": r"projection loss|differentiable",
                "multiview": r"multi[-_ ]?view",
            },
        ),
        **common_flags(summary, target),
        "evidence": evidence,
        "next_required": "Implement and run true HairGS-style hair topology with real head/hair RGB or masks, VGGT features, root proposals, strand parameters, projection loss, and multiview consistency.",
    }


def audit_cloud_a() -> dict[str, Any]:
    target = TARGETS["cloud_a"]
    summary = load_json(target["summary"])
    status = load_json(target["status_json"])
    genealogy = summary.get("genealogy", {})
    genealogy_text = json.dumps(genealogy, ensure_ascii=False).lower()
    cases = genealogy.get("cases", []) if isinstance(genealogy, dict) else []
    procedural_cases = [
        {
            "index": idx,
            "case_builder": case.get("case_builder"),
            "note": case.get("note"),
            "seed": case.get("seed"),
            "shell_offsets": case.get("shell_offsets"),
        }
        for idx, case in enumerate(cases)
        if isinstance(case, dict) and case.get("case_builder") == "make_procedural_sdf_case"
    ]
    used_procedural = bool(procedural_cases) or "make_procedural_sdf_case" in genealogy_text
    evidence = [
        path_evidence(target["summary"], "Cloud-A downloaded summary used for fallback verdict."),
        path_evidence(target["status_json"], "Report-level V8 status used for Cloud-A cross-check."),
        json_evidence("summary", target["summary"], "/genealogy/synthetic_mesh_supervision", summary, "Cloud-A declares synthetic mesh supervision."),
        json_evidence("summary", target["summary"], "/genealogy/external_case_roots", summary, "No external case roots were loaded."),
        json_evidence("summary", target["summary"], "/genealogy/cases/0/case_builder", summary, "First case uses procedural fallback builder."),
        json_evidence("summary", target["summary"], "/genealogy/cases/0/note", summary, "First case records missing query/template fallback."),
        json_evidence("summary", target["summary"], "/inputs/query_cache", summary, "Configured query cache path existed in config but was not available to the remote run."),
        json_evidence("summary", target["summary"], "/inputs/template_payload", summary, "Configured template path existed in config but was not available to the remote run."),
        json_evidence("summary", target["summary"], "/contract/synthetic_mesh_supervision_allowed", summary, "The V8 contract allowed synthetic supervision for this smoke."),
        json_evidence("summary", target["summary"], "/strict_teacher_passes", summary, "No strict teacher pass exists."),
        json_evidence("status_report", target["status_json"], "/genealogy/cases/0/case_builder", status, "Status report repeats procedural case builder."),
        first_source_evidence(target["dataset_script"], r"def make_procedural_sdf_case", "Dataset has an explicit procedural case generator."),
        first_source_evidence(target["dataset_script"], r"Fallback used when query/template assets are absent", "Dataset records fallback reason in genealogy."),
        first_source_evidence(target["dataset_script"], r"not self\.query_cache.*is_file", "Dataset branches to procedural fallback when assets are absent."),
        first_source_evidence(target["script"], r"synthetic_mesh_supervision_allowed", "Cloud-A training smoke explicitly allows synthetic mesh supervision."),
        first_source_evidence(target["modal_wrapper"], r"b_fus3d2_human_dataset_train\.py", "Cloud-A launcher routes to the bounded dataset smoke."),
    ]
    return {
        "target": target["label"],
        "question": target["question"],
        "answer": "procedural fallback, not real assets",
        "fail_fast_verdict": "FAIL_FAST_PROCEDURAL_FALLBACK",
        "promotion_allowed": False,
        "summary_path": str(target["summary"]),
        "status_json_path": str(target["status_json"]),
        "script_path": str(target["script"]),
        "dataset_script_path": str(target["dataset_script"]),
        "output_dir": str(target["output_dir"]),
        "used_procedural_fallback": used_procedural,
        "procedural_cases": procedural_cases,
        "real_asset_checks": {
            "external_case_roots_loaded": bool(genealogy.get("external_case_roots")),
            "query_cache_case_loaded": any(
                isinstance(case, dict) and case.get("case_builder") == "load_query_cache_case" for case in cases
            ),
            "external_surface_sdf_case_loaded": any(
                isinstance(case, dict) and case.get("case_builder") == "load_external_case" for case in cases
            ),
            "all_cases_procedural": bool(cases) and len(procedural_cases) == len(cases),
            "has_exportable_predictions": False,
        },
        "comparison": summary.get("comparison", {}),
        "source_hits": source_hits(
            target["dataset_script"],
            {
                "make_procedural_sdf_case": r"def make_procedural_sdf_case",
                "fallback_note": r"fallback used when query/template assets are absent",
                "query_cache_loader": r"load_query_cache_case",
                "external_case_loader": r"load_external_case",
            },
        ),
        **common_flags(summary, target),
        "evidence": evidence,
        "next_required": "Package the real query cache/template/RGB/mask/camera/token assets into Modal and rerun with fallback disabled or treated as a hard error.",
    }


def context_summary() -> dict[str, Any]:
    manifest = load_json(CONTEXT_FILES["manifest"])
    launch = load_json(CONTEXT_FILES["launch_status"])
    gate = load_json(CONTEXT_FILES["gate_status"])
    referee = load_json(CONTEXT_FILES["artifact_referee"])
    smoke_utils = CONTEXT_FILES["smoke_utils"]
    return {
        "manifest_path": str(CONTEXT_FILES["manifest"]),
        "manifest_research_only": manifest.get("research_only"),
        "manifest_no_teacher_export": manifest.get("no_teacher_export"),
        "manifest_no_candidate_export": manifest.get("no_candidate_export"),
        "manifest_no_predictions_write": manifest.get("no_predictions_write"),
        "launch_attempted": launch.get("launch_attempted"),
        "launch_mode": launch.get("mode"),
        "gate_status": gate.get("status"),
        "gate_verdict": gate.get("verdict"),
        "artifact_referee_verdict": referee.get("verdict"),
        "artifact_referee_aggregate": referee.get("aggregate", {}),
        "smoke_utils_path": str(smoke_utils),
        "smoke_utils_flags": source_hits(
            smoke_utils,
            {
                "research_flags": r"RESEARCH_FLAGS",
                "strict_facts": r"STRICT_FACTS",
                "make_human_points": r"def make_human_points",
            },
        ),
        "evidence": [
            path_evidence(CONTEXT_FILES["manifest"], "V8 research cloud manifest."),
            json_evidence("manifest", CONTEXT_FILES["manifest"], "/research_only", manifest, "Manifest keeps all jobs research-only."),
            json_evidence("manifest", CONTEXT_FILES["manifest"], "/no_teacher_export", manifest, "Manifest blocks teacher export."),
            json_evidence("launch_status", CONTEXT_FILES["launch_status"], "/launch_attempted", launch, "Launch-agent dry-run flag, separate from downloaded Cloud-A/B/C/D artifacts."),
            json_evidence("gate_status", CONTEXT_FILES["gate_status"], "/research_gate/formal_guard_still_blocked", gate, "Formal guard remained blocked."),
            json_evidence("artifact_referee", CONTEXT_FILES["artifact_referee"], "/aggregate/predictions_export", referee, "Referee blocks predictions export."),
            first_source_evidence(smoke_utils, r"def make_human_points", "Shared V8 helper provides synthetic human point generator."),
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_target_markdown(path: Path, item: dict[str, Any]) -> None:
    lines = [
        f"# {item['target']}",
        "",
        f"Question: {item['question']}",
        "",
        f"Answer: `{item['answer']}`",
        f"Fail-fast verdict: `{item['fail_fast_verdict']}`",
        f"Promotion allowed: `{item['promotion_allowed']}`",
        "",
        "## Evidence",
        "",
    ]
    for ev in item.get("evidence", []):
        if ev.get("kind") == "json_field":
            lines.append(
                f"- `{ev['relative_path']}` `{ev['json_pointer']}` = `{json.dumps(ev['value'], ensure_ascii=False)}`"
            )
        elif ev.get("kind") == "source_line":
            lines.append(f"- `{ev['relative_path']}:{ev['line']}` `{ev['text']}`")
        elif ev.get("kind") == "path":
            lines.append(f"- `{ev['relative_path']}` exists={ev['exists']} size={ev['size_bytes']}")
        else:
            lines.append(f"- `{ev.get('relative_path', ev.get('path'))}` {ev.get('kind')}")
    lines.extend(
        [
            "",
            "## Required Next Step",
            "",
            item["next_required"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V9 Reality Audit",
        "",
        f"Generated UTC: `{summary['created_utc']}`",
        f"Status: `{summary['status']}`",
        "",
        "Fail-fast reality audit over V8 Cloud-A/B/C/D artifacts and scripts. This report does not modify guards and does not launch cloud.",
        "",
        "## Verdicts",
        "",
    ]
    for key in ("a5x", "b_hand10", "b_hair3", "cloud_a"):
        item = summary["targets"][key]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- question: {item['question']}",
                f"- answer: `{item['answer']}`",
                f"- fail_fast_verdict: `{item['fail_fast_verdict']}`",
                f"- promotion_allowed: `{item['promotion_allowed']}`",
                f"- summary: `{rel(Path(item['summary_path']))}`",
                "",
            ]
        )
        for ev in item.get("evidence", [])[:8]:
            if ev.get("kind") == "json_field":
                value = json.dumps(ev["value"], ensure_ascii=False)
                lines.append(f"- evidence: `{ev['relative_path']}` `{ev['json_pointer']}` = `{value}`")
            elif ev.get("kind") == "source_line":
                lines.append(f"- evidence: `{ev['relative_path']}:{ev['line']}` `{ev['text']}`")
            elif ev.get("kind") == "path":
                lines.append(f"- evidence: `{ev['relative_path']}` exists={ev['exists']} size={ev['size_bytes']}")
        lines.extend(["", f"Next required: {item['next_required']}", ""])

    lines.extend(
        [
            "## Overall",
            "",
            f"`{summary['overall_verdict']}`",
            "",
            "## Context Flags",
            "",
            "```json",
            json.dumps(summary["context"], indent=2, ensure_ascii=False, sort_keys=True)[:20000],
            "```",
            "",
            "## Full JSON",
            "",
            "```json",
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)[:70000],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary() -> dict[str, Any]:
    targets = {
        "a5x": audit_a5x(),
        "b_hand10": audit_hand10(),
        "b_hair3": audit_hair3(),
        "cloud_a": audit_cloud_a(),
    }
    fail_fast = {key: item["fail_fast_verdict"] for key, item in targets.items()}
    promotion_allowed = {key: item["promotion_allowed"] for key, item in targets.items()}
    return {
        "schema_version": "20260507_v9_reality_audit_v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "v9_reality_audit_complete_fail_fast",
        "scope": {
            "owned_tool": str(REPO_ROOT / "tools/v9_reality_audit.py"),
            "owned_reports": [
                str(DEFAULT_REPORT_MD),
                str(DEFAULT_REPORT_JSON),
            ],
            "owned_output_root": str(DEFAULT_OUTPUT),
            "cloud_launch_attempted": False,
            "guards_modified": False,
        },
        "targets": targets,
        "fail_fast_verdicts": fail_fast,
        "promotion_allowed": promotion_allowed,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "context": context_summary(),
        "overall_verdict": (
            "V8 Cloud-A/B/C/D positive signals are research/proxy/fallback only: "
            "A5-X is synthetic external-backend labeling, B-hand10 is a synthetic/procedural decoder proxy, "
            "B-hair3 is a synthetic HairGS-style topology proxy, and Cloud-A used procedural fallback cases. "
            "No target may be promoted to teacher, candidate, predictions export, or strict gate progress."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", summary)
    for key, item in summary["targets"].items():
        write_json(args.output_dir / f"{key}_reality_audit.json", item)
        write_target_markdown(args.output_dir / f"{key}_reality_audit.md", item)

    write_json(args.report_json, summary)
    write_report(args.report_md, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "fail_fast_verdicts": summary["fail_fast_verdicts"],
                "overall_verdict": summary["overall_verdict"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
