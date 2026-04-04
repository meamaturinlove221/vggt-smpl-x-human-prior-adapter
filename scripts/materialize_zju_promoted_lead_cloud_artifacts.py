import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
LOCAL_MANIFEST_JSON = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"

BASELINE_CONFIG = "training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
CANDIDATE_CONFIG = "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize cloud launch artifacts for the promoted hybrid-ring stable lead."
    )
    parser.add_argument("--date-tag", default="20260403")
    parser.add_argument("--version-suffix", default="")
    parser.add_argument(
        "--output-subdir",
        default="zju_source_policy_research_loop/cloud_runs/20260403_promoted_hybrid_ring_latest_lead_v1",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    args = parse_args()
    local_manifest = load_json(LOCAL_MANIFEST_JSON)
    current_lead = local_manifest.get("current_lead", {}) or {}
    suffix = f".{args.version_suffix}" if str(args.version_suffix).strip() else ""
    decision_json = OUTPUT_ROOT / f"cloud_promotion_decision.source_policy_hybrid_ring_regularization.{args.date_tag}{suffix}.json"
    decision_md = OUTPUT_ROOT / f"cloud_promotion_decision.source_policy_hybrid_ring_regularization.{args.date_tag}{suffix}.md"
    cloud_pair_json = OUTPUT_ROOT / f"zju_next_cloud_pair.source_policy_hybrid_ring_regularization.{args.date_tag}{suffix}.json"
    cloud_pair_md = OUTPUT_ROOT / f"zju_next_cloud_pair.source_policy_hybrid_ring_regularization.{args.date_tag}{suffix}.md"

    decision = {
        "checked_at": now_iso(),
        "artifact_kind": "cloud_promotion_decision",
        "decision": "PROMOTE_CURRENT_LOCAL_LEAD_TO_CLOUD_VALIDATION",
        "family": str(current_lead.get("family", "source_policy_hybrid_ring_regularization")),
        "candidate_shape": str(current_lead.get("first_candidate_shape", "stablelead_nearest_plus_uniform_tail")),
        "current_local_lead_config": str(current_lead.get("config", CANDIDATE_CONFIG)),
        "baseline_reference_config": BASELINE_CONFIG,
        "cloud_gate": True,
        "launch_cloud_now": True,
        "why_now": [
            "source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail remains the current promoted local lead.",
            "Residual-case, focal-only, translation-only, FL/T coupling, and the later higher-level camera-depth audit tickets all failed to dislodge it.",
            "The user explicitly authorized autonomous downstream decisions and cloud execution on the latest valid result.",
        ],
        "cloud_scope": {
            "mode": "single_config_cloud_validation",
            "output_subdir": args.output_subdir,
            "modal_gpu": "A100-80GB",
            "limit_train_batches": 100,
            "limit_val_batches": 20,
            "max_epochs": 1,
        },
        "still_forbidden": [
            "Do not launch a second local ticket before this cloud validation finishes.",
            "Do not reopen dead_same_day camera-object cousins.",
            "Do not launch more than one active Modal app for this validation.",
        ],
    }

    cloud_pair = {
        "schema_version": 1,
        "problem_id": f"source_policy_hybrid_ring_regularization_cloud_validation_{args.date_tag}{('_' + args.version_suffix) if str(args.version_suffix).strip() else ''}",
        "baseline_config": BASELINE_CONFIG,
        "candidate_config": str(current_lead.get("config", CANDIDATE_CONFIG)),
        "extra_overrides": "",
        "local_gate_doc": str(
            current_lead.get("gate_references", {}).get("long_gate_reference_summary", "")
        ),
        "acceptance_cases": [
            "CoreView_390 promoted-hybrid-ring cloud validation against the previous nearest-rawpool reference",
        ],
        "acceptance_metrics": [
            "loss_camera",
            "loss_T",
            "loss_conf_depth",
            "loss_reg_depth",
        ],
        "throughput_profile": "modal_a100_single_config_100x20",
        "cloud_gate": True,
        "launch_cloud_now": True,
        "notes": (
            "Materialized by explicit autonomous cloud decision on April 3, 2026. This pair validates the promoted "
            "hybrid-ring local lead on Modal against the previous nearest-rawpool reference."
        ),
        "output_subdir": args.output_subdir,
        "probe_reference_config": "zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal",
        "probe_label_current": "promoted_hybrid_ring",
        "probe_label_reference": "previous_nearest_rawpool",
    }

    write_json(decision_json, decision)
    write_text(decision_md, render_md("Cloud Promotion Decision", decision))
    write_json(cloud_pair_json, cloud_pair)
    write_text(cloud_pair_md, render_md("Cloud Pair", cloud_pair))
    print(
        json.dumps(
            {
                "cloud_promotion_decision": str(decision_json.relative_to(REPO_ROOT)).replace("\\", "/"),
                "cloud_pair": str(cloud_pair_json.relative_to(REPO_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
