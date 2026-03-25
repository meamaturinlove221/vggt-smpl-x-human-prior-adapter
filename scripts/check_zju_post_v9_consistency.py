import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "output" / "geometry_post_v9_nightly_state" / "state.json"
TRAINING_QUESTION_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
LOCAL_NIGHTLY_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Check consistency across post-v9 nightly status files.")
    parser.add_argument(
        "--state",
        type=Path,
        default=STATE_PATH,
        help="Path to persistent nightly state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "geometry_post_v9_nightly_state" / "consistency_check.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_path(path_like: str) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path.resolve())


def main():
    args = parse_args()
    state = load_json(args.state)
    latest_run_dir = Path(state["latest_run_dir"])
    nightly_decision = load_json(latest_run_dir / "nightly_decision.json")
    brief = load_json(latest_run_dir / "next_training_question_brief.json")
    question_manifest = load_json(TRAINING_QUESTION_MANIFEST)
    nightly_manifest = load_json(LOCAL_NIGHTLY_MANIFEST)

    expected_lead = normalize_path(nightly_manifest["current_lead"]["config"])
    expected_blocker = str(nightly_manifest["current_cloud_blocker"])

    checks = {
        "current_lead_config": {
            "expected": expected_lead,
            "state": normalize_path(state["current_lead_config"]),
            "nightly_decision": normalize_path(nightly_decision["current_lead_config"]),
            "brief": normalize_path(brief["candidate_config"]),
            "training_question_manifest": normalize_path(question_manifest["candidate_config"]),
        },
        "patch_collection_stop": {
            "expected": True,
            "state": bool(state["patch_collection_stop"]),
            "nightly_decision": bool(nightly_decision["patch_collection_stop"]),
            "brief": bool(brief["patch_collection_stop"]),
            "training_question_manifest": bool(question_manifest["patch_collection_stop"]),
            "local_nightly_manifest": bool(nightly_manifest["patch_collection_stop"]),
        },
        "ready_for_new_training_question": {
            "expected": True,
            "state": bool(state["ready_for_new_training_question"]),
            "nightly_decision": bool(nightly_decision["ready_for_new_training_question"]),
            "brief": bool(brief["ready_for_new_training_question"]),
            "training_question_manifest": bool(question_manifest["ready_for_new_training_question"]),
            "local_nightly_manifest": bool(nightly_manifest["ready_for_new_training_question"]),
            "training_question_manifest_status": str(question_manifest["status"]),
            "brief_status": str(brief["status"]),
        },
        "cloud_gate": {
            "expected": False,
            "state": bool(state["cloud_gate"]),
            "nightly_decision": bool(nightly_decision["cloud_gate"]),
            "brief": bool(brief["cloud_gate"]),
            "training_question_manifest": bool(question_manifest["cloud_gate"]),
            "local_nightly_manifest": bool(nightly_manifest["cloud_gate"]),
        },
        "launch_cloud_now": {
            "expected": False,
            "state": bool(state["launch_cloud_now"]),
            "nightly_decision": bool(nightly_decision["launch_cloud_now"]),
            "brief": bool(brief["launch_cloud_now"]),
            "training_question_manifest": bool(question_manifest["launch_cloud_now"]),
            "local_nightly_manifest": bool(nightly_manifest["launch_cloud_now"]),
        },
        "current_cloud_blocker": {
            "expected": expected_blocker,
            "state": str(state["current_cloud_blocker"]),
            "nightly_decision": str(nightly_decision["current_cloud_blocker"]),
            "brief": str(brief["current_cloud_blocker"]),
            "training_question_manifest": str(question_manifest["current_cloud_blocker"]),
            "local_nightly_manifest": str(nightly_manifest["current_cloud_blocker"]),
        },
    }

    mismatches = []
    for field, payload in checks.items():
        expected = payload["expected"]
        for source_name, value in payload.items():
            if source_name == "expected":
                continue
            if field == "ready_for_new_training_question" and source_name in {"training_question_manifest_status", "brief_status"}:
                if value != "ready_for_new_training_question":
                    mismatches.append(f"{field}:{source_name}={value}")
                continue
            if value != expected:
                mismatches.append(f"{field}:{source_name}={value}")

    result = {
        "ok": len(mismatches) == 0,
        "latest_run_dir": str(latest_run_dir.resolve()),
        "checks": checks,
        "mismatches": mismatches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    lines = [
        "# ZJU post-v9 nightly consistency check",
        "",
        f"- ok: `{result['ok']}`",
        f"- latest_run_dir: `{result['latest_run_dir']}`",
        "",
        "## Checks",
        "",
    ]
    for field, payload in result["checks"].items():
        lines.append(f"### {field}")
        lines.append("")
        for key, value in payload.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if result["mismatches"]:
        lines.append("## Mismatches")
        lines.append("")
        for item in result["mismatches"]:
            lines.append(f"- `{item}`")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
