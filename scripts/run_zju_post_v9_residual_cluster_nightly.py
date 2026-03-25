import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_post_v9_residual_cluster_v1.json"
DEFAULT_CLOUD_TEMPLATE = REPO_ROOT / "scripts" / "manifests" / "zju_next_cloud_pair_v1.json"
DEFAULT_TRAINING_QUESTION_TEMPLATE = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
DEFAULT_BASELINE_CONFIG = "training/config/zju_vggt_geom_unproject_minimal.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "geometry_post_v9_residual_cluster_nightly"
DEFAULT_STATE_DIR = REPO_ROOT / "output" / "geometry_post_v9_nightly_state"
PREFLIGHT_PS1 = REPO_ROOT / "scripts" / "invoke_modal_zju_preflight.ps1"
SWEEP_PY = REPO_ROOT / "scripts" / "run_zju_geometry_view_sweep.py"
REGION_SUMMARY_PY = REPO_ROOT / "scripts" / "summarize_zju_region_case_summaries.py"
REGION_COMPARE_PY = REPO_ROOT / "scripts" / "compare_zju_region_batch_summaries.py"
LEGACY_BACKFILL_PS1 = REPO_ROOT / "scripts" / "run_legacy_backfill_from_current_manifest.ps1"
LEGACY_SUMMARY_PY = REPO_ROOT / "scripts" / "summarize_legacy_backfill_manifest.py"
SEARCH_PY = REPO_ROOT / "scripts" / "search_zju_hybrid_source_sets.py"
SWEEP_COMPARE_PY = REPO_ROOT / "scripts" / "compare_zju_geometry_sweeps.py"

HEADROOM_FG_MAX = 0.001
HEADROOM_FULL_MIN = -0.001
HEADROOM_BG_MIN = -0.0015
LEGACY_GAP_DOMINANT_MAX = 0.001


class NightlyError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the post-v9 ZJU local nightly decision protocol."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Frozen residual-cluster manifest.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for nightly run artifacts.",
    )
    parser.add_argument(
        "--state_dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Persistent nightly state directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "diagnose", "bounded_search"],
        help="Run diagnose only, bounded-search only, or auto-advance when allowed.",
    )
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def run_checked(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = run_cmd(args, cwd=cwd)
    if result.returncode != 0:
        raise NightlyError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


def make_case_key(seq_name: str, frame_id: int, target_camera: str) -> str:
    return f"{seq_name}|{frame_id}|{target_camera}"


def make_case_id(seq_name: str, frame_id: int, target_camera: str) -> str:
    return f"{seq_name}_frame_{frame_id:06d}_{target_camera}"


def build_default_state(manifest: dict) -> dict:
    return {
        "frozen_manifest_version": str(manifest.get("frozen_manifest_version", "v9")),
        "bounded_search_consumed": False,
        "pending_bounded_search_group": None,
        "latest_decision": "continue_diagnose",
        "last_reason": "",
        "patch_collection_stop": False,
        "ready_for_new_training_question": False,
        "cloud_gate": False,
        "launch_cloud_now": False,
        "current_lead_config": "",
        "current_cloud_blocker": "",
        "latest_run_dir": "",
        "generated_cloud_template": "",
        "generated_training_question_brief": "",
        "suggested_manifest_path": "",
    }


def load_state(state_path: Path, manifest: dict) -> dict:
    if not state_path.exists():
        return build_default_state(manifest)
    payload = load_json(state_path)
    defaulted = build_default_state(manifest)
    defaulted.update(payload)
    defaulted["cloud_gate"] = False
    return defaulted


def save_state(state_path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["cloud_gate"] = False
    write_json(state_path, payload)


def write_decision_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# ZJU post-v9 nightly decision",
        "",
        f"- state: `{payload['state']}`",
        f"- latest_decision: `{payload['latest_decision']}`",
        f"- preflight_ok: `{payload['preflight_ok']}`",
        f"- input_frozen_ok: `{payload['input_frozen_ok']}`",
        f"- residual_case_count: `{payload['residual_case_count']}`",
        f"- allow_bounded_search: `{payload['allow_bounded_search']}`",
        f"- bounded_search_group: `{payload['bounded_search_group']}`",
        f"- allow_manifest_upgrade: `{payload['allow_manifest_upgrade']}`",
        f"- next_manifest_version: `{payload['next_manifest_version']}`",
        f"- patch_collection_stop: `{payload['patch_collection_stop']}`",
        f"- ready_for_new_training_question: `{payload['ready_for_new_training_question']}`",
        f"- cloud_gate: `{payload['cloud_gate']}`",
        f"- launch_cloud_now: `{payload['launch_cloud_now']}`",
        f"- reason: `{payload['reason']}`",
    ]
    if payload.get("artifacts"):
        lines.extend(["", "## Artifacts", ""])
        for key, value in sorted(payload["artifacts"].items()):
            lines.append(f"- {key}: `{value}`")
    if payload.get("case_labels"):
        lines.extend(["", "## Case Labels", ""])
        for key, value in sorted(payload["case_labels"].items()):
            lines.append(f"- {key}: `{value}`")
    write_text(path, "\n".join(lines) + "\n")


def round_case_summary_path(row: dict) -> Path:
    case_dir = repo_resolve(row["case_dir"])
    return case_dir / "summary.json"


def summary_has_region_diagnostics(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = load_json(summary_path)
    except Exception:
        return False
    region_payload = payload.get("region_diagnostics", {})
    branches = region_payload.get("branches", {})
    point_regions = branches.get("point_map", {}).get("regions", {})
    depth_regions = branches.get("depth_unproject", {}).get("regions", {})
    return bool(point_regions) and bool(depth_regions)


def load_round(round_root: Path) -> dict:
    root = repo_resolve(round_root)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise NightlyError(f"Missing sweep summary: {summary_path}")
    payload = load_json(summary_path)
    keyed = {}
    for row in payload.get("rows", []):
        keyed[(int(row["frame_id"]), str(row["target_camera"]))] = row
    return {
        "root": root,
        "summary_json": summary_path,
        "payload": payload,
        "keyed": keyed,
        "manifest": payload.get("manifest", {}),
    }


def resolve_template_reports(round_info: dict, expected_view_profile: str) -> list[str]:
    profiles = round_info["manifest"].get("profiles", [])
    matched = [
        str(repo_resolve(profile["template_report"]))
        for profile in profiles
        if str(profile.get("view_profile", "")) == expected_view_profile
    ]
    if matched:
        return matched
    return [str(repo_resolve(profile["template_report"])) for profile in profiles]


def ensure_case_summary(
    case: dict,
    round_info: dict,
    expected_view_profile: str,
    supplement_root: Path,
    round_label: str,
) -> Path:
    key = (int(case["frame_id"]), str(case["target_camera"]))
    row = round_info["keyed"].get(key)
    if row is not None:
        existing = round_case_summary_path(row)
        if summary_has_region_diagnostics(existing):
            return existing.resolve()

    round_manifest = round_info["manifest"]
    template_reports = resolve_template_reports(round_info, expected_view_profile)
    local_zju_root = str(repo_resolve(round_manifest["local_zju_root"]))
    checkpoint = str(round_manifest.get("checkpoint", ""))
    python_exe = str(round_manifest.get("python_exe", "")) or sys.executable
    output_root = ensure_dir(supplement_root / round_label)
    cmd = [
        python_exe,
        str(SWEEP_PY),
        "--template_reports",
        *template_reports,
        "--local_zju_root",
        local_zju_root,
        "--checkpoint",
        checkpoint,
        "--output_root",
        str(output_root),
        "--frame_ids",
        str(int(case["frame_id"])),
        "--target_cameras",
        str(case["target_camera"]),
        "--source_policy",
        str(round_manifest.get("source_policy", "fixed_template")),
        "--skip_existing",
    ]
    source_override_json = str(round_manifest.get("source_override_json", "")).strip()
    if source_override_json:
        cmd.extend(["--source_override_json", str(repo_resolve(source_override_json))])
    run_checked(cmd, cwd=REPO_ROOT)
    case_dir = output_root / expected_view_profile / f"frame_{int(case['frame_id']):06d}_{case['target_camera']}"
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        raise NightlyError(f"Supplemented case is still missing summary.json: {summary_path}")
    return summary_path.resolve()


def run_region_summary(label: str, summary_paths: list[Path], output_dir: Path, python_exe: str) -> Path:
    cmd = [
        python_exe,
        str(REGION_SUMMARY_PY),
        "--output_dir",
        str(output_dir),
        "--label",
        label,
        "--summary_json",
        *[str(path) for path in summary_paths],
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    return (output_dir / "summary.json").resolve()


def run_region_compare(
    summary_a: Path,
    summary_b: Path,
    output_dir: Path,
    label_a: str,
    label_b: str,
    title: str,
    python_exe: str,
    ignore_view_profile: bool = False,
) -> Path:
    cmd = [
        python_exe,
        str(REGION_COMPARE_PY),
        "--summary_a",
        str(summary_a),
        "--summary_b",
        str(summary_b),
        "--label_a",
        label_a,
        "--label_b",
        label_b,
        "--output_dir",
        str(output_dir),
        "--title",
        title,
    ]
    if ignore_view_profile:
        cmd.append("--ignore_view_profile")
    run_checked(cmd, cwd=REPO_ROOT)
    return (output_dir / "summary.json").resolve()


def build_current_case_manifest(
    manifest: dict,
    resolved_cases: list[dict],
    path: Path,
) -> Path:
    payload = {
        "label": str(manifest.get("label", "zju_post_v9_residual_cluster_v1")),
        "cases": [],
    }
    for item in resolved_cases:
        payload["cases"].append(
            {
                "seq_name": item["seq_name"],
                "frame_id": item["frame_id"],
                "target_camera": item["target_camera"],
                "view_profile": item["main_view_profile"],
                "current_summary_json": str(item["main_summary_json"]),
            }
        )
    write_json(path, payload)
    return path


def run_legacy_backfill(
    case_manifest_json: Path,
    output_root: Path,
    legacy_view_profile_tag: str,
) -> Path:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(LEGACY_BACKFILL_PS1),
        "-CaseManifestJson",
        str(case_manifest_json),
        "-OutRoot",
        str(output_root),
        "-LegacyViewProfileTag",
        legacy_view_profile_tag,
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    manifest_json = output_root / "backfill_manifest.json"
    if not manifest_json.exists():
        raise NightlyError(f"Legacy backfill did not produce {manifest_json}")
    return manifest_json.resolve()


def run_legacy_summary(backfill_manifest: Path, output_dir: Path, python_exe: str) -> Path:
    cmd = [
        python_exe,
        str(LEGACY_SUMMARY_PY),
        "--backfill_manifest",
        str(backfill_manifest),
        "--output_dir",
        str(output_dir),
        "--label",
        "legacy_gap_summary",
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    return (output_dir / "summary.json").resolve()


def run_preflight(run_dir: Path) -> None:
    log_dir = ensure_dir(run_dir / "preflight")
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PREFLIGHT_PS1),
        "-StopRepoProcesses",
    ]
    result = run_cmd(cmd, cwd=REPO_ROOT)
    write_text(log_dir / "stdout.log", result.stdout)
    write_text(log_dir / "stderr.log", result.stderr)
    if result.returncode != 0:
        payload = {
            "preflight_ok": False,
            "reason": "invoke_modal_zju_preflight.ps1 failed",
            "stdout_log": str((log_dir / "stdout.log").resolve()),
            "stderr_log": str((log_dir / "stderr.log").resolve()),
        }
        write_json(run_dir / "preflight_failed.json", payload)
        write_text(
            run_dir / "preflight_failed.md",
            "# Preflight Failed\n\n"
            f"- stdout_log: `{payload['stdout_log']}`\n"
            f"- stderr_log: `{payload['stderr_log']}`\n",
        )
        raise NightlyError("Preflight failed; see preflight_failed.json")


def keyed_rows(rows: list[dict]) -> dict:
    return {
        make_case_key(str(row["seq_name"]), int(row["frame_id"]), str(row["target_camera"])): row
        for row in rows
    }


def classify_case(compare_row: dict, legacy_row: dict | None) -> str:
    fg_delta_raw = compare_row.get("fg_human_depth_minus_point_mae_delta")
    full_delta = compare_row.get("full_depth_minus_point_mae_delta")
    bg_far_delta = compare_row.get("bg_far_depth_minus_point_mae_delta")
    bg_bottom_delta = compare_row.get("bg_bottom_band_depth_minus_point_mae_delta")
    if fg_delta_raw is None:
        headroom = False
    else:
        fg_delta = float(fg_delta_raw)
        headroom = fg_delta <= HEADROOM_FG_MAX and (
        (full_delta is not None and float(full_delta) <= HEADROOM_FULL_MIN)
        or (bg_far_delta is not None and float(bg_far_delta) <= HEADROOM_BG_MIN)
        or (bg_bottom_delta is not None and float(bg_bottom_delta) <= HEADROOM_BG_MIN)
        )
    if headroom:
        return "shared_source_policy_headroom"
    if legacy_row is not None and float(legacy_row["gap_delta_abs"]) < LEGACY_GAP_DOMINANT_MAX:
        return "legacy_gap_dominant"
    return "bounded_local_residual"


def generate_cloud_template(run_dir: Path, manifest: dict) -> Path:
    if DEFAULT_CLOUD_TEMPLATE.exists():
        template = load_json(DEFAULT_CLOUD_TEMPLATE)
    else:
        template = {
            "schema_version": 1,
            "problem_id": "",
            "baseline_config": "",
            "candidate_config": "",
            "extra_overrides": "",
            "local_gate_doc": "",
            "acceptance_cases": [],
            "acceptance_metrics": [
                "full_depth_minus_point_mae",
                "fg_human_depth_minus_point_mae",
                "bg_far_depth_minus_point_mae",
                "bg_bottom_band_depth_minus_point_mae",
                "legacy_gap_depth_minus_point",
            ],
            "throughput_profile": "",
            "cloud_gate": False,
            "launch_cloud_now": False,
        }

    # Fill the frozen residual set into the generated run-local template so the
    # next cloud question starts from the same accepted local gate.
    template["acceptance_cases"] = [
        {
            "seq_name": case["seq_name"],
            "frame_id": case["frame_id"],
            "target_camera": case["target_camera"],
        }
        for case in manifest["cases"]
    ]
    template["cloud_gate"] = False
    template["launch_cloud_now"] = False
    path = run_dir / "cloud_experiment_template.json"
    write_json(path, template)
    return path


def build_next_training_question_brief(manifest: dict, cluster_summary: dict) -> dict:
    template = {}
    if DEFAULT_TRAINING_QUESTION_TEMPLATE.exists():
        template = load_json(DEFAULT_TRAINING_QUESTION_TEMPLATE)

    case_labels = {case["case_id"]: case["label"] for case in cluster_summary["cases"]}
    headroom_cases = sorted(
        case["case_id"] for case in cluster_summary["cases"] if case["label"] == "shared_source_policy_headroom"
    )
    bounded_cases = sorted(
        case["case_id"] for case in cluster_summary["cases"] if case["label"] == "bounded_local_residual"
    )
    legacy_gap_cases = sorted(
        case["case_id"] for case in cluster_summary["cases"] if case["label"] == "legacy_gap_dominant"
    )

    repeated_group = cluster_summary["cluster_groups"].get("repeated_cameras", {})
    frame1170_group = cluster_summary["cluster_groups"].get("frame1170_trio", {})

    brief = {
        "status": "ready_for_new_training_question",
        "problem_family": "source_policy_rule_regularization",
        "recommended_problem_id": "zju_6src_source_policy_regularization_v1",
        "main_manifest_version": str(manifest.get("frozen_manifest_version", "v9")),
        "baseline_config": DEFAULT_BASELINE_CONFIG,
        "candidate_config": "",
        "candidate_config_status": "not_implemented_yet",
        "geometry_direction": "keep depth + camera -> unproject -> render",
        "question": (
            "Can a single training-time or data-time source-policy rule reduce the post-v9 "
            "6src residual cluster without harming fg_human quality, instead of collecting "
            "narrow frame-aware overrides?"
        ),
        "why_now": [
            "The post-v9 residual cluster is now bounded locally.",
            "The single allowed bounded-search round was consumed without producing a reusable shared donor pattern.",
            "The next step is no longer best framed as another manual override family.",
        ],
        "local_evidence": {
            "residual_case_count": int(cluster_summary["case_count"]),
            "headroom_cases": headroom_cases,
            "bounded_local_residual_cases": bounded_cases,
            "legacy_gap_dominant_cases": legacy_gap_cases,
            "cluster_groups": cluster_summary["cluster_groups"],
            "case_labels": case_labels,
            "repeated_cameras_label_counts": repeated_group.get("label_counts", {}),
            "frame1170_trio_label_counts": frame1170_group.get("label_counts", {}),
        },
        "acceptance_cases": [
            {
                "seq_name": case["seq_name"],
                "frame_id": case["frame_id"],
                "target_camera": case["target_camera"],
                "label": case["label"],
            }
            for case in cluster_summary["cases"]
        ],
        "acceptance_metrics": [
            "full_depth_minus_point_mae",
            "fg_human_depth_minus_point_mae",
            "bg_far_depth_minus_point_mae",
            "bg_bottom_band_depth_minus_point_mae",
            "legacy_gap_depth_minus_point",
        ],
        "non_goals": [
            "Do not reopen global zju_min_depth_conf sweeps.",
            "Do not reopen reliable-region or bottom-only auxiliary unproject losses.",
            "Do not restore the ghost stack.",
            "Do not collect another manual override family by default.",
        ],
        "required_before_cloud": [
            "Define an actual candidate implementation for source-policy regularization or source-selection learning.",
            "Materialize a candidate config path instead of leaving candidate_config empty.",
            "Keep cloud_gate=false until the new training question is explicitly approved.",
        ],
        "cloud_gate": False,
        "launch_cloud_now": False,
    }

    # Allow a repo-level template to provide the stable framing while the nightly
    # fills in the dynamic local evidence from the latest residual cluster.
    for key, value in template.items():
        if key in {"acceptance_cases", "local_evidence", "cloud_gate", "launch_cloud_now"}:
            continue
        brief[key] = value

    brief["local_evidence"] = {
        **brief.get("local_evidence", {}),
        "residual_case_count": int(cluster_summary["case_count"]),
        "headroom_cases": headroom_cases,
        "bounded_local_residual_cases": bounded_cases,
        "legacy_gap_dominant_cases": legacy_gap_cases,
        "cluster_groups": cluster_summary["cluster_groups"],
        "case_labels": case_labels,
        "repeated_cameras_label_counts": repeated_group.get("label_counts", {}),
        "frame1170_trio_label_counts": frame1170_group.get("label_counts", {}),
    }
    brief["acceptance_cases"] = [
        {
            "seq_name": case["seq_name"],
            "frame_id": case["frame_id"],
            "target_camera": case["target_camera"],
            "label": case["label"],
        }
        for case in cluster_summary["cases"]
    ]
    brief["cloud_gate"] = False
    brief["launch_cloud_now"] = False
    return brief


def write_next_training_question_brief(run_dir: Path, manifest: dict, cluster_summary: dict) -> tuple[Path, Path]:
    brief = build_next_training_question_brief(manifest, cluster_summary)
    json_path = run_dir / "next_training_question_brief.json"
    md_path = run_dir / "next_training_question_brief.md"
    write_json(json_path, brief)

    lines = [
        "# ZJU next training question brief",
        "",
        f"- status: `{brief['status']}`",
        f"- problem_family: `{brief['problem_family']}`",
        f"- recommended_problem_id: `{brief['recommended_problem_id']}`",
        f"- main_manifest_version: `{brief['main_manifest_version']}`",
        f"- baseline_config: `{brief['baseline_config']}`",
        f"- candidate_config_status: `{brief['candidate_config_status']}`",
        f"- cloud_gate: `{brief['cloud_gate']}`",
        f"- launch_cloud_now: `{brief['launch_cloud_now']}`",
        "",
        "## Question",
        "",
        brief["question"],
        "",
        "## Why Now",
        "",
    ]
    lines.extend(f"- {item}" for item in brief["why_now"])
    lines.extend(["", "## Acceptance Cases", ""])
    lines.extend(
        f"- {item['seq_name']} / frame {item['frame_id']} / {item['target_camera']} ({item['label']})"
        for item in brief["acceptance_cases"]
    )
    lines.extend(["", "## Non Goals", ""])
    lines.extend(f"- {item}" for item in brief["non_goals"])
    lines.extend(["", "## Required Before Cloud", ""])
    lines.extend(f"- {item}" for item in brief["required_before_cloud"])
    write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def build_cluster_summary(
    manifest: dict,
    resolved_cases: list[dict],
    compare_v9_vs_ref: dict,
    legacy_summary: dict,
) -> dict:
    compare_rows = keyed_rows(compare_v9_vs_ref.get("matched_cases", []))
    legacy_rows = keyed_rows(legacy_summary.get("rows", []))
    cases = []
    for item in resolved_cases:
        key = make_case_key(item["seq_name"], item["frame_id"], item["target_camera"])
        compare_row = compare_rows[key]
        legacy_row = legacy_rows.get(key)
        label = classify_case(compare_row, legacy_row)
        cases.append(
            {
                "case_id": item["case_id"],
                "seq_name": item["seq_name"],
                "frame_id": item["frame_id"],
                "target_camera": item["target_camera"],
                "main_view_profile": item["main_view_profile"],
                "reference_view_profile": item["reference_view_profile"],
                "main_summary_json": str(item["main_summary_json"]),
                "baseline_summary_json": str(item["baseline_summary_json"]),
                "reference_summary_json": str(item["reference_summary_json"]),
                "label": label,
                "full_depth_minus_point_mae_delta_v9_to_12src": compare_row.get("full_depth_minus_point_mae_delta"),
                "fg_human_depth_minus_point_mae_delta_v9_to_12src": compare_row.get("fg_human_depth_minus_point_mae_delta"),
                "bg_far_depth_minus_point_mae_delta_v9_to_12src": compare_row.get("bg_far_depth_minus_point_mae_delta"),
                "bg_bottom_band_depth_minus_point_mae_delta_v9_to_12src": compare_row.get("bg_bottom_band_depth_minus_point_mae_delta"),
                "legacy_gap_point": None if legacy_row is None else legacy_row["legacy_gap_point"],
                "legacy_gap_depth": None if legacy_row is None else legacy_row["legacy_gap_depth"],
                "legacy_gap_delta_abs": None if legacy_row is None else legacy_row["gap_delta_abs"],
            }
        )

    case_by_id = {case["case_id"]: case for case in cases}
    groups = {}
    for group_name, group_cases in manifest["cluster_groups"].items():
        group_ids = [
            make_case_id(item["seq_name"], int(item["frame_id"]), item["target_camera"])
            for item in group_cases
        ]
        present = [case_by_id[group_id] for group_id in group_ids if group_id in case_by_id]
        groups[group_name] = {
            "case_ids": group_ids,
            "present_case_ids": [case["case_id"] for case in present],
            "label_counts": dict(Counter(case["label"] for case in present)),
        }
    return {
        "label": str(manifest.get("label", "")),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "cases": cases,
        "cluster_groups": groups,
    }


def choose_search_group(cluster_summary: dict, manifest: dict, state: dict) -> str | None:
    if state.get("bounded_search_consumed", False):
        return None
    case_by_id = {case["case_id"]: case for case in cluster_summary["cases"]}
    for group_name in manifest.get("search_priority", []):
        group = cluster_summary["cluster_groups"].get(group_name, {})
        present_ids = group.get("present_case_ids", [])
        headroom_cases = [
            case_by_id[case_id]
            for case_id in present_ids
            if case_by_id[case_id]["label"] == "shared_source_policy_headroom"
        ]
        if len(headroom_cases) >= 2:
            return group_name
    return None


def write_cluster_summary(run_dir: Path, payload: dict) -> tuple[Path, Path]:
    json_path = run_dir / "cluster_summary.json"
    md_path = run_dir / "cluster_summary.md"
    write_json(json_path, payload)
    lines = [
        "# post-v9 residual cluster summary",
        "",
        f"- case_count: `{payload['case_count']}`",
        "",
        "## Cluster Groups",
        "",
    ]
    for group_name, group in payload["cluster_groups"].items():
        lines.append(f"- {group_name}: `{json.dumps(group['label_counts'], ensure_ascii=False)}`")
    lines.extend(["", "## Cases", ""])
    for case in payload["cases"]:
        lines.append(
            "- {case_id}: `{label}` | fg_human delta `{fg}` | full delta `{full}` | bg_far delta `{bg_far}` | bg_bottom delta `{bg_bottom}` | legacy gap abs `{gap}`".format(
                case_id=case["case_id"],
                label=case["label"],
                fg=case["fg_human_depth_minus_point_mae_delta_v9_to_12src"],
                full=case["full_depth_minus_point_mae_delta_v9_to_12src"],
                bg_far=case["bg_far_depth_minus_point_mae_delta_v9_to_12src"],
                bg_bottom=case["bg_bottom_band_depth_minus_point_mae_delta_v9_to_12src"],
                gap=case["legacy_gap_delta_abs"],
            )
        )
    write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def collect_pattern_keys(search_summary: dict) -> list[tuple]:
    best = search_summary.get("best_guard_pass_variant")
    if not best:
        return []
    added = tuple(sorted(best.get("added_sources_vs_reference", [])))
    removed = tuple(sorted(best.get("removed_sources_vs_reference", [])))
    keys = []
    if added:
        keys.append(("added", added))
    if added or removed:
        keys.append(("swap", removed, added))
    return keys


def build_nearest_ring_sources(target_camera: str, source_count: int, ring_order: list[str]) -> list[str]:
    if not ring_order:
        return []
    if target_camera not in ring_order:
        return []
    target_idx = ring_order.index(target_camera)
    selected = []
    for ring_step in range(1, len(ring_order)):
        for offset in (ring_step, -ring_step):
            camera = ring_order[(target_idx + offset) % len(ring_order)]
            if camera == target_camera or camera in selected:
                continue
            selected.append(camera)
            if len(selected) == source_count:
                return selected
    return selected


def build_nearest_family_candidate_pool(target_camera: str, reference_sources: list[str], ring_order: list[str]) -> list[str]:
    nearest_sources = build_nearest_ring_sources(target_camera, len(reference_sources), ring_order)
    return list(dict.fromkeys(reference_sources + nearest_sources))


def run_search_case(
    case: dict,
    search_dir: Path,
    main_round: dict,
    python_exe: str,
) -> Path:
    main_summary = load_json(Path(case["main_summary_json"]))
    reference_sources = [str(camera) for camera in main_summary["case"]["source_cameras"]]
    ring_order = [str(camera) for camera in main_round["manifest"].get("camera_ring_order", [])]
    candidate_pool = build_nearest_family_candidate_pool(case["target_camera"], reference_sources, ring_order)
    report_json = Path(case["main_summary_json"]).parent / "synthetic_report.json"
    out_dir = ensure_dir(search_dir / case["case_id"])
    cmd = [
        python_exe,
        str(SEARCH_PY),
        "--report_json",
        str(report_json),
        "--output_dir",
        str(out_dir),
        "--reference_sources",
        ",".join(reference_sources),
        "--candidate_pool",
        ",".join(candidate_pool),
        "--reference_variant",
        "v9",
        "--max_swaps",
        "1",
        "--fg_human_guard_max_delta",
        "0.001",
        "--skip_existing",
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    summary_json = out_dir / "summary.json"
    if not summary_json.exists():
        raise NightlyError(f"Search summary missing: {summary_json}")
    return summary_json.resolve()


def update_override_manifest(base_manifest: dict, selected_rows: list[dict], manifest_path: Path) -> Path:
    keyed = {}
    for row in base_manifest.get("cases", []):
        key = (str(row["seq_name"]), str(row["view_profile"]), int(row["frame_id"]), str(row["target_camera"]))
        keyed[key] = dict(row)
    for row in selected_rows:
        case = row["case"]
        best = row["search"]["best_guard_pass_variant"]
        key = (case["seq_name"], case["main_view_profile"], case["frame_id"], case["target_camera"])
        keyed[key] = {
            "seq_name": case["seq_name"],
            "view_profile": case["main_view_profile"],
            "frame_id": case["frame_id"],
            "target_camera": case["target_camera"],
            "override_name": f"nightly_{best['variant_name']}",
            "source_cameras": list(best["source_cameras"]),
        }
    updated = {
        "label": f"{base_manifest.get('label', 'v9')}_v10_nightly",
        "cases": sorted(
            keyed.values(),
            key=lambda item: (str(item["view_profile"]), int(item["frame_id"]), str(item["target_camera"])),
        ),
    }
    write_json(manifest_path, updated)
    return manifest_path


def run_candidate_rollout(main_round: dict, override_manifest_path: Path, output_root: Path, python_exe: str) -> Path:
    manifest = main_round["manifest"]
    template_reports = [str(repo_resolve(profile["template_report"])) for profile in manifest.get("profiles", [])]
    cmd = [
        python_exe,
        str(SWEEP_PY),
        "--template_reports",
        *template_reports,
        "--local_zju_root",
        str(repo_resolve(manifest["local_zju_root"])),
        "--checkpoint",
        str(manifest.get("checkpoint", "")),
        "--output_root",
        str(output_root),
        "--frame_ids",
        ",".join(str(frame_id) for frame_id in manifest.get("frame_ids", [])),
        "--target_cameras",
        ",".join(str(camera) for camera in manifest.get("target_cameras", [])),
        "--source_policy",
        str(manifest.get("source_policy", "uniform_ring")),
        "--source_override_json",
        str(override_manifest_path),
        "--skip_existing",
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    summary_json = output_root / "summary.json"
    if not summary_json.exists():
        raise NightlyError(f"Candidate sweep summary missing: {summary_json}")
    return summary_json.resolve()


def run_sweep_compare(round_a: Path, round_b: Path, output_dir: Path, python_exe: str) -> Path:
    cmd = [
        python_exe,
        str(SWEEP_COMPARE_PY),
        "--round_a",
        str(round_a),
        "--round_b",
        str(round_b),
        "--label_a",
        "v9",
        "--label_b",
        "v10_candidate",
        "--output_dir",
        str(output_dir),
    ]
    run_checked(cmd, cwd=REPO_ROOT)
    comparison_json = output_dir / "comparison.json"
    if not comparison_json.exists():
        raise NightlyError(f"Sweep comparison missing: {comparison_json}")
    return comparison_json.resolve()


def run_bounded_search_phase(
    run_dir: Path,
    cluster_summary: dict,
    manifest: dict,
    main_round: dict,
    state: dict,
    python_exe: str,
) -> dict:
    selected_group = state.get("pending_bounded_search_group")
    if not selected_group:
        raise NightlyError("bounded_search mode requested but no pending_bounded_search_group is set.")
    cases_by_id = {case["case_id"]: case for case in cluster_summary["cases"]}
    group_case_ids = cluster_summary["cluster_groups"][selected_group]["present_case_ids"]
    target_cases = [
        cases_by_id[case_id]
        for case_id in group_case_ids
        if cases_by_id[case_id]["label"] == "shared_source_policy_headroom"
    ]
    search_root = ensure_dir(run_dir / "bounded_search")
    search_runs = []
    pattern_counts = Counter()
    for case in target_cases:
        summary_json = run_search_case(case, search_root, main_round, python_exe)
        summary = load_json(summary_json)
        search_runs.append({"case": case, "summary_json": str(summary_json), "search": summary})
        for pattern in collect_pattern_keys(summary):
            pattern_counts[pattern] += 1

    selected_pattern = None
    for pattern, count in pattern_counts.most_common():
        if count >= 2:
            selected_pattern = pattern
            break

    decision = {
        "state": "S4_DECIDE_STOP_OR_CLOUD_READY",
        "latest_decision": "ready_for_new_training_question",
        "preflight_ok": True,
        "input_frozen_ok": True,
        "residual_case_count": cluster_summary["case_count"],
        "cluster_groups": cluster_summary["cluster_groups"],
        "case_labels": {case["case_id"]: case["label"] for case in cluster_summary["cases"]},
        "allow_bounded_search": False,
        "bounded_search_group": selected_group,
        "allow_manifest_upgrade": False,
        "next_manifest_version": None,
        "patch_collection_stop": True,
        "ready_for_new_training_question": True,
        "cloud_gate": False,
        "launch_cloud_now": False,
        "reason": "",
        "artifacts": {},
    }
    decision["artifacts"]["bounded_search_root"] = str(search_root.resolve())

    if selected_pattern is None:
        decision["reason"] = "bounded search consumed; no shared donor pattern across at least two headroom cases"
        return decision

    matching_runs = [
        item
        for item in search_runs
        if selected_pattern in collect_pattern_keys(item["search"])
    ]
    improved_runs = []
    for item in matching_runs:
        current_summary = load_json(Path(item["case"]["main_summary_json"]))
        current_decision = str(current_summary["decision"]["decision"])
        candidate_decision = str(item["search"]["best_guard_pass_variant"]["decision"])
        fg_delta = float(item["search"]["best_guard_pass_variant"]["fg_human_depth_minus_point_mae_delta_vs_reference"])
        if current_decision == "point_map" and candidate_decision == "depth_unproject" and fg_delta <= 0.001:
            improved_runs.append(item)

    if len(improved_runs) < 2:
        decision["reason"] = "bounded search consumed; shared donor pattern exists but fewer than two residuals improve from point_map to depth_unproject"
        return decision

    generated_dir = ensure_dir(run_dir / "generated_manifests")
    base_override_manifest = load_json(repo_resolve(manifest["main_override_manifest"]))
    v10_manifest_path = update_override_manifest(
        base_override_manifest,
        improved_runs,
        generated_dir / "zju_6src_hardcontrol_hybrid_v10_post_v9_nightly.json",
    )
    candidate_sweep_root = ensure_dir(run_dir / "candidate_v10_sweep")
    candidate_summary = run_candidate_rollout(main_round, v10_manifest_path, candidate_sweep_root, python_exe)
    comparison_json = run_sweep_compare(
        main_round["summary_json"],
        candidate_summary,
        run_dir / "candidate_v10_vs_v9_compare",
        python_exe,
    )
    comparison = load_json(comparison_json)
    overall = comparison["overall_common"]["transition_counts"]
    decision["artifacts"]["candidate_manifest"] = str(v10_manifest_path.resolve())
    decision["artifacts"]["candidate_sweep_summary"] = str(candidate_summary)
    decision["artifacts"]["candidate_compare_json"] = str(comparison_json)

    if int(overall["regressed_from_depth"]) != 0:
        decision["reason"] = "bounded search consumed; candidate manifest regressed at least one depth_unproject case during rollout safety"
        return decision

    decision["latest_decision"] = "ready_for_new_training_question"
    decision["allow_manifest_upgrade"] = True
    decision["next_manifest_version"] = "v10"
    decision["patch_collection_stop"] = True
    decision["ready_for_new_training_question"] = True
    decision["reason"] = "bounded search consumed; v10 candidate passed rollout safety locally, but cloud remains locked until a new training question is chosen"
    return decision


def run_diagnose_phase(
    run_dir: Path,
    manifest: dict,
    state_dir: Path,
    state: dict,
) -> tuple[dict, dict]:
    main_round = load_round(repo_resolve(manifest["main_round"]))
    baseline_round = load_round(repo_resolve(manifest["baseline_round"]))
    reference_round = load_round(repo_resolve(manifest["reference_round"]))
    python_exe = str(main_round["manifest"].get("python_exe", "")) or sys.executable
    supplement_root = ensure_dir(state_dir / "supplements")

    resolved_cases = []
    for case in manifest["cases"]:
        seq_name = str(case["seq_name"])
        frame_id = int(case["frame_id"])
        target_camera = str(case["target_camera"])
        baseline_summary_json = ensure_case_summary(
            case,
            baseline_round,
            str(manifest["main_view_profile"]),
            supplement_root,
            "baseline_round12",
        )
        main_summary_json = ensure_case_summary(
            case,
            main_round,
            str(manifest["main_view_profile"]),
            supplement_root,
            "main_round15_v9",
        )
        reference_summary_json = ensure_case_summary(
            case,
            reference_round,
            str(manifest["reference_view_profile"]),
            supplement_root,
            "reference_round3_12src",
        )
        resolved_cases.append(
            {
                "case_id": make_case_id(seq_name, frame_id, target_camera),
                "seq_name": seq_name,
                "frame_id": frame_id,
                "target_camera": target_camera,
                "main_view_profile": str(manifest["main_view_profile"]),
                "reference_view_profile": str(manifest["reference_view_profile"]),
                "baseline_summary_json": baseline_summary_json,
                "main_summary_json": main_summary_json,
                "reference_summary_json": reference_summary_json,
            }
        )

    cluster_manifest = {
        "label": str(manifest.get("label", "")),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": resolved_cases,
    }
    cluster_manifest_path = run_dir / "cluster_manifest.json"
    write_json(cluster_manifest_path, cluster_manifest)

    baseline_region_summary = run_region_summary(
        "baseline6src_region_summary",
        [Path(item["baseline_summary_json"]) for item in resolved_cases],
        run_dir / "baseline6src_region_summary",
        python_exe,
    )
    main_region_summary = run_region_summary(
        "v9_region_summary",
        [Path(item["main_summary_json"]) for item in resolved_cases],
        run_dir / "v9_region_summary",
        python_exe,
    )
    reference_region_summary = run_region_summary(
        "uniform12src_region_summary",
        [Path(item["reference_summary_json"]) for item in resolved_cases],
        run_dir / "uniform12src_region_summary",
        python_exe,
    )
    v9_vs_baseline_json = run_region_compare(
        baseline_region_summary,
        main_region_summary,
        run_dir / "v9_vs_baseline_region_compare",
        "baseline6src",
        "v9",
        "post-v9 residual cluster: baseline6src vs v9",
        python_exe,
        ignore_view_profile=False,
    )
    v9_vs_uniform_json = run_region_compare(
        main_region_summary,
        reference_region_summary,
        run_dir / "v9_vs_uniform12src_region_compare",
        "v9",
        "uniform12src",
        "post-v9 residual cluster: v9 vs 12src uniform",
        python_exe,
        ignore_view_profile=True,
    )

    current_case_manifest = build_current_case_manifest(
        manifest,
        resolved_cases,
        run_dir / "legacy_case_manifest.json",
    )
    legacy_backfill_manifest = run_legacy_backfill(
        current_case_manifest,
        ensure_dir(state_dir / "legacy_backfill_cache"),
        str(manifest.get("legacy_view_profile_tag", "legacy_from_current_manifest")),
    )
    legacy_gap_summary_json = run_legacy_summary(
        legacy_backfill_manifest,
        run_dir / "legacy_gap_summary",
        python_exe,
    )

    compare_v9_vs_ref = load_json(v9_vs_uniform_json)
    legacy_summary = load_json(legacy_gap_summary_json)
    cluster_summary = build_cluster_summary(manifest, resolved_cases, compare_v9_vs_ref, legacy_summary)
    cluster_summary_json, cluster_summary_md = write_cluster_summary(run_dir, cluster_summary)

    decision = {
        "state": "S4_DECIDE_STOP_OR_CLOUD_READY",
        "latest_decision": "continue_diagnose",
        "preflight_ok": True,
        "input_frozen_ok": True,
        "residual_case_count": len(resolved_cases),
        "cluster_groups": cluster_summary["cluster_groups"],
        "case_labels": {case["case_id"]: case["label"] for case in cluster_summary["cases"]},
        "allow_bounded_search": False,
        "bounded_search_group": None,
        "allow_manifest_upgrade": False,
        "next_manifest_version": None,
        "patch_collection_stop": False,
        "ready_for_new_training_question": False,
        "cloud_gate": False,
        "launch_cloud_now": False,
        "reason": "",
        "artifacts": {
            "cluster_manifest_json": str(cluster_manifest_path.resolve()),
            "baseline6src_region_summary_json": str(baseline_region_summary),
            "v9_region_summary_json": str(main_region_summary),
            "uniform12src_region_summary_json": str(reference_region_summary),
            "v9_vs_baseline_region_compare_json": str(v9_vs_baseline_json),
            "v9_vs_uniform12src_region_compare_json": str(v9_vs_uniform_json),
            "legacy_gap_summary_json": str(legacy_gap_summary_json),
            "cluster_summary_json": str(cluster_summary_json.resolve()),
            "cluster_summary_md": str(cluster_summary_md.resolve()),
        },
    }

    search_group = choose_search_group(cluster_summary, manifest, state)
    if search_group is not None:
        decision["latest_decision"] = "ready_for_one_bounded_search"
        decision["allow_bounded_search"] = True
        decision["bounded_search_group"] = search_group
        decision["reason"] = f"diagnose complete; {search_group} has at least two shared_source_policy_headroom cases"
    else:
        decision["latest_decision"] = "ready_for_new_training_question"
        decision["patch_collection_stop"] = True
        decision["ready_for_new_training_question"] = True
        decision["reason"] = "diagnose complete; no cluster group satisfied the bounded-search gate"
    return decision, {
        "manifest": manifest,
        "main_round": main_round,
        "cluster_summary": cluster_summary,
        "python_exe": python_exe,
    }


def main():
    args = parse_args()
    manifest = load_json(repo_resolve(args.manifest))
    state_dir = ensure_dir(repo_resolve(args.state_dir))
    state_path = state_dir / "state.json"
    state = load_state(state_path, manifest)
    run_dir = ensure_dir(repo_resolve(args.output_root) / now_tag())

    try:
        run_preflight(run_dir)
        decision, context = run_diagnose_phase(run_dir, manifest, state_dir, state)
        state["pending_bounded_search_group"] = decision["bounded_search_group"]
        state["latest_decision"] = decision["latest_decision"]
        state["last_reason"] = decision["reason"]
        state["patch_collection_stop"] = bool(decision["patch_collection_stop"])
        state["ready_for_new_training_question"] = bool(decision["ready_for_new_training_question"])
        state["launch_cloud_now"] = bool(decision["launch_cloud_now"])
        state["latest_run_dir"] = str(run_dir.resolve())

        should_run_search = (
            args.mode in {"auto", "bounded_search"}
            and decision["allow_bounded_search"]
            and not state.get("bounded_search_consumed", False)
        )
        if should_run_search:
            search_run_dir = ensure_dir(run_dir / "bounded_search_phase")
            decision = run_bounded_search_phase(
                search_run_dir,
                context["cluster_summary"],
                manifest,
                context["main_round"],
                state,
                context["python_exe"],
            )
            decision["artifacts"]["diagnose_run_dir"] = str(run_dir.resolve())
            state["bounded_search_consumed"] = True
            state["pending_bounded_search_group"] = None
            state["latest_decision"] = decision["latest_decision"]
            state["last_reason"] = decision["reason"]
            state["patch_collection_stop"] = bool(decision["patch_collection_stop"])
            state["ready_for_new_training_question"] = bool(decision["ready_for_new_training_question"])
            state["launch_cloud_now"] = bool(decision["launch_cloud_now"])
            state["latest_run_dir"] = str(search_run_dir.resolve())

        if decision["patch_collection_stop"]:
            cloud_template = generate_cloud_template(run_dir, manifest)
            decision["artifacts"]["cloud_experiment_template_json"] = str(cloud_template.resolve())
            state["generated_cloud_template"] = str(cloud_template.resolve())
            brief_json, brief_md = write_next_training_question_brief(run_dir, manifest, context["cluster_summary"])
            brief_payload = load_json(brief_json)
            decision["current_lead_config"] = str(brief_payload.get("candidate_config", ""))
            decision["current_cloud_blocker"] = str(brief_payload.get("current_cloud_blocker", ""))
            decision["artifacts"]["next_training_question_brief_json"] = str(brief_json.resolve())
            decision["artifacts"]["next_training_question_brief_md"] = str(brief_md.resolve())
            state["generated_training_question_brief"] = str(brief_json.resolve())
            state["current_lead_config"] = str(brief_payload.get("candidate_config", ""))
            state["current_cloud_blocker"] = str(brief_payload.get("current_cloud_blocker", state.get("last_reason", "")))

        write_json(run_dir / "nightly_decision.json", decision)
        write_decision_markdown(run_dir / "nightly_decision.md", decision)
        save_state(state_path, state)
        print(run_dir / "nightly_decision.md")
    except Exception as exc:
        failure = {
            "state": "failed",
            "latest_decision": "continue_diagnose",
            "preflight_ok": False,
            "input_frozen_ok": False,
            "residual_case_count": 0,
            "cluster_groups": {},
            "case_labels": {},
            "allow_bounded_search": False,
            "bounded_search_group": None,
            "allow_manifest_upgrade": False,
            "next_manifest_version": None,
            "patch_collection_stop": False,
            "ready_for_new_training_question": False,
            "cloud_gate": False,
            "launch_cloud_now": False,
            "reason": str(exc),
            "artifacts": {},
        }
        write_json(run_dir / "nightly_decision.json", failure)
        write_decision_markdown(run_dir / "nightly_decision.md", failure)
        state["latest_decision"] = "continue_diagnose"
        state["last_reason"] = str(exc)
        state["patch_collection_stop"] = False
        state["ready_for_new_training_question"] = False
        state["launch_cloud_now"] = False
        state["latest_run_dir"] = str(run_dir.resolve())
        save_state(state_path, state)
        raise


if __name__ == "__main__":
    main()
