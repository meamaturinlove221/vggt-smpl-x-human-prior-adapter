import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
DEFAULT_APPROVED_PROBLEM_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.json"
DEFAULT_RESEARCH_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "research_loop_status.json"
RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
DEFAULT_INTERP_SMOOTHSTEP_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.interpolated_eligibility_shaping.smoothstep_taper.json"
)
DEFAULT_INTERP_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.interpolated_eligibility_shaping.json"
DEFAULT_PARTIAL_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.partial_joint_depth_routing.json"
DEFAULT_DISAGREEMENT_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.conf_reg_disagreement_routing.json"
DEFAULT_UNPROJECT_CONSISTENCY_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.unproject_consistency_routing.json"
)
DEFAULT_UNPROJECT_AUX_CONFGATE_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.unproject_aux_confgate.json"
)
DEFAULT_SOURCE_POLICY_HYBRID_RING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.source_policy_hybrid_ring_regularization.json"
)
DEFAULT_RESIDUAL_CASE_COVERAGE_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.residual_case_coverage_rebalancing.json"
)
DEFAULT_HARDTAIL_BUCKET_GRANULARITY_REFINEMENT_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.hardtail_bucket_granularity_refinement.json"
)
DEFAULT_SOFT_TAIL_EXPOSURE_REBALANCING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.soft_tail_exposure_rebalancing.json"
)
DEFAULT_HYBRID_TAIL_EXPOSURE_BALANCING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.hybrid_tail_exposure_balancing.json"
)
DEFAULT_TAIL_CONF_BRANCH_DECOUPLING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_conf_branch_decoupling.json"
)
DEFAULT_TAIL_SOURCE_POOL_TEMPERING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_source_pool_tempering.json"
)
DEFAULT_TAIL_ANCHOR_STABILIZATION_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_anchor_stabilization.json"
)
DEFAULT_TAIL_POSE_BRANCH_DECOUPLING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_pose_branch_decoupling.json"
)
DEFAULT_TAIL_INTRINSICS_BRANCH_DECOUPLING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_intrinsics_branch_decoupling.json"
)
DEFAULT_TAIL_COUNTERBALANCE_COHORT_MIXING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_counterbalance_cohort_mixing.json"
)
DEFAULT_TAIL_ANCHOR_RESERVE_HYBRIDIZATION_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_anchor_reserve_hybridization.json"
)
DEFAULT_TAIL_MANIFEST_FOCAL_REINFORCEMENT_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_manifest_focal_reinforcement.json"
)
DEFAULT_TAIL_STREAM_SELECTIVE_FOCAL_REINFORCEMENT_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_stream_selective_focal_reinforcement.json"
)
DEFAULT_TAIL_CONTRACT_ANCHOR_REPLAY_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_contract_anchor_replay.json"
)
DEFAULT_TAIL_CONTRACT_VIEWSET_REPLAY_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_contract_viewset_replay.json"
)
DEFAULT_TAIL_DUAL_SUPERVISION_REBALANCING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.tail_dual_supervision_rebalancing.json"
)
DEFAULT_DEFAULT_STREAM_INTRINSICS_COUNTERBALANCE_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.default_stream_intrinsics_counterbalance.json"
)
DEFAULT_TWO_STAGE_OBJECTIVE_DECOUPLING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.two_stage_objective_decoupling.json"
)
DEFAULT_CAMERA_FOCAL_OBJECTIVE_ISOLATION_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.camera_focal_objective_isolation.json"
)
DEFAULT_CAMERA_TRANSLATION_OBJECTIVE_ISOLATION_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.camera_translation_objective_isolation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Arm exactly one approved_problem.json from an existing research-loop seed without widening the "
            "single-problem single-candidate overnight contract."
        )
    )
    parser.add_argument(
        "--seed",
        choices=[
            "source_policy_hybrid_ring_regularization",
            "residual_case_coverage_rebalancing",
            "hardtail_bucket_granularity_refinement",
            "soft_tail_exposure_rebalancing",
            "hybrid_tail_exposure_balancing",
            "tail_conf_branch_decoupling",
            "tail_source_pool_tempering",
            "tail_anchor_stabilization",
            "tail_pose_branch_decoupling",
            "tail_intrinsics_branch_decoupling",
            "tail_counterbalance_cohort_mixing",
            "tail_anchor_reserve_hybridization",
            "tail_manifest_focal_reinforcement",
            "tail_stream_selective_focal_reinforcement",
            "tail_contract_anchor_replay",
            "tail_contract_viewset_replay",
            "tail_dual_supervision_rebalancing",
            "default_stream_intrinsics_counterbalance",
            "two_stage_objective_decoupling",
            "camera_focal_objective_isolation",
            "camera_translation_objective_isolation",
            "unproject_aux_confgate",
            "unproject_consistency_routing",
            "conf_reg_disagreement_routing",
        ],
        default="source_policy_hybrid_ring_regularization",
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--approved-problem-path", type=Path, default=DEFAULT_APPROVED_PROBLEM_PATH)
    parser.add_argument(
        "--approval-note",
        default="Manually approved single-problem single-candidate research loop run.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_path_for_name(seed_name: str) -> Path:
    if seed_name == "source_policy_hybrid_ring_regularization":
        return DEFAULT_SOURCE_POLICY_HYBRID_RING_SEED_PATH
    if seed_name == "residual_case_coverage_rebalancing":
        return DEFAULT_RESIDUAL_CASE_COVERAGE_SEED_PATH
    if seed_name == "hardtail_bucket_granularity_refinement":
        return DEFAULT_HARDTAIL_BUCKET_GRANULARITY_REFINEMENT_SEED_PATH
    if seed_name == "soft_tail_exposure_rebalancing":
        return DEFAULT_SOFT_TAIL_EXPOSURE_REBALANCING_SEED_PATH
    if seed_name == "hybrid_tail_exposure_balancing":
        return DEFAULT_HYBRID_TAIL_EXPOSURE_BALANCING_SEED_PATH
    if seed_name == "tail_conf_branch_decoupling":
        return DEFAULT_TAIL_CONF_BRANCH_DECOUPLING_SEED_PATH
    if seed_name == "tail_source_pool_tempering":
        return DEFAULT_TAIL_SOURCE_POOL_TEMPERING_SEED_PATH
    if seed_name == "tail_anchor_stabilization":
        return DEFAULT_TAIL_ANCHOR_STABILIZATION_SEED_PATH
    if seed_name == "tail_pose_branch_decoupling":
        return DEFAULT_TAIL_POSE_BRANCH_DECOUPLING_SEED_PATH
    if seed_name == "tail_intrinsics_branch_decoupling":
        return DEFAULT_TAIL_INTRINSICS_BRANCH_DECOUPLING_SEED_PATH
    if seed_name == "tail_counterbalance_cohort_mixing":
        return DEFAULT_TAIL_COUNTERBALANCE_COHORT_MIXING_SEED_PATH
    if seed_name == "tail_anchor_reserve_hybridization":
        return DEFAULT_TAIL_ANCHOR_RESERVE_HYBRIDIZATION_SEED_PATH
    if seed_name == "tail_manifest_focal_reinforcement":
        return DEFAULT_TAIL_MANIFEST_FOCAL_REINFORCEMENT_SEED_PATH
    if seed_name == "tail_stream_selective_focal_reinforcement":
        return DEFAULT_TAIL_STREAM_SELECTIVE_FOCAL_REINFORCEMENT_SEED_PATH
    if seed_name == "tail_contract_anchor_replay":
        return DEFAULT_TAIL_CONTRACT_ANCHOR_REPLAY_SEED_PATH
    if seed_name == "tail_contract_viewset_replay":
        return DEFAULT_TAIL_CONTRACT_VIEWSET_REPLAY_SEED_PATH
    if seed_name == "tail_dual_supervision_rebalancing":
        return DEFAULT_TAIL_DUAL_SUPERVISION_REBALANCING_SEED_PATH
    if seed_name == "default_stream_intrinsics_counterbalance":
        return DEFAULT_DEFAULT_STREAM_INTRINSICS_COUNTERBALANCE_SEED_PATH
    if seed_name == "two_stage_objective_decoupling":
        return DEFAULT_TWO_STAGE_OBJECTIVE_DECOUPLING_SEED_PATH
    if seed_name == "camera_focal_objective_isolation":
        return DEFAULT_CAMERA_FOCAL_OBJECTIVE_ISOLATION_SEED_PATH
    if seed_name == "camera_translation_objective_isolation":
        return DEFAULT_CAMERA_TRANSLATION_OBJECTIVE_ISOLATION_SEED_PATH
    if seed_name == "unproject_aux_confgate":
        return DEFAULT_UNPROJECT_AUX_CONFGATE_SEED_PATH
    if seed_name == "unproject_consistency_routing":
        return DEFAULT_UNPROJECT_CONSISTENCY_SEED_PATH
    if seed_name == "conf_reg_disagreement_routing":
        return DEFAULT_DISAGREEMENT_SEED_PATH
    raise ValueError(f"Unknown seed: {seed_name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_live_truth_for_arm(payload: dict, *, force: bool) -> None:
    if force or not DEFAULT_RESEARCH_STATUS_PATH.exists():
        return
    research_status = load_json(DEFAULT_RESEARCH_STATUS_PATH)
    family = str(payload.get("family", "")).strip()
    allowed_families = [str(item).strip() for item in (research_status.get("allowed_families", []) or []) if str(item).strip()]
    current_priority_family = str(research_status.get("current_priority_family", "")).strip()

    if allowed_families and family not in allowed_families:
        raise SystemExit(
            f"Refusing to arm {family}: live truth currently allows only {allowed_families}."
        )
    if current_priority_family and family != current_priority_family:
        raise SystemExit(
            f"Refusing to arm {family}: live truth current_priority_family is {current_priority_family}."
        )
    if not allowed_families and not current_priority_family:
        raise SystemExit(
            f"Refusing to arm {family}: live truth currently selects no family for arming."
        )


def import_research_loop_module():
    spec = importlib.util.spec_from_file_location("zju_research_loop_module", RESEARCH_LOOP_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load research-loop module from {RESEARCH_LOOP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_payload(payload: dict) -> None:
    research_loop = import_research_loop_module()
    issues = research_loop.validate_approved_problem(
        payload,
        max_approved_problems_per_night=1,
        max_candidates_per_problem=1,
    )
    if issues:
        raise SystemExit(
            "Refusing to arm approved_problem.json because the single-problem single-candidate contract "
            f"failed validation: {issues}"
        )


def refresh_research_loop(python_exe: str) -> None:
    result = subprocess.run(
        [python_exe, str(RESEARCH_LOOP_SCRIPT)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "approved_problem.json was armed, but research-loop status refresh failed.\n"
            f"STDOUT:\n{result.stdout.strip()}\n"
            f"STDERR:\n{result.stderr.strip()}"
        )


def main() -> int:
    args = parse_args()
    seed_path = seed_path_for_name(args.seed)
    payload = load_json(seed_path)
    payload["approved"] = True
    payload["approved_at"] = datetime.now().isoformat(timespec="seconds")
    payload["approval_note"] = args.approval_note
    payload["approval_source_seed"] = str(seed_path.resolve())
    validate_payload(payload)
    validate_live_truth_for_arm(payload, force=args.force)

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.approved_problem_path.exists() and not args.force:
        raise SystemExit(
            "approved_problem.json already exists; refuse to overwrite an active approval without --force."
        )

    write_json(args.approved_problem_path, payload)
    refresh_research_loop(args.python_exe)
    print(f"[armed] {args.approved_problem_path}")
    print(f"[seed] {seed_path}")
    print(f"[refreshed] {RESEARCH_LOOP_SCRIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
