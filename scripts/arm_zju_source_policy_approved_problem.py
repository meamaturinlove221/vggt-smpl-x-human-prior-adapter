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
