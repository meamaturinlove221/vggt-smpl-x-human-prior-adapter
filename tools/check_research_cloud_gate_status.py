from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.research_cloud_common import (
    REQUIRED_RESEARCH_FLAGS,
    default_research_metadata,
    json_ready,
    repo_root,
    validate_research_metadata,
    write_json,
)


REPO_ROOT = repo_root()
SCHEMA_VERSION = "20260507_v8_research_cloud_manifest_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "research_cloud_gate_probe"
DEFAULT_MANIFEST = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "v8_research_cloud_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "20260507_v8_research_cloud_gate_status.md"
DEFAULT_JSON = REPO_ROOT / "reports" / "20260507_v8_research_cloud_gate_status.json"


LANES = {
    "b_fus3d2_human_dataset_train": ("Cloud-A", "B-Fus3D2 human dataset train smoke"),
    "a5x_external_dense_teacher_intake": ("Cloud-B", "A5-X external dense teacher intake smoke"),
    "b_hand10_hggt_style_hand_decoder": ("Cloud-C", "B-hand10 HGGT-style hand decoder smoke"),
    "b_hair3_hairgs_topology": ("Cloud-D", "B-hair3 HairGS-style topology smoke"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only cloud gate for bounded V8 preflight jobs. This is separate from the formal strict "
            "candidate/teacher guard and never permits prediction, teacher, candidate, registry, or pass export."
        )
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lane", choices=sorted(LANES), default="b_fus3d2_human_dataset_train")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--write-default-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_formal_guard(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, *args], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = int(proc.returncode)
    return payload


def default_manifest(output_root: Path, *, max_steps: int, max_cases: int, max_hours: float) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for lane, (job_id, job_name) in LANES.items():
        output_dir = output_root.parent / job_id.replace("-", "_") / lane
        metadata = default_research_metadata(
            job_id=job_id,
            job_name=job_name,
            output_dir=output_dir,
            max_steps=max_steps,
            max_cases=max_cases,
            max_hours=max_hours,
        )
        metadata.update({"lane": lane, "post_job_referee_required": True})
        jobs.append(metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        **REQUIRED_RESEARCH_FLAGS,
        "created_by": "tools/check_research_cloud_gate_status.py",
        "jobs": jobs,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        reasons.append("manifest schema_version is not current")
    for key, expected in REQUIRED_RESEARCH_FLAGS.items():
        if manifest.get(key) is not expected:
            reasons.append(f"manifest {key} is not {expected}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        reasons.append("manifest jobs must be a non-empty list")
        return reasons
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            reasons.append(f"jobs[{index}] is not an object")
            continue
        job_id = str(job.get("job_id", ""))
        if job_id in seen:
            reasons.append(f"duplicate job_id {job_id}")
        seen.add(job_id)
        if job.get("lane") not in LANES:
            reasons.append(f"{job_id or index}: unsupported lane {job.get('lane')!r}")
        if job.get("post_job_referee_required") is not True:
            reasons.append(f"{job_id or index}: post_job_referee_required is not true")
        reasons.extend(f"{job_id or index}: {reason}" for reason in validate_research_metadata(job))
    return reasons


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Research Cloud Gate Status",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This gate only allows bounded research-cloud preflight jobs under",
        "`output/surface_research_cloud_preflight`. It does not alter the formal strict guard.",
        "",
        "## Formal Guard",
        "",
        "```json",
        json.dumps(summary["formal_guards"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Research Decision",
        "",
        "```json",
        json.dumps(summary["research_gate"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Jobs",
        "",
        "```json",
        json.dumps(summary["jobs"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.manifest is None:
        manifest = default_manifest(args.output_dir, max_steps=args.max_steps, max_cases=args.max_cases, max_hours=args.max_hours)
        manifest_path = DEFAULT_MANIFEST
        if args.write_default_manifest:
            write_json(manifest_path, manifest)
    else:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_manifest(manifest_path)

    formal_candidate = run_formal_guard(["tools/check_cloud_gate_status.py", "--json"])
    formal_teacher = run_formal_guard(["tools/check_cloud_gate_status.py", "--teacher-supervised", "--json"])
    manifest_reasons = validate_manifest(manifest)
    formal_still_blocked = (
        bool(formal_candidate.get("cloud_allowed")) is False
        and bool(formal_teacher.get("cloud_allowed")) is False
        and int(formal_candidate.get("strict_candidate_passes", 0) or 0) == 0
        and int(formal_teacher.get("strict_teacher_passes", 0) or 0) == 0
    )
    research_allowed = not manifest_reasons
    summary = {
        "status": "research_cloud_preflight_allowed" if research_allowed else "research_cloud_preflight_blocked",
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "formal_guards": {"candidate": formal_candidate, "teacher_supervised": formal_teacher},
        "formal_guard_still_blocked": formal_still_blocked,
        "research_gate": {
            "research_cloud_train": research_allowed,
            "research_cloud_infer": research_allowed,
            "research_cloud_artifact_download": research_allowed,
            "allowed": research_allowed,
            "reasons": manifest_reasons,
            "required_post_job_referee": True,
            "forbidden_outputs": [
                "predictions.npz",
                "candidate package",
                "teacher package",
                "strict registry write",
                "strict pass write",
            ],
        },
        "jobs": manifest.get("jobs", []),
        "manifest": manifest,
    }
    write_json(args.report_json, summary)
    write_report(args.report_md, summary)
    if args.json:
        print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(json_ready({"status": summary["status"], "research_allowed": research_allowed, "formal_guard_still_blocked": formal_still_blocked, "reasons": manifest_reasons}), indent=2, ensure_ascii=False))
    return 0 if research_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
