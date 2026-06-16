from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from tools.research_cloud_common import (
        FORBIDDEN_PATH_WORDS,
        default_research_metadata,
        normalize_output_dir,
        now_utc,
        validate_research_metadata,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - keeps the script runnable from tools/.
    from research_cloud_common import (  # type: ignore
        FORBIDDEN_PATH_WORDS,
        default_research_metadata,
        normalize_output_dir,
        now_utc,
        validate_research_metadata,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_GUARD = REPO_ROOT / "tools" / "check_research_cloud_gate_status.py"
MANIFEST_ROOT = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "launch_manifest_agent"
REPORT_MD = REPO_ROOT / "reports" / "20260507_v8_research_cloud_launch_agent_status.md"
REPORT_JSON = REPO_ROOT / "reports" / "20260507_v8_research_cloud_launch_agent_status.json"

DEFAULT_SCENE_SUBDIR = "surface_research_assets/0012_11_frame0000_60views_human_crop"
DEFAULT_MESH_SEED_SUBPATH = "surface_research_assets/a3_visual_hull_mesh_project_t96_g56_s4_mesh.npz"
DEFAULT_TEMPLATE_PAYLOAD_SUBPATH = "surface_research_assets/connected_human_surface_template_payload.npz"

FORBIDDEN_JOB_WORDS = tuple(FORBIDDEN_PATH_WORDS) + ("registry", "teacher", "candidate")


@dataclass(frozen=True)
class ResearchJob:
    job_id: str
    job_name: str
    cloud_lane: str
    modal_lane: str
    output_leaf: str
    max_steps: int
    max_cases: int
    max_hours: float
    target_size: int
    view_indices: str
    gpu: str = "A100-40GB"
    mesh_seed_subpath: str = ""
    template_payload_subpath: str = ""
    eval_view_indices: str = ""
    extra_args: dict[str, str | int | float | bool] = field(default_factory=dict)

    def output_dir(self) -> Path:
        return MANIFEST_ROOT / self.output_leaf

    def modal_output_subdir(self) -> str:
        return f"surface_research_cloud_preflight/launch_manifest_agent/{self.output_leaf}"


DEFAULT_JOBS: tuple[ResearchJob, ...] = (
    ResearchJob(
        job_id="cloud_a_a5_colmap_cuda_v8_tri_adj6",
        job_name="Cloud-A bounded A5 COLMAP CUDA v8 adjacency-6 smoke",
        cloud_lane="Cloud-A",
        modal_lane="A5_known_camera_colmap_workspace",
        output_leaf="cloud_a_a5_colmap_cuda_v8_tri_adj6",
        max_steps=1,
        max_cases=1,
        max_hours=0.75,
        target_size=256,
        view_indices="0,1,2,3,4,5",
        extra_args={
            "dense_mode": "triangulated",
            "execute_colmap": True,
            "a5_dry_run": False,
        },
    ),
    ResearchJob(
        job_id="cloud_b_a4_neus_sdf_t48_step16",
        job_name="Cloud-B bounded A4 NeuS SDF smoke",
        cloud_lane="Cloud-B",
        modal_lane="A4_neus_sdf_surface",
        output_leaf="cloud_b_a4_neus_sdf_t48_step16",
        max_steps=16,
        max_cases=1,
        max_hours=0.5,
        target_size=48,
        view_indices="0,30",
        mesh_seed_subpath=DEFAULT_MESH_SEED_SUBPATH,
        eval_view_indices="10,40",
        extra_args={
            "steps": 16,
            "ray_batch_size": 2048,
            "samples_per_ray": 48,
            "hidden_dim": 96,
            "pos_frequencies": 5,
            "sdf_grid": 48,
            "sdf_beta": 0.045,
            "density_scale": 38.0,
            "surface_entropy_weight": 0.01,
            "eikonal_weight": 0.02,
            "residual_scale": 0.28,
        },
    ),
    ResearchJob(
        job_id="cloud_c_a4_part_local_sdf_t48_step16",
        job_name="Cloud-C bounded A4.1 part-local SDF smoke",
        cloud_lane="Cloud-C",
        modal_lane="A4_1_part_local_sdf",
        output_leaf="cloud_c_a4_part_local_sdf_t48_step16",
        max_steps=16,
        max_cases=1,
        max_hours=0.5,
        target_size=48,
        view_indices="0,30",
        mesh_seed_subpath=DEFAULT_MESH_SEED_SUBPATH,
        eval_view_indices="10,40",
        extra_args={
            "steps": 16,
            "ray_batch_size": 2048,
            "samples_per_ray": 48,
            "hidden_dim": 96,
            "pos_frequencies": 6,
            "sdf_grid": 48,
            "sdf_beta": 0.035,
            "density_scale": 45.0,
            "surface_entropy_weight": 0.01,
            "eikonal_weight": 0.03,
            "residual_scale": 0.35,
            "part_carriers": "head_hair_hands",
        },
    ),
    ResearchJob(
        job_id="cloud_d_b2_surface_tokens_t96_step4",
        job_name="Cloud-D bounded B2 surface-token diagnostic smoke",
        cloud_lane="Cloud-D",
        modal_lane="B2_surface_tokens",
        output_leaf="cloud_d_b2_surface_tokens_t96_step4",
        max_steps=4,
        max_cases=1,
        max_hours=0.35,
        target_size=96,
        view_indices="0,10,20,30,40,50",
        template_payload_subpath=DEFAULT_TEMPLATE_PAYLOAD_SUBPATH,
        extra_args={
            "steps": 4,
            "token_grid": 5,
            "family_token_grid_overrides": "hand=2,face=2,hair=3",
            "token_hidden": 64,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or launch bounded V8 research-only Modal cloud jobs. The script "
            "fails closed unless tools/check_research_cloud_gate_status.py exists "
            "and explicitly allows research cloud."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "launch"),
        default="dry-run",
        help="dry-run writes manifests/reports only; launch also calls Modal after the research guard allows it.",
    )
    parser.add_argument(
        "--lanes",
        default="Cloud-A,Cloud-B,Cloud-C,Cloud-D",
        help="Comma-separated Cloud-A/B/C/D lanes to include.",
    )
    parser.add_argument("--scene-subdir", default=DEFAULT_SCENE_SUBDIR)
    parser.add_argument("--manifest-root", default=str(MANIFEST_ROOT))
    parser.add_argument("--report-md", default=str(REPORT_MD))
    parser.add_argument("--report-json", default=str(REPORT_JSON))
    parser.add_argument("--guard", default=str(RESEARCH_GUARD))
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--allow-launch",
        action="store_true",
        help="Required with --mode launch to make accidental cloud submission a two-key action.",
    )
    return parser.parse_args()


def split_lanes(value: str) -> set[str]:
    lanes = {item.strip() for item in value.split(",") if item.strip()}
    unknown = lanes.difference({job.cloud_lane for job in DEFAULT_JOBS})
    if unknown:
        raise ValueError(f"Unsupported cloud lanes: {', '.join(sorted(unknown))}")
    return lanes


def check_forbidden_text(label: str, value: str) -> None:
    lowered = value.replace("\\", "/").lower()
    for word in FORBIDDEN_JOB_WORDS:
        if word in lowered:
            raise ValueError(f"{label} contains forbidden research-only token {word!r}: {value}")


def guard_status(guard_path: Path, max_age_hours: float) -> dict[str, Any]:
    status: dict[str, Any] = {
        "guard": str(guard_path),
        "research_cloud_allowed": False,
        "guard_exists": guard_path.is_file(),
        "returncode": None,
        "reasons": [],
    }
    if not guard_path.is_file():
        status["reasons"].append("research cloud guard missing; main thread owns tools/check_research_cloud_gate_status.py")
        return status

    cmd = [
        sys.executable,
        str(guard_path),
        "--json",
        "--max-age-hours",
        str(float(max_age_hours)),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    status.update(
        {
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        status["reasons"].append(f"research guard did not return JSON: {exc}")
        return status

    status["guard_payload"] = payload
    allowed = bool(
        payload.get("research_cloud_allowed")
        or payload.get("cloud_allowed")
        or payload.get("allowed")
    )
    status["research_cloud_allowed"] = allowed and result.returncode == 0
    if not status["research_cloud_allowed"]:
        reasons = payload.get("reasons") or payload.get("blocked_reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        status["reasons"].extend(str(reason) for reason in reasons)
        if result.returncode != 0:
            status["reasons"].append(f"research guard returned {result.returncode}")
        if not status["reasons"]:
            status["reasons"].append("research guard did not explicitly allow research cloud")
    return status


def modal_command(job: ResearchJob, *, scene_subdir: str, download_dir: Path) -> list[str]:
    check_forbidden_text("scene_subdir", scene_subdir)
    check_forbidden_text("output_subdir", job.modal_output_subdir())
    output_dir = normalize_output_dir(job.output_dir())
    metadata = default_research_metadata(
        job_id=job.job_id,
        job_name=job.job_name,
        output_dir=output_dir,
        max_steps=job.max_steps,
        max_cases=job.max_cases,
        max_hours=job.max_hours,
    )
    metadata.update(
        {
            "cloud_lane": job.cloud_lane,
            "modal_lane": job.modal_lane,
            "no_strict_pass_write": True,
            "scene_subdir": scene_subdir,
            "modal_output_subdir": job.modal_output_subdir(),
        }
    )
    metadata_reasons = validate_research_metadata(metadata)
    if metadata_reasons:
        raise ValueError(f"{job.job_id} metadata failed research validation: {metadata_reasons}")

    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools\\run_modal_utf8.ps1",
        "run",
        "modal_surface_research_preflight.py::run_research",
        "--lane",
        job.modal_lane,
        "--scene-subdir",
        scene_subdir,
        "--output-subdir",
        job.modal_output_subdir(),
        "--view-indices",
        job.view_indices,
        "--target-size",
        str(int(job.target_size)),
        "--expected-gpu",
        job.gpu,
        "--download-local-dir",
        str(download_dir),
    ]
    if job.mesh_seed_subpath:
        check_forbidden_text("mesh_seed_subpath", job.mesh_seed_subpath)
        cmd.extend(["--mesh-seed-subpath", job.mesh_seed_subpath])
    if job.template_payload_subpath:
        check_forbidden_text("template_payload_subpath", job.template_payload_subpath)
        cmd.extend(["--template-payload-subpath", job.template_payload_subpath])
    if job.eval_view_indices:
        cmd.extend(["--eval-view-indices", job.eval_view_indices])
    for key, value in job.extra_args.items():
        opt = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
        else:
            cmd.extend([opt, str(value)])
    return cmd


def manifest_for_job(job: ResearchJob, *, scene_subdir: str) -> dict[str, Any]:
    output_dir = normalize_output_dir(job.output_dir())
    download_dir = output_dir / "downloaded_artifacts"
    cmd = modal_command(job, scene_subdir=scene_subdir, download_dir=download_dir)
    metadata = default_research_metadata(
        job_id=job.job_id,
        job_name=job.job_name,
        output_dir=output_dir,
        max_steps=job.max_steps,
        max_cases=job.max_cases,
        max_hours=job.max_hours,
    )
    metadata.update(
        {
            "cloud_lane": job.cloud_lane,
            "modal_lane": job.modal_lane,
            "no_strict_pass_write": True,
            "scene_subdir": scene_subdir,
            "modal_output_subdir": job.modal_output_subdir(),
            "modal_command": cmd,
            "download_local_dir": str(download_dir),
            "mesh_seed_subpath": job.mesh_seed_subpath,
            "template_payload_subpath": job.template_payload_subpath,
            "target_size": job.target_size,
            "view_indices": job.view_indices,
            "eval_view_indices": job.eval_view_indices,
            "extra_args": dict(job.extra_args),
        }
    )
    reasons = validate_research_metadata(metadata)
    if reasons:
        raise ValueError(f"{job.job_id} metadata failed research validation: {reasons}")
    return metadata


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# V8 Research Cloud Launch Agent Status",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Guard",
        "",
        "```json",
        json.dumps(report["guard_status"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        f"- mode = `{report['mode']}`",
        f"- launch_attempted = `{str(report['launch_attempted']).lower()}`",
        f"- manifest_root = `{report['manifest_root']}`",
        f"- report_json = `{report['report_json']}`",
        "",
        "Research-only invariants:",
        "",
        "```text",
        "research_only = true",
        "no_export = true",
        "no_predictions_write = true",
        "no_registry_write = true",
        "no_teacher_export = true",
        "no_candidate_export = true",
        "no_strict_pass_write = true",
        "```",
        "",
        "## Jobs",
        "",
    ]
    for job in report["jobs"]:
        lines.extend(
            [
                f"### {job['cloud_lane']} - {job['job_id']}",
                "",
                f"- status = `{job['status']}`",
                f"- modal_lane = `{job['modal_lane']}`",
                f"- output_dir = `{job['output_dir']}`",
                f"- manifest = `{job['manifest_path']}`",
                f"- max_steps = `{job['max_steps']}`",
                f"- max_cases = `{job['max_cases']}`",
                f"- max_hours = `{job['max_hours']}`",
                "",
                "Command:",
                "",
                "```powershell",
                " ".join(job["modal_command"]),
                "```",
                "",
            ]
        )
        if job.get("launch_result"):
            lines.extend(
                [
                    "Launch result:",
                    "",
                    "```json",
                    json.dumps(job["launch_result"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_launch_command(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    return {
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
    }


def main() -> int:
    args = parse_args()
    selected_lanes = split_lanes(args.lanes)
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    report_md = Path(args.report_md).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve()
    guard_path = Path(args.guard).expanduser().resolve()
    scene_subdir = args.scene_subdir.strip().replace("\\", "/").strip("/")
    check_forbidden_text("scene_subdir", scene_subdir)

    global MANIFEST_ROOT
    MANIFEST_ROOT = manifest_root

    guard = guard_status(guard_path, args.max_age_hours)
    requested_launch = args.mode == "launch"
    launch_allowed = requested_launch and bool(args.allow_launch) and bool(guard["research_cloud_allowed"])
    if requested_launch and not args.allow_launch:
        guard["reasons"].append("--mode launch requires --allow-launch")

    jobs: list[dict[str, Any]] = []
    for job in DEFAULT_JOBS:
        if job.cloud_lane not in selected_lanes:
            continue
        manifest = manifest_for_job(job, scene_subdir=scene_subdir)
        job_dir = Path(manifest["output_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "launch_manifest.json"
        write_json(manifest_path, manifest)

        job_report = {
            **manifest,
            "status": "ready_not_launched",
            "manifest_path": str(manifest_path),
            "launch_result": None,
        }
        if not guard["research_cloud_allowed"]:
            job_report["status"] = "blocked_by_research_guard"
        elif requested_launch and not args.allow_launch:
            job_report["status"] = "blocked_by_two_key_launch"
        elif launch_allowed:
            job_report["status"] = "launching"
            launch_result = run_launch_command(manifest["modal_command"])
            job_report["launch_result"] = launch_result
            job_report["status"] = "launched" if launch_result["returncode"] == 0 else "launch_failed"
        jobs.append(job_report)

    status = "blocked_research_guard"
    if guard["research_cloud_allowed"] and not requested_launch:
        status = "dry_run_ready_research_guard_green"
    elif guard["research_cloud_allowed"] and requested_launch and not args.allow_launch:
        status = "blocked_two_key_launch"
    elif launch_allowed:
        status = "launch_completed" if all(job["status"] == "launched" for job in jobs) else "launch_failed"

    report = {
        "status": status,
        "created_utc": now_utc(),
        "mode": args.mode,
        "launch_attempted": bool(launch_allowed),
        "manifest_root": str(manifest_root),
        "report_md": str(report_md),
        "report_json": str(report_json),
        "guard_status": guard,
        "jobs": jobs,
        "owned_outputs": [
            str(REPO_ROOT / "tools" / "launch_v8_research_cloud_jobs.py"),
            str(report_md),
            str(report_json),
            str(manifest_root),
        ],
    }
    write_json(report_json, report)
    write_markdown_report(report_md, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status.startswith(("dry_run_ready", "launch_completed", "blocked_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
