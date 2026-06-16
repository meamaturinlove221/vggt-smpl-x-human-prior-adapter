from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, REPO_ROOT, json_ready, read_summary, write_json


FRAMES = (0, 1, 2)
ADJACENT_FRAMES = (1, 2)
DEFAULT_REGISTRY = REPORTS / "20260508_v14_strict_gate_registry_refresh.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def scene_dir(frame: int) -> Path:
    return REPO_ROOT / f"output/4k4d_scenes/0012_11_frame{frame:04d}_12views_tmf"


def prediction_dir(frame: int) -> Path:
    return REPO_ROOT / f"output/modal_results/0012_11_frame{frame:04d}_60views"


def scene_row(frame: int) -> dict[str, Any]:
    scene = scene_dir(frame)
    image_dir = scene / "images"
    mask_dir = scene / "masks"
    images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
    masks = sorted(mask_dir.glob("*.png")) if mask_dir.is_dir() else []
    manifest = scene / "scene_manifest.json"
    prior_maps = scene / "prior_maps.npz"
    ready = bool(scene.is_dir() and len(images) >= 12 and len(masks) >= 12 and manifest.is_file() and prior_maps.is_file())
    return {
        "frame": frame,
        "scene": str(scene.resolve()),
        "exists": scene.is_dir(),
        "image_count": len(images),
        "mask_count": len(masks),
        "manifest": file_row(manifest),
        "prior_maps": file_row(prior_maps),
        "ready_for_bounded_vggt": ready,
    }


def prediction_row(frame: int) -> dict[str, Any]:
    out = prediction_dir(frame)
    return {
        "frame": frame,
        "output_dir": str(out.resolve()),
        "predictions": file_row(out / "predictions.npz"),
        "summary": file_row(out / "summary.json"),
        "ready": bool((out / "predictions.npz").is_file() and (out / "summary.json").is_file()),
    }


def local_vggt_command(frame: int) -> list[str]:
    return [
        "python",
        "tools/run_local_vggt_inference.py",
        "--scene-dir",
        f"output/4k4d_scenes/0012_11_frame{frame:04d}_12views_tmf",
        "--output-dir",
        f"output/modal_results/0012_11_frame{frame:04d}_60views",
        "--device",
        "cuda",
        "--target-size",
        "518",
        "--overwrite",
    ]


def modal_upload_run_command(frame: int) -> list[str]:
    scene_name = f"0012_11_frame{frame:04d}_12views_tmf"
    return [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools\\run_modal_utf8.ps1",
        "run",
        "modal_4k4d_vggt_infer.py::run_scene_from_local",
        "--local-scene-dir",
        f"output\\4k4d_scenes\\{scene_name}",
        "--remote-scene-subdir",
        f"scenes\\{scene_name}",
        "--output-subdir",
        f"evals/v15_tmf/{scene_name}_vggt518",
        "--download-local-dir",
        f"output\\modal_results\\0012_11_frame{frame:04d}_60views",
    ]


def modal_chunk_download_command(frame: int) -> list[str]:
    scene_name = f"0012_11_frame{frame:04d}_12views_tmf"
    return [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools\\run_modal_utf8.ps1",
        "run",
        "modal_4k4d_vggt_infer.py::download_prediction_chunks_rpc",
        "--remote-output-subdir",
        f"evals/v15_tmf/{scene_name}_vggt518",
        "--local-output-dir",
        f"output\\modal_results\\0012_11_frame{frame:04d}_60views",
        "--chunk-views",
        "1",
        "--resume-existing",
        "True",
        "--max-chunk-retries",
        "3",
    ]


def run_json_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    payload["returncode"] = int(proc.returncode)
    return payload


def formal_guards(registry: Path) -> dict[str, Any]:
    base = [sys.executable, "tools/check_cloud_gate_status.py", "--registry", str(registry), "--json"]
    return {
        "candidate": run_json_command(base),
        "teacher_supervised": run_json_command(
            [sys.executable, "tools/check_cloud_gate_status.py", "--registry", str(registry), "--teacher-supervised", "--json"]
        ),
    }


def cuda_probe() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_import": False,
        "cuda_available": False,
        "supported_for_local_vggt": False,
        "reason": "torch was not probed",
    }
    try:
        import torch

        info.update(
            {
                "torch_import": True,
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
                "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else [],
            }
        )
        if not torch.cuda.is_available():
            info["reason"] = "torch.cuda.is_available() is false"
            return info

        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "capability": [int(major), int(minor)],
                "capability_tag": sm,
                "memory_total_bytes": int(props.total_memory),
            }
        )
        arch_list = set(info.get("arch_list") or [])
        if arch_list and sm not in arch_list:
            info["reason"] = f"current PyTorch CUDA build does not list {sm} in supported arch list"
            return info
        try:
            probe = torch.ones((1,), dtype=torch.float32, device="cuda")
            probe = probe + 1
            torch.cuda.synchronize()
            info["cuda_smoke_value"] = float(probe.detach().cpu()[0])
        except Exception as exc:  # noqa: BLE001
            info["reason"] = f"cuda smoke allocation/op failed: {exc!r}"
            return info
        info["supported_for_local_vggt"] = True
        info["reason"] = "cuda smoke passed and capability is in torch arch list"
        return info
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"torch probe failed: {exc!r}"
        return info


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V15 TMF Prediction Dispatch",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only dispatch/readiness report. This script does not write predictions, teacher/candidate packages, registries, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## CUDA",
        "",
        "```json",
        json.dumps(summary["local_cuda"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Formal Guard",
        "",
        "```json",
        json.dumps(summary["formal_guards"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Bounded Commands",
        "",
    ]
    commands = summary.get("dispatch_commands", {})
    if commands:
        for label, row in commands.items():
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- allowed_by_this_report: `{str(row['allowed_by_this_report']).lower()}`",
                    f"- reason: {row['reason']}",
                    "",
                    "```powershell",
                    " ".join(row["command"]),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["- none", ""])
    lines.extend(["## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 T-line TMF prediction dispatch/readiness.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_tmf_prediction_dispatch.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_tmf_prediction_dispatch.md")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    scenes = [scene_row(frame) for frame in FRAMES]
    predictions = [prediction_row(frame) for frame in FRAMES]
    scene_by_frame = {row["frame"]: row for row in scenes}
    pred_by_frame = {row["frame"]: row for row in predictions}
    missing_adjacent = [frame for frame in ADJACENT_FRAMES if not pred_by_frame[frame]["ready"]]
    adjacent_scenes_ready = all(scene_by_frame[frame]["ready_for_bounded_vggt"] for frame in ADJACENT_FRAMES)
    adjacent_predictions_ready = not missing_adjacent

    local_cuda = cuda_probe()
    guards = formal_guards(args.registry.expanduser().resolve())
    formal_cloud_unblocked = bool(guards["candidate"].get("cloud_allowed") or guards["teacher_supervised"].get("cloud_allowed"))
    v14_t = read_summary(REPORTS / "20260508_v14_tmf_prediction_readiness.json")
    v14_f = read_summary(REPORTS / "20260508_v14_fus3d_region_backend_readiness.json")
    v14_dline = read_summary(REPORTS / "20260508_v14_dline_promotion_report.json")

    dispatch_commands: dict[str, dict[str, Any]] = {}
    for frame in missing_adjacent:
        local_allowed = bool(adjacent_scenes_ready and local_cuda.get("supported_for_local_vggt"))
        dispatch_commands[f"frame{frame:04d}_local_vggt"] = {
            "command": local_vggt_command(frame),
            "allowed_by_this_report": local_allowed,
            "reason": (
                "bounded local CUDA inference is feasible for this frame"
                if local_allowed
                else f"local CUDA unsupported for this PyTorch/GPU combination: {local_cuda.get('reason')}"
            ),
        }
        dispatch_commands[f"frame{frame:04d}_formal_modal_upload_run"] = {
            "command": modal_upload_run_command(frame),
            "allowed_by_this_report": False,
            "reason": (
                "formal Modal inference remains blocked by tools/check_cloud_gate_status.py; do not set "
                "VGGT_ALLOW_CLOUD_WITHOUT_STRICT_PASS for this dispatch"
            ),
        }
        dispatch_commands[f"frame{frame:04d}_formal_modal_chunk_download"] = {
            "command": modal_chunk_download_command(frame),
            "allowed_by_this_report": False,
            "reason": "remote prediction chunk export/download is also behind the formal cloud inference guard",
        }

    blockers: list[str] = []
    if not adjacent_scenes_ready:
        blockers.append("Adjacent frame0001/frame0002 TMF scene inputs are incomplete.")
    if missing_adjacent:
        blockers.append(f"Missing adjacent-frame predictions for: {', '.join(f'frame{frame:04d}' for frame in missing_adjacent)}.")
    if missing_adjacent and not local_cuda.get("supported_for_local_vggt"):
        blockers.append(f"Local CUDA is not usable for VGGT here: {local_cuda.get('reason')}.")
    if not formal_cloud_unblocked:
        blockers.append("Formal cloud guard is blocked; cloud inference/upload/download commands are recorded only, not authorized.")
    if v14_dline.get("strict_candidate_passes", 0) or v14_dline.get("strict_teacher_passes", 0):
        blockers.append("Unexpected strict pass detected in V14 D-line source; re-audit before dispatch.")

    if adjacent_predictions_ready:
        status = "v15_tmf_predictions_available_research_only"
        decision = "Frame0001/frame0002 predictions are present; T-line can proceed to a real canonical teacher audit, still with no strict pass written here."
    elif adjacent_scenes_ready and local_cuda.get("supported_for_local_vggt"):
        status = "v15_tmf_local_dispatch_ready"
        decision = "Frame0001/frame0002 scenes are ready and local CUDA is supported, so bounded local VGGT commands are ready to run; this report did not run them."
    elif adjacent_scenes_ready:
        status = "v15_tmf_dispatch_blocked_local_cuda_unsupported"
        decision = "Frame0001/frame0002 scenes are ready, but predictions are missing and local CUDA is unsupported by the current PyTorch build; only exact bounded commands and guard constraints were written."
    else:
        status = "v15_tmf_dispatch_blocked_scene_inputs_missing"
        decision = "T-line dispatch is blocked because the adjacent TMF scene inputs are incomplete."

    summary = {
        "task": "v15_tmf_prediction_dispatch",
        "created_utc": utc_now(),
        "status": status,
        "scenes": scenes,
        "predictions": predictions,
        "missing_adjacent_frames": missing_adjacent,
        "adjacent_scenes_ready": adjacent_scenes_ready,
        "frame0001_0002_predictions_ready": adjacent_predictions_ready,
        "local_cuda": local_cuda,
        "formal_guards": guards,
        "formal_cloud_unblocked": False,
        "formal_cloud_actual_guard_allowed": formal_cloud_unblocked,
        "dispatch_commands": dispatch_commands,
        "research_cloud_constraints": {
            "formal_modal_inference_allowed_by_this_report": False,
            "do_not_bypass_formal_guard": True,
            "forbidden_override": "VGGT_ALLOW_CLOUD_WITHOUT_STRICT_PASS",
            "allowed_research_cloud_only": "bounded non-prediction research-cloud jobs under output/surface_research_cloud_preflight after tools/check_research_cloud_gate_status.py allows them",
            "forbidden_outputs": [
                "predictions.npz from research-cloud preflight lanes",
                "teacher package",
                "candidate package",
                "strict gate registry write",
                "strict pass write",
            ],
        },
        "source_status": {
            "v14_tmf": v14_t.get("status"),
            "v14_fus3d": v14_f.get("status"),
            "v14_dline": v14_dline.get("status"),
        },
        "canonical_teacher_ready": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write_by_this_dispatch": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "decision": decision,
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "output": args.output_json, "missing_adjacent_frames": missing_adjacent}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
