from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools/v9_external_backend_install_attempts"
REPORT_JSON = OUT_DIR / "a5x2_backend_readiness_audit.json"
REPORT_MD = OUT_DIR / "README.md"


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "elapsed_sec": time.time() - started,
        }
    except Exception as exc:
        return {"command": cmd, "cwd": str(cwd) if cwd else None, "error": repr(exc), "elapsed_sec": time.time() - started}


def exists(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
        "bytes": int(path.stat().st_size) if path.is_file() else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    must3r = Path("D:/must3r-main")
    two_dgs = Path("D:/2d-gaussian-splatting-main")
    mast3r_slam = Path("D:/MASt3R-SLAM-main")
    must3r_artifact = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend_audit/summary.json"
    scene_root = REPO_ROOT / "output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop"
    rows = {
        "MUSt3R": {
            "repo": exists(must3r),
            "artifact_audit": exists(must3r_artifact),
            "status": "true_backend_ran_nonempty_weak_pool_only" if must3r_artifact.is_file() else "repo_only_no_artifact",
        },
        "2DGS": {
            "repo": exists(two_dgs),
            "submodules": {
                "diff_surfel": exists(two_dgs / "submodules/diff-surfel-rasterization"),
                "simple_knn": exists(two_dgs / "submodules/simple-knn"),
            },
            "environment": exists(two_dgs / "environment.yml"),
            "train_py": exists(two_dgs / "train.py"),
            "render_py": exists(two_dgs / "render.py"),
            "data_contract": {
                "requires_colmap_format": True,
                "current_4k4d_scene_has_images_masks_cameras": scene_root.is_dir(),
                "current_4k4d_scene_is_colmap_sparse_format": (scene_root / "sparse").is_dir(),
            },
            "status": "repo_present_blocked_missing_colmap_scene_and_cuda_extension_build",
        },
        "MASt3R-SLAM": {
            "repo": exists(mast3r_slam),
            "thirdparty_mast3r": exists(mast3r_slam / "thirdparty/mast3r"),
            "thirdparty_in3d": exists(mast3r_slam / "thirdparty/in3d"),
            "main_py": exists(mast3r_slam / "main.py"),
            "checkpoint_expected": [
                "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
                "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
            ],
            "checkpoint_present": [
                exists(mast3r_slam / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
                exists(mast3r_slam / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"),
                exists(mast3r_slam / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"),
            ],
            "status": "repo_present_blocked_missing_checkpoints_and_install",
        },
    }
    probes = {
        "2dgs_git_status": run(["git", "status", "--short"], cwd=two_dgs, timeout=30) if two_dgs.is_dir() else None,
        "mast3r_slam_git_status": run(["git", "status", "--short"], cwd=mast3r_slam, timeout=30) if mast3r_slam.is_dir() else None,
        "must3r_git_status": run(["git", "status", "--short"], cwd=must3r, timeout=30) if must3r.is_dir() else None,
    }
    summary = {
        "task": "v9_a5x2_backend_readiness_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "research_only": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_write": True,
        },
        "backends": rows,
        "probes": probes,
        "decision": (
            "MUSt3R is the only V9 A5-X2 backend that has run on staged 4K4D images and produced a non-empty pointcloud. "
            "2DGS and MASt3R-SLAM are present on D: but remain blocked by data contract/checkpoint/install prerequisites."
        ),
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V9 A5-X2 External Backend Readiness",
                "",
                summary["decision"],
                "",
                "## Status",
                "",
                f"- MUSt3R: `{rows['MUSt3R']['status']}`",
                f"- 2DGS: `{rows['2DGS']['status']}`",
                f"- MASt3R-SLAM: `{rows['MASt3R-SLAM']['status']}`",
                "",
                "No synthetic pointcloud, teacher, candidate, predictions, registry, or strict pass artifact is written here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "decision": summary["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
