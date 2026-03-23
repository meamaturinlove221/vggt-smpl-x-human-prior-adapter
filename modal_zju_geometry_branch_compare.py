import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REQUIREMENTS_TRAINING = "requirements_training.txt"
DEFAULT_REQUIREMENTS = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.1",
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
    "hydra-core",
    "omegaconf",
    "fvcore",
    "iopath",
    "wcmatch",
    "tensorboard",
    "tqdm",
    "opencv-python",
]


def _load_requirements(path: Path, seen: set[str] | None = None) -> list[str]:
    seen = set() if seen is None else seen
    packages: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            nested = (path.parent / line[3:].strip()).resolve()
            packages.extend(_load_requirements(nested, seen))
            continue
        if line not in seen:
            seen.add(line)
            packages.append(line)
    return packages


def _resolve_requirements() -> list[str]:
    candidates = [
        REPO_ROOT / REQUIREMENTS_TRAINING,
        Path.cwd() / REQUIREMENTS_TRAINING,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_requirements(candidate)
    return list(DEFAULT_REQUIREMENTS)


def _decode_config_blob(blob: str) -> dict:
    text = blob.lstrip("\ufeff")
    if text.startswith("base64:"):
        text = base64.b64decode(text[len("base64:") :]).decode("utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoded = base64.b64decode(text).decode("utf-8-sig")
        return json.loads(decoded)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


TRAINING_REQUIREMENTS = _resolve_requirements()
APP_NAME = os.environ.get("VGGT_ZJU_GEOM_COMPARE_MODAL_APP_NAME", "vggt-zju-geometry-branch-compare")
DATA_VOLUME_NAME = os.environ.get("VGGT_ZJU_GEOM_COMPARE_DATA_VOLUME", "vggt-zju-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_ZJU_GEOM_COMPARE_OUTPUT_VOLUME", "vggt-out")
GPU_SPEC = os.environ.get("VGGT_ZJU_GEOM_COMPARE_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_ZJU_GEOM_COMPARE_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_ZJU_GEOM_COMPARE_MEMORY_MB", "49152"))
TIMEOUT_SEC = int(os.environ.get("VGGT_ZJU_GEOM_COMPARE_TIMEOUT_SEC", str(8 * 60 * 60)))
OUTPUT_CHECKPOINT_FALLBACKS = ("weights/model.pt", "pretrained_weights/model.pt")

CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    ".github/**",
    "__pycache__",
    "__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/**",
    ".venv*",
    ".venv*/**",
    "training/logs",
    "training/logs/**",
    "training/**/ckpts",
    "training/**/ckpts/**",
    "output",
    "output/**",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.npz",
    "*.ply",
    "*.obj",
    "*.glb",
    "*.zip",
    "*.tar",
    "*.tar.*",
]

COMPARE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(*TRAINING_REQUIREMENTS)
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path=(REMOTE_CODE_DIR / "scripts").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
    .add_local_dir(
        str(REPO_ROOT / "vggt"),
        remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=False)


@dataclass
class BranchCompareCase:
    case_id: str
    report_json: str | None = None
    report_json_b64: str | None = None


@dataclass
class BranchCompareBatchConfig:
    cases: list[dict]
    zju_subdir: str = "zju_mocap"
    checkpoint_subpath: str = "checkpoints/model.pt"
    exp_name: str = "zju_geometry_branch_compare_batch"
    output_subdir: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    conf_percentile: float = 25.0
    export_max_points: int = 100000
    render_max_points: int = 500000
    z_tolerance: float = 0.02
    min_conf: float = 1e-6
    primary_branch: str = "depth_unproject"
    skip_save_predictions: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "BranchCompareBatchConfig":
        return BranchCompareBatchConfig(**_decode_config_blob(blob))


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _resolve_output_root(exp_name: str, output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_exp = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in exp_name).strip("_")
    safe_exp = safe_exp or "zju_geometry_branch_compare_batch"
    return Path(str(REMOTE_OUTPUT_DIR / "geometry_compare" / f"{run_tag}_{safe_exp}"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_volume.commit()


def _decode_case_report_text(case: BranchCompareCase) -> str:
    if case.report_json_b64:
        return base64.b64decode(case.report_json_b64).decode("utf-8")

    if isinstance(case.report_json, str):
        return case.report_json

    # PowerShell ConvertTo-Json can accidentally serialize the file-content string
    # as a PSCustomObject with a `value` field plus PS* metadata. Recover the raw
    # original report text when we see that legacy shape.
    if isinstance(case.report_json, dict) and isinstance(case.report_json.get("value"), str):
        return case.report_json["value"]

    raise ValueError(f"Unsupported report payload for case {case.case_id}.")


def _resolve_checkpoint_path(remote_subpath: str) -> Path | None:
    if not remote_subpath.strip():
        return None
    target = _remote_data_path(remote_subpath)
    if target.exists():
        return target

    candidates = [_remote_output_path(remote_subpath)] + [
        _remote_output_path(path) for path in OUTPUT_CHECKPOINT_FALLBACKS
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[modal-zju-geom-compare] using checkpoint from output volume: {candidate}", flush=True)
            return candidate

    raise FileNotFoundError(
        "Could not resolve checkpoint from data volume or output volume fallbacks.\n"
        f"- requested data path: {target}\n"
        + "\n".join(f"- tried output fallback: {path}" for path in candidates)
    )


def _load_cfg_from_output_volume(cfg_subpath: str) -> BranchCompareBatchConfig:
    cfg_path = _remote_output_path(cfg_subpath)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found on output volume: {cfg_path}")
    return BranchCompareBatchConfig.from_json(cfg_path.read_text(encoding="utf-8-sig"))


def _summary_row(case_id: str, case_dir: Path, summary: dict) -> dict:
    return {
        "case_id": case_id,
        "view_profile": summary["case"]["view_profile"],
        "source_count": summary["case"]["source_count"],
        "decision": summary["decision"]["decision"],
        "mae_winner": summary["decision"]["mae_winner"],
        "coverage_winner": summary["decision"]["coverage_winner"],
        "point_mae": summary["branches"]["point_map"]["metrics"]["mae"],
        "depth_mae": summary["branches"]["depth_unproject"]["metrics"]["mae"],
        "point_cov": summary["branches"]["point_map"]["render"]["coverage_ratio"],
        "depth_cov": summary["branches"]["depth_unproject"]["render"]["coverage_ratio"],
        "summary_md": str(case_dir / "summary.md"),
    }


def _write_batch_markdown(path: Path, rows: list[dict], failures: list[dict]) -> None:
    depth_wins = [row for row in rows if row["decision"] == "depth_unproject"]
    point_wins = [row for row in rows if row["decision"] == "point_map"]
    ties = [row for row in rows if row["decision"] == "tie"]
    lines = [
        "# ZJU Geometry Branch Compare Batch",
        "",
        f"- runs: `{len(rows)}`",
        f"- depth_unproject_wins: `{len(depth_wins)}`",
        f"- point_map_wins: `{len(point_wins)}`",
        f"- ties: `{len(ties)}`",
        "",
        "| Case | Profile | Sources | Decision | MAE Winner | Cov Winner | Point MAE | Depth MAE | Point Cov | Depth Cov | Summary |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {view_profile} | {source_count} | {decision} | {mae_winner} | {coverage_winner} | "
            "{point_mae:.4f} | {depth_mae:.4f} | {point_cov:.4f} | {depth_cov:.4f} | `{summary_md}` |".format(
                **row
            )
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['case_id']}`: {failure['error'] or 'unknown error'}")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This batch keeps the mentor-aligned first-round definition: compare only `point_map` vs `depth + camera` branch outputs.",
            "- It does not add `unproject_geometry` loss or revive the old ghost stack.",
            "- `depth + camera` wins when it lowers MAE without losing coverage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_volume.commit()


def _run_compare_case(
    cfg: BranchCompareBatchConfig,
    case: BranchCompareCase,
    report_path: Path,
    case_dir: Path,
    checkpoint_path: Path | None,
) -> dict:
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    compare_script = remote_code_dir / "scripts" / "compare_geometry_branches_zju_report.py"
    zju_root = _remote_data_path(cfg.zju_subdir)
    cmd = [
        sys.executable,
        str(compare_script),
        "--report_json",
        str(report_path),
        "--local_zju_root",
        zju_root.as_posix(),
        "--output_dir",
        case_dir.as_posix(),
        "--device",
        cfg.device,
        "--dtype",
        cfg.dtype,
        "--conf_percentile",
        str(cfg.conf_percentile),
        "--export_max_points",
        str(cfg.export_max_points),
        "--render_max_points",
        str(cfg.render_max_points),
        "--z_tolerance",
        str(cfg.z_tolerance),
        "--min_conf",
        str(cfg.min_conf),
        "--primary_branch",
        cfg.primary_branch,
    ]
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint", checkpoint_path.as_posix()])
    if cfg.skip_save_predictions:
        cmd.append("--skip_save_predictions")

    print(f"[modal-zju-geom-compare] case={case.case_id}", flush=True)
    print(f"[modal-zju-geom-compare] command={shlex.join(cmd)}", flush=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = repo_pythonpath if not existing_pythonpath else repo_pythonpath + os.pathsep + existing_pythonpath
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run(cmd, cwd=str(remote_code_dir), env=env, check=True)
    output_volume.commit()

    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for case {case.case_id}: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_branch_compare_batch(cfg: BranchCompareBatchConfig) -> None:
    checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    output_root = _resolve_output_root(cfg.exp_name, cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "batch_status.json"
    status = {
        "exp_name": cfg.exp_name,
        "output_root": output_root.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix() if checkpoint_path is not None else "",
        "state": "running",
        "cases": [],
    }
    _write_json(status_path, status)

    rows: list[dict] = []
    failures: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="zju_geom_compare_") as tmpdir:
        tmp_root = Path(tmpdir)
        for index, raw_case in enumerate(cfg.cases):
            case = BranchCompareCase(**raw_case)
            case_dir = output_root / case.case_id
            report_path = tmp_root / f"case_{index:03d}_{case.case_id}.json"
            report_text = _decode_case_report_text(case)
            report_path.write_text(report_text, encoding="utf-8")

            case_status = {
                "case_id": case.case_id,
                "state": "running",
                "output_dir": case_dir.as_posix(),
            }
            status["cases"].append(case_status)
            _write_json(status_path, status)

            try:
                summary = _run_compare_case(cfg, case, report_path, case_dir, checkpoint_path)
                rows.append(_summary_row(case.case_id, case_dir, summary))
                case_status["state"] = "completed"
            except Exception as exc:
                case_status["state"] = "failed"
                case_status["error"] = str(exc)
                failures.append({"case_id": case.case_id, "error": str(exc)})
                status["state"] = "failed"
                _write_json(status_path, status)
                raise

            _write_json(status_path, status)

    batch_json_path = output_root / "batch_summary.json"
    batch_md_path = output_root / "batch_summary.md"
    batch_json_path.write_text(
        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_volume.commit()
    _write_batch_markdown(batch_md_path, rows, failures)

    status["state"] = "completed"
    _write_json(status_path, status)


@app.function(
    image=COMPARE_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_zju_geometry_branch_compare_batch(cfg_json: str) -> None:
    _run_branch_compare_batch(BranchCompareBatchConfig.from_json(cfg_json))


@app.function(
    image=COMPARE_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_zju_geometry_branch_compare_batch_from_cfg_path(cfg_subpath: str) -> None:
    _run_branch_compare_batch(_load_cfg_from_output_volume(cfg_subpath))
