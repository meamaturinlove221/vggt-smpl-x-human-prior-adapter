from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")

APP_NAME = os.environ.get("VGGT_MODAL_V30_APP_NAME", "vggt-v30-prior-enabled-predictions")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_V30_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-40GB"))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V30_TIMEOUT_SEC", str(6 * 60 * 60)))

V30_REMOTE_ROOT = "surface_research_cloud_preflight/V30_prior_enabled_predictions"
FORBIDDEN_TOKENS = (
    "strict_pass",
    "strict_gate_registry",
    "formal_candidate",
    "candidate_package",
    "teacher_package",
    "teacher_export",
    "candidate_export",
    "registry_refresh",
)


def _load_requirements(path: Path) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        packages.append(line)
    return packages


def _requirements() -> list[str]:
    path = REPO_ROOT / "requirements.txt"
    if path.is_file():
        return _load_requirements(path)
    return ["torch", "torchvision", "numpy", "Pillow", "huggingface_hub", "einops", "safetensors"]


def _normalize_subpath(value: str, *, require_v30_root: bool) -> str:
    cleaned = (value or "").replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("empty volume path")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"parent traversal forbidden: {value}")
    lower = cleaned.lower()
    if require_v30_root and not lower.startswith(V30_REMOTE_ROOT.lower()):
        raise ValueError(f"V30 outputs must stay under {V30_REMOTE_ROOT}: {value}")
    if Path(cleaned).name.lower() == "predictions.npz":
        raise ValueError("V30 must not write predictions.npz")
    hits = [token for token in FORBIDDEN_TOKENS if token in lower]
    if hits:
        raise ValueError(f"formal-output token(s) forbidden in V30 path {value}: {hits}")
    return cleaned


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"V30 local scene/checkpoint directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir, require_v30_root=False)
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() == "predictions.npz":
                raise ValueError(f"refusing to upload formal predictions file: {path}")
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


CODE_SYNC_IGNORE = [".git", ".git/**", "__pycache__", "__pycache__/**", ".venv*", ".venv*/**", "output", "output/**", "reports", "reports/**"]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_requirements())
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class V30Config:
    scene_subdirs: dict[str, str]
    output_subdir: str = V30_REMOTE_ROOT
    checkpoint_subdir: str = ""
    checkpoint_filename: str = ""
    target_size: int = 518
    controls: tuple[str, ...] = ("real", "zero", "shuffle", "random-region", "prior-dropout")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "V30Config":
        payload = json.loads(blob)
        if isinstance(payload.get("controls"), list):
            payload["controls"] = tuple(payload["controls"])
        return V30Config(**payload)


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return payload
    return {}


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SEC,
    volumes={str(REMOTE_DATA_DIR): data_volume, str(REMOTE_OUTPUT_DIR): output_volume},
)
def run_prior_enabled_predictions(config_json: str) -> dict[str, Any]:
    import numpy as np
    import torch

    import sys

    sys.path.insert(0, str(REMOTE_CODE_DIR))
    from vggt.models.vggt import VGGT

    cfg = V30Config.from_json(config_json)
    out_dir = Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(cfg.output_subdir, require_v30_root=True)))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "task": "v30_prior_enabled_vggt_predictions_remote",
        "research_only": True,
        "output_dir": str(out_dir),
        "controls": list(cfg.controls),
        "human_prior_channels": 0,
        "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
        "blockers": [],
        "no_predictions_npz": True,
    }

    checkpoint_path = None
    if cfg.checkpoint_subdir and cfg.checkpoint_filename:
        checkpoint_path = Path(str(REMOTE_DATA_DIR / _normalize_subpath(cfg.checkpoint_subdir, require_v30_root=False))) / cfg.checkpoint_filename
    if checkpoint_path is None or not checkpoint_path.is_file():
        summary["blockers"].append("No prior-enabled checkpoint was provided on the data volume.")
        _write_json(out_dir / "research_prior_effect.json", summary)
        _write_json(out_dir / "v30_remote_guard.json", summary)
        output_volume.commit()
        return summary

    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(payload)
    keys = [str(key) for key in state.keys()]
    has_prior = any("human_prior_adapter" in key for key in keys)
    if not has_prior:
        summary["blockers"].append("Provided checkpoint has no HumanPriorAdapter weights.")
        _write_json(out_dir / "research_prior_effect.json", summary)
        _write_json(out_dir / "v30_remote_guard.json", summary)
        output_volume.commit()
        return summary

    # The remote runner intentionally refuses to infer architecture from arbitrary
    # checkpoints. V31 is expected to produce explicit architecture metadata.
    summary["blockers"].append("Prior-enabled checkpoint metadata is missing explicit architecture/channel counts required for safe V30 inference.")
    summary["checkpoint_key_count"] = len(keys)
    summary["has_human_prior_adapter_weights"] = True
    _write_json(out_dir / "research_prior_effect.json", summary)
    _write_json(out_dir / "v30_remote_guard.json", summary)
    output_volume.commit()
    return summary


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir, require_v30_root=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel = Path(entry.path)
        try:
            rel = rel.relative_to(prefix)
        except ValueError:
            pass
        if rel.name.lower() == "predictions.npz":
            raise RuntimeError(f"V30 safety stop: predictions.npz found at {entry.path}")
        dest = local_dir / rel
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            output_volume.read_file_into_fileobj(entry.path, handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run Modal V30 only when a checkpoint is supplied.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-filename", default="")
    parser.add_argument("--target-size", type=int, default=518)
    args = parser.parse_args()

    if not args.execute:
        print("V30 Modal entrypoint ready. Use --execute only with a prior-enabled checkpoint directory.")
        return 0

    if args.checkpoint_dir is None or not args.checkpoint_filename:
        raise SystemExit("--execute requires --checkpoint-dir and --checkpoint-filename")

    checkpoint_subdir = _upload_dir(args.checkpoint_dir, "v30_input/prior_enabled_checkpoint")
    scenes = {}
    for frame in ("frame0000", "frame0001", "frame0002"):
        frame_num = int(frame.replace("frame", ""))
        scene = REPO_ROOT / f"output/4k4d_scenes/0012_11_frame{frame_num:04d}_12views_tmf"
        scenes[frame] = _upload_dir(scene, f"v30_input/scenes/{frame}")
    cfg = V30Config(
        scene_subdirs=scenes,
        checkpoint_subdir=checkpoint_subdir,
        checkpoint_filename=args.checkpoint_filename,
        target_size=args.target_size,
    )
    with modal.enable_output():
        result = run_prior_enabled_predictions.remote(cfg.to_json())
    local_dir = REPO_ROOT / "output/surface_research_cloud_preflight/V30_prior_enabled_predictions"
    _download_volume_dir(V30_REMOTE_ROOT, local_dir)
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
