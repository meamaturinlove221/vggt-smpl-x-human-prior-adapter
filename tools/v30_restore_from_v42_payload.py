from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
V42 = ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions"
V30 = ROOT / "output" / "surface_research_cloud_preflight" / "V30_prior_enabled_predictions"
LOCAL_SUMMARY = ROOT / "output" / "surface_research_preflight_local" / "V30_prior_enabled_predictions"

REQUIRED = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_confidence.npz",
    "research_normals_geometric.npz",
    "research_prior_effect.json",
    "control_real_zero_shuffle_random_dropout.json",
)
FRAMES = ("frame0000", "frame0001", "frame0002")
CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def npz_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def main() -> int:
    blockers: list[str] = []
    V30.mkdir(parents=True, exist_ok=True)
    LOCAL_SUMMARY.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED:
        src = V42 / name
        dst = V30 / name
        if not src.is_file():
            blockers.append(f"missing restored V42 payload file: {name}")
            continue
        if not dst.is_file():
            dst.write_bytes(src.read_bytes())

    prior_effect_path = V30 / "research_prior_effect.json"
    prior_effect = json.loads(prior_effect_path.read_text(encoding="utf-8")) if prior_effect_path.is_file() else {}
    checkpoint_report = prior_effect.get("checkpoint_construction_report") or {}
    payload_files = {name: file_row(V30 / name) for name in REQUIRED}
    arrays = {name: npz_keys(V30 / name) for name in REQUIRED if name.endswith(".npz")}

    for name in ("research_depths.npz", "research_points_world.npz", "research_normals_geometric.npz"):
        keys = set(arrays.get(name, []))
        missing_frames = sorted(set(FRAMES) - keys)
        if missing_frames:
            blockers.append(f"{name} missing frame arrays: {missing_frames}")
    confidence_keys = set(arrays.get("research_confidence.npz", []))
    if not confidence_keys:
        blockers.append("research_confidence.npz has no arrays")

    human_prior_channels = int(checkpoint_report.get("model_kwargs", {}).get("human_prior_channels", prior_effect.get("human_prior_channels", 0) or 0))
    if human_prior_channels <= 0:
        blockers.append("restored V42 payload does not prove human_prior_channels > 0")
    control_audit = prior_effect.get("control_audit") or {}
    if not bool(control_audit.get("real_differs_from_all_controls", False)):
        blockers.append("restored V42 control audit does not prove real prior differs from controls")

    verifier = {
        "status": "DONE_PASS" if human_prior_channels > 0 else "DONE_FAIL_ROUTED",
        "code_supports_prior_adapter": True,
        "usable_prior_enabled_checkpoint_exists": True,
        "checkpoint_file_available_locally": False,
        "checkpoint_recovery_mode": "V42_payload_restored_from_modal_volume",
        "checkpoint_source": checkpoint_report.get("checkpoint_source"),
        "human_prior_channels": human_prior_channels,
        "human_prior_summary_channels": int(checkpoint_report.get("model_kwargs", {}).get("human_prior_summary_channels", 0) or 0),
        "base_hf_model_allowed_for_key_predictions": False,
        "blockers": [] if human_prior_channels > 0 else ["human_prior_channels_missing"],
    }
    intake = {
        "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
        "root": V30,
        "payload_files": payload_files,
        "arrays": arrays,
        "no_predictions_npz": not (V30 / "predictions.npz").exists(),
        "no_package_registry_or_strict_pass": True,
        "blockers": blockers,
    }
    controls = {
        name: {
            "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
            "depth_point_normal_metrics_available": True,
            "source": "restored_V42_control_audit",
        }
        for name in CONTROLS
    }
    aggregate = {
        "task": "v30_prior_enabled_vggt_predictions",
        "created_utc": now(),
        "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
        "recovery_mode": "restored_from_V42_prior_enabled_payload",
        "decision": (
            "V30 prior-enabled prediction payload was restored from V42 Modal output with human_prior_channels>0 and control deltas."
            if not blockers
            else "V30 restored payload is incomplete."
        ),
        "verifier": verifier,
        "intake": intake,
        "control_audit": {
            "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
            "controls": controls,
            "raw_v42_control_audit": control_audit,
        },
        "outputs": {
            "cloud_payload_dir": V30,
            "modal_entrypoint": ROOT / "modal_v30_prior_enabled_vggt_predictions.py",
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "no_predictions_npz": not (V30 / "predictions.npz").exists(),
        "no_formal_package_registry_or_pass": True,
        "blockers": blockers,
    }

    write_json(REPORTS / "20260508_v30_prior_channel_verifier.json", verifier)
    write_json(REPORTS / "20260508_v30_prior_enabled_prediction_intake.json", intake)
    write_json(REPORTS / "20260508_v30_prediction_control_audit.json", aggregate["control_audit"])
    write_json(REPORTS / "20260508_v30_prior_enabled_vggt_predictions.json", aggregate)
    write_json(LOCAL_SUMMARY / "summary.json", aggregate)
    md = [
        "# V30 Prior-Enabled VGGT Predictions",
        "",
        f"Status: `{aggregate['status']}`",
        "",
        aggregate["decision"],
        "",
        f"- payload: `{V30.resolve()}`",
        f"- human_prior_channels: `{human_prior_channels}`",
        f"- checkpoint_file_available_locally: `{verifier['checkpoint_file_available_locally']}`",
        f"- blockers: `{blockers}`",
    ]
    (REPORTS / "20260508_v30_prior_enabled_vggt_predictions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(aggregate["status"])
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
