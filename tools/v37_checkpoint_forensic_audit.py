#!/usr/bin/env python
"""V37 forensic audit for the V31 research checkpoint.

This audit intentionally distinguishes compact residual/teacher checkpoints
from loadable VGGT state_dict checkpoints. It does not promote or package
anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V31_CKPT = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train" / "v31_candidate_research_checkpoint.npz"
V31_SUMMARY = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train" / "summary.json"
REPORT_JSON = ROOT / "reports" / "20260509_v37_checkpoint_forensic_audit.json"
REPORT_MD = ROOT / "reports" / "20260509_v37_checkpoint_forensic_audit.md"


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "scalar": bool(value.shape == ()),
    }
    if value.shape == ():
        try:
            row["value"] = value.item()
        except Exception:  # noqa: BLE001
            row["value_repr"] = repr(value)
    elif np.issubdtype(value.dtype, np.number):
        finite = np.isfinite(value)
        row.update(
            {
                "finite_ratio": float(finite.mean()) if finite.size else 0.0,
                "min": float(np.nanmin(value)) if value.size else None,
                "max": float(np.nanmax(value)) if value.size else None,
                "mean": float(np.nanmean(value)) if value.size else None,
            }
        )
    return row


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ckpt_info = _file_info(V31_CKPT)
    summary = _load_json(V31_SUMMARY)
    blockers: list[str] = []
    npz_keys: list[str] = []
    key_summaries: dict[str, Any] = {}
    has_state_dict = False
    has_adapter_weights = False
    has_vggt_heads = False
    has_optimizer_state = False
    has_config = False
    human_prior_channels = None
    train_target = None
    loss_or_controls = None

    if not V31_CKPT.is_file():
        blockers.append("missing_v31_checkpoint")
    else:
        with np.load(V31_CKPT, allow_pickle=True) as payload:
            npz_keys = list(payload.files)
            for key in npz_keys:
                value = payload[key]
                key_summaries[key] = _array_summary(value)
                lower = key.lower()
                has_state_dict = has_state_dict or "state_dict" in lower or lower in {"model", "model_state"}
                has_adapter_weights = has_adapter_weights or "human_prior_adapter" in lower or "prior_adapter" in lower
                has_vggt_heads = has_vggt_heads or lower.startswith(("depth_head", "point_head", "camera_head", "normal_head"))
                has_optimizer_state = has_optimizer_state or "optimizer" in lower
                has_config = has_config or lower in {"config", "model_config", "training_config"}
                if lower == "human_prior_channels":
                    human_prior_channels = value.item() if value.shape == () else value.tolist()
                if lower in {"v24_teacher_npz", "v26_temporal_npz", "checkpoint_kind"}:
                    train_target = train_target or {}
                    try:
                        train_target[lower] = value.item()
                    except Exception:  # noqa: BLE001
                        train_target[lower] = str(value)

    if summary:
        loss_or_controls = {
            "status": summary.get("status"),
            "control_metrics_present": isinstance(summary.get("control_metrics"), dict),
            "real_wins_controls": summary.get("real_wins_controls"),
            "normal_source": summary.get("normal_source"),
        }

    expected_residual_keys = {"temporal_blend", "depth_scale", "control_bias", "region_offsets", "region_names"}
    residual_like = expected_residual_keys.issubset(set(npz_keys))
    loadable_vggt_state_dict = bool(has_state_dict and has_adapter_weights and has_vggt_heads)

    if loadable_vggt_state_dict:
        conclusion_case = "A"
        conclusion = "V31 appears directly convertible to a prior-enabled VGGT checkpoint."
        status = "DONE_PASS"
    elif residual_like:
        conclusion_case = "B"
        conclusion = "V31 is a compact residual/teacher research checkpoint; it cannot be loaded as VGGT state_dict and requires V38/V39 adapter training."
        status = "DONE_FAIL_ROUTED"
    else:
        conclusion_case = "C"
        conclusion = "V31 contains insufficient model-loading information and can only be treated as teacher evidence."
        status = "DONE_FAIL_ROUTED"

    if not loadable_vggt_state_dict:
        blockers.append("v31_checkpoint_not_loadable_vggt_state_dict")
    if not has_adapter_weights:
        blockers.append("missing_human_prior_adapter_weights")
    if not has_vggt_heads:
        blockers.append("missing_vggt_head_weights")

    audit = {
        "task": "v37_checkpoint_forensic_audit",
        "status": status,
        "research_only": True,
        "no_formal_outputs": True,
        "v31_checkpoint": ckpt_info,
        "v31_summary": _file_info(V31_SUMMARY),
        "npz_keys": npz_keys,
        "key_summaries": key_summaries,
        "checks": {
            "has_state_dict": has_state_dict,
            "has_human_prior_adapter_weights": has_adapter_weights,
            "has_vggt_head_weights": has_vggt_heads,
            "has_optimizer_state": has_optimizer_state,
            "has_config": has_config,
            "human_prior_channels": human_prior_channels,
            "residual_like_checkpoint": residual_like,
            "loadable_vggt_state_dict": loadable_vggt_state_dict,
        },
        "train_target": train_target,
        "loss_or_control_record": loss_or_controls,
        "conclusion_case": conclusion_case,
        "conclusion": conclusion,
        "blockers": blockers,
        "route_decision": "V38_prior_enabled_checkpoint_scaffold",
    }
    REPORT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# V37 Checkpoint Forensic Audit",
        "",
        f"status: `{status}`",
        f"conclusion_case: `{conclusion_case}`",
        "",
        conclusion,
        "",
        "## Checkpoint",
        f"- path: `{ckpt_info['path']}`",
        f"- size: `{ckpt_info['size']}`",
        f"- keys: `{npz_keys}`",
        "",
        "## Loadability",
        f"- has_state_dict: `{has_state_dict}`",
        f"- has_human_prior_adapter_weights: `{has_adapter_weights}`",
        f"- has_vggt_head_weights: `{has_vggt_heads}`",
        f"- residual_like_checkpoint: `{residual_like}`",
        f"- loadable_vggt_state_dict: `{loadable_vggt_state_dict}`",
        "",
        "## Blockers",
        f"`{blockers}`",
        "",
        "## Route",
        "`V38_prior_enabled_checkpoint_scaffold`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "conclusion_case": conclusion_case, "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
