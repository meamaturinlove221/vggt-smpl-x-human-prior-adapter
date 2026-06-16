#!/usr/bin/env python
"""V41b real prior-enabled VGGT checkpoint builder.

This audit constructs a loadable VGGT state_dict from the local base VGGT
checkpoint plus the repo's native HumanPriorAdapter. It explicitly rejects the
V39 PriorAdapterMicrofit compact checkpoint as a HumanPriorAdapter source unless
direct key evidence exists under aggregator.human_prior_adapter.*.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt.models.vggt import VGGT  # noqa: E402


OUT = ROOT / "output" / "surface_research_preflight_local" / "V41b_real_prior_enabled_checkpoint"
REPORT_JSON = ROOT / "reports" / "20260509_v41b_real_prior_enabled_checkpoint_builder.json"
REPORT_MD = ROOT / "reports" / "20260509_v41b_real_prior_enabled_checkpoint_builder.md"
BASE_CHECKPOINT = Path(
    os.environ.get("V41B_BASE_CHECKPOINT", r"C:\Users\WINDOWS\.cache\torch\hub\checkpoints\model.pt")
)
NATIVE_CONFIG = ROOT / "training" / "config" / "4k4d_smplx_native_prior.yaml"
V38_DIR = ROOT / "output" / "surface_research_preflight_local" / "V38_prior_enabled_checkpoint_scaffold"
V39_DIR = ROOT / "output" / "surface_research_preflight_local" / "V39_adapter_microfit"

FORBIDDEN_FILENAMES = {"predictions.npz"}
FORBIDDEN_TOKENS = (
    "strict_pass",
    "strict_gate_registry",
    "formal_candidate",
    "candidate_package",
    "teacher_package",
    "teacher_export",
    "candidate_export",
)

DEFAULT_MODEL_KWARGS: dict[str, Any] = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_point": True,
    "enable_depth": True,
    "enable_normal": True,
    "enable_track": False,
    "human_prior_channels": 29,
    "human_prior_summary_channels": 27,
    "human_prior_hidden_dim": 128,
    "human_prior_gate_init": 0.0,
    "human_prior_multi_scale_factors": [1, 2, 4],
    "human_prior_enable_input_fusion": True,
    "human_prior_enable_frame_fusion": True,
    "human_prior_enable_global_fusion": True,
    "human_prior_enable_summary_fusion": True,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def partial_sha256(path: Path, max_bytes: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "mtime": path.stat().st_mtime if path.is_file() else None,
        "sha256_first_1m": partial_sha256(path) if path.is_file() else None,
    }


def scan_forbidden(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not root.exists():
        return findings
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix().lower()
        if path.is_file() and path.name.lower() in FORBIDDEN_FILENAMES:
            findings.append({"path": str(path.resolve()), "reason": "forbidden formal predictions.npz"})
        hits = [token for token in FORBIDDEN_TOKENS if token in rel]
        if hits:
            findings.append({"path": str(path.resolve()), "reason": "formal-output token(s): " + ",".join(hits)})
    return findings


def torch_load(path: Path, map_location: str = "cpu", mmap: bool = False) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=mmap)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if is_tensor_mapping(payload):
        return dict(payload), "top_level_tensor_mapping"
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state", "model_state_dict"):
            candidate = payload.get(key)
            if is_tensor_mapping(candidate):
                return dict(candidate), key
    raise TypeError(f"Unsupported checkpoint payload for state_dict extraction: {type(payload)!r}")


def strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "_orig_mod.", "model.", "state_dict.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def key_bucket(key: str) -> str:
    if key.startswith("aggregator.human_prior_adapter."):
        return "aggregator.human_prior_adapter"
    if key.startswith("aggregator."):
        return "aggregator"
    if key.startswith("camera_head."):
        return "camera_head"
    if key.startswith("depth_head."):
        return "depth_head"
    if key.startswith("point_head."):
        return "point_head"
    if key.startswith("normal_head."):
        return "normal_head"
    if key.startswith("track_head."):
        return "track_head"
    return key.split(".", 1)[0]


def tensor_brief(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
    }


def state_dict_features(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = list(state_dict.keys())
    has = {
        "aggregator": any(k.startswith("aggregator.") for k in keys),
        "camera_head": any(k.startswith("camera_head.") for k in keys),
        "depth_head": any(k.startswith("depth_head.") for k in keys),
        "point_head": any(k.startswith("point_head.") for k in keys),
        "normal_head": any(k.startswith("normal_head.") for k in keys),
        "track_head": any(k.startswith("track_head.") for k in keys),
        "human_prior_adapter": any(k.startswith("aggregator.human_prior_adapter.") for k in keys),
        "human_prior_projector": "aggregator.human_prior_adapter.proj.0.weight" in state_dict,
        "human_prior_frame_fusions": any(k.startswith("aggregator.human_prior_adapter.frame_fusions.") for k in keys),
        "human_prior_global_fusions": any(k.startswith("aggregator.human_prior_adapter.global_fusions.") for k in keys),
        "human_prior_summary_fusions": any(k.startswith("aggregator.human_prior_adapter.global_summary_fusions.") for k in keys),
    }
    full_vggt = bool(has["aggregator"] and has["camera_head"] and has["depth_head"] and has["point_head"])
    prior_enabled = bool(full_vggt and has["human_prior_adapter"] and has["human_prior_projector"])
    adapter_weight = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_weight = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    hidden_dim = int(adapter_weight.shape[0]) if adapter_weight is not None else None
    prior_channels = int(adapter_weight.shape[1]) if adapter_weight is not None else None
    summary_channels = int(summary_weight.shape[1]) if summary_weight is not None else None
    return {
        "key_count": len(keys),
        "tensor_count_by_bucket": dict(Counter(key_bucket(k) for k in keys)),
        "has": has,
        "full_vggt_state_dict": full_vggt,
        "prior_enabled_full_vggt_state_dict": prior_enabled,
        "human_prior_channels": prior_channels,
        "human_prior_summary_channels": summary_channels,
        "human_prior_hidden_dim": hidden_dim,
        "sample_keys": keys[:24],
    }


def load_native_model_kwargs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs = dict(DEFAULT_MODEL_KWARGS)
    evidence: dict[str, Any] = {"config_path": str(config_path.resolve()), "exists": config_path.is_file(), "source": "defaults"}
    if config_path.is_file():
        try:
            import yaml

            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            model_cfg = payload.get("model") or {}
            if isinstance(model_cfg, Mapping):
                for key, value in model_cfg.items():
                    if key.startswith("_"):
                        continue
                    if key in kwargs:
                        kwargs[key] = value
                evidence.update({"source": "training/config/4k4d_smplx_native_prior.yaml:model", "model_config": dict(model_cfg)})
        except Exception as exc:  # noqa: BLE001
            evidence.update({"source": "defaults_after_yaml_error", "error": f"{type(exc).__name__}: {exc}"})
    if isinstance(kwargs.get("human_prior_multi_scale_factors"), tuple):
        kwargs["human_prior_multi_scale_factors"] = list(kwargs["human_prior_multi_scale_factors"])
    for key in ("img_size", "patch_size", "embed_dim", "human_prior_channels", "human_prior_summary_channels", "human_prior_hidden_dim"):
        kwargs[key] = int(kwargs[key])
    for key in (
        "enable_camera",
        "enable_point",
        "enable_depth",
        "enable_normal",
        "enable_track",
        "human_prior_enable_input_fusion",
        "human_prior_enable_frame_fusion",
        "human_prior_enable_global_fusion",
        "human_prior_enable_summary_fusion",
    ):
        kwargs[key] = bool(kwargs[key])
    kwargs["human_prior_gate_init"] = float(kwargs["human_prior_gate_init"])
    kwargs["human_prior_multi_scale_factors"] = [int(v) for v in kwargs["human_prior_multi_scale_factors"]]
    return kwargs, evidence


def inspect_v39_surrogate(v39_path: Path, target_adapter_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(v39_path.resolve()),
        "file": file_info(v39_path),
        "loaded": False,
        "classification": "missing",
        "accepted_as_human_prior_adapter_source": False,
    }
    if not v39_path.is_file():
        row["reason"] = "missing_v39_checkpoint"
        return row
    try:
        payload = torch_load(v39_path, map_location="cpu", mmap=False)
        state, state_source = extract_state_dict(payload)
        keys = list(state.keys())
        direct_adapter_keys = [k for k in keys if k.startswith("aggregator.human_prior_adapter.")]
        target_adapter_keys = [k for k in target_adapter_state if k.startswith("aggregator.human_prior_adapter.")]
        exact_key_matches = [k for k in direct_adapter_keys if k in target_adapter_state and tuple(state[k].shape) == tuple(target_adapter_state[k].shape)]
        shape_only_matches = []
        target_shapes = {k: tuple(v.shape) for k, v in target_adapter_state.items() if k.startswith("aggregator.human_prior_adapter.")}
        for source_key, source_tensor in state.items():
            matches = [target_key for target_key, target_shape in target_shapes.items() if tuple(source_tensor.shape) == target_shape]
            if matches:
                shape_only_matches.append({"source_key": source_key, "source_shape": list(source_tensor.shape), "target_key_sample": matches[:4], "match_count": len(matches)})
        features = state_dict_features(state) if any(k.startswith("aggregator.") for k in keys) else {
            "key_count": len(keys),
            "tensor_count_by_bucket": dict(Counter(key_bucket(k) for k in keys)),
            "full_vggt_state_dict": False,
            "prior_enabled_full_vggt_state_dict": False,
            "sample_keys": keys[:24],
        }
        model_class = payload.get("model_class") if isinstance(payload, Mapping) else None
        compact_surrogate = bool(
            model_class == "PriorAdapterMicrofit"
            or {"region_bias", "gate_logit"}.issubset(set(keys))
            or any(k.startswith("net.") for k in keys)
        )
        accepted = bool(features.get("prior_enabled_full_vggt_state_dict") and len(exact_key_matches) >= max(8, int(0.5 * len(target_adapter_keys))))
        row.update(
            {
                "loaded": True,
                "state_source": state_source,
                "checkpoint_kind": payload.get("checkpoint_kind") if isinstance(payload, Mapping) else None,
                "model_class": model_class,
                "features": features,
                "direct_human_prior_adapter_key_count": len(direct_adapter_keys),
                "target_human_prior_adapter_key_count": len(target_adapter_keys),
                "exact_human_prior_adapter_key_matches": exact_key_matches[:32],
                "exact_human_prior_adapter_key_match_count": len(exact_key_matches),
                "shape_only_matches_to_adapter_count": len(shape_only_matches),
                "shape_only_matches_to_adapter_sample": shape_only_matches[:12],
                "compact_adapter_surrogate": compact_surrogate,
                "classification": "compact_adapter_surrogate" if compact_surrogate else "unknown_non_full_vggt",
                "accepted_as_human_prior_adapter_source": accepted,
                "rejection_reason": None
                if accepted
                else "no direct aggregator.human_prior_adapter.* key mapping evidence; V39 PriorAdapterMicrofit remains compact surrogate",
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"loaded": False, "classification": "load_error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def inspect_v38_scaffold() -> dict[str, Any]:
    config = load_json(V38_DIR / "config.json")
    summary = load_json(V38_DIR / "summary.json")
    load_report = load_json(V38_DIR / "load_report.json")
    return {
        "dir": str(V38_DIR.resolve()),
        "state_dict": file_info(V38_DIR / "state_dict.pt"),
        "config": config,
        "load_report": load_report,
        "summary_status": summary.get("status"),
        "classification": "full_prior_enabled_scaffold_untrained_adapter",
        "used_as_source_for_v41b": False,
        "reason": "V41b uses local base model.pt for transferred VGGT weights; V38 metadata says scaffold/trained_adapter false.",
    }


def build_base_index(base_state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    index: dict[str, str] = {}
    for key in base_state:
        variants = (key, strip_known_prefixes(key))
        for variant in variants:
            index.setdefault(variant, key)
    return index


def transfer_base_weights(model: VGGT, base_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    target_state = model.state_dict()
    base_index = build_base_index(base_state)
    transferred: list[dict[str, Any]] = []
    shape_mismatch: list[dict[str, Any]] = []
    newly_initialized: list[str] = []
    used_base_keys: set[str] = set()

    for target_key, target_tensor in target_state.items():
        source_key = base_index.get(target_key)
        if source_key is None:
            newly_initialized.append(target_key)
            continue
        source_tensor = base_state[source_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch.append(
                {
                    "target_key": target_key,
                    "source_key": source_key,
                    "target": tensor_brief(target_tensor),
                    "source": tensor_brief(source_tensor),
                }
            )
            newly_initialized.append(target_key)
            continue
        with torch.no_grad():
            target_tensor.copy_(source_tensor.detach().to(device=target_tensor.device, dtype=target_tensor.dtype))
        used_base_keys.add(source_key)
        transferred.append(
            {
                "target_key": target_key,
                "source_key": source_key,
                "shape": list(target_tensor.shape),
                "dtype": str(target_tensor.dtype),
                "numel": int(target_tensor.numel()),
                "bytes": int(target_tensor.numel() * target_tensor.element_size()),
            }
        )

    base_unused = [key for key in base_state.keys() if key not in used_base_keys]
    newly_by_bucket = Counter(key_bucket(k) for k in newly_initialized)
    transferred_by_bucket = Counter(key_bucket(item["target_key"]) for item in transferred)
    base_unused_by_bucket = Counter(key_bucket(k) for k in base_unused)
    non_adapter_new = [
        key
        for key in newly_initialized
        if not key.startswith("aggregator.human_prior_adapter.") and not key.startswith("normal_head.")
    ]
    target_numel = int(sum(t.numel() for t in target_state.values()))
    transferred_numel = int(sum(item["numel"] for item in transferred))
    return {
        "target_tensor_count": len(target_state),
        "base_tensor_count": len(base_state),
        "target_numel": target_numel,
        "transferred_tensor_count": len(transferred),
        "transferred_numel": transferred_numel,
        "transferred_fraction_of_target_numel": float(transferred_numel / max(1, target_numel)),
        "transferred_bytes": int(sum(item["bytes"] for item in transferred)),
        "transferred_by_bucket": dict(transferred_by_bucket),
        "newly_initialized_target_key_count": len(newly_initialized),
        "newly_initialized_target_keys_by_bucket": dict(newly_by_bucket),
        "newly_initialized_target_keys_sample": newly_initialized[:80],
        "newly_initialized_non_adapter_non_normal_keys": non_adapter_new[:80],
        "newly_initialized_non_adapter_non_normal_key_count": len(non_adapter_new),
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatches": shape_mismatch[:80],
        "base_unused_key_count": len(base_unused),
        "base_unused_by_bucket": dict(base_unused_by_bucket),
        "base_unused_keys_sample": base_unused[:80],
        "base_weight_source_used": True,
    }


def save_checkpoint(path: Path, model: VGGT, model_kwargs: dict[str, Any], transfer_report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": "v41b_real_prior_enabled_vggt_from_local_base_plus_repo_human_prior_adapter",
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "created_utc": utc_now(),
            "source_base_checkpoint": str(BASE_CHECKPOINT.resolve()),
            "base_weight_transfer_summary": {
                "transferred_tensor_count": transfer_report.get("transferred_tensor_count"),
                "newly_initialized_target_key_count": transfer_report.get("newly_initialized_target_key_count"),
                "newly_initialized_target_keys_by_bucket": transfer_report.get("newly_initialized_target_keys_by_bucket"),
                "shape_mismatch_count": transfer_report.get("shape_mismatch_count"),
            },
            "human_prior_adapter_source": "repo.vggt.models.human_prior.HumanPriorAdapter newly initialized in target VGGT",
            "human_prior_adapter_training_status": "untrained_repo_adapter_initialized_not_v39_microfit",
            "v39_prior_adapter_microfit_used": False,
            "external_mano_flame_hairgs_used": False,
            "formal_outputs_written": False,
        },
        path,
    )


def strict_reload_report(checkpoint_path: Path, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = torch_load(checkpoint_path, map_location="cpu", mmap=True)
    state, state_source = extract_state_dict(payload)
    model = VGGT(**model_kwargs)
    result = model.load_state_dict(state, strict=True)
    features = state_dict_features(state)
    report = {
        "status": "DONE_PASS",
        "checkpoint": file_info(checkpoint_path),
        "state_source": state_source,
        "strict_load": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "features": features,
        "model_kwargs": model_kwargs,
        "has_human_prior_adapter_module": model.aggregator.human_prior_adapter is not None,
        "human_prior_channels": int(model.aggregator.human_prior_channels),
        "human_prior_summary_channels": int(model.aggregator.human_prior_summary_channels),
        "has_normal_head": model.normal_head is not None,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "classification": "prior_enabled_full_vggt_state_dict"
        if features.get("prior_enabled_full_vggt_state_dict")
        else "not_prior_enabled_full_vggt_state_dict",
    }
    del model
    del state
    del payload
    gc.collect()
    return report


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    outputs = summary.get("outputs", {})
    transfer = summary.get("base_weight_transfer", {})
    load_report = summary.get("load_report", {})
    v39 = summary.get("v39_surrogate_audit", {})
    lines = [
        "# V41b Real Prior-Enabled Checkpoint Builder",
        "",
        f"status: `{summary.get('status')}`",
        f"decision: `{summary.get('decision')}`",
        "",
        "## Outputs",
        "",
        f"- checkpoint: `{outputs.get('checkpoint_pt')}`",
        f"- load_report: `{outputs.get('load_report_json')}`",
        f"- base_weight_transfer_report: `{outputs.get('base_weight_transfer_report_json')}`",
        "",
        "## Evidence",
        "",
        f"- base checkpoint: `{summary.get('base_checkpoint', {}).get('path')}`",
        f"- strict reload: `{load_report.get('strict_load')}`",
        f"- full prior-enabled VGGT state_dict: `{load_report.get('features', {}).get('prior_enabled_full_vggt_state_dict')}`",
        f"- transferred tensors: `{transfer.get('transferred_tensor_count')}`",
        f"- newly initialized keys by bucket: `{transfer.get('newly_initialized_target_keys_by_bucket')}`",
        f"- shape mismatches: `{transfer.get('shape_mismatch_count')}`",
        "",
        "## V39 Surrogate Check",
        "",
        f"- classification: `{v39.get('classification')}`",
        f"- model_class: `{v39.get('model_class')}`",
        f"- accepted_as_human_prior_adapter_source: `{v39.get('accepted_as_human_prior_adapter_source')}`",
        f"- reason: `{v39.get('rejection_reason')}`",
        "",
        "## Safety",
        "",
        f"- formal_outputs_written: `{summary.get('formal_outputs_written')}`",
        f"- external_mano_flame_hairgs_used: `{summary.get('external_mano_flame_hairgs_used')}`",
        f"- forbidden_hits: `{summary.get('forbidden_hits')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def assert_safe_clean(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    expected_parent = (ROOT / "output" / "surface_research_preflight_local").resolve()
    if resolved.name != "V41b_real_prior_enabled_checkpoint" or resolved.parent != expected_parent:
        raise RuntimeError(f"Refusing to clean unexpected output path: {resolved}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    if args.clean and out_dir.exists():
        assert_safe_clean(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    model_kwargs, config_evidence = load_native_model_kwargs(Path(args.config))
    if args.enable_track:
        model_kwargs["enable_track"] = True
    if args.img_size is not None:
        model_kwargs["img_size"] = int(args.img_size)

    base_path = Path(args.base_checkpoint)
    hard_evidence_path = out_dir / "hard_evidence.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    load_report_path = out_dir / "load_report.json"
    transfer_report_path = out_dir / "base_weight_transfer_report.json"

    if not base_path.is_file():
        hard = {
            "status": "DONE_FAIL_ROUTED",
            "created_utc": utc_now(),
            "reason": "missing_local_base_checkpoint",
            "base_checkpoint": file_info(base_path),
            "model_kwargs": model_kwargs,
        }
        write_json(hard_evidence_path, hard)
        write_json(load_report_path, {"status": "DONE_FAIL_ROUTED", "reason": "checkpoint_not_constructed"})
        write_json(transfer_report_path, hard)
        return hard

    base_payload = torch_load(base_path, map_location="cpu", mmap=True)
    base_state, base_state_source = extract_state_dict(base_payload)
    base_features = state_dict_features(base_state)

    model = VGGT(**model_kwargs)
    target_features_before = state_dict_features(model.state_dict())
    transfer_report = transfer_base_weights(model, base_state)
    transfer_report.update(
        {
            "created_utc": utc_now(),
            "base_checkpoint": file_info(base_path),
            "base_state_source": base_state_source,
            "base_features": base_features,
            "target_model_kwargs": model_kwargs,
            "target_features_before_transfer": target_features_before,
            "target_features_after_transfer": state_dict_features(model.state_dict()),
            "v38_scaffold_audit": inspect_v38_scaffold(),
        }
    )
    write_json(transfer_report_path, transfer_report)

    v39_audit = inspect_v39_surrogate(V39_DIR / "v39_adapter_only_checkpoint.pt", model.state_dict())
    del base_state
    del base_payload
    gc.collect()

    non_adapter_new = int(transfer_report.get("newly_initialized_non_adapter_non_normal_key_count", 0))
    if args.no_checkpoint:
        hard = {
            "status": "DONE_FAIL_ROUTED",
            "created_utc": utc_now(),
            "reason": "no_checkpoint_flag_set",
            "base_weight_transfer_report": str(transfer_report_path.resolve()),
            "v39_surrogate_audit": v39_audit,
        }
        write_json(hard_evidence_path, hard)
        load_report = {"status": "DONE_FAIL_ROUTED", "reason": "checkpoint_not_written_no_checkpoint_flag"}
    else:
        save_checkpoint(checkpoint_path, model, model_kwargs, transfer_report)
        del model
        gc.collect()
        load_report = strict_reload_report(checkpoint_path, model_kwargs)

    write_json(load_report_path, load_report)
    forbidden_hits = scan_forbidden(out_dir)
    checkpoint_written = checkpoint_path.is_file() and not args.no_checkpoint
    strict_ok = bool(load_report.get("strict_load") and not load_report.get("missing_keys") and not load_report.get("unexpected_keys"))
    prior_full = bool(load_report.get("features", {}).get("prior_enabled_full_vggt_state_dict"))
    transfer_ok = bool(
        transfer_report.get("shape_mismatch_count") == 0
        and non_adapter_new == 0
        and transfer_report.get("transferred_tensor_count", 0) > 0
    )
    v39_rejected = not bool(v39_audit.get("accepted_as_human_prior_adapter_source"))
    status = "DONE_PASS" if checkpoint_written and strict_ok and prior_full and transfer_ok and v39_rejected and not forbidden_hits else "DONE_FAIL_ROUTED"
    decision = (
        "constructed_base_transferred_prior_enabled_vggt_checkpoint_with_untrained_repo_human_prior_adapter"
        if status == "DONE_PASS"
        else "not_promoted_review_hard_evidence"
    )
    summary = {
        "task": "v41b_real_prior_enabled_checkpoint_builder",
        "created_utc": utc_now(),
        "status": status,
        "decision": decision,
        "research_only": True,
        "formal_outputs_written": False,
        "external_mano_flame_hairgs_used": False,
        "base_checkpoint": file_info(base_path),
        "native_config_evidence": config_evidence,
        "model_kwargs": model_kwargs,
        "real_prior_enabled_checkpoint_constructed": bool(status == "DONE_PASS"),
        "adapter_training_status": "repo_HumanPriorAdapter_initialized_untrained",
        "v39_prior_adapter_microfit_used": False,
        "outputs": {
            "checkpoint_pt": str(checkpoint_path.resolve()) if checkpoint_written else None,
            "hard_evidence_json": str(hard_evidence_path.resolve()) if hard_evidence_path.is_file() else None,
            "load_report_json": str(load_report_path.resolve()),
            "base_weight_transfer_report_json": str(transfer_report_path.resolve()),
        },
        "base_weight_transfer": transfer_report,
        "load_report": load_report,
        "v39_surrogate_audit": v39_audit,
        "v38_scaffold_audit": transfer_report.get("v38_scaffold_audit"),
        "checks": {
            "checkpoint_written": checkpoint_written,
            "strict_reload_ok": strict_ok,
            "prior_enabled_full_vggt_state_dict": prior_full,
            "base_transfer_ok": transfer_ok,
            "non_adapter_non_normal_new_key_count": non_adapter_new,
            "v39_rejected_as_human_prior_adapter_source": v39_rejected,
        },
        "forbidden_hits": forbidden_hits,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(REPORT_JSON, summary)
    write_markdown(REPORT_MD, summary)
    print(
        json.dumps(
            {
                "status": status,
                "checkpoint": summary["outputs"]["checkpoint_pt"],
                "load_report": summary["outputs"]["load_report_json"],
                "base_weight_transfer_report": summary["outputs"]["base_weight_transfer_report_json"],
                "decision": decision,
            },
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct and audit a real prior-enabled VGGT checkpoint from local base model.pt.")
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=NATIVE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--enable-track", action="store_true", help="Also keep/transfer the base track head.")
    parser.add_argument("--img-size", type=int, default=None, help="Override img_size; default follows SMPL-X native prior config.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Write hard evidence only; do not save checkpoint.pt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
        return 0 if summary.get("status") == "DONE_PASS" else 2
    except Exception as exc:  # noqa: BLE001
        OUT.mkdir(parents=True, exist_ok=True)
        evidence = {
            "task": "v41b_real_prior_enabled_checkpoint_builder",
            "created_utc": utc_now(),
            "status": "DONE_FAIL_ROUTED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "formal_outputs_written": False,
            "external_mano_flame_hairgs_used": False,
        }
        write_json(OUT / "hard_evidence.json", evidence)
        write_json(OUT / "load_report.json", {"status": "DONE_FAIL_ROUTED", "error": evidence["error"]})
        write_json(OUT / "base_weight_transfer_report.json", evidence)
        write_json(REPORT_JSON, evidence)
        write_markdown(REPORT_MD, evidence)
        print(json.dumps({"status": "DONE_FAIL_ROUTED", "error": evidence["error"], "hard_evidence": str((OUT / "hard_evidence.json").resolve())}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
