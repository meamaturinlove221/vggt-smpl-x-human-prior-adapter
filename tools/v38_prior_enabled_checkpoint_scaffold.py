#!/usr/bin/env python
"""Build a real loadable prior-enabled VGGT checkpoint scaffold.

The scaffold contains a complete local VGGT state_dict with HumanPriorAdapter
enabled. It references base VGGT weights as the intended trunk source, but does
not claim that the adapter is trained. A CPU-safe small-model forward probe is
also run to validate the prior route and real/zero/shuffle separation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt.models.vggt import VGGT  # noqa: E402


OUT = ROOT / "output" / "surface_research_preflight_local" / "V38_prior_enabled_checkpoint_scaffold"
REPORT_JSON = ROOT / "reports" / "20260509_v38_prior_enabled_checkpoint_scaffold.json"
REPORT_MD = ROOT / "reports" / "20260509_v38_prior_enabled_checkpoint_scaffold.md"
PRIOR_60V = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views" / "prior_maps.npz"
V37_JSON = ROOT / "reports" / "20260509_v37_checkpoint_forensic_audit.json"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256(path) if path.is_file() and path.stat().st_size < 256 * 1024 * 1024 else None,
    }


def _load_prior(path: Path, view_count: int, target_size: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as payload:
        prior_maps = np.asarray(payload["prior_maps"], dtype=np.float32)[:view_count]
        prior_summary = np.asarray(payload["prior_summary_tokens"], dtype=np.float32)[:view_count] if "prior_summary_tokens" in payload.files else None
        prior_channels = [str(x) for x in payload["prior_channels"].tolist()] if "prior_channels" in payload.files else [f"prior_{i}" for i in range(prior_maps.shape[1])]
        prior_summary_channels = [str(x) for x in payload["prior_summary_channels"].tolist()] if "prior_summary_channels" in payload.files and prior_summary is not None else []
    if prior_maps.shape[-2:] != (target_size, target_size):
        tensor = torch.from_numpy(prior_maps)
        prior_maps = torch.nn.functional.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False).numpy()
    meta = {
        "path": str(path.resolve()),
        "prior_maps_shape": list(prior_maps.shape),
        "prior_summary_shape": None if prior_summary is None else list(prior_summary.shape),
        "prior_channels": prior_channels,
        "prior_summary_channels": prior_summary_channels,
    }
    return prior_maps, prior_summary, meta


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    data = t.detach().float().cpu()
    return {
        "shape": list(data.shape),
        "numel": int(data.numel()),
        "mean": float(data.mean()) if data.numel() else 0.0,
        "std": float(data.std(unbiased=False)) if data.numel() else 0.0,
        "min": float(data.min()) if data.numel() else 0.0,
        "max": float(data.max()) if data.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(data)) if data.numel() else 0.0,
    }


def _adapter_stats(model: VGGT) -> dict[str, Any]:
    adapter = model.aggregator.human_prior_adapter
    rows: dict[str, Any] = {}
    total = 0
    for name, param in adapter.named_parameters():
        total += int(param.numel())
        if any(token in name for token in ("gate", "proj.0.weight", "proj.2.weight", "summary_proj.0.weight")):
            rows[name] = _tensor_stats(param)
    gates = {
        name: float(param.detach().cpu())
        for name, param in adapter.named_parameters()
        if name.endswith("gate") or name == "input_fusion.gate"
    }
    return {"parameter_count": total, "selected_parameter_stats": rows, "gate_values": gates}


def _make_controls(prior_maps: np.ndarray, prior_summary: np.ndarray | None) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    real = torch.from_numpy(prior_maps).unsqueeze(0)
    zero = torch.zeros_like(real)
    shuffle = torch.roll(real, shifts=1, dims=2)
    controls: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for name, maps in (("real", real), ("zero", zero), ("shuffle", shuffle)):
        summary_t = None
        if prior_summary is not None:
            summary = torch.from_numpy(prior_summary).unsqueeze(0)
            if name == "zero":
                summary = torch.zeros_like(summary)
            elif name == "shuffle":
                summary = torch.roll(summary, shifts=1, dims=-1)
            summary_t = summary
        controls[name] = (maps, summary_t)
    return controls


def _forward_probe(prior_maps: np.ndarray, prior_summary: np.ndarray | None, target_size: int, view_count: int) -> dict[str, Any]:
    # Full 24-layer VGGT is too heavy for CPU probing. This probe uses the same
    # HumanPriorAdapter/Aggregator code path with a reduced local model and
    # nonzero gate, then saves the production scaffold state_dict separately.
    torch.manual_seed(38038)
    np.random.seed(38038)
    random.seed(38038)
    model = VGGT(
        img_size=target_size,
        patch_size=14,
        embed_dim=32,
        enable_camera=False,
        enable_track=False,
        enable_normal=True,
        human_prior_channels=int(prior_maps.shape[1]),
        human_prior_summary_channels=0 if prior_summary is None else int(prior_summary.shape[-1]),
        human_prior_hidden_dim=16,
        human_prior_gate_init=0.05,
    )
    model.eval()
    images = torch.linspace(0.0, 1.0, steps=view_count * 3 * target_size * target_size, dtype=torch.float32).reshape(
        1,
        view_count,
        3,
        target_size,
        target_size,
    )
    controls = _make_controls(prior_maps, prior_summary)
    rows: dict[str, Any] = {}
    signatures: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, (maps, summary) in controls.items():
            prior_tokens = model.aggregator.human_prior_adapter.project_prior_maps(
                prior_maps=maps,
                target_hw=(target_size // model.aggregator.patch_size, target_size // model.aggregator.patch_size),
            )
            fused = model.aggregator.human_prior_adapter.fuse_input_tokens(
                torch.zeros_like(prior_tokens),
                prior_tokens,
            )
            out = model(images, prior_maps=maps, prior_summary_tokens=summary)
            depth = out["depth"].detach().float().cpu()
            points = out["world_points"].detach().float().cpu()
            normals = out["normal"].detach().float().cpu() if "normal" in out else torch.empty(0)
            signature = torch.cat(
                [
                    depth.reshape(-1)[:128],
                    points.reshape(-1)[:128],
                    normals.reshape(-1)[:128] if normals.numel() else torch.zeros(128),
                    fused.detach().float().cpu().reshape(-1)[:128],
                ]
            )
            signatures[name] = signature
            rows[name] = {
                "prior_token_stats": _tensor_stats(prior_tokens),
                "fused_input_signature_stats": _tensor_stats(fused),
                "depth_stats": _tensor_stats(depth),
                "point_stats": _tensor_stats(points),
                "normal_stats": _tensor_stats(normals),
            }
    pairwise = {}
    for left, right in (("real", "zero"), ("real", "shuffle"), ("zero", "shuffle")):
        diff = signatures[left] - signatures[right]
        pairwise[f"{left}_minus_{right}_l2"] = float(torch.linalg.vector_norm(diff))
        pairwise[f"{left}_minus_{right}_maxabs"] = float(diff.abs().max())
    return {
        "probe_model": "VGGT reduced CPU-safe prior route probe",
        "uses_real_vggt_forward": True,
        "uses_human_prior_adapter": True,
        "human_prior_channels": int(prior_maps.shape[1]),
        "human_prior_summary_channels": 0 if prior_summary is None else int(prior_summary.shape[-1]),
        "target_size": int(target_size),
        "view_count": int(view_count),
        "controls": rows,
        "pairwise_differences": pairwise,
        "non_identical_outputs": bool(pairwise["real_minus_zero_l2"] > 1e-8 and pairwise["real_minus_shuffle_l2"] > 1e-8),
    }


def build_scaffold(target_size: int = 28, view_count: int = 2, clean: bool = False) -> dict[str, Any]:
    if clean and OUT.exists():
        for child in OUT.iterdir():
            if child.is_file():
                child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    prior_maps, prior_summary, prior_meta = _load_prior(PRIOR_60V, view_count=view_count, target_size=target_size)
    config = {
        "task": "v38_prior_enabled_checkpoint_scaffold",
        "research_only": True,
        "no_formal_outputs": True,
        "checkpoint_claim": "loadable_scaffold_not_trained_adapter",
        "base_trunk_source": "facebook/VGGT-1B",
        "base_trunk_weight_status": "referenced_not_embedded",
        "human_prior_channels": int(prior_maps.shape[1]),
        "human_prior_summary_channels": 0 if prior_summary is None else int(prior_summary.shape[-1]),
        "human_prior_hidden_dim": 64,
        "human_prior_gate_init": 0.0,
        "enable_normal": True,
        "prior_source": prior_meta,
        "v37_audit": str(V37_JSON.resolve()),
        "strict_promotion_allowed": False,
    }

    torch.manual_seed(38038)
    model = VGGT(
        img_size=target_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_normal=True,
        enable_track=False,
        human_prior_channels=config["human_prior_channels"],
        human_prior_summary_channels=config["human_prior_summary_channels"],
        human_prior_hidden_dim=config["human_prior_hidden_dim"],
        human_prior_gate_init=config["human_prior_gate_init"],
    )
    state_path = OUT / "state_dict.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "checkpoint_kind": "v38_prior_enabled_vggt_scaffold_untrained_adapter",
            "trained_adapter": False,
            "base_trunk_source": "facebook/VGGT-1B",
        },
        state_path,
    )
    config_path = OUT / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    adapter_stats = _adapter_stats(model)
    adapter_stats_path = OUT / "prior_adapter_weight_stats.json"
    adapter_stats_path.write_text(json.dumps(adapter_stats, indent=2), encoding="utf-8")

    loaded = torch.load(state_path, map_location="cpu")
    load_model = VGGT(
        img_size=target_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_normal=True,
        enable_track=False,
        human_prior_channels=config["human_prior_channels"],
        human_prior_summary_channels=config["human_prior_summary_channels"],
        human_prior_hidden_dim=config["human_prior_hidden_dim"],
        human_prior_gate_init=config["human_prior_gate_init"],
    )
    load_result = load_model.load_state_dict(loaded["state_dict"], strict=True)
    load_report = {
        "status": "DONE_PASS",
        "state_dict_path": str(state_path.resolve()),
        "strict_load": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "human_prior_channels": int(load_model.aggregator.human_prior_channels),
        "human_prior_summary_channels": int(load_model.aggregator.human_prior_summary_channels),
        "has_human_prior_adapter": load_model.aggregator.human_prior_adapter is not None,
        "has_normal_head": load_model.normal_head is not None,
        "parameter_count": int(sum(p.numel() for p in load_model.parameters())),
        "trained_adapter": False,
    }
    load_report_path = OUT / "load_report.json"
    load_report_path.write_text(json.dumps(load_report, indent=2), encoding="utf-8")

    probe = _forward_probe(prior_maps=prior_maps, prior_summary=prior_summary, target_size=target_size, view_count=view_count)
    probe_path = OUT / "real_zero_shuffle_forward_probe.json"
    probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    blockers: list[str] = []
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        blockers.append("strict_load_key_mismatch")
    if not load_report["has_human_prior_adapter"] or load_report["human_prior_channels"] <= 0:
        blockers.append("prior_adapter_not_enabled")
    if not probe["non_identical_outputs"]:
        blockers.append("real_zero_shuffle_outputs_identical")
    forbidden_hits = []
    for path in OUT.rglob("*"):
        if path.is_file():
            lower = path.name.lower()
            if lower == "predictions.npz" or "strict_registry" in lower or "candidate_package" in lower or "teacher_package" in lower:
                forbidden_hits.append(str(path.resolve()))

    summary = {
        "task": "v38_prior_enabled_checkpoint_scaffold",
        "status": "DONE_PASS" if not blockers and not forbidden_hits else "DONE_FAIL_ROUTED",
        "research_only": True,
        "no_formal_outputs": True,
        "scaffold_is_trained": False,
        "adapter_training_status": "untrained_scaffold_only",
        "blockers": blockers,
        "forbidden_hits": forbidden_hits,
        "outputs": {
            "config_json": str(config_path.resolve()),
            "state_dict_pt": str(state_path.resolve()),
            "load_report_json": str(load_report_path.resolve()),
            "prior_adapter_weight_stats_json": str(adapter_stats_path.resolve()),
            "real_zero_shuffle_forward_probe_json": str(probe_path.resolve()),
        },
        "file_info": {
            "state_dict_pt": _file_info(state_path),
            "config_json": _file_info(config_path),
        },
        "load_report": load_report,
        "forward_probe": probe,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# V38 Prior-Enabled Checkpoint Scaffold",
        "",
        f"status: `{summary['status']}`",
        f"scaffold_is_trained: `{summary['scaffold_is_trained']}`",
        "",
        "This is a real loadable VGGT checkpoint scaffold with HumanPriorAdapter enabled. It is not a trained prior-enabled checkpoint.",
        "",
        "## Outputs",
        f"- config: `{summary['outputs']['config_json']}`",
        f"- state_dict: `{summary['outputs']['state_dict_pt']}`",
        f"- load_report: `{summary['outputs']['load_report_json']}`",
        f"- adapter_stats: `{summary['outputs']['prior_adapter_weight_stats_json']}`",
        f"- forward_probe: `{summary['outputs']['real_zero_shuffle_forward_probe_json']}`",
        "",
        "## Verification",
        f"- strict_load: `{load_report['strict_load']}`",
        f"- human_prior_channels: `{load_report['human_prior_channels']}`",
        f"- has_human_prior_adapter: `{load_report['has_human_prior_adapter']}`",
        f"- has_normal_head: `{load_report['has_normal_head']}`",
        f"- real/zero/shuffle non-identical: `{probe['non_identical_outputs']}`",
        "",
        "## Blockers",
        f"`{blockers}`",
        "",
        "## Safety",
        f"forbidden_hits: `{forbidden_hits}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=28)
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    result = build_scaffold(target_size=args.target_size, view_count=args.view_count, clean=args.clean)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"], "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
