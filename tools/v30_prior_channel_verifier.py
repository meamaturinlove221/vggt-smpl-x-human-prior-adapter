from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Any

from v30_prior_common import (
    CLOUD_ROOT,
    LOCAL_ROOT,
    REPO_ROOT,
    file_info,
    json_ready,
    prior_case_rows,
    scene_rows,
    scan_forbidden,
    utc_now,
    write_json,
)


REPORT_JSON = REPO_ROOT / "reports/20260508_v30_prior_channel_verifier.json"


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _code_checks() -> dict[str, Any]:
    vggt_py = REPO_ROOT / "vggt/models/vggt.py"
    aggregator_py = REPO_ROOT / "vggt/models/aggregator.py"
    human_prior_py = REPO_ROOT / "vggt/models/human_prior.py"
    vggt_text = _read_text(vggt_py)
    aggregator_text = _read_text(aggregator_py)
    human_prior_text = _read_text(human_prior_py)
    checks = {
        "vggt_file": file_info(vggt_py),
        "aggregator_file": file_info(aggregator_py),
        "human_prior_file": file_info(human_prior_py),
        "human_prior_adapter_class_present": "class HumanPriorAdapter" in human_prior_text,
        "vggt_constructor_has_human_prior_channels": "human_prior_channels=0" in vggt_text,
        "vggt_forward_accepts_prior_maps": "prior_maps: torch.Tensor = None" in vggt_text,
        "vggt_forward_accepts_prior_summary_tokens": "prior_summary_tokens: torch.Tensor = None" in vggt_text,
        "vggt_passes_prior_maps_to_aggregator": "prior_maps=prior_maps" in vggt_text,
        "aggregator_imports_human_prior_adapter": "from vggt.models.human_prior import HumanPriorAdapter" in aggregator_text,
        "aggregator_instantiates_adapter_when_channels_positive": "if human_prior_channels > 0" in aggregator_text,
        "aggregator_raises_when_prior_maps_without_adapter": "prior_maps were provided" in aggregator_text
        and "human_prior_channels=0" in aggregator_text,
        "normal_head_optional_present": "self.normal_head" in vggt_text and "enable_normal" in vggt_text,
    }
    checks["code_supports_prior_adapter"] = all(
        bool(checks[key])
        for key in (
            "human_prior_adapter_class_present",
            "vggt_constructor_has_human_prior_channels",
            "vggt_forward_accepts_prior_maps",
            "vggt_forward_accepts_prior_summary_tokens",
            "vggt_passes_prior_maps_to_aggregator",
            "aggregator_imports_human_prior_adapter",
            "aggregator_instantiates_adapter_when_channels_positive",
        )
    )
    return checks


def _instantiate_probe() -> dict[str, Any]:
    try:
        from vggt.models.vggt import VGGT

        model = VGGT(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            enable_camera=False,
            enable_track=False,
            enable_normal=True,
            human_prior_channels=29,
            human_prior_summary_channels=27,
            human_prior_hidden_dim=16,
        )
        return {
            "ok": True,
            "human_prior_channels": int(model.aggregator.human_prior_channels),
            "human_prior_summary_channels": int(model.aggregator.human_prior_summary_channels),
            "has_human_prior_adapter": model.aggregator.human_prior_adapter is not None,
            "has_depth_head": model.depth_head is not None,
            "has_point_head": model.point_head is not None,
            "has_normal_head": model.normal_head is not None,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _checkpoint_candidates() -> list[dict[str, Any]]:
    roots = [
        REPO_ROOT / "output/surface_research_preflight_local",
        REPO_ROOT / "output/surface_research_cloud_preflight",
        REPO_ROOT / "logs",
        REPO_ROOT / "external_models",
    ]
    suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            lower = path.as_posix().lower()
            likely_vggt = any(token in lower for token in ("vggt", "checkpoint", "ckpt", "prior", "teacher"))
            likely_prior = any(token in lower for token in ("human_prior", "smplx", "v22", "v27", "v31", "prior"))
            excluded_specialist = any(token in lower for token in ("hhand", "hhair", "detail_normal_refiner", "hair", "hand11", "hair4"))
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "likely_vggt": likely_vggt,
                    "likely_prior_enabled": likely_prior and not excluded_specialist,
                    "excluded_specialist_or_non_vggt": excluded_specialist,
                }
            )
    rows.sort(key=lambda item: (bool(item["likely_prior_enabled"]), item["mtime"]), reverse=True)
    return rows[:100]


def _state_dict_probe(candidates: list[dict[str, Any]], max_load: int = 8) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [{"torch_import_ok": False, "error": f"{type(exc).__name__}: {exc}"}]

    loaded = 0
    for row in candidates:
        if not row.get("likely_prior_enabled"):
            continue
        if row.get("size", 0) > 2_000_000_000:
            continue
        path = Path(row["path"])
        probe = {"path": str(path), "loaded": False}
        try:
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
                state = payload["state_dict"]
            elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
                state = payload["model"]
            elif isinstance(payload, dict):
                state = payload
            else:
                state = {}
            keys = [str(key) for key in state.keys()]
            probe.update(
                {
                    "loaded": True,
                    "key_count": len(keys),
                    "has_human_prior_adapter_weights": any("human_prior_adapter" in key for key in keys),
                    "has_vggt_aggregator_weights": any("aggregator" in key for key in keys),
                    "has_depth_head_weights": any("depth_head" in key for key in keys),
                    "has_point_head_weights": any("point_head" in key for key in keys),
                    "has_normal_head_weights": any("normal_head" in key for key in keys),
                    "sample_keys": keys[:20],
                }
            )
        except Exception as exc:  # noqa: BLE001
            probe["error"] = f"{type(exc).__name__}: {exc}"
        probes.append(probe)
        loaded += 1
        if loaded >= max_load:
            break
    return probes


def run_verifier() -> dict[str, Any]:
    code = _code_checks()
    instantiate = _instantiate_probe()
    candidates = _checkpoint_candidates()
    probes = _state_dict_probe(candidates)
    usable = [
        item
        for item in probes
        if item.get("loaded")
        and item.get("has_human_prior_adapter_weights")
        and item.get("has_vggt_aggregator_weights")
        and item.get("has_depth_head_weights")
        and item.get("has_point_head_weights")
    ]
    blockers: list[str] = []
    if not code.get("code_supports_prior_adapter"):
        blockers.append("Local VGGT code does not fully support HumanPriorAdapter prior routing.")
    if not instantiate.get("ok") or not instantiate.get("has_human_prior_adapter"):
        blockers.append("Could not instantiate a prior-enabled VGGT probe locally.")
    if not usable:
        blockers.append(
            "No usable prior-enabled VGGT checkpoint/state_dict with HumanPriorAdapter plus VGGT heads was found under local research roots."
        )

    summary = {
        "task": "v30_prior_channel_verifier",
        "created_utc": utc_now(),
        "status": "DONE_PASS" if code.get("code_supports_prior_adapter") and instantiate.get("ok") else "DONE_FAIL_ROUTED",
        "code_supports_prior_adapter": bool(code.get("code_supports_prior_adapter")),
        "usable_prior_enabled_checkpoint_exists": bool(usable),
        "base_hf_model_allowed_for_key_predictions": False,
        "decision": (
            "Code-level prior route is present, but V30 cannot generate key predictions unless a prior-enabled checkpoint exists."
            if not usable
            else "A usable prior-enabled checkpoint was detected for V30 prediction execution."
        ),
        "code": code,
        "instantiate_probe": instantiate,
        "scene_rows": scene_rows(),
        "prior_case_rows": prior_case_rows(),
        "checkpoint_candidates": candidates,
        "checkpoint_state_dict_probes": probes,
        "usable_prior_enabled_checkpoints": usable,
        "blockers": blockers,
        "forbidden_findings": scan_forbidden(LOCAL_ROOT) + scan_forbidden(CLOUD_ROOT),
    }
    write_json(REPORT_JSON, summary)
    write_json(LOCAL_ROOT / "v30_prior_channel_verifier.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=REPORT_JSON)
    args = parser.parse_args()
    summary = run_verifier()
    if args.json_out != REPORT_JSON:
        write_json(args.json_out, summary)
    print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
