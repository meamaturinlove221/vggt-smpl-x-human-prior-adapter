"""Guard old residual-composer routes from becoming V100+ main candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_MAIN_ROUTES = {
    "V770_world_points_plus_sparse_delta_direct_composer",
    "V999_teacher_residual_composer",
    "V129_guarded_mix",
    "HumanRAM_blend_postcompose",
    "support_only_direct_residual",
    "observation_only_direct_residual",
    "no_teacher_residual_without_semantic_bottleneck",
    "old_residual_composer",
    "final_guarded_mix",
}

ALLOWED_DIAGNOSTIC_USAGE = {"baseline", "ablation", "diagnostic", "failure_comparison"}


def check_route_config(config: dict[str, Any]) -> dict[str, Any]:
    route_name = str(config.get("route_name") or config.get("route") or "")
    usage = str(config.get("usage", "main")).lower()
    uses_teacher_postcompose = bool(config.get("uses_teacher_postcompose", False))
    uses_v999_residual_copy = bool(config.get("uses_v999_residual_copy", False))
    support_outputs_residual = bool(config.get("support_outputs_residual", False))
    observation_outputs_residual = bool(config.get("observation_outputs_residual", False))
    violations: list[str] = []

    if usage not in ALLOWED_DIAGNOSTIC_USAGE and route_name in FORBIDDEN_MAIN_ROUTES:
        violations.append(f"forbidden_main_route:{route_name}")
    if uses_teacher_postcompose:
        violations.append("teacher_postcompose_enabled")
    if uses_v999_residual_copy:
        violations.append("v999_residual_copy_enabled")
    if support_outputs_residual:
        violations.append("support_branch_direct_residual_enabled")
    if observation_outputs_residual and usage == "main":
        violations.append("observation_branch_direct_residual_enabled")

    return {
        "route_name": route_name,
        "usage": usage,
        "allowed": not violations,
        "violations": violations,
        "allowed_diagnostic_usage": sorted(ALLOWED_DIAGNOSTIC_USAGE),
        "forbidden_main_routes": sorted(FORBIDDEN_MAIN_ROUTES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = check_route_config(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
