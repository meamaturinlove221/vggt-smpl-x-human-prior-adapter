from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True, encoding="utf-8", errors="replace").strip()


def main() -> None:
    binding = read_json(REPORTS / "V30400000000_best_binding.json")
    modal_export = read_json(REPORTS / "V41500000000_modal_camera_mask_fullview_export.json")
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    region_eval = read_json(REPORTS / "V41500000000_core_controls_eval.json")
    region_rows = read_csv(REPORTS / "V41500000000_region_metrics.csv")
    required_regions = {"full_body", "head_face", "hairline", "left_hand", "right_hand"}
    true_region_rows = [r for r in region_rows if r.get("group") == "true_camera_bound_transport"]
    true_regions_ok = required_regions.issubset({r["region"] for r in true_region_rows if r.get("status") == "ok"})
    true_normals_ok = all(float(r.get("normal_nonzero_ratio", "0") or 0) > 0.99 for r in true_region_rows)
    groups = {r["group"] for r in projection["ranked_groups"]}
    required_groups = {
        "true_camera_bound_transport",
        "random_surface_semantic",
        "shuffled_surface_semantic",
        "local_knn_smoothing_surface",
        "no_surface_graph",
        "random_surface_graph",
        "observation_only",
        "support_only",
        "no_sparseconv_mlp",
        "no_teacher",
    }
    status_short = git(["status", "--short"])
    changed_v50 = bool(git(["diff", "--name-only", "--", "V50", "V50R2", "strict_registry", "registry"]))
    mentor_ready = bool(
        binding.get("binding_passed")
        and modal_export.get("status") == "DONE_FULLVIEW_EXPORT"
        and modal_export.get("cuda_available")
        and projection.get("true_beats_all_controls_camera_bound")
        and projection.get("true_rank") == 1
        and set(modal_export.get("groups", [])) >= required_groups
        and groups >= required_groups
        and region_eval.get("regions_ok")
        and true_regions_ok
        and true_normals_ok
        and region_eval.get("npz_testzip") is None
        and projection.get("npz_testzip") is None
        and not changed_v50
    )
    payload = {
        "created_utc": now(),
        "final_gate": "V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED" if mentor_ready else "CONTINUE_AUTO_EVOLUTION",
        "mentor_ready_camera_bound": mentor_ready,
        "binding_passed": bool(binding.get("binding_passed")),
        "binding_best": binding.get("best"),
        "modal_export_status": modal_export.get("status"),
        "gpu_type": modal_export.get("gpu_type"),
        "core_groups_complete": set(modal_export.get("groups", [])) >= required_groups,
        "projection_true_rank": projection.get("true_rank"),
        "projection_true_margin": projection.get("true_camera_bound_margin"),
        "projection_true_beats_all_controls": projection.get("true_beats_all_controls_camera_bound"),
        "regions_ok": bool(region_eval.get("regions_ok") and true_regions_ok),
        "true_regions_ok": true_regions_ok,
        "true_normals_ok": true_normals_ok,
        "npz_internal_clean": bool(region_eval.get("npz_testzip") is None and projection.get("npz_testzip") is None),
        "no_promotion": True,
        "no_registry_or_v50_v50r2_diff": not changed_v50,
        "active_candidate_replaced": False,
        "worktree_dirty": bool(status_short),
        "worktree_dirty_summary": status_short.splitlines()[:80],
        "remaining_limitations": [
            "Residual-vs-input ranking still favors low-motion controls, so the claim is camera-bound semantic transport, not unrestricted residual superiority.",
            "The coordinate binding is solved by automatic search to SMC 0021_03, not by user-provided ground-truth transform.",
            "Active candidate remains V11700; this is advisor package evidence, not promotion.",
        ],
    }
    write_json(REPORTS / "V41600000000_camera_bound_mentor_gate_decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
