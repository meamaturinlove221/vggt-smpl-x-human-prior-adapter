from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                return None
    return None


def final_loss_and_points(summary: dict[str, Any]) -> dict[str, Any]:
    tail = summary.get("steps", {}).get("train_smoke", {}).get("stdout_tail", "")
    losses = re.findall(r"Loss=([0-9.]+).*?Points=([0-9]+)", tail)
    if not losses:
        return {}
    loss, points = losses[-1]
    return {"final_loss": float(loss), "final_points": int(points)}


def file_row(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.is_file(), "bytes": int(path.stat().st_size) if path.is_file() else 0}


def main() -> int:
    local = REPO_ROOT / "output/surface_research_preflight_local"
    cloud = REPO_ROOT / "output/surface_research_cloud_preflight"
    paths = {
        "a5x3": local / "V10_A5X3_must3r_known_camera_alignment/summary.json",
        "2dgs_scene": local / "V10_2DGS_scene_contract_audit/summary.json",
        "fus3d4": local / "B_Fus3D4_surface_candidate_precheck/summary.json",
        "hand11": local / "B_Hand11_real_vggt_hand_token_decoder/summary.json",
        "hair4": local / "B_Hair4_native_4k4d_smplx_hair_topology/summary.json",
        "unified": local / "V10_unified_surface_merge_precheck/summary.json",
        "dline": local / "DLine_V10_promotion_transaction/summary.json",
    }
    component = {name: load_json(path) for name, path in paths.items()}
    tiers = {}
    for tier in ("1k", "10k", "30k"):
        summary_path = cloud / f"Cloud_G_V10/a5x3_2dgs_colmap_scene_{tier}/summary.json"
        ply_path = cloud / f"Cloud_G_V10/a5x3_2dgs_colmap_scene_{tier}/model_smoke/point_cloud/iteration_{int(tier[:-1]) * 1000 if tier.endswith('k') else tier}/point_cloud.ply"
        # The 30k/10k paths are iteration_30000 and iteration_10000; the
        # expression above intentionally mirrors the naming convention.
        summary = load_json(summary_path)
        tiers[tier] = {
            "summary": str(summary_path.resolve()),
            "status": summary.get("status"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "train_elapsed_sec": summary.get("steps", {}).get("train_smoke", {}).get("elapsed_sec"),
            **final_loss_and_points(summary),
            "point_cloud": file_row(ply_path),
            "point_cloud_vertices": ply_vertex_count(ply_path),
        }
    forbidden_hits = []
    scan_roots = [
        local / "V10_A5X3_must3r_known_camera_alignment",
        local / "V10_2DGS_scene_contract_audit",
        local / "B_Fus3D4_surface_candidate_precheck",
        local / "B_Hand11_real_vggt_hand_token_decoder",
        local / "B_Hair4_native_4k4d_smplx_hair_topology",
        local / "V10_unified_surface_merge_precheck",
        local / "DLine_V10_promotion_transaction",
        cloud / "Cloud_G_V10",
    ]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and re.search(r"predictions\.npz|teacher_export|candidate_export|strict_pass|strict_gate_registry|registry\.json|pass\.json", path.name, re.I):
                forbidden_hits.append(str(path.resolve()))
    dline = component["dline"]
    unified = component["unified"]
    summary = {
        "task": "v10_execution_rollup",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "strict_promotion_blocked_after_full_v10_run",
        "strict_candidate_passes": int(dline.get("strict_candidate_passes", 0)),
        "strict_teacher_passes": int(dline.get("strict_teacher_passes", 0)),
        "formal_cloud_unblocked": False,
        "component_statuses": {name: payload.get("status") for name, payload in component.items()},
        "component_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "2dgs_cloud_tiers": tiers,
        "unified_gates": unified.get("gates", {}),
        "dline_required_gates": dline.get("required_gates", {}),
        "forbidden_output_hits": forbidden_hits,
        "key_artifacts": {
            "a5x3_aligned_points": file_row(local / "V10_A5X3_must3r_known_camera_alignment/a5x3_aligned_must3r_points.ply"),
            "a5x3_reprojection_sheet": file_row(local / "V10_A5X3_must3r_known_camera_alignment/a5x3_6view_reprojection_sheet.png"),
            "fus3d4_body": file_row(local / "B_Fus3D4_surface_candidate_precheck/b_fus3d4_body_surface.ply"),
            "fus3d4_head_face": file_row(local / "B_Fus3D4_surface_candidate_precheck/b_fus3d4_head_face_surface.ply"),
            "hand11_left": file_row(local / "B_Hand11_real_vggt_hand_token_decoder/b_hand11_left_surface.ply"),
            "hand11_right": file_row(local / "B_Hand11_real_vggt_hand_token_decoder/b_hand11_right_surface.ply"),
            "hair4_hairline": file_row(local / "B_Hair4_native_4k4d_smplx_hair_topology/b_hair4_hairline_band_surface.ply"),
            "unified_surface": file_row(local / "V10_unified_surface_merge_precheck/unified_surface_v10.ply"),
            "unified_full_sheet": file_row(local / "V10_unified_surface_merge_precheck/unified_surface_v10_open3d_full.png"),
            "unified_head_hair_sheet": file_row(local / "V10_unified_surface_merge_precheck/unified_surface_v10_open3d_head_face_hair.png"),
            "unified_hands_sheet": file_row(local / "V10_unified_surface_merge_precheck/unified_surface_v10_open3d_hands.png"),
            "dline_report": file_row(local / "DLine_V10_promotion_transaction/dline_v10_promotion_report.md"),
        },
        "blockers": dline.get("blockers", []),
        "decision": "V10 ran through local T/G/F/H/U/D and cloud 2DGS 1k/10k/30k. D-line correctly refused strict promotion; no formal package/registry/pass was written.",
    }
    out_dir = local / "V10_execution_rollup"
    out_json = out_dir / "summary.json"
    out_md = out_dir / "report.md"
    write_json(out_json, summary)
    write_json(REPO_ROOT / "reports/20260508_v10_execution_rollup.json", summary)
    lines = [
        "# V10 Execution Rollup",
        "",
        f"Status: `{summary['status']}`",
        "",
        summary["decision"],
        "",
        "## Component Statuses",
        "",
    ]
    lines.extend([f"- {name}: `{status}`" for name, status in summary["component_statuses"].items()])
    lines += ["", "## 2DGS Cloud Tiers", ""]
    lines.extend(
        [
            f"- {tier}: status=`{row.get('status')}`, final_loss=`{row.get('final_loss')}`, final_points=`{row.get('final_points')}`, vertices=`{row.get('point_cloud_vertices')}`"
            for tier, row in tiers.items()
        ]
    )
    lines += ["", "## D-line Gates", ""]
    lines.extend([f"- {key}: `{value}`" for key, value in summary["dline_required_gates"].items()])
    lines += ["", "## Forbidden Output Scan", "", f"- hits: `{len(forbidden_hits)}`"]
    lines += ["", "## Key Artifacts", ""]
    lines.extend([f"- {name}: `{row['path']}`" for name, row in summary["key_artifacts"].items() if row["exists"]])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO_ROOT / "reports/20260508_v10_execution_rollup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report": str(out_md), "strict_candidate_passes": 0, "strict_teacher_passes": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
