from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"
BOARDS = REPO / "boards"

AUDIT = REPORTS / "V10170000000000000000_flatness_geometry_audit.csv"
DECISION = REPORTS / "V10170000000000000000_flatness_and_depth_render_decision.json"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    created_at = now()
    rows = read_csv(AUDIT)
    by_label = {r["label"]: r for r in rows}
    baseline = by_label["baseline"]
    candidate = by_label["candidate"]
    candidate_thickness = as_float(candidate, "human_pca_thickness_ratio")
    baseline_thickness = as_float(baseline, "human_pca_thickness_ratio")
    thickness_gain = candidate_thickness - baseline_thickness
    candidate_z = as_float(candidate, "human_bbox_z")
    baseline_z = as_float(baseline, "human_bbox_z")
    z_gain = candidate_z - baseline_z
    render_board = BOARDS / "V10170000000000000000_0012_11_frame001_oblique_depth_pointcloud_audit.png"

    status = "V10180_RENDER_REPAIR_AUXILIARY_AND_GEOMETRY_REPRESENTATION_REPAIR_REQUIRED"
    goal_path = GOALS / "V10180000000000000000_auto_evolved_depth_cue_or_geometry_repair_route.md"
    write_text(
        goal_path,
        f"""# V10180 Auto-Evolved Depth-Cue Plus Geometry Repair Route

Created: {created_at}

The user's visual concern is valid: the V10150 board looked too 2D and lacked point-cloud depth feel.

Evidence:
- V10150 renderer used a direct `points[:, :2]` projection, so the original board was effectively 2D/orthographic.
- V10170 generated an oblique depth-shaded board: `{render_board}`.
- Candidate human PCA thickness ratio: {candidate_thickness:.6f}.
- Baseline human PCA thickness ratio: {baseline_thickness:.6f}.
- Candidate z-range gain over baseline: {z_gain:.6f}.

Decision:
- Rendering must be repaired for all future mentor boards: oblique view, depth shading, side/local 3D views, same scene/same bounds.
- Rendering repair is auxiliary only. It cannot turn this candidate into mentor-ready.
- The candidate does not provide meaningfully stronger 3D morphology than the VGGT baseline, so route back to representation/geometry repair.

Next representation route:
1. Preserve real VGGT baseline high-confidence RGB/detail and real environment points.
2. Add canonical SMPL-X surfel/graph support only in visible weak regions.
3. Optimize explicit 3D thickness/side-view/limb-continuity objectives, not just front-view projection.
4. Produce human-main full-scene RGB point cloud plus same-scene controls.
5. Use projection only as auxiliary.
6. Face detail remains not applicable; allowed claim: {ALLOWED_FACE_CLAIM}.

Forbidden final claims:
- mentor-ready
- projection-only pass
- render-only pass
- facial detail improved
- metric-only pass
- route exhausted
""",
    )

    report = f"""# V10180 Depth-Cue And Geometry Repair Decision

结论：用户指出的二维化问题成立。

## 渲染问题

V10150 的主图渲染只使用 `points[:, :2]`，再按 `z` 排序，没有斜视透视、深度着色或侧向厚度提示。因此原图天然会像二维贴片，不能作为最终导师视觉板。

## 几何问题

V10170 的斜视深度板已经补了深度线索：

`{render_board}`

但候选本身没有比 baseline 明显更 3D：

- baseline PCA thickness ratio: `{baseline_thickness:.6f}`
- candidate PCA thickness ratio: `{candidate_thickness:.6f}`
- thickness gain: `{thickness_gain:.6f}`
- baseline z range: `{baseline_z:.6f}`
- candidate z range: `{candidate_z:.6f}`
- z range gain: `{z_gain:.6f}`

所以不能只继续调 viewer/截图角度。渲染需要修，但主路线必须回到表示和几何重构。

## 下一步

进入 V10180/V10190：把所有导师板换成 oblique/depth-cued 3D 渲染作为检查工具，同时重构 candidate 的 3D morphology：canonical SMPL-X surfel/graph 支撑可见弱区，增加 thickness / side-view / limb-continuity 目标，保留真实 VGGT 环境。Projection 仍然只能是辅助。

Face detail 仍不适用；只允许写 `{ALLOWED_FACE_CLAIM}`。
"""
    report_path = REPORTS / "V10180000000000000000_depth_cue_and_geometry_repair_decision.md"
    write_text(report_path, report)

    payload = {
        "created_at": created_at,
        "status": status,
        "user_observation_valid": True,
        "render_repair_needed": True,
        "geometry_repair_needed": True,
        "render_only_repair_allowed_as_final": False,
        "baseline_human_pca_thickness_ratio": baseline_thickness,
        "candidate_human_pca_thickness_ratio": candidate_thickness,
        "candidate_thickness_gain": thickness_gain,
        "baseline_human_z_range": baseline_z,
        "candidate_human_z_range": candidate_z,
        "candidate_z_range_gain": z_gain,
        "oblique_depth_board": str(render_board),
        "audit_csv": str(AUDIT),
        "report": str(report_path),
        "next_goal": str(goal_path),
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "next_action": "Continue, but route back to geometry/representation repair rather than only adjusting render/viewer.",
    }
    out = REPORTS / "V10180000000000000000_depth_cue_and_geometry_repair_decision.json"
    write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
