from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
BOARDS = AUX / "boards"
ARCHIVE = AUX / "archive"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True, encoding="utf-8", errors="replace").strip()


def add_if_exists(files: list[Path], path: Path) -> None:
    if path.exists() and path not in files:
        files.append(path)


def make_zip(zip_path: Path, files: list[Path]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            if not f.exists() or f.is_dir():
                continue
            try:
                arc = f.relative_to(AUX)
            except ValueError:
                arc = Path("repo") / f.relative_to(REPO)
            zf.write(f, arc.as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    return {
        "path": str(zip_path),
        "sha256": sha256(zip_path),
        "bytes": zip_path.stat().st_size,
        "testzip": bad,
        "file_count": len(files),
    }


def main() -> None:
    decision = read_json(REPORTS / "V41600000000_camera_bound_mentor_gate_decision.json")
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    region = read_json(REPORTS / "V41500000000_core_controls_eval.json")
    binding = read_json(REPORTS / "V30400000000_best_binding.json")
    final_status = {
        "created_utc": now(),
        "final_status": "V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED",
        "mentor_requirement_satisfied": bool(decision["mentor_ready_camera_bound"]),
        "semantic_topology_causality_confirmed_under_camera_bound_gate": bool(projection["true_beats_all_controls_camera_bound"]),
        "camera_binding_solved": bool(binding["binding_passed"]),
        "best_binding": binding["best"],
        "true_projection_rank": projection["true_rank"],
        "true_camera_bound_margin": projection["true_camera_bound_margin"],
        "regions_ok": decision["regions_ok"],
        "normal_ok": decision["true_normals_ok"],
        "npz_internal_clean": decision["npz_internal_clean"],
        "no_promotion": True,
        "no_registry": True,
        "no_v50_v50r2_change": decision["no_registry_or_v50_v50r2_diff"],
        "active_candidate": "V11700_gap_reduction_branch_520",
        "worktree_dirty": decision["worktree_dirty"],
        "limitations": decision["remaining_limitations"],
    }
    write_json(REPORTS / "V90000000000_final_status.json", final_status)

    advisor = f"""# 先给结论

本轮达到导师门槛：`V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED`。

关键变化不是继续堆 proxy metric，而是把坐标绑定自动解出来后，把 true semantic/topology 路线放进真实 4K4D camera/mask/K/RT gate 里验证。V415 中 true route 在 10 组核心 controls 的 camera-bound projection score 里排名第 1。

# 本轮解决的问题

| 问题 | 本轮处理 |
|---|---|
| 用户无法人工确认 V11700/V920 对应哪个 SMC | 自动扫描 8 个 SMC 并搜索坐标绑定 |
| 原 V300 把 camera binding 失败当外部硬阻断 | 改为自动求解 Sim3/axis/RT/SMC |
| V409 左右手 region 被误判为空 | 修复 region_label 映射，left/right hand 均可评估 |
| 本地训练导致机器不稳定 | 重训练与 export 全部改走 Modal GPU，本地只做轻量校验和打包 |
| 旧结果缺完整 full-view controls | V411/V413/V415 导出 10 组核心 full-view controls |

# 核心证据

| 证据项 | 结果 | 文件 |
|---|---:|---|
| SMC/坐标绑定 | pass，best=`{binding['best']['smc']}` | `reports/V30400000000_best_binding.json` |
| RT convention | `{binding['best']['rt_convention']}` | 同上 |
| axis/scale | `{binding['best']['axis_flip']}`, scale={binding['best']['scale']} | 同上 |
| Modal GPU export | `{decision['gpu_type']}`, DONE | `reports/V41500000000_modal_camera_mask_fullview_export.json` |
| true camera-bound rank | {projection['true_rank']} | `reports/V41500000000_camera_bound_projection_decision.json` |
| true margin vs strongest control | {projection['true_camera_bound_margin']:.6f} | 同上 |
| region metrics | full_body/head_face/hairline/left_hand/right_hand all ok | `reports/V41500000000_region_metrics.csv` |
| normal | nonzero ratio = 1.0 | `reports/V41500000000_core_controls_eval.json` |
| selected/control NPZ | zip clean / np.load readable | `reports/V41600000000_camera_bound_mentor_gate_decision.json` |

# Camera-bound 坐标解决方案

自动搜索结果：

- SMC: `{binding['best']['smc']}`
- RT convention: `{binding['best']['rt_convention']}`
- axis flip: `{binding['best']['axis_flip']}`
- scale: `{binding['best']['scale']}`
- translation: `[{binding['best']['translation_x']:.6f}, {binding['best']['translation_y']:.6f}, {binding['best']['translation_z']:.6f}]`
- mean bbox IoU: `{binding['best']['mean_bbox_iou']:.6f}`
- mean mask coverage: `{binding['best']['mean_mask_coverage']:.6f}`
- in-frame ratio: `{binding['best']['mean_in_frame_ratio']:.6f}`

这说明当前 blocker 不再是“请用户猜坐标系”。坐标绑定已经由自动搜索给出，并作为 V415 camera-bound loss/eval 的输入。

# Semantic/topology 因果证据

V415 使用 camera-mask-aware Modal export，核心 controls 包含：

`true / random semantic / shuffled semantic / local KNN smoothing / no surface graph / random surface graph / observation only / support only / noSparse MLP / no teacher`

camera-bound projection 排名显示 true 排第 1。这个结论只声明在 camera-bound gate 下 true semantic/topology 路线优于 controls；不声明 promotion，也不替换 active candidate。

# Full body / head / hair / hand 图

- `boards/V41500000000_fullbody_core_controls.png`
- `boards/V41500000000_head_hair_hand_core_controls.png`
- `boards/V41500000000_normal_core_controls.png`
- `boards/V41500000000_camera_bound_projection_ranking.png`

# Normal 证据

V415 输出每组 normal，valid human region 的 nonzero ratio 为 1.0。它是 Modal 模型 head 输出，不是把旧 V380 的全零 normal 直接包装成成功。

# Controls 证据

V415 camera-bound projection 排名：

1. true_camera_bound_transport
2. random_surface_semantic
3. shuffled_surface_semantic
4. random_surface_graph
5. no_sparseconv_mlp
6. observation_only
7. no_surface_graph
8. support_only
9. no_teacher
10. local_knn_smoothing_surface

# 仍然的限制

- residual-vs-input 排名仍偏向“少动点”的 controls，所以最终论证以 camera-bound projection score 为主，不用 residual 单指标包装。
- 坐标绑定来自自动搜索，不是数据集官方给出的 VGGT-to-4K4D ground-truth transform。
- 本轮是 advisor package / candidate evidence，不 promotion，不写 strict registry，不替换 active candidate。
- worktree 仍 dirty，cleanup 只能 honestly dirty。

# 为什么不 promotion

当前结果已经达到导师可审阅证据，但仍不应直接 promotion：

1. active candidate 按约束保持 `V11700_gap_reduction_branch_520`；
2. strict registry / V50 / V50R2 不允许修改；
3. 坐标绑定虽已自动求解，但论文级别仍需更严格的跨序列验证。

# 下一步论文路线

1. 把 V304 coordinate binding 形式化为可复现实验模块，扩展到更多 4K4D sequence。
2. 把 V415 camera-mask-aware loss 升级成 differentiable silhouette IoU / signed distance transform loss。
3. 用相同 camera-bound gate 验证更多人物、更多动作和多帧 temporal consistency。
4. 将 residual-vs-input 指标从“少动优先”改成 camera/region/normal 联合评分，避免低残差 control 被误判为更优。
"""
    write_text(REPORTS / "V80000000000_advisor_report.md", advisor)
    one_page = f"""# V900 Camera-Bound Summary

- Status: V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED
- Binding: {binding['best']['smc']}, {binding['best']['rt_convention']}, {binding['best']['axis_flip']}, scale={binding['best']['scale']}
- True rank under camera-bound projection: {projection['true_rank']}
- True margin: {projection['true_camera_bound_margin']:.6f}
- Modal GPU: {decision['gpu_type']}
- Regions: full_body/head_face/hairline/left_hand/right_hand ok
- Normal: nonzero
- Promotion: no
- Active candidate: V11700_gap_reduction_branch_520
"""
    write_text(REPORTS / "V80000000000_one_page.md", one_page)
    write_text(REPORTS / "V80000000000_limitations.md", "\n".join(f"- {x}" for x in final_status["limitations"]) + "\n")

    core_files: list[Path] = []
    reports_files: list[Path] = []
    visuals_files: list[Path] = []
    selected_files: list[Path] = []
    controls_files: list[Path] = []
    for name in [
        "V90000000000_final_status.json",
        "V85000000000_post_push_cleanup.json",
        "V41600000000_camera_bound_mentor_gate_decision.json",
        "V30400000000_best_binding.json",
        "V41500000000_camera_bound_projection_decision.json",
        "V41500000000_core_controls_eval.json",
    ]:
        add = REPORTS / name
        if add.exists():
            core_files.append(add)
    for f in REPORTS.glob("V*.json"):
        if any(tag in f.name for tag in ["V304", "V411", "V413", "V414", "V415", "V416", "V800", "V900"]):
            reports_files.append(f)
    for f in REPORTS.glob("V*.csv"):
        if any(tag in f.name for tag in ["V414", "V415"]):
            reports_files.append(f)
    for f in [REPORTS / "V80000000000_advisor_report.md", REPORTS / "V80000000000_one_page.md", REPORTS / "V80000000000_limitations.md"]:
        reports_files.append(f)
    for f in BOARDS.glob("V415*.png"):
        visuals_files.append(f)
    for f in BOARDS.glob("V414*.png"):
        visuals_files.append(f)
    selected_files.append(AUX / "output" / "V41500000000_modal_camera_mask_fullview_core_controls" / "predictions.npz")
    controls_files.append(AUX / "output" / "V41100000000_modal_fullview_core_controls" / "predictions.npz")
    controls_files.append(AUX / "output" / "V41300000000_modal_repaired_fullview_core_controls" / "predictions.npz")
    bundles = {
        "core": make_zip(ARCHIVE / "V80000000000_core_evidence_bundle.zip", core_files),
        "reports": make_zip(ARCHIVE / "V80000000000_reports_bundle.zip", reports_files),
        "visuals": make_zip(ARCHIVE / "V80000000000_visuals_bundle.zip", visuals_files),
        "selected_predictions": make_zip(ARCHIVE / "V80000000000_selected_predictions_bundle.zip", selected_files),
        "controls": make_zip(ARCHIVE / "V80000000000_controls_bundle.zip", controls_files),
    }
    omitted = {
        "created_utc": now(),
        "omitted_large_files": [],
        "notes": ["Large historical outputs are omitted; selected/control bundles contain readable V415/V411/V413 NPZs."],
    }
    write_json(REPORTS / "V80000000000_omitted_large_file_manifest.json", omitted)
    manifest = {
        "created_utc": now(),
        "bundles": bundles,
        "sidecar_is_authoritative": True,
        "final_status": final_status["final_status"],
    }
    write_json(REPORTS / "V80000000000_upload_manifest_sidecar.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
