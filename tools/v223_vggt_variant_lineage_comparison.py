from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output" / "mentor_report_v50r2" / "vggt_variant_lineage_comparison"
MD_REPORT = REPORTS / "20260509_v50r2_vggt_variant_lineage_comparison.md"
MD_MENTOR_REPORT = REPORTS / "20260509_v50r2_vggt_variant_lineage_comparison_mentor.md"
JSON_REPORT = REPORTS / "20260509_v50r2_vggt_variant_lineage_comparison.json"
CSV_REPORT = REPORTS / "20260509_v50r2_vggt_variant_lineage_comparison.csv"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def method_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "method": "base_full_6v_vggt_preprocess_full",
            "family": "VGGT baseline / original full image",
            "protocol": "0012_11 frame0000, original 6-view sparse protocol",
            "source": "reports/20260422_mentor_taskboard_status.md",
            "artifact_status": "historical_report_only",
            "full_points": 40882,
            "head_points": 8994,
            "face_points": 4177,
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "",
            "normal_angle_vs_full": "",
            "translation_l2_vs_full": "",
            "visual_result": "baseline weak; head/face ROI sparse",
            "mentor_reading": "This is the baseline VGGT point-cloud quality the mentor asked to compare against.",
            "current_file_evidence": "Original predictions path is no longer present locally; numeric evidence is preserved in report.",
        },
        {
            "method": "human_crop_6v_vggt",
            "family": "VGGT + human crop input preprocessing",
            "protocol": "same 6-view case, crop preprocessing",
            "source": "reports/20260422_mentor_taskboard_status.md; reports/20260422_humancrop_mainline_bootstrap.md",
            "artifact_status": "historical_report_only",
            "full_points": 111094,
            "head_points": 24441,
            "face_points": 11523,
            "depth_mae_vs_full": 0.0400,
            "world_l2_vs_full": 0.0453,
            "normal_angle_vs_full": 0.3762,
            "translation_l2_vs_full": 0.0554,
            "visual_result": "large occupancy gain; not sufficient as final head/face quality",
            "mentor_reading": "Crop is useful as an input route because the person occupies more pixels, matching the mentor's recording. It improves retained points, but does not by itself prove clearer 3D face detail.",
            "current_file_evidence": "Original predictions path is no longer present locally; report and scene path references remain.",
        },
        {
            "method": "human_crop_hardmask_6v_vggt",
            "family": "VGGT + hard human mask/crop",
            "protocol": "same 6-view case, hard-mask crop preprocessing",
            "source": "reports/20260422_mentor_taskboard_status.md",
            "artifact_status": "historical_report_only",
            "full_points": 111078,
            "head_points": 24437,
            "face_points": 10712,
            "depth_mae_vs_full": 0.0481,
            "world_l2_vs_full": 0.0504,
            "normal_angle_vs_full": 0.4963,
            "translation_l2_vs_full": 0.0966,
            "visual_result": "occupancy improves, global geometry shifts more than plain crop",
            "mentor_reading": "Hard-mask crop is not the best default because it disturbs geometry more than plain human crop.",
            "current_file_evidence": "Original predictions path is no longer present locally; numeric evidence is preserved in report.",
        },
        {
            "method": "human_crop_softmatte_6v_vggt",
            "family": "VGGT + soft matte crop",
            "protocol": "same 6-view case, soft-matte crop preprocessing",
            "source": "reports/20260422_mentor_taskboard_status.md",
            "artifact_status": "historical_report_only",
            "full_points": 151734,
            "head_points": 33382,
            "face_points": 15127,
            "depth_mae_vs_full": 0.0586,
            "world_l2_vs_full": 0.0480,
            "normal_angle_vs_full": 0.4832,
            "translation_l2_vs_full": 0.1160,
            "visual_result": "densest ROI, but less stable than plain crop",
            "mentor_reading": "Soft matte raises point count most, but the project should not claim quality from point count alone.",
            "current_file_evidence": "Original predictions path is no longer present locally; numeric evidence is preserved in report.",
        },
        {
            "method": "normal_r16_xview_selfgeom",
            "family": "VGGT + normal/depth/point self-geometry",
            "protocol": "6-view headshoulder, same-protocol signfix comparison",
            "source": "reports/20260428_normal_r16_xview_selfgeom_eval_on6v_headshoulder.md",
            "artifact_status": "historical_report_only",
            "full_points": 184213,
            "head_points": 40527,
            "face_points": 14981,
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "",
            "normal_angle_vs_full": "",
            "translation_l2_vs_full": "",
            "visual_result": "normal consistency partly improves, but face ROI is below signfix and Open3D remains shell-like",
            "mentor_reading": "This directly answers the mentor's normal/depth coupling suggestion: the coupling was implemented and helped some consistency metrics, but did not translate into a modeled face/head point cloud.",
            "current_file_evidence": "Original predictions path is no longer present locally; preview directory is empty after cleanup.",
        },
        {
            "method": "r32_selfgeom_crop_weakprior",
            "family": "VGGT + crop + weak SMPL-X prior + self-geometry",
            "protocol": "6-view full/headshoulder/headface matrix",
            "source": "reports/20260503_selfgeom_crop_weakprior_pivot_status.md",
            "artifact_status": "historical_report_only",
            "full_points": "",
            "head_points": "",
            "face_points": "",
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "",
            "normal_angle_vs_full": "",
            "translation_l2_vs_full": "",
            "visual_result": "negative strict visual review; shell/ghost head-face and fragmented hands",
            "mentor_reading": "Combining crop and self-geometry was a reasonable VGGT line, but it still failed the visual gate.",
            "current_file_evidence": "Local inference directories remain, but no predictions.npz is present in this checkout.",
        },
        {
            "method": "v25_base_vggt_research_prediction",
            "family": "VGGT base model research prediction",
            "protocol": "frame0000 first six V42/V50R2 views",
            "source": "reports/20260509_v50r2_vggt_vertical_baseline_comparison.md",
            "artifact_status": "recomputed_current",
            "full_points": "mean valid 11575.5 per view in V50R2 comparison",
            "head_points": "mean valid 4019.17 per view",
            "face_points": "included in head_face region",
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "baseline reference",
            "normal_angle_vs_full": "normal not available",
            "translation_l2_vs_full": "",
            "visual_result": "basic full-body point-map output exists; no normal route",
            "mentor_reading": "This is the current same-protocol base VGGT comparison used for V50R2.",
            "current_file_evidence": "output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_points_world.npz",
        },
        {
            "method": "v42_prior_enabled_vggt",
            "family": "VGGT + SMPL-X prior-enabled prediction",
            "protocol": "frame0000 first six V42/V50R2 views",
            "source": "reports/20260509_v50r2_vggt_vertical_baseline_comparison.md",
            "artifact_status": "recomputed_current",
            "full_points": "mean valid 11575.5 per view",
            "head_points": "mean valid 4019.17 per view",
            "face_points": "included in head_face region",
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "mean L2 vs V25 all pixels = 0.00053544",
            "normal_angle_vs_full": "geometric normal available",
            "translation_l2_vs_full": "",
            "visual_result": "normal evidence available; main point-map coordinate change from V25 is small",
            "mentor_reading": "This is the true prior-enabled VGGT path; the honest conclusion is that point-map coordinates changed little.",
            "current_file_evidence": "output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_points_world.npz",
        },
        {
            "method": "v50r2_candidate",
            "family": "VGGT candidate package with SMPL-X native evidence",
            "protocol": "V50R2 rebuilt/frozen candidate",
            "source": "reports/20260509_v50r2_vggt_vertical_baseline_comparison.md; reports/V399_v50r2_final_promotion_controller.md",
            "artifact_status": "recomputed_current",
            "full_points": "same main point map as V42",
            "head_points": "tighter packaged head/face evidence, mean valid 2450.83 per view in comparison",
            "face_points": "packaged region evidence, not a larger raw point count",
            "depth_mae_vs_full": "",
            "world_l2_vs_full": "V50R2 candidate points vs V42 max abs = 0.0",
            "normal_angle_vs_full": "candidate geometric normals available",
            "translation_l2_vs_full": "",
            "visual_result": "formal candidate closure and region evidence; not a new visibly sharper full-body point field",
            "mentor_reading": "V50R2 is defensible as a candidate package, but detail improvement over base VGGT must be described cautiously.",
            "current_file_evidence": "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files",
        },
    ]
    return rows


def current_file_audit() -> list[dict[str, Any]]:
    paths = [
        "output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_points_world.npz",
        "output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_points_world.npz",
        "output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__candidate_points.npz",
        "output/modal_results/20260421_6views_preprocess_full_b40/predictions.npz",
        "output/modal_results/20260421_6views_preprocess_crop_b40/predictions.npz",
        "output/modal_results/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder/predictions.npz",
        "output/local_inference_results/r32_confstable_geomonly1_on6v_fullbody/predictions.npz",
    ]
    audited = []
    for rel in paths:
        path = ROOT / rel
        audited.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)) if path.exists() else None,
            }
        )
    return audited


def write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    lines = [
        "# V50R2 VGGT-derived point-cloud lineage comparison",
        "",
        f"- Created UTC: `{now()}`",
        "- Scope: only VGGT-derived point-cloud routes are compared in the main table.",
        "- Non-VGGT teacher/sensor routes such as Kinect, COLMAP/MVS, 2DGS, PSHuman, DepthPro, and Sapiens are treated as reference or failure-route evidence, not as VGGT baseline variants.",
        "- Important evidence boundary: several April VGGT variant directories remain locally, but their original `predictions.npz` files are no longer present after cleanup. Those rows are therefore historical-report evidence, not newly recomputed point clouds.",
        "",
        "## What this answers",
        "",
        "The mentor asked to compare the current result against baseline VGGT and other VGGT-based point-cloud estimation variants before deciding the next improvement direction. The comparison below separates two questions:",
        "",
        "1. Which VGGT route gave measurable occupancy or consistency gains?",
        "2. Which route actually produced clearer, more stable human point-cloud geometry in Open3D?",
        "",
        "The answer is not the same. Crop and softmatte raised point counts substantially. Normal/self-geometry improved several consistency metrics. The prior-enabled V42/V50R2 route added normal and package evidence. But none of these, by themselves, proves a large visible improvement in full head/face/hair/right-hand geometry over the best VGGT baseline.",
        "",
        "## Main lineage table",
        "",
        "| method | family | evidence | full | head | face | visual / mentor reading |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {method} | {family} | {artifact_status} | {full_points} | {head_points} | {face_points} | {visual_result} |".format(
                **{k: str(v).replace("|", "/") for k, v in r.items()}
            )
        )
    lines += [
        "",
        "## Key readings",
        "",
        "- `human_crop` is the clearest early win over full-image baseline: full points rose from about `40.9k` to `111.1k`, head from `9.0k` to `24.4k`, and face from about `4.2k` / `3.7k` to roughly `9.6k-11.5k` depending on report source.",
        "- `human_crop_softmatte` had the highest ROI point count, but also larger global deltas, so it is not automatically a better geometry result.",
        "- `normal_r16_xview_selfgeom` implemented the mentor's depth/point/normal coupling idea and improved some consistency rows, but same-protocol face ROI stayed below signfix and the Open3D result remained shell-like.",
        "- `r32_selfgeom_crop_weakprior` combined crop, self-geometry, and weak SMPL-X prior, but the visual gate still failed because head/face and hands remained shell/fragmented.",
        "- `V42` / `V50R2` are the current prior-enabled VGGT path. They are useful for candidate closure and normal/region evidence, but the current full point-map coordinates are not visibly far from base VGGT: V42 vs V25 mean L2 is `0.00053544`, and V50R2 main candidate point map equals V42 (`max_abs = 0.0`).",
        "",
        "## File audit for recomputation",
        "",
        "| path | exists | size | mtime |",
        "|---|---:|---:|---|",
    ]
    for a in audit:
        lines.append(f"| `{a['path']}` | {a['exists']} | {a['size']} | {a['mtime']} |")
    lines += [
        "",
        "## Mentor-facing conclusion",
        "",
        "The strongest honest statement is: baseline VGGT already recovers a rough full-body point cloud; crop helps occupancy and image detail visibility; depth/point/normal self-consistency is implemented and technically active; SMPL-X prior-enabled V50R2 makes the result package stricter and supplies normals/region evidence. However, the current evidence does not yet show a decisive visible improvement in detailed human geometry over all VGGT baselines, especially for face/hairline/right hand. The next useful work should target local point-map-changing geometry, not only more reports or point-count increases.",
        "",
    ]
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_mentor_md(rows: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    lines = [
        "# VGGT 系列人体点云路线纵向比较补充",
        "",
        f"- 生成时间 UTC：`{now()}`",
        "- 比较范围：只比较“基于 VGGT 输出 depth / point map / normal 的人体点云路线”。",
        "- Kinect、COLMAP/MVS、2DGS、PSHuman、DepthPro、Sapiens 等外部 teacher 或传感器路线不放进主表；它们只能作为失败路线或参考证据。",
        "- 证据边界：4 月底很多 VGGT 变体的目录还在，但原始 `predictions.npz` 已经不在当前本地工作树里。因此这些路线只能按历史报告记录比较，不能伪装成这次重新复算过的点云结果。",
        "",
        "## 这部分回答导师什么问题",
        "",
        "导师要求先看纵向比较：baseline VGGT 的点云效果怎样，基于 VGGT 的其他人体点云估计方式有没有改善，数值和点云视觉都要看，再判断改进空间。",
        "",
        "这次对比后可以比较明确地说：",
        "",
        "1. 原始 full-image VGGT baseline 能恢复粗全身，但 head / face ROI 很弱。",
        "2. crop / softmatte 是早期最明显的输入侧提升，主要改善人体在输入里的占比和 ROI 点数。",
        "3. normal / depth / point 自一致路线落实了导师说的几何耦合，但没有单独把 6-view face/head 点云变成清晰连续的人脸几何。",
        "4. SMPL-X prior-enabled V42 / V50R2 补齐了 prior、normal、region evidence 和 candidate package 闭环，但主 point-map 坐标相对 base VGGT 的改变很小。",
        "5. 所以当前不能说“整体细节已经明显超过所有 VGGT baseline”。更稳妥的说法是：工程闭环已完成，候选包可提交；真正的视觉细节改进仍集中在 head/face/hairline 和 right hand。",
        "",
        "## 纵向比较表",
        "",
        "| 路线 | 类型 | 证据状态 | full 点数 | head 点数 | face 点数 | 结论 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {method} | {family} | {artifact_status} | {full_points} | {head_points} | {face_points} | {visual_result} |".format(
                **{k: str(v).replace("|", "/") for k, v in r.items()}
            )
        )
    lines += [
        "",
        "## 逐路线说明",
        "",
        "### 1. 原始 full-image VGGT baseline",
        "",
        "早期记录中，原始 full-image VGGT 在 6-view 下 full ROI 约 `40.9k`，head ROI 约 `9.0k`，face ROI 约 `4.2k`。这说明 baseline 不是完全失败，它能给出粗的人体点云；但对导师关心的头脸细节来说，点数和视觉连续性都不足。",
        "",
        "### 2. human crop / hardmask / softmatte",
        "",
        "crop 线是最明确的输入侧收益。`human_crop` 把 full ROI 从约 `40.9k` 提到约 `111.1k`，head 从约 `9.0k` 提到约 `24.4k`，face 从约 `3.7k-4.2k` 提到约 `9.6k-11.5k`。这和导师录音里说的“人占比太小，看不清细节，crop 有道理”一致。",
        "",
        "但 crop 的收益主要是 occupancy 和输入占比，不等价于真正的 3D 面部几何变清晰。`softmatte` 点数最高，但全局几何扰动也更大，所以不能只按点数说它最好。",
        "",
        "### 3. normal / depth / point 自一致 VGGT",
        "",
        "`r16_xview_selfgeom` 和后面的 `r32_selfgeom_crop_weakprior` 对应导师说的 depth、point、normal 要耦合，而不是三个分支各学各的。历史结果显示，一些 normal-depth-point consistency 指标确实改善；但同协议 face ROI 没有超过 signfix reference，Open3D 视觉仍然偏 shell-like，不能作为最终头脸点云质量通过。",
        "",
        "这说明导师的方向是对的：normal 是必要的几何约束；但当前 normal/self-geometry 还不是单独充分条件。它能让输出之间更自洽，但缺少高质量、连续、对齐的局部 head/face teacher 或直接 point-map 优化时，点云视觉仍然不够。",
        "",
        "### 4. V42 prior-enabled VGGT 与 V50R2 candidate",
        "",
        "当前可复算的 V25/V42/V50R2 对比显示：V42 prior-enabled 相比 V25 base VGGT 的全像素 point-map mean L2 为 `0.00053544`；V50R2 candidate 主点图与 V42 的 max abs 差异为 `0.0`。这说明 V50R2 的主全身 point-map 不是在 V42 后又大幅变形出一套新几何。",
        "",
        "V50R2 的主要价值在于：补齐 SMPL-X native prior-enabled 路线、normal availability、region evidence、head/hand package、formal replay、D-line candidate package 和 registry 闭环。它是 candidate 交付闭环，而不是“视觉细节全面碾压 baseline”的证据。",
        "",
        "## 当前可复算文件审计",
        "",
        "| 路径 | 是否存在 | 大小 | 修改时间 |",
        "|---|---:|---:|---|",
    ]
    for a in audit:
        lines.append(f"| `{a['path']}` | {a['exists']} | {a['size']} | {a['mtime']} |")
    lines += [
        "",
        "## 给导师的结论口径",
        "",
        "如果导师问“其它基于 VGGT 做点云估计的结果有没有明显更好”，目前应回答：有局部路线收益，但没有一个旧路线真正达到最终要求。crop 对点数和输入占比帮助最大；normal/self-geometry 对一致性有帮助；SMPL-X prior-enabled V50R2 完成了候选包和几何证据闭环。但从最终点云视觉看，head/face/hairline/right hand 的细节仍然不够，下一步需要直接改善 target-view point map 或引入更可靠的独立局部几何证据。",
        "",
    ]
    MD_MENTOR_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = method_rows()
    audit = current_file_audit()
    write_csv(rows)
    payload = {
        "task": "v223_vggt_variant_lineage_comparison",
        "created_utc": now(),
        "status": "DONE_PASS",
        "scope": "VGGT-derived point-cloud routes only",
        "rows": rows,
        "file_audit": audit,
        "reports": {
            "md": str(MD_REPORT),
            "mentor_md": str(MD_MENTOR_REPORT),
            "json": str(JSON_REPORT),
            "csv": str(CSV_REPORT),
        },
    }
    JSON_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(rows, audit)
    write_mentor_md(rows, audit)
    print(json.dumps(payload["reports"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
