from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
JSON_REPORT = REPORTS / "20260509_v50r2_self_supervision_status.json"
MD_REPORT = REPORTS / "20260509_v50r2_self_supervision_status.md"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = {
        "task": "v50r2_self_supervision_status",
        "created_utc": now(),
        "status": "DONE_PASS",
        "implemented": True,
        "mentor_recording_item": "depth can be converted to normal; depth, point, and normal should be geometrically coupled rather than learned as unrelated heads",
        "implementation": [
            {
                "name": "normal_from_point_map_supervision",
                "path": "training/loss.py",
                "evidence_lines": "compute_normal_loss derives target normals from dense point maps and restricts supervision to valid human masks",
            },
            {
                "name": "depth_to_normal_consistency",
                "path": "training/loss.py",
                "evidence_lines": "predicted depth is unprojected to camera points, point_map_to_normal_map builds depth-derived normals, then cosine normal loss compares them with predicted normals",
            },
            {
                "name": "depth_to_point_geometry_chain",
                "path": "training/loss.py",
                "evidence_lines": "compute_unproject_geometry_loss differentiably reconstructs world points from predicted depth and predicted camera pose",
            },
            {
                "name": "selfgeom_training_configs",
                "path": "training/config",
                "evidence_lines": "depthnormal_coupled, xview_selfgeom, teacherless selfgeom, and depth_point_normal_direct configs enable depth-normal, depth-point, point-normal, and cross-view consistency variants",
            },
        ],
        "effect_vs_without": {
            "positive": "normal-depth-point consistency metrics improved in earlier r2/r3/r16 evaluations",
            "negative": "same-protocol 6-view Open3D face/head point cloud did not reliably beat the signfix/reference visual gate",
            "v50r2_claim": "self-geometry is used as auxiliary consistency/audit evidence; it is not the sole source of V50R2 strict candidate pass and it is not an independent teacher",
        },
        "mentor_report_updates": [
            "reports/20260509_v50r2_mentor_report_human_geometry_recovery.md",
            "output/mentor_report_v50r2/images/05_vggt_smplx_native_architecture.png",
        ],
    }
    JSON_REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD_REPORT.write_text(
        "\n".join(
            [
                "# V50R2 自监督 / 几何自一致性状态",
                "",
                "结论：目前版本已经实现导师录音里提到的 depth / point / normal 几何自一致约束，但它是辅助约束和审查证据，不是独立 teacher，也不是 V50R2 通过 candidate gate 的唯一原因。",
                "",
                "## 如何实现",
                "",
                "- `training/loss.py` 的 `compute_normal_loss` 从 dense point map 在线计算 target normal，并把 normal 监督限制在有效人体区域。",
                "- 同一函数支持把预测 depth 反投影成 camera points，再计算 depth-derived normal，和网络输出 normal 做 cosine consistency。",
                "- `compute_unproject_geometry_loss` 用预测 depth + 预测 camera pose 可微反投影到 world points，再和 world point / human prior pseudo point 对齐。",
                "- `training/config` 下的 `depthnormal_coupled`、`xview_selfgeom`、`teacherless selfgeom`、`depth_point_normal_direct` 配置记录了 depth-normal、depth-point、point-normal 和 cross-view self-geometry 的实验路线。",
                "",
                "## 效果",
                "",
                "历史 r2 / r3 / r16 实验显示，normal-depth-point consistency 指标有改善；但同协议 6-view Open3D face/head 点云没有稳定超过 signfix/reference visual gate。因此汇报时应写成：自监督几何一致性已经实现并产生可测量改善，但它本身没有单独解决头脸连续表面问题。",
                "",
                "## V50R2 中的定位",
                "",
                "V50R2 把这条线作为 candidate consistency / normal-depth audit 使用。它支持 candidate 几何自洽审查，但不能包装成 strict teacher pass。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
