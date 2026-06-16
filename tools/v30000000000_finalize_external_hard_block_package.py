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
OUTPUT = AUX / "output"
ARCHIVE = AUX / "archive"
STATUS = "V∞_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str]) -> dict[str, Any]:
    p = subprocess.run(args, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"args": args, "returncode": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-8000:]}


def write_reports() -> dict[str, Any]:
    v290 = json.loads((REPORTS / "V29000000000_decision.json").read_text(encoding="utf-8"))
    final = {
        "created_utc": now(),
        "final_status": STATUS,
        "mentor_requirement_satisfied": False,
        "semantic_topology_causality_confirmed": False,
        "dominant_failure": "trusted camera coordinate binding blocks nonzero reprojection-consistent residuals",
        "best_candidate": "V11700_gap_reduction_branch_520 remains active; V220/V240/V250/V260 are evidence candidates only",
        "no_promotion": True,
        "no_strict_registry": True,
        "no_v50_v50r2_modification": True,
        "active_candidate_replaced": False,
        "hard_block": v290,
    }
    write_json(REPORTS / "V30000000000_final_status.json", final)
    checklist = {
        "created_utc": now(),
        "required_user_actions": v290["user_action_checklist"],
        "required_files_or_confirmations": [
            "matching *_annots.smc for the exact V11700/V920 subject/action if not G:\\数据集\\datasets\\data_used_in_4K4D\\annotations\\0012_11_annots.smc",
            "world-coordinate transform from VGGT world_points to 4K4D camera coordinates",
            "trusted 518x518 reprojection masks/silhouettes for camera ids 00,01,15,30,45,59",
        ],
    }
    write_json(REPORTS / "V30000000000_user_action_checklist.json", checklist)

    advisor = f"""# 先给结论

本轮没有达到导师要求的 mentor-ready。原因不是 SMPL-X 主模型缺失，也不是 Modal/GPU 不可用，而是可信 camera-bound reprojection 门卡住：4K4D SMC 的 K/RT 可以读取，但当前 V11700/V920 world_points 与该 K/RT 坐标系不一致；所有非零 residual 在 reprojection trust 下都失败，V290 最优 scale 退化为 0。

# 本轮解决的问题

- 完成 V210 semantic-gate graph 修复：修掉 support/semantic tensor 维度和 MHA 维度错误。
- 完成 V220 10 组 × 5 seeds full mentor-gate matrix，true_surface_transformer 排第一。
- 完成 V230 sample-to-fullview 投影，生成 6×518×518 predictions 和新 3D boards。
- 完成 V240 dense full-view inference 和 connected-component region metrics。
- 完成 V250 dense teacher visual route，但明确不作为 independent causality。
- 完成 V260 TSDF/SDF backend metric，true 仍排第一。
- 完成 V270 4K4D SMC camera binding：K/RT 可读取，camera ids 00/01/15/30/45/59 可绑定。
- 完成 V280/V290 reprojection-aware scale search，确认非零 residual 无法通过当前 camera-bound trust。

# 核心证据

| 项目 | 结论 |
|---|---|
| Formal controls | V220 true_surface_transformer 高于 random、strong shuffled、local smoothing、noSparse、no/random graph、observation/support、no_teacher |
| Full-view tensors | V230/V240/V250 均写出 6×518×518 predictions |
| Region metrics | full_body/head_face/hairline/left_hand/right_hand 已评估，component_count 已加入 |
| Camera binding | 4K4D `0012_11_annots.smc` 可读，K/RT 存在 |
| Reprojection gate | 失败；V270 true 排名最差，V280/V290 最佳可行 scale=0 |
| Normal | 非零 residual 存在，但不能越过 camera-bound gate |

# 为什么可信

- 没有 promotion。
- 没有 strict registry。
- 没有修改 V50/V50R2。
- 没有替换 active candidate。
- 没有把 V11700/V770/V999 post-compose 当 hidden target。
- V250 明确标记 teacher route limitation，没有写成 independent causality。
- V290 给出明确用户动作清单，而不是继续 proxy 循环。

# 仍然没解决的问题

1. 当前 V11700/V920 world coordinate 与读取到的 4K4D K/RT 之间缺可信变换。
2. 不能确认 `0012_11_annots.smc` 是否就是 V11700/V920 的匹配 subject/action。
3. 缺少可信的 518×518 silhouettes/reprojection masks 来做 hard reprojection gate。

# 给导师看的图

- `D:\\vggt\\vggt-main\\local_report_auxiliary\\V600_quality_rebuild\\boards\\V23000000000_fullbody.png`
- `D:\\vggt\\vggt-main\\local_report_auxiliary\\V600_quality_rebuild\\boards\\V24000000000_fullbody.png`
- `D:\\vggt\\vggt-main\\local_report_auxiliary\\V600_quality_rebuild\\boards\\V25000000000_fullbody.png`
- `D:\\vggt\\vggt-main\\local_report_auxiliary\\V600_quality_rebuild\\boards\\V26000000000_tsdf_visual.png`

# 下一步论文路线

先补齐 camera/world 坐标系绑定，再回到 V290：如果非零 residual 能通过 reprojection trust，再跑 V220/V240/V260 全控制复验；否则不要继续做 semantic/topology route 包装。
"""
    (REPORTS / "V30000000000_advisor_report.md").write_text(advisor, encoding="utf-8")
    (REPORTS / "V30000000000_one_page.md").write_text(
        "结论：未达 mentor-ready；真实外部硬阻断是 camera/world 坐标绑定缺失。需要用户提供匹配 SMC 或 VGGT-to-4K4D world transform。\n",
        encoding="utf-8",
    )
    (REPORTS / "V30000000000_limitations.md").write_text(
        "- V290 最优 residual scale=0。\n- 非零 geometry improvement 与当前 camera-bound reprojection 不兼容。\n- 需要人工确认匹配 calibration/coordinate transform。\n",
        encoding="utf-8",
    )
    return final


def zip_files(out: Path, files: list[Path]) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists() and f.is_file():
                zf.write(f, f.relative_to(AUX) if f.is_relative_to(AUX) else f.name)
    with zipfile.ZipFile(out, "r") as zf:
        bad = zf.testzip()
    return {"path": str(out), "sha256": sha256(out), "size": out.stat().st_size, "zip_clean": bad is None, "bad_member": bad}


def make_sample_npz(src: Path, dst: Path) -> None:
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src, allow_pickle=False) as z:
        payload = {}
        for k in z.files:
            a = z[k]
            if a.ndim >= 3 and a.shape[:3] == (6, 518, 518):
                payload[k] = a[:, ::8, ::8, ...]
            else:
                payload[k] = a
    np.savez_compressed(dst, **payload)


def bundles() -> dict[str, Any]:
    tmp = OUTPUT / "V30000000000_bundle_samples"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    selected = []
    controls = []
    for group in ["true_surface_transformer", "random_semantic", "local_knn_smoothing_surface", "observation_only", "support_only"]:
        src = OUTPUT / "V25000000000_visual_first_predictions" / f"{group}_seed0" / "predictions.npz"
        dst = tmp / f"{group}_seed0_predictions_sample.npz"
        if src.exists():
            make_sample_npz(src, dst)
            if group == "true_surface_transformer":
                selected.append(dst)
            else:
                controls.append(dst)
    core_files = [
        REPORTS / "V30000000000_final_status.json",
        REPORTS / "V29000000000_decision.json",
        REPORTS / "V27000000000_camera_binding_report.json",
        REPORTS / "V30000000000_user_action_checklist.json",
    ]
    report_files = list(REPORTS.glob("V2*.json")) + list(REPORTS.glob("V2*.csv")) + [
        REPORTS / "V30000000000_advisor_report.md",
        REPORTS / "V30000000000_one_page.md",
        REPORTS / "V30000000000_limitations.md",
    ]
    visual_files = list(BOARDS.glob("V23000000000*.png")) + list(BOARDS.glob("V24000000000*.png")) + list(BOARDS.glob("V25000000000*.png")) + list(BOARDS.glob("V26000000000*.png"))
    manifest = {
        "created_utc": now(),
        "bundles": {
            "core": zip_files(ARCHIVE / "V30000000000_core_evidence_bundle.zip", core_files),
            "reports": zip_files(ARCHIVE / "V30000000000_reports_bundle.zip", report_files),
            "visuals": zip_files(ARCHIVE / "V30000000000_visuals_bundle.zip", visual_files),
            "selected_predictions": zip_files(ARCHIVE / "V30000000000_selected_predictions_bundle.zip", selected),
            "controls": zip_files(ARCHIVE / "V30000000000_controls_bundle.zip", controls),
        },
    }
    omitted = {
        "created_utc": now(),
        "omitted_reason": "Large full dense predictions omitted from upload bundles; sampled npz included for readability.",
        "large_roots": [
            str(OUTPUT / "V24000000000_dense_fullview_predictions"),
            str(OUTPUT / "V25000000000_visual_first_predictions"),
        ],
    }
    write_json(REPORTS / "V30000000000_upload_manifest_sidecar.json", manifest)
    write_json(REPORTS / "V30000000000_omitted_large_file_manifest.json", omitted)
    return manifest


def cleanup() -> dict[str, Any]:
    status = run_cmd(["git", "status", "--short"])
    branch = run_cmd(["git", "branch", "--show-current"])
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    modal_apps = run_cmd(["modal", "app", "list"])
    v50 = run_cmd(["git", "diff", "--", "V50", "V50R2", "docs/V50", "docs/V50R2"])
    cleanup_doc = {
        "created_utc": now(),
        "git_status_short": status,
        "branch": branch["stdout"].strip(),
        "commit": commit["stdout"].strip(),
        "worktree_clean_claim": False,
        "modal_apps": modal_apps,
        "registry_written": False,
        "v50_v50r2_diff": v50,
        "active_candidate": "V11700_gap_reduction_branch_520",
    }
    write_json(REPORTS / "V30000000000_post_push_cleanup.json", cleanup_doc)
    return cleanup_doc


def main() -> None:
    write_reports()
    manifest = bundles()
    cleanup_doc = cleanup()
    print(json.dumps({"status": STATUS, "manifest": manifest, "cleanup": cleanup_doc["branch"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
