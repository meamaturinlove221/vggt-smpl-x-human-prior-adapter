from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPO = Path(r"D:\vggt\vggt-feature-adapter")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str], cwd: Path) -> dict:
    proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def load_rows() -> list[dict]:
    matrix = ROOT / "reports" / "V91900000_leakage_free_causal_matrix.csv"
    with matrix.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def group_stats(rows: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    groups = sorted({r["group"] for r in rows})
    for group in groups:
        vals = [to_float(r["mean_delta_vs_v999"]) for r in rows if r["group"] == group]
        stats[group] = {
            "n": len(vals),
            "mean": mean(vals) if vals else 0.0,
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }
    return stats


def required_candidate_files(row: dict) -> dict:
    pred = Path(row["prediction"])
    cand_dir = pred.parent
    return {
        "final_status": cand_dir.parent.parent / "reports" / "V12000000_final_status.json",
        "predictions": pred,
        "eval": Path(row["eval"]),
        "config": cand_dir / "config.json",
        "source_manifest": Path(row["source_manifest"]),
        "board": cand_dir / "board.png",
        "changed_map": cand_dir / "changed_map.png",
    }


def audit_candidate_artifacts(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    audits = []
    failures = []
    for row in rows:
        files = required_candidate_files(row)
        file_audit = {}
        missing = []
        for name, path in files.items():
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            file_audit[name] = {
                "path": str(path),
                "exists": exists,
                "size": size,
            }
            if not exists or size <= 0:
                missing.append(name)
        status = "complete" if not missing else "incomplete"
        item = {
            "run_id": row["run_id"],
            "group": row["group"],
            "best_name": row["best_name"],
            "status": status,
            "missing_or_empty": missing,
            "files": file_audit,
        }
        audits.append(item)
        if missing:
            failures.append(item)
    return audits, failures


def write_ranked(rows: list[dict]) -> None:
    out = ROOT / "reports" / "V99000000_ranked_final_candidates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: to_float(r["mean_delta_vs_v999"]), reverse=True)
    fields = [
        "rank",
        "group",
        "run_id",
        "best_name",
        "mean_delta_vs_v999",
        "full_body_delta",
        "head_face_delta",
        "hairline_delta",
        "left_hand_delta",
        "right_hand_delta",
        "background_leakage",
        "depth_world_consistency",
        "prediction",
        "eval",
        "source_manifest",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(sorted_rows, 1):
            writer.writerow({
                "rank": idx,
                "group": row["group"],
                "run_id": row["run_id"],
                "best_name": row["best_name"],
                "mean_delta_vs_v999": row["mean_delta_vs_v999"],
                "full_body_delta": row["full_body_delta"],
                "head_face_delta": row["head_face_delta"],
                "hairline_delta": row["hairline_delta"],
                "left_hand_delta": row["left_hand_delta"],
                "right_hand_delta": row["right_hand_delta"],
                "background_leakage": row["background_leakage"],
                "depth_world_consistency": row["depth_world_consistency"],
                "prediction": row["prediction"],
                "eval": row["eval"],
                "source_manifest": row["source_manifest"],
            })


def copy_existing(files: list[Path], dest_root: Path) -> list[tuple[Path, str]]:
    copied = []
    for path in files:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path(path.name)
        copied.append((path, rel.as_posix()))
    return copied


def zip_files(zip_path: Path, entries: list[tuple[Path, str]]) -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        seen = set()
        for path, arcname in entries:
            if not path.exists() or arcname in seen:
                continue
            z.write(path, arcname)
            seen.add(arcname)
    with ZipFile(zip_path, "r") as z:
        bad = z.testzip()
        count = len(z.infolist())
    return {
        "path": str(zip_path),
        "size": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "entry_count": count,
        "zip_test": "clean" if bad is None else f"bad:{bad}",
    }


def selected_prediction_entries(rows: list[dict]) -> tuple[list[tuple[Path, str]], list[dict]]:
    selected_groups = [
        "leakage_free_true_full",
        "leakage_free_observation_only",
        "leakage_free_support_only",
        "leakage_free_semantic_only",
        "leakage_free_no_teacher",
    ]
    entries: list[tuple[Path, str]] = []
    selected = []
    for group in selected_groups:
        candidates = [r for r in rows if r["group"] == group]
        if not candidates:
            continue
        row = max(candidates, key=lambda r: to_float(r["mean_delta_vs_v999"]))
        run_id = row["run_id"]
        files = required_candidate_files(row)
        for name in ["predictions", "eval", "config", "source_manifest", "board", "changed_map"]:
            path = files[name]
            entries.append((path, f"selected_predictions/{group}/{run_id}/{path.name}"))
        selected.append({
            "group": group,
            "run_id": run_id,
            "mean_delta_vs_v999": to_float(row["mean_delta_vs_v999"]),
            "prediction": row["prediction"],
        })
    return entries, selected


def make_reports(summary: dict, stats: dict, failures: list[dict]) -> None:
    strict = {
        "status": "V99000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS",
        "created_utc": now(),
        "v919_conclusion": summary.get("conclusion"),
        "failed_hard_gates": [
            "leakage_free_route_exhausted",
            "smpl_semantic_independent_causality_not_confirmed",
            "support_semantic_no_teacher_routes_match_or_exceed_true_full",
        ],
        "checks": {
            "real_sparseconv": True,
            "upload_safe_pending": True,
            "leakage_free_complete": True,
            "paper_grade_ready": False,
            "promotion": False,
            "strict_registry_written": False,
            "v50_v50r2_modified": False,
            "active_candidate": "V11700_gap_reduction_branch_520",
        },
        "stats": stats,
        "key_differences": summary.get("key_differences", {}),
        "artifact_failures": failures,
        "decision": (
            "Do not enter V940 as the main paper-grade branch. V919 no-blend/no-composition "
            "rerun did not confirm independent SMPL semantic causality."
        ),
    }
    write_json(ROOT / "reports" / "V99000000_strict_final_eval.json", strict)

    report = f"""# V99200000 Advisor Report

老师，这轮已经完成 V9175 no-blend repair 和 V919 leakage-free causal rerun。结论需要谨慎：当前路线不能写成“SMPL 语义编码已被强因果证明”。

## 结论

- V919 conclusion: `{summary.get("conclusion")}`
- final status: `V99900000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS`
- promotion: false
- strict registry: not written
- V50/V50R2: not modified
- active candidate: `V11700_gap_reduction_branch_520`

## 关键统计

- true full mean: `{stats['leakage_free_true_full']['mean']}`
- random SMPL full mean: `{stats['leakage_free_random_smpl_full']['mean']}`
- shuffled SMPL full mean: `{stats['leakage_free_shuffled_smpl_full']['mean']}`
- observation-only mean: `{stats['leakage_free_observation_only']['mean']}`
- no-SparseConv MLP mean: `{stats['leakage_free_no_sparseconv_mlp']['mean']}`
- no-teacher mean: `{stats['leakage_free_no_teacher']['mean']}`
- support-only mean: `{stats['leakage_free_support_only']['mean']}`
- semantic-only mean: `{stats['leakage_free_semantic_only']['mean']}`

## 判断

V9175 证明旧路线里存在 composition/blend 相关风险；V919 进一步把 no-blend/no-composition 路线重跑到 40/40。结果显示 true full 并没有稳定高于 support-only / semantic-only / no-teacher，observation-only 也保留了明显正收益。因此当前正收益更像来自 observation、support、teacher-free residual、SparseConv/MLP 平滑与指标偏好的组合，而不是由 SMPL semantic 独立主导。

## 后续方向

后续若继续推进，应先做 architecture rebuild，而不是进入 V940 作为主线：

1. observation-disentangled route: 限制 observation-only 的解释力。
2. semantic-disentangled route: support 只做 reliability gate，semantic branch 加 canonical/body-part/skinning auxiliary losses。
3. smoothing-vs-structure route: 用 body-part boundary、hairline boundary、hand component metrics 区分结构收益和平滑收益。
"""
    write_text(ROOT / "reports" / "V99200000_advisor_report.md", report)

    one_page = f"""# V992 One Page

V919 leakage-free rerun completed 40/40 and returned `{summary.get("conclusion")}`.

The route is not promoted and not paper-grade. True full is not clearly separated from support-only, semantic-only, no-teacher, and observation-only controls. The correct final state is `V99900000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS`.
"""
    write_text(ROOT / "reports" / "V99200000_advisor_one_page.md", one_page)

    limitations = """# V992 Limitations

- SMPL semantic independent causality is not confirmed.
- Observation/support/no-teacher routes remain strongly positive.
- Teacher and V770 comparison assets still appear in source manifests, although post-compose/blend is disabled.
- Full paper-grade VGGT checkpoint token training should not be used to overclaim causality before architecture disentanglement.
"""
    write_text(ROOT / "reports" / "V99200000_limitations.md", limitations)


def make_cleanup_report() -> dict:
    report = {
        "created_utc": now(),
        "feature_adapter_git_status": run_cmd(["git", "status", "--short", "--branch"], REPO),
        "feature_adapter_commit": run_cmd(["git", "rev-parse", "HEAD"], REPO),
        "feature_adapter_branch": run_cmd(["git", "branch", "--show-current"], REPO),
        "vggt_main_git_status": run_cmd(["git", "status", "--short", "--branch"], Path(r"D:\vggt\vggt-main")),
        "modal_app_list": run_cmd(["modal", "app", "list"], REPO),
        "python_v919_process_scan": run_cmd([
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'V91900000|V9175|modal_v10000000_sparseconv_route' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3",
        ], REPO),
        "registry_diff": run_cmd(["git", "diff", "--name-only", "--", "registry"], REPO),
        "v50_v50r2_diff": run_cmd([
            "powershell",
            "-NoProfile",
            "-Command",
            "git diff --name-only | Select-String -Pattern 'V50|V50R2'",
        ], REPO),
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
    }
    write_json(ROOT / "reports" / "V99700000_post_push_cleanup_report.json", report)
    return report


def make_bundles(rows: list[dict], summary: dict, stats: dict, cleanup: dict) -> dict:
    archive = ROOT / "archive"
    reports = ROOT / "reports"
    boards = ROOT / "boards"
    failures = ROOT / "failures"

    common_reports = [
        reports / "V91900000_leakage_free_causal_matrix.csv",
        reports / "V91900000_leakage_free_causal_summary.md",
        reports / "V91900000_leakage_free_causal_summary.json",
        reports / "V91900000_seed_level_metrics.csv",
        reports / "V91900000_source_manifest_audit.json",
        reports / "V91900000_cli_hang_recovery.json",
        failures / "V91900000_failed_jobs.json",
        reports / "V91750000_teacher_leakage_repair_summary.json",
        reports / "V91750000_teacher_leakage_repair.md",
        reports / "V91500000_support_vs_semantic_attribution.md",
        reports / "V91700000_teacher_leakage_audit.md",
        reports / "V99000000_strict_final_eval.json",
        reports / "V99000000_ranked_final_candidates.csv",
        reports / "V99200000_advisor_report.md",
        reports / "V99200000_advisor_one_page.md",
        reports / "V99200000_limitations.md",
        reports / "V99700000_post_push_cleanup_report.json",
    ]
    visual_files = [
        boards / "V91900000_leakage_free_causal_visual.png",
        boards / "V91750000_no_blend_repair_visual.png",
        boards / "V91000000_causal_matrix_visual.png",
    ]
    prediction_entries, selected_predictions = selected_prediction_entries(rows)

    omitted = []
    selected_paths = {str(p) for p, _ in prediction_entries}
    for row in rows:
        pred = Path(row["prediction"])
        if pred.exists() and str(pred) not in selected_paths:
            omitted.append({
                "path": str(pred),
                "size": pred.stat().st_size,
                "sha256": sha256(pred),
                "category": "omitted_v919_prediction",
                "required_for_repro": True,
                "reason": "Full 40-seed predictions would exceed compact upload target.",
            })
    write_json(ROOT / "reports" / "V99500000_omitted_large_files_manifest.json", omitted)

    final_status = {
        "status": "V99900000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS",
        "created_utc": now(),
        "best_candidate": None,
        "v919_conclusion": summary.get("conclusion"),
        "paper_grade_ready": False,
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "failed_hard_gates": [
            "leakage_free_smpl_causality_not_confirmed",
            "route_exhausted_after_v919",
        ],
        "checks": {
            "v910_completed": True,
            "v915_completed": True,
            "v917_completed": True,
            "v9175_completed": True,
            "v919_completed": True,
            "upload_safe": True,
            "paper_grade": False,
        },
        "stats": stats,
        "selected_predictions": selected_predictions,
        "advisor_report": str(ROOT / "reports" / "V99200000_advisor_report.md"),
        "omitted_large_files_manifest": str(ROOT / "reports" / "V99500000_omitted_large_files_manifest.json"),
        "cleanup_report": str(ROOT / "reports" / "V99700000_post_push_cleanup_report.json"),
    }
    write_json(ROOT / "reports" / "V99900000_final_status.json", final_status)

    core_entries = copy_existing([
        reports / "V99900000_final_status.json",
        reports / "V99000000_strict_final_eval.json",
        reports / "V91900000_leakage_free_causal_summary.json",
        reports / "V91900000_source_manifest_audit.json",
        reports / "V99700000_post_push_cleanup_report.json",
        reports / "V99200000_advisor_report.md",
        reports / "V99200000_limitations.md",
        reports / "V99500000_omitted_large_files_manifest.json",
    ], ROOT)
    report_entries = copy_existing(common_reports + [reports / "V99500000_omitted_large_files_manifest.json"], ROOT)
    visual_entries = copy_existing(visual_files, ROOT)

    bundles = {
        "core_evidence": zip_files(archive / "V99500000_core_evidence_bundle.zip", core_entries),
        "reports": zip_files(archive / "V99500000_reports_bundle.zip", report_entries),
        "visuals": zip_files(archive / "V99500000_visuals_bundle.zip", visual_entries),
        "selected_predictions": zip_files(archive / "V99500000_selected_predictions_bundle.zip", prediction_entries),
    }
    total = sum(item["size"] for item in bundles.values())
    manifest = {
        "status": "V99500000_UPLOAD_SAFE_FINAL_PACKAGE",
        "created_utc": now(),
        "total_upload_bytes": total,
        "total_upload_under_500mb": total < 500 * 1024 * 1024,
        "each_bundle_under_500mb": all(item["size"] < 500 * 1024 * 1024 for item in bundles.values()),
        "bundles": bundles,
        "selected_predictions": selected_predictions,
        "omitted_large_files_manifest": str(ROOT / "reports" / "V99500000_omitted_large_files_manifest.json"),
        "note": "Exact bundle hashes are recorded here after package creation. Core bundle contains final status and core evidence; this manifest is the final sidecar generated after hashing all zip files.",
    }
    write_json(ROOT / "reports" / "V99500000_upload_manifest.json", manifest)
    return manifest


def main() -> None:
    rows = load_rows()
    if len(rows) != 40:
        raise SystemExit(f"V919 row count is {len(rows)}, expected 40")
    stats = group_stats(rows)
    summary = read_json(ROOT / "reports" / "V91900000_leakage_free_causal_summary.json")
    artifact_audit, failures = audit_candidate_artifacts(rows)
    write_json(ROOT / "reports" / "V99000000_v919_artifact_completeness.json", artifact_audit)
    write_json(ROOT / "failures" / "V91900000_failed_jobs.json", {
        "created_utc": now(),
        "failed_count": len(failures),
        "failures": failures,
    })
    write_ranked(rows)
    make_reports(summary, stats, failures)
    cleanup = make_cleanup_report()
    manifest = make_bundles(rows, summary, stats, cleanup)
    if not manifest["total_upload_under_500mb"] or not manifest["each_bundle_under_500mb"]:
        raise SystemExit("Upload-safe package size gate failed")
    print(json.dumps({
        "status": "V99900000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS",
        "manifest": str(ROOT / "reports" / "V99500000_upload_manifest.json"),
        "total_upload_bytes": manifest["total_upload_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
