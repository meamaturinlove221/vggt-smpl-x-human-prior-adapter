from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAIN = Path(r"D:\vggt\vggt-main")
ROOT = MAIN / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
ARCHIVE = ROOT / "archive"
OUTPUT = ROOT / "output"
RUN_ROOT = OUTPUT / "V10000000_V12000000_modal_sparseconv"

MB = 1024 * 1024
VISUAL_LIMIT = 150 * MB
REPORT_LIMIT = 50 * MB
THIN_LIMIT = 450 * MB
SHARD_LIMIT = 450 * MB
UPLOAD_LIMIT = 500 * MB


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path, *, hash_it: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}
    if row["exists"] and hash_it:
        row["sha256"] = sha256(path)
    return row


def zip_clean(path: Path) -> str:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip() or "clean"


def make_zip(path: Path, files: list[Path], base: Path = ROOT) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    seen: set[str] = set()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for file in files:
            if not file.is_file():
                continue
            try:
                arc = file.relative_to(base).as_posix()
            except ValueError:
                arc = file.name
            if arc in seen:
                stem = Path(arc).stem
                suffix = Path(arc).suffix
                arc = f"duplicates/{stem}_{len(seen)}{suffix}"
            seen.add(arc)
            zf.write(file, arc)
    row = file_row(path)
    row["zip_test"] = zip_clean(path)
    row["entry_count"] = len(seen)
    row["under_500mb"] = bool(row["size"] <= UPLOAD_LIMIT)
    return row


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def run_record(run_id: str) -> dict[str, Any] | None:
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    if not status_path.is_file():
        return None
    status = read_json(status_path)
    best = status.get("best", {})
    pred = run_dir / "candidates" / str(best.get("name")) / "predictions.npz"
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "status": status,
        "best": best,
        "prediction": pred if pred.is_file() else None,
        "ranked": run_dir / "reports" / "V10800000_ranked_candidates.csv",
        "advisor": run_dir / "reports" / "V11100000_advisor_summary.md",
        "boards": sorted((run_dir / "boards").glob("*.png")) if (run_dir / "boards").is_dir() else [],
        "thin": run_dir / "V11000000_thin_review_bundle.zip",
        "full": run_dir / "V11000000_full_sparseconv_archive.zip",
    }


def default_run_ids() -> list[str]:
    return [
        "V100_formal_20260520",
        "V125_no_v129_full_20260520",
        "V126_no_v129_highscale_fast_20260520",
        "V125_smpl_only_20260520",
        "V125_no_semantic_20260520",
        "V157_observation_only_20260520",
        "V157_random_smpl_only_20260520",
        "V157_random_smpl_full_20260520",
    ]


def audit_previous_bundles() -> dict[str, Any]:
    v150 = read_json(REPORTS / "V15000000_final_status.json", {})
    previous = []
    for key in ("thin_review_bundle", "visual_mentor_bundle"):
        p = Path(v150.get("bundles", {}).get(key, {}).get("path", ""))
        row = file_row(p)
        row["bundle_key"] = key
        row["over_500mb"] = bool(row["size"] > UPLOAD_LIMIT)
        previous.append(row)
    audit = {
        "created_utc": now(),
        "upload_limit_bytes": UPLOAD_LIMIT,
        "previous_bundles": previous,
        "previous_over_limit": [r for r in previous if r["over_500mb"]],
    }
    write_json(REPORTS / "V15200000_previous_bundle_size_audit.json", audit)
    policy = {
        "created_utc": now(),
        "limits": {
            "visual_bundle": VISUAL_LIMIT,
            "report_bundle": REPORT_LIMIT,
            "thin_review_bundle": THIN_LIMIT,
            "candidate_shard": SHARD_LIMIT,
            "absolute_upload_limit": UPLOAD_LIMIT,
        },
        "rule": "No upload-facing bundle may exceed 500MB; V141 thin bundle is superseded if over limit.",
    }
    write_json(REPORTS / "V15200000_packaging_policy.json", policy)
    return audit


def summarize_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        ranked = read_csv(rec["ranked"])
        best = rec["best"]
        pure_spconv_rows = [row for row in ranked if row.get("blend") == "spconv"]
        pure_spconv = pure_spconv_rows[0] if pure_spconv_rows else {}
        rows.append(
            {
                "run_id": rec["run_id"],
                "status": rec["status"].get("status"),
                "best_name": best.get("name"),
                "teacher_mode": best.get("teacher_mode", rec["status"].get("teacher_mode")),
                "feature_mode": best.get("feature_mode", rec["status"].get("feature_mode")),
                "blend": best.get("blend"),
                "mean_delta_vs_v999": best.get("mean_delta_vs_v999"),
                "full_body_delta": best.get("full_body_delta"),
                "head_face_delta": best.get("head_face_delta"),
                "hairline_delta": best.get("hairline_delta"),
                "left_hand_delta": best.get("left_hand_delta"),
                "right_hand_delta": best.get("right_hand_delta"),
                "changed_vs_v999": best.get("changed_vs_v999"),
                "real_sparse_backend": rec["status"].get("real_sparse_backend"),
                "pure_spconv_name": pure_spconv.get("name"),
                "pure_spconv_mean_delta_vs_v999": pure_spconv.get("mean_delta_vs_v999"),
                "pure_spconv_full_body_delta": pure_spconv.get("full_body_delta"),
                "pure_spconv_head_face_delta": pure_spconv.get("head_face_delta"),
                "pure_spconv_hairline_delta": pure_spconv.get("hairline_delta"),
                "pure_spconv_left_hand_delta": pure_spconv.get("left_hand_delta"),
                "pure_spconv_right_hand_delta": pure_spconv.get("right_hand_delta"),
                "prediction_size": rec["prediction"].stat().st_size if rec["prediction"] else 0,
                "prediction": str(rec["prediction"]) if rec["prediction"] else "",
            }
        )
    write_csv(REPORTS / "V15700000_causal_ablation_v2.csv", rows)
    full = next((r for r in rows if r["run_id"] == "V126_no_v129_highscale_fast_20260520"), None)
    obs = next((r for r in rows if r["run_id"] == "V157_observation_only_20260520"), None)
    rnd = next((r for r in rows if r["run_id"] == "V157_random_smpl_only_20260520"), None)
    rnd_full = next((r for r in rows if r["run_id"] == "V157_random_smpl_full_20260520"), None)
    smpl = next((r for r in rows if r["run_id"] == "V125_smpl_only_20260520"), None)
    def f(row: dict[str, Any] | None, key: str) -> float:
        try:
            return float((row or {}).get(key) or 0.0)
        except Exception:
            return 0.0
    def fp(row: dict[str, Any] | None) -> float:
        return f(row, "pure_spconv_mean_delta_vs_v999")
    conclusion = {
        "created_utc": now(),
        "full_no_v129_run": full,
        "observation_only_control": obs,
        "random_smpl_only_control": rnd,
        "random_smpl_full_control": rnd_full,
        "smpl_only_control": smpl,
        "random_smpl_full_weaker_than_full": bool(rnd_full and full and fp(rnd_full) < fp(full) * 0.75),
        "random_smpl_only_weaker_than_full": bool(rnd and full and fp(rnd) < fp(full) * 0.75),
        "observation_only_weaker_than_full": bool(obs and full and fp(obs) < fp(full) * 0.75),
        "smpl_signal_positive": bool(smpl and fp(smpl) > 0.00005),
        "full_no_v129_positive": bool(full and fp(full) > 0.0004),
        "comparison_basis": "pure_spconv blend, not spconv_humanram_mix, to avoid HumanRAM blend leakage in causal controls",
    }
    write_json(REPORTS / "V15700000_causal_conclusion.json", conclusion)
    md = [
        "# V15700000 Causal Conclusion",
        "",
        f"- full no-V129 positive: `{conclusion['full_no_v129_positive']}`",
        f"- SMPL signal positive: `{conclusion['smpl_signal_positive']}`",
        f"- random SMPL full weaker than full: `{conclusion['random_smpl_full_weaker_than_full']}`",
        f"- random SMPL only weaker than full: `{conclusion['random_smpl_only_weaker_than_full']}`",
        f"- observation-only weaker than full: `{conclusion['observation_only_weaker_than_full']}`",
        "- comparison basis: `pure_spconv` blend",
        "",
        "These are V151-V220 causal-v2 controls over the V150 evidence chain.",
        "",
    ]
    (REPORTS / "V15700000_causal_conclusion.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def make_advisor_v2(records: list[dict[str, Any]], causal_rows: list[dict[str, Any]]) -> dict[str, Path]:
    final = read_json(REPORTS / "V15000000_final_status.json", {})
    causal = read_json(REPORTS / "V15700000_causal_conclusion.json", {})
    report = REPORTS / "V18000000_advisor_report.md"
    one_page = REPORTS / "V18000000_advisor_one_page.md"
    limitations = REPORTS / "V18000000_advisor_limitations.md"
    lines = [
        "# V18000000 Advisor Report",
        "",
        "## 结论",
        "",
        "当前路线已经从 SMPL-X 直接 patch/replacement 转为 SMPL 三维结构 feature encoding：HumanRAM-style token feature 与 NeuralBody-style voxel feature + real spconv SparseConv3D 两条线都保留证据。",
        "",
        f"- V150 status: `{final.get('status')}`",
        "- V220 status: `V22000000_FINAL_ADVISOR_READY_NOT_PROMOTED` if all upload and cleanup gates pass.",
        "- 不 promotion，不写 strict registry，不修改 V50/V50R2，active candidate 保持 V11700。",
        "",
        "## V151-V220 新增证据",
        "",
        f"- full no-V129 positive: `{causal.get('full_no_v129_positive')}`",
        f"- SMPL signal positive: `{causal.get('smpl_signal_positive')}`",
        f"- random SMPL full weaker than full: `{causal.get('random_smpl_full_weaker_than_full')}`",
        f"- random SMPL only weaker than full: `{causal.get('random_smpl_only_weaker_than_full')}`",
        f"- observation-only weaker than full: `{causal.get('observation_only_weaker_than_full')}`",
        "",
        "## 关键候选",
        "",
    ]
    for row in causal_rows:
        lines.append(
            f"- `{row['run_id']}` / `{row['best_name']}`: mean_vs_v999={row.get('mean_delta_vs_v999')}, "
            f"full={row.get('full_body_delta')}, head={row.get('head_face_delta')}, hair={row.get('hairline_delta')}, "
            f"L/R hand={row.get('left_hand_delta')}/{row.get('right_hand_delta')}"
        )
    lines += [
        "",
        "## 打包规则",
        "",
        "V141 旧 thin bundle 为 688MB，超过 500MB 上传限制；V190 重新生成 visual/report/thin/shard 包，所有上传包必须低于 500MB。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    one_page.write_text("\n".join(lines[:30]) + "\n", encoding="utf-8")
    limitations.write_text(
        "# V18000000 Limitations\n\n"
        "- 当前仍为 REVIEW/ADVISOR ready，不做 promotion。\n"
        "- 如果后续导师要求更强泛化，需要更多帧/更多 subject 的 V117/V770 predictions。\n"
        "- hand/object 与 hairline 仍应作为后续视觉强化重点。\n",
        encoding="utf-8",
    )
    return {"report": report, "one_page": one_page, "limitations": limitations}


def upload_safe_bundles(records: list[dict[str, Any]], advisor: dict[str, Path]) -> dict[str, Any]:
    previous_audit = read_json(REPORTS / "V15200000_previous_bundle_size_audit.json", {})
    report_files = [
        REPORTS / "V15000000_final_status.json",
        REPORTS / "V12500000_causal_ablation_results.csv",
        REPORTS / "V12500000_causal_attribution.json",
        REPORTS / "V15700000_causal_ablation_v2.csv",
        REPORTS / "V15700000_causal_conclusion.json",
        REPORTS / "V15700000_causal_conclusion.md",
        REPORTS / "V15200000_previous_bundle_size_audit.json",
        REPORTS / "V15200000_packaging_policy.json",
        advisor["report"],
        advisor["one_page"],
        advisor["limitations"],
    ]
    visual_files = sorted(BOARDS.glob("V13000000_*.png")) + sorted(BOARDS.glob("V16500000_*.png"))
    if not visual_files:
        visual_files = sorted(BOARDS.glob("V*.png"))[:20]
    top_predictions = [r["prediction"] for r in records if r.get("prediction") and Path(r["prediction"]).is_file()]
    top_predictions = [Path(p) for p in top_predictions]
    report_bundle = make_zip(ARCHIVE / "V19000000_reports_bundle.zip", report_files)
    visual_bundle = make_zip(ARCHIVE / "V19000000_visual_bundle.zip", visual_files + [advisor["one_page"]])

    thin_files = report_files + visual_files + top_predictions[:5]
    thin_bundle = make_zip(ARCHIVE / "V19000000_thin_review_bundle.zip", thin_files)

    shards = []
    current: list[Path] = []
    current_size = 0
    shard_idx = 0
    for pred in top_predictions:
        size = pred.stat().st_size
        if current and current_size + size > SHARD_LIMIT:
            shards.append(make_zip(ARCHIVE / f"V19000000_candidate_shard_{shard_idx:03d}.zip", current))
            shard_idx += 1
            current = []
            current_size = 0
        current.append(pred)
        current_size += size
    if current:
        shards.append(make_zip(ARCHIVE / f"V19000000_candidate_shard_{shard_idx:03d}.zip", current))

    omitted = []
    for path in [
        ARCHIVE / "V14100000_thin_review_bundle.zip",
        ARCHIVE / "V14100000_visual_mentor_bundle.zip",
    ]:
        row = file_row(path)
        row["reason"] = "superseded_by_upload_safe_bundle" if path.name == "V14100000_thin_review_bundle.zip" else "retained_but_repacked"
        omitted.append(row)
    for rec in records:
        full = Path(rec["full"]) if rec.get("full") else None
        if full and full.is_file():
            row = file_row(full)
            row["reason"] = "large_full_archive_manifest_only"
            omitted.append(row)

    manifest = {
        "created_utc": now(),
        "limits": {"visual": VISUAL_LIMIT, "reports": REPORT_LIMIT, "thin": THIN_LIMIT, "shard": SHARD_LIMIT, "absolute": UPLOAD_LIMIT},
        "previous_bundle_size_audit": previous_audit,
        "bundles": {
            "visual": visual_bundle,
            "reports": report_bundle,
            "thin_review": thin_bundle,
            "candidate_shards": shards,
        },
        "all_upload_bundles_under_500mb": all(
            b["size"] <= UPLOAD_LIMIT
            for b in [visual_bundle, report_bundle, thin_bundle] + shards
        ),
        "omitted_large_files": omitted,
    }
    write_json(REPORTS / "V19000000_upload_manifest.json", manifest)
    return manifest


def cleanup_report(name: str) -> dict[str, Any]:
    report = {
        "created_utc": now(),
        "git_status": run_cmd(["git", "status", "--short", "--branch"], cwd=Path(r"D:\vggt\vggt-feature-adapter")),
        "modal_apps": run_cmd(["modal", "app", "list"], cwd=Path(r"D:\vggt\vggt-feature-adapter")),
        "process_scan_note": "No long-running Modal route should remain; current reporting command may appear transiently if inspected externally.",
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
    }
    write_json(REPORTS / name, report)
    return report


def final_status(manifest: dict[str, Any], causal_rows: list[dict[str, Any]], cleanup: dict[str, Any], *, post_push: bool) -> dict[str, Any]:
    causal = read_json(REPORTS / "V15700000_causal_conclusion.json", {})
    upload_ok = bool(manifest.get("all_upload_bundles_under_500mb"))
    causal_ok = bool(causal.get("full_no_v129_positive") and causal.get("smpl_signal_positive"))
    random_ok = causal.get("random_smpl_full_weaker_than_full") or causal.get("random_smpl_only_weaker_than_full")
    obs_ok = causal.get("observation_only_weaker_than_full")
    status = "V22000000_FINAL_ADVISOR_READY_NOT_PROMOTED" if upload_ok and causal_ok and random_ok and obs_ok else "V22000000_READY_BUT_LIMITATIONS_DISCLOSED"
    final = {
        "created_utc": now(),
        "status": status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "causal": causal,
        "causal_rows": causal_rows,
        "upload_manifest": manifest,
        "cleanup_report": cleanup,
        "post_push_cleanup": bool(post_push),
        "failed_hard_gates": [] if status == "V22000000_FINAL_ADVISOR_READY_NOT_PROMOTED" else [
            key for key, ok in {
                "upload_safe": upload_ok,
                "causal_positive": causal_ok,
                "random_smpl_control": bool(random_ok),
                "observation_only_control": bool(obs_ok),
            }.items() if not ok
        ],
    }
    write_json(REPORTS / "V22000000_final_status.json", final)
    return final


def run(run_ids: list[str], *, post_push: bool = False) -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    records = [r for rid in run_ids if (r := run_record(rid))]
    if not records:
        raise RuntimeError("No SparseConv run records found")
    audit_previous_bundles()
    causal_rows = summarize_runs(records)
    advisor = make_advisor_v2(records, causal_rows)
    manifest = upload_safe_bundles(records, advisor)
    cleanup = cleanup_report("V19500000_cleanup_report.json" if post_push else "V17000000_post_push_cleanup_report.json")
    final = final_status(manifest, causal_rows, cleanup, post_push=post_push)
    print(json.dumps(final, indent=2, ensure_ascii=True))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", default=default_run_ids())
    parser.add_argument("--post-push", action="store_true")
    args = parser.parse_args()
    run(args.run_ids, post_push=args.post_push)


if __name__ == "__main__":
    main()
