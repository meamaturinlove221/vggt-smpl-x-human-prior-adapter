import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "compare_geometry_branches_zju_report.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ZJU geometry baselines for discovered old report.json cases.")
    parser.add_argument(
        "--report_roots",
        nargs="+",
        default=[
            r"G:\项目备份\vggt_小感度不起作用\vggt\infer_out",
            r"G:\项目备份\vggt原版60°相机推理结果\infer_out",
        ],
        help="Roots to scan for report.json files.",
    )
    parser.add_argument(
        "--local_zju_root",
        type=str,
        default=r"G:\数据集\datasets\ZJU_MoCap\data\zju_mocap",
        help="Local ZJU root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"G:\项目备份\vggt_小感度不起作用\vggt\model.pt",
        help="Checkpoint used for all runs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/geometry_baseline_zju_batch",
        help="Output root for per-case runs and batch summary.",
    )
    parser.add_argument(
        "--include_smoke",
        action="store_true",
        help="Include smoke report cases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def discover_reports(roots, include_smoke):
    candidates = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for report_path in root_path.rglob("report.json"):
            text = str(report_path).lower()
            if (not include_smoke) and ("smoke" in text):
                continue
            candidates.append(report_path)
    return sorted(candidates)


def normalize_profile(report_path, payload):
    meta = payload.get("meta", {})
    profile = str(meta.get("view_profile", "")).strip()
    if profile:
        return profile
    parts = report_path.parts
    if "infer_out" in parts:
        idx = parts.index("infer_out")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown_profile"


def dedupe_reports(report_paths):
    def preference_key(path):
        path_text = path.as_posix().lower()
        return (
            1 if "vggt_raw_viewcount" in path_text else 0,
            int(path.stat().st_mtime),
            path_text,
        )

    deduped = {}
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        source_cameras = tuple(meta.get("src_cameras", []))
        key = (
            str(meta.get("seq_name", "")),
            int(meta.get("frame_id", -1)),
            str(meta.get("tgt_camera", "")),
            source_cameras,
        )
        current = deduped.get(key)
        if current is None or preference_key(report_path) > preference_key(current):
            deduped[key] = report_path
    return list(sorted(deduped.values()))


def run_case(report_path, args):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    meta = payload["meta"]
    profile = normalize_profile(report_path, payload)
    case_dir = Path(args.output_root) / f"{meta['seq_name']}_frame_{int(meta['frame_id']):06d}_{meta['tgt_camera']}_{profile}"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--report_json",
        str(report_path),
        "--local_zju_root",
        str(args.local_zju_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output_dir",
        str(case_dir),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
    ]
    print(f"[zju-batch] case={case_dir.name}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path = case_dir / "batch_run.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + ("\n" if proc.stdout else "") + (proc.stderr or ""), encoding="utf-8")
    result = {
        "report_json": str(report_path),
        "case_dir": str(case_dir),
        "returncode": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "error": "",
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "").strip().splitlines()[-1] if (proc.stderr or proc.stdout) else ""
        return result
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        result["status"] = "failed"
        result["error"] = "summary.json missing"
        return result
    result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return result


def write_markdown(path, rows, failures):
    depth_wins = [row for row in rows if row["decision"] == "depth_unproject"]
    point_wins = [row for row in rows if row["decision"] == "point_map"]
    ties = [row for row in rows if row["decision"] == "tie"]
    lines = [
        "# ZJU Geometry Baseline Batch",
        "",
        f"- runs: `{len(rows)}`",
        f"- depth_unproject_wins: `{len(depth_wins)}`",
        f"- point_map_wins: `{len(point_wins)}`",
        f"- ties: `{len(ties)}`",
        "",
        "| Profile | Sources | Decision | MAE Winner | Cov Winner | Point MAE | Depth MAE | Point Cov | Depth Cov | Summary |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {view_profile} | {source_count} | {decision} | {mae_winner} | {coverage_winner} | {point_mae:.4f} | {depth_mae:.4f} | {point_cov:.4f} | {depth_cov:.4f} | `{summary_md}` |".format(
                **row
            )
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['report_json']}`: {failure['error'] or 'unknown error'}")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `depth + camera` is the decision winner when it has lower MAE and no worse coverage.",
            "- A `tie` means one branch won MAE while the other won coverage.",
            "- On the human-domain cases, if `depth + camera` keeps winning or staying competitive, the geometry-first route remains justified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = discover_reports(args.report_roots, args.include_smoke)
    reports = dedupe_reports(reports)
    print(f"[zju-batch] discovered={len(reports)}", flush=True)

    results = [run_case(report, args) for report in reports]
    failures = [result for result in results if result["status"] != "ok"]
    rows = []
    for result in results:
        if result["status"] != "ok":
            continue
        summary = result["summary"]
        rows.append(
            {
                "view_profile": summary["case"]["view_profile"],
                "source_count": summary["case"]["source_count"],
                "decision": summary["decision"]["decision"],
                "mae_winner": summary["decision"]["mae_winner"],
                "coverage_winner": summary["decision"]["coverage_winner"],
                "point_mae": summary["branches"]["point_map"]["metrics"]["mae"],
                "depth_mae": summary["branches"]["depth_unproject"]["metrics"]["mae"],
                "point_cov": summary["branches"]["point_map"]["render"]["coverage_ratio"],
                "depth_cov": summary["branches"]["depth_unproject"]["render"]["coverage_ratio"],
                "summary_md": str(Path(result["case_dir"]) / "summary.md"),
                "report_json": result["report_json"],
            }
        )

    batch_json = output_root / "batch_summary.json"
    batch_md = output_root / "batch_summary.md"
    batch_json.write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(batch_md, rows, failures)
    print(batch_md, flush=True)


if __name__ == "__main__":
    main()
