import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_geometry_branches_zju_report.py"

REGION_ORDER = ("fg_human", "bg_far", "bg_bottom_band")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-source-out region diagnostics for ZJU geometry cases."
    )
    parser.add_argument(
        "--report_json",
        nargs="+",
        required=True,
        help="One or more synthetic/original report.json files defining the baseline source sets.",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_exe", type=str, default="")
    parser.add_argument("--local_zju_root", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument("--export_max_points", type=int, default=100000)
    parser.add_argument("--render_max_points", type=int, default=750000)
    parser.add_argument("--z_tolerance", type=float, default=0.02)
    parser.add_argument("--min_conf", type=float, default=1e-6)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--primary_branch",
        type=str,
        default="depth_unproject",
        choices=["depth_unproject", "point_map", "auto"],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def detect_local_zju_root() -> Path:
    candidates = [
        Path("G:/数据集/datasets/ZJU_MoCap/data/zju_mocap"),
        Path("G:/项目备份/Redo_viewpoints_at_60°_intervals_add_random_perturbations_vggt/datasets/ZJU_MoCap/data/zju_mocap"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not auto-detect local ZJU root; pass --local_zju_root.")


def detect_checkpoint() -> Path:
    candidates = [
        Path("G:/项目备份/Redo_viewpoints_at_60°_intervals_add_random_perturbations_vggt/vggt/model.pt"),
        Path("G:/项目备份/vggt_小感度不起作用/vggt/model.pt"),
        REPO_ROOT / "model.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not auto-detect checkpoint; pass --checkpoint.")


def resolve_python_executable(requested: str) -> str:
    requested = str(requested).strip()
    if requested:
        requested_path = Path(requested)
        if requested_path.is_file():
            return str(requested_path.resolve())
        return requested

    candidates = [
        REPO_ROOT / ".venv5080" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return sys.executable


def load_ring_order(seq_dir: Path) -> list[str]:
    annots = np.load(seq_dir / "annots.npy", allow_pickle=True).item()
    rows = []
    for index, (rotation, translation) in enumerate(zip(annots["cams"]["R"], annots["cams"]["T"]), start=1):
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        center = (-rotation.T @ translation).reshape(3)
        azimuth = float(np.degrees(np.arctan2(center[0], center[2])))
        rows.append((f"Camera_B{index}", azimuth))
    rows.sort(key=lambda item: item[1])
    return [camera for camera, _ in rows]


def signed_ring_offset(source_camera: str, target_camera: str, ring_order: list[str]) -> int:
    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    ring_len = len(ring_order)
    raw = (camera_to_index[source_camera] - camera_to_index[target_camera]) % ring_len
    if raw > ring_len / 2:
        raw -= ring_len
    return int(raw)


def shortest_ring_distance(source_camera: str, target_camera: str, ring_order: list[str]) -> int:
    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    ring_len = len(ring_order)
    raw = abs(camera_to_index[source_camera] - camera_to_index[target_camera])
    return int(min(raw, ring_len - raw))


def write_variant_report(path: Path, template_payload: dict, kept_sources: list[str], dropped_source: str | None) -> None:
    payload = json.loads(json.dumps(template_payload))
    meta = payload["meta"]
    meta["src_cameras"] = list(kept_sources)
    meta["num_src_views_actual"] = int(len(kept_sources))
    meta["num_total_cams"] = int(len(kept_sources) + 1)
    frame_stem = f"{int(meta['frame_id']):06d}.jpg"
    zju_root = str(meta.get("zju_root", "")).replace("\\", "/").rstrip("/")
    seq_name = str(meta["seq_name"])
    meta["src_image_paths"] = [f"{zju_root}/{seq_name}/{camera}/{frame_stem}" for camera in kept_sources]
    meta["variant_tag"] = "baseline" if dropped_source is None else f"drop_{dropped_source}"
    save_json(path, payload)


def run_compare(
    *,
    python_exe: str,
    report_json: Path,
    output_dir: Path,
    local_zju_root: Path,
    checkpoint: Path,
    args,
) -> None:
    summary_path = output_dir / "summary.json"
    if args.skip_existing and summary_path.is_file():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(COMPARE_SCRIPT),
        "--report_json",
        str(report_json),
        "--output_dir",
        str(output_dir),
        "--local_zju_root",
        str(local_zju_root),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--conf_percentile",
        str(args.conf_percentile),
        "--export_max_points",
        str(args.export_max_points),
        "--render_max_points",
        str(args.render_max_points),
        "--z_tolerance",
        str(args.z_tolerance),
        "--min_conf",
        str(args.min_conf),
        "--primary_branch",
        args.primary_branch,
        "--skip_save_predictions",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def extract_region_value(summary_payload: dict, region_name: str, branch_name: str, metric_key: str):
    return (
        summary_payload
        .get("region_diagnostics", {})
        .get("branches", {})
        .get(branch_name, {})
        .get("regions", {})
        .get(region_name, {})
        .get("render_metrics", {})
        .get(metric_key)
    )


def extract_region_coverage(summary_payload: dict, region_name: str, branch_name: str):
    return (
        summary_payload
        .get("region_diagnostics", {})
        .get("branches", {})
        .get(branch_name, {})
        .get("regions", {})
        .get(region_name, {})
        .get("coverage_ratio")
    )


def extract_case_metrics(summary_payload: dict) -> dict:
    branches = summary_payload["branches"]
    region_payload = summary_payload.get("region_diagnostics", {})
    result = {
        "decision": str(summary_payload.get("decision", {}).get("decision", "n/a")),
        "full_depth_minus_point_mae": float(branches["depth_unproject"]["metrics"]["mae"]) - float(branches["point_map"]["metrics"]["mae"]),
        "full_depth_minus_point_cov": float(branches["depth_unproject"]["render"]["coverage_ratio"]) - float(branches["point_map"]["render"]["coverage_ratio"]),
    }
    for region_name in REGION_ORDER:
        depth_mae = extract_region_value(summary_payload, region_name, "depth_unproject", "mae")
        point_mae = extract_region_value(summary_payload, region_name, "point_map", "mae")
        depth_cov = extract_region_coverage(summary_payload, region_name, "depth_unproject")
        point_cov = extract_region_coverage(summary_payload, region_name, "point_map")
        result[f"{region_name}_depth_minus_point_mae"] = None if depth_mae is None or point_mae is None else float(depth_mae) - float(point_mae)
        result[f"{region_name}_depth_minus_point_cov"] = None if depth_cov is None or point_cov is None else float(depth_cov) - float(point_cov)
        result[f"{region_name}_winner"] = (
            region_payload.get("comparison", {}).get(region_name, {}).get("mae_winner", "n/a")
        )
    return result


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def build_case_summary(
    *,
    case_meta: dict,
    ring_order: list[str],
    baseline_payload: dict,
    variant_payloads: list[tuple[str, dict]],
) -> dict:
    baseline_metrics = extract_case_metrics(baseline_payload)
    rows = []
    for dropped_source, payload in variant_payloads:
        variant_metrics = extract_case_metrics(payload)
        row = {
            "dropped_source": dropped_source,
            "ring_distance_to_target": shortest_ring_distance(dropped_source, case_meta["tgt_camera"], ring_order),
            "signed_ring_offset": signed_ring_offset(dropped_source, case_meta["tgt_camera"], ring_order),
            "decision": variant_metrics["decision"],
            "full_depth_minus_point_mae": variant_metrics["full_depth_minus_point_mae"],
            "full_depth_minus_point_mae_delta_vs_baseline": variant_metrics["full_depth_minus_point_mae"] - baseline_metrics["full_depth_minus_point_mae"],
            "full_depth_minus_point_cov": variant_metrics["full_depth_minus_point_cov"],
            "full_depth_minus_point_cov_delta_vs_baseline": variant_metrics["full_depth_minus_point_cov"] - baseline_metrics["full_depth_minus_point_cov"],
        }
        for region_name in REGION_ORDER:
            key_mae = f"{region_name}_depth_minus_point_mae"
            key_cov = f"{region_name}_depth_minus_point_cov"
            row[key_mae] = variant_metrics[key_mae]
            baseline_mae = baseline_metrics[key_mae]
            row[f"{key_mae}_delta_vs_baseline"] = None if row[key_mae] is None or baseline_mae is None else row[key_mae] - baseline_mae
            row[key_cov] = variant_metrics[key_cov]
            baseline_cov = baseline_metrics[key_cov]
            row[f"{key_cov}_delta_vs_baseline"] = None if row[key_cov] is None or baseline_cov is None else row[key_cov] - baseline_cov
            row[f"{region_name}_winner"] = variant_metrics[f"{region_name}_winner"]
        rows.append(row)

    rows.sort(
        key=lambda item: (
            float("inf") if item["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"] is None else item["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"],
            float("inf") if item["full_depth_minus_point_mae_delta_vs_baseline"] is None else item["full_depth_minus_point_mae_delta_vs_baseline"],
        )
    )

    improved_bottom = [
        row for row in rows
        if row["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"] is not None
        and row["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"] < 0.0
    ]
    improved_bottom_and_full = [
        row for row in improved_bottom
        if row["full_depth_minus_point_mae_delta_vs_baseline"] is not None
        and row["full_depth_minus_point_mae_delta_vs_baseline"] < 0.0
    ]
    return {
        "case": {
            "seq_name": case_meta["seq_name"],
            "frame_id": int(case_meta["frame_id"]),
            "view_profile": case_meta["view_profile"],
            "target_camera": case_meta["tgt_camera"],
            "source_cameras": list(case_meta["src_cameras"]),
        },
        "baseline": baseline_metrics,
        "rows": rows,
        "aggregate": {
            "drop_variants": len(rows),
            "improved_bottom_band_mae_variants": len(improved_bottom),
            "improved_bottom_band_and_full_mae_variants": len(improved_bottom_and_full),
            "best_bottom_band_variant": None if not rows else rows[0]["dropped_source"],
            "best_bottom_band_delta": None if not rows else rows[0]["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"],
        },
    }


def write_case_markdown(path: Path, payload: dict) -> None:
    case = payload["case"]
    baseline = payload["baseline"]
    rows = payload["rows"]
    lines = [
        f"# Leave-One-Out Region Diagnostics: {case['seq_name']} / frame {case['frame_id']} / {case['target_camera']}",
        "",
        f"- view_profile: `{case['view_profile']}`",
        f"- baseline source_cameras: `{','.join(case['source_cameras'])}`",
        f"- baseline full depth-point MAE: `{fmt(baseline['full_depth_minus_point_mae'])}`",
        f"- baseline full depth-point coverage: `{fmt(baseline['full_depth_minus_point_cov'])}`",
        f"- baseline bg_bottom_band depth-point MAE: `{fmt(baseline['bg_bottom_band_depth_minus_point_mae'])}`",
        f"- baseline bg_bottom_band depth-point coverage: `{fmt(baseline['bg_bottom_band_depth_minus_point_cov'])}`",
        "",
        "## Drop Ranking",
        "",
        "| Dropped Source | Ring Dist | Signed Offset | Full Delta Change | fg_human Change | bg_far Change | bg_bottom Change | bg_bottom Cov Change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{dropped}` | {dist} | {offset} | {full_delta} | {fg_delta} | {bg_far_delta} | {bg_bottom_delta} | {bg_bottom_cov_delta} |".format(
                dropped=row["dropped_source"],
                dist=row["ring_distance_to_target"],
                offset=row["signed_ring_offset"],
                full_delta=fmt(row["full_depth_minus_point_mae_delta_vs_baseline"]),
                fg_delta=fmt(row["fg_human_depth_minus_point_mae_delta_vs_baseline"]),
                bg_far_delta=fmt(row["bg_far_depth_minus_point_mae_delta_vs_baseline"]),
                bg_bottom_delta=fmt(row["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"]),
                bg_bottom_cov_delta=fmt(row["bg_bottom_band_depth_minus_point_cov_delta_vs_baseline"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# ZJU Source Leave-One-Out Region Diagnostics",
        "",
        f"- cases: `{payload['aggregate']['cases']}`",
        f"- drop_variants: `{payload['aggregate']['drop_variants']}`",
        f"- improved_bottom_band_mae_variants: `{payload['aggregate']['improved_bottom_band_mae_variants']}`",
        f"- improved_bottom_band_and_full_mae_variants: `{payload['aggregate']['improved_bottom_band_and_full_mae_variants']}`",
        "",
        "## Cases",
        "",
        "| Case | Baseline bg_bottom | Best Drop | Best bg_bottom Change | Best Full Change |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for case in payload["cases"]:
        best = case["rows"][0] if case["rows"] else None
        lines.append(
            "| `{case_id}` | {baseline_bottom} | {best_drop} | {best_bottom_delta} | {best_full_delta} |".format(
                case_id=f"{case['case']['view_profile']} / {case['case']['target_camera']}",
                baseline_bottom=fmt(case["baseline"]["bg_bottom_band_depth_minus_point_mae"]),
                best_drop="n/a" if best is None else best["dropped_source"],
                best_bottom_delta="n/a" if best is None else fmt(best["bg_bottom_band_depth_minus_point_mae_delta_vs_baseline"]),
                best_full_delta="n/a" if best is None else fmt(best["full_depth_minus_point_mae_delta_vs_baseline"]),
            )
        )
    lines.extend(
        [
            "",
            "## Overall Readout",
            "",
            "- Negative `bg_bottom_band` delta means dropping that source camera makes `depth_unproject` relatively better than `point_map` in the bottom band.",
            "- Negative full-frame delta means the same drop also helps the whole image, not just the bottom band.",
            "- This report is intended to tell us whether the next move should be source-camera policy refinement before any cloud run.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    local_zju_root = Path(args.local_zju_root).resolve() if args.local_zju_root else detect_local_zju_root()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else detect_checkpoint()
    python_exe = resolve_python_executable(args.python_exe)

    batch_cases = []
    aggregate_drop_variants = 0
    aggregate_improved_bottom = 0
    aggregate_improved_bottom_and_full = 0

    for report_path_raw in args.report_json:
        report_path = Path(report_path_raw).resolve()
        report_payload = load_json(report_path)
        case_meta = report_payload["meta"]
        case_id = f"{case_meta['seq_name']}_frame_{int(case_meta['frame_id']):06d}_{case_meta['tgt_camera']}_{case_meta['view_profile']}"
        case_root = output_root / case_id
        report_out_dir = case_root / "reports"
        baseline_report = report_out_dir / "baseline_report.json"
        write_variant_report(baseline_report, report_payload, list(case_meta["src_cameras"]), dropped_source=None)

        baseline_out = case_root / "baseline"
        run_compare(
            python_exe=python_exe,
            report_json=baseline_report,
            output_dir=baseline_out,
            local_zju_root=local_zju_root,
            checkpoint=checkpoint,
            args=args,
        )
        baseline_payload = load_json(baseline_out / "summary.json")

        ring_order = load_ring_order(local_zju_root / str(case_meta["seq_name"]))
        variant_payloads = []
        for dropped_source in case_meta["src_cameras"]:
            kept_sources = [camera for camera in case_meta["src_cameras"] if camera != dropped_source]
            variant_report = report_out_dir / f"drop_{dropped_source}.json"
            write_variant_report(variant_report, report_payload, kept_sources, dropped_source=dropped_source)
            variant_out = case_root / f"drop_{dropped_source}"
            run_compare(
                python_exe=python_exe,
                report_json=variant_report,
                output_dir=variant_out,
                local_zju_root=local_zju_root,
                checkpoint=checkpoint,
                args=args,
            )
            variant_payloads.append((str(dropped_source), load_json(variant_out / "summary.json")))

        case_summary = build_case_summary(
            case_meta=case_meta,
            ring_order=ring_order,
            baseline_payload=baseline_payload,
            variant_payloads=variant_payloads,
        )
        save_json(case_root / "summary.json", case_summary)
        write_case_markdown(case_root / "summary.md", case_summary)
        batch_cases.append(case_summary)
        aggregate_drop_variants += int(case_summary["aggregate"]["drop_variants"])
        aggregate_improved_bottom += int(case_summary["aggregate"]["improved_bottom_band_mae_variants"])
        aggregate_improved_bottom_and_full += int(case_summary["aggregate"]["improved_bottom_band_and_full_mae_variants"])

    batch_payload = {
        "aggregate": {
            "cases": len(batch_cases),
            "drop_variants": aggregate_drop_variants,
            "improved_bottom_band_mae_variants": aggregate_improved_bottom,
            "improved_bottom_band_and_full_mae_variants": aggregate_improved_bottom_and_full,
        },
        "cases": batch_cases,
    }
    save_json(output_root / "summary.json", batch_payload)
    write_batch_markdown(output_root / "summary.md", batch_payload)
    print(f"[done] Wrote {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
