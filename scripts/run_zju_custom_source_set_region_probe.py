import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_geometry_branches_zju_report.py"
REGION_ORDER = ("fg_human", "fg_edge", "bg_far", "bg_bottom_band")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run region diagnostics for custom source-camera variants on one ZJU case."
    )
    parser.add_argument("--report_json", type=Path, required=True, help="Template report.json for the target case.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for this probe run.")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant definition: variant_name=Camera_B1,Camera_B2,...",
    )
    parser.add_argument(
        "--reference_variant",
        type=str,
        default="",
        help="Variant name used as the comparison reference. Defaults to the first variant.",
    )
    parser.add_argument("--python_exe", type=str, default="")
    parser.add_argument("--local_zju_root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument("--export_max_points", type=int, default=100000)
    parser.add_argument("--render_max_points", type=int, default=750000)
    parser.add_argument("--z_tolerance", type=float, default=0.02)
    parser.add_argument("--min_conf", type=float, default=1e-6)
    parser.add_argument(
        "--primary_branch",
        type=str,
        default="depth_unproject",
        choices=["depth_unproject", "point_map", "auto"],
    )
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def detect_checkpoint() -> Path:
    candidates = [
        REPO_ROOT / "model.pt",
        Path("G:/项目备份/Redo_viewpoints_at_60°_intervals_add_random_perturbations_vggt/vggt/model.pt"),
        Path("G:/项目备份/vggt_小感度不起作用/vggt/model.pt"),
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


def resolve_local_zju_root(args_root: Path | None, report_payload: dict) -> Path:
    if args_root is not None and args_root.is_dir():
        return args_root.resolve()
    meta_root = Path(str(report_payload["meta"].get("zju_root", "")))
    if meta_root.is_dir():
        return meta_root.resolve()
    raise FileNotFoundError("Could not resolve local ZJU root; pass --local_zju_root.")


def parse_variant(text: str) -> tuple[str, list[str]]:
    name, sep, raw_sources = str(text).partition("=")
    if not sep:
        raise ValueError(f"Invalid --variant value: {text!r}")
    variant_name = name.strip()
    source_cameras = [item.strip() for item in raw_sources.split(",") if item.strip()]
    if not variant_name or not source_cameras:
        raise ValueError(f"Invalid --variant value: {text!r}")
    return variant_name, source_cameras


def write_variant_report(path: Path, template_payload: dict, variant_name: str, source_cameras: list[str]) -> None:
    payload = json.loads(json.dumps(template_payload))
    meta = payload["meta"]
    meta["src_cameras"] = list(source_cameras)
    meta["num_src_views_actual"] = int(len(source_cameras))
    meta["num_total_cams"] = int(len(source_cameras) + 1)
    frame_stem = f"{int(meta['frame_id']):06d}.jpg"
    zju_root = str(meta.get("zju_root", "")).replace("\\", "/").rstrip("/")
    seq_name = str(meta["seq_name"])
    meta["src_image_paths"] = [f"{zju_root}/{seq_name}/{camera}/{frame_stem}" for camera in source_cameras]
    meta["variant_tag"] = variant_name
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
        summary_payload.get("region_diagnostics", {})
        .get("branches", {})
        .get(branch_name, {})
        .get("regions", {})
        .get(region_name, {})
        .get("render_metrics", {})
        .get(metric_key)
    )


def extract_region_coverage(summary_payload: dict, region_name: str, branch_name: str):
    return (
        summary_payload.get("region_diagnostics", {})
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
        result[f"{region_name}_winner"] = region_payload.get("comparison", {}).get(region_name, {}).get("mae_winner", "n/a")
    return result


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def build_variant_row(variant_name: str, source_cameras: list[str], metrics: dict, ref_name: str, ref_sources: list[str], ref_metrics: dict) -> dict:
    row = {
        "variant_name": variant_name,
        "source_cameras": list(source_cameras),
        "decision": metrics["decision"],
        "added_sources_vs_reference": [camera for camera in source_cameras if camera not in ref_sources],
        "removed_sources_vs_reference": [camera for camera in ref_sources if camera not in source_cameras],
        "full_depth_minus_point_mae": metrics["full_depth_minus_point_mae"],
        "full_depth_minus_point_cov": metrics["full_depth_minus_point_cov"],
        "reference_variant": ref_name,
        "full_depth_minus_point_mae_delta_vs_reference": metrics["full_depth_minus_point_mae"] - ref_metrics["full_depth_minus_point_mae"],
        "full_depth_minus_point_cov_delta_vs_reference": metrics["full_depth_minus_point_cov"] - ref_metrics["full_depth_minus_point_cov"],
    }
    for region_name in REGION_ORDER:
        key_mae = f"{region_name}_depth_minus_point_mae"
        key_cov = f"{region_name}_depth_minus_point_cov"
        row[key_mae] = metrics[key_mae]
        row[key_cov] = metrics[key_cov]
        ref_mae = ref_metrics[key_mae]
        ref_cov = ref_metrics[key_cov]
        row[f"{key_mae}_delta_vs_reference"] = None if row[key_mae] is None or ref_mae is None else row[key_mae] - ref_mae
        row[f"{key_cov}_delta_vs_reference"] = None if row[key_cov] is None or ref_cov is None else row[key_cov] - ref_cov
        row[f"{region_name}_winner"] = metrics[f"{region_name}_winner"]
    return row


def write_markdown(path: Path, summary_payload: dict) -> None:
    case = summary_payload["case"]
    reference = summary_payload["reference"]
    rows = summary_payload["rows"]
    lines = [
        f"# Custom Source-Set Region Probe: {case['seq_name']} / frame {case['frame_id']} / {case['target_camera']}",
        "",
        f"- view_profile: `{case['view_profile']}`",
        f"- reference_variant: `{reference['variant_name']}`",
        f"- reference_sources: `{','.join(reference['source_cameras'])}`",
        f"- reference_full_depth_minus_point_mae: `{fmt(reference['metrics']['full_depth_minus_point_mae'])}`",
        f"- reference_fg_human_depth_minus_point_mae: `{fmt(reference['metrics']['fg_human_depth_minus_point_mae'])}`",
        f"- reference_bg_far_depth_minus_point_mae: `{fmt(reference['metrics']['bg_far_depth_minus_point_mae'])}`",
        f"- reference_bg_bottom_band_depth_minus_point_mae: `{fmt(reference['metrics']['bg_bottom_band_depth_minus_point_mae'])}`",
        "",
        "## Variant Ranking",
        "",
        "| Variant | Added vs Ref | Removed vs Ref | Decision | Full Delta vs Ref | fg_human Delta vs Ref | fg_edge Delta vs Ref | bg_far Delta vs Ref | bg_bottom Delta vs Ref |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{variant}` | `{added}` | `{removed}` | `{decision}` | {full_delta} | {fg_delta} | {edge_delta} | {far_delta} | {bottom_delta} |".format(
                variant=row["variant_name"],
                added=",".join(row["added_sources_vs_reference"]) or "-",
                removed=",".join(row["removed_sources_vs_reference"]) or "-",
                decision=row["decision"],
                full_delta=fmt(row["full_depth_minus_point_mae_delta_vs_reference"]),
                fg_delta=fmt(row["fg_human_depth_minus_point_mae_delta_vs_reference"]),
                edge_delta=fmt(row["fg_edge_depth_minus_point_mae_delta_vs_reference"]),
                far_delta=fmt(row["bg_far_depth_minus_point_mae_delta_vs_reference"]),
                bottom_delta=fmt(row["bg_bottom_band_depth_minus_point_mae_delta_vs_reference"]),
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Negative deltas mean the variant makes `depth_unproject` relatively better than the reference variant on that metric.",
            "- This probe is intended to test narrow source-set refinements before any new training or cloud action.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    template_payload = load_json(args.report_json.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else detect_checkpoint()
    local_zju_root = resolve_local_zju_root(args.local_zju_root, template_payload)
    python_exe = resolve_python_executable(args.python_exe)

    variants = []
    for raw_variant in args.variant:
        name, source_cameras = parse_variant(raw_variant)
        variants.append((name, source_cameras))
    variant_names = [name for name, _ in variants]
    reference_variant = args.reference_variant or variant_names[0]
    if reference_variant not in variant_names:
        raise ValueError(f"reference_variant {reference_variant!r} not found in variants {variant_names}")

    variant_metrics = {}
    variant_sources = {}
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, source_cameras in variants:
        variant_report = report_dir / f"{variant_name}.json"
        variant_out = output_dir / variant_name
        write_variant_report(variant_report, template_payload, variant_name, source_cameras)
        run_compare(
            python_exe=python_exe,
            report_json=variant_report,
            output_dir=variant_out,
            local_zju_root=local_zju_root,
            checkpoint=checkpoint,
            args=args,
        )
        variant_payload = load_json(variant_out / "summary.json")
        variant_metrics[variant_name] = extract_case_metrics(variant_payload)
        variant_sources[variant_name] = list(source_cameras)

    ref_metrics = variant_metrics[reference_variant]
    ref_sources = variant_sources[reference_variant]
    rows = [
        build_variant_row(name, variant_sources[name], variant_metrics[name], reference_variant, ref_sources, ref_metrics)
        for name in variant_names
    ]
    rows.sort(
        key=lambda item: (
            float("inf") if item["bg_bottom_band_depth_minus_point_mae_delta_vs_reference"] is None else item["bg_bottom_band_depth_minus_point_mae_delta_vs_reference"],
            float("inf") if item["full_depth_minus_point_mae_delta_vs_reference"] is None else item["full_depth_minus_point_mae_delta_vs_reference"],
        )
    )

    meta = template_payload["meta"]
    summary_payload = {
        "case": {
            "seq_name": meta["seq_name"],
            "frame_id": int(meta["frame_id"]),
            "view_profile": meta["view_profile"],
            "target_camera": meta["tgt_camera"],
        },
        "reference": {
            "variant_name": reference_variant,
            "source_cameras": ref_sources,
            "metrics": ref_metrics,
        },
        "rows": rows,
    }
    save_json(output_dir / "summary.json", summary_payload)
    write_markdown(output_dir / "summary.md", summary_payload)
    print(f"[done] Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
