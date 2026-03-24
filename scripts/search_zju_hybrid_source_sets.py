import argparse
import itertools
import json
from pathlib import Path

from run_zju_custom_source_set_region_probe import (
    detect_checkpoint,
    extract_case_metrics,
    load_json,
    resolve_local_zju_root,
    resolve_python_executable,
    run_compare,
    save_json,
    write_variant_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate and rank narrow local source-set hybrids for one ZJU target case."
    )
    parser.add_argument("--report_json", type=Path, required=True, help="Template report.json for the target case.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for this search run.")
    parser.add_argument(
        "--reference_sources",
        type=str,
        required=True,
        help="Comma-separated reference source cameras to preserve unless swapped out.",
    )
    parser.add_argument(
        "--candidate_pool",
        type=str,
        default="",
        help="Comma-separated source-camera pool to search. Defaults to reference union report src_cameras.",
    )
    parser.add_argument(
        "--reference_variant",
        type=str,
        default="uniform",
        help="Variant name used for the reference source set.",
    )
    parser.add_argument(
        "--max_swaps",
        type=int,
        default=2,
        help="Maximum number of cameras to swap out of the reference source set.",
    )
    parser.add_argument(
        "--must_include",
        action="append",
        default=[],
        help="Camera that must be present in every searched candidate. Repeatable.",
    )
    parser.add_argument(
        "--must_exclude",
        action="append",
        default=[],
        help="Camera that must be absent from every searched candidate. Repeatable.",
    )
    parser.add_argument(
        "--fg_human_guard_max_delta",
        type=float,
        default=0.001,
        help="Maximum allowed fg_human MAE delta vs reference to count as guard-pass.",
    )
    parser.add_argument(
        "--top_k_markdown",
        type=int,
        default=20,
        help="How many ranked candidates to print in the markdown summary.",
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


def parse_camera_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def sorted_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def validate_reference_sources(reference_sources: list[str]) -> list[str]:
    if not reference_sources:
        raise ValueError("reference_sources must not be empty")
    if len(reference_sources) != len(set(reference_sources)):
        raise ValueError(f"reference_sources contains duplicates: {reference_sources}")
    return reference_sources


def build_candidate_pool(reference_sources: list[str], template_payload: dict, candidate_pool_arg: str) -> list[str]:
    report_sources = [str(item) for item in template_payload["meta"].get("src_cameras", [])]
    if candidate_pool_arg.strip():
        pool = parse_camera_csv(candidate_pool_arg)
    else:
        pool = list(reference_sources) + report_sources
    pool = sorted_unique(pool)
    missing_reference = [camera for camera in reference_sources if camera not in pool]
    if missing_reference:
        raise ValueError(f"candidate_pool is missing reference cameras: {missing_reference}")
    return pool


def generate_variant_specs(
    reference_sources: list[str],
    candidate_pool: list[str],
    max_swaps: int,
    must_include: list[str],
    must_exclude: list[str],
    reference_variant_name: str,
) -> list[dict]:
    reference_set = set(reference_sources)
    extra_candidates = [camera for camera in candidate_pool if camera not in reference_set]
    variant_specs = []
    seen_sets = set()

    reference_key = tuple(reference_sources)
    seen_sets.add(reference_key)
    variant_specs.append(
        {
            "variant_name": reference_variant_name,
            "source_cameras": list(reference_sources),
            "added_vs_reference": [],
            "removed_vs_reference": [],
            "swap_count": 0,
        }
    )

    for swap_count in range(1, max_swaps + 1):
        for removed_sources in itertools.combinations(reference_sources, swap_count):
            kept_sources = [camera for camera in reference_sources if camera not in removed_sources]
            for added_sources in itertools.combinations(extra_candidates, swap_count):
                candidate_sources = kept_sources + list(added_sources)
                candidate_set = set(candidate_sources)
                if any(camera not in candidate_set for camera in must_include):
                    continue
                if any(camera in candidate_set for camera in must_exclude):
                    continue
                key = tuple(candidate_sources)
                if key in seen_sets:
                    continue
                seen_sets.add(key)
                variant_name = f"s{swap_count}_{len(variant_specs):03d}"
                variant_specs.append(
                    {
                        "variant_name": variant_name,
                        "source_cameras": candidate_sources,
                        "added_vs_reference": list(added_sources),
                        "removed_vs_reference": list(removed_sources),
                        "swap_count": swap_count,
                    }
                )
    return variant_specs


def decision_rank(decision: str) -> int:
    if decision == "depth_unproject":
        return 0
    if decision == "tie":
        return 1
    return 2


def rank_rows(rows: list[dict], fg_human_guard_max_delta: float) -> list[dict]:
    def row_key(item: dict):
        fg_delta = item.get("fg_human_depth_minus_point_mae_delta_vs_reference")
        bottom_delta = item.get("bg_bottom_band_depth_minus_point_mae_delta_vs_reference")
        full_delta = item.get("full_depth_minus_point_mae_delta_vs_reference")
        far_delta = item.get("bg_far_depth_minus_point_mae_delta_vs_reference")
        guard_fail = 0 if fg_delta is not None and fg_delta <= fg_human_guard_max_delta else 1
        return (
            1 if item["variant_name"] == item["reference_variant"] else 0,
            guard_fail,
            decision_rank(item["decision"]),
            float("inf") if bottom_delta is None else bottom_delta,
            float("inf") if full_delta is None else full_delta,
            float("inf") if far_delta is None else far_delta,
            float("inf") if fg_delta is None else fg_delta,
        )

    return sorted(rows, key=row_key)


def build_variant_row(
    spec: dict,
    metrics: dict,
    reference_variant: str,
    reference_metrics: dict,
    fg_human_guard_max_delta: float,
) -> dict:
    row = {
        "variant_name": spec["variant_name"],
        "source_cameras": list(spec["source_cameras"]),
        "decision": metrics["decision"],
        "swap_count": int(spec["swap_count"]),
        "added_sources_vs_reference": list(spec["added_vs_reference"]),
        "removed_sources_vs_reference": list(spec["removed_vs_reference"]),
        "reference_variant": reference_variant,
        "full_depth_minus_point_mae": metrics["full_depth_minus_point_mae"],
        "full_depth_minus_point_cov": metrics["full_depth_minus_point_cov"],
        "full_depth_minus_point_mae_delta_vs_reference": metrics["full_depth_minus_point_mae"] - reference_metrics["full_depth_minus_point_mae"],
        "full_depth_minus_point_cov_delta_vs_reference": metrics["full_depth_minus_point_cov"] - reference_metrics["full_depth_minus_point_cov"],
    }
    for region_name in ("fg_human", "fg_edge", "bg_far", "bg_bottom_band"):
        key_mae = f"{region_name}_depth_minus_point_mae"
        key_cov = f"{region_name}_depth_minus_point_cov"
        row[key_mae] = metrics[key_mae]
        row[key_cov] = metrics[key_cov]
        ref_mae = reference_metrics[key_mae]
        ref_cov = reference_metrics[key_cov]
        row[f"{key_mae}_delta_vs_reference"] = None if row[key_mae] is None or ref_mae is None else row[key_mae] - ref_mae
        row[f"{key_cov}_delta_vs_reference"] = None if row[key_cov] is None or ref_cov is None else row[key_cov] - ref_cov
        row[f"{region_name}_winner"] = metrics[f"{region_name}_winner"]
    fg_delta = row["fg_human_depth_minus_point_mae_delta_vs_reference"]
    row["fg_human_guard_pass"] = bool(fg_delta is not None and fg_delta <= fg_human_guard_max_delta)
    return row


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_markdown(path: Path, payload: dict) -> None:
    case = payload["case"]
    search = payload["search"]
    reference = payload["reference"]
    rows = payload["rows"][: search["top_k_markdown"]]
    lines = [
        f"# Hybrid Source-Set Search: {case['seq_name']} / frame {case['frame_id']} / {case['target_camera']}",
        "",
        f"- view_profile: `{case['view_profile']}`",
        f"- reference_variant: `{reference['variant_name']}`",
        f"- reference_sources: `{','.join(reference['source_cameras'])}`",
        f"- candidate_pool: `{','.join(search['candidate_pool'])}`",
        f"- max_swaps: `{search['max_swaps']}`",
        f"- fg_human_guard_max_delta: `{fmt(search['fg_human_guard_max_delta'])}`",
        f"- variant_count: `{search['variant_count']}`",
        f"- guard_pass_count: `{search['guard_pass_count']}`",
        "",
        "## Top Ranked Variants",
        "",
        "| Variant | Swaps | Guard | Added vs Ref | Removed vs Ref | Decision | Full Delta | fg_human Delta | bg_far Delta | bg_bottom Delta |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{variant}` | {swaps} | `{guard}` | `{added}` | `{removed}` | `{decision}` | {full_delta} | {fg_delta} | {far_delta} | {bottom_delta} |".format(
                variant=row["variant_name"],
                swaps=row["swap_count"],
                guard="pass" if row["fg_human_guard_pass"] else "fail",
                added=",".join(row["added_sources_vs_reference"]) or "-",
                removed=",".join(row["removed_sources_vs_reference"]) or "-",
                decision=row["decision"],
                full_delta=fmt(row["full_depth_minus_point_mae_delta_vs_reference"]),
                fg_delta=fmt(row["fg_human_depth_minus_point_mae_delta_vs_reference"]),
                far_delta=fmt(row["bg_far_depth_minus_point_mae_delta_vs_reference"]),
                bottom_delta=fmt(row["bg_bottom_band_depth_minus_point_mae_delta_vs_reference"]),
            )
        )

    best_row = payload.get("best_guard_pass_variant")
    lines.extend(["", "## Readout", ""])
    if best_row is None:
        lines.append("- No candidate satisfied the `fg_human` guard threshold in this search.")
    else:
        lines.append(
            "- Best guard-pass candidate: `{name}` with full delta `{full}`, fg_human delta `{fg}`, bg_far delta `{far}`, bg_bottom delta `{bottom}`, decision `{decision}`.".format(
                name=best_row["variant_name"],
                full=fmt(best_row["full_depth_minus_point_mae_delta_vs_reference"]),
                fg=fmt(best_row["fg_human_depth_minus_point_mae_delta_vs_reference"]),
                far=fmt(best_row["bg_far_depth_minus_point_mae_delta_vs_reference"]),
                bottom=fmt(best_row["bg_bottom_band_depth_minus_point_mae_delta_vs_reference"]),
                decision=best_row["decision"],
            )
        )
    lines.append("- Negative deltas mean the candidate makes `depth_unproject` relatively better than the reference variant on that metric.")
    lines.append("- Guard-pass means `fg_human` MAE delta vs reference is within the configured local tolerance.")
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

    reference_sources = validate_reference_sources(parse_camera_csv(args.reference_sources))
    candidate_pool = build_candidate_pool(reference_sources, template_payload, args.candidate_pool)
    must_include = sorted_unique([camera.strip() for camera in args.must_include if camera.strip()])
    must_exclude = sorted_unique([camera.strip() for camera in args.must_exclude if camera.strip()])

    variant_specs = generate_variant_specs(
        reference_sources=reference_sources,
        candidate_pool=candidate_pool,
        max_swaps=args.max_swaps,
        must_include=must_include,
        must_exclude=must_exclude,
        reference_variant_name=args.reference_variant,
    )

    metrics_by_variant = {}
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for spec in variant_specs:
        variant_report = report_dir / f"{spec['variant_name']}.json"
        variant_out = output_dir / spec["variant_name"]
        write_variant_report(variant_report, template_payload, spec["variant_name"], spec["source_cameras"])
        run_compare(
            python_exe=python_exe,
            report_json=variant_report,
            output_dir=variant_out,
            local_zju_root=local_zju_root,
            checkpoint=checkpoint,
            args=args,
        )
        metrics_by_variant[spec["variant_name"]] = extract_case_metrics(load_json(variant_out / "summary.json"))

    reference_metrics = metrics_by_variant[args.reference_variant]
    rows = [
        build_variant_row(
            spec=spec,
            metrics=metrics_by_variant[spec["variant_name"]],
            reference_variant=args.reference_variant,
            reference_metrics=reference_metrics,
            fg_human_guard_max_delta=args.fg_human_guard_max_delta,
        )
        for spec in variant_specs
    ]
    rows = rank_rows(rows, fg_human_guard_max_delta=args.fg_human_guard_max_delta)

    guard_pass_rows = [row for row in rows if row["variant_name"] != args.reference_variant and row["fg_human_guard_pass"]]
    best_guard_pass_variant = guard_pass_rows[0] if guard_pass_rows else None

    meta = template_payload["meta"]
    summary_payload = {
        "case": {
            "seq_name": str(meta["seq_name"]),
            "frame_id": int(meta["frame_id"]),
            "view_profile": str(meta["view_profile"]),
            "target_camera": str(meta["tgt_camera"]),
        },
        "reference": {
            "variant_name": args.reference_variant,
            "source_cameras": list(reference_sources),
            "metrics": reference_metrics,
        },
        "search": {
            "candidate_pool": list(candidate_pool),
            "max_swaps": int(args.max_swaps),
            "must_include": list(must_include),
            "must_exclude": list(must_exclude),
            "fg_human_guard_max_delta": float(args.fg_human_guard_max_delta),
            "variant_count": int(len(rows)),
            "guard_pass_count": int(len(guard_pass_rows)),
            "top_k_markdown": int(args.top_k_markdown),
        },
        "best_guard_pass_variant": best_guard_pass_variant,
        "rows": rows,
    }
    save_json(output_dir / "summary.json", summary_payload)
    write_markdown(output_dir / "summary.md", summary_payload)
    print(f"[done] Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
