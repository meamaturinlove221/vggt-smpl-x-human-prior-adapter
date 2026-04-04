import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic hero/benchmark manifests for teacher-fixed visual lift."
    )
    parser.add_argument("--sweep-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--hero-count", type=int, default=5)
    parser.add_argument("--benchmark-count", type=int, default=20)
    return parser.parse_args()


def _load_case_summary(case_dir: Path) -> dict:
    return json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))


def _case_payload(row: dict, case_dir: Path) -> dict:
    summary = _load_case_summary(case_dir)
    case = summary["case"]
    return {
        "case_id": "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(case["seq_name"]),
            frame=int(case["frame_id"]),
            target=str(case["target_camera"]),
        ),
        "seq_name": str(case["seq_name"]),
        "frame_id": int(case["frame_id"]),
        "target_camera": str(case["target_camera"]),
        "source_cameras": list(case["source_cameras"]),
        "source_count": int(case["source_count"]),
        "selection_reason": {
            "legacy_depth_mae": float(summary["branches"]["depth_unproject"]["metrics"]["mae"]),
            "legacy_depth_ssim": float(summary["branches"]["depth_unproject"]["metrics"]["ssim"]),
            "legacy_depth_coverage_ratio": float(summary["branches"]["depth_unproject"]["render"]["coverage_ratio"]),
            "legacy_point_mae": float(summary["branches"]["point_map"]["metrics"]["mae"]),
            "legacy_point_ssim": float(summary["branches"]["point_map"]["metrics"]["ssim"]),
        },
        "legacy_case_dir": str(case_dir),
    }


def main() -> int:
    args = parse_args()
    sweep_summary_path = Path(args.sweep_summary_json).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(sweep_summary_path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    rows = [row for row in rows if str(row.get("view_profile")) == "6src_hist" and int(row.get("source_count", 0)) == 6]
    rows.sort(
        key=lambda row: (
            -float(row["depth_mae"]),
            float(row["depth_cov"]),
            str(row["target_camera"]),
            int(row["frame_id"]),
        )
    )

    selected = []
    seen = set()
    for row in rows:
        key = (int(row["frame_id"]), str(row["target_camera"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= max(int(args.hero_count), int(args.benchmark_count)):
            break

    if len(selected) < int(args.benchmark_count):
        raise ValueError("Not enough 6src_hist cases to build benchmark manifest.")

    hero_rows = selected[: int(args.hero_count)]
    benchmark_rows = selected[: int(args.benchmark_count)]
    hero_cases = [_case_payload(row, Path(row["case_dir"])) for row in hero_rows]
    benchmark_cases = [_case_payload(row, Path(row["case_dir"])) for row in benchmark_rows]

    output_payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "artifact_kind": "teacher_fixed_visual_lift_benchmark_manifest",
        "source_summary_json": str(sweep_summary_path),
        "selection_rule": "Take the hardest deterministic 6src_hist cases ranked by legacy depth_unproject MAE descending, then use the top hero_count for hero cases and the top benchmark_count for benchmark cases.",
        "hero_case_count": len(hero_cases),
        "benchmark_case_count": len(benchmark_cases),
        "hero_cases": hero_cases,
        "benchmark_cases": benchmark_cases,
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Teacher-Fixed Visual Lift Benchmark Manifest",
        "",
        f"- source_summary_json: `{sweep_summary_path}`",
        f"- hero_case_count: `{len(hero_cases)}`",
        f"- benchmark_case_count: `{len(benchmark_cases)}`",
        "",
        "## Hero Cases",
        "",
    ]
    for case in hero_cases:
        lines.append(
            "- `{case_id}` target={target} sources={sources} legacy_depth_mae={mae:.6f} legacy_depth_ssim={ssim:.6f} legacy_depth_cov={cov:.6f}".format(
                case_id=case["case_id"],
                target=case["target_camera"],
                sources=case["source_cameras"],
                mae=case["selection_reason"]["legacy_depth_mae"],
                ssim=case["selection_reason"]["legacy_depth_ssim"],
                cov=case["selection_reason"]["legacy_depth_coverage_ratio"],
            )
        )
    lines.extend(["", "## Benchmark Cases", ""])
    for case in benchmark_cases:
        lines.append(
            "- `{case_id}` target={target} sources={sources} legacy_depth_mae={mae:.6f} legacy_depth_ssim={ssim:.6f} legacy_depth_cov={cov:.6f}".format(
                case_id=case["case_id"],
                target=case["target_camera"],
                sources=case["source_cameras"],
                mae=case["selection_reason"]["legacy_depth_mae"],
                ssim=case["selection_reason"]["legacy_depth_ssim"],
                cov=case["selection_reason"]["legacy_depth_coverage_ratio"],
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
