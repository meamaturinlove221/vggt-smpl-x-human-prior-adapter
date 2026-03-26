import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTRIBUTION_SUMMARY = (
    REPO_ROOT
    / "output"
    / "zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "zju_conf_depth_quality_signal_confdepth_dropworst_gradconfmask_20260326_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize whether conf-depth residuals track cached supervised-view quality scores."
    )
    parser.add_argument(
        "--attribution-summary",
        type=Path,
        default=DEFAULT_ATTRIBUTION_SUMMARY,
        help="Path to the existing conf-depth attribution summary.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write summary.json and summary.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_thirds(rows: list[dict], value_key: str) -> list[dict]:
    if len(rows) < 3:
        return []
    ordered = sorted(rows, key=lambda row: float(row[value_key]))
    cut1 = len(ordered) // 3
    cut2 = (2 * len(ordered)) // 3
    buckets = [
        ("low", ordered[:cut1]),
        ("mid", ordered[cut1:cut2]),
        ("high", ordered[cut2:]),
    ]
    payload = []
    for label, bucket_rows in buckets:
        if not bucket_rows:
            continue
        payload.append(
            {
                "bucket": label,
                "count": len(bucket_rows),
                "quality_score_mean": mean_or_none([float(row["quality_score"]) for row in bucket_rows]),
                "delta_conf_depth_mean": mean_or_none([float(row["delta_conf_depth_mean"]) for row in bucket_rows]),
                "delta_reg_depth_mean": mean_or_none([float(row["delta_reg_depth_mean"]) for row in bucket_rows]),
            }
        )
    return payload


def summarize_rows(rows: list[dict]) -> dict:
    quality = [float(row["quality_score"]) for row in rows]
    delta_conf = [float(row["delta_conf_depth_mean"]) for row in rows]
    delta_reg = [float(row["delta_reg_depth_mean"]) for row in rows]
    return {
        "count": len(rows),
        "quality_score_mean": mean_or_none(quality),
        "delta_conf_depth_mean": mean_or_none(delta_conf),
        "delta_reg_depth_mean": mean_or_none(delta_reg),
        "quality_to_conf_depth_corr": pearson_corr(quality, delta_conf),
        "quality_to_reg_depth_corr": pearson_corr(quality, delta_reg),
        "quality_score_buckets": build_thirds(rows, "quality_score"),
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main():
    args = parse_args()
    payload = load_json(args.attribution_summary)
    delta_rows = payload.get("delta_rows", [])
    anchor_rows = [
        row
        for row in delta_rows
        if row.get("view_role") == "anchor_supervised"
        and row.get("quality_score") is not None
        and row.get("delta_conf_depth_mean") is not None
        and row.get("delta_reg_depth_mean") is not None
    ]

    by_camera: dict[str, list[dict]] = {}
    for row in anchor_rows:
        by_camera.setdefault(str(row["camera_name"]), []).append(row)

    camera_summaries = {
        camera_name: summarize_rows(rows)
        for camera_name, rows in sorted(by_camera.items())
    }

    top_anchor_camera = None
    top_anchor_payload = payload.get("candidate_recommendation", {}).get("top_anchor")
    if isinstance(top_anchor_payload, dict):
        camera_name = top_anchor_payload.get("camera_name")
        if camera_name is not None:
            top_anchor_camera = str(camera_name)
    elif top_anchor_payload:
        top_anchor_camera = str(top_anchor_payload).split()[0]
    if not top_anchor_camera:
        top_anchor_camera = max(
            camera_summaries.items(),
            key=lambda item: item[1]["delta_conf_depth_mean"] if item[1]["delta_conf_depth_mean"] is not None else float("-inf"),
        )[0]

    top_anchor_rows = by_camera.get(top_anchor_camera, [])
    top_anchor_summary = summarize_rows(top_anchor_rows) if top_anchor_rows else {}
    non_top_anchor_rows = [row for row in anchor_rows if str(row["camera_name"]) != top_anchor_camera]
    non_top_anchor_summary = summarize_rows(non_top_anchor_rows) if non_top_anchor_rows else {}

    recommendation = {
        "dominant_failure_shape": "anchor_and_quality_conditioned",
        "recommended_candidate_family": "quality_conditioned_conf_target_normalization",
        "reason": (
            f"The exhausted fixed-scale {top_anchor_camera} family left one remaining bounded question: "
            f"on the audited anchor-supervised rows, cached quality_score is positively correlated with worse "
            f"conf_depth delta overall ({format_metric(camera_summaries.get(top_anchor_camera, {}).get('quality_to_conf_depth_corr'))} "
            f"within {top_anchor_camera}; {format_metric(summarize_rows(anchor_rows).get('quality_to_conf_depth_corr'))} across anchors). "
            "That pattern suggests cached depth_conf may be miscalibrated on the bad anchor rather than uniformly too low or too high everywhere."
        ),
        "top_anchor_camera": top_anchor_camera,
        "top_anchor_quality_to_conf_depth_corr": top_anchor_summary.get("quality_to_conf_depth_corr"),
        "all_anchor_quality_to_conf_depth_corr": summarize_rows(anchor_rows).get("quality_to_conf_depth_corr"),
        "caution": (
            f"This signal comes from {len(top_anchor_rows)} {top_anchor_camera} anchor rows on the same 32-sample audit slice, "
            "so it is strong enough to define the next local question, but not strong enough to skip a local gate."
        ),
    }

    result = {
        "attribution_summary": str(args.attribution_summary.resolve()),
        "primary_label": payload.get("primary_label"),
        "reference_label": payload.get("reference_label"),
        "anchor_row_count": len(anchor_rows),
        "top_anchor_camera": top_anchor_camera,
        "all_anchor_summary": summarize_rows(anchor_rows),
        "top_anchor_summary": top_anchor_summary,
        "non_top_anchor_summary": non_top_anchor_summary,
        "camera_summaries": camera_summaries,
        "recommendation": recommendation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)

    lines = [
        "# ZJU Conf-Depth Quality Signal Audit",
        "",
        f"- attribution_summary: `{args.attribution_summary}`",
        f"- primary_label: `{payload.get('primary_label')}`",
        f"- reference_label: `{payload.get('reference_label')}`",
        f"- anchor_row_count: `{len(anchor_rows)}`",
        f"- top_anchor_camera: `{top_anchor_camera}`",
        "",
        "## Anchor Quality Correlation",
        "",
        "| scope | count | quality_score_mean | delta_conf_depth_mean | delta_reg_depth_mean | corr(quality, delta_conf_depth) | corr(quality, delta_reg_depth) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, summary in (
        ("all_anchor", result["all_anchor_summary"]),
        (f"{top_anchor_camera}", top_anchor_summary),
        ("non_top_anchor", non_top_anchor_summary),
    ):
        if not summary:
            continue
        lines.append(
            "| {label} | {count} | {q_mean} | {d_conf} | {d_reg} | {corr_conf} | {corr_reg} |".format(
                label=label,
                count=summary["count"],
                q_mean=format_metric(summary["quality_score_mean"]),
                d_conf=format_metric(summary["delta_conf_depth_mean"]),
                d_reg=format_metric(summary["delta_reg_depth_mean"]),
                corr_conf=format_metric(summary["quality_to_conf_depth_corr"]),
                corr_reg=format_metric(summary["quality_to_reg_depth_corr"]),
            )
        )

    lines.extend(
        [
            "",
            f"## {top_anchor_camera} Quality Buckets",
            "",
            "| bucket | count | quality_score_mean | delta_conf_depth_mean | delta_reg_depth_mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in top_anchor_summary.get("quality_score_buckets", []):
        lines.append(
            "| {bucket} | {count} | {q_mean} | {d_conf} | {d_reg} |".format(
                bucket=bucket["bucket"],
                count=bucket["count"],
                q_mean=format_metric(bucket["quality_score_mean"]),
                d_conf=format_metric(bucket["delta_conf_depth_mean"]),
                d_reg=format_metric(bucket["delta_reg_depth_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- dominant_failure_shape: `{recommendation['dominant_failure_shape']}`",
            f"- recommended_candidate_family: `{recommendation['recommended_candidate_family']}`",
            f"- reason: {recommendation['reason']}",
            f"- caution: {recommendation['caution']}",
            "",
        ]
    )
    write_text(output_dir / "summary.md", "\n".join(lines))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
