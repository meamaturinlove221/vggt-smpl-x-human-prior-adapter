import argparse
import json
import re
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"Loss/(?P<phase>train|val)_(?P<name>[A-Za-z0-9_]+):\s+"
    r"(?P<current>-?\d+(?:\.\d+)?)\s+\((?P<average>-?\d+(?:\.\d+)?)\)"
)
STEP_PATTERN = re.compile(
    r"(?P<phase>Train|Val) Epoch:\s+\[\d+\]\[\s*(?P<step>\d+)/(?P<total>\d+)\]"
)


def parse_log(log_path: Path) -> dict:
    phase_summaries = {"train": None, "val": None}

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        step_match = STEP_PATTERN.search(raw_line)
        metric_matches = list(METRIC_PATTERN.finditer(raw_line))
        if not step_match or not metric_matches:
            continue

        phase = step_match.group("phase").lower()
        summary = {
            "step": int(step_match.group("step")),
            "total": int(step_match.group("total")),
            "metrics": {},
        }
        for metric_match in metric_matches:
            metric_phase = metric_match.group("phase")
            if metric_phase != phase:
                continue
            summary["metrics"][metric_match.group("name")] = {
                "current": float(metric_match.group("current")),
                "average": float(metric_match.group("average")),
            }
        phase_summaries[phase] = summary

    return phase_summaries


def metric_average(summary: dict, metric_name: str):
    if summary is None:
        return None
    metric = summary["metrics"].get(metric_name)
    if metric is None:
        return None
    return metric["average"]


def build_phase_rows(baseline_phase: dict, candidate_phase: dict) -> list[dict]:
    metric_names = sorted(
        set((baseline_phase or {}).get("metrics", {}).keys())
        | set((candidate_phase or {}).get("metrics", {}).keys())
    )
    rows = []
    for metric_name in metric_names:
        baseline_value = metric_average(baseline_phase, metric_name)
        candidate_value = metric_average(candidate_phase, metric_name)
        delta = None
        if baseline_value is not None and candidate_value is not None:
            delta = candidate_value - baseline_value
        rows.append(
            {
                "metric": metric_name,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
            }
        )
    return rows


def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two ZJU fine-tune log.txt files and emit a compact summary."
    )
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--candidate-log", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="ZJU Fine-Tune Run Comparison",
        help="Title written into the markdown summary.",
    )
    args = parser.parse_args()

    baseline = parse_log(args.baseline_log)
    candidate = parse_log(args.candidate_log)

    train_rows = build_phase_rows(baseline["train"], candidate["train"])
    val_rows = build_phase_rows(baseline["val"], candidate["val"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": args.title,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_log": str(args.baseline_log),
        "candidate_log": str(args.candidate_log),
        "train": {
            "baseline": baseline["train"],
            "candidate": candidate["train"],
            "rows": train_rows,
        },
        "val": {
            "baseline": baseline["val"],
            "candidate": candidate["val"],
            "rows": val_rows,
        },
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# {args.title}",
        "",
        f"- {args.baseline_label}: `{args.baseline_log}`",
        f"- {args.candidate_label}: `{args.candidate_log}`",
        "",
    ]

    for phase_name, rows in (("Train", train_rows), ("Val", val_rows)):
        phase_key = phase_name.lower()
        baseline_phase = baseline[phase_key]
        candidate_phase = candidate[phase_key]
        md_lines.append(f"## {phase_name}")
        md_lines.append("")
        if baseline_phase is not None:
            md_lines.append(
                f"- {args.baseline_label} last step: `{baseline_phase['step']}` / `{baseline_phase['total']}`"
            )
        if candidate_phase is not None:
            md_lines.append(
                f"- {args.candidate_label} last step: `{candidate_phase['step']}` / `{candidate_phase['total']}`"
            )
        md_lines.append("")
        md_lines.append(
            f"| metric | {args.baseline_label} | {args.candidate_label} | delta ({args.candidate_label} - {args.baseline_label}) |"
        )
        md_lines.append("| --- | ---: | ---: | ---: |")
        for row in rows:
            md_lines.append(
                f"| `{row['metric']}` | {format_float(row['baseline'])} | {format_float(row['candidate'])} | {format_float(row['delta'])} |"
            )
        md_lines.append("")

    summary_md = output_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[done] Wrote {summary_md}")
    print(f"[done] Wrote {summary_json}")


if __name__ == "__main__":
    main()
