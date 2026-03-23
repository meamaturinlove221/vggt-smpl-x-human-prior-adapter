import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def avg_metric(summary: dict, phase: str, side: str, metric: str):
    try:
        return summary[phase][side]["metrics"][metric]["average"]
    except Exception:
        return None


def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize multiple geometry pair summary.json files into one markdown table."
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Pair item in the form label=path/to/summary.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="Geometry Pair Sweep Summary",
        help="Markdown title.",
    )
    args = parser.parse_args()

    rows = []
    for raw_item in args.item:
        if "=" not in raw_item:
            raise ValueError(f"Expected label=path form, got: {raw_item}")
        label, path_str = raw_item.split("=", 1)
        path = Path(path_str)
        summary = load_summary(path)
        row = {
            "label": label,
            "path": str(path),
            "train_baseline_objective": avg_metric(summary, "train", "baseline", "loss_objective"),
            "train_candidate_objective": avg_metric(summary, "train", "candidate", "loss_objective"),
            "val_baseline_objective": avg_metric(summary, "val", "baseline", "loss_objective"),
            "val_candidate_objective": avg_metric(summary, "val", "candidate", "loss_objective"),
            "val_candidate_unproject": avg_metric(summary, "val", "candidate", "loss_unproject_geometry"),
            "val_candidate_conf_depth": avg_metric(summary, "val", "candidate", "loss_conf_depth"),
            "val_baseline_conf_depth": avg_metric(summary, "val", "baseline", "loss_conf_depth"),
        }
        if row["val_baseline_objective"] is not None and row["val_candidate_objective"] is not None:
            row["val_objective_delta"] = row["val_candidate_objective"] - row["val_baseline_objective"]
        else:
            row["val_objective_delta"] = None
        rows.append(row)

    rows.sort(key=lambda item: (item["val_objective_delta"] is None, item["val_objective_delta"]))

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# {args.title}",
        "",
        "| label | val baseline obj | val candidate obj | val delta | val baseline conf_depth | val candidate conf_depth | val candidate unproject | train baseline obj | train candidate obj | source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {label} | {vbo} | {vco} | {delta} | {vbcd} | {vccd} | {vcu} | {tbo} | {tco} | `{path}` |".format(
                label=row["label"],
                vbo=fmt(row["val_baseline_objective"]),
                vco=fmt(row["val_candidate_objective"]),
                delta=fmt(row["val_objective_delta"]),
                vbcd=fmt(row["val_baseline_conf_depth"]),
                vccd=fmt(row["val_candidate_conf_depth"]),
                vcu=fmt(row["val_candidate_unproject"]),
                tbo=fmt(row["train_baseline_objective"]),
                tco=fmt(row["train_candidate_objective"]),
                path=row["path"],
            )
        )
    md_lines.append("")

    output.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[done] Wrote {output}")


if __name__ == "__main__":
    main()
