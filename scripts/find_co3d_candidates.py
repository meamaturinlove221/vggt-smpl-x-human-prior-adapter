import argparse
import json
import os
from pathlib import Path


SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Find likely CO3D dataset and annotation directories.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["C:\\", "D:\\", "G:\\"],
        help="Root directories to scan.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum recursion depth from each root.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/co3d_candidates.json",
        help="Where to save the structured result.",
    )
    parser.add_argument(
        "--roots-env",
        type=str,
        default="CO3D_SCAN_ROOTS_JSON",
        help="Optional environment variable that contains a JSON array of roots. Used when present.",
    )
    parser.add_argument(
        "--roots-file",
        type=str,
        default="",
        help="Optional UTF-8 JSON file that contains a JSON array of roots. Used before --roots-env.",
    )
    return parser.parse_args()


def depth_from_root(root: Path, path: Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 10**9
    return len(rel.parts)


def score_annotation_dir(path: Path):
    train_hits = 0
    test_hits = 0
    sample_files = []
    for category in SEEN_CATEGORIES:
        train_file = path / f"{category}_train.jgz"
        test_file = path / f"{category}_test.jgz"
        if train_file.exists():
            train_hits += 1
            if len(sample_files) < 5:
                sample_files.append(str(train_file))
        if test_file.exists():
            test_hits += 1
            if len(sample_files) < 5:
                sample_files.append(str(test_file))

    score = train_hits * 2 + test_hits
    if score <= 0:
        return None
    if (train_hits + test_hits) < 3:
        return None

    return {
        "path": str(path),
        "kind": "annotation_dir",
        "score": score,
        "train_hits": train_hits,
        "test_hits": test_hits,
        "sample_files": sample_files,
    }


def score_dataset_dir(path: Path):
    category_hits = 0
    image_dir_hits = 0
    depth_dir_hits = 0
    depth_mask_dir_hits = 0
    sample_dirs = []

    for category in SEEN_CATEGORIES:
        category_dir = path / category
        if not category_dir.is_dir():
            continue
        category_hits += 1
        if len(sample_dirs) < 5:
            sample_dirs.append(str(category_dir))

        try:
            for seq_dir in category_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                if (seq_dir / "images").is_dir():
                    image_dir_hits += 1
                if (seq_dir / "depths").is_dir():
                    depth_dir_hits += 1
                if (seq_dir / "depth_masks").is_dir():
                    depth_mask_dir_hits += 1
                if image_dir_hits >= 3 and depth_dir_hits >= 3:
                    break
        except PermissionError:
            continue

    score = category_hits * 3 + image_dir_hits * 2 + depth_dir_hits + depth_mask_dir_hits
    if score <= 0:
        return None
    if category_hits < 3:
        return None
    if image_dir_hits == 0 and depth_dir_hits == 0 and depth_mask_dir_hits == 0:
        return None

    return {
        "path": str(path),
        "kind": "dataset_dir",
        "score": score,
        "category_hits": category_hits,
        "image_dir_hits": image_dir_hits,
        "depth_dir_hits": depth_dir_hits,
        "depth_mask_dir_hits": depth_mask_dir_hits,
        "sample_dirs": sample_dirs,
    }


def scan_root(root: Path, max_depth: int):
    candidates = []
    if not root.exists():
        return candidates

    stack = [root]
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        current_depth = depth_from_root(root, current)
        if current_depth > max_depth:
            continue

        if current != root:
            anno_candidate = score_annotation_dir(current)
            if anno_candidate:
                candidates.append(anno_candidate)

            dataset_candidate = score_dataset_dir(current)
            if dataset_candidate:
                candidates.append(dataset_candidate)

        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except (PermissionError, OSError):
            continue
        stack.extend(children)

    return candidates


def write_markdown(output_md: Path, roots, candidates):
    lines = [
        "# CO3D Candidate Scan",
        "",
        f"- roots: `{', '.join(roots)}`",
        "",
    ]

    dataset_rows = [item for item in candidates if item["kind"] == "dataset_dir"]
    anno_rows = [item for item in candidates if item["kind"] == "annotation_dir"]

    def append_table(title, rows, columns):
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.append("- none found")
            lines.append("")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            values = [str(row.get(column, "")) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    append_table(
        "Likely Dataset Dirs",
        sorted(dataset_rows, key=lambda item: item["score"], reverse=True),
        ["path", "score", "category_hits", "image_dir_hits", "depth_dir_hits", "depth_mask_dir_hits"],
    )
    append_table(
        "Likely Annotation Dirs",
        sorted(anno_rows, key=lambda item: item["score"], reverse=True),
        ["path", "score", "train_hits", "test_hits"],
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.roots_file:
        roots = [Path(root) for root in json.loads(Path(args.roots_file).read_text(encoding="utf-8-sig"))]
    else:
        roots_blob = os.environ.get(args.roots_env, "").strip()
        if roots_blob:
            roots = [Path(root) for root in json.loads(roots_blob)]
        else:
            roots = [Path(root) for root in args.roots]
    if not roots:
        raise ValueError("No scan roots were provided.")
    all_candidates = []
    for root in roots:
        print(f"[find-co3d] scanning {root}", flush=True)
        all_candidates.extend(scan_root(root, args.max_depth))

    all_candidates.sort(key=lambda item: (item["kind"], item["score"]), reverse=True)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "roots": [str(root) for root in roots],
                "max_depth": args.max_depth,
                "candidates": all_candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(output_json.with_suffix(".md"), [str(root) for root in roots], all_candidates)
    print(f"[find-co3d] wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
