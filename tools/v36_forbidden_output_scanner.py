from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_JSON = REPORTS / "20260508_v36_forbidden_output_scan.json"
DEFAULT_MD = REPORTS / "20260508_v36_forbidden_output_scan.md"
DEFAULT_ROOTS = [
    REPO_ROOT / "output" / "surface_research_preflight_local",
    REPO_ROOT / "output" / "surface_research_cloud_preflight",
]


FORBIDDEN_NAMES = {
    "predictions.npz",
    "candidate_package",
    "teacher_package",
    "strict_gate_registry",
}
FORBIDDEN_SUBSTRINGS = (
    "formal_candidate",
    "strict_pass",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def scan_root(root: Path) -> list[Path]:
    hits: list[Path] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.as_posix().lower()
        if path.name.lower() in FORBIDDEN_NAMES or any(token in rel for token in FORBIDDEN_SUBSTRINGS):
            # Historical DLine directories from old runs are not current V29-V36 outputs.
            if "dline_v12_tmf_promotion_transaction" in rel:
                continue
            hits.append(path)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan V29-V36 research roots for forbidden formal outputs.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    args = parser.parse_args()
    all_hits: list[Path] = []
    root_rows = []
    for root in args.roots:
        hits = scan_root(root)
        all_hits.extend(hits)
        root_rows.append({"root": root, "hit_count": len(hits), "hits": hits[:50]})
    status = "DONE_PASS" if not all_hits else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v36_forbidden_output_scanner",
        "created_utc": utc_now(),
        "status": status,
        "hit_count": len(all_hits),
        "roots": root_rows,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
    }
    write_json(args.output_json, summary)
    lines = [
        "# V36 Forbidden Output Scan",
        "",
        f"Status: `{status}`",
        f"Hit count: `{len(all_hits)}`",
        "",
        "## Roots",
        "",
    ]
    for row in root_rows:
        lines.append(f"- {row['root']}: `{row['hit_count']}`")
    lines.extend(["", "## Hits", ""])
    lines.extend([f"- `{hit}`" for hit in all_hits] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "hit_count": len(all_hits)}, ensure_ascii=False))
    return 0 if not all_hits else 2


if __name__ == "__main__":
    raise SystemExit(main())
