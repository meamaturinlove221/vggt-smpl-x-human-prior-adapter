from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local normal-line teacher/candidate gate artifacts under the "
            "current strict mentor protocol. This is intentionally read-only over "
            "outputs and only writes a registry report."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "output/normal_line_multiview_20260428",
            "output/normal_line_multiview_20260430",
        ],
    )
    parser.add_argument("--report-json", default="reports/20260501_strict_gate_registry.json")
    parser.add_argument("--report-md", default="reports/20260501_teacher_gate_blocker_status.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bool_path(payload: dict[str, Any] | None, *keys: str) -> bool:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return False
        cur = cur.get(key)
    return bool(cur)


def candidate_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    package_mode = str(payload.get("package_mode", "legacy_or_unknown"))
    gates = {
        "numeric": bool_path(payload, "numeric_gate", "pass"),
        "fullbody": bool_path(payload, "fullbody_gate", "pass"),
        "fullbody_provenance": bool_path(payload, "fullbody_provenance_gate", "pass"),
        "normal": bool_path(payload, "normal_gate", "pass"),
        "shape": bool_path(payload, "shape_gate", "pass"),
        "visual": bool_path(payload, "visual_gate", "pass"),
        "aux_ok": bool(payload.get("aux_ok")),
        "render_artifacts_ok": bool(payload.get("render_artifacts_ok") or payload.get("render_ok")),
    }
    strict_pass = bool(
        package_mode == "full_mentor_gate"
        and not bool(payload.get("cloud_upload_blocked", True))
        and all(gates.values())
    )
    apparent_green_but_not_strict = bool(
        not bool(payload.get("cloud_upload_blocked", True)) and not strict_pass
    )
    failed = [name for name, ok in gates.items() if not ok]
    if package_mode != "full_mentor_gate":
        failed.append(f"not_full_mentor_gate:{package_mode}")
    return {
        "kind": "candidate",
        "name": payload.get("candidate_name") or path.parent.name,
        "path": str(path),
        "output_dir": str(path.parent),
        "package_mode": package_mode,
        "cloud_upload_blocked": bool(payload.get("cloud_upload_blocked", True)),
        "strict_mentor_pass": strict_pass,
        "apparent_green_but_not_strict": apparent_green_but_not_strict,
        "gates": gates,
        "failed": failed,
        "contact_sheet": payload.get("contact_sheet"),
        "report": payload.get("report"),
    }


def teacher_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    gates = {
        "numeric": bool(gate.get("numeric_pass")),
        "visual": bool(gate.get("visual_pass")),
        "overall": bool(gate.get("pass")),
    }
    strict_pass = bool(gates["numeric"] and gates["visual"] and gates["overall"])
    failed = [name for name, ok in gates.items() if not ok]
    return {
        "kind": "teacher",
        "name": path.parent.name,
        "path": str(path),
        "output_dir": str(path.parent),
        "source_kind": payload.get("source_kind"),
        "source_path": payload.get("source_path"),
        "strict_teacher_pass": strict_pass,
        "gates": gates,
        "failed": failed,
    }


def scan(roots: list[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    teachers: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("candidate_gate_summary.json"):
            payload = load_json(path)
            if payload is not None:
                candidates.append(candidate_record(path, payload))
        for path in root.rglob("teacher_gate_summary.json"):
            payload = load_json(path)
            if payload is not None:
                teachers.append(teacher_record(path, payload))
    candidates.sort(key=lambda row: (row["strict_mentor_pass"], row["output_dir"]))
    teachers.sort(key=lambda row: (row["strict_teacher_pass"], row["output_dir"]))
    strict_candidate_passes = [row for row in candidates if row["strict_mentor_pass"]]
    strict_teacher_passes = [row for row in teachers if row["strict_teacher_pass"]]
    apparent_green = [row for row in candidates if row["apparent_green_but_not_strict"]]
    full_gate_numeric_visual_fails = [
        row
        for row in candidates
        if row["package_mode"] == "full_mentor_gate"
        and row["gates"].get("numeric")
        and not row["gates"].get("visual")
    ]
    numeric_teacher_visual_fails = [
        row
        for row in teachers
        if row["gates"].get("numeric") and not row["gates"].get("visual")
    ]
    return {
        "counts": {
            "candidates": len(candidates),
            "teachers": len(teachers),
            "strict_candidate_passes": len(strict_candidate_passes),
            "strict_teacher_passes": len(strict_teacher_passes),
            "legacy_or_diagnostic_apparent_green": len(apparent_green),
            "full_gate_numeric_pass_visual_fail": len(full_gate_numeric_visual_fails),
            "teacher_numeric_pass_visual_fail": len(numeric_teacher_visual_fails),
        },
        "strict_candidate_passes": strict_candidate_passes,
        "strict_teacher_passes": strict_teacher_passes,
        "legacy_or_diagnostic_apparent_green": apparent_green,
        "full_gate_numeric_pass_visual_fail": full_gate_numeric_visual_fails,
        "teacher_numeric_pass_visual_fail": numeric_teacher_visual_fails,
        "candidates": candidates,
        "teachers": teachers,
    }


def bullet_rows(rows: list[dict[str, Any]], max_rows: int = 40) -> list[str]:
    lines: list[str] = []
    for row in rows[:max_rows]:
        failed = ", ".join(row.get("failed") or [])
        name = row.get("name")
        out = row.get("output_dir")
        lines.append(f"- `{name}`: failed=`{failed}`; output=`{out}`")
    if len(rows) > max_rows:
        lines.append(f"- ... {len(rows) - max_rows} more omitted from markdown; see JSON registry.")
    return lines


def write_markdown(path: Path, registry: dict[str, Any]) -> None:
    counts = registry["counts"]
    lines = [
        "# Teacher/Candidate Strict Gate Blocker Status",
        "",
        "Date: 2026-05-01",
        "",
        "## Current Truth",
        "",
        (
            "No local candidate or teacher currently passes the strict mentor gate. "
            "Cloud upload remains blocked. Numeric point counts, normal consistency, "
            "or depth-compatible teacher coverage are not accepted without explicit "
            "Open3D visual pass and full-body/hand bottom-line pass."
        ),
        "",
        "## Counts",
        "",
        f"- Candidate gate summaries scanned: `{counts['candidates']}`",
        f"- Teacher gate summaries scanned: `{counts['teachers']}`",
        f"- Strict full mentor candidate passes: `{counts['strict_candidate_passes']}`",
        f"- Strict teacher passes: `{counts['strict_teacher_passes']}`",
        f"- Legacy/diagnostic apparent green packages: `{counts['legacy_or_diagnostic_apparent_green']}`",
        f"- Full mentor packages with numeric pass but visual fail: `{counts['full_gate_numeric_pass_visual_fail']}`",
        f"- Teacher packages with numeric pass but visual fail: `{counts['teacher_numeric_pass_visual_fail']}`",
        "",
        "## Strict Passes",
        "",
        "None.",
        "",
        "## Numeric Positive But Visual Negative Candidates",
        "",
    ]
    lines.extend(bullet_rows(registry["full_gate_numeric_pass_visual_fail"]))
    lines.extend(
        [
            "",
            "## Numeric Positive But Visual Negative Teachers",
            "",
        ]
    )
    lines.extend(bullet_rows(registry["teacher_numeric_pass_visual_fail"]))
    lines.extend(
        [
            "",
            "## Legacy Or Diagnostic Apparent Green",
            "",
            (
                "These entries have old or diagnostic status fields that can look green, "
                "but they are not strict full mentor gates. They must not be used for a "
                "pass claim without re-packaging under the current full protocol."
            ),
            "",
        ]
    )
    lines.extend(bullet_rows(registry["legacy_or_diagnostic_apparent_green"]))
    lines.extend(
        [
            "",
            "## Frozen / Negative Routes",
            "",
            "- HART-style PnP camera replacement: local ablation did not improve head/face Open3D or beat the VGGT camera-head chain.",
            "- r16/r18/r19 more epoch or same-config retry: consistency gains did not translate to face/head point cloud quality.",
            "- r57/r58/r59/r60 and r61-r68: blocked by strict same-protocol, normal/shape, full-body/hand, or visual gates.",
            "- TSDF from signfix depth and 60-view direct/fused surfaces: numeric/depth-compatible positives are shell-like or coordinate/depth incompatible.",
            "- Kinect/MVS/COLMAP/external pointcloud projection patches: not a passing teacher under same-protocol depth/visual gates.",
            "- Visual hull/keypoint/MediaPipe relief/SMPL-X face scaffold: numeric gains do not produce modeled personal face/head/hairline geometry.",
            "",
            "## Active Blocker",
            "",
            (
                "The local repo still lacks a continuous, aligned, visually valid "
                "head/face/hairline surface teacher that can be projected back to "
                "`0012_11_frame0000_6views_sparseproto_headshoulder_crop` and pass "
                "both numeric and explicit Open3D visual teacher gates."
            ),
            "",
            "## Allowed Next Actions Before Any Training",
            "",
            "- Kinect coordinate convention audit only: no projection patch, no training, no cloud unless the resulting teacher passes strict gate.",
            "- SMPL-X weak full-body/hand anchor audit only: not a face teacher and not a pass claim.",
            "- Multi-view consistent face surface teacher design: must produce one shared 3D surface and pass teacher-gate before one-frame overfit.",
            "",
            "## Cloud Policy",
            "",
            "No cloud upload until a teacher passes strict teacher gate and a local one-frame overfit passes the full candidate gate, including full-body/hands and explicit visual review.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    roots = [Path(root).resolve() for root in args.roots]
    registry = scan(roots)
    report_json = Path(args.report_json).resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(Path(args.report_md).resolve(), registry)
    print(json.dumps(registry["counts"], indent=2, ensure_ascii=False))
    return 0 if not registry["strict_candidate_passes"] and not registry["strict_teacher_passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
