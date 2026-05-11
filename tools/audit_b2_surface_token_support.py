from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FAMILY_ORDER = ("body", "hand", "face", "hair")
CRITICAL_FAMILIES = ("face", "hand")
SUMMARY_MIN_VIEW_SUPPORT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only B2 surface-token support audit. It compares projected "
            "token support with raster token support for body/hand/face/hair "
            "and writes research-only markdown/json diagnostics. It never reads "
            "large images, imports Open3D, writes strict pass state, or changes "
            "the B2 optimizer."
        )
    )
    parser.add_argument("--summary", required=True, type=Path, help="B2 surface_token_b2_summary.json")
    parser.add_argument(
        "--token-diagnostics",
        type=Path,
        default=None,
        help="Optional B2 token diagnostics .json or .csv. If omitted, the script tries the summary directory.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default=None, help="Optional report name. Defaults to the B2 run directory name.")
    parser.add_argument("--min-view-support", type=int, default=SUMMARY_MIN_VIEW_SUPPORT)
    parser.add_argument("--min-critical-min-view-fraction", type=float, default=0.25)
    parser.add_argument("--min-critical-visible-fraction", type=float, default=0.50)
    parser.add_argument("--min-critical-min-view-tokens", type=int, default=8)
    parser.add_argument("--warn-raster-gap-fraction", type=float, default=0.25)
    parser.add_argument("--top-tokens", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "b2_surface_token_support"


def default_report_name(summary_path: Path) -> str:
    if summary_path.parent.name == "B2_surface_tokens" and summary_path.parent.parent.name:
        return sanitize_name(summary_path.parent.parent.name)
    return sanitize_name(summary_path.parent.name or summary_path.stem)


def discover_token_diagnostics(summary_path: Path) -> Path | None:
    candidates = [
        summary_path.parent / "surface_token_b2_token_diagnostics.json",
        summary_path.parent / "surface_token_b2_token_diagnostics.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def first_number(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        out = safe_float(value, None)
        if out is not None:
            return out
    return default


def first_int(*values: Any, default: int | None = None) -> int | None:
    for value in values:
        out = safe_int(value, None)
        if out is not None:
            return out
    return default


def fraction(count: int | None, total: int | None) -> float | None:
    if count is None or total is None or total <= 0:
        return None
    return float(count) / float(total)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if abs(float(value)) >= 1000.0:
        return f"{float(value):.1f}"
    return f"{float(value):.{digits}f}"


def count_frac(count: int | None, total: int | None) -> str:
    if count is None or total is None:
        return "n/a"
    return f"{count}/{total} ({pct(fraction(count, total))})"


def normal_dispersion_from_angle(angle_deg: float | None) -> float | None:
    if angle_deg is None:
        return None
    return float(max(0.0, 1.0 - math.cos(math.radians(max(0.0, min(180.0, angle_deg))))))


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def summarize_values(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": sum(clean) / len(clean),
        "p10": percentile(clean, 0.10),
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "max": max(clean),
    }


def load_token_diagnostics(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = load_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("tokens") or payload.get("token_diagnostics") or []
        else:
            rows = []
        return [as_dict(row) for row in rows]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported token diagnostics extension: {path}")


def family_order_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def family_rows_by_name(rows: list[Any], key: str = "family") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_dict = as_dict(row)
        family = str(row_dict.get(key, "")).strip()
        if family:
            out[family] = row_dict
    return out


def numeric_field(rows: list[dict[str, Any]], key: str, *, visible_only: bool = False) -> list[float]:
    values: list[float] = []
    for row in rows:
        if visible_only and (safe_int(row.get("raster_visible_views"), 0) or 0) <= 0:
            continue
        value = safe_float(row.get(key), None)
        if value is not None:
            values.append(value)
    return values


def token_normal_dispersion(row: dict[str, Any]) -> float | None:
    direct = first_number(
        row.get("raster_normal_dispersion"),
        row.get("normal_dispersion"),
        row.get("raster_token_normal_dispersion"),
    )
    if direct is not None:
        return direct
    return normal_dispersion_from_angle(safe_float(row.get("raster_normal_angular_std_deg"), None))


def token_evidence(rows: list[dict[str, Any]], min_view_support: int, top_tokens: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family", "")).strip()
        if family:
            by_family[family].append(row)
        part = str(row.get("part_name", "")).strip()
        if part:
            by_part[part].append(row)

    family_stats: dict[str, Any] = {}
    for family, family_tokens in sorted(by_family.items(), key=lambda item: family_order_key(item[0])):
        token_count = len(family_tokens)
        projected_support = [safe_int(row.get("projected_support_views"), 0) or 0 for row in family_tokens]
        raster_support = [safe_int(row.get("raster_visible_views"), 0) or 0 for row in family_tokens]
        visible_rows = [row for row in family_tokens if (safe_int(row.get("raster_visible_views"), 0) or 0) > 0]
        normal_dispersion_values = [
            value
            for value in (token_normal_dispersion(row) for row in visible_rows)
            if value is not None and math.isfinite(value)
        ]
        gap_rows = [
            row
            for row in family_tokens
            if (safe_int(row.get("projected_support_views"), 0) or 0) >= min_view_support
            and (safe_int(row.get("raster_visible_views"), 0) or 0) < min_view_support
        ]
        no_raster_rows = [
            row
            for row in family_tokens
            if (safe_int(row.get("projected_support_views"), 0) or 0) > 0
            and (safe_int(row.get("raster_visible_views"), 0) or 0) <= 0
        ]
        sorted_gap_rows = sorted(
            family_tokens,
            key=lambda row: (
                safe_int(row.get("raster_visible_views"), 0) or 0,
                -(safe_int(row.get("projected_support_views"), 0) or 0),
                safe_float(row.get("raster_target_fraction"), 0.0) or 0.0,
                -(safe_float(row.get("raster_rgb_residual_mean"), 0.0) or 0.0),
            ),
        )
        worst_rows: list[dict[str, Any]] = []
        for row in sorted_gap_rows[: max(0, int(top_tokens))]:
            angle = safe_float(row.get("raster_normal_angular_std_deg"), None)
            worst_rows.append(
                {
                    "token_id": safe_int(row.get("token_id"), None),
                    "part_name": row.get("part_name"),
                    "projected_support_views": safe_int(row.get("projected_support_views"), None),
                    "raster_visible_views": safe_int(row.get("raster_visible_views"), None),
                    "raster_target_fraction": safe_float(row.get("raster_target_fraction"), None),
                    "raster_rgb_residual_mean": safe_float(row.get("raster_rgb_residual_mean"), None),
                    "raster_depth_std": safe_float(row.get("raster_depth_std"), None),
                    "raster_normal_angular_std_deg": angle,
                    "normal_dispersion_approx": token_normal_dispersion(row),
                    "visibility_gate": safe_float(row.get("visibility_gate"), None),
                }
            )
        family_stats[family] = {
            "token_count": token_count,
            "visible_token_count": sum(1 for value in raster_support if value > 0),
            "visible_token_fraction": fraction(sum(1 for value in raster_support if value > 0), token_count),
            "projected_tokens_with_min_view_support": sum(1 for value in projected_support if value >= min_view_support),
            "projected_min_view_fraction": fraction(sum(1 for value in projected_support if value >= min_view_support), token_count),
            "raster_tokens_with_min_view_support": sum(1 for value in raster_support if value >= min_view_support),
            "raster_min_view_fraction": fraction(sum(1 for value in raster_support if value >= min_view_support), token_count),
            "projected_min_view_but_raster_below_min_count": len(gap_rows),
            "projected_visible_but_no_raster_count": len(no_raster_rows),
            "raster_visible_views": summarize_values([float(value) for value in raster_support]),
            "projected_support_views": summarize_values([float(value) for value in projected_support]),
            "target_fraction_visible_tokens": summarize_values(numeric_field(family_tokens, "raster_target_fraction", visible_only=True)),
            "rgb_residual_visible_tokens": summarize_values(numeric_field(family_tokens, "raster_rgb_residual_mean", visible_only=True)),
            "depth_std_visible_tokens": summarize_values(numeric_field(family_tokens, "raster_depth_std", visible_only=True)),
            "normal_angular_std_deg_visible_tokens": summarize_values(
                numeric_field(family_tokens, "raster_normal_angular_std_deg", visible_only=True)
            ),
            "normal_dispersion_approx_visible_tokens": summarize_values(normal_dispersion_values),
            "worst_support_tokens": worst_rows,
        }

    part_stats: dict[str, Any] = {}
    for part, part_tokens in sorted(by_part.items()):
        token_count = len(part_tokens)
        raster_support = [safe_int(row.get("raster_visible_views"), 0) or 0 for row in part_tokens]
        projected_support = [safe_int(row.get("projected_support_views"), 0) or 0 for row in part_tokens]
        part_stats[part] = {
            "family": part_tokens[0].get("family") if part_tokens else None,
            "token_count": token_count,
            "projected_tokens_with_min_view_support": sum(1 for value in projected_support if value >= min_view_support),
            "raster_tokens_with_min_view_support": sum(1 for value in raster_support if value >= min_view_support),
            "raster_visible_token_fraction": fraction(sum(1 for value in raster_support if value > 0), token_count),
        }

    return {"family": family_stats, "part": part_stats}


def extract_family_audit(
    summary: dict[str, Any],
    token_stats: dict[str, Any] | None,
    *,
    min_view_support: int,
    warn_raster_gap_fraction: float,
) -> list[dict[str, Any]]:
    summary_block = as_dict(summary.get("summary"))
    token_meta = as_dict(summary_block.get("token_meta"))
    token_family_histogram = as_dict(token_meta.get("token_family_histogram"))
    projected_meta = as_dict(summary_block.get("projected_meta"))
    projected_by_family = family_rows_by_name(as_list(projected_meta.get("family_projected_visibility")))

    raster_source = "final_raster_meta" if isinstance(summary_block.get("final_raster_meta"), dict) else "initial_raster_meta"
    raster_meta = as_dict(summary_block.get(raster_source))
    raster_by_family = family_rows_by_name(as_list(raster_meta.get("family_raster_visibility")))
    token_by_family = as_dict(as_dict(token_stats or {}).get("family"))

    families = set(FAMILY_ORDER)
    families.update(str(key) for key in token_family_histogram.keys())
    families.update(projected_by_family.keys())
    families.update(raster_by_family.keys())
    families.update(token_by_family.keys())

    rows: list[dict[str, Any]] = []
    for family in sorted(families, key=family_order_key):
        projected = as_dict(projected_by_family.get(family))
        raster = as_dict(raster_by_family.get(family))
        token = as_dict(token_by_family.get(family))

        token_count = first_int(
            token_family_histogram.get(family),
            projected.get("token_count"),
            raster.get("token_count"),
            token.get("token_count"),
            default=0,
        )
        projected_min_count = first_int(
            token.get("projected_tokens_with_min_view_support"),
            projected.get("projected_tokens_with_min_view_support"),
            projected.get("projected_tokens_with_two_views"),
            default=None,
        )
        projected_visible_fraction = first_number(projected.get("projected_visible_token_fraction"), default=None)
        projected_min_fraction = fraction(projected_min_count, token_count)
        projected_mean_support_views = first_number(projected.get("projected_mean_support_views"), default=None)

        raster_visible_count = first_int(raster.get("visible_token_count"), token.get("visible_token_count"), default=None)
        raster_visible_fraction = first_number(
            raster.get("visible_token_fraction"),
            token.get("visible_token_fraction"),
            fraction(raster_visible_count, token_count),
            default=None,
        )
        raster_min_count = first_int(
            raster.get("tokens_with_min_view_support"),
            token.get("raster_tokens_with_min_view_support"),
            default=None,
        )
        raster_min_fraction = fraction(raster_min_count, token_count)
        pixel_count = first_number(raster.get("pixel_count"), default=None)
        target_fraction_mean = first_number(raster.get("target_fraction_mean"), default=None)
        rgb_residual_mean = first_number(raster.get("rgb_residual_mean"), default=None)
        depth_std_mean = first_number(raster.get("depth_std_mean"), default=None)
        normal_angular_std_deg_mean = first_number(raster.get("normal_angular_std_deg_mean"), default=None)
        normal_dispersion_mean = first_number(
            raster.get("normal_dispersion_mean"),
            raster.get("normal_dispersion"),
            normal_dispersion_from_angle(normal_angular_std_deg_mean),
            default=None,
        )

        min_gap_fraction = None
        if projected_min_fraction is not None and raster_min_fraction is not None:
            min_gap_fraction = projected_min_fraction - raster_min_fraction
        visible_gap_fraction = None
        if projected_visible_fraction is not None and raster_visible_fraction is not None:
            visible_gap_fraction = projected_visible_fraction - raster_visible_fraction

        flags: list[str] = []
        if raster_min_count is None:
            flags.append("raster_family_summary_missing")
        if min_gap_fraction is not None and min_gap_fraction >= float(warn_raster_gap_fraction):
            flags.append("projected_vs_raster_min_view_gap")
        if visible_gap_fraction is not None and visible_gap_fraction >= float(warn_raster_gap_fraction):
            flags.append("projected_vs_raster_visible_gap")
        if projected_min_count and raster_min_count == 0:
            flags.append("raster_min_view_support_collapse")
        if token_count and pixel_count is not None and pixel_count / max(1, token_count) < 1.0:
            flags.append("low_raster_pixels_per_token")

        rows.append(
            {
                "family": family,
                "token_count": token_count,
                "projected_visible_token_fraction": projected_visible_fraction,
                "projected_tokens_with_min_view_support": projected_min_count,
                "projected_min_view_fraction": projected_min_fraction,
                "projected_mean_support_views": projected_mean_support_views,
                "raster_visible_token_count": raster_visible_count,
                "raster_visible_token_fraction": raster_visible_fraction,
                "raster_tokens_with_min_view_support": raster_min_count,
                "raster_min_view_fraction": raster_min_fraction,
                "projected_minus_raster_visible_fraction": visible_gap_fraction,
                "projected_minus_raster_min_view_fraction": min_gap_fraction,
                "pixel_count": pixel_count,
                "pixels_per_token": (pixel_count / max(1, token_count)) if token_count and pixel_count is not None else None,
                "target_fraction_mean": target_fraction_mean,
                "rgb_residual_mean": rgb_residual_mean,
                "depth_std_mean": depth_std_mean,
                "normal_angular_std_deg_mean": normal_angular_std_deg_mean,
                "normal_dispersion_approx_mean": normal_dispersion_mean,
                "flags": flags,
                "raster_source": raster_source,
                "token_diagnostics": token,
            }
        )
    return rows


def required_min_view_tokens(token_count: int, min_fraction: float, min_tokens: int) -> int:
    if token_count <= 0:
        return 0
    return max(int(min_tokens), int(math.ceil(float(min_fraction) * float(token_count))))


def make_decision(args: argparse.Namespace, family_audit: list[dict[str, Any]], contract_checks: dict[str, Any]) -> dict[str, Any]:
    hard_failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for row in family_audit:
        family = str(row.get("family"))
        token_count = int(row.get("token_count") or 0)
        raster_min_count = safe_int(row.get("raster_tokens_with_min_view_support"), 0) or 0
        raster_min_fraction = safe_float(row.get("raster_min_view_fraction"), 0.0) or 0.0
        raster_visible_fraction = safe_float(row.get("raster_visible_token_fraction"), 0.0) or 0.0
        required_tokens = required_min_view_tokens(
            token_count,
            float(args.min_critical_min_view_fraction),
            int(args.min_critical_min_view_tokens),
        )

        if family in CRITICAL_FAMILIES and token_count > 0:
            if raster_min_count < required_tokens or raster_min_fraction < float(args.min_critical_min_view_fraction):
                hard_failures.append(
                    {
                        "family": family,
                        "reason": "critical_raster_tokens_with_min_view_support_too_low",
                        "observed_tokens_with_min_view_support": raster_min_count,
                        "required_tokens_with_min_view_support": required_tokens,
                        "observed_fraction": raster_min_fraction,
                        "required_fraction": float(args.min_critical_min_view_fraction),
                    }
                )
            if raster_visible_fraction < float(args.min_critical_visible_fraction):
                hard_failures.append(
                    {
                        "family": family,
                        "reason": "critical_visible_token_fraction_too_low",
                        "observed_fraction": raster_visible_fraction,
                        "required_fraction": float(args.min_critical_visible_fraction),
                    }
                )

        for flag in row.get("flags", []):
            warnings.append({"family": family, "reason": flag})

    for violation in contract_checks.get("violations", []):
        hard_failures.append({"family": "contract", "reason": violation})

    if hard_failures:
        label = "STOP_FIX_CARRIER_ROI_SAMPLING"
        recommendation = (
            "Stop before any direct B2 optimize steps. Fix carrier coverage and ROI sampling for face/hand raster "
            "support, then rerun step0 raster diagnostics and this audit."
        )
        direct_optimize_steps_allowed = False
    else:
        label = "GO_RESEARCH_ONLY_DIAGNOSTICS"
        recommendation = (
            "Raster token support is sufficient for further research-only B2 diagnostics. This still does not allow "
            "teacher export, candidate export, strict pass writes, train/infer/export changes, or cloud unblock claims."
        )
        direct_optimize_steps_allowed = True

    return {
        "label": label,
        "direct_optimize_steps_allowed": direct_optimize_steps_allowed,
        "strict_pass_allowed": False,
        "teacher_or_candidate_export_allowed": False,
        "cloud_unblock_signal_allowed": False,
        "recommendation": recommendation,
        "hard_failures": hard_failures,
        "warnings": warnings,
    }


def contract_checks(summary: dict[str, Any]) -> dict[str, Any]:
    contract = as_dict(summary.get("contract"))
    block = as_dict(summary.get("summary"))
    checks = {
        "research_only": bool(contract.get("research_only") is True and block.get("research_only") is True),
        "no_teacher_export": bool(block.get("no_teacher_export") is True),
        "no_candidate_export": bool(block.get("no_candidate_export") is True),
        "no_strict_pass_write": bool(block.get("no_strict_pass_write") is True),
        "strict_candidate_passes": safe_int(block.get("strict_candidate_passes"), None),
        "strict_teacher_passes": safe_int(block.get("strict_teacher_passes"), None),
        "formal_train_infer_export": block.get("formal_train_infer_export"),
    }
    violations: list[str] = []
    if not checks["research_only"]:
        violations.append("summary_does_not_confirm_research_only_true")
    if not checks["no_teacher_export"]:
        violations.append("summary_does_not_confirm_no_teacher_export")
    if not checks["no_candidate_export"]:
        violations.append("summary_does_not_confirm_no_candidate_export")
    if not checks["no_strict_pass_write"]:
        violations.append("summary_does_not_confirm_no_strict_pass_write")
    if checks["strict_candidate_passes"] != 0:
        violations.append("strict_candidate_passes_not_zero")
    if checks["strict_teacher_passes"] != 0:
        violations.append("strict_teacher_passes_not_zero")
    checks["violations"] = violations
    return checks


def view_diagnostic_summary(summary: dict[str, Any]) -> dict[str, Any]:
    block = as_dict(summary.get("summary"))
    raster_meta = as_dict(block.get("final_raster_meta") or block.get("initial_raster_meta"))
    views = as_list(raster_meta.get("view_diagnostics"))
    return {
        "view_count": len(views),
        "pred_pixels": summarize_values([safe_float(as_dict(row).get("pred_pixels"), 0.0) or 0.0 for row in views]),
        "target_pixels": summarize_values([safe_float(as_dict(row).get("target_pixels"), 0.0) or 0.0 for row in views]),
        "iou": summarize_values([safe_float(as_dict(row).get("iou"), 0.0) or 0.0 for row in views]),
        "rgb_residual_mean": summarize_values([safe_float(as_dict(row).get("rgb_residual_mean"), 0.0) or 0.0 for row in views]),
        "depth_std": summarize_values([safe_float(as_dict(row).get("depth_std"), 0.0) or 0.0 for row in views]),
        "normal_dispersion": summarize_values([safe_float(as_dict(row).get("normal_dispersion"), 0.0) or 0.0 for row in views]),
    }


def family_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("family")): row for row in rows}


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# B2 Surface Token Support Audit: {payload['name']}",
        "",
        "This is a read-only, research-only diagnostic. It does not write a strict pass, teacher, candidate, train job, infer job, export, or cloud unblock signal.",
        "",
        "## Decision",
        "",
        f"- recommendation: `{payload['decision']['label']}`",
        f"- direct optimize steps allowed: `{str(payload['decision']['direct_optimize_steps_allowed']).lower()}`",
        f"- strict pass allowed: `false`",
        f"- reason: {payload['decision']['recommendation']}",
        "",
        "## Gate Truth",
        "",
        "```text",
        "research_only = true",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "teacher/candidate export = blocked",
        "formal cloud unblock signal = blocked",
        "```",
        "",
        "## Inputs",
        "",
        f"- summary: `{payload['inputs']['summary']}`",
        f"- token diagnostics: `{payload['inputs'].get('token_diagnostics') or 'not provided'}`",
        f"- views: `{payload['run'].get('views')}`",
        f"- target_size: `{payload['run'].get('target_size')}`",
        f"- token_grid: `{payload['run'].get('token_grid')}`",
        f"- max_steps: `{payload['run'].get('max_steps')}`",
        f"- min_view_support: `{payload['thresholds']['min_view_support']}`",
        "",
        "## Family Support Gap",
        "",
        "| Family | Tokens | Projected visible | Projected >=min | Raster visible | Raster >=min | Min-view gap | Pixels | Target frac | RGB residual | Depth std | Normal disp | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in payload["family_audit"]:
        token_count = safe_int(row.get("token_count"), 0) or 0
        flags = ", ".join(row.get("flags", [])) or "-"
        lines.append(
            "| {family} | {tokens} | {proj_vis} | {proj_min} | {ras_vis} | {ras_min} | {gap} | {pixels} | {target} | {rgb} | {depth} | {normal} | {flags} |".format(
                family=row.get("family"),
                tokens=token_count,
                proj_vis=pct(safe_float(row.get("projected_visible_token_fraction"), None)),
                proj_min=count_frac(safe_int(row.get("projected_tokens_with_min_view_support"), None), token_count),
                ras_vis=count_frac(safe_int(row.get("raster_visible_token_count"), None), token_count),
                ras_min=count_frac(safe_int(row.get("raster_tokens_with_min_view_support"), None), token_count),
                gap=pct(safe_float(row.get("projected_minus_raster_min_view_fraction"), None)),
                pixels=num(safe_float(row.get("pixel_count"), None), 1),
                target=num(safe_float(row.get("target_fraction_mean"), None), 3),
                rgb=num(safe_float(row.get("rgb_residual_mean"), None), 3),
                depth=num(safe_float(row.get("depth_std_mean"), None), 3),
                normal=num(safe_float(row.get("normal_dispersion_approx_mean"), None), 3),
                flags=flags,
            )
        )

    lines.extend(["", "## Critical Findings", ""])
    if payload["decision"]["hard_failures"]:
        for failure in payload["decision"]["hard_failures"]:
            family = failure.get("family")
            reason = failure.get("reason")
            if "observed_tokens_with_min_view_support" in failure:
                lines.append(
                    "- `{family}` {reason}: raster tokens_with_min_view_support is {observed}/{required} "
                    "({observed_frac} vs required {required_frac}).".format(
                        family=family,
                        reason=reason,
                        observed=failure.get("observed_tokens_with_min_view_support"),
                        required=failure.get("required_tokens_with_min_view_support"),
                        observed_frac=pct(safe_float(failure.get("observed_fraction"), None)),
                        required_frac=pct(safe_float(failure.get("required_fraction"), None)),
                    )
                )
            elif "observed_fraction" in failure:
                lines.append(
                    "- `{family}` {reason}: observed {observed} vs required {required}.".format(
                        family=family,
                        reason=reason,
                        observed=pct(safe_float(failure.get("observed_fraction"), None)),
                        required=pct(safe_float(failure.get("required_fraction"), None)),
                    )
                )
            else:
                lines.append(f"- `{family}` {reason}.")
    else:
        lines.append("- No critical face/hand raster support failure under the configured thresholds.")

    token_stats = as_dict(payload.get("token_diagnostics_summary"))
    token_family = as_dict(token_stats.get("family"))
    if token_family:
        lines.extend(["", "## Token Diagnostics Evidence", ""])
        for family in ("face", "hand", "hair"):
            stats = as_dict(token_family.get(family))
            if not stats:
                continue
            normal_stats = as_dict(stats.get("normal_dispersion_approx_visible_tokens"))
            rgb_stats = as_dict(stats.get("rgb_residual_visible_tokens"))
            depth_stats = as_dict(stats.get("depth_std_visible_tokens"))
            target_stats = as_dict(stats.get("target_fraction_visible_tokens"))
            lines.extend(
                [
                    f"### {family}",
                    "",
                    f"- visible_token_fraction: `{pct(safe_float(stats.get('visible_token_fraction'), None))}`",
                    f"- tokens_with_min_view_support: `{count_frac(safe_int(stats.get('raster_tokens_with_min_view_support'), None), safe_int(stats.get('token_count'), None))}`",
                    f"- projected_min_view_but_raster_below_min_count: `{stats.get('projected_min_view_but_raster_below_min_count')}`",
                    f"- target_fraction visible p50/p90: `{num(safe_float(target_stats.get('p50'), None))}` / `{num(safe_float(target_stats.get('p90'), None))}`",
                    f"- rgb_residual visible p50/p90: `{num(safe_float(rgb_stats.get('p50'), None))}` / `{num(safe_float(rgb_stats.get('p90'), None))}`",
                    f"- depth_std visible p50/p90: `{num(safe_float(depth_stats.get('p50'), None))}` / `{num(safe_float(depth_stats.get('p90'), None))}`",
                    f"- normal_dispersion approx visible p50/p90: `{num(safe_float(normal_stats.get('p50'), None))}` / `{num(safe_float(normal_stats.get('p90'), None))}`",
                    "",
                    "| Token | Part | Projected views | Raster views | Target frac | RGB residual | Depth std | Normal disp |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for token in as_list(stats.get("worst_support_tokens")):
                token_row = as_dict(token)
                lines.append(
                    "| {token_id} | {part} | {projected} | {raster} | {target} | {rgb} | {depth} | {normal} |".format(
                        token_id=token_row.get("token_id"),
                        part=token_row.get("part_name"),
                        projected=token_row.get("projected_support_views"),
                        raster=token_row.get("raster_visible_views"),
                        target=num(safe_float(token_row.get("raster_target_fraction"), None)),
                        rgb=num(safe_float(token_row.get("raster_rgb_residual_mean"), None)),
                        depth=num(safe_float(token_row.get("raster_depth_std"), None)),
                        normal=num(safe_float(token_row.get("normal_dispersion_approx"), None)),
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## Next Action",
            "",
        ]
    )
    if payload["decision"]["direct_optimize_steps_allowed"]:
        lines.append("- Continue only as research-only B2 diagnostics; do not write strict pass, teacher, candidate, or cloud unblock artifacts.")
    else:
        lines.append("- Fix face/hand carrier coverage and ROI sampling before increasing B2 optimize steps, then rerun the step0 raster diagnostics and this audit.")
    lines.append("- Keep formal VGGT train/infer/export untouched until a separate strict gate actually passes.")
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    summary_path = args.summary.resolve()
    summary = load_json(summary_path)
    diagnostics_path = args.token_diagnostics.resolve() if args.token_diagnostics else discover_token_diagnostics(summary_path)
    diagnostics_rows = load_token_diagnostics(diagnostics_path) if diagnostics_path else []
    diagnostics_summary = token_evidence(diagnostics_rows, int(args.min_view_support), int(args.top_tokens)) if diagnostics_rows else {}
    checks = contract_checks(summary)
    family_audit = extract_family_audit(
        summary,
        diagnostics_summary,
        min_view_support=int(args.min_view_support),
        warn_raster_gap_fraction=float(args.warn_raster_gap_fraction),
    )
    decision = make_decision(args, family_audit, checks)
    summary_block = as_dict(summary.get("summary"))
    name = sanitize_name(args.name) if args.name else default_report_name(summary_path)
    payload = {
        "audit_status": "diagnostic_complete",
        "name": name,
        "research_only": True,
        "read_only": True,
        "writes_strict_pass": False,
        "writes_teacher_or_candidate": False,
        "reads_large_images": False,
        "uses_open3d": False,
        "inputs": {
            "summary": str(summary_path),
            "token_diagnostics": str(diagnostics_path.resolve()) if diagnostics_path else None,
        },
        "thresholds": {
            "min_view_support": int(args.min_view_support),
            "min_critical_min_view_fraction": float(args.min_critical_min_view_fraction),
            "min_critical_visible_fraction": float(args.min_critical_visible_fraction),
            "min_critical_min_view_tokens": int(args.min_critical_min_view_tokens),
            "warn_raster_gap_fraction": float(args.warn_raster_gap_fraction),
        },
        "run": {
            "status": summary.get("status"),
            "decision": summary.get("decision"),
            "views": summary_block.get("views"),
            "target_size": summary_block.get("target_size"),
            "max_steps": summary_block.get("max_steps"),
            "diagnostics_only": summary_block.get("diagnostics_only"),
            "token_grid": summary_block.get("token_grid"),
            "token_hidden": summary_block.get("token_hidden"),
            "token_count": as_dict(summary_block.get("token_meta")).get("token_count"),
            "avg_initial_iou": summary_block.get("avg_initial_iou"),
            "avg_final_iou": summary_block.get("avg_final_iou"),
        },
        "contract_checks": checks,
        "family_audit": family_audit,
        "token_diagnostics_summary": diagnostics_summary,
        "view_diagnostics_summary": view_diagnostic_summary(summary),
        "decision": decision,
    }
    return payload, name


def main() -> None:
    args = parse_args()
    payload, name = build_payload(args)
    output_dir = args.output_dir.resolve()
    json_path = output_dir / f"{name}_b2_surface_token_support_audit.json"
    md_path = output_dir / f"{name}_b2_surface_token_support_audit.md"
    payload["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    write_json(json_path, payload)
    write_text(md_path, render_markdown(payload))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
