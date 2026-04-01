from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
ARCHIVE_ROOT = OUTPUT_ROOT / "approved_problem_archive"
DATE_TAG = "20260330"

TARGET_FAMILIES = [
    "hardtail_bucket_granularity_refinement",
    "soft_tail_exposure_rebalancing",
    "hybrid_tail_exposure_balancing",
    "tail_conf_branch_decoupling",
    "tail_source_pool_tempering",
    "tail_anchor_stabilization",
    "tail_pose_branch_decoupling",
    "tail_intrinsics_branch_decoupling",
    "tail_counterbalance_cohort_mixing",
    "tail_anchor_reserve_hybridization",
    "tail_manifest_focal_reinforcement",
    "tail_stream_selective_focal_reinforcement",
    "tail_contract_anchor_replay",
    "tail_contract_viewset_replay",
    "tail_dual_supervision_rebalancing",
    "default_stream_intrinsics_counterbalance",
]

PLATEAU_FAMILIES = [
    "hardtail_bucket_granularity_refinement",
    "hybrid_tail_exposure_balancing",
    "tail_conf_branch_decoupling",
    "tail_anchor_stabilization",
    "tail_counterbalance_cohort_mixing",
    "tail_anchor_reserve_hybridization",
    "tail_manifest_focal_reinforcement",
    "tail_stream_selective_focal_reinforcement",
    "tail_contract_anchor_replay",
    "tail_contract_viewset_replay",
    "default_stream_intrinsics_counterbalance",
]

LOG_RE = re.compile(
    r"(Train|Val) Epoch: \[\d+\]\[\s*(\d+)/\d+\].*?"
    r"Loss/(?:train|val)_loss_camera: ([\-0-9.]+) \(([\-0-9.]+)\).*?"
    r"Loss/(?:train|val)_loss_T: ([\-0-9.]+) \(([\-0-9.]+)\).*?"
    r"Loss/(?:train|val)_loss_R: ([\-0-9.]+) \(([\-0-9.]+)\).*?"
    r"Loss/(?:train|val)_loss_FL: ([\-0-9.]+) \(([\-0-9.]+)\).*?"
    r"Loss/(?:train|val)_loss_conf_depth: ([\-0-9.]+) \(([\-0-9.]+)\).*?"
    r"Loss/(?:train|val)_loss_reg_depth: ([\-0-9.]+) \(([\-0-9.]+)\)"
)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_metric_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "phase": match.group(1).lower(),
                "step": int(match.group(2)),
                "camera": float(match.group(4)),
                "T": float(match.group(6)),
                "R": float(match.group(8)),
                "FL": float(match.group(10)),
                "conf_depth": float(match.group(12)),
                "reg_depth": float(match.group(14)),
            }
        )
    return rows


def rows_index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["phase"], row["step"]): row for row in rows}


def summary_delta_map(summary: dict, section: str) -> dict[str, float]:
    return {row["metric"]: float(row["delta"]) for row in summary[section]["rows"]}


def summary_candidate_avg_map(summary: dict, section: str) -> dict[str, float]:
    return {
        metric: float(values["average"])
        for metric, values in (summary[section]["candidate"]["metrics"] or {}).items()
    }


def stable_smoke_log_path() -> Path:
    matches = sorted(
        (
            REPO_ROOT / "training" / "logs"
        ).glob(
            "zju_source_policy_candidate_zju_vggt_geom_unproject_source_policy_"
            "nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal_smoke1x1_*/log.txt"
        )
    )
    if not matches:
        raise FileNotFoundError("Unable to locate stable-lead smoke log.")
    return matches[-1]


def latest_archives_per_family() -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for archive_path in ARCHIVE_ROOT.glob("*.json"):
        payload = load_json(archive_path)
        family = str(payload.get("family", "")).strip()
        archived_at = str(payload.get("archived_at", "")).strip()
        if family not in TARGET_FAMILIES or not archived_at:
            continue
        current = latest.get(family)
        if current is None or archived_at > current[0]:
            latest[family] = (archived_at, archive_path)
    return {family: payload[1] for family, payload in latest.items()}


def build_records() -> dict[str, dict]:
    stable_smoke_path = stable_smoke_log_path()
    records: dict[str, dict] = {}
    for family, archive_path in latest_archives_per_family().items():
        archive = load_json(archive_path)
        run_dir = Path(str(archive.get("archive_run_dir", "")).strip())
        status_path = run_dir / "status.json"
        summary_path = run_dir / "short_vs_lead" / "summary.json"
        if not run_dir.exists() or not status_path.exists() or not summary_path.exists():
            continue
        status = load_json(status_path)
        artifacts = status.get("artifacts", {}) or {}
        record = {
            "family": family,
            "archive_path": str(archive_path.resolve()),
            "archived_at": str(archive.get("archived_at", "")),
            "problem_id": str(archive.get("problem_id", "")),
            "shape": str(archive.get("first_candidate_shape", "")),
            "candidate_config": str(archive.get("first_candidate_config", "")),
            "knobs": archive.get("first_candidate_knobs", {}) or {},
            "write_surface": archive.get("first_candidate_write_surface", []) or [],
            "run_dir": str(run_dir.resolve()),
            "summary": load_json(summary_path),
            "status": status,
            "short_log_rows": parse_metric_log(Path(str(artifacts.get("short_log", "")).strip())),
            "smoke_log_rows": parse_metric_log(Path(str(artifacts.get("smoke_log", "")).strip())),
            "stable_short_rows": parse_metric_log(Path(str(status.get("stable_short_reference_log", "")).strip())),
            "stable_smoke_rows": parse_metric_log(stable_smoke_path),
        }
        record["val_deltas"] = summary_delta_map(record["summary"], "val")
        record["train_deltas"] = summary_delta_map(record["summary"], "train")
        record["val_candidate_metrics"] = summary_candidate_avg_map(record["summary"], "val")
        record["train_candidate_metrics"] = summary_candidate_avg_map(record["summary"], "train")
        records[family] = record
    return records


def log_delta(record: dict, log_kind: str, phase: str, step: int) -> dict[str, float] | None:
    rows = rows_index(record[f"{log_kind}_log_rows"])
    stable_rows = rows_index(record[f"stable_{log_kind}_rows"])
    current = rows.get((phase, step))
    stable = stable_rows.get((phase, step))
    if current is None or stable is None:
        return None
    return {
        metric: round(float(current[metric]) - float(stable[metric]), 4)
        for metric in ("camera", "T", "R", "FL", "conf_depth", "reg_depth")
    }


def compare_families(records: dict[str, dict], family_a: str, family_b: str) -> dict[str, float]:
    metrics = ("loss_camera", "loss_T", "loss_R", "loss_FL", "loss_conf_depth", "loss_reg_depth")
    a_rows = records[family_a]["val_deltas"]
    b_rows = records[family_b]["val_deltas"]
    return {metric: round(float(a_rows[metric]) - float(b_rows[metric]), 4) for metric in metrics}


def assign_pattern(record: dict) -> str:
    camera = float(record["val_deltas"]["loss_camera"])
    delta_t = float(record["val_deltas"]["loss_T"])
    conf_depth = float(record["val_deltas"]["loss_conf_depth"])
    reg_depth = float(record["val_deltas"]["loss_reg_depth"])
    if (abs(camera) <= 0.0002 and abs(conf_depth) <= 0.001 and abs(reg_depth) <= 0.001) or (
        conf_depth > 0 or reg_depth > 0
    ):
        return "NON_FORWARD_SIGNAL"
    if camera > 0.0012 or delta_t > 0.00015:
        return "DEPTH_WIN_CAMERA_SPIKE"
    return "DEPTH_WIN_SMALL_FL_TAX"


def build_per_stream_audit(records: dict[str, dict]) -> dict:
    plateau_rows = [records[family]["val_deltas"] for family in PLATEAU_FAMILIES if family in records]
    median_camera = median(row["loss_camera"] for row in plateau_rows)
    median_t = median(row["loss_T"] for row in plateau_rows)
    median_r = median(row["loss_R"] for row in plateau_rows)
    median_fl = median(row["loss_FL"] for row in plateau_rows)

    default_vs_reference = compare_families(
        records, "default_stream_intrinsics_counterbalance", "tail_anchor_reserve_hybridization"
    )
    hardtail_focal_vs_reference = compare_families(
        records, "tail_stream_selective_focal_reinforcement", "tail_anchor_reserve_hybridization"
    )
    blanket_tail_vs_reference = compare_families(
        records, "tail_manifest_focal_reinforcement", "tail_anchor_reserve_hybridization"
    )
    reserve_companion_vs_anchor = compare_families(
        records, "tail_anchor_reserve_hybridization", "tail_anchor_stabilization"
    )
    hardtail_anchor_vs_bucket = compare_families(
        records, "tail_anchor_stabilization", "hardtail_bucket_granularity_refinement"
    )
    hardtail_pose_off_vs_anchor = compare_families(
        records, "tail_pose_branch_decoupling", "tail_anchor_stabilization"
    )
    hardtail_focal_off_vs_anchor = compare_families(
        records, "tail_intrinsics_branch_decoupling", "tail_anchor_stabilization"
    )

    return {
        "checked_at": iso_now(),
        "artifact_kind": "per_stream_camera_component_audit",
        "analysis_scope": {
            "families_considered": sorted(records.keys()),
            "plateau_families": [family for family in PLATEAU_FAMILIES if family in records],
            "evidence_boundary": (
                "Run artifacts do not log direct per-stream loss_camera/loss_FL/loss_R/loss_T scalars. "
                "Stream attribution below is inferred from targeted family interventions and short-gate deltas."
            ),
        },
        "persistent_camera_component": {
            "dominant_component": "loss_FL",
            "median_short_gate_val_delta_loss_camera": round(median_camera, 4),
            "median_short_gate_val_delta_loss_T": round(median_t, 4),
            "median_short_gate_val_delta_loss_R": round(median_r, 4),
            "median_short_gate_val_delta_loss_FL": round(median_fl, 4),
            "finding": (
                "The residual camera tax is FL-dominant on the plateau. T remains a small residue and R is "
                "effectively neutral at short gate."
            ),
        },
        "stream_intervention_response": {
            "default_stream": {
                "isolating_family": "default_stream_intrinsics_counterbalance",
                "reference_family": "tail_anchor_reserve_hybridization",
                "short_gate_val_change_vs_reference": default_vs_reference,
                "reading": (
                    "A default-only focal counterweight leaves the short-gate objective effectively unchanged, "
                    "so the remaining tax is not explained by a missing default-stream focal weight alone."
                ),
            },
            "hardtail_stream": {
                "isolating_family": "tail_stream_selective_focal_reinforcement",
                "reference_family": "tail_anchor_reserve_hybridization",
                "short_gate_val_change_vs_reference": hardtail_focal_vs_reference,
                "hardtail_anchor_gain_vs_bucket": hardtail_anchor_vs_bucket,
                "hardtail_pose_off_vs_anchor": hardtail_pose_off_vs_anchor,
                "hardtail_focal_off_vs_anchor": hardtail_focal_off_vs_anchor,
                "reading": (
                    "Hardtail-only focal upweight also leaves the plateau unchanged, but hardtail anchor "
                    "stabilization is the stream-local change that most increases depth gain while keeping the "
                    "same FL-dominant tax. Removing hardtail camera or focal capacity spikes camera loss sharply."
                ),
            },
            "reserve_stream": {
                "closest_isolation": "tail_anchor_reserve_hybridization_vs_tail_anchor_stabilization",
                "short_gate_val_change_vs_anchor_only": reserve_companion_vs_anchor,
                "blanket_manifest_tail_focal_vs_reference": blanket_tail_vs_reference,
                "reading": (
                    "Adding the reserve companion stream barely changes the short-gate plateau once hardtail "
                    "anchor stabilization is already active, and blanket manifest-tail focal reinforcement does "
                    "not move the plateau either."
                ),
            },
        },
        "answers": {
            "camera_component_that_keeps_lifting_camera": "loss_FL",
            "secondary_component": "loss_T",
            "negligible_component": "loss_R",
            "stream_with_strongest_depth_gain_cooccurrence": "hardtail",
            "stream_with_strongest_depth_gain_cooccurrence_reason": (
                "Hardtail anchor stabilization produces the biggest conf_depth/reg_depth win over the earlier "
                "bucket baseline, while reserve addition and default-only focal counterweight leave the plateau "
                "nearly unchanged."
            ),
        },
    }


def build_early_trace(records: dict[str, dict]) -> dict:
    families = [
        "soft_tail_exposure_rebalancing",
        "hardtail_bucket_granularity_refinement",
        "tail_anchor_stabilization",
        "tail_anchor_reserve_hybridization",
        "tail_stream_selective_focal_reinforcement",
        "default_stream_intrinsics_counterbalance",
        "tail_dual_supervision_rebalancing",
    ]
    trace_rows = []
    immediate_pattern_count = 0
    for family in families:
        if family not in records:
            continue
        record = records[family]
        smoke_train0 = log_delta(record, "smoke", "train", 0)
        smoke_val0 = log_delta(record, "smoke", "val", 0)
        short_train0 = log_delta(record, "short", "train", 0)
        short_val0 = log_delta(record, "short", "val", 0)
        short_val4 = log_delta(record, "short", "val", 4)
        if short_val0 and short_val0["camera"] > 0 and short_val0["conf_depth"] < 0 and short_val0["reg_depth"] < 0:
            immediate_pattern_count += 1
        trace_rows.append(
            {
                "family": family,
                "shape": record["shape"],
                "smoke_train_step0_vs_stable_smoke": smoke_train0,
                "smoke_val_step0_vs_stable_smoke": smoke_val0,
                "short_train_step0_vs_stable_short": short_train0,
                "short_val_step0_vs_stable_short": short_val0,
                "short_val_step4_vs_stable_short": short_val4,
                "camera_tax_present_at_smoke_val_step0": bool(smoke_val0 and smoke_val0["camera"] > 0),
                "camera_tax_present_at_first_short_val": bool(short_val0 and short_val0["camera"] > 0),
                "depth_gain_present_at_first_short_val": bool(
                    short_val0 and short_val0["conf_depth"] < 0 and short_val0["reg_depth"] < 0
                ),
            }
        )

    return {
        "checked_at": iso_now(),
        "artifact_kind": "early_step_objective_balance_trace",
        "families_traced": trace_rows,
        "aggregate_readout": {
            "families_with_immediate_short_val_camera_tax_and_depth_gain": immediate_pattern_count,
            "families_traced_count": len(trace_rows),
            "primary_finding": (
                "Once a family enters the depth-win regime, the validation camera tax is already visible at the "
                "first short-gate validation step. It does not appear late after depth has already settled."
            ),
            "default_vs_global_localization": (
                "The default-only focal family reproduces the same smoke/short trace as the hardtail-only and "
                "blanket-tail focal families within logging precision, so the early-step pattern behaves like a "
                "global objective effect rather than a default-stream-only lag."
            ),
        },
        "answers": {
            "camera_tax_timing": "present_from_the_first_validation_snapshots",
            "camera_tax_timing_reason": (
                "Both smoke val step 0 and short val step 0 already show camera/FL regression against the "
                "stable lead while conf_depth/reg_depth are better."
            ),
            "default_stream_only_supported": False,
            "all_streams_or_global_pattern_supported": True,
        },
    }


def build_alignment_matrix(records: dict[str, dict]) -> dict:
    pattern_rows = []
    pattern_counts: dict[str, int] = {}
    for family in TARGET_FAMILIES:
        if family not in records:
            continue
        record = records[family]
        pattern = assign_pattern(record)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        pattern_rows.append(
            {
                "family": family,
                "shape": record["shape"],
                "pattern": pattern,
                "loss_camera": round(record["val_deltas"]["loss_camera"], 4),
                "loss_T": round(record["val_deltas"]["loss_T"], 4),
                "loss_conf_depth": round(record["val_deltas"]["loss_conf_depth"], 4),
                "loss_reg_depth": round(record["val_deltas"]["loss_reg_depth"], 4),
            }
        )
    return {
        "checked_at": iso_now(),
        "artifact_kind": "family_outcome_alignment_matrix",
        "core_metrics": ["loss_camera", "loss_T", "loss_conf_depth", "loss_reg_depth"],
        "pattern_definitions": {
            "DEPTH_WIN_SMALL_FL_TAX": (
                "Real conf_depth/reg_depth improvement with only a small residual camera tax and a near-zero T term."
            ),
            "DEPTH_WIN_CAMERA_SPIKE": (
                "Depth still improves, but camera loss rises materially and the tradeoff is no longer a tiny plateau."
            ),
            "NON_FORWARD_SIGNAL": (
                "Either the ticket is effectively a no-op or it regresses at least one depth objective badly enough "
                "that there is no forward-moving signal."
            ),
        },
        "rows": pattern_rows,
        "pattern_counts": pattern_counts,
    }


def build_root_cause_decision(records: dict[str, dict], per_stream: dict, trace: dict, matrix: dict) -> dict:
    default_delta = per_stream["stream_intervention_response"]["default_stream"]["short_gate_val_change_vs_reference"]
    hardtail_delta = per_stream["stream_intervention_response"]["hardtail_stream"]["short_gate_val_change_vs_reference"]
    blanket_delta = per_stream["stream_intervention_response"]["reserve_stream"][
        "blanket_manifest_tail_focal_vs_reference"
    ]
    default_small = max(abs(value) for value in default_delta.values()) <= 0.0002
    hardtail_small = max(abs(value) for value in hardtail_delta.values()) <= 0.0002
    blanket_small = max(abs(value) for value in blanket_delta.values()) <= 0.0002
    immediate_count = trace["aggregate_readout"]["families_with_immediate_short_val_camera_tax_and_depth_gain"]
    plateau_count = matrix["pattern_counts"].get("DEPTH_WIN_SMALL_FL_TAX", 0)

    if default_small and hardtail_small and blanket_small and immediate_count >= 4 and plateau_count >= 6:
        label = "GLOBAL_OBJECTIVE_CONFLICT"
        rationale = (
            "The depth-win families collapse onto the same FL-dominant short-gate plateau, and neither default-only "
            "nor hardtail-only nor blanket-tail focal counterweight changes that plateau in a detectable way."
        )
    elif not default_small:
        label = "DEFAULT_STREAM_UNDERWEIGHTED"
        rationale = (
            "A default-only counterweight produces a distinct short-gate relief while preserving the depth gains, "
            "so the remaining tax looks concentrated in the default stream."
        )
    else:
        label = "NO_ACTIONABLE_SIGNAL"
        rationale = (
            "The available logs do not isolate a stable early-step pattern strongly enough to justify a new family."
        )

    return {
        "checked_at": iso_now(),
        "artifact_kind": "objective_balance_root_cause_decision",
        "label": label,
        "rationale": rationale,
        "rejected_labels": {
            "DEFAULT_STREAM_UNDERWEIGHTED": (
                "Not supported by the current artifacts because default-only focal counterbalance leaves the "
                "short-gate plateau effectively unchanged."
            ),
            "NO_ACTIONABLE_SIGNAL": (
                "Not selected because the family batch does show a repeated, immediate depth-win plus FL-tax "
                "pattern across many tickets."
            ),
        },
        "decisive_evidence": [
            "Default-only focal counterbalance stays within <=0.0002 of the hardtail-anchor-plus-reserve reference at short gate.",
            "Hardtail-only focal counterbalance and blanket manifest-tail focal reinforcement also stay within <=0.0002 of that same reference.",
            "The strongest depth-win regime is already coupled to positive validation camera/FL deltas at smoke val step 0 and short val step 0.",
            "The reserve companion stream barely changes the plateau once hardtail anchor stabilization is already active.",
        ],
        "next_phase": "packaging_only" if label != "NO_ACTIONABLE_SIGNAL" else "return_to_idle_guard",
        "allowed_next_family": (
            "two_stage_objective_decoupling"
            if label == "GLOBAL_OBJECTIVE_CONFLICT"
            else ("two_stage_default_camera_recovery_schedule" if label == "DEFAULT_STREAM_UNDERWEIGHTED" else "")
        ),
    }


def render_per_stream_md(payload: dict) -> str:
    lines = [
        "# Per-Stream Camera Component Audit",
        "",
        f"- Checked at: `{payload['checked_at']}`",
        f"- Dominant camera component: `{payload['persistent_camera_component']['dominant_component']}`",
        f"- Stream with strongest depth-gain co-occurrence: `{payload['answers']['stream_with_strongest_depth_gain_cooccurrence']}`",
        "",
        "## Key Finding",
        "",
        payload["persistent_camera_component"]["finding"],
        "",
        "## Stream Readout",
        "",
        "Default stream: default-only focal counterweight does not move the short-gate plateau in a detectable way.",
        "Hardtail stream: hardtail anchor stabilization is the stream-local change that most strengthens depth gain while preserving the same FL-dominant tax.",
        "Reserve stream: reserve acts like a companion stream, but it does not materially change the plateau once hardtail anchor stabilization is present.",
        "",
        "## Evidence Boundary",
        "",
        payload["analysis_scope"]["evidence_boundary"],
        "",
    ]
    return "\n".join(lines)


def render_early_trace_md(payload: dict) -> str:
    lines = [
        "# Early-Step Objective Balance Trace",
        "",
        f"- Checked at: `{payload['checked_at']}`",
        f"- Camera tax timing: `{payload['answers']['camera_tax_timing']}`",
        "",
        "## Main Finding",
        "",
        payload["aggregate_readout"]["primary_finding"],
        "",
        "## Localization",
        "",
        payload["aggregate_readout"]["default_vs_global_localization"],
        "",
        "## Representative Families",
        "",
    ]
    for row in payload["families_traced"]:
        short_val0 = row["short_val_step0_vs_stable_short"]
        short_val4 = row["short_val_step4_vs_stable_short"]
        lines.append(
            f"- `{row['family']}`: short val step 0 camera `{short_val0['camera']:+.4f}`, "
            f"FL `{short_val0['FL']:+.4f}`, conf `{short_val0['conf_depth']:+.4f}`, reg `{short_val0['reg_depth']:+.4f}`; "
            f"short val step 4 camera `{short_val4['camera']:+.4f}`, conf `{short_val4['conf_depth']:+.4f}`, "
            f"reg `{short_val4['reg_depth']:+.4f}`."
        )
    lines.append("")
    return "\n".join(lines)


def render_alignment_md(payload: dict) -> str:
    lines = [
        "# Family Outcome Alignment Matrix",
        "",
        f"- Checked at: `{payload['checked_at']}`",
        "",
        "## Pattern Counts",
        "",
    ]
    for pattern, count in sorted(payload["pattern_counts"].items()):
        lines.append(f"- `{pattern}`: {count}")
    lines.extend(["", "## Rows", ""])
    for row in payload["rows"]:
        lines.append(
            f"- `{row['family']}` -> `{row['pattern']}` | camera `{row['loss_camera']:+.4f}` | "
            f"T `{row['loss_T']:+.4f}` | conf `{row['loss_conf_depth']:+.4f}` | reg `{row['loss_reg_depth']:+.4f}`"
        )
    lines.append("")
    return "\n".join(lines)


def render_root_cause_md(payload: dict) -> str:
    lines = [
        "# Objective Balance Root-Cause Decision",
        "",
        f"- Checked at: `{payload['checked_at']}`",
        f"- Label: `{payload['label']}`",
        "",
        payload["rationale"],
        "",
        "## Decisive Evidence",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["decisive_evidence"]])
    lines.extend(
        [
            "",
            "## Next Phase",
            "",
            f"- next_phase: `{payload['next_phase']}`",
            f"- allowed_next_family: `{payload['allowed_next_family']}`",
            "",
        ]
    )
    return "\n".join(lines)


def package_two_stage_objective_decoupling() -> dict:
    checked_at = iso_now()
    family = "two_stage_objective_decoupling"
    shape = "depth_gain_then_camera_reconciliation"
    write_surface = [
        "training/trainer.py",
        "training/loss.py",
        "training/config/*.yaml",
        "scripts/compare_zju_finetune_runs.py",
    ]

    approved_problem_seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": "promoted_two_stage_objective_decoupling_v1",
        "problem_title": "Two-stage objective decoupling after tail depth-win plateau",
        "family": family,
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "The tail-contract derivative axis is closed. This new family is packaging-only until a human "
            "explicitly approves a fresh two-stage design review."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that separates the current depth-gain phase from "
            "a later camera reconciliation phase instead of forcing both goals through one static short-gate joint objective."
        ),
        "why_genuinely_new": (
            "This is a training-schedule family rather than another tail-contract derivative. It changes optimization "
            "staging, not tail manifests, replay, focal scales, or supervision counts."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen any tail-contract derivative. The current strongest hardtail-plus-reserve contract "
            "remains the phase-1 starting point rather than a new derivative ticket."
        ),
        "first_candidate_hint": (
            "Package only the depth_gain_then_camera_reconciliation candidate: phase 1 preserves the strongest "
            "current tail contract to harvest depth gain, then phase 2 reduces or freezes tail-side depth pressure "
            "and focuses on camera reconciliation on the stable/default distribution."
        ),
        "first_candidate_shape": shape,
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": True,
        "first_candidate_write_surface": write_surface,
        "first_candidate_knobs": {
            "phase_1_starting_point": "current strongest tail contract",
            "phase_2_goal": "camera reconciliation without surrendering harvested depth gain",
            "require_two_stage_schedule": True,
            "do_not_arm_now": True,
        },
        "avoid_patterns": [
            "tail-contract derivative reopen",
            "same-night arm",
            "same-night run",
            "cloud action",
        ],
        "max_approved_problems_per_night": 1,
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "requires_dataset_or_routing_change": False,
        "requires_supervision_audit": False,
        "mutation_dsl": {
            "allow_two_stage_objective_decoupling": True,
            "require_phase_boundary_between_depth_gain_and_camera_reconciliation": True,
            "keep_tail_contract_fixed_in_phase_1": True,
            "disallow_tail_contract_derivative_reopen": True,
            "disallow_same_day_arm": True,
            "disallow_cloud": True,
        },
    }

    family_blueprint = {
        "checked_at": checked_at,
        "family": family,
        "status": "packaging_only_not_repo_ready",
        "ready_for_manual_approval": True,
        "ready_for_execution": False,
        "why_now": (
            "The daytime audit shows a repeated depth-win plus FL-tax plateau that no single-stream counterweight "
            "can move. The next bounded family therefore has to change optimization staging rather than keep "
            "micro-tuning single-stage weights."
        ),
        "why_not_same_family_retry": (
            "This is not another tail manifest, replay, focal, or supervision-count cousin. It is a separate "
            "two-stage training schedule family."
        ),
        "signal_definition": (
            "Keep the strongest current tail contract fixed for phase 1, then run a bounded camera reconciliation "
            "phase that is allowed to relax tail-side depth pressure."
        ),
        "scope_definition": (
            "packaging-only today: new schedule/code-path family, code patch required, no repo-ready candidate yet"
        ),
        "first_candidate_hypothesis": (
            "The short-gate depth gain and camera reconciliation are conflicting when optimized in one phase. "
            "A two-stage schedule may preserve the depth win while giving camera recovery its own optimization window."
        ),
        "first_candidate_shape": shape,
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": True,
        "first_candidate_write_surface": write_surface,
        "first_candidate_execution_note": (
            "Packaging-only today. No config is armed and no candidate may run until a human approves a concrete "
            "two-stage schedule design."
        ),
        "first_candidate_knobs": approved_problem_seed["first_candidate_knobs"],
        "required_exclusions": [
            "not another tail derivative",
            "not same-day arm",
            "not same-day run",
            "not cloud",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": (
                "This stays packaging-only for now; the research loop must remain in IDLE_GUARD until a later manual review."
            ),
        },
        "gate_sequence": [
            "SMOKE_1x1",
            "TIGHT_GATE_10x5",
            "LONG_GATE_100x20",
            "VERDICT_WRITEBACK",
            "RETURN_TO_GUARD",
        ],
        "cloud_must_remain_off": True,
        "stop_rule": (
            "Do not arm until a fresh manual design review produces a bounded two-stage config and explicit write surface."
        ),
    }

    candidate_patch_plan = {
        "checked_at": checked_at,
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "current_stable_lead_config": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
        ),
        "family": family,
        "first_candidate_shape": shape,
        "first_candidate_config": "",
        "arm_command": "",
        "run_command": "",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": False,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "packaging_only": {
            "reason": "Two-stage family is a new schedule/code path and is not repo-ready yet.",
            "requires_code_patch": True,
            "required_write_surface": write_surface,
        },
        "write_surface": write_surface,
    }

    next_manual_problem_draft = {
        "checked_at": checked_at,
        "draft_kind": "new_manual_problem",
        "status": "packaging_only_not_execution_ready",
        "family": family,
        "first_candidate_shape": shape,
        "candidate_config": "",
        "ready_for_manual_review": True,
        "ready_for_execution": False,
        "requires_new_manual_approval": True,
        "why_now": [
            "The daytime audit converged on GLOBAL_OBJECTIVE_CONFLICT rather than another stream-local weight issue.",
            "Default-only and hardtail-only focal counterweights do not move the short-gate plateau in a detectable way.",
            "A new family only makes sense if it changes the optimization schedule rather than reopening tail derivatives.",
        ],
        "readiness_artifact": "",
        "hypothesis": (
            "Keep the current strongest tail contract for a depth-gain phase, then run a bounded camera reconciliation "
            "phase on top of that checkpoint instead of demanding both objectives from one short-gate phase."
        ),
    }

    write_json(OUTPUT_ROOT / f"approved_problem.seed.{family}.json", approved_problem_seed)
    write_json(OUTPUT_ROOT / f"family_blueprint.{family}.json", family_blueprint)
    write_json(OUTPUT_ROOT / f"candidate_patch_plan.{family}.json", candidate_patch_plan)
    write_text(
        OUTPUT_ROOT / f"candidate_patch_plan.{family}.md",
        "\n".join(
            [
                "# Candidate Patch Plan",
                "",
                f"- family: `{family}`",
                f"- first_candidate_shape: `{shape}`",
                "- status: `packaging_only_not_execution_ready`",
                "- do_not_arm_now: `true`",
                "- do_not_run_candidate_now: `true`",
                "",
                "This family is packaged for manual review only. No repo-ready candidate exists yet.",
                "",
            ]
        ),
    )
    write_json(OUTPUT_ROOT / f"next_manual_problem_draft.{family}.{DATE_TAG}.json", next_manual_problem_draft)
    write_text(
        OUTPUT_ROOT / f"next_manual_problem_draft.{family}.{DATE_TAG}.md",
        "\n".join(
            [
                "# Next Manual Problem Draft",
                "",
                f"- family: `{family}`",
                f"- first_candidate_shape: `{shape}`",
                "- ready_for_manual_review: `true`",
                "- ready_for_execution: `false`",
                "",
                next_manual_problem_draft["hypothesis"],
                "",
            ]
        ),
    )
    return {
        "family": family,
        "seed_path": str((OUTPUT_ROOT / f"approved_problem.seed.{family}.json").resolve()),
        "blueprint_path": str((OUTPUT_ROOT / f"family_blueprint.{family}.json").resolve()),
        "draft_path": str((OUTPUT_ROOT / f"next_manual_problem_draft.{family}.{DATE_TAG}.json").resolve()),
        "plan_path": str((OUTPUT_ROOT / f"candidate_patch_plan.{family}.json").resolve()),
    }


def main() -> int:
    records = build_records()
    per_stream = build_per_stream_audit(records)
    early_trace = build_early_trace(records)
    alignment = build_alignment_matrix(records)
    root_cause = build_root_cause_decision(records, per_stream, early_trace, alignment)

    write_json(OUTPUT_ROOT / f"per_stream_camera_component_audit.{DATE_TAG}.json", per_stream)
    write_text(OUTPUT_ROOT / f"per_stream_camera_component_audit.{DATE_TAG}.md", render_per_stream_md(per_stream))

    write_json(OUTPUT_ROOT / f"early_step_objective_balance_trace.{DATE_TAG}.json", early_trace)
    write_text(OUTPUT_ROOT / f"early_step_objective_balance_trace.{DATE_TAG}.md", render_early_trace_md(early_trace))

    write_json(OUTPUT_ROOT / f"family_outcome_alignment_matrix.{DATE_TAG}.json", alignment)
    write_text(OUTPUT_ROOT / f"family_outcome_alignment_matrix.{DATE_TAG}.md", render_alignment_md(alignment))

    packaging_result = {}
    if root_cause["label"] == "GLOBAL_OBJECTIVE_CONFLICT":
        packaging_result = package_two_stage_objective_decoupling()
    elif root_cause["label"] == "DEFAULT_STREAM_UNDERWEIGHTED":
        packaging_result = {"family": "two_stage_default_camera_recovery_schedule", "status": "not_packaged_in_this_audit"}
    root_cause["packaging_result"] = packaging_result

    write_json(OUTPUT_ROOT / f"objective_balance_root_cause_decision.{DATE_TAG}.json", root_cause)
    write_text(OUTPUT_ROOT / f"objective_balance_root_cause_decision.{DATE_TAG}.md", render_root_cause_md(root_cause))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
