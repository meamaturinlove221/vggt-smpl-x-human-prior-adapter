import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
RESEARCH_ROOT = OUTPUT_ROOT / "zju_source_policy_research_loop"
AUTOLOOP_ROOT = OUTPUT_ROOT / "autoloop_teacher_geometry_anchor_specific_correspondence"

FAMILY = "teacher_geometry_anchor_specific_render_artifact_audit"
FIRST_SHAPE = "stablelead_anchor_specific_render_artifact_layer_suppression_maskedhuman_v1"
KEY_CASES = [
    "CoreView_390_frame_001170_Camera_B4",
    "CoreView_390_frame_000600_Camera_B4",
    "CoreView_390_frame_001080_Camera_B14",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    checked_at = datetime.now().astimezone().isoformat()
    research_status = _read_json(RESEARCH_ROOT / "research_loop_status.json")
    watch = _read_json(OUTPUT_ROOT / "zju_source_policy_research_watch" / "latest_watch_snapshot.json")
    task_plan = _read_json(OUTPUT_ROOT / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json")
    prior_result = _read_json(RESEARCH_ROOT / "teacher_geometry_anchor_specific_correspondence_audit_result.json")
    prior_postmortem = _read_json(RESEARCH_ROOT / "teacher_geometry_anchor_specific_correspondence_audit_postmortem.json")
    prior_best = _read_json(AUTOLOOP_ROOT / "best_state.json")

    summary_path = Path(prior_best["summary_json"])
    summary = _read_json(summary_path)
    variant = str(prior_best["compare"]["variant"])
    checkpoint = str(summary["checkpoint"])
    eval_root = OUTPUT_ROOT / FAMILY / f"evidence_eval.{datetime.now().strftime('%Y%m%d')}"
    eval_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    baseline_rows = [row for row in summary["rows"] if row["variant"] == "baseline_depth_unproject"]
    baseline_by_case = {row["case_id"]: row for row in baseline_rows}
    for case_id in KEY_CASES:
        base_row = baseline_by_case[case_id]
        manifest_rows.append(
            {
                "seq_name": base_row["case"]["seq_name"],
                "frame_id": base_row["case"]["frame_id"],
                "target_camera": base_row["case"]["target_camera"],
                "source_cameras": base_row["case"]["source_cameras"],
            }
        )

    manifest_path = eval_root / "render_artifact_cases_manifest.json"
    proxy_config_path = eval_root / "anchor_specific_best_proxy_config.json"
    evidence_summary_path = RESEARCH_ROOT / f"{FAMILY}_evidence_summary.json"
    seed_path = RESEARCH_ROOT / f"{FAMILY}_problem_seed.json"
    blueprint_path = RESEARCH_ROOT / f"{FAMILY}_blueprint.json"
    manual_packet_json_path = RESEARCH_ROOT / f"{FAMILY}_manual_packet.json"
    manual_packet_md_path = RESEARCH_ROOT / f"{FAMILY}_manual_packet.md"
    key_panels_zip_path = RESEARCH_ROOT / f"{FAMILY}_key_panels.zip"

    _write_json(manifest_path, {"cases": manifest_rows})
    _write_json(proxy_config_path, prior_best["mutation"]["proxy_config"])

    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"),
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(eval_root),
            "--checkpoint",
            checkpoint,
            "--case-set",
            "cases",
            "--variants",
            variant,
            "--device",
            "cpu",
            "--proxy-config-json",
            str(proxy_config_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "score_render_artifact_progress.py"),
            "--summary-json",
            str(eval_root / "summary.json"),
            "--variant",
            variant,
            "--output-json",
            str(evidence_summary_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    evidence_summary = _read_json(evidence_summary_path)

    problem_seed = {
        "checked_at": checked_at,
        "family": FAMILY,
        "prior_family": prior_result["family"],
        "problem_statement": (
            "Correspondence-side overrides already improve some rebound cases, but the remaining failure still looks like "
            "inside-fg rehydrated render artifact: duplicated visible lobes, component rebound, and peak rebound persist even when "
            "off-body support stays low."
        ),
        "current_live_truth": {
            "state": research_status["state"],
            "approved_problem_present": research_status["approved_problem_present"],
            "same_family_retry_forbidden": research_status["same_family_retry_forbidden"],
            "cloud_must_remain_off": research_status["cloud_must_remain_off"],
        },
        "key_cases": KEY_CASES,
        "baseline_references": {
            "baseline_render_family": "teacher_geometry_anchor_specific_correspondence_audit",
            "best_state_json": str(AUTOLOOP_ROOT / "best_state.json"),
            "best_variant": variant,
        },
    }
    _write_json(seed_path, problem_seed)

    blueprint = {
        "checked_at": checked_at,
        "family": FAMILY,
        "first_candidate_shape": FIRST_SHAPE,
        "allowed_write_surface_now": [
            "scripts/evaluate_teacher_visual_lift_cases.py",
            "scripts/score_render_artifact_progress.py",
            "scripts/package_teacher_geometry_anchor_specific_render_artifact_problem.py",
            "scripts/prepare_teacher_geometry_anchor_specific_render_artifact_ready.py",
        ],
        "forbidden_actions_now": [
            "reopen teacher_geometry_anchor_specific_correspondence_audit",
            "create approved_problem.json",
            "arm",
            "run",
            "cloud",
            "training",
            "stable lead config changes",
        ],
        "audit_focus": [
            "inside-fg visible connected-component behavior",
            "primary-vs-secondary lobe competition",
            "multilayer overlap and duplicate lobe residuals",
            "render residual / hole / bridge-break failure modes",
        ],
        "key_cases": KEY_CASES,
    }
    _write_json(blueprint_path, blueprint)

    with zipfile.ZipFile(key_panels_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for case_id in KEY_CASES:
            case_dir = eval_root / case_id / "renders"
            for name in [
                "comparison_panel.png",
                f"target_baseline_candidate_renderdiff_fgmask_{variant}.png",
                f"fg_visible_components_colored_{variant}.png",
                f"fg_primary_vs_secondary_lobe_{variant}.png",
                f"fg_multilayer_overlap_heatmap_{variant}.png",
                f"fg_peak_map_{variant}.png",
                f"fg_hole_bridge_panel_{variant}.png",
                f"support_inside_fg_{variant}.png",
                f"support_outside_fg_{variant}.png",
                f"support_overlay_on_fg_{variant}.png",
            ]:
                path = case_dir / name
                if path.exists():
                    zf.write(path, arcname=f"{case_id}/{name}")

    per_case_lines = []
    for case in evidence_summary["per_case"]:
        per_case_lines.append(
            {
                "case_id": case["case_id"],
                "artifact_type": case["artifact_type"],
                "delta_fg_connected_components": case["delta_fg_connected_components"],
                "delta_fg_peak_count": case["delta_fg_peak_count"],
                "delta_fg_peak_count_after_render": case["delta_fg_peak_count_after_render"],
                "delta_fg_multilayer_overlap_ratio": case["delta_fg_multilayer_overlap_ratio"],
                "delta_fg_duplicate_lobe_ratio": case["delta_fg_duplicate_lobe_ratio"],
                "delta_fg_secondary_mass_ratio": case["delta_fg_secondary_mass_ratio"],
                "delta_largest_fg_visible_component_ratio": case["delta_largest_fg_visible_component_ratio"],
                "delta_masked_l1": case["delta_masked_l1"],
                "delta_masked_ssim": case["delta_masked_ssim"],
                "delta_off_body_support_ratio": case["delta_off_body_support_ratio"],
                "fg_visible_rgb_coverage_ratio": case["fg_visible_rgb_coverage_ratio"],
                "applied_anchor_rules": case["applied_anchor_rules"],
                "effective_source_subset": case["effective_source_subset"],
            }
        )

    manual_packet = {
        "checked_at": checked_at,
        "family": FAMILY,
        "summary": {
            "prior_family_status": prior_result["status"],
            "best_control_accept": prior_best["compare"]["control_accept"],
            "best_hero_accept": prior_best["compare"]["hero_accept"],
            "best_local20_accept": prior_best["compare"]["local20_accept"],
            "reason_to_switch": (
                "The best correspondence candidate improved some rebound cases but failed to generalize past control, "
                "so the remaining blocker is more likely inside-fg render artifact than correspondence rule search."
            ),
        },
        "facts": [
            "Best correspondence candidate only reached control_accept=true, hero_accept=false, local20_accept=false.",
            "CoreView_390_frame_001170_Camera_B4 still rebounded on peak count even after anchor-specific rule matching.",
            "Support-outside-fg stayed low, so the residual failure is no longer dominated by off-body leakage.",
            "FG coverage remained materially present, so the residual failure is not simply human erasure.",
            "Baseline-vs-candidate render panels still show highly similar duplicated inside-fg geometry structure.",
        ],
        "key_cases": per_case_lines,
        "evidence_summary_json": str(evidence_summary_path.relative_to(REPO_ROOT)),
        "key_panels_zip": str(key_panels_zip_path.relative_to(REPO_ROOT)),
        "watch_state": {
            "research_state": research_status["state"],
            "watch_review_packet": watch["research"]["summary"]["current_review_packet"],
            "task_mode_status": task_plan["task_mode_status"],
        },
    }
    _write_json(manual_packet_json_path, manual_packet)

    lines = [
        f"# {FAMILY}",
        "",
        f"- checked_at: `{checked_at}`",
        f"- prior family: `{prior_result['family']}`",
        f"- prior status: `{prior_result['status']}`",
        f"- best mutation: `{prior_best['mutation']['mutation_id']}`",
        f"- best variant: `{variant}`",
        f"- best gates: `control={prior_best['compare']['control_accept']}` / `hero={prior_best['compare']['hero_accept']}` / `local20={prior_best['compare']['local20_accept']}`",
        "",
        "## Why This Family",
        "",
        "Correspondence-side overrides already improved some rebound cases, but the remaining blocker does not look like missing support or off-body leakage anymore.",
        "",
        "## Key Facts",
    ]
    lines.extend(f"- {item}" for item in manual_packet["facts"])
    lines.extend(
        [
            "",
            "## Three-Case Evidence",
        ]
    )
    for case in per_case_lines:
        lines.extend(
            [
                f"- `{case['case_id']}`: `{case['artifact_type']}`; "
                f"`d_components={case['delta_fg_connected_components']}`, "
                f"`d_peaks={case['delta_fg_peak_count']}`, "
                f"`d_render_peaks={case['delta_fg_peak_count_after_render']}`, "
                f"`d_multilayer={case['delta_fg_multilayer_overlap_ratio']:.4f}`, "
                f"`d_duplicate_lobe={case['delta_fg_duplicate_lobe_ratio']:.4f}`, "
                f"`d_masked_l1={case['delta_masked_l1']:.4f}`, "
                f"`d_masked_ssim={case['delta_masked_ssim']:.4f}`, "
                f"`d_off_body={case['delta_off_body_support_ratio']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- evidence summary: `{evidence_summary_path.relative_to(REPO_ROOT)}`",
            f"- key panels zip: `{key_panels_zip_path.relative_to(REPO_ROOT)}`",
            f"- evidence eval root: `{eval_root.relative_to(REPO_ROOT)}`",
        ]
    )
    _write_md(manual_packet_md_path, "\n".join(lines))
    print(json.dumps({"evidence_summary": str(evidence_summary_path), "manual_packet": str(manual_packet_json_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
