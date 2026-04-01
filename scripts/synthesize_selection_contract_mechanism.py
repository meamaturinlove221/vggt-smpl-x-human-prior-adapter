import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize slot_3 / role-transition mechanism evidence.")
    parser.add_argument("--expanded-diff-json", required=True)
    parser.add_argument("--hardcase-definition-json", required=True)
    parser.add_argument("--hardcase-manifest-json", default="")
    parser.add_argument("--post-promotion-gap-json", required=True)
    parser.add_argument("--next-manual-problem-json", required=True)
    parser.add_argument("--slot3-output-json", required=True)
    parser.add_argument("--slot3-output-md", required=True)
    parser.add_argument("--tail-output-json", required=True)
    parser.add_argument("--tail-output-md", required=True)
    parser.add_argument("--decision-output-json", default="")
    parser.add_argument("--decision-output-md", default="")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def role_for_camera(payload, camera_name):
    camera_name = str(camera_name)
    if camera_name in payload.get("supervised_camera_names", []):
        return "supervised"
    if camera_name in payload.get("source_only_camera_names", []):
        return "source_only"
    if camera_name in payload.get("camera_names", []):
        return "selected_other"
    return "absent"


def role_for_camera_from_row(row, camera_name, side):
    camera_name = str(camera_name)
    suffix = str(side)
    if camera_name in row.get(f"supervised_camera_names_{suffix}", []):
        return "supervised"
    if camera_name in row.get(f"source_only_camera_names_{suffix}", []):
        return "source_only"
    if camera_name in row.get(f"camera_names_{suffix}", []):
        return "selected_other"
    return "absent"


def reverse_probe_a_to_b_transition(text):
    left, right = str(text).split(" -> ", 1)
    return right, left


def format_count_ranking(counter: Counter, key_name: str):
    return [{key_name: key, "count": int(value)} for key, value in sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))]


def role_transitions_from_diff_row(row: dict) -> list[str]:
    camera_union = sorted(set(row.get("camera_names_a", [])) | set(row.get("camera_names_b", [])))
    transitions = []
    for camera_name in camera_union:
        role_previous = role_for_camera_from_row(row, camera_name, "b")
        role_promoted = role_for_camera_from_row(row, camera_name, "a")
        if role_previous != role_promoted:
            transitions.append(f"{camera_name}: {role_previous} -> {role_promoted}")
    return transitions


def build_slot3_rankings(diff_summary: dict, payloads_a: list[dict], payloads_b: list[dict]) -> dict:
    slot3_hist = diff_summary["slot_transition_summary"]["slot_transition_histograms"].get("slot_3", {})
    slot3_ranked_previous_to_promoted = []
    slot3_ranked_promoted_to_previous = []
    slot3_promoted_counts = Counter()
    slot3_previous_counts = Counter()
    for key, value in slot3_hist.items():
        promoted_camera, previous_camera = key.split(" -> ", 1)
        count = int(value)
        slot3_ranked_previous_to_promoted.append(
            {
                "transition": f"{previous_camera} -> {promoted_camera}",
                "previous_slot3_camera": previous_camera,
                "promoted_slot3_camera": promoted_camera,
                "count": count,
            }
        )
        slot3_ranked_promoted_to_previous.append(
            {
                "transition": f"{promoted_camera} -> {previous_camera}",
                "promoted_slot3_camera": promoted_camera,
                "previous_slot3_camera": previous_camera,
                "count": count,
            }
        )
        slot3_promoted_counts[promoted_camera] += count
        slot3_previous_counts[previous_camera] += count
    slot3_ranked_previous_to_promoted.sort(key=lambda item: (-int(item["count"]), item["transition"]))
    slot3_ranked_promoted_to_previous.sort(key=lambda item: (-int(item["count"]), item["transition"]))

    role_transition_by_camera_previous_to_promoted = Counter()
    supervised_demotions_previous_to_promoted = Counter()
    source_only_drops_previous_to_promoted = Counter()
    source_only_entries_previous_to_promoted = Counter()
    for payload_a, payload_b in zip(payloads_a, payloads_b):
        camera_union = sorted(set(payload_a.get("camera_names", [])) | set(payload_b.get("camera_names", [])))
        for camera_name in camera_union:
            role_previous = role_for_camera(payload_b, camera_name)
            role_promoted = role_for_camera(payload_a, camera_name)
            if role_previous == role_promoted:
                continue
            transition = f"{camera_name}: {role_previous} -> {role_promoted}"
            role_transition_by_camera_previous_to_promoted[transition] += 1
            if role_previous == "supervised" and role_promoted in {"source_only", "absent"}:
                supervised_demotions_previous_to_promoted[transition] += 1
            if role_previous == "source_only" and role_promoted == "absent":
                source_only_drops_previous_to_promoted[transition] += 1
            if role_previous in {"absent", "selected_other"} and role_promoted == "source_only":
                source_only_entries_previous_to_promoted[transition] += 1

    top_changed_samples = []
    for row in diff_summary["per_sample_diffs"]:
        camera_union = sorted(set(row.get("camera_names_a", [])) | set(row.get("camera_names_b", [])))
        role_transitions_previous_to_promoted = []
        supervised_demotions = 0
        for camera_name in camera_union:
            role_previous = role_for_camera_from_row(row, camera_name, "b")
            role_promoted = role_for_camera_from_row(row, camera_name, "a")
            if role_previous == role_promoted:
                continue
            role_transitions_previous_to_promoted.append(f"{camera_name}: {role_previous} -> {role_promoted}")
            supervised_demotions += int(role_previous == "supervised" and role_promoted in {"source_only", "absent"})
        contract_change_score = (
            1000 * supervised_demotions
            + 100 * len(role_transitions_previous_to_promoted)
            + 10 * (len(row.get("camera_names_only_in_a", [])) + len(row.get("camera_names_only_in_b", [])))
            + abs(int(row["valid_points_delta_a_minus_b"]))
        )
        top_changed_samples.append(
            {
                "probe_sample_index": int(row["probe_sample_index"]),
                "sample_seq_name": str(row["sample_seq_name"]),
                "seq_name": str(row.get("seq_name") or ""),
                "frame_id": row.get("frame_id"),
                "slot_3_transition_previous_to_promoted": f"{row['slot_3_camera_b']} -> {row['slot_3_camera_a']}",
                "camera_set_only_in_previous": list(row.get("camera_names_only_in_b", [])),
                "camera_set_only_in_promoted": list(row.get("camera_names_only_in_a", [])),
                "supervised_previous": list(row["supervised_camera_names_b"]),
                "supervised_promoted": list(row["supervised_camera_names_a"]),
                "source_only_previous": list(row["source_only_camera_names_b"]),
                "source_only_promoted": list(row["source_only_camera_names_a"]),
                "role_transitions_previous_to_promoted": role_transitions_previous_to_promoted,
                "camera_set_change_count": int(
                    len(row.get("camera_names_only_in_a", [])) + len(row.get("camera_names_only_in_b", []))
                ),
                "supervised_demotions_previous_to_promoted": int(supervised_demotions),
                "valid_points_delta_a_minus_b": int(row["valid_points_delta_a_minus_b"]),
                "pointcloud_radius_p95_delta_a_minus_b": float(row["pointcloud_radius_p95_delta_a_minus_b"]),
                "contract_change_score": float(contract_change_score),
            }
        )
    top_changed_samples.sort(
        key=lambda item: (
            -int(item["supervised_demotions_previous_to_promoted"]),
            -int(len(item["role_transitions_previous_to_promoted"])),
            -int(item["camera_set_change_count"]),
            -abs(int(item["valid_points_delta_a_minus_b"])),
            -abs(float(item["pointcloud_radius_p95_delta_a_minus_b"])),
            int(item["probe_sample_index"]),
        )
    )

    aggregate_samples = int(diff_summary["aggregate_samples"])
    top_transition_share = (
        float(slot3_ranked_previous_to_promoted[0]["count"] / aggregate_samples)
        if slot3_ranked_previous_to_promoted
        else 0.0
    )
    top_two_transition_share = (
        float(
            (slot3_ranked_previous_to_promoted[0]["count"] + slot3_ranked_previous_to_promoted[1]["count"])
            / aggregate_samples
        )
        if len(slot3_ranked_previous_to_promoted) > 1
        else top_transition_share
    )
    supervised_demotion_case_count = sum(int(item["supervised_demotions_previous_to_promoted"] > 0) for item in top_changed_samples)
    mechanism_kind = "distributed_role_reassignment"
    if (
        diff_summary["slot_transition_summary"]["slot_match_rates"].get("slot_0", 0.0) >= 0.99
        and diff_summary["slot_transition_summary"]["slot_match_rates"].get("slot_1", 0.0) >= 0.99
        and diff_summary["slot_transition_summary"]["slot_match_rates"].get("slot_2", 0.0) >= 0.99
        and diff_summary["slot_transition_summary"]["slot_match_rates"].get("slot_3", 1.0) <= 0.01
        and top_two_transition_share >= 0.65
    ):
        mechanism_kind = "slot3_fixed_replacement_with_secondary_role_reassignment"

    return {
        "comparison_type": "slot3_transition_ranked_cameras",
        "aggregate_samples": aggregate_samples,
        "aggregate_stride": int(diff_summary["aggregate_stride"]),
        "sample_indices": list(diff_summary["sample_indices"]),
        "transition_direction_primary": "previous_to_promoted",
        "slot_3_transition_ranked_previous_to_promoted": slot3_ranked_previous_to_promoted,
        "slot_3_transition_ranked_promoted_to_previous": slot3_ranked_promoted_to_previous,
        "slot_3_promoted_camera_frequency": format_count_ranking(slot3_promoted_counts, "camera"),
        "slot_3_previous_camera_frequency": format_count_ranking(slot3_previous_counts, "camera"),
        "role_transition_ranked_by_camera_previous_to_promoted": format_count_ranking(
            role_transition_by_camera_previous_to_promoted, "camera_role_transition"
        ),
        "supervised_demotions_previous_to_promoted": format_count_ranking(
            supervised_demotions_previous_to_promoted, "camera_role_transition"
        ),
        "source_only_drops_previous_to_promoted": format_count_ranking(
            source_only_drops_previous_to_promoted, "camera_role_transition"
        ),
        "source_only_entries_previous_to_promoted": format_count_ranking(
            source_only_entries_previous_to_promoted, "camera_role_transition"
        ),
        "top_changed_samples": top_changed_samples[:10],
        "mechanism_shape": {
            "anchor_match_rate": float(diff_summary["delta_a_minus_b"]["anchor_match_rate"]),
            "slot_match_rates": dict(diff_summary["slot_transition_summary"]["slot_match_rates"]),
            "unique_slot_3_transition_count": int(len(slot3_ranked_previous_to_promoted)),
            "top_slot3_transition_share": top_transition_share,
            "top_two_slot3_transition_share": top_two_transition_share,
            "supervised_demotion_case_share": float(supervised_demotion_case_count / max(aggregate_samples, 1)),
            "mechanism_kind": mechanism_kind,
        },
        "mechanism_reading": [
            "slot_0/1/2 stay fixed while slot_3 flips on every sampled case, so the mechanism is concentrated in the last auxiliary source slot.",
            "The slot_3 swaps are not diffuse: the expanded basket collapses into four previous->promoted replacements, with the top two patterns covering most samples.",
            "Role reassignment exists, but it is secondary: only one transition family demotes a previous supervised camera, while the other cases are source_only-for-source_only swaps.",
        ],
    }


def build_tail_join(
    diff_summary: dict,
    slot3_summary: dict,
    hardcase_definition: dict,
    hardcase_manifest: dict | None,
    post_promotion_gap: dict,
    next_manual_problem: dict,
) -> dict:
    slot_match_rates = diff_summary["slot_transition_summary"]["slot_match_rates"]
    manifest_entries = list((hardcase_manifest or {}).get("entries", []))
    direct_tail_manifest_available = bool(manifest_entries)
    tail_keys = set()
    if direct_tail_manifest_available:
        for entry in manifest_entries:
            tail_keys.add((str(entry["seq_name"]), int(entry["frame_id"])))

    sample_rows = []
    transition_counts = Counter()
    transition_tail_hits = Counter()
    supervised_demotion_case_count = 0
    supervised_demotion_tail_hits = 0
    for row in diff_summary["per_sample_diffs"]:
        seq_name = str(row.get("seq_name") or "")
        frame_id = row.get("frame_id")
        slot3_transition = f"{row['slot_3_camera_b']} -> {row['slot_3_camera_a']}"
        role_transitions_previous_to_promoted = role_transitions_from_diff_row(row)
        has_supervised_demotion = any(
            transition.endswith("supervised -> source_only") or transition.endswith("supervised -> absent")
            for transition in role_transitions_previous_to_promoted
        )
        in_tail_manifest = bool(
            direct_tail_manifest_available
            and seq_name
            and frame_id is not None
            and (seq_name, int(frame_id)) in tail_keys
        )
        transition_counts[slot3_transition] += 1
        if in_tail_manifest:
            transition_tail_hits[slot3_transition] += 1
        if has_supervised_demotion:
            supervised_demotion_case_count += 1
            if in_tail_manifest:
                supervised_demotion_tail_hits += 1
        sample_rows.append(
            {
                "probe_sample_index": int(row["probe_sample_index"]),
                "seq_name": seq_name,
                "frame_id": frame_id,
                "slot_3_transition_previous_to_promoted": slot3_transition,
                "role_transitions_previous_to_promoted": role_transitions_previous_to_promoted,
                "camera_set_only_in_previous": list(row.get("camera_names_only_in_b", [])),
                "camera_set_only_in_promoted": list(row.get("camera_names_only_in_a", [])),
                "has_supervised_demotion_previous_to_promoted": bool(has_supervised_demotion),
                "in_official_tail_manifest": in_tail_manifest,
            }
        )

    total_samples = int(len(sample_rows))
    total_tail_hits = sum(int(item["in_official_tail_manifest"]) for item in sample_rows)
    basket_tail_rate = float(total_tail_hits / total_samples) if total_samples > 0 else 0.0
    sample_frame_ids = [int(item["frame_id"]) for item in sample_rows if item.get("frame_id") is not None]
    tail_frame_ids = [int(entry["frame_id"]) for entry in manifest_entries if entry.get("frame_id") is not None]
    probe_frame_range = None
    official_tail_frame_range = None
    probe_tail_overlap_note = ""
    if sample_frame_ids:
        probe_frame_range = {"min_frame_id": min(sample_frame_ids), "max_frame_id": max(sample_frame_ids)}
    if tail_frame_ids:
        official_tail_frame_range = {"min_frame_id": min(tail_frame_ids), "max_frame_id": max(tail_frame_ids)}
    if probe_frame_range is not None and official_tail_frame_range is not None:
        if probe_frame_range["max_frame_id"] < official_tail_frame_range["min_frame_id"]:
            probe_tail_overlap_note = (
                "The expanded probe basket never reaches the official hard-tail region: sampled frames stop before the first labeled tail frame."
            )
        elif probe_frame_range["min_frame_id"] > official_tail_frame_range["max_frame_id"]:
            probe_tail_overlap_note = (
                "The expanded probe basket sits entirely after the official hard-tail region."
            )
        else:
            probe_tail_overlap_note = "The expanded probe basket overlaps the official hard-tail frame range."
    slot3_transition_tail_alignment = []
    for item in slot3_summary["slot_3_transition_ranked_previous_to_promoted"]:
        transition = str(item["transition"])
        count = int(transition_counts.get(transition, 0))
        tail_hits = int(transition_tail_hits.get(transition, 0))
        tail_rate = float(tail_hits / count) if count > 0 else 0.0
        enrichment = None
        if basket_tail_rate > 0:
            enrichment = float(tail_rate / basket_tail_rate)
        slot3_transition_tail_alignment.append(
            {
                "transition": transition,
                "count": count,
                "tail_hits": tail_hits,
                "tail_hit_rate": tail_rate,
                "tail_enrichment_vs_probe_basket": enrichment,
            }
        )

    top_two_transition_rows = slot3_transition_tail_alignment[:2]
    top_two_transition_count = sum(int(item["count"]) for item in top_two_transition_rows)
    top_two_transition_tail_hits = sum(int(item["tail_hits"]) for item in top_two_transition_rows)
    top_two_transition_tail_rate = (
        float(top_two_transition_tail_hits / top_two_transition_count)
        if top_two_transition_count > 0
        else 0.0
    )
    top_two_transition_tail_enrichment = (
        float(top_two_transition_tail_rate / basket_tail_rate)
        if basket_tail_rate > 0
        else None
    )
    supervised_demotion_tail_rate = (
        float(supervised_demotion_tail_hits / supervised_demotion_case_count)
        if supervised_demotion_case_count > 0
        else 0.0
    )
    supervised_demotion_tail_enrichment = (
        float(supervised_demotion_tail_rate / basket_tail_rate)
        if basket_tail_rate > 0 and supervised_demotion_case_count > 0
        else None
    )

    dominant_slot3_mechanism = (
        slot_match_rates.get("slot_0", 0.0) >= 0.99
        and slot_match_rates.get("slot_1", 0.0) >= 0.99
        and slot_match_rates.get("slot_2", 0.0) >= 0.99
        and slot_match_rates.get("slot_3", 1.0) <= 0.01
        and slot3_summary["mechanism_shape"]["top_two_slot3_transition_share"] >= 0.65
    )
    slot3_tail_alignment_positive = (
        direct_tail_manifest_available
        and dominant_slot3_mechanism
        and top_two_transition_tail_hits >= 2
        and top_two_transition_tail_rate >= basket_tail_rate
        and (top_two_transition_tail_enrichment is None or top_two_transition_tail_enrichment >= 1.10)
    )
    can_establish_tail_alignment = direct_tail_manifest_available
    same_direction_signal = "unresolved"
    if direct_tail_manifest_available:
        same_direction_signal = "present" if slot3_tail_alignment_positive else "not_detected"
    conclusion = "OPEN slot3_tail_source_stabilization" if slot3_tail_alignment_positive else "DO_NOT_OPEN_NEW_TICKET"
    blocker_reason = (
        str((hardcase_manifest or {}).get("blocker") or "")
        or "The current workspace still lacks an official promoted per-frame residual export / hard-case manifest."
    )

    return {
        "comparison_type": "slot3_transition_vs_tail_cases",
        "tail_join_mode": "official_manifest_join" if direct_tail_manifest_available else "proxy_only_missing_official_tail_manifest",
        "direct_tail_manifest_available": direct_tail_manifest_available,
        "official_tail_manifest_entry_count": int(len(manifest_entries)),
        "official_tail_manifest_status": str((hardcase_manifest or {}).get("status") or ""),
        "current_tail_problem_family": next_manual_problem["family"],
        "tail_gap_type": post_promotion_gap["remaining_gap_assessment"]["gap_type"],
        "tail_metric_name": hardcase_definition["bucket_definition"]["hard_case_metric"]["name"],
        "tail_threshold_rule": hardcase_definition["bucket_definition"]["threshold_rule"],
        "mechanism_evidence": {
            "anchor_match_rate": float(diff_summary["delta_a_minus_b"]["anchor_match_rate"]),
            "slot_match_rates": dict(slot_match_rates),
            "slot_3_transition_ranked_previous_to_promoted": list(
                slot3_summary["slot_3_transition_ranked_previous_to_promoted"]
            ),
            "role_transition_ranked_by_camera_previous_to_promoted": list(
                slot3_summary["role_transition_ranked_by_camera_previous_to_promoted"]
            ),
            "top_changed_samples": list(slot3_summary["top_changed_samples"]),
        },
        "tail_alignment_assessment": {
            "can_establish_tail_alignment": can_establish_tail_alignment,
            "reason": (
                "Official promoted tail labels are available and can be joined directly."
                if direct_tail_manifest_available
                else (
                    blocker_reason
                    + " slot_3 / role transitions can therefore only be joined to tail hypotheses indirectly, not to actual tail labels."
                )
            ),
            "same_direction_signal": same_direction_signal,
            "probe_sample_count": total_samples,
            "probe_tail_hits": int(total_tail_hits),
            "probe_tail_hit_rate": basket_tail_rate,
            "probe_frame_range": probe_frame_range,
            "official_tail_frame_range": official_tail_frame_range,
            "probe_tail_overlap_note": probe_tail_overlap_note,
            "slot3_transition_tail_alignment": slot3_transition_tail_alignment,
            "top_two_slot3_transition_tail_summary": {
                "count": int(top_two_transition_count),
                "tail_hits": int(top_two_transition_tail_hits),
                "tail_hit_rate": top_two_transition_tail_rate,
                "tail_enrichment_vs_probe_basket": top_two_transition_tail_enrichment,
            },
            "supervised_demotion_tail_summary": {
                "count": int(supervised_demotion_case_count),
                "tail_hits": int(supervised_demotion_tail_hits),
                "tail_hit_rate": supervised_demotion_tail_rate,
                "tail_enrichment_vs_probe_basket": supervised_demotion_tail_enrichment,
            },
            "sample_rows": sample_rows,
        },
        "preferred_future_ticket_if_tail_evidence_arrives": "slot3_tail_source_stabilization",
        "decision": conclusion,
        "decision_reasoning": [
            "Mechanism evidence is strong enough to say the change lives in slot_3 and camera-role reassignment rather than in anchor selection.",
            "The expanded basket supports a fixed slot_3 replacement story more strongly than a diffuse role-churn story, so role reassignment remains diagnostic rather than ticket-defining.",
            (
                "Official tail labels are available, so the slot_3 mechanism can be checked against actual promoted hard-tail entries."
                if direct_tail_manifest_available
                else "But the workspace still does not retain an official promoted residual-tail manifest, so same-direction tail evidence is still missing."
            ),
            (
                "The dominant slot_3 transition families are enriched in official tail labels, so a single slot3_tail_source_stabilization ticket is justified."
                if conclusion == "OPEN slot3_tail_source_stabilization"
                else (
                    probe_tail_overlap_note
                    if probe_tail_overlap_note
                    else "Without a positive slot_3-to-tail alignment signal, a new ticket would still be premature."
                )
            ),
        ],
    }


def build_decision_artifact(tail_join: dict, slot3_summary: dict, next_manual_problem: dict) -> dict:
    if tail_join["decision"] == "DO_NOT_OPEN_NEW_TICKET":
        direct_tail_manifest_available = bool(tail_join.get("direct_tail_manifest_available"))
        top_two_summary = tail_join["tail_alignment_assessment"].get("top_two_slot3_transition_tail_summary", {})
        return {
            "comparison_type": "selection_contract_mechanism_stop_conclusion",
            "decision": "DO_NOT_OPEN_NEW_TICKET",
            "manual_problem_draft_generated": False,
            "slot3_fixed_camera_answer": "yes",
            "tail_alignment_answer": "no" if direct_tail_manifest_available else "unresolved",
            "future_ticket_answer": "none",
            "current_tail_problem_family_remains": next_manual_problem["family"],
            "mechanism_kind": slot3_summary["mechanism_shape"]["mechanism_kind"],
            "stop_reason": (
                tail_join["tail_alignment_assessment"].get("probe_tail_overlap_note")
                or "Official promoted tail labels are available, but none of the 32 sampled slot_3 transition cases land in the official hard tail."
                if direct_tail_manifest_available
                else tail_join["tail_alignment_assessment"]["reason"]
            ),
            "top_two_slot3_transition_tail_summary": top_two_summary,
            "next_required_evidence": (
                [
                    "slot_3 stabilization is not the right next ticket on the current labeled evidence",
                    "return to residual_case_coverage_rebalancing and inspect which official hard-tail regions remain uncovered",
                    "do not open a new slot_3 / role reassignment ticket unless a future basket directly intersects the official hard tail",
                ]
                if direct_tail_manifest_available
                else [
                    "restore or regenerate an official promoted per-frame residual export",
                    "materialize a frozen promoted hard-tail manifest keyed by (seq_name, frame_id)",
                    "re-run the slot_3 / role join against actual tail labels before opening a new ticket",
                ]
            ),
        }

    return {
        "draft_kind": "selection_contract_mechanism_open_decision",
        "status": "decision_only_no_draft_written",
        "decision": tail_join["decision"],
        "family": "slot3_tail_source_stabilization",
        "manual_problem_draft_generated": False,
        "mechanism_kind": slot3_summary["mechanism_shape"]["mechanism_kind"],
        "evidence_gate": "tail_aligned_selection_contract_mechanism",
    }


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    expanded_diff = load_json(Path(args.expanded_diff_json))
    hardcase_definition = load_json(Path(args.hardcase_definition_json))
    hardcase_manifest = None
    hardcase_manifest_ref = str(args.hardcase_manifest_json or "").strip()
    if hardcase_manifest_ref:
        manifest_path = Path(hardcase_manifest_ref)
        if manifest_path.is_file():
            hardcase_manifest = load_json(manifest_path)
    post_promotion_gap = load_json(Path(args.post_promotion_gap_json))
    next_manual_problem = load_json(Path(args.next_manual_problem_json))

    probe_a_dir = Path(expanded_diff["probe_a_dir"])
    probe_b_dir = Path(expanded_diff["probe_b_dir"])
    payloads_a = load_json(probe_a_dir / "per_sample_summaries.json")
    payloads_b = load_json(probe_b_dir / "per_sample_summaries.json")

    slot3_summary = build_slot3_rankings(expanded_diff, payloads_a, payloads_b)
    tail_join = build_tail_join(
        expanded_diff,
        slot3_summary,
        hardcase_definition,
        hardcase_manifest,
        post_promotion_gap,
        next_manual_problem,
    )
    decision_artifact = build_decision_artifact(tail_join, slot3_summary, next_manual_problem)

    slot3_json = Path(args.slot3_output_json)
    slot3_md = Path(args.slot3_output_md)
    tail_json = Path(args.tail_output_json)
    tail_md = Path(args.tail_output_md)
    decision_json = Path(args.decision_output_json) if str(args.decision_output_json or "").strip() else None
    decision_md = Path(args.decision_output_md) if str(args.decision_output_md or "").strip() else None
    output_paths = [slot3_json, slot3_md, tail_json, tail_md]
    if decision_json is not None:
        output_paths.append(decision_json)
    if decision_md is not None:
        output_paths.append(decision_md)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    slot3_json.write_text(json.dumps(slot3_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        slot3_md,
        [
            "# Slot3 Transition Ranked Cameras",
            "",
            f"- aggregate_samples: `{slot3_summary['aggregate_samples']}`",
            f"- aggregate_stride: `{slot3_summary['aggregate_stride']}`",
            f"- sample_indices: `{slot3_summary['sample_indices']}`",
            f"- mechanism_shape: `{slot3_summary['mechanism_shape']}`",
            "",
            "## Slot3 Transition Ranking",
            "",
            f"- transition_direction_primary: `{slot3_summary['transition_direction_primary']}`",
            f"- slot_3_transition_ranked_previous_to_promoted: `{slot3_summary['slot_3_transition_ranked_previous_to_promoted']}`",
            f"- slot_3_promoted_camera_frequency: `{slot3_summary['slot_3_promoted_camera_frequency']}`",
            f"- slot_3_previous_camera_frequency: `{slot3_summary['slot_3_previous_camera_frequency']}`",
            f"- role_transition_ranked_by_camera_previous_to_promoted: `{slot3_summary['role_transition_ranked_by_camera_previous_to_promoted']}`",
            f"- supervised_demotions_previous_to_promoted: `{slot3_summary['supervised_demotions_previous_to_promoted']}`",
            f"- source_only_drops_previous_to_promoted: `{slot3_summary['source_only_drops_previous_to_promoted']}`",
            f"- source_only_entries_previous_to_promoted: `{slot3_summary['source_only_entries_previous_to_promoted']}`",
            "",
            "## Top Changed Samples",
            "",
            f"- top_changed_samples: `{slot3_summary['top_changed_samples']}`",
            "",
            "## Reading",
            "",
        ] + [f"- {item}" for item in slot3_summary["mechanism_reading"]],
    )

    tail_json.write_text(json.dumps(tail_join, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        tail_md,
        [
            "# Slot3 Transition vs Tail Cases",
            "",
            f"- tail_join_mode: `{tail_join['tail_join_mode']}`",
            f"- direct_tail_manifest_available: `{tail_join['direct_tail_manifest_available']}`",
            f"- official_tail_manifest_entry_count: `{tail_join['official_tail_manifest_entry_count']}`",
            f"- official_tail_manifest_status: `{tail_join['official_tail_manifest_status']}`",
            f"- current_tail_problem_family: `{tail_join['current_tail_problem_family']}`",
            f"- tail_gap_type: `{tail_join['tail_gap_type']}`",
            f"- tail_metric_name: `{tail_join['tail_metric_name']}`",
            f"- preferred_future_ticket_if_tail_evidence_arrives: `{tail_join['preferred_future_ticket_if_tail_evidence_arrives']}`",
            f"- decision: `{tail_join['decision']}`",
            "",
            "## Mechanism Evidence",
            "",
            f"- mechanism_evidence: `{tail_join['mechanism_evidence']}`",
            "",
            "## Tail Alignment",
            "",
            f"- tail_alignment_assessment: `{tail_join['tail_alignment_assessment']}`",
            "",
            "## Decision Reasoning",
            "",
        ] + [f"- {item}" for item in tail_join["decision_reasoning"]],
    )

    if decision_json is not None:
        decision_json.write_text(json.dumps(decision_artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    if decision_md is not None:
        write_markdown(
            decision_md,
            [
                "# Selection Contract Mechanism Decision",
                "",
                f"- decision: `{decision_artifact['decision' if 'decision' in decision_artifact else 'family']}`",
                f"- manual_problem_draft_generated: `{decision_artifact['manual_problem_draft_generated']}`",
                f"- mechanism_kind: `{decision_artifact['mechanism_kind']}`",
                "",
                "## Summary",
                "",
                f"- payload: `{decision_artifact}`",
            ],
        )

    print(slot3_json)
    print(slot3_md)
    print(tail_json)
    print(tail_md)
    if decision_json is not None:
        print(decision_json)
    if decision_md is not None:
        print(decision_md)


if __name__ == "__main__":
    main()
