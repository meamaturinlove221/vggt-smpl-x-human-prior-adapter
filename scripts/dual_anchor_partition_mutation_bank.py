from __future__ import annotations


def _base_proxy_config() -> dict:
    return {
        "render_mode": "rehydrated",
        "source_subset": [0, 1, 4, 5],
        "alpha_floor": 0.25,
        "support_gate_pow": 1.15,
        "baseline_blend": 0.30,
        "gaussian_sigma": 0.8,
        "morph_close_k": 3,
        "label_majority_k": 5,
        "label_smooth_mix": 0.45,
        "consensus_margin_floor": 0.06,
        "coverage_floor_ratio": 0.97,
        "coverage_floor_mix": 0.25,
        "visible_floor_ratio": 0.99,
        "visible_floor_mix": 0.25,
        "visible_mass_floor_ratio": 0.99,
        "visible_mass_floor_mix": 0.30,
        "casewise_source_drop_topk": 0,
        "casewise_source_keep_min": 3,
        "partition_rules": [],
    }


def _mutation(
    mutation_id: str,
    stage: str,
    family: str,
    description: str,
    proxy_config: dict,
    *,
    source_subset: list[int] | None = None,
) -> dict:
    return {
        "mutation_id": mutation_id,
        "stage": stage,
        "family": family,
        "description": description,
        "proxy_config": proxy_config,
        "source_subset": list(source_subset or proxy_config.get("source_subset", [])),
    }


def _normalize_subset(source_subset: list[int], source_count: int) -> list[int]:
    subset = sorted(dict.fromkeys(int(v) for v in source_subset if 0 <= int(v) < source_count))
    return subset


def _partition_rule(partition_seed: dict, source_subset: list[int], override: dict | None = None) -> dict:
    rule = {
        "rule_id": str(partition_seed.get("rule_id", "rebound_partition")),
        "target_camera": partition_seed.get("target_camera"),
        "frame_min": partition_seed.get("frame_min"),
        "frame_max": partition_seed.get("frame_max"),
        "case_ids": list(partition_seed.get("case_ids", [])),
        "proxy_config": {"source_subset": list(source_subset)},
    }
    if override:
        rule["proxy_config"].update(dict(override))
    return rule


def stage_a_partition_mutations(
    *,
    source_count: int,
    base_config: dict,
    base_subset: list[int],
    partition_seed: dict,
) -> list[dict]:
    base = {**_base_proxy_config(), **base_config}
    base["partition_rules"] = []
    subset = _normalize_subset(base_subset, source_count) or [0, 1, 4, 5]
    full_subset = list(range(source_count))
    swap_in_pool = [idx for idx in full_subset if idx not in subset]
    rows: list[dict] = []

    def add_row(mutation_id: str, part_subset: list[int], desc: str) -> None:
        part_subset_n = _normalize_subset(part_subset, source_count)
        if len(part_subset_n) < 3:
            return
        if any(existing["proxy_config"]["partition_rules"][0]["proxy_config"]["source_subset"] == part_subset_n for existing in rows):
            return
        proxy_config = {
            **base,
            "partition_rules": [_partition_rule(partition_seed, part_subset_n)],
        }
        rows.append(
            _mutation(
                mutation_id,
                "A",
                "dual_anchor_partition_subset_search",
                desc,
                proxy_config,
                source_subset=part_subset_n,
            )
        )

    add_row(
        "prior_partition_hold",
        subset,
        f"Keep the global best subset {subset} but only apply it to the rebound partition.",
    )
    for incoming in swap_in_pool:
        for outgoing in subset:
            swapped = [idx for idx in subset if idx != outgoing] + [incoming]
            add_row(
                f"partition_swap_out{outgoing}_in{incoming}",
                swapped,
                f"For the rebound partition, swap out source {outgoing} for source {incoming}.",
            )
            if len(rows) >= 10:
                return rows

    for incoming in swap_in_pool:
        add_row(
            f"partition_add_{incoming}",
            subset + [incoming],
            f"For the rebound partition, keep the global subset and add source {incoming}.",
        )
        if len(rows) >= 10:
            return rows

    return rows[:10]


def stage_b_partition_tuning(
    partition_seed: dict,
    *,
    prefix: str,
    seed_config: dict,
    source_subset: list[int],
) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config}
    base["partition_rules"] = []
    subset = list(source_subset)
    configs = [
        ("soft_blend030", 0.20, 0.30, 1.00, 0.90, 0.8, 3, 0.04),
        ("mid_blend045", 0.25, 0.45, 1.00, 1.00, 0.8, 3, 0.05),
        ("sharper_margin", 0.25, 0.30, 0.99, 1.10, 1.2, 5, 0.06),
        ("smooth_margin", 0.35, 0.30, 1.00, 0.90, 1.2, 5, 0.03),
        ("soft_rescue", 0.35, 0.45, 1.00, 0.85, 1.2, 7, 0.03),
        ("crisp_rescue", 0.20, 0.45, 0.99, 1.10, 0.8, 3, 0.05),
    ]
    rows: list[dict] = []
    for name, alpha_floor, baseline_blend, coverage_floor_ratio, support_gate_pow, gaussian_sigma, morph_close_k, consensus_margin_floor in configs:
        override = {
            "alpha_floor": alpha_floor,
            "baseline_blend": baseline_blend,
            "coverage_floor_ratio": coverage_floor_ratio,
            "support_gate_pow": support_gate_pow,
            "gaussian_sigma": gaussian_sigma,
            "morph_close_k": morph_close_k,
            "consensus_margin_floor": consensus_margin_floor,
        }
        proxy_config = {
            **base,
            "partition_rules": [_partition_rule(partition_seed, subset, override)],
        }
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "dual_anchor_partition_rehydrated_tuning",
                (
                    "Tune only the rebound partition with "
                    f"alpha_floor={alpha_floor:.2f}, baseline_blend={baseline_blend:.2f}, "
                    f"support_gate_pow={support_gate_pow:.2f}, consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                proxy_config,
                source_subset=subset,
            )
        )
    return rows


def stage_c_partition_label(
    partition_seed: dict,
    *,
    prefix: str,
    seed_config: dict,
    source_subset: list[int],
) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config}
    base["partition_rules"] = []
    subset = list(source_subset)
    configs = [
        ("label_soft", 3, 0.25, 0.04, 0.8, 3),
        ("label_mid", 5, 0.45, 0.05, 1.2, 5),
        ("label_strong", 7, 0.65, 0.06, 1.6, 5),
    ]
    rows: list[dict] = []
    for name, label_majority_k, label_smooth_mix, consensus_margin_floor, gaussian_sigma, morph_close_k in configs:
        override = {
            "label_majority_k": label_majority_k,
            "label_smooth_mix": label_smooth_mix,
            "consensus_margin_floor": consensus_margin_floor,
            "gaussian_sigma": gaussian_sigma,
            "morph_close_k": morph_close_k,
        }
        proxy_config = {
            **base,
            "partition_rules": [_partition_rule(partition_seed, subset, override)],
        }
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "C",
                "dual_anchor_partition_label_consistency",
                (
                    "Add label consistency only on the rebound partition: "
                    f"label_majority_k={label_majority_k}, label_smooth_mix={label_smooth_mix:.2f}, "
                    f"consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                proxy_config,
                source_subset=subset,
            )
        )
    return rows
