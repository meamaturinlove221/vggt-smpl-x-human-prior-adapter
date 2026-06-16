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
    return sorted(dict.fromkeys(int(v) for v in source_subset if 0 <= int(v) < source_count))


def _anchor_rule(anchor_seed: dict, source_subset: list[int], override: dict | None = None) -> dict:
    rule = {
        "rule_id": str(anchor_seed.get("rule_id", "anchor_specific_rule")),
        "seq_name": anchor_seed.get("seq_name"),
        "target_camera": anchor_seed.get("target_camera"),
        "frame_min": anchor_seed.get("frame_min"),
        "frame_max": anchor_seed.get("frame_max"),
        "case_ids": list(anchor_seed.get("case_ids", [])),
        "proxy_config": {"source_subset": list(source_subset)},
    }
    if override:
        rule["proxy_config"].update(dict(override))
    return rule


def stage_a_anchor_seed_mutations(
    *,
    source_count: int,
    base_config: dict,
    base_subset: list[int],
    anchor_seed: dict,
) -> list[dict]:
    base = {**_base_proxy_config(), **base_config}
    base["partition_rules"] = []
    subset = _normalize_subset(base_subset, source_count) or [0, 1, 4, 5]
    full_subset = list(range(source_count))
    swap_in_pool = [idx for idx in full_subset if idx not in subset]
    rows: list[dict] = []

    def add_row(mutation_id: str, anchor_subset: list[int], desc: str) -> None:
        anchor_subset_n = _normalize_subset(anchor_subset, source_count)
        if len(anchor_subset_n) < 3:
            return
        if any(existing["proxy_config"]["partition_rules"][0]["proxy_config"]["source_subset"] == anchor_subset_n for existing in rows):
            return
        proxy_config = {**base, "partition_rules": [_anchor_rule(anchor_seed, anchor_subset_n)]}
        rows.append(
            _mutation(
                mutation_id,
                "A",
                "anchor_specific_subset_search",
                desc,
                proxy_config,
                source_subset=anchor_subset_n,
            )
        )

    add_row(
        "anchor_hold",
        subset,
        f"Keep the prior honest global subset {subset} as the anchor-specific starting point.",
    )
    for incoming in swap_in_pool:
        for outgoing in subset:
            swapped = [idx for idx in subset if idx != outgoing] + [incoming]
            add_row(
                f"anchor_swap_out{outgoing}_in{incoming}",
                swapped,
                f"Only on the rebound anchor, swap out source {outgoing} for source {incoming}.",
            )
            if len(rows) >= 10:
                return rows
    for incoming in swap_in_pool:
        add_row(
            f"anchor_add_{incoming}",
            subset + [incoming],
            f"Only on the rebound anchor, keep the prior subset and add source {incoming}.",
        )
        if len(rows) >= 10:
            return rows
    return rows[:10]


def stage_b_anchor_correspondence_tuning(
    anchor_seed: dict,
    *,
    prefix: str,
    seed_config: dict,
    source_subset: list[int],
) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config}
    base["partition_rules"] = []
    subset = list(source_subset)
    configs = [
        ("corr_soft", 0.04, 3, 0.25, 0.95, 0.8, 3),
        ("corr_mid", 0.05, 5, 0.45, 1.00, 1.2, 5),
        ("corr_strong", 0.06, 5, 0.55, 1.05, 1.2, 5),
        ("corr_sharp", 0.07, 5, 0.45, 1.15, 0.8, 3),
        ("corr_blur", 0.05, 5, 0.35, 0.95, 1.6, 7),
        ("corr_close", 0.05, 7, 0.55, 1.00, 1.2, 7),
        ("corr_margin08", 0.08, 5, 0.45, 1.05, 1.2, 5),
        ("corr_margin03", 0.03, 3, 0.25, 0.90, 0.8, 3),
        ("corr_mix065", 0.05, 7, 0.65, 1.00, 1.2, 5),
        ("corr_pow085", 0.05, 5, 0.45, 0.85, 1.2, 5),
        ("corr_pow125", 0.05, 5, 0.45, 1.25, 0.8, 3),
        ("corr_close3", 0.05, 5, 0.45, 1.00, 0.8, 3),
    ]
    rows: list[dict] = []
    for name, margin, majority_k, smooth_mix, support_pow, sigma, close_k in configs:
        override = {
            "consensus_margin_floor": margin,
            "label_majority_k": majority_k,
            "label_smooth_mix": smooth_mix,
            "support_gate_pow": support_pow,
            "gaussian_sigma": sigma,
            "morph_close_k": close_k,
        }
        proxy_config = {**base, "partition_rules": [_anchor_rule(anchor_seed, subset, override)]}
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "anchor_specific_correspondence_tuning",
                (
                    "Tune only the rebound anchor correspondence behavior: "
                    f"margin={margin:.2f}, label_majority_k={majority_k}, label_smooth_mix={smooth_mix:.2f}, "
                    f"support_gate_pow={support_pow:.2f}, gaussian_sigma={sigma:.1f}, morph_close_k={close_k}."
                ),
                proxy_config,
                source_subset=subset,
            )
        )
    return rows


def stage_c_anchor_label_smoothing(
    anchor_seed: dict,
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
        ("label_close7", 7, 0.55, 0.05, 1.2, 7),
        ("label_smooth_blur", 5, 0.65, 0.05, 1.6, 7),
        ("label_crisp", 3, 0.25, 0.06, 0.8, 3),
    ]
    rows: list[dict] = []
    for name, majority_k, smooth_mix, margin, sigma, close_k in configs:
        override = {
            "label_majority_k": majority_k,
            "label_smooth_mix": smooth_mix,
            "consensus_margin_floor": margin,
            "gaussian_sigma": sigma,
            "morph_close_k": close_k,
        }
        proxy_config = {**base, "partition_rules": [_anchor_rule(anchor_seed, subset, override)]}
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "C",
                "anchor_specific_label_smoothing",
                (
                    "Apply conservative anchor-specific label smoothing: "
                    f"label_majority_k={majority_k}, label_smooth_mix={smooth_mix:.2f}, "
                    f"margin={margin:.2f}, gaussian_sigma={sigma:.1f}, morph_close_k={close_k}."
                ),
                proxy_config,
                source_subset=subset,
            )
        )
    return rows
