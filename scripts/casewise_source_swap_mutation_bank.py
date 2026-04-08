from __future__ import annotations


def _base_proxy_config() -> dict:
    return {
        "render_mode": "rehydrated",
        "source_subset": [],
        "alpha_floor": 0.25,
        "support_gate_pow": 1.00,
        "baseline_blend": 0.30,
        "gaussian_sigma": 0.8,
        "morph_close_k": 3,
        "label_majority_k": 3,
        "label_smooth_mix": 0.25,
        "consensus_margin_floor": 0.05,
        "coverage_floor_ratio": 0.97,
        "coverage_floor_mix": 0.25,
        "visible_floor_ratio": 0.99,
        "visible_floor_mix": 0.25,
        "visible_mass_floor_ratio": 0.99,
        "visible_mass_floor_mix": 0.30,
        "casewise_source_drop_topk": 0,
        "casewise_source_keep_min": 3,
        "source_poison_offbody_weight": 1.00,
        "source_poison_bottom_weight": 0.90,
        "source_poison_fragment_weight": 0.80,
        "source_poison_peak_weight": 0.30,
        "source_fg_reward_weight": 0.75,
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


def stage_a_source_swap_mutations(*, source_count: int, prior_best_subset: list[int]) -> list[dict]:
    base = _base_proxy_config()
    full_subset = list(range(source_count))
    prior = list(prior_best_subset) if prior_best_subset else [0, 1, 4, 5]
    if not prior:
        prior = full_subset[:4]
    swap_in_pool = [idx for idx in full_subset if idx not in prior]
    rows: list[dict] = []

    def add_row(mutation_id: str, subset: list[int], desc: str) -> None:
        nonlocal rows
        subset = sorted(dict.fromkeys(int(v) for v in subset if 0 <= int(v) < source_count))
        if len(subset) < 3:
            return
        if any(existing["source_subset"] == subset for existing in rows):
            return
        rows.append(
            _mutation(
                mutation_id,
                "A",
                "casewise_source_swap_search",
                desc,
                {**base, "source_subset": subset},
                source_subset=subset,
            )
        )

    add_row(
        "prior_hold",
        prior,
        f"Hold the prior best subset {prior} as the source-swap baseline with casewise drop disabled.",
    )

    for incoming in swap_in_pool:
        for outgoing in prior:
            swapped = [idx for idx in prior if idx != outgoing] + [incoming]
            add_row(
                f"swap_out{outgoing}_in{incoming}",
                swapped,
                f"Swap out source {outgoing} for source {incoming} from prior subset {prior}.",
            )
            if len(rows) >= 10:
                return rows

    for incoming in swap_in_pool:
        add_row(
            f"add_{incoming}",
            prior + [incoming],
            f"Keep the prior subset and add source {incoming} back for anchor robustness.",
        )
        if len(rows) >= 10:
            return rows

    return rows[:10]


def stage_b_rehydrated_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset), "casewise_source_drop_topk": 0}
    configs = [
        ("soft_blend030", 0.20, 0.30, 0.97, 0.25, 0.99, 0.25, 0.99, 0.30, 1.00, 0.8, 3, 0.05),
        ("mid_blend045", 0.25, 0.45, 0.97, 0.25, 0.99, 0.25, 0.99, 0.30, 1.00, 0.8, 3, 0.05),
        ("sharper_margin", 0.25, 0.30, 0.98, 0.30, 0.99, 0.30, 0.99, 0.30, 1.15, 1.2, 5, 0.06),
        ("smooth_margin", 0.35, 0.30, 0.99, 0.35, 1.00, 0.35, 1.00, 0.35, 0.90, 1.2, 5, 0.03),
        ("soft_rescue", 0.35, 0.45, 1.00, 0.40, 1.00, 0.40, 1.00, 0.40, 0.85, 1.2, 7, 0.03),
        ("crisp_rescue", 0.20, 0.45, 0.98, 0.30, 0.99, 0.30, 0.99, 0.30, 1.10, 0.8, 3, 0.05),
    ]
    rows: list[dict] = []
    for (
        name,
        alpha_floor,
        baseline_blend,
        coverage_floor_ratio,
        coverage_floor_mix,
        visible_floor_ratio,
        visible_floor_mix,
        visible_mass_floor_ratio,
        visible_mass_floor_mix,
        support_gate_pow,
        gaussian_sigma,
        morph_close_k,
        consensus_margin_floor,
    ) in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "casewise_source_swap_rehydrated_tuning",
                (
                    "Tune source-swap rehydrated render with "
                    f"alpha_floor={alpha_floor:.2f}, baseline_blend={baseline_blend:.2f}, "
                    f"support_gate_pow={support_gate_pow:.2f}, consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                {
                    **base,
                    "alpha_floor": alpha_floor,
                    "baseline_blend": baseline_blend,
                    "coverage_floor_ratio": coverage_floor_ratio,
                    "coverage_floor_mix": coverage_floor_mix,
                    "visible_floor_ratio": visible_floor_ratio,
                    "visible_floor_mix": visible_floor_mix,
                    "visible_mass_floor_ratio": visible_mass_floor_ratio,
                    "visible_mass_floor_mix": visible_mass_floor_mix,
                    "support_gate_pow": support_gate_pow,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_close_k": morph_close_k,
                    "consensus_margin_floor": consensus_margin_floor,
                },
                source_subset=source_subset,
            )
        )
    return rows


def stage_c_label_swap_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset), "casewise_source_drop_topk": 0}
    configs = [
        ("label_soft", 3, 0.25, 0.04, 0.8, 3),
        ("label_mid", 5, 0.45, 0.05, 1.2, 5),
        ("label_strong", 7, 0.65, 0.06, 1.6, 5),
    ]
    rows: list[dict] = []
    for name, label_majority_k, label_smooth_mix, consensus_margin_floor, gaussian_sigma, morph_close_k in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "C",
                "casewise_source_swap_label_consistency",
                (
                    "Escalate label consistency only after source-swap smoke honesty: "
                    f"label_majority_k={label_majority_k}, label_smooth_mix={label_smooth_mix:.2f}, "
                    f"consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                {
                    **base,
                    "label_majority_k": label_majority_k,
                    "label_smooth_mix": label_smooth_mix,
                    "consensus_margin_floor": consensus_margin_floor,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_close_k": morph_close_k,
                },
                source_subset=source_subset,
            )
        )
    return rows
