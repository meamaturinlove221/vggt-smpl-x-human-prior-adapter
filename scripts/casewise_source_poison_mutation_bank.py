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
        "casewise_source_drop_topk": 1,
        "casewise_source_keep_min": 3,
        "source_poison_offbody_weight": 1.00,
        "source_poison_bottom_weight": 0.90,
        "source_poison_fragment_weight": 0.90,
        "source_poison_peak_weight": 0.40,
        "source_fg_reward_weight": 0.70,
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


def stage_a_source_poison_mutations(*, source_count: int, prior_best_subset: list[int]) -> list[dict]:
    base = _base_proxy_config()
    full_subset = list(range(source_count))
    seed_subsets = []
    for subset in [prior_best_subset, full_subset, [0, 1, 2, 4, 5], [0, 1, 3, 4, 5], [0, 1, 4, 5]]:
        if subset and subset not in seed_subsets:
            seed_subsets.append(list(subset))

    presets = [
        ("balanced_d1", 1, 1.00, 0.90, 0.90, 0.40, 0.70),
        ("fragment_d1", 1, 0.80, 0.70, 1.25, 0.60, 0.55),
        ("peak_d1", 1, 0.75, 0.65, 1.00, 0.95, 0.55),
        ("balanced_d2", 2, 1.00, 0.90, 1.00, 0.45, 0.80),
    ]

    rows: list[dict] = []
    for subset_idx, subset in enumerate(seed_subsets, start=1):
        for name, drop_topk, offbody_w, bottom_w, fragment_w, peak_w, fg_reward_w in presets:
            rows.append(
                _mutation(
                    f"seed{subset_idx:02d}_{name}",
                    "A",
                    "casewise_source_poison_search",
                    (
                        f"Casewise poison-source search on subset {subset} with drop_topk={drop_topk}, "
                        f"offbody={offbody_w:.2f}, bottom={bottom_w:.2f}, fragment={fragment_w:.2f}, "
                        f"peak={peak_w:.2f}, fg_reward={fg_reward_w:.2f}."
                    ),
                    {
                        **base,
                        "source_subset": list(subset),
                        "casewise_source_drop_topk": drop_topk,
                        "source_poison_offbody_weight": offbody_w,
                        "source_poison_bottom_weight": bottom_w,
                        "source_poison_fragment_weight": fragment_w,
                        "source_poison_peak_weight": peak_w,
                        "source_fg_reward_weight": fg_reward_w,
                    },
                    source_subset=subset,
                )
            )
            if len(rows) >= 12:
                return rows
    return rows


def stage_b_rehydrated_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
    configs = [
        ("soft_blend015", 0.15, 0.15, 0.97, 0.25, 0.99, 0.25, 0.99, 0.30, 0.95, 0.8, 3, 0.05),
        ("mid_blend030", 0.25, 0.30, 0.97, 0.25, 0.99, 0.25, 0.99, 0.30, 1.00, 0.8, 3, 0.05),
        ("mid_blend045", 0.25, 0.45, 0.98, 0.30, 1.00, 0.30, 1.00, 0.35, 1.00, 1.2, 5, 0.04),
        ("smooth_margin", 0.35, 0.30, 0.99, 0.35, 1.00, 0.35, 1.00, 0.35, 0.90, 1.2, 5, 0.03),
        ("sharper_margin", 0.25, 0.30, 0.98, 0.30, 0.99, 0.30, 0.99, 0.30, 1.20, 1.6, 5, 0.06),
        ("soft_rescue", 0.35, 0.45, 1.00, 0.40, 1.00, 0.40, 1.00, 0.40, 0.85, 1.2, 7, 0.03),
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
                "casewise_poison_rehydrated_tuning",
                (
                    "Tune poison-aware rehydrated render with "
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


def stage_c_label_poison_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
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
                "casewise_poison_label_consistency",
                (
                    "Escalate label consistency only after poison pruning is honest: "
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
