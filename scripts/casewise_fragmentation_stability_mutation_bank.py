from __future__ import annotations

from itertools import combinations


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
        "coverage_floor_ratio": 0.82,
        "coverage_floor_mix": 0.30,
        "visible_floor_ratio": 0.98,
        "visible_floor_mix": 0.25,
        "visible_mass_floor_ratio": 0.98,
        "visible_mass_floor_mix": 0.30,
        "casewise_source_drop_topk": 1,
        "casewise_source_keep_min": 3,
        "source_poison_offbody_weight": 1.00,
        "source_poison_bottom_weight": 0.90,
        "source_poison_fragment_weight": 0.75,
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


def _unique_subsets(subsets: list[list[int]]) -> list[list[int]]:
    seen = set()
    out: list[list[int]] = []
    for subset in subsets:
        key = tuple(int(idx) for idx in subset)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def stage_a_casewise_source_mutations(source_count: int, *, prior_subsets: list[list[int]]) -> list[dict]:
    base = _base_proxy_config()
    full_subset = list(range(source_count))
    seed_subsets = _unique_subsets(
        [
            full_subset,
            *prior_subsets,
            [idx for idx in full_subset if idx != 3] if source_count > 3 else full_subset,
            [idx for idx in full_subset if idx not in {2, 3}] if source_count > 3 else full_subset,
        ]
    )

    poison_presets = [
        ("drop1_balanced", 1, 1.00, 0.90, 0.75, 0.30, 0.75),
        ("drop1_fragment", 1, 0.80, 0.70, 1.05, 0.40, 0.60),
        ("drop2_balanced", 2, 1.00, 0.90, 0.80, 0.30, 0.85),
        ("drop2_fragment", 2, 0.75, 0.65, 1.10, 0.45, 0.65),
    ]

    rows: list[dict] = []
    subset_budget = max(1, min(3, len(seed_subsets)))
    for subset_rank, subset in enumerate(seed_subsets[:subset_budget], start=1):
        for preset_name, drop_topk, offbody_w, bottom_w, fragment_w, peak_w, fg_reward_w in poison_presets:
            rows.append(
                _mutation(
                    f"subset{subset_rank:02d}_{preset_name}",
                    "A",
                    "casewise_source_prune_search",
                    (
                        f"Casewise source pruning on subset {subset} with drop_topk={drop_topk}, "
                        f"offbody={offbody_w:.2f}, bottom={bottom_w:.2f}, "
                        f"fragment={fragment_w:.2f}, peak={peak_w:.2f}, fg_reward={fg_reward_w:.2f}."
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
            if len(rows) >= 10:
                return rows[:10]

    if len(rows) < 10 and source_count >= 4:
        for a, b in combinations(full_subset, 2):
            subset = [idx for idx in full_subset if idx not in {a, b}]
            mutation_id = f"fallback_drop_{a}_{b}"
            if any(row["mutation_id"] == mutation_id for row in rows):
                continue
            rows.append(
                _mutation(
                    mutation_id,
                    "A",
                    "casewise_source_prune_search",
                    f"Fallback casewise source pruning seed dropping sources {a} and {b}.",
                    {
                        **base,
                        "source_subset": subset,
                        "casewise_source_drop_topk": 1,
                    },
                    source_subset=subset,
                )
            )
            if len(rows) >= 10:
                break
    return rows[:10]


def stage_b_casewise_rehydrated_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
    configs = [
        ("soft_blend015", 0.15, 0.15, 0.95, 0.20, 0.98, 0.25, 0.98, 0.30, 0.85, 0.8, 3),
        ("mid_blend030", 0.25, 0.30, 0.97, 0.25, 0.99, 0.25, 0.99, 0.30, 1.00, 0.8, 3),
        ("mid_blend045", 0.25, 0.45, 0.98, 0.30, 1.00, 0.30, 1.00, 0.35, 1.00, 1.2, 5),
        ("hi_floor_softpow", 0.35, 0.30, 0.99, 0.35, 1.00, 0.35, 1.00, 0.35, 0.90, 1.2, 5),
        ("mid_sharppow", 0.25, 0.30, 0.98, 0.30, 0.99, 0.30, 0.99, 0.30, 1.20, 1.6, 5),
        ("soft_rescue", 0.35, 0.45, 1.00, 0.40, 1.00, 0.40, 1.00, 0.40, 0.85, 1.2, 7),
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
    ) in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "casewise_rehydrated_tuning",
                (
                    "Tune casewise rehydrated render with "
                    f"alpha_floor={alpha_floor:.2f}, baseline_blend={baseline_blend:.2f}, "
                    f"coverage_floor_ratio={coverage_floor_ratio:.2f}, visible_floor_ratio={visible_floor_ratio:.2f}, "
                    f"visible_mass_floor_ratio={visible_mass_floor_ratio:.2f}, support_gate_pow={support_gate_pow:.2f}, "
                    f"gaussian_sigma={gaussian_sigma:.1f}, morph_close_k={morph_close_k}."
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
                },
                source_subset=source_subset,
            )
        )
    return rows


def stage_c_casewise_label_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
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
                "casewise_label_consistency",
                (
                    "Escalate label consistency only after casewise pruning is honest: "
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
