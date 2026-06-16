from __future__ import annotations

from itertools import combinations


def _base_proxy_config() -> dict:
    return {
        "render_mode": "rehydrated",
        "source_subset": [],
        "alpha_floor": 0.25,
        "support_gate_pow": 1.0,
        "baseline_blend": 0.30,
        "gaussian_sigma": 0.8,
        "morph_close_k": 3,
        "label_majority_k": 3,
        "label_smooth_mix": 0.25,
        "consensus_margin_floor": 0.05,
        "coverage_floor_ratio": 0.80,
        "coverage_floor_mix": 0.30,
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


def stage_a_source_subset_mutations(source_count: int) -> list[dict]:
    base = _base_proxy_config()
    full_subset = list(range(source_count))
    rows: list[dict] = []
    for drop_idx in full_subset:
        subset = [idx for idx in full_subset if idx != drop_idx]
        rows.append(
            _mutation(
                f"subset_leave_one_out_drop_{drop_idx}",
                "A",
                "source_subset_search",
                f"Leave-one-out source subset search dropping source index {drop_idx}.",
                {**base, "source_subset": subset},
                source_subset=subset,
            )
        )

    preferred_pairs = [(1, 2), (1, 3), (1, 5), (2, 3)]
    for a, b in preferred_pairs:
        if a >= source_count or b >= source_count:
            continue
        subset = [idx for idx in full_subset if idx not in {a, b}]
        rows.append(
            _mutation(
                f"subset_drop_{a}_{b}",
                "A",
                "source_subset_search",
                f"Two-source subset search dropping source indices {a} and {b}.",
                {**base, "source_subset": subset},
                source_subset=subset,
            )
        )

    if len(rows) < 10 and source_count >= 4:
        for a, b in combinations(full_subset, 2):
            mutation_id = f"subset_drop_{a}_{b}"
            if any(row["mutation_id"] == mutation_id for row in rows):
                continue
            subset = [idx for idx in full_subset if idx not in {a, b}]
            rows.append(
                _mutation(
                    mutation_id,
                    "A",
                    "source_subset_search",
                    f"Two-source subset search dropping source indices {a} and {b}.",
                    {**base, "source_subset": subset},
                    source_subset=subset,
                )
            )
            if len(rows) >= 10:
                break

    return rows[:10]


def stage_b_rehydrated_mutations(source_subset: list[int], *, prefix: str) -> list[dict]:
    base = {**_base_proxy_config(), "source_subset": list(source_subset)}
    configs = [
        ("rehydrated_softfloor_blend015", 0.15, 0.15, 0.78, 0.15, 0.75, 0.8, 3),
        ("rehydrated_softfloor_blend030", 0.15, 0.30, 0.78, 0.25, 1.00, 0.8, 3),
        ("rehydrated_midfloor_blend030", 0.25, 0.30, 0.80, 0.30, 1.00, 1.2, 3),
        ("rehydrated_midfloor_blend045", 0.25, 0.45, 0.82, 0.35, 1.00, 1.2, 5),
        ("rehydrated_highfloor_softpow", 0.35, 0.30, 0.84, 0.45, 0.85, 1.2, 5),
        ("rehydrated_midfloor_sharppow", 0.25, 0.30, 0.82, 0.35, 1.25, 1.6, 5),
    ]
    rows: list[dict] = []
    for name, alpha_floor, baseline_blend, coverage_floor_ratio, coverage_floor_mix, support_gate_pow, gaussian_sigma, morph_close_k in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "rehydrated_render_tuning",
                (
                    "Tune rehydrated render with "
                    f"alpha_floor={alpha_floor:.2f}, baseline_blend={baseline_blend:.2f}, "
                    f"coverage_floor_ratio={coverage_floor_ratio:.2f}, coverage_floor_mix={coverage_floor_mix:.2f}, "
                    f"support_gate_pow={support_gate_pow:.2f}, gaussian_sigma={gaussian_sigma:.1f}, "
                    f"morph_close_k={morph_close_k}."
                ),
                {
                    **base,
                    "alpha_floor": alpha_floor,
                    "baseline_blend": baseline_blend,
                    "coverage_floor_ratio": coverage_floor_ratio,
                    "coverage_floor_mix": coverage_floor_mix,
                    "support_gate_pow": support_gate_pow,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_close_k": morph_close_k,
                },
                source_subset=source_subset,
            )
        )
    return rows


def stage_c_label_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
    configs = [
        ("label_consistency_soft", 3, 0.25, 0.03),
        ("label_consistency_mid", 5, 0.45, 0.05),
        ("label_consistency_strong", 7, 0.65, 0.08),
    ]
    rows: list[dict] = []
    for name, label_majority_k, label_smooth_mix, consensus_margin_floor in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "C",
                "label_consistency",
                (
                    "Add label consistency only after visible coverage is safe: "
                    f"label_majority_k={label_majority_k}, label_smooth_mix={label_smooth_mix:.2f}, "
                    f"consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                {
                    **base,
                    "label_majority_k": label_majority_k,
                    "label_smooth_mix": label_smooth_mix,
                    "consensus_margin_floor": consensus_margin_floor,
                },
                source_subset=source_subset,
            )
        )
    return rows
