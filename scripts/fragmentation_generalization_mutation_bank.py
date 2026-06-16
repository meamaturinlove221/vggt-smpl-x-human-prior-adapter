from __future__ import annotations

from itertools import combinations


def _base_proxy_config() -> dict:
    return {
        "render_mode": "rehydrated",
        "source_subset": [],
        "alpha_floor": 0.25,
        "support_gate_pow": 1.25,
        "baseline_blend": 0.30,
        "gaussian_sigma": 1.6,
        "morph_close_k": 5,
        "label_majority_k": 3,
        "label_smooth_mix": 0.25,
        "consensus_margin_floor": 0.05,
        "coverage_floor_ratio": 0.82,
        "coverage_floor_mix": 0.30,
        "visible_floor_ratio": 1.00,
        "visible_floor_mix": 0.30,
        "visible_mass_floor_ratio": 0.98,
        "visible_mass_floor_mix": 0.30,
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
    rows = []
    for subset in subsets:
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)
        rows.append(list(subset))
    return rows


def stage_a_seed_mutations(source_count: int, *, prior_subsets: list[list[int]]) -> list[dict]:
    base = _base_proxy_config()
    full_subset = list(range(source_count))
    candidate_subsets = _unique_subsets(
        [
            *prior_subsets,
            full_subset,
            [idx for idx in full_subset if idx != 3] if source_count > 3 else full_subset,
            [idx for idx in full_subset if idx not in {2, 3}] if source_count > 3 else full_subset,
            [idx for idx in full_subset if idx not in {1, 3}] if source_count > 3 else full_subset,
        ]
    )

    rows: list[dict] = []
    for rank, subset in enumerate(candidate_subsets[:6], start=1):
        rows.append(
            _mutation(
                f"seed_subset_{rank:02d}",
                "A",
                "fragmentation_seed_search",
                f"Seed fragmentation-generalization search with source subset {subset}.",
                {**base, "source_subset": subset},
                source_subset=subset,
            )
        )

    if len(rows) < 6:
        for a, b in combinations(full_subset, 2):
            subset = [idx for idx in full_subset if idx not in {a, b}]
            if any(item["source_subset"] == subset for item in rows):
                continue
            rows.append(
                _mutation(
                    f"seed_drop_{a}_{b}",
                    "A",
                    "fragmentation_seed_search",
                    f"Fallback seed search dropping sources {a} and {b}.",
                    {**base, "source_subset": subset},
                    source_subset=subset,
                )
            )
            if len(rows) >= 6:
                break
    return rows[:6]


def stage_b_generalization_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
    configs = [
        ("softsmooth_v1", 1.00, 0.30, 1.2, 5, 5, 0.35, 0.04, 0.30, 0.98, 0.98),
        ("softsmooth_v2", 1.00, 0.35, 1.6, 7, 5, 0.45, 0.04, 0.30, 0.99, 0.98),
        ("majority_focus_v1", 1.05, 0.35, 1.6, 7, 7, 0.55, 0.05, 0.32, 1.00, 0.99),
        ("softgate_anchor_v1", 0.90, 0.45, 1.2, 5, 5, 0.45, 0.03, 0.35, 1.00, 0.99),
        ("conservative_v1", 1.10, 0.30, 0.8, 3, 3, 0.25, 0.06, 0.25, 0.98, 0.98),
        ("midclose_v1", 1.00, 0.30, 1.6, 5, 5, 0.45, 0.05, 0.30, 1.00, 0.99),
    ]
    rows: list[dict] = []
    for (
        name,
        support_gate_pow,
        baseline_blend,
        gaussian_sigma,
        morph_close_k,
        label_majority_k,
        label_smooth_mix,
        consensus_margin_floor,
        visible_floor_mix,
        visible_floor_ratio,
        visible_mass_floor_ratio,
    ) in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "B",
                "fragmentation_generalization_tuning",
                (
                    "Tune consensus-margin generalization with "
                    f"support_gate_pow={support_gate_pow:.2f}, baseline_blend={baseline_blend:.2f}, "
                    f"gaussian_sigma={gaussian_sigma:.1f}, morph_close_k={morph_close_k}, "
                    f"label_majority_k={label_majority_k}, label_smooth_mix={label_smooth_mix:.2f}, "
                    f"consensus_margin_floor={consensus_margin_floor:.2f}."
                ),
                {
                    **base,
                    "support_gate_pow": support_gate_pow,
                    "baseline_blend": baseline_blend,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_close_k": morph_close_k,
                    "label_majority_k": label_majority_k,
                    "label_smooth_mix": label_smooth_mix,
                    "consensus_margin_floor": consensus_margin_floor,
                    "visible_floor_ratio": visible_floor_ratio,
                    "visible_floor_mix": visible_floor_mix,
                    "visible_mass_floor_ratio": visible_mass_floor_ratio,
                    "visible_mass_floor_mix": max(base.get("visible_mass_floor_mix", 0.30), visible_floor_mix),
                },
                source_subset=source_subset,
            )
        )
    return rows


def stage_c_label_generalization_mutations(source_subset: list[int], *, prefix: str, seed_config: dict) -> list[dict]:
    base = {**_base_proxy_config(), **seed_config, "source_subset": list(source_subset)}
    configs = [
        ("labelsmooth_soft", 5, 0.45, 1.6, 5),
        ("labelsmooth_mid", 7, 0.55, 1.6, 7),
        ("labelsmooth_strong", 7, 0.65, 2.0, 7),
    ]
    rows: list[dict] = []
    for name, label_majority_k, label_smooth_mix, gaussian_sigma, morph_close_k in configs:
        rows.append(
            _mutation(
                f"{prefix}_{name}",
                "C",
                "fragmentation_label_generalization",
                (
                    "Escalate label/generalization smoothing only after smoke+control are honest: "
                    f"label_majority_k={label_majority_k}, label_smooth_mix={label_smooth_mix:.2f}, "
                    f"gaussian_sigma={gaussian_sigma:.1f}, morph_close_k={morph_close_k}."
                ),
                {
                    **base,
                    "label_majority_k": label_majority_k,
                    "label_smooth_mix": label_smooth_mix,
                    "gaussian_sigma": gaussian_sigma,
                    "morph_close_k": morph_close_k,
                },
                source_subset=source_subset,
            )
        )
    return rows
