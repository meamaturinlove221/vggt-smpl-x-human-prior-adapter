from itertools import combinations


def _base_proxy_config() -> dict:
    return {
        "render_mode": "rehydrated",
        "source_subset": [],
        "alpha_floor": 0.15,
        "support_gate_pow": 1.0,
        "baseline_blend": 0.30,
        "gaussian_sigma": 0.8,
        "morph_close_k": 3,
        "label_majority_k": 5,
        "label_smooth_mix": 0.45,
        "consensus_margin_floor": 0.05,
        "coverage_floor_ratio": 0.78,
        "coverage_floor_mix": 0.35,
    }


def _mutation(mid: str, family: str, description: str, proxy_config: dict, *, dropped_sources: list[int] | None = None) -> dict:
    return {
        "mutation_id": mid,
        "family": family,
        "description": description,
        "proxy_config": proxy_config,
        "dropped_sources": list(dropped_sources or []),
    }


def build_mutation_candidates(
    *,
    failure_class: str,
    source_count: int,
    tried_ids: set[str],
    best_subset: list[int] | None = None,
) -> list[dict]:
    best_subset = list(best_subset or list(range(source_count)))
    candidates = []

    def add(item: dict) -> None:
        if item["mutation_id"] not in tried_ids:
            candidates.append(item)

    for drop_idx in best_subset:
        subset = [idx for idx in best_subset if idx != drop_idx]
        add(
            _mutation(
                f"source_leave_one_out_drop_{drop_idx}",
                "source_selection",
                f"Leave-one-out source pruning by removing source index {drop_idx}.",
                {**_base_proxy_config(), "source_subset": subset},
                dropped_sources=[drop_idx],
            )
        )

    render_configs = [
        ("rehydrated_alpha015_blend015_pow075", 0.15, 0.15, 0.75),
        ("rehydrated_alpha025_blend030_pow100", 0.25, 0.30, 1.00),
        ("rehydrated_alpha035_blend045_pow125", 0.35, 0.45, 1.25),
    ]
    if failure_class in {"erasure_win", "no_movement"}:
        for name, alpha_floor, baseline_blend, gate_pow in render_configs:
            add(
                _mutation(
                    name,
                    "render_rehydration",
                    f"Use rehydrated render alpha_floor={alpha_floor:.2f}, baseline_blend={baseline_blend:.2f}, support_gate_pow={gate_pow:.2f}.",
                    {
                        **_base_proxy_config(),
                        "source_subset": best_subset,
                        "alpha_floor": alpha_floor,
                        "baseline_blend": baseline_blend,
                        "support_gate_pow": gate_pow,
                    },
                )
            )

    if len(best_subset) >= 4:
        for keep_pair in list(combinations(best_subset, len(best_subset) - 2))[:6]:
            subset = list(keep_pair)
            dropped = [idx for idx in best_subset if idx not in subset]
            add(
                _mutation(
                    f"source_prune_drop_{'_'.join(str(idx) for idx in dropped)}",
                    "source_pruning",
                    f"Prune two sources {dropped} while keeping the remaining source set fixed.",
                    {**_base_proxy_config(), "source_subset": subset},
                    dropped_sources=dropped,
                )
            )

    label_configs = [
        ("label_consistency_soft", 3, 0.25, 0.03),
        ("label_consistency_mid", 5, 0.45, 0.05),
        ("label_consistency_strong", 7, 0.65, 0.08),
    ]
    if failure_class in {"fragmentation_win", "no_movement"}:
        for name, majority_k, mix, margin_floor in label_configs:
            add(
                _mutation(
                    name,
                    "source_label_consistency",
                    f"Adjust majority smoothing to k={majority_k}, mix={mix}, margin_floor={margin_floor}.",
                    {
                        **_base_proxy_config(),
                        "source_subset": best_subset,
                        "label_majority_k": majority_k,
                        "label_smooth_mix": mix,
                        "consensus_margin_floor": margin_floor,
                    },
                )
            )

    coverage_configs = [
        ("coverage_floor_soft", 0.80, 0.35),
        ("coverage_floor_mid", 0.82, 0.50),
        ("coverage_floor_strong", 0.85, 0.65),
    ]
    if failure_class == "erasure_win":
        for name, floor_ratio, floor_mix in coverage_configs:
            add(
                _mutation(
                    name,
                    "coverage_floor",
                    f"Raise coverage floor to {floor_ratio:.2f} with mix {floor_mix:.2f}.",
                    {
                        **_base_proxy_config(),
                        "source_subset": best_subset,
                        "coverage_floor_ratio": floor_ratio,
                        "coverage_floor_mix": floor_mix,
                    },
                )
            )

    smoothing_configs = [
        ("support_smoothing_soft", 0.8, 3),
        ("support_smoothing_mid", 1.2, 5),
        ("support_smoothing_strong", 1.6, 7),
    ]
    if failure_class in {"fragmentation_win", "background_only_win"}:
        for name, sigma, close_k in smoothing_configs:
            add(
                _mutation(
                    name,
                    "support_smoothing",
                    f"Apply support smoothing sigma={sigma}, close_k={close_k}.",
                    {
                        **_base_proxy_config(),
                        "source_subset": best_subset,
                        "gaussian_sigma": sigma,
                        "morph_close_k": close_k,
                    },
                )
            )

    return candidates
