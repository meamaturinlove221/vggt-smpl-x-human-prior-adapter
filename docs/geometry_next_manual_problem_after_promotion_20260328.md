# Geometry Next Manual Problem After Promotion (2026-03-28)

## Proposed Family

- family: `residual_case_coverage_rebalancing`
- status: `draft_after_promotion`

## Problem Statement

The promoted source-policy lead fixed the global source-slot allocation question strongly enough that the next research question should move away from global routing knobs.

The new question is:

`Can we improve the remaining post-promotion residual by rebalancing dataset-level exposure toward a mined hard-case bucket while keeping the promoted source policy fixed?`

## Why This Is Genuinely New

- It is not a retry of `source_policy_hybrid_ring_regularization`; the promoted source policy stays fixed.
- It is not a cousin of the five closed routing or unprojection families; it changes case coverage, not pixel-level loss routing.
- It targets a new failure mode class: long-tail residual concentration after a globally improved lead.

## Why Not Reopen A Frozen Family

- The five closed families all failed before long gate and therefore did not show evidence of a recoverable frontier.
- The promoted source-policy family already won and was manually promoted, so same-family local tinkering is explicitly forbidden.
- The open gap is no longer "find a better global knob." It is "find which hard-case tail still dominates residual error after the better global knob is already in place."

## Proposed First Candidate

- first candidate shape: `promotedlead_hardcase_bucket_mix`
- candidate idea:
  keep the promoted source policy unchanged, mine a residual hard-case bucket from post-promotion evidence, and mix that bucket into the default training stream with a bounded ratio

## Execution Feasibility

- `first_candidate_requires_code_patch=true`
- `config-only=false`
- `ready_for_manual_review=true`
- `ready_for_execution=false`

Minimal write surface:

- `training/data/datasets/zju_vggt_geom.py`
- `training/config/*.yaml`

Conditional fallback only if the existing composition path proves insufficient in a later dry run:

- `training/data/composed_dataset.py`

## Why Execution Is Not Ready Yet

- The hard-case bucket definition and sampler contract are now canonicalized.
- The official frame-level hard-case manifest still cannot be mined honestly from the remaining promoted-run artifacts because the promoted fine-tune checkpoint is not retained.
- Without that real manifest, the first candidate would still be using a placeholder rather than the reviewed residual tail.

## Manual Review Recommendation

- `ready_for_manual_review=true`
- recommended next step:
  review whether the research team agrees that the remaining post-promotion gap is primarily a dataset-level hard-case coverage problem, not another source-policy or routing problem
