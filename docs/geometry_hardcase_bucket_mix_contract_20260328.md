# Geometry Hard-Case Bucket Mix Contract (2026-03-28)

## Goal

Convert the new family into a single-ticket execution contract without writing code yet.

Family:

- `residual_case_coverage_rebalancing`

First candidate shape:

- `promotedlead_hardcase_bucket_mix`

## Contract Summary

The promoted source policy remains fixed.

The first ticket should test only one new lever:

- mix the default promoted-lead training stream with a static hard-case bucket manifest

This is a dataset-level case-coverage experiment, not a loss-routing or source-policy experiment.

## Proposed Execution Shape

- promoted source policy config stays unchanged as the base behavior
- train stream becomes a two-stream composition:
  - default promoted-lead stream
  - hard-case bucket stream
- first-ticket mix ratio:
  - `default_stream : hard_bucket = 4 : 1`

## Config Surface

- `config_only = false`

But the contract is intentionally narrow:

- single family
- single candidate
- single mix ratio
- strong return-to-guard semantics

## Exact Minimal Files To Touch If Code Patch Starts

Required:

- `training/data/datasets/zju_vggt_geom.py`
- one new config under `training/config/`

Proposed first config filename:

- `training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_hardcasebucketmix4to1_minimal.yaml`

Why `zju_vggt_geom.py` is required:

- the current dataset constructor has no manifest or subset argument for frame-level hard-case filtering
- the new family needs a canonical `sample_manifest_path` or equivalent whitelist input

Why `composed_dataset.py` is not currently required:

- `ComposedDataset` already supports multiple base datasets
- `BaseDataset.__len__` is driven by `len_train`
- `TupleConcatDataset` already samples across the concatenated dataset population
- therefore the mix ratio can be expressed by composing two base datasets and setting their train lengths accordingly

Conditional only if the current composition path proves insufficient in dry config compose:

- `training/data/composed_dataset.py`

## Required Dataset Contract

The future code path must support:

- a hard-case manifest input path
- filtering the train split down to that manifest
- preserving the promoted source policy inside both the default and hard-case streams
- leaving the test split unchanged

The future hard-case manifest should be external to the repo code path, for example under:

- `output/zju_source_policy_research_loop/hardcase_bucket_entries.promotedlead.v1.json`

## Candidate Budget And Stop Rules

- `candidate_budget = 1`
- `same_night_second_candidate_forbidden = true`
- `same_night_cousin_sweep_forbidden = true`
- `do_not_modify_promoted_lead = true`
- `cloud_must_remain_off = true`

## Execution-Readiness Assessment

- `ready_for_manual_review = true`
- `ready_for_execution = false`

Why not execution-ready yet:

1. the hard-case bucket definition is now canonical
2. `ZjuVggtGeomDataset` now accepts a train-only frame-level manifest/subset input
3. the new config now exists
4. but the official hard-case manifest still cannot be materialized honestly because the promoted fine-tune checkpoint is not retained and the surviving promoted-run artifacts only contain aggregate summaries

## Final Assessment

This family is now contract-ready for the next preparation step, but not execution-ready.

The last blocker is no longer "what question should we ask?" The last blocker is:

- `materialize one real frozen hard-case manifest from promoted-lead per-frame residual evidence`
