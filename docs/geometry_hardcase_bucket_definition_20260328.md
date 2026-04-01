# Geometry Hard-Case Bucket Definition (2026-03-28)

## Scope

This document defines the canonical post-promotion hard-case bucket for the new family:

- `residual_case_coverage_rebalancing`

This is a planning artifact only.

- no `approved_problem.json`
- no `arm`
- no candidate run
- no cloud action

## Authoritative Preconditions

The definition below is only valid while the live state remains:

- `research_loop_status.state = IDLE_GUARD`
- `approved_problem_present = false`
- `current_priority_family = ""`
- `auto-next ticket = false`
- promoted local lead:
  `training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml`
- `cloud_must_remain_off = true`

## Bucket Intent

The promoted lead already solved the global source-selection problem well enough that the next question should not reopen source-policy or routing cousins.

The bucket therefore targets the remaining residual tail:

- not the global average stream
- not another pixel-routing knob
- not another same-family source-policy retry

## Canonical Case Granularity

The canonical bucket unit is:

- `frame-level training entry`

Primary key fields:

- `seq_name`
- `frame_id`

Diagnostic metadata to retain, but not use as the serving key:

- `promoted_anchor_camera`
- `selected_supervised_camera_names`
- `selected_source_only_camera_names`

Reason:

- `ZjuVggtGeomDataset` already builds a frame-indexed `sequence_list`, so frame-level bucketing matches the existing dataset boundary.
- Keeping the serving key at frame granularity keeps the future sampler contract narrow and avoids turning this into an anchor-policy rewrite.

## Canonical Hard-Case Metric

The canonical hard-case score is:

- `joint_depth_geom_tail_score`

Definition:

`joint_depth_geom_tail_score = 0.45 * pct(conf_depth) + 0.35 * pct(reg_depth) + 0.20 * pct(unproject_geometry)`

Where:

- each `pct(...)` term is the within-split percentile rank of the promoted-lead residual for that frame-level entry
- higher is worse
- the residual is computed only from promoted-lead outputs
- the residual is aggregated over active supervised views for that entry

Additional rules:

- `camera` and `T` remain tracked as diagnostics, not primary bucket drivers
- the bucket is defined by post-promotion residual under the promoted lead, not by delta-vs-previous-lead

Reason:

- the promoted lead already won broadly on camera, translation, depth, and geometry
- the remaining gap should be mined from the residual tail under the new lead itself
- using percentile ranks keeps the score scale-stable across metrics

## Threshold Rule

The canonical threshold rule is:

- select the worst `8%` of eligible training entries by `joint_depth_geom_tail_score`

Tie-break order:

1. worse `conf_depth`
2. worse `reg_depth`
3. worse `unproject_geometry`

Eligibility:

- split must be `train`
- entry must have valid promoted-lead residual readout for all three primary metrics

## Static Or Refreshable

The bucket is:

- `refreshable between manual-review cycles`
- `static within a ticket`

Meaning:

- a mining pass may regenerate the bucket before a future manual approval
- once a bucket manifest is chosen for an approved ticket, that manifest must be frozen for the whole ticket

## Allowed Mix Ratio Candidates

Allowed default-stream to hard-bucket mix candidates:

- `4:1` recommended first ticket
- `3:1` allowed only by a new manual decision

The current recommended first candidate ratio is:

- `4:1`

Reason:

- it increases exposure to the residual tail without collapsing into near-global resampling

## Anti-Expansion Constraints

The bucket must not expand into a disguised global resampling policy.

Hard constraints:

- bucket coverage must stay `<= 8%` of eligible training entries
- bucket selection must come from an explicit manifest of frame entries
- bucket membership must be defined from promoted-lead residuals, not from a generic wholefg/global routing heuristic
- bucket membership must not change source policy, source-view pool, or loss routing
- no second ratio on the same night
- no multiple bucket variants on the same night

Interpretation constraints:

- if the selected tail starts looking like "almost every low-confidence frame", the definition is too broad and must be rejected
- if the bucket cannot be explained as a residual tail under the promoted lead, it is not a valid hard-case bucket for this family

## Future Materialization Shape

The future machine-readable manifest should eventually contain frame-entry rows shaped like:

- `seq_name`
- `frame_id`
- `joint_depth_geom_tail_score`
- `conf_depth_percentile`
- `reg_depth_percentile`
- `unproject_geometry_percentile`
- `promoted_anchor_camera`

That manifest is not created by this document. This document only defines the canonical truth it must follow.

## Readiness Conclusion

- `ready_for_manual_review = true`
- `ready_for_execution = false`

Remaining blocker:

- the hard-case bucket has now been defined conceptually, but the actual promoted-lead frame manifest still cannot be mined honestly because the promoted fine-tune checkpoint is not retained in the current workspace and the remaining live artifacts only preserve aggregate summaries
