# Geometry Post-Promotion Gap Mining (2026-03-28)

## Authoritative Live State

- `research_loop_status.json` is `IDLE_GUARD`
- `approved_problem_present=false`
- `current_priority_family=""`
- `auto-next ticket=false`
- promoted local lead:
  `training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml`
- `cloud_must_remain_off=true`

This note is a synthesis-only artifact. It does not arm a ticket, run a candidate, or change the promoted lead.

## What The Promotion Already Solved

The promoted lead came from the source-policy side, not from loss-routing or auxiliary geometry routing.

Evidence against the previous local lead:

- short gate 10/5:
  - `camera -0.0039`
  - `T 0.0000`
  - `conf_depth -0.0516`
  - `reg_depth -0.0565`
  - `loss_unproject_geometry -0.0563`
  - `loss_grad_depth -0.1925`
- long gate 100/20:
  - `camera -0.0454`
  - `T -0.0208`
  - `conf_depth -0.1101`
  - `reg_depth -0.0704`
  - `loss_unproject_geometry -0.1402`
  - `loss_grad_depth -0.4086`

Interpretation:

- The win is broad, not narrow. Camera, translation, depth confidence, depth regression, gradient depth, and unprojection geometry all improved together.
- Because the winning change was a source-slot policy change, the strongest positive slice is the source-selection-sensitive slice.
- Unprojection geometry improved as a downstream effect even though direct unproject-routing families failed, which argues that the main bottleneck was not another global unproject residual routing knob.

## What Still Looks Unsolved

The live artifacts do not show another obvious global scalar or routing knob that is still underexploited.

What remains open is a different question:

- The promoted lead fixed the average behavior strongly.
- The remaining error is therefore more likely concentrated in a long-tail of difficult cases than in the global average recipe.
- The current live artifacts do not yet isolate those hard cases into an actionable training-time handle.

This means the next research gap is best framed as:

`Which residual cases remain under-covered after the promoted source policy, and can dataset-level hard-case coverage rebalance them without reopening source-policy or routing families?`

## Why The Old Families Should Stay Closed

- `interpolated_eligibility_shaping`: died at short gate and only softened an already wrong-side routing frontier.
- `partial_joint_depth_routing`: died at short gate and stayed inside the same routing idea class.
- `conf_reg_disagreement_routing`: died at short gate and still behaved like a global residual-routing knob.
- `unproject_consistency_routing`: died at short gate; direct unprojection residual routing was not the missing lever.
- `unproject_aux_confgate`: died at short gate; auxiliary unproject gating did not translate into better global metrics.
- `source_policy_hybrid_ring_regularization`: already spent its single-ticket family budget and is now the promoted local lead, so a same-family retry is contract-forbidden.

## Recommended New Family

Recommended genuinely new family:

- `residual_case_coverage_rebalancing`

Why this is genuinely new:

- It acts at dataset-level case coverage, not per-pixel loss routing.
- It does not alter the promoted intra-sample source-slot policy.
- It asks whether the remaining residual is concentrated in under-covered hard cases, which none of the closed families tested.

## Proposed First Candidate

- first candidate shape: `promotedlead_hardcase_bucket_mix`
- intent: keep the promoted source policy fixed, but rebalance training exposure toward a small mined residual bucket of hard cases while preserving a stable proportion of the promoted default stream

## Execution Feasibility

- `ready_for_manual_review=true`
- `ready_for_execution=false`
- `config-only=false`

Minimal planned write surface:

- `training/data/composed_dataset.py`
- `training/config/*.yaml`

Possible conditional extension if sampler metadata is missing:

- `training/data/datasets/zju_vggt_geom.py`

Current blockers before execution-ready:

- We do not yet have a canonical post-promotion hard-case bucket definition.
- The sampler contract for mixing a residual bucket with the promoted default stream is not yet specified.
- The first ticket should only be prepared after that bucket definition is reviewed as a genuinely new manual problem.

## Bottom Line

The promoted lead solved the global source-selection question well enough that reopening source-policy or routing cousins would be low-value.

The real remaining research gap is now case-selective residual coverage:

- not another same-family source-policy tweak
- not another global per-pixel routing knob
- not another unprojection auxiliary gate

The next manual problem should therefore target post-promotion hard-case coverage as a new family, starting from the promoted lead rather than trying to out-tune it locally.
