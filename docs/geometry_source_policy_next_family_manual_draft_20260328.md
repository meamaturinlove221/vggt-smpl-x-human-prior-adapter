# Geometry Source Policy Next Family Manual Draft (2026-03-28)

## Current State

- Track A remains guard-only `steady_hold`.
- Track B remains `IDLE_GUARD`.
- The stable lead is [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml).
- `cloud_gate=false` and `launch_cloud_now=false`.
- There is no active [approved_problem.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/approved_problem.json).
- [repo_process_allowlist.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/repo_process_allowlist.json) is empty.

## Formal Failure Boundary

This five-family batch is closed:

- `interpolated_eligibility_shaping / smoothstep_taper`
  - formal result: `dead_same_day`
  - gate reached: `short_gate_10x5`
  - `delta_conf_depth=+0.0003`
  - `delta_reg_depth=+0.0000`
- `partial_joint_depth_routing / conf_branch_smoothstep_subset`
  - formal result: `dead_same_day`
  - gate reached: `short_gate_10x5`
  - `delta_conf_depth=+0.0015`
  - `delta_reg_depth=+0.0002`
- `conf_reg_disagreement_routing / anchor_disagreement_joint_routing`
  - formal result: `dead_same_day`
  - gate reached: `short_gate_10x5`
  - `delta_conf_depth=+0.0051`
  - `delta_reg_depth=+0.0010`
- `unproject_consistency_routing / anchor_unproject_consistency_joint_routing`
  - formal result: `dead_same_day`
  - gate reached: `short_gate_10x5`
  - `delta_conf_depth=+0.0060`
  - `delta_reg_depth=+0.0012`
- `unproject_aux_confgate / stablelead_unproject_aux_confgate_w005`
  - formal result: `dead_same_day`
  - gate reached: `short_gate_10x5`
  - `delta_conf_depth=+0.0079`
  - `delta_reg_depth=+0.0013`

## Shared Unresolved Dimension

- All five first tickets were real research failures, not infra failures.
- All five kept `camera/T` broadly stable but still lost on the depth terms.
- The first four changed how the main depth losses were routed or reweighted.
- The fifth changed only the auxiliary `unproject_geometry` branch and still failed.
- None asked whether the next step should stop touching loss routing entirely and instead regularize the source-policy rule itself on top of the current stable lead.

## Frozen Directions

- no interpolated cousins
- no partial cousins
- no conf-reg-disagreement cousins
- no unproject-consistency cousins
- no unproject-aux cousins
- no wholefg cousins
- no edge-band cousins
- no hard-threshold reopen
- no pow-like reopen
- no plain `anchor_view_only` reopen
- no reliable-region auxiliary-unproject reopen
- no bottom-only auxiliary-unproject reopen
- no uniform-only rawpool reopen
- no `min_supervised_views=2` reopen
- no `max_depth_conf` anchor reopen
- no trainmix reopen

## New Manual Problem Draft

### Proposed Family

- `source_policy_hybrid_ring_regularization`

### First Candidate Shape

- `stablelead_nearest_plus_uniform_tail`

### Core Hypothesis

- The current stable lead already fixes the earlier camera/T regression, but still leaves a depth-side gap.
- The last closed batch exhausted both depth-loss routing cousins and one auxiliary-unproject cousin.
- A more credible next family is to leave the stable lead's loss route untouched and regularize only the source-policy rule that chooses rawpool source views.
- Earlier source-policy evidence showed:
  - pure `uniform_rawpool` broadened coverage but hurt camera/T too much
  - pure `nearest_rawpool` is the stable training lead
  - so the next bounded rule should keep nearest-ring locality while forcing one wider-coverage tail source
- So the first candidate should keep the stable lead intact and change only:
  - `zju_source_policy: nearest_ring -> nearest_plus_uniform_tail`

### Why This Is Genuinely New

- It changes the data-side source-view rule, not the conf/reg depth routing maps.
- It does not change anchor-conditioned depth-loss selectivity at all.
- It does not retune the auxiliary unproject branch either.
- It therefore sits outside the closed routing-plus-auxiliary batch.

### Why It Is Not A Cousin

- Not an interpolated cousin:
  - no depth-conf taper is applied to the main depth routing masks
- Not a partial cousin:
  - no conf-branch subset routing is introduced
- Not a conf-reg-disagreement cousin:
  - no disagreement map is built inside `regression_loss(...)`
- Not an unproject-consistency cousin:
  - it does not route depth supervision from unprojection residual
- Not an unproject-aux cousin:
  - it does not change `loss.unproject_geometry`
- Not a rejected sampler reopen:
  - it is neither pure `uniform_rawpool`, nor `min_supervised_views=2`, nor `max_depth_conf` anchor, nor `trainmix50`

### Signal Definition Draft

- Keep the current stable lead:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)
- Change only the source-policy rule:
  - `zju_source_policy = nearest_plus_uniform_tail`
  - keep `zju_source_view_pool = geom_plus_raw`
  - keep `zju_min_supervised_views = 1`
  - keep `respect_conf_mask_in_grad_conf = True`

## Minimal Write Surface

- one dataset-side source-policy helper in [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
- one new config under [training/config](/f:/vggt/vggt-main/training/config)
- planning / seed / blueprint files for the research loop

## Feasibility Readout

- Dataset plumbing required: yes, but bounded
- Loss-side implementation required: no
- Compare or runner plumbing required: no
- Why:
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py) already has ring-order helpers and source-policy selection hooks
  - the candidate keeps the current stable lead's loss route unchanged
  - once the hybrid source-policy helper is added, the first candidate is a stable-lead-derived config-only follow-up

## Execution Outcome

- The first approved ticket has already run.
- Formal result: `provisional_lead`
- Gate reached: `long_gate_100x20`
- Research is back in `IDLE_GUARD`.
- The archived result now waits for a fresh manual promotion decision.

## Current Decision State

- `ready_for_manual_approval = false`
- `ready_for_execution = false`
- `waiting_for_manual_promotion_decision = true`
- `auto_arm_forbidden = true`
- `auto_run_forbidden = true`

### Why This Draft Still Matters Now

- It records why `source_policy_hybrid_ring_regularization` was considered genuinely new.
- It is the source document for the current provisional local lead.
- The next human action is promotion review, not another implementation or auto-run.

## Operational Rule

- keep guard healthy
- keep research local
- keep cloud off
- keep the first ticket archived
- wait for a fresh manual promotion decision before any further move
