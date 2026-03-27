# Geometry Source Policy Failure Boundary And Next Problem (2026-03-27)

## Executive Status

- Track A remains guard-only `steady_hold`.
- Track B remains `IDLE_GUARD`.
- The current stable lead is [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml).
- `cloud_gate` stays `false`.
- There is no active [approved_problem.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/approved_problem.json).
- [repo_process_allowlist.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/repo_process_allowlist.json) remains empty.
- The research loop now has no auto-recommended next family; [frontier_ledger.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/frontier_ledger.json) records `recommended_next_families=[]`.

## Formal First-Ticket Outcomes

### 1. `interpolated_eligibility_shaping / smoothstep_taper`

- Approved ticket:
  - [20260327_130957_camera_b1_interpolated_eligibility_shaping_v1.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/approved_problem_archive/20260327_130957_camera_b1_interpolated_eligibility_shaping_v1.json)
- `10/5` summary:
  - [summary.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/runs/20260327_130618_zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfsmoothstepp60jointdepthscale0875_minimal/short_vs_lead/summary.json)
- Formal result:
  - entered `10/5`
  - verdict = `dead_same_day`
  - `camera/T` stayed flat
  - `conf_depth: 0.2288 -> 0.2291` (`+0.0003`)
  - `reg_depth: 0.1759 -> 0.1759` (`+0.0000`)

### 2. `partial_joint_depth_routing / conf_branch_smoothstep_subset`

- Approved ticket:
  - [20260327_135659_camera_b1_partial_joint_depth_routing_v1.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/approved_problem_archive/20260327_135659_camera_b1_partial_joint_depth_routing_v1.json)
- `10/5` summary:
  - [summary.json](/f:/vggt/vggt-main/output/zju_source_policy_research_loop/runs/20260327_135211_zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5partialjointconfsmoothstepp60jointdepthscale0875_minimal/short_vs_lead/summary.json)
- Formal result:
  - entered `10/5`
  - verdict = `dead_same_day`
  - `camera/T` stayed flat
  - `conf_depth: 0.2288 -> 0.2303` (`+0.0015`)
  - `reg_depth: 0.1759 -> 0.1761` (`+0.0002`)

## Boundary Summary

- These are both research failures, not infra failures.
- Both tickets stayed inside the same broad edge-band, raw `depth_conf`-driven routing line.
- Both tickets improved or held train-side depth losses, but neither produced a validation crossing at `10/5`.
- The partial ticket improved train depth losses more strongly than the interpolated ticket, yet validation regressed more. That is a useful negative result: more aggressive redistribution within the same signal family appears to overfit train without fixing the validation blocker.
- The common unresolved dimension is therefore not "make the same `depth_conf` rule slightly softer or slightly more branch-specific." The shared failure suggests the remaining headroom is not another interpolated/partial near-neighbor inside the same pixel-selectivity family.
- [geometry_direction_status_20260323_threshold_and_pow2_completed.md](/f:/vggt/vggt-main/docs/geometry_direction_status_20260323_threshold_and_pow2_completed.md) already froze hard-threshold and sharpened-weighting directions earlier, and the newer edge-band threshold failure froze the abrupt-rule variant again from the current line.

## What Is Frozen From This State

- no `interpolated_eligibility_shaping` cousin sweep
- no `partial_joint_depth_routing` cousin sweep
- no fallback to whole-foreground near-neighbors
- no fallback to edge-band scalar or edge-band decoupled near-neighbors
- no reopening hard `depth_conf` thresholds
- no reopening pow-like sharpened weighting
- no reopening plain `anchor_view_only`

## New Manual Problem Draft

### Proposed Family

- `conf_reg_disagreement_routing`

### Draft Question

- Can the next candidate route depth supervision by a new disagreement signal between the confidence-depth and regression-depth branches on anchor-supervised pixels, instead of reusing raw `depth_conf` shaping or the existing partial conf-branch subset, so that the residual `Camera_B1` depth gap is attacked on a new dimension without harming `camera/T`?

### Disagreement Signal Definition Draft

- Start from the existing per-pixel tensors already computed inside [loss.py](/f:/vggt/vggt-main/training/loss.py):
  - `pred_depth_conf`
  - `reg_map = ||gt - pred||`
- Restrict the signal to anchor-supervised pixels only:
  - pixels must already be valid in `point_masks`
  - pixels must already be valid in `conf_depth_point_masks`
  - pixels must belong to the current `selection_anchor_view_index`
- Build one detached disagreement map:
  - normalize detached `reg_map` within the active anchor-supervised mask
  - compare it against detached inverse confidence `1 - pred_depth_conf`
  - use the absolute gap between those two terms as the routing signal
- Interpret the signal as "the confidence branch and the regression branch disagree on how trustworthy this anchor pixel is," which is different from simply reusing raw `depth_conf` as the routing score.

### Why This Is Not An Interpolated Or Partial Cousin

- It changes the routing signal itself, from raw `depth_conf` to branch disagreement.
- It does not reopen the same `smoothstep` family with a new cutoff, percentile, or interpolation curve.
- It does not reopen the same partial conf-branch subset with a different subset boundary or scale.
- It does not rely on another whole-foreground, edge-band, bottom-band, hard-threshold, or pow-like reskin.
- It targets the possibility that the current `depth_conf` ranking is misaligned with the validation failure mode, which neither failed first ticket directly challenged.

### Minimum Implementation Surface For Future Review

- [loss.py](/f:/vggt/vggt-main/training/loss.py)
- one new candidate config under [training/config](/f:/vggt/vggt-main/training/config)

### Feasibility Readout

- Existing tensors are sufficient for a first ticket:
  - [loss.py](/f:/vggt/vggt-main/training/loss.py) already has both `pred_depth_conf` and `reg_map` in the same depth-loss path.
  - [composed_dataset.py](/f:/vggt/vggt-main/training/data/composed_dataset.py) already forwards `point_masks`, `conf_depth_point_masks`, `depth_conf_maps`, `selection_anchor_camera`, `selection_anchor_view_index`, and `selection_anchor_quality_score`.
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py) already constructs anchor-aware supervised masks and depth-confidence maps per view.
  - [trainer.py](/f:/vggt/vggt-main/training/trainer.py) already preserves `selection_anchor_view_index` and related metadata through batch repetition.
- New dataset plumbing is not required for the first ticket.
- Compare plumbing is already generic:
  - [compare_zju_finetune_runs.py](/f:/vggt/vggt-main/scripts/compare_zju_finetune_runs.py) compares whatever `Loss/train_*` and `Loss/val_*` metrics appear in the log, so the standard formal verdict path can stay unchanged.
- The smallest expected write surface is therefore:
  - `training/loss.py`
  - one new config in `training/config`
- No new runner or cloud automation work is required.

### Single First-Candidate Shape

- `anchor_disagreement_joint_routing`
- Only one first shape is allowed for review.
- Draft behavior:
  - use the detached disagreement map on anchor-supervised pixels only
  - downscale the confidence-target term where disagreement is high
  - upweight the regression-target term on the same pixels
  - leave non-anchor and non-supervised pixels unchanged
- This is intentionally not a menu of cousins. The first ticket should test whether disagreement itself is a useful routing signal before any shape sweep exists.

### Execution Readiness

- No dataset blocker is present.
- No compare blocker is present.
- The only earlier blocker has now been resolved in [loss.py](/f:/vggt/vggt-main/training/loss.py):
  - the new disagreement routing path now applies an explicit anchor-only mask instead of relying on the older conf-target path that hardcoded `anchor_conditioned_conf_target_anchor_view_only=False`
- The first ticket is now executable from the repo state without another code patch.

### Hard Constraints On The Draft

- not an `interpolated_eligibility_shaping` cousin
- not a `partial_joint_depth_routing` cousin
- not a wholefg reopen
- not an edge-band reopen
- not a hard-threshold reopen
- not a pow-style reopen
- not a plain `anchor_view_only` reopen
- manual review only; no active approval may be generated from this draft tonight
- if ever approved later, it still has to obey the single-ticket contract: `1/1 -> 10/5 -> 100/20 -> return_to_guard`

### Manual-Approval Readiness

- `ready_for_manual_approval = true`
- `ready_for_execution = true`
- Reason:
  - the proposal is a new routing dimension rather than a same-family reskin
  - the first ticket can stay single-candidate
  - the write surface is small and local
  - there is no missing dataset or compare plumbing blocker
  - the anchor-only disagreement detail is already implemented inside the repo

## Operational Rule Tonight

- refresh guard state only
- keep research in `IDLE_GUARD`
- do not arm any new approved problem
- do not run [run_zju_source_policy_research_candidate.py](/f:/vggt/vggt-main/scripts/run_zju_source_policy_research_candidate.py)
- do not treat heartbeat, watch-cycle growth, or snapshot timestamps as research progress

## Decision For Tomorrow

- The next step is not "run another ticket."
- The next step is manual review of a genuinely new problem statement.
- Only after that manual review may a new approved problem be authored.
