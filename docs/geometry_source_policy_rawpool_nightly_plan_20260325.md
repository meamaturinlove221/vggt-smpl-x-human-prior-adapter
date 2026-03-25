# Geometry Source-Policy Rawpool Nightly Plan (2026-03-25)

## Summary

- The current local training lead is:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)
- This lead keeps the same `nearest_ring + geom_plus_raw` input path and preserves the `depth + camera -> unproject -> render` geometry chain.
- Relative to the previous `confdepth_dropworst_only` lead, it improves `val_loss_conf_depth` and `val_loss_reg_depth` while keeping `val_loss_camera` and `val_loss_T` flat.
- It is still not baseline-clean on `conf_depth` and `reg_depth`, so cloud remains closed.
- The latest completed `conf_depth attribution v3` pass points to one narrow next move:
  - `anchor_conditioned_conf_target_normalization`
  - worst positive camera-level delta remains concentrated in anchor `Camera_B1`
- The single allowed anchor-conditioned follow-up was exercised locally as `anchorb1confscale05` and failed the tighter gate.
- The nightly default is now a hold state: `preflight -> consistency check -> status emit`. No training runs unless a fresh manual training question explicitly approves one genuinely new candidate.

## Current Lead

- Gate note:
  - [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md)
- Compare versus previous lead:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_lead_20260325_v1/summary.md)
- Compare versus baseline:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_baseline_20260325_v1/summary.md)
- Current attribution evidence:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3/summary.md)
  - On the audited 32-sample val slice, `conf_depth` remains anchor-only.
  - The only clearly positive camera-level delta is concentrated in anchor `Camera_B1`.
  - The current local recommendation is `anchor_conditioned_conf_target_normalization`.
- Failed anchor-conditioned follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_baseline_20260325_v1/summary.md)
  - The train-only `Camera_B1` conf-target scale `0.5` candidate reduced train `conf_depth`, but on the same val metric it regressed `loss_conf_depth` from `0.2289 -> 0.3233` and `loss_reg_depth` from `0.1760 -> 0.1901` versus the current lead.
- Historical stage marker, not current lead:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml)
  - [geometry_source_policy_rawpool_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_local_gate_20260325.md)

## Frozen Non-Reentry List

- `min_supervised_views = 2`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md)
- `source_anchor_policy = max_depth_conf`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md)
- `dropworst_supervised`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md)
- `trainmix50`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_trainmix50_vs_lead_20260325_v1/summary.md)
- `active_view_mean`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
- `anchorb1confscale05`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
- "exclude rawpool-only views from geometry supervision" as a standalone candidate
  - [summary.md](/f:/vggt/vggt-main/output/zju_source_policy_supervision_audit_nearest_rawpool_20260325_v1/summary.md)

## Completed Diagnostic

- Planned entrypoint:
  - [audit_zju_conf_depth_attribution.py](/f:/vggt/vggt-main/scripts/audit_zju_conf_depth_attribution.py)
- Diagnostic target:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)
- Required outputs for `conf_depth attribution v3`:
  - anchor summary
  - frame summary
  - region summary
  - candidate recommendation
- Result:
  - `dominant_failure_shape = anchor_conditioned`
  - `recommended_candidate_family = anchor_conditioned_conf_target_normalization`
- The rawpool supervision-leak question is already treated as settled local evidence. Re-run that audit only when a future candidate changes dataset or loss routing.

## Default Nightly Sequence

1. Preflight
  - Require `modal app list` to be empty.
  - Require no stale repo-scoped `python / powershell / modal` processes.
  - Do not stop active Modal apps automatically.
2. Run consistency check
   - Refresh the local status snapshot.
   - Do not rerun `conf_depth attribution v3` by default.
   - Do not open a sibling candidate automatically after the failed `anchorb1confscale05` round.
3. Emit status only
   - Keep the current lead fixed.
   - Keep `cloud_gate=false`.

## Single Candidate Local Gate

- This mode is allowed only when exactly one genuinely new local candidate is checked in.
- That candidate must be a non-redundant `conf_depth semantics` hypothesis.
- It must be materially different from:
  - `min_supervised_views = 2`
  - `source_anchor_policy = max_depth_conf`
  - `dropworst_supervised`
  - `trainmix50`
  - `active_view_mean`
  - "exclude rawpool-only views from geometry supervision" as a standalone change
- Sequence:
  1. Run preflight.
  2. Run the supervision audit only if dataset or loss routing changed.
  3. Run `1 train / 1 val` smoke.
  4. Run one tighter local gate against:
     - baseline
     - current `confdepth_dropworst_gradconfmask` lead
  5. Stop after that one candidate.
- This mode is no longer automatic; it requires a fresh manual training question because the first anchor-conditioned round has already been consumed and failed.

## Promotion Gate

- A candidate may move beyond nightly local gate only if it simultaneously:
  - keeps `val_loss_camera` no worse than the current `confdepth_dropworst_gradconfmask` lead
  - keeps `val_loss_T` no worse than the current `confdepth_dropworst_gradconfmask` lead
  - improves `val_loss_conf_depth` versus the current `confdepth_dropworst_gradconfmask` lead
  - improves `val_loss_reg_depth` versus the current `confdepth_dropworst_gradconfmask` lead
  - introduces no new cache/path/raw-image failures
- If `conf_depth` and `reg_depth` do not both continue to fall, the candidate is dead on that same night.

## Cloud Gate

- `cloud_gate` stays `false`.
- No Modal launch is allowed from this state.
- Cloud may be reconsidered only after:
  - a stricter local gate passes against both baseline and the current lead
  - `consistency_check` remains clean
  - a fresh manual cloud pair decision is made

## Single Source Of Truth

- [consistency_check.md](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/consistency_check.md)
- [geometry_post_v9_next_training_question_20260325.md](/f:/vggt/vggt-main/docs/geometry_post_v9_next_training_question_20260325.md)
- [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json)
- [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json)
