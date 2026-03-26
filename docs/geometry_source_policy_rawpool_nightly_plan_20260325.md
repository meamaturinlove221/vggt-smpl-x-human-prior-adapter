# Geometry Source-Policy Rawpool Nightly Plan (2026-03-25)

## Summary

- The current local training lead is:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)
- This lead keeps the same `nearest_ring + geom_plus_raw` input path and preserves the `depth + camera -> unproject -> render` geometry chain.
- Relative to the previous `confdepth_dropworst_only` lead, it improves `val_loss_conf_depth` and `val_loss_reg_depth` while keeping `val_loss_camera` and `val_loss_T` flat.
- It is still not baseline-clean on `conf_depth` and `reg_depth`, so cloud remains closed.
- The completed `conf_depth attribution v3` pass localized the remaining positive camera-level gap to `Camera_B1`, especially the foreground bottom band.
- The broader whole-frame `anchorb1confscale05` follow-up failed the tighter gate, and the narrower `anchorb1bottom20confscale05` follow-up later failed the stricter `100 train / 20 val` long gate after only a marginal short-gate win.
- A follow-up quality-signal audit defined the next bounded question, and the first `Camera_B1 + quality>=3.0 + foreground-bottom20 + scale0.5` candidate has now been tested and frozen after failing the tighter promotion rule.
- A postmortem on that candidate shows the hard `q>=3.0` gate was over-selective: it still excluded `4 / 10` bad `Camera_B1` audit rows and touched only `0.7476%` of conf-depth-supervised pixels on a 512-sample train slice.
- A follow-up quality-region boundary check shows the region was over-selective too: all `10 / 10` bad `Camera_B1` audit rows still have positive non-bottom `conf_depth` deltas, while `q>=3.0 + whole_foreground` covers `22.6975%` of conf-depth-supervised pixels on a fresh 512-sample train slice versus only `0.7983%` for `q>=3.0 + bottom20`.
- A further rule-shape sweep shows the next question is not another broader hard-threshold whole-foreground sibling either: `q>=3.0 + whole_foreground` still omits `4 / 10` bad audit rows, `q>=2.75 + whole_foreground` still omits `1 / 10`, and the best-supported next shape is a continuous `Camera_B1` whole-foreground quality-conditioned rule.
- The continuous loss knob now exists locally, but both tested whole-foreground continuous rules are frozen: the first exact `linear qmin -> qmax + whole_foreground + scale0.5` candidate failed the tighter gate, and the softer `scale0.75` fallback recommended by the local soft sweep failed too.
- The nightly default remains a hold state: `preflight -> consistency check -> status emit`. No training runs unless a fresh manual training question explicitly approves one genuinely new candidate.
- The overnight automation wrapper is now [run_zju_source_policy_rawpool_overnight_watch.py](/f:/vggt/vggt-main/scripts/run_zju_source_policy_rawpool_overnight_watch.py): it only repeats machine-readable `steady_hold`, refreshes its lock heartbeat while sleeping, and re-checks that the stable lead has not drifted, `cloud_gate=false`, consistency, Modal apps, and repo-scoped processes after each cycle.

## Current Lead

- Gate note:
  - [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md)
- Compare versus previous lead:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_lead_20260325_v1/summary.md)
- Compare versus baseline:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_baseline_20260325_v1/summary.md)
- Attribution evidence that motivated the lead:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3/summary.md)
  - On the audited 32-sample val slice, `conf_depth` was already anchor-only.
  - The only clearly positive camera-level delta was concentrated in anchor `Camera_B1`, with extra error in the foreground bottom band.
- Broader failed B1 follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_baseline_20260325_v1/summary.md)
  - The train-only whole-frame `Camera_B1` conf-target scale `0.5` candidate regressed both `loss_conf_depth` and `loss_reg_depth` versus the pre-B1 lead.
- Historical stage marker, not current lead:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml)
  - [geometry_source_policy_rawpool_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_local_gate_20260325.md)
- Failed long-gate successor:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_current_lead_longgate_100t_20v_overnight_20260326_v1_vs_previous_lead/summary.md)
  - The narrow B1-bottom20 successor lost the stricter long gate and no longer counts as current lead.
- Quality-signal audit:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_quality_signal_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - Within `Camera_B1`, `corr(quality_score, delta_conf_depth)=0.6555`.

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
- `anchorb1bottom20confscale05`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_current_lead_longgate_100t_20v_overnight_20260326_v1_vs_previous_lead/summary.md)
- `anchorb1qge3bottom20confscale05`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_vs_lead_20260326_v1/summary.md)
- `anchorb1qge3bottom20confscale05` postmortem
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_conditioned_candidate_postmortem_anchorb1qge3bottom20_20260326_v1/summary.md)
- `anchorb1qlinearwholefgconfscale05`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale05_vs_lead_20260326_v1/summary.md)
- `anchorb1qlinearwholefgconfscale075`
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale075_vs_lead_20260326_v1/summary.md)
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
  - `dominant_failure_shape = anchor_and_quality_conditioned`
  - `recommended_candidate_family = quality_conditioned_conf_target_normalization`
- First bounded follow-up from that recommendation:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_vs_lead_20260326_v1/summary.md)
  - `loss_conf_depth: 0.2288 -> 0.2287`
  - `loss_reg_depth: 0.1759 -> 0.1759`
- Candidate postmortem:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_conditioned_candidate_postmortem_anchorb1qge3bottom20_20260326_v1/summary.md)
  - bad-anchor rows below threshold: `4 / 10`
  - affected conf-depth-supervised pixel fraction: `0.7476%`
- Boundary check:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_region_boundary_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - positive non-bottom `conf_depth` rows on bad `Camera_B1`: `10 / 10`
  - `q>=3.0 + whole_foreground` conf-depth-supervised pixel fraction: `22.6975%`
  - `q>=3.0 + bottom20` conf-depth-supervised pixel fraction on the same fresh slice: `0.7983%`
- Rule-shape sweep:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_rule_shape_sweep_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - `q>=3.0 + whole_foreground` omitted bad audit rows: `4 / 10`
  - `q>=2.75 + whole_foreground` omitted bad audit rows: `1 / 10`
  - continuous `qmin -> qmax + whole_foreground` effective reduction fraction on the same fresh slice: `5.1155%`
- First exact continuous whole-foreground follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale05_vs_lead_20260326_v1/summary.md)
  - `loss_camera: 0.0219 -> 0.0218`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2289 -> 0.2776`
  - `loss_reg_depth: 0.1760 -> 0.1848`
- Softer continuous-rule sweep:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_continuous_soft_rule_sweep_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - recommended fallback rule: `linear_qmin_qmax_wholefg_scale075`
  - effective reduction fraction: `5.9433% -> 2.9716%`
  - bad `Camera_B1` audit rows still covered: `10 / 10`
- First softer continuous whole-foreground follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale075_vs_lead_20260326_v1/summary.md)
  - `loss_camera: 0.0219 -> 0.0218`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2289 -> 0.2483`
  - `loss_reg_depth: 0.1760 -> 0.1797`
- That recommendation replaced the exhausted fixed-scale B1 family, but the first quality-conditioned member did not clear promotion, its hard threshold proved over-selective, the follow-up boundary check showed `bottom20` is too narrow, the rule-shape sweep ruled out broader hard-threshold whole-foreground siblings, the first exact continuous whole-foreground candidate failed, and the first softer whole-foreground fallback failed as well. Default nightly therefore returns to `steady_hold` and still does not auto-train.
- The rawpool supervision-leak question remains settled local evidence. Re-run that audit only when a future candidate changes dataset or loss routing.

## Default Nightly Sequence

1. Preflight
  - Require `modal app list` to be empty.
  - Require no stale repo-scoped `python / powershell / modal` processes.
  - For guard-only `steady_hold`, preflight still checks Modal/apps/process cleanliness but no longer requires cloud-launch memory headroom, because this mode never launches cloud work.
  - Do not stop active Modal apps automatically.
2. Run consistency check
   - Refresh the local status snapshot.
   - Do not rerun attribution or quality audits by default.
   - Do not open a fixed-scale B1 sibling automatically after the exhausted B1-conditioned follow-up family.
3. Emit status only
   - Keep the current lead fixed.
   - Keep `cloud_gate=false`.

## Overnight Watch

- Entrypoint:
  - [run_zju_source_policy_rawpool_overnight_watch.py](/f:/vggt/vggt-main/scripts/run_zju_source_policy_rawpool_overnight_watch.py)
- Role:
  - Repeat `steady_hold` on a fixed interval during unattended overnight execution.
  - Keep a live lock file and refresh the sleep heartbeat so the watch can be distinguished from a hung process.
  - Keep both `active_watch.json` and `latest_session.json` updated so the current unattended session can be inspected without waiting for process exit.
  - Re-check during sleep heartbeats that the state still points at the stable `confdepth_dropworst_gradconfmask` lead, there are still zero active Modal apps, and there are still zero unexpected repo-scoped `python/powershell/pwsh/modal` processes, instead of waiting until the next full cycle.
  - Keep `latest_guard_snapshot.json` updated with the most recent successful guard check so the last clean Modal/process snapshot, `consistency_ok`, the current state lead, and the state cloud flags are visible at a fixed path.
  - Re-check after each cycle that `consistency_check` is clean, `current_lead_config` still matches the stable `confdepth_dropworst_gradconfmask` lead, `cloud_gate=false`, `launch_cloud_now=false`, there are zero active Modal apps, and there are zero unexpected repo-scoped `python/powershell/pwsh/modal` processes.
- Non-goals:
  - Do not auto-open `single_candidate_local_gate`.
  - Do not train.
  - Do not launch cloud.

## Single Candidate Local Gate

- This mode is allowed only when exactly one genuinely new local candidate is checked in.
- That candidate must be a non-redundant `conf_depth semantics` hypothesis.
- Under the current evidence, any future candidate still has to justify a quality-conditioned conf-target rule rather than another constant camera-only scale.
- It must also justify why it is broader or more continuous than the rejected hard `q>=3.0` gate and why it is not defaulting back to the already-too-narrow `bottom20` region.
- Under the newest evidence, it must also justify why it is continuous rather than another broader hard-threshold whole-foreground sibling.
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
- This mode is not automatic; it requires a fresh manual training question after both tested fixed-scale B1-conditioned follow-ups were exhausted, the stable lead was restored to `confdepth_dropworst_gradconfmask`, the first quality-conditioned follow-up was frozen, and both the first exact and first softer whole-foreground continuous candidates were frozen too.
- If that fresh question is approved, it must justify an even softer or reshaped rule beyond the already-frozen exact `linear qmin -> qmax + whole_foreground + scale0.5` and softer `scale0.75` variants.

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
