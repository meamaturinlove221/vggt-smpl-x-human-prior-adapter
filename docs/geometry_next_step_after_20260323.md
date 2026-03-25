# Geometry Next Step After 2026-03-23

## What Finished

- The local `6src` hard/control source-policy repair loop is now fully closed:
  - [geometry_6src_hardcontrol_local_completion_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_hardcontrol_local_completion_20260324.md)
  - key readout:
    - the expanded `B15` search produced `14` guard-pass `depth_unproject` candidates
    - the standardized `B1+B4+B12+B15` sweep now flips the full hard/control set to `depth_unproject = 5 / 5`
    - `uniform -> v5 hybrid` moves average depth-point MAE from `+0.001521 -> -0.001548`
    - `uniform -> v5 hybrid` moves average depth-point coverage from `+0.010582 -> +0.065431`
- The repaired `6src hybrid v5` batch has now also passed strict same-source legacy-native validation:
  - [geometry_6src_hybrid_v5_legacy_backfill_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_v5_legacy_backfill_20260324.md)
  - key readout:
    - matched-source legacy backfill now gives `depth better than point = 5 / 5`
    - average legacy gap improves from `point = 0.007913` to `depth = 0.006365`
    - the local backfill wrapper now has a controlled retry for old-script-only `sim3 rmse_after` failures
- The repaired `6src hybrid v5` batch has now also passed a full all-target local rollout-safety sweep:
  - [geometry_6src_hybrid_v5_transfer_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_v5_transfer_gate_20260324.md)
  - key readout:
    - all `23` `6src_hist / frame 1080` targets were re-swept locally
    - `depth_unproject` decisions improve from `11 / 23` to `15 / 23`
    - `point_map` decisions drop from `6 / 23` to `2 / 23`
    - average depth-point geometry gain moves from `-0.000117` to `+0.000550`
    - average depth-point coverage gain moves from `+0.062199` to `+0.074123`
    - only the intended `B1 / B4 / B12 / B15` cases change, and all four change in the favorable direction
    - there are no collateral regressions on the remaining `19` cameras
- A broader local multi-frame follow-up is now also complete:
  - [geometry_6src_hybrid_v5_multiframe_local_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_v5_multiframe_local_gate_20260324.md)
  - key readout:
    - [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py) now accepts override manifests with `frame_id = 0`
    - the repaired `v5` manifest was re-run locally on `B1 / B4 / B12 / B15` across `frame 0 / 600 / 1080 / 1170`
    - decision counts move from `depth = 0, point = 14, tie = 2` to `depth = 7, point = 1, tie = 8`
    - average depth-point geometry gain moves from `-0.002419` to `+0.000355`
    - average depth-point coverage gain moves from `-0.024763` to `+0.055556`
    - there are still no regressions to `point_map` from an existing `depth_unproject` win
    - but broader multi-frame transfer is not fully closed yet, especially at `frame 600`
    - a bounded `B12 @ frame 600` local family search ran `64` variants and found `0` `depth` / `0` `tie` candidates
- The broader local multi-frame transfer question is now also closed under a
  frame-aware `v6` manifest:
  - [geometry_6src_frameaware_v6_multiframe_transfer_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_frameaware_v6_multiframe_transfer_gate_20260324.md)
  - key readout:
    - `B12 @ frame 600` was repaired with a frame-aware nearest-family override
    - targeted `B1 / B4 / B12 / B15 x frame 0 / 600 / 1080 / 1170` moves from
      `depth = 0, point = 14, tie = 2` to `depth = 8, point = 0, tie = 8`
    - `v5 -> v6` adds one more `point -> depth` flip and no regressions
    - all-target `92`-case multiframe rollout safety moves from
      `depth = 34, point = 28, tie = 30` to `depth = 42, point = 14, tie = 36`
    - average depth-point geometry gain moves from `-0.000530` to `-0.000018`
    - average depth-point coverage gain moves from `+0.051624` to `+0.065943`
    - only the intended `16` override cases change, and all changes are
      favorable or neutral
    - there are no non-target collateral regressions
- The `zju_min_depth_conf=p60` reliability pair completed and was pulled locally:
  - [summary.md](/f:/vggt/vggt-main/output/zju_depth_target_reliability_pair_cloud/20260323_170500_zju_depth_target_reliability_pair_4000step_p60_v2/summary.md)
  - result: the filtered-target candidate did **not** beat the baseline on validation objective
- The hard-case legacy-gap batch has now also been fully finalized locally:
  - [summary.md](/f:/vggt/vggt-main/output/legacy_gap_batch_20260323_hardcase_v1/summary.md)
  - [summary.json](/f:/vggt/vggt-main/output/legacy_gap_batch_20260323_hardcase_v1/summary.json)
- A new mentor-aligned reliable-region `unproject_geometry` branch was implemented and locally gated:
  - [geometry_reliable_region_unproject_local_gate_20260323.md](/f:/vggt/vggt-main/docs/geometry_reliable_region_unproject_local_gate_20260323.md)
  - result:
    - implementation and smoke both passed
    - short local A/B objective did **not** beat the baseline
    - decision: keep local only, do not move this branch to cloud yet
- A narrower bottom-only follow-up for the same auxiliary branch was also gated locally:
  - [geometry_bottom_only_unproject_local_gate_20260323.md](/f:/vggt/vggt-main/docs/geometry_bottom_only_unproject_local_gate_20260323.md)
  - result:
    - implementation passed
    - short local A/B objective again did **not** beat the baseline
    - decision: keep cloud off for this branch as well
- The `6src` hard cameras were re-materialized locally with full compare outputs and region-level aggregation:
  - [geometry_6src_hardcase_region_status_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_hardcase_region_status_20260323.md)
  - key readout:
    - `depth_unproject` still wins `fg_human` MAE on `3 / 4`
    - but loses `bg_far` on `4 / 4`
    - and loses `bg_bottom_band` on `4 / 4`
    - so the current sparse hard-camera failure is mainly bottom/background driven, not mainly human-region driven
- The matched-source `12src_uniform` cases were also re-materialized locally and summarized with the same region-level tooling:
  - [geometry_sparse_region_contrast_20260323.md](/f:/vggt/vggt-main/docs/geometry_sparse_region_contrast_20260323.md)
  - key readout:
    - `12src_uniform` moves the full-frame result to `depth_unproject` win-or-tie
    - `fg_human` becomes strongly depth-favored
    - `bg_far` no longer shows the universal collapse seen in `6src` hard cameras
    - `bg_bottom_band` weakness remains, but is much smaller than in `6src` hard cameras
- A new `6src_uniform` hard/control current-current follow-up was run locally:
  - [geometry_6src_uniform_policy_followup_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_uniform_policy_followup_20260323.md)
  - key readout:
    - full-frame average geometry gap improves from `+0.002257 -> +0.001521`
    - full-frame average coverage gap improves from `-0.042405 -> +0.010582`
    - `bg_far` improves strongly
    - `bg_bottom_band` improves only slightly and remains the strongest unresolved region
    - `fg_human` / `fg_edge` are not a free win under `uniform_ring`
- A new region-level policy contrast pass is now complete for both `12src` and `6src`:
  - [geometry_region_policy_contrast_20260323.md](/f:/vggt/vggt-main/docs/geometry_region_policy_contrast_20260323.md)
  - key readout:
    - `12src_uniform` improves not just full-frame MAE but also `fg_edge`, `bg_far`, and `bg_bottom_band`
    - `6src_uniform` still mainly helps `bg_far`
    - `6src` `bg_bottom_band` remains the sharpest unresolved sparse failure
- A new `6src` leave-one-source-out local batch is now complete:
  - [geometry_6src_source_loo_status_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_source_loo_status_20260323.md)
  - key readout:
    - `bg_bottom_band` is still source-selection-sensitive under `6src`
    - but helpful drops are case-specific
    - and naive pruning often hurts `fg_human`
- A new `6src` custom source-set swap probe is now complete:
  - [geometry_6src_source_swap_probe_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_source_swap_probe_20260323.md)
  - key readout:
    - the first custom-probe readout exposed a local `500000` vs `750000`
      render-setting inconsistency
    - after standardization, `Camera_B4` no longer survives as a valid rescue
    - `Camera_B12` still survives as a narrow two-source hybrid rescue
    - the corrected batch-level source of truth is now:
      [geometry_6src_hybrid_policy_standardized_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_policy_standardized_20260323.md)
    - so the remaining `6src` bottom-band problem is still source-policy-shaped,
      but only one local override is currently validated
- A new local runner repair + stage-2 probe pass is now complete:
  - [geometry_6src_stage2_local_probes_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_stage2_local_probes_20260323.md)
  - key readout:
    - the local `custom probe / leave-one-out / view sweep` runners now default
      to repo-local `.venv5080`
    - `leave-one-out` is now standardized to `render_max_points = 750000`
    - `B1` and `B15` were re-probed locally under the repaired runner chain
    - that stage itself still ended with no new safe override
    - but a later expanded local search and standardized rerun has now promoted
      `B1`, `B4`, and `B15` as well
    - so the current local hard/control rescue set is no longer `B12`-only
- A new first-pass local residual probe is now also complete for the strongest
  post-`v6` unresolved target:
  - [geometry_6src_b2_policy_probe_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_policy_probe_20260324.md)
  - key readout:
    - `Camera_B2` remains `point_map = 4 / 4` under the post-`v6`
      `uniform_ring` baseline
    - `nearest_ring` improves two frames from `point` to `tie`, but gives
      `0` depth wins and worsens average geometry gain
    - `rotate_template_offsets` gives `0` favorable decision flips
    - so `B2` is not a cheap generic-policy rescue and should move to a
      targeted custom source-set search if this residual axis stays active
- A new `B2` custom-search upgrade gate is now also complete:
  - [geometry_6src_b2_v7_upgrade_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_v7_upgrade_gate_20260324.md)
  - key readout:
    - a `358`-variant `frame 600` custom search found `60` guard-pass
      candidates and recovered `s1_013` as the best balanced local fix
    - direct four-frame transfer of `s1_013` gives
      `depth = 3, point = 0, tie = 1` against the old `B2 4 / 4 point`
      baseline
    - full four-frame `B2` rollout is not safe enough yet, because
      `frame 1080` only reaches `tie` and `frame 1170` misses the local
      `fg_human` guard
    - but a partial `v7` promotion for `B2 @ frame 0 / 600` is now accepted
      locally
    - all-target `v6 -> v7` moves `depth = 42 -> 44` and `point = 14 -> 12`
      with no non-target regressions
- A new `B2` completion gate is now also complete:
  - [geometry_6src_b2_v8_completion_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_v8_completion_gate_20260324.md)
  - key readout:
    - `frame 1080` custom search promoted `s1_029_s1013_family`
    - `frame 1170` custom search promoted `s2_130_s1013_family`
    - `v8` all-target sweep now runs cleanly with `92` rows and `0` failures
    - all-target `v7 -> v8` moves `depth = 44 -> 46` and `point = 12 -> 10`
    - only `Camera_B2 @ frame 1080 / 1170` change, both from `point_map` to
      `depth_unproject`
    - there are still no non-target collateral regressions
- A new post-`v8` frame-1170 residual cleanup is now also complete:
  - [geometry_6src_b13_b23_v9_upgrade_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b13_b23_v9_upgrade_gate_20260324.md)
  - key readout:
    - the strongest `frame 1170` post-`v8` residuals, `Camera_B13` and
      `Camera_B23`, were searched locally under narrow `uniform + nearest`
      one-swap families
    - neither target produced a clean all-frame transfer family
    - but both produced a safe target-frame repair
    - those two repairs were promoted into a narrow frame-aware `v9` manifest
    - all-target `v8 -> v9` moves `depth = 46 -> 48` and `point = 10 -> 8`
    - only `Camera_B13 @ frame 1170` and `Camera_B23 @ frame 1170` change
    - there are still no non-target collateral regressions
- A new post-`v9` stop gate is now also complete:
  - [geometry_post_v9_b11_stop_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_post_v9_b11_stop_gate_20260324.md)
  - key readout:
    - exactly one more post-`v9` residual cleanup round was run on
      `Camera_B11`
    - `swap Camera_B6 -> Camera_B17` is the best local `B11` family
    - it repairs `frame 1170` and preserves the existing `frame 1080` depth win
    - but it still does not repair `frame 600`, and its transfer is already too
      narrow to justify another manifest promotion
    - decision: keep `v9` frozen as the main local manifest and stop adding
      manual residual overrides for now

## What We Learned

- Global cached-target filtering is not the next mainline:
  - `baseline_raw_target val loss_objective = 0.0559`
  - `baseline_min_depth_conf val loss_objective = 0.0577`
  - decision: do **not** carry this scalar `p60` threshold into `unproject_geometry`
- The geometry branch itself is still viable overall, but not uniformly:
  - hard-case batch: `depth_better_than_point = 5 / 8`
  - average legacy gap:
    - `depth_unproject = 0.008434`
    - `point_map = 0.009209`
  - so `depth_unproject` is slightly better overall against legacy native, but the margin is small and not universal
- The pattern is not "all cameras are fixed by depth+camera":
  - `12src_nested / Camera_B5` control: `depth_unproject` wins
  - under the pre-repair `6src_hist` controls, `Camera_B5` still slightly favored `point_map`
  - under the same pre-repair readout, `Camera_B15`, `Camera_B17` also favored `point_map`
  - `12src_nested / Camera_B3`, `Camera_B8`: `depth_unproject` leads
- This keeps pointing back to sparse-view geometry/source-policy sensitivity rather than image-side ghost logic or a single global depth-target threshold.
- Source-policy sensitivity is now more specifically localized:
  - `bg_far` responds strongly to source-policy changes
  - `12src` `bg_bottom_band` also improves materially under `uniform_ring`
  - but `6src` `bg_bottom_band` remains much more stubborn than the rest of the image
  - even inside `6src`, leave-one-out analysis shows the bottom-band gap can still move under source removal, but not in a uniform or tradeoff-free way
- The latest source-swap probe further sharpens that:
  - the early `B12`-only and `B12+B4` batches were useful intermediate steps,
    but no longer reflect the latest local state
  - the expanded local search eventually recovered a standardized `B15`
    override as well
  - the current corrected hard/control source-policy batch is now:
    - `B1 + B4 + B12 + B15`
  - and that batch moves the full hard/control set to `depth_unproject = 5 / 5`
  - regionally, `bg_bottom_band` is still the weakest area
  - but it is no longer strong enough to flip any of the repaired full-frame
    decisions back to `point_map`
  - and the repaired batch now also survives strict same-source legacy-native
    backfill
  - and a later all-target rollout-safety sweep shows the repaired manifest
    only changes the intended `B1 / B4 / B12 / B15` targets, with no collateral
    regressions on the remaining `19`
  - the broader multi-frame follow-up first showed that plain `v5` was only a
    partial transfer beyond `frame 1080`
  - but the later frame-aware `v6` follow-up closes that repaired
    `B1 / B4 / B12 / B15` multi-frame slice as well
  - the first `B2` generic-policy probe showed `B2` was not a cheap
    `nearest/rotate` rescue
  - the later `B2` custom-search gate first contracted that residual by
    accepting a safe partial `v7` upgrade on `frame 0 / 600`
  - and the later `v8` completion gate closes the remaining
    `B2 @ frame 1080 / 1170` slice as well
  - a later post-`v8` follow-up then searched `Camera_B13` and `Camera_B23`
    around `frame 1170`
  - broad four-frame transfer still did not hold cleanly for either target
  - but narrow frame-aware `frame 1170` repairs for both targets were promoted
    safely into `v9`
  - so the strongest post-`v9` local residual cluster is now smaller:
    - `frame 0`: `B13 / B23`
    - `frame 600`: `B11 / B13`
    - `frame 1080`: `B23`
    - `frame 1170`: `B11 / B16 / B7`
  - a final extra local residual round on `B11` did find another usable
    target-frame family
  - but that family still did not cleanly resolve the repeated `frame 600`
    `B11` slice
  - so the local source-policy line has now crossed into diminishing returns:
    it can still produce narrower patches, but not obviously cleaner reusable
    rules

## Immediate Decision

- Stop this branch:
  - no more `zju_min_depth_conf` threshold follow-up for now
  - no propagation of this global target filter into the next `unproject_geometry` run
- Hold this branch locally:
  - reliable-region `unproject_geometry` is now implemented
  - but its first short local gate did not beat baseline
  - the narrower bottom-only follow-up also did not beat baseline
  - so this auxiliary-region family is not yet worth a Modal launch
- Keep this branch:
  - `depth + camera -> unproject -> render` remains the preferred geometry direction to test
- Shift the next experiment target to:
  - keep the next question local
  - freeze `v9` as the current local frame-aware main manifest
  - do not open another manual residual override round yet
  - specifically, treat the current post-`v9` residual cluster as the handoff
    point into a new explicit training/ablation question
  - while keeping `12src_uniform` as the best current sparse geometry-friendly
    reference
  - and treating the completed local `6src hybrid v5/v6` evidence as closure of
    the original sparse hard/control repair loop and its direct multi-frame
    transfer gate, with the later `v7/v8/v9` updates extending that closure to
    the later repaired `B2` subset and the additional `frame 1170`
    `B13 / B23` repairs as well

## Next Local Step

- The next step should still be chosen deliberately, not opened on cloud by
  inertia.
- It should not be another target-threshold pair.
- It should not be either of the current auxiliary-region
  `unproject_geometry` variants.
- The old local blocker for the original `frame 1080` repair scope is now
  cleared:
  - the `6src` hard/control failure set is no longer unresolved locally
  - the repaired `v5` manifest also passes the all-target rollout-safety gate
    for `frame 1080`
- But the next local residual blocker is still active:
  - the old `B12 @ frame 600` blocker is now closed under the frame-aware `v6`
    manifest
  - and the old `B2 4 / 4 point` residual is now fully closed by `v7/v8`
  - and the old `frame 1170 B13 / B23` pair is now also locally closed by `v9`
  - but the post-`v9` residual cluster remains
  - and one more `B11` round showed that the next step is no longer a cleaner
    manual override family
- But there still should not be any automatic new Modal launch:
  - the next cloud run should only start once the next experiment target is
    chosen explicitly
- The best next minimal step is now to choose between:
  - turning the current residual evidence into a new explicit local
    training/ablation hypothesis
  - or, if that question is still not clear enough, doing read-only diagnosis on
    the repeated remaining residuals `B11 / B13 / B23` without adding new
    override families
  - and only after that, choosing whether any fresh cloud run is justified
- Operationally:
  - keep the future cloud throughput settings on hand
  - keep Modal clean until that next question is chosen

## Engineering Fixes That Are Now In Place

- [modal_zju_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_zju_geometry_minimal_finetune.py)
  - live mirroring to `driver_live.log`
  - periodic output-volume commit during training
  - higher-throughput defaults for future runs
- [monitor_modal_depth_target_pair.py](/f:/vggt/vggt-main/scripts/monitor_modal_depth_target_pair.py)
  - now monitors `driver_live.log` first
- [run_legacy_gap_batch.py](/f:/vggt/vggt-main/scripts/run_legacy_gap_batch.py)
  - now resumes from already-downloaded legacy cases
  - now resolves copied legacy PNGs from the local run directory even when `report.json` still points to `/mnt/out/...`
- [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py)
  - now accepts override manifests with valid `frame_id = 0` cases
  - now also accepts override manifests saved with a UTF-8 BOM on Windows
  - now also resolves a real local checkpoint path before dispatching
    per-case compares, avoiding PowerShell path-mangling on non-ASCII
    checkpoint locations

## Practical Readout

- The mentor concern about unreliable ZJU cached depth targets was worth testing.
- The diagnosis remains useful, but the simplest scalar treatment failed.
- The first two mentor-aligned auxiliary-region training treatments also failed the local gate.
- The project should now move back to the more stable mainline:
  - keep the geometry chain
  - stop global target-threshold tuning
  - keep the new auxiliary-region code local-only for now
  - treat the original `6src hybrid v5/v6` repair loop and its direct
    multi-frame transfer gate as closed
  - treat the later `v7/v8/v9` updates as safe local extensions, not as a
    reason to open cloud by inertia
  - keep the next residual source-policy question local for now
  - `frame 1080`, the old `B12 @ frame 600` line, the old `B2` slice, and the
    old `frame 1170 B13 / B23` pair no longer block cloud on their own
  - but the extra `B11` round also says the project should stop treating manual
    override collection as the default next move
  - and only reopen cloud once there is a deliberately chosen new
    training/ablation question, or the remaining residual cleanup has been
    judged complete enough to stop being the active local focus
