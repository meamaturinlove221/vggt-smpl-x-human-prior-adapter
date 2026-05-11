# B-Fus3D15 Latent Grid Evidence Control Comparison

Status: `research_only_control_comparison_not_decoder_not_teacher_not_candidate`

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher_export = blocked
candidate_export = blocked
```

## Compared Runs

```text
real:
  output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/
shuffle:
  output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/
zero:
  output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/
```

## Readout

```text
real:
  supported_ratio = 0.502229
  boundary_like_ratio = 0.658779
  evidence_score_mean = 0.415409
  token_cosine_mean = 0.867266
  token_cosine_p10 = 0.758634
  token_cosine_p90 = 0.947897

shuffle:
  supported_ratio = 0.502229
  boundary_like_ratio = 0.658779
  evidence_score_mean = 0.409682
  token_cosine_mean = 0.765353
  token_cosine_p10 = 0.655371
  token_cosine_p90 = 0.873613

zero:
  supported_ratio = 0.502229
  boundary_like_ratio = 0.658779
  evidence_score_mean = 0.335843
  token_cosine_mean = null
```

## Decision

The raw mask/RGB visibility terms are identical across controls, as expected.
The real VGGT token evidence has noticeably higher cross-view token cosine than
the shuffled control, and the zero-token control removes the token term entirely.
This means B-Fus3D15 is not purely a bbox/mask artifact; there is a measurable
token-consistency signal available to a future learned 3D field backend.

This still does not establish a surface, teacher, candidate, or strict pass. The
only allowed next B-line action is one bounded B16 latent-field smoke with fixed
seed, fixed grid, fixed controls, negative controls, and Open3D precheck.

## Blocked Actions

```text
do_not_train_from_B15_alone
do_not_export_teacher_or_candidate
do_not_write_strict_registry
do_not_unblock_cloud
do_not_restart_B14_sparse_offset_loop
do_not_tune_hidden_steps_thresholds_after_B16
```
