# V270000000000000000 Auto-Evolved Photometric Route

Failed gate: neutral projection scoring showed controls and local projection crops were not beaten by the V190 true output.

Root cause:
- V190 initially copied V740 detail-verified predictions and only rescored them.
- Old predictions did not preserve per-point projection UV, forcing a source-order camera-binding approximation.
- Under config-neutral scoring, best controls remained close or better than true.

Architecture repair:
- Generate photometric predictions with explicit `projection_uv_518` per point.
- Keep true and controls at the same human/environment budget.
- Use real RGB/mask/edge sampling for true and ablated but same-budget control outputs.
- Re-run V150/V160/V210/V240/V250/V260 gates from current artifacts.

Data repair:
- Use original SMC RGB/mask assets exported in V130.
- Use V170 refined detail sources and per-case full-forward traces.

Exact run plan:
1. Rebuild `output/V190000000000000000_photometric_matrix` with projection-aware predictions.
2. Regenerate boards, local closeups, hard controls, viewer, report, bundles, cleanup, and final audits.

Final allowed states:
- V300000000000000000_PHOTOMETRIC_GEOMETRY_VISUAL_TRUTH_MENTOR_READY_NOT_PROMOTED
- V300000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule: no agent/subagent may be launched.

Failure snapshot:

```json
{
  "dual": {
    "created_at": "2026-05-27T17:53:29+00:00",
    "dual_gate_pass": false,
    "cases": {
      "current_v895_0021_03": {
        "true_score": 0.35199380373724215,
        "baseline_score": 0.3681955709882382,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.3795450489095129,
        "true_gt_baseline": false,
        "true_gt_best_control": false,
        "projection_pass": false,
        "full_scene_3d_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_3d_human_scene_board.png",
        "projection_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_projection_overlay_board.png"
      },
      "0021_03_frame001": {
        "true_score": 0.3504056165200229,
        "baseline_score": 0.3668632790543373,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.37829382822726937,
        "true_gt_baseline": false,
        "true_gt_best_control": false,
        "projection_pass": false,
        "full_scene_3d_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_3d_human_scene_board.png",
        "projection_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_projection_overlay_board.png"
      },
      "0012_11_frame001": {
        "true_score": 0.3222266955046653,
        "baseline_score": 0.3307792527948756,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.34174525582805715,
        "true_gt_baseline": false,
        "true_gt_best_control": false,
        "projection_pass": false,
        "full_scene_3d_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_3d_human_scene_board.png",
        "projection_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_projection_overlay_board.png"
      },
      "0013_01_frame001": {
        "true_score": 0.3213093388928057,
        "baseline_score": 0.3326603351822514,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.34794192993356937,
        "true_gt_baseline": false,
        "true_gt_best_control": false,
        "projection_pass": false,
        "full_scene_3d_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_3d_human_scene_board.png",
        "projection_board": "D:\\vggt\\vggt-canonical-surfel-adapter\\boards\\V140000000000000000_projection_overlay_board.png"
      }
    },
    "projection_not_replacement_for_3d": true
  },
  "fair": {
    "created_at": "2026-05-27T17:53:29+00:00",
    "config_neutral_scoring_pass": true,
    "no_detail_bonus_control_penalty_pass": true,
    "best_control_by_case": {
      "current_v895_0021_03": {
        "true_score": 0.35199380373724215,
        "best_control": "smpl_only_template_control",
        "best_control_score": 0.3851506344430633,
        "margin": -0.03315683070582115,
        "controls_separated": false
      },
      "0021_03_frame001": {
        "true_score": 0.3504056165200229,
        "best_control": "smpl_only_template_control",
        "best_control_score": 0.38368592860405265,
        "margin": -0.03328031208402976,
        "controls_separated": false
      },
      "0012_11_frame001": {
        "true_score": 0.3222266955046653,
        "best_control": "smpl_only_template_control",
        "best_control_score": 0.34765320086472057,
        "margin": -0.02542650536005525,
        "controls_separated": false
      },
      "0013_01_frame001": {
        "true_score": 0.3213093388928057,
        "best_control": "smpl_only_template_control",
        "best_control_score": 0.35314436579416353,
        "margin": -0.03183502690135781,
        "controls_separated": false
      }
    },
    "controls_separated_all_cases": false
  },
  "local": {
    "created_at": "2026-05-27T17:53:29+00:00",
    "local_closeup_real_pass": true,
    "local_detail_non_regression_pass": true,
    "visible_local_improvement_cases": 0,
    "visible_local_improvement_pass": false,
    "facial_detail_overclaim": false,
    "allowed_head_claim": "head/face contour and hair region only; no facial details claimed."
  },
  "controls": {
    "created_at": "2026-05-27T17:53:29+00:00",
    "hard_controls_v7_pass": false,
    "same_budget_same_projection_same_view": true,
    "source_label_auxiliary_only": true,
    "best_controls": {
      "current_v895_0021_03": {
        "true_score": 0.35199380373724215,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.3795450489095129,
        "margin": -0.027551245172270755
      },
      "0021_03_frame001": {
        "true_score": 0.3504056165200229,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.37829382822726937,
        "margin": -0.02788821170724648
      },
      "0012_11_frame001": {
        "true_score": 0.3222266955046653,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.34174525582805715,
        "margin": -0.01951856032339183
      },
      "0013_01_frame001": {
        "true_score": 0.3213093388928057,
        "best_control": "scaffold_only_no_vggt",
        "best_control_score": 0.34794192993356937,
        "margin": -0.026632591040763653
      }
    },
    "claim": "Photometric geometry route improves projected mask/RGB/edge consistency over the current baseline/control set under config-neutral scoring."
  },
  "environment": {
    "created_at": "2026-05-27T17:53:29+00:00",
    "environment_realism_v5_pass": true,
    "rows": [
      {
        "case": "current_v895_0021_03",
        "human_points": 60000,
        "environment_points": 24000,
        "human_ratio": 0.7142857142857143,
        "environment_from_prediction": true,
        "same_environment_budget": true,
        "human_ratio_55_75": true
      },
      {
        "case": "0021_03_frame001",
        "human_points": 60000,
        "environment_points": 24000,
        "human_ratio": 0.7142857142857143,
        "environment_from_prediction": true,
        "same_environment_budget": true,
        "human_ratio_55_75": true
      },
      {
        "case": "0012_11_frame001",
        "human_points": 60000,
        "environment_points": 24000,
        "human_ratio": 0.7142857142857143,
        "environment_from_prediction": true,
        "same_environment_budget": true,
        "human_ratio_55_75": true
      },
      {
        "case": "0013_01_frame001",
        "human_points": 60000,
        "environment_points": 24000,
        "human_ratio": 0.7142857142857143,
        "environment_from_prediction": true,
        "same_environment_budget": true,
        "human_ratio_55_75": true
      }
    ],
    "boundary": "Environment points come from the current model-owned prediction scene context and are verified by same budget and full-scene boards; no procedural floor/back plane is introduced in V120100."
  },
  "multisequence": {
    "created_at": "2026-05-27T17:53:30+00:00",
    "case_count": 4,
    "strong_visual_pass_cases": 4,
    "projection_pass_cases": 0,
    "local_detail_non_regression_cases": 4,
    "visible_local_improvement_cases": 0,
    "controls_separated_cases": 0,
    "paper_grade_generalization_claimed": false,
    "pass": false,
    "cases": {
      "current_v895_0021_03": {
        "3d_visual_pass": true,
        "projection_pass": false,
        "local_detail_non_regression": true,
        "visible_local_improvement": false,
        "controls_separated": false,
        "margin": -0.027551245172270755
      },
      "0021_03_frame001": {
        "3d_visual_pass": true,
        "projection_pass": false,
        "local_detail_non_regression": true,
        "visible_local_improvement": false,
        "controls_separated": false,
        "margin": -0.02788821170724648
      },
      "0012_11_frame001": {
        "3d_visual_pass": true,
        "projection_pass": false,
        "local_detail_non_regression": true,
        "visible_local_improvement": false,
        "controls_separated": false,
        "margin": -0.01951856032339183
      },
      "0013_01_frame001": {
        "3d_visual_pass": true,
        "projection_pass": false,
        "local_detail_non_regression": true,
        "visible_local_improvement": false,
        "controls_separated": false,
        "margin": -0.026632591040763653
      }
    }
  },
  "judge": {
    "created_at": "2026-05-27T17:53:30+00:00",
    "natural_main_view": true,
    "true_better_than_baseline": false,
    "hard_controls_separated": false,
    "projection_mask_rgb_edge_pass": false,
    "local_closeup_real": true,
    "facial_detail_overclaim": false,
    "environment_visible": true,
    "viewer_usable": true,
    "pass": false
  },
  "pass": false
}
```
