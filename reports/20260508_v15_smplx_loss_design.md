# V15 SMPL-X Native Prior Loss Design

Date: 2026-05-08

## Scope

This design stays inside the existing VGGT prior case/config/model/loss surface:

- Dense conditioning enters as `prior_maps`.
- Pooled SMPL-X conditioning enters as `prior_summary_tokens`.
- Supervision uses existing dataset keys: `prior_depths`, `prior_points`, `prior_normals` or `teacher_normals`, `teacher_mask`, and `smplx_*_anchor_mask`.
- No trainer, model, or dataset rewrite is required for the V15 path.

## Existing integration points

`DNA4K4DPseudoDataset` loads `prior_maps` and `prior_summary_tokens` from `inputs.npz`, passes them through optional prior pose noise, and returns them in the batch.

`Trainer._step()` calls:

```python
model(
    images=batch["images"],
    prior_maps=batch.get("prior_maps"),
    prior_summary_tokens=batch.get("prior_summary_tokens"),
)
```

`VGGT.forward()` forwards those tensors to `Aggregator.forward()`.

`HumanPriorAdapter.project_prior_maps()` projects `[B, S, C, H, W]` maps to patch tokens. `PriorFusionBlock` applies a gated residual update. V15 keeps `human_prior_gate_init: 0.0`, so initial model behavior is unchanged until training moves the gate.

## V15 case contract

`tools/v15_build_smplx_vggt_prior_case.py` builds a native case from an existing prepared case. The default `smplx_native` channel policy keeps:

- `silhouette`
- every dense channel named `smplx_*`
- every summary channel named `smplx_summary_*`

For current V2 SMPL-X cases this produces:

- `model.human_prior_channels: 29`
- `model.human_prior_summary_channels: 27`

The builder writes:

- `inputs.npz`: filtered `prior_maps`, filtered `prior_summary_tokens`, and `prior_mask`
- `targets.npz`: copied targets plus native `teacher_mask`, `smplx_bodyhand_anchor_mask`, `smplx_body_anchor_mask`, hand masks, and `smplx_native_visible_mask`
- `case_manifest.json`: updated prior channel metadata and `prior_input_meta.channel_groups`
- `v15_smplx_native_prior_summary.json`: shape/status/blocker summary

## Loss design

`training/loss_smplx_native_prior.py` adds `SMPLXNativePriorLoss`, a compatibility wrapper around the existing `MultitaskLoss`. It does not fork the loss math. It provides:

- optional default native-prior config
- optional required-key validation
- `loss_smplx_native_missing_key_count` for smoke/debug logging

`training/config/4k4d_smplx_native_prior.yaml` points Hydra to that wrapper and reuses existing human-prior terms:

- direct prior depth loss on `prior_depths` and `prior_mask`
- direct prior point loss on `prior_points` and `prior_mask`
- optional normal loss on `prior_normals` or `teacher_normals`
- depth/point consistency
- SMPL-X weak body/hand anchor through existing `smplx_*_anchor_mask` keys

## Blockers and guardrails

- The config expects V15-built case roots. It should not be launched before running the builder or overriding `smplx_native_case_roots`.
- If the template case lacks `prior_depths`, native depth loss becomes a dummy zero loss.
- If the template case lacks `prior_points`, native point loss becomes a dummy zero loss.
- If the template case lacks `prior_normals` and `teacher_normals`, native normal loss becomes a dummy zero loss.
- Hand masks are preserved when present. If missing, the builder creates conservative empty hand masks and uses the native visible mask as body/bodyhand fallback.
- This is not a formal training pass, cloud run, or strict gate. It is a local integration design plus compile/smoke target.
