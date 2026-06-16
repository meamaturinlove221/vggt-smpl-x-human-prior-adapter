# V523 Auto-Evolved Observation-Anchor Control/Part-Binding Repair Route

Repo: `D:\vggt\vggt-canonical-surfel-adapter`

Branch: `codex/volume-aware-3d-morphology`

No promotion. No registry. No V50/V50R2 modification. No active candidate replacement. No `git add .`.

## Trigger

V521 recovered the central visual direction: the human is readable again in full-scene RGB point-cloud panels with partial environment. However it remains fail-closed because the V42/VGGT visible anchor is itself very strong and the student does not visibly or metrically beat the competitive visible-anchor baseline.

## Required Repair

1. Separate source roles:
   - V50R2: visual floor / teacher / reference only.
   - V42 visible observation: input anchor, not a claimed VGGT baseline win.
   - VGGT baseline/control: use the actual prior student/VGGT baseline artifacts from V514/V517/V520 for mentor comparison.
2. Repair body-part binding:
   - head/hair, torso/clothing, arm/hand, leg/foot labels must be robust enough for local gates.
   - Do not use coarse y/x-only labels as final body-part evidence.
3. Preserve V521 visual readability:
   - Do not return to V519/V520 free-template smear.
   - Any new residual/completion must be identity-safe when it would degrade the visible anchor.
4. Re-run gates:
   - V509 full-scene insertion.
   - V510 local fidelity.
   - V511 anti-2D.
   - V512 manual mentor gate.

## Forbidden Success Claims

Do not claim success for raw visible-anchor recovery alone, teacher-only/crop-only output, metric-only improvement, projection-only evidence, or route-created-only status.

## Allowed Next Statuses

- `V523_OBSERVATION_ANCHOR_CONTROL_PART_BINDING_REPAIR_READY_FOR_V512_NOT_PROMOTED`
- `V523_OBSERVATION_ANCHOR_REPAIR_FAIL_CLOSED_CONTINUE_MODEL_REPAIR_NOT_PROMOTED`
