# VGGT-SMPL-X Human Prior Adapter

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter architecture" width="100%" />
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#why-this-repo-exists">Why this repo</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#evidence-standard">Evidence standard</a> ·
  <a href="#current-status">Status</a>
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_research_route-2563eb" />
  <img alt="backbone" src="https://img.shields.io/badge/backbone-VGGT-7c3aed" />
  <img alt="prior" src="https://img.shields.io/badge/prior-SMPL--X-d97706" />
</p>

A research adapter for adding **SMPL-X structural priors** to **VGGT-style multi-view geometry**.

The idea is to keep VGGT as the geometry owner, then give it a cleaner route for using aligned human-prior evidence during training and evaluation.

---

## Why this repo exists

VGGT already estimates cameras, depth, point maps, and tracks from multi-view RGB. For human-scene reconstruction, the difficult part is often local structure and part consistency. A result can look acceptable by metric while still failing in the final 3D view.

This repo explores a model-side route for injecting SMPL-X prior evidence into VGGT without replacing the VGGT prediction path.

The route focuses on:

- real-camera alignment;
- prior input construction;
- prior geometry supervision;
- baseline and control comparison;
- full-scene point-cloud evidence.

---

## What this repo adds

| Part | Role |
| --- | --- |
| SMPL-X prior source | Builds a posed structural reference from available parameters. |
| View-aligned prior rendering | Projects prior evidence into the same camera views used by VGGT. |
| `prior_maps` style input | Sends image-space human cues into the model route. |
| `prior_depths` / `prior_points` / `prior_mask` style targets | Adds geometry supervision aligned with real cameras. |
| HumanPriorAdapter route | Adds a light prior path while keeping the main VGGT backbone intact. |
| Evidence gate | Keeps metrics, 3D visual results, and advisor-facing evidence separate. |

---

## Pipeline

```text
Multi-view RGB + real cameras
        │
        ├── VGGT backbone
        │        └── cameras / depth / point maps / tracks
        │
SMPL-X parameters
        │
        └── prior rendering under real cameras
                 ├── prior_maps
                 ├── prior_depths
                 ├── prior_points
                 └── prior_mask
                          │
                          └── HumanPriorAdapter
                                   │
                                   └── VGGT-owned human-aware scene geometry
                                            │
                                            └── full-scene RGB point-cloud evidence
```

---

## Project split

| Repository | Position |
| --- | --- |
| `VGGT-SMPL-X-Human-Prior-Adapter` | Model-side prior injection and supervision route. |
| `VGGT-ZJU-MoCap-Adapter` | Dataset adaptation, camera alignment, and trusted cases. |
| `vggt_for_4k_4d` | 4K4D-style cases, baseline/control comparison, and visual evidence. |
| `vggt-human-prior-builder` | Release-safe public preprocessing recipe and schema boundary. |

---

## Evidence standard

| Level | Meaning | Final claim |
| --- | --- | --- |
| Metric pass | Losses or geometry metrics improve. | Insufficient alone. |
| Visual pass | The 3D output is visibly more structured. | Useful but still needs comparison. |
| Advisor pass | The gain is visible in a human-main full-scene RGB point cloud under the same bounds and view. | Main target. |

Debug views are useful for locating problems. Isolated scatter views, projection overlays, SMPL-X-only views, and teacher references should stay as supporting evidence.

---

## What to look at first

1. Start from the architecture figure at `docs/figures/vggt_smplx_human_prior_adapter_architecture.svg`.
2. Read the route summary above.
3. Use `README_CN.md` for the Chinese project story and result-display links.
4. Compare against vanilla VGGT and control variants before making result claims.

---

## Current status

This repo is an active research route. It defines a clean place for SMPL-X prior construction, prior input, prior supervision, and later baseline/control comparison.

A checkpoint should only be promoted after the exported evidence passes the full-scene visual gate.
