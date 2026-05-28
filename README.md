# VGGT-SMPL-X Human Prior Adapter

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter architecture" width="100%" />
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#why-this-repo-exists">Why this repo</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#parallel-engineering-record">Parallel engineering</a> ·
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

## Parallel engineering record

This route was later reviewed inside a larger parallel engineering loop. The task moved from “connecting SMPL-X to VGGT” toward a more complete sparse-view human geometry recovery workflow. Several routes were made runnable, while the upper bound of 6-view head / face point-cloud quality also became clearer.

The main chain can be summarized in four layers:

1. **Pose-aligned SMPL-X driver**: reads pose / shape / expression / translation / scale, and places the parametric body into the current pose and scene coordinate system.
2. **Dense prior maps**: projects the posed mesh into real cameras and generates view-aligned dense priors, including depth, camera/world points, normals, visibility, canonical coordinates, and body-part features.
3. **Input-side / layer-wise fusion**: RGB keeps real appearance and scene context; prior maps provide pose-aligned geometric positions; masks restrict where the human prior should take effect. The prior is not only concatenated once at the input side, but also participates during multi-layer feature evolution.
4. **Output-side supervision**: the training side supports depth / point / normal / point-normal geometric supervision, with ROI and boundary weighting.

The role of SMPL / SMPL-X here is not to serve as the final result. It is a pose-aligned geometry prior that provides coarse body position, depth, surface direction, and region constraints. The real question is still whether the downstream model can generate a clearer, more continuous, and more stable 3D human point cloud under sparse-view conditions.

This stage also made one lesson clear: adding more losses or increasing ROI point count does not automatically mean better geometry. If the teacher is not continuous, aligned, and complete enough on visible surfaces, the result can easily become a pseudo-positive case where point count increases but Open3D evaluation becomes worse.

---

## Checked routes and failure boundaries

The parallel experiments checked several directions:

- projected targetpatch / summary-token patch;
- point-normal / humancrop finetuning from the same checkpoint;
- TeacherGeom / ROI combo;
- confidence-collapse pseudo-positive cases, where face ROI point count increases but confidence thresholding or Open3D evaluation shows worse geometry;
- external teacher routes such as NormalBae, Sapiens, DepthAnything, and DepthPro.

The conclusion is fairly clear: the current bottleneck is not a lack of scripts. It is the lack of a high-quality, continuous, aligned head / face geometry teacher, or the lack of a local geometry optimization method that can directly improve sparse-view target-view surfaces.

The next route therefore has to move toward harder components:

- real 3D learned residual;
- multi-view detail supervision;
- baseline high-confidence detail preservation;
- SMPL feature-conditioned local geometry branch;
- human-main full-scene visual gate.

---

## Current result snapshot

<p align="center">
  <img src="docs/figures/parallel_engineering_result_snapshot.svg" alt="parallel engineering result snapshot" width="100%" />
</p>

<p align="center"><sub>6-view face/head ROI re-audit: local facial structure is visible, but continuity and stability are still not enough.</sub></p>

<p align="center">
  <img src="docs/figures/external_reference_control.svg" alt="external reference control" width="100%" />
</p>

<p align="center"><sub>External geometry reference routes are recorded only for camera, mask, and teacher-quality audits. They are not student outputs.</sub></p>

The safe conclusion at this point is that the 6-view setting has produced promising local facial results, but flaws remain. Under the same protocol, the 6-view face / head point cloud has not yet reached the final requirement of being clear, continuous, and stable enough.

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

The main evidence must come from a VGGT student route and should be shown as a full-scene RGB point cloud where the human is the subject while some environment context remains visible.

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

---

## Added figures

```text
docs/figures/vggt_smplx_human_prior_adapter_architecture.svg
docs/figures/parallel_engineering_result_snapshot.svg
docs/figures/external_reference_control.svg
```
