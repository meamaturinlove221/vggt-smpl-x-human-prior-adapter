# VGGT-SMPL-X Human Prior Adapter

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter architecture" width="100%" />
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#one-sentence-summary">Summary</a> ·
  <a href="#relation-to-vggt">Relation to VGGT</a> ·
  <a href="#what-this-repository-adds">What this adds</a> ·
  <a href="#engineering-record">Engineering record</a> ·
  <a href="#visual-record">Visual record</a> ·
  <a href="#repository-role">Repository role</a>
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_research_route-2563eb" />
  <img alt="baseline" src="https://img.shields.io/badge/baseline-VGGT-7c3aed" />
  <img alt="prior" src="https://img.shields.io/badge/prior-SMPL--X-d97706" />
  <img alt="evidence" src="https://img.shields.io/badge/evidence-full--scene_point_cloud-0f766e" />
</p>

## One-sentence summary

This repository records the model-side route for adding **SMPL-X human structural priors** to a **VGGT-style feed-forward visual geometry pipeline**. The goal is to keep VGGT as the geometry owner while providing human-region structural cues, prior geometry targets, and evidence-aware evaluation for sparse-view human-scene reconstruction.

## Relation to VGGT

The original VGGT baseline predicts visual geometry directly from one or more RGB views: cameras, depth maps, point maps, and tracks. That makes it a strong general scene geometry backbone. Human-centric reconstruction, however, adds a different difficulty: the output must preserve recognizable human topology, including head, torso, arms, legs, hands, and feet, while still staying in the surrounding scene.

This repository is a delta on top of the VGGT baseline:

| VGGT baseline | This repository |
| --- | --- |
| RGB-driven feed-forward geometry | RGB geometry plus SMPL-X human structural prior |
| General camera / depth / point / track prediction | Human-region prior maps, prior depth, prior points, prior normals, and prior masks |
| Scene-level geometry output | Human-aware scene geometry under the same camera and scene context |
| Standard metric / visual inspection | Explicit separation of metrics, visual diagnostics, and full-scene point-cloud evidence |
| Generic point-map prediction | Human topology constraints through pose-aligned body priors |

The key design choice is that SMPL-X is not used as a final replacement. It is a structured prior and supervision source. The final output still needs to come from the VGGT student route.

## What this repository adds

### 1. Pose-aligned SMPL-X prior construction

The route starts with SMPL-X pose / shape / expression / translation / scale parameters and places the body prior into the current pose and scene coordinate system. This creates a human structural reference that can be projected into real camera views.

### 2. View-aligned prior rendering

The posed SMPL-X mesh is rendered under the same cameras used by the RGB inputs. The route can provide:

- `prior_maps`: image-space human cues for the model;
- `prior_depths`: human-region depth references;
- `prior_points`: camera/world point references;
- `prior_normals`: local surface direction references;
- `prior_mask`: valid human-prior supervision regions.

These signals are useful only when they remain aligned with the RGB, mask, depth, and camera chain.

### 3. HumanPriorAdapter / supervision path

The repository keeps the main VGGT route intact and adds a light human-prior path. The prior branch can be used as input-side guidance, intermediate feature conditioning, or output-side supervision. This keeps the project closer to a VGGT extension rather than a separate SMPL-X regressor.

### 4. Baseline and control awareness

The project is organized around comparison rather than isolated results. The expected comparison set includes vanilla VGGT, no-prior runs, prior-conditioned runs, teacher/reference routes, and visual controls.

## Engineering record

The route was later reviewed inside a larger sparse-view human geometry loop. Several engineering directions were made runnable or auditable:

- projected target patch / summary-token patch routes;
- point-normal and human-crop fine-tuning attempts from the same checkpoint;
- TeacherGeom / ROI combinations;
- confidence-collapse pseudo-positive cases where ROI point count increased but Open3D or confidence-based inspection got worse;
- external reference routes such as NormalBae, Sapiens, DepthAnything, and DepthPro;
- 6-view face/head ROI re-audits;
- full-scene human-main visual evidence checks.

The main lesson is that adding more loss terms or more ROI points is not automatically a geometry improvement. If the teacher is not continuous, aligned, and complete on visible surfaces, the student can inherit noisy or incomplete structure.

## Visual record

<p align="center">
  <img src="docs/figures/parallel_engineering_result_snapshot.svg" alt="parallel engineering result snapshot" width="100%" />
</p>

<p align="center"><sub>6-view face/head ROI re-audit. Local facial structure is visible, while continuity and stability still need stronger geometry support.</sub></p>

<p align="center">
  <img src="docs/figures/external_reference_control.svg" alt="external reference control" width="100%" />
</p>

<p align="center"><sub>External reference routes are used for camera, mask, and teacher-quality auditing. They are reference controls, not student outputs.</sub></p>

## Repository role

This repository is the model-prior side of the broader human-prior VGGT project stack.

| Repository | Role |
| --- | --- |
| `VGGT-SMPL-X-Human-Prior-Adapter` | Model-side SMPL-X prior injection and supervision route |
| `VGGT-ZJU-Mocap-Adapter` | Dataset adaptation, camera alignment, mask audit, and trusted case export |
| `vggt-human-prior-builder` | Release-safe public preprocessing recipe and schema boundary |
| `TuringResearch_plus` | MCP-first research workflow engine used for evidence and planning support |

## What the project demonstrates

This project demonstrates my work on:

- reading and adapting a visual geometry foundation model baseline;
- designing a human-prior route that does not replace the model output with a template;
- aligning human priors to real cameras;
- building prior maps, prior depth, prior points, prior normals, and prior masks;
- separating teacher/reference outputs from student/model-owned outputs;
- organizing experiments around baseline, controls, evidence, and reproducible project records.

## Results and project links

The following Yuque pages contain additional stage records and project displays:

- https://www.yuque.com/maturinlove221/gqr279/emwf87ku108nzvez
- https://www.yuque.com/maturinlove221/gqr279/fg8lq33tgbwiagtt
