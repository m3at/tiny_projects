# Validation record — 2026-09-06

Completed headlessly on macOS ARM64, CPython 3.13.5. The locked environment uses
MuJoCo 3.12.0, Gymnasium 1.3.0, PyTorch 2.14.0 and Stable-Baselines3 2.9.0.
Linux rendering instructions are provided but were not tested on this host.

## Executed checks

- `make setup`: installed/synchronized the locked project environment successfully.
- `make data`: verified the pinned archive SHA-256 and extracted 6,702 SVGs, with
  upstream notices and provenance. Repeated execution succeeded from the local archive.
- `make check`: Ruff passed; both integration tests passed. These cover IK at workspace
  extremes, Gymnasium API compliance, deterministic dynamics, five ordered 永 strokes,
  arc-length sample spacing, inter-stroke lifts, paper reset and invalid-action rejection.
- `make train`: 17,920 expert labels from 28 noisy demonstration episodes; a CPU MLP
  trained for 35 epochs with seed 7. Checkpoint: `runs/bc.pt`.
- `make validate`: passed API, per-character <4 mm RMSE, >95% drawing contact,
  >98% raised-pen clearance, >80% improvement over stationary baseline, and actual
  offscreen animation gates. No held-out episode truncated.
- `make ppo`: completed 32,768 transitions from scratch with seed 7; saved
  `runs/ppo.zip`. Reloaded it for held-out evaluation and an offscreen 永 animation.
- `make demo CHARS=水`: reloaded the cloned policy and exported PNG, GIF, NPZ and JSON.
- `make build`: wheel built successfully; archive inspection confirmed `shodo/arm.xml`
  is included, so the installed package carries its robot model.

Gymnasium reports two advisory warnings for intentionally unbounded observations
(velocity and tracking-error features). Its API/determinism checks pass. The surrounding
shell's different active virtualenv also produces a uv warning; uv correctly uses
this project's `.venv`.

## Held-out results

Training: 一二三十木大人. Held out: 永水日山. Values are 3D tip-to-reference RMSE
in millimeters; the mean is the unweighted mean of the four glyph RMSEs.

| Controller | 永 | 水 | 日 | 山 | Mean |
|---|---:|---:|---:|---:|---:|
| Analytic teacher | 0.718 | 0.732 | 0.760 | 0.713 | 0.731 |
| Behavior cloning | 0.708 | 0.721 | 0.751 | 0.704 | 0.721 |
| Stationary | 89.692 | 82.044 | 85.876 | 98.979 | 89.148 |

PPO independently achieved mean held-out RMSE **1.479 mm** after 32,768 training
transitions. Its detailed results are in `runs/evaluation-ppo.json`. This is one seed
and a simple target-conditioned tracking task; it is not a convergence study.
Both teacher and cloned policy achieved 100% drawing contact and raised-pen clearance
at evaluated control steps on all four held-out characters.

A warmed benchmark of five complete teacher 永 rollouts measured approximately
12,835 control steps/second, including dynamics and ink rasterization, excluding
initial SVG parsing and rendering. Each control step integrates ten physics steps.
This measurement is host/load dependent, not a portable performance guarantee.

## Inspect the output

- `runs/validation.json`: current validation result, package versions and source hashes.
- `runs/validation-rollout.gif`: 213 frames covering the 21.28-second simulated drawing.
- `runs/validation-rollout-scene.png`: final arm scene alongside the actual ink raster.
- `runs/validation-rollout.png`: full-resolution paper; visually inspected as 永.
- `runs/validation-rollout.npz`: executed tip, reference, joints and stroke IDs.
- `runs/evaluation.json`: per-character teacher, cloned and stationary metrics.
- `runs/bc.json`, `runs/ppo.json`: training settings and provenance.

The scene and ink output were visually inspected. The rendering shows the moving
articulated arm and a separate labeled paper view. Ink derives from physical tip
positions, never from copying the target. Earlier exploratory render files are
retained in `runs/initial-pass/`; current outputs use the names above.

## Scope of the evidence

This validates an installed, reproducible environment for training and visualizing
a virtual brush arm, with real stroke-order data and two working learning pipelines.
The brush is vertical and uses a procedural compression/deposition model with no hard
paper contact, bristle mechanics, force sensing or calibrated ink physics. KanjiVG
contains centerline geometry, not measured expert shodo motion. The results establish
path tracking and pen lifts, not brush-style mastery or transfer to a real robot.
