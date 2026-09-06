# Validation record

The numerical audit dated 2026-09-06 covers 5,460 broader-cohort episodes, 800 material-grid episodes, native/reduced numerical studies and multi-seed training comparisons. Broad-cohort and material-grid results refer to their recorded source/checkpoint hashes. Regression tests, nominal controller validation, coordinate registration and device compatibility are checked separately below.

Host: macOS ARM64, 8 CPU cores, 24 GiB RAM, CPython 3.13.5. Core packages: MuJoCo 3.12.0, Gymnasium 1.3.0, NumPy 2.5.2, PyTorch 2.14.0, Stable-Baselines3 2.9.0. Matplotlib 3.11.1 is a development-only plotting dependency. Linux rendering is documented but not tested on this host.

## Completed checks

- `make check` passes 88 tests, Ruff lint and formatting checks. Coverage includes arm/brush mechanics, conservative ink transport, native roots/beam response, control and checkpoint contracts, device routing, immutable path caches, world/raster/render coordinates, paper-edge footprints, decoded MP4 timing, encoding failures and curated CLI output.
- Calibrated offscreen tests locate ink at asymmetric world coordinates in both views and reject a deliberately mirrored image. Edge tests check the full physical paper extent and single clipping of Gaussian footprints: half the mass remains at an exact straight edge and one quarter at a corner. Force-limit and warning-triggered truncations flush pending transport, preserving paper/simulation timing, drying and pigment accounting.
- Default BC, plain PPO and fixed seed-7 residual PPO each pass 99-check validation: tracking, visible ink, contact, lifts, force, orientation, motor limits, pigment conservation, stationary-baseline improvement, and offscreen rendering.
- BC trained on the reduced brush also passes 107 native-backend validation checks on all four held-out glyphs, without native retraining. This is model-to-model transfer, not evidence of transfer to hardware.
- All 6,702 local KanjiVG files pass finite-trajectory, stroke-order and spacing checks. The explicit paper-local coordinate construction preserves held-out step counts and stroke IDs; world paths agree within 2.23e-16 m. Evidence: `runs/coordinate-audit/`.
- Data/robot downloads verify the pinned archive and 62 Panda files (including 59 meshes). An isolated wheel installation loaded the asset manifest and Panda.
- BC/PPO reloads preserve deterministic actions on the same device. Apple Metal BC and residual-PPO training pass compatibility checks, including CPU continuation from 2,048 to 4,096 transitions; CPU/Metal loaded actions differ by at most 7.45e-9 on the recorded probe. Wrong-seed resume and concurrent writers are rejected without changing the saved model. These short runs test compatibility, not controller quality; evidence and device microbenchmarks are in `runs/device-smoke/`.
- All eight PPO checkpoints pass fresh 99-check nominal validations; their tracking RMSE agrees with the recorded comparison within 5.23e-11 mm. `runs/controller-validation.json` records the per-controller checks. The larger cohorts and material grids are not rerun by this nominal check.
- The established BC and fixed residual controllers also complete all four held-out glyphs on Metal without truncation. Their maximum CPU/Metal tracking-RMSE difference is 6.69e-9 mm (`runs/device-smoke/heldout-portability.json`); this is inference portability, not cross-device training equivalence.

The two Gymnasium warnings concern intentionally unbounded observation limits; API and determinism checks pass.

## Fixed four-character comparison

Training: 一二三十木大人. Held out: 永水日山. BC used 56 noisy episodes (35,840 labels), 60 epochs, seed 7. PPO variants each requested one million transitions (rounded to complete rollout batches).

Values are tracking RMSE in millimeters: loaded ink-center XY plus nominal-tip Z. These are not purely handle-tip errors; offsets intentionally compensate brush drag.

| Controller | 永 | 水 | 日 | 山 | Mean |
|---|---:|---:|---:|---:|---:|
| Teacher | 1.261 | 1.241 | 1.212 | 1.183 | 1.224 |
| BC | 1.262 | 1.241 | 1.210 | 1.182 | 1.224 |
| Plain PPO | 1.129 | 1.136 | 1.040 | 1.101 | 1.102 |
| Residual PPO | 0.925 | 0.886 | 0.882 | 0.863 | 0.889 |
| Stationary | 89.998 | 82.407 | 86.221 | 99.317 | 89.486 |

Residual ink-center XY RMSE is 0.813/0.724/0.678/0.687 mm respectively. Its pressure error is higher than the teacher's on these nominal tests; tracking improvement alone is not an across-the-board improvement.

The contact-offset ablation gives teacher ink-center error on 永 of 5.368 mm without compensation and 1.267 mm with compensation. The compensated handle intentionally departs farther from the target centerline.

## Longer training and objective alignment

All three roughly-three-million-transition residual checkpoints pass 99 validation checks. Fixed-budget nominal means on the same four glyphs are:

| Training seed | Tracking RMSE (mm) | Ink XY RMSE (mm) | Force RMSE (N) |
|---|---:|---:|---:|
| 7, resumed | 0.814 | 0.839 | 0.060 |
| 17, fresh | 1.082 | 1.100 | 0.053 |
| 27, fresh | 0.955 | 0.902 | 0.102 |

Fresh runs completed 3,000,320 transitions. Seed 7 continued from 1,001,472 to 3,002,368 with optimizer/checkpoint state but a new environment/RNG sequence; it is not a bitwise uninterrupted run. These results show seed variability and tradeoffs, not universal improvement with a larger training budget. The primary broad cohort uses the fixed 1M checkpoint, not a longer-training checkpoint.

On 永, seed 7's drawing XY error worsened 0.813→0.887 mm after continuation, while air XY error improved 0.799→0.555 mm and air Z error 0.586→0.425 mm. A better whole-episode score can therefore hide worse ink tracking. A separate opt-in `--ink-objective` experiment adds dense loaded-ink accuracy credit during drawing; its 1M run is complete and passes 99 nominal checks. Mean ink error improves from 0.726 to 0.638 mm, but mean force error worsens from 0.044 to 0.144 N. The complete 100-episode material grid gives ink 0.651 mm, force 0.164 N, worst force error 0.324 N, no truncation and minimum coverage 99.64%. This is not promoted to the default: an ink-only emphasis sacrifices pressure fidelity. Its 780-episode broad cohort is complete: mean ink error 0.629 mm, 95th percentile 0.769 mm, mean force error 0.116 N, no truncation or missing ink, minimum visible coverage 99.92%.

The combination with the pressure preset also passes 99 nominal checks: mean ink error 0.681 mm and force error 0.0437 N. Its 100-episode material grid and 780-episode broad cohort are complete. On the recorded 永 drawing samples, the ink-only policy's signed force error averages −0.133 N, versus +0.011 N for the default residual and −0.039 N for the combination: the ink-only improvement comes with systematic under-loading, not just force noise. The wrapper changes no observations, deposition or dynamics; the default objective is unchanged.

## Mechanical and transport checks

The independent native cantilever audit varies 3/6/12/24 segments and timestep, checks 7/19/37 representative bundles, and tests load scaling and recovery. All cases preserve mass within 1 ppm, match discrete beam theory within 0.23%, and recover to within 1 micrometer after unloading. The discrete fixed-first-segment beam approaches continuum theory as segments increase; a short transient test must not be mistaken for a settled response.

Straight native rods produce touchdown spikes above 20 N despite passing bending and solver-warning checks. A 36-case sweep varies curvature, approach speed and timestep, retaining failed stress cases. The native reference preset uses 1 mm stress-free curvature and a 1 mm/s final approach. One complete 永 achieved peak force 0.439 N, maximum penetration 0.0564 mm, ink-center error 2.137 mm and visible coverage 97.27%, without truncation. Its narrower ink trace differs visibly from the reduced model; neither is a calibrated physical brush. Whole-episode tracking RMSE is not directly comparable across these backends because the slower native approach adds many air/approach samples; compare ink error and coverage separately. Handle–bristle collisions are excluded by explicit collision categories so the clamp defines attachment. The eight-case refinement study has no truncation through 19 bundles × 12 segments on 一永. At the common 0.05 ms timestep on 永, 7×6 / 7×12 / 19×6 / 19×12 give ink errors 2.142 / 2.171 / 2.003 / 2.100 mm and peaks 0.455 / 0.394 / 0.397 / 0.370 N. Bundle counts alter contact geometry; segment counts alter the clamp discretization. These are sensitivity measurements, not interchangeable models of identical hairs.

The completed four-glyph timestep study compares 0.2/0.1/0.05 ms with the same seven-bundle, six-segment model. From default 0.1 ms to 0.05 ms, each glyph's ink RMSE changes by less than 0.029 mm, integrated force-vector difference is below 0.326%, and microstep peak-force difference is below 3.60%. Maximum penetration falls from 0.046–0.056 mm to 0.023–0.027 mm. This supports timestep consistency at the reported tolerances, not a claim of exact constraints or geometry-independent physical accuracy. Finer segments change the clamp discretization and force response. The complete 20-case timestep/friction study also tests friction 0.44 and 0.66; none truncated. The separate faster-native preset passes 107 teacher/BC-transfer checks. Its full-永 timestep comparison gives ink 2.307→2.235 mm and peak 0.664→0.651 N from .1 to .05 ms: greater ink sensitivity than the slower reference.

For the reduced brush, the completed 48-case bundle/timestep audit gives mean tracking RMSE 1.224 mm at 2 ms versus 1.243 mm at 0.5 ms (four held-out glyphs, 19 bundles). Peak load is 0.73677 versus 0.73655 N.

Ink-grid tests at 128²/256²/512² and multiple transport cadences conserve pigment to roundoff. On 永, painted area varies by less than 1.6% and centerline coverage remains 100%. Requested transport intervals round up to a controller boundary (e.g. 0.05 seconds executes at 0.06 seconds). Diffusion is separately checked against the analytical second-moment increment.

## Material stress tests and broader glyph cohorts

The completed 300-episode plant-only material sweep spans 0.5–1.5 times nominal stiffness and friction while keeping target force fixed. No episode truncated. Mean ink-center error: teacher 1.214 mm, BC 1.217 mm, residual PPO 0.758 mm. Mean pressure error: teacher 0.135 N, BC 0.135 N, residual PPO 0.140 N. This exposes pressure robustness limitations outside the default ±20% training range.

A separate pressure-focused teacher/BC experiment improves force tracking with stronger feedback and ±50% training variation. On half-stiffness 永, BC force RMSE falls from 0.249 to 0.114 N; overall tracking RMSE increases from 1.175 to 1.529 mm. The teacher/BC full grid is complete: pressure BC mean ink error is 1.193 mm, mean force error 0.0480 N, and worst force error 0.115 N across 100 episodes. The pressure-focused residual has completed 1M transitions and passes 99 nominal validation checks. Its material grid is also complete. Each row below averages the same 100 plant perturbations (25 stiffness/friction combinations × four glyphs):

| Controller | Ink XY RMSE (mm) | Force RMSE (N) | Worst force RMSE (N) |
|---|---:|---:|---:|
| Default BC | 1.217 | 0.135 | 0.252 |
| Default residual | 0.758 | 0.140 | 0.274 |
| Pressure BC | 1.193 | 0.048 | 0.115 |
| Pressure residual | 0.754 | 0.061 | 0.112 |
| Ink residual | 0.651 | 0.164 | 0.324 |
| Pressure + ink residual | 0.686 | 0.062 | 0.148 |

No controller truncated or missed all ink. The pressure residual's minimum visible coverage was 99.79%, maximum load 0.839 N. Its mean force error is 56.3% lower than the default residual's, with similar ink accuracy, but pressure BC still has lower mean force error. These are separate training/reward settings, not a universal winner. Adding the ink bonus to the pressure profile reduces mean ink error another 9.0% with similar mean force error, but increases the worst force error from 0.112 to 0.148 N. Its minimum coverage is 99.79% and maximum load 0.823 N. This combination is a useful accuracy/pressure compromise, not a dominance claim or a new default.

The fixed 260-glyph × three-material-seed × four-controller comparison is complete: 3,120 episodes, no truncation or missing ink. Each row summarizes 780 episodes:

| Fixed controller | Mean ink RMSE (mm) | 95th percentile ink (mm) | Mean force RMSE (N) |
|---|---:|---:|---:|
| Teacher | 1.216 | 1.349 | 0.045 |
| BC | 1.218 | 1.357 | 0.047 |
| Plain PPO | 0.945 | 1.071 | 0.083 |
| Residual PPO | 0.744 | 0.917 | 0.075 |

Residual PPO reduces mean ink error 38.9% relative to BC, but has higher mean force error. The paired mean improvement is 0.4743 mm; its 10,000-resample glyph-bootstrap 95% interval is [0.4696, 0.4790] mm (seed 20260906, all 260 glyph pairs). Resampling is by glyph after averaging its three material seeds, not by correlated episodes. This interval is conditional on the fixed materials/checkpoints, not a measure of training or hardware uncertainty. Minimum visible coverage across all four policies is 99.64%, maximum load 1.0604 N. Artifact metadata records the source and checkpoint hashes used by each run.

Three separate exploratory extensions use exactly the same 260 glyphs and material seeds. All 2,340 additional episodes completed without truncation or missing ink:

| Exploratory residual | Mean ink RMSE (mm) | 95th percentile ink (mm) | Mean force RMSE (N) |
|---|---:|---:|---:|
| Pressure-focused | 0.763 | 0.932 | 0.041 |
| Ink-focused | 0.629 | 0.769 | 0.116 |
| Pressure + ink | 0.665 | 0.829 | 0.035 |

The combination has lower mean ink and force errors than the fixed default residual on this cohort, but the ink-only variant still has lower ink error and the pressure-only variant has lower worst force error on the broader material grid. The combination's minimum coverage is 100% and maximum load 0.789 N. These extensions were selected after earlier results; they are exploratory, not a blinded model-selection test. The four-controller comparison uses fixed settings.

Material pairing checks confirm identical full brush parameters and paper fiber hashes for all 2,340 extension rows (`material-seed-pairing.json`). All cohorts have the same glyph order and material seeds as the fixed comparison, with no duplicate episode keys. The three actual material realizations are stiffness 231.008/250.367/237.401 N/m and friction 0.63739/0.47541/0.50904 for seeds 7/17/27. All three sampled stiffnesses are above nominal 220 N/m; the separate 25-condition stress grid covers softer material. Do not interpret three random draws as exhaustive coverage of the training range.

The fixed cohort's torque peak is sampled at controller boundaries; canonical validation records the peak across physics microsteps and additionally records force impulse. Do not compare those torque summaries as though their sampling were identical.

## Performance measurements

Measured sequentially after training/cohort jobs stopped, on the host listed above. Each end-to-end measurement uses 永, fixed seed 7, one warm-up episode and three timed episodes. It includes teacher, controller, dynamics and ink; excludes model loading, reset and rendering. Rates are host/load dependent, not training throughput.

| Preset | Median control steps/s | Trial range | Simulated seconds / wall second |
|---|---:|---:|---:|
| Default: 19 bundles, 256² ink | 1,044.2 | 1,043.1–1,045.7 | 20.88 |
| Quality: 37 bundles, 512² ink | 770.3 | 744.5–771.5 | 15.41 |
| Native reference: 7×6 rods, 0.1 ms | 36.02 | 35.81–36.03 | 0.720 |

All trials completed with identical tracking/peak-force metrics within each preset. The native reference takes 2,524 control steps versus 1,064 for reduced models because of its slow approach. Do not confuse fewer reference steps with faster physics.

The separate paired IK microbenchmark uses seven trials of 1,000 calls at a warm fixed air pose and verifies equal Jacobians/positions before timing. Replacing full `mj_forward` with the required kinematics/center-of-mass stages reduces median prerequisite-plus-Jacobian time from 4.760 to 1.767 µs (2.69×) for reduced, and 26.126 to 3.471 µs (7.53×) for native. These are component speedups, not whole-simulator speedups. Exact rollout-equivalence checks separately guard physical behavior. Reports: `benchmark-default/benchmark.json`, `benchmark-quality/benchmark.json`, `benchmark-native/benchmark.json`, and `ik-benchmark.json`, all under `runs`.

## Artifacts

`runs/README.md` indexes selected reference recordings and demos. Other exploratory policies retain their numerical results, named trajectories and final ink images; movies can be regenerated from their checkpoints. `runs/recording-audit.json` verifies the stored GIF audit recordings' decoding, simulated timing, matching paper views and finite trajectory columns.

MP4 export checks decode actual H.264 frames and verify state timing, final holds, `yuv420p`, odd-dimension padding, a front-loaded playback index and preservation of existing videos on encoder failure. `make demo CHARS=自在` produces 1.02/0.94 MiB videos, approximately 90% smaller than the corresponding 10.20/9.14 MiB GIFs. Their decoded final frames have PSNR 46.56/46.20 dB against the original RGB renders. Exact sizes, duration/frame-count checks and the limited final-frame comparison are recorded in `runs/demo-video-checks.json`; these are encoding checks, not new physics or aesthetic scores.

- `runs/validation.json`, `validation-rollout.*`: default BC validation/recording.
- `runs/residual-seed7/validation.json`, `validation-rollout.*`: fixed 1M residual result.
- `runs/cable-curved-one-mm.*`: native full-glyph reference recording.
- `runs/mechanics/cantilever.json`, `mechanics-summary.svg`: beam/contact checks.
- `runs/contact-sweep.json`: passing and deliberately failing touchdown cases.
- `runs/dataset-audit.json`, `ink-convergence.json`: geometry/transport audits.
- `runs/material-sweep/report.json`: completed plant-only material stress grid.
- `runs/heldout-sweep/`, `heldout-summary.json`, `heldout-distribution.svg`: completed four-controller broad cohort, paired statistics and ink-error distributions.
- `runs/native-audit/`: completed eight-case native refinement study.
- `runs/heldout-pressure/`, `heldout-ink/`, `heldout-pressure-ink/`: completed exploratory cohorts; corresponding `*-summary.json` files and figures hold results.
- `runs/pressure-ink-residual/validation-rollout.*`: combined-controller recording.

The numerical audit includes headless recording/figure inspection, full GIF decoding, finite named trajectory histories, deterministic rollout equivalence and isolated wheel installation. Those checks establish artifact integrity and reproducibility, not physical calibration or visual coordinate registration.

## Limits of this evidence

The canonical four-glyph set was inspected during iterative engineering: it is validation evidence, not a blinded final test. The larger cohort uses four canonical glyphs plus 256 fixed-seed sampled glyphs, with three paired material realizations. Glyph-bootstrap intervals describe variation over that cohort, conditional on those materials and the fixed trained checkpoint; they do not quantify training-seed or hardware uncertainty. The separate three-seed study addresses training variability only at its stated protocol and budget.

This is supplied-path control, not recognition, aesthetic calligraphy learning or hardware transfer. KanjiVG pressure/tilt/timing are procedural. Reduced bristles are massless approximations; native bundles omit inter-hair locking and capillary clumping. Ink is a continuously fed conservative porous-paper approximation, not a calibrated fluid/chemical model. Numerical agreement does not establish physical calibration. See [PHYSICS.md](PHYSICS.md) for equations, primary sources and detailed limitations.
