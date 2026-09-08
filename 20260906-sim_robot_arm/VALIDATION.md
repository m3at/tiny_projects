# Current validation

Results are for the B601-RS with a rigid vertical brush, dated 2026-09-08. Earlier Panda controller, native-transfer and broad-cohort results are not carried forward. Source hashes, package versions, configuration and seeds accompany the artifacts. These are uncalibrated simulation results, not hardware or aesthetic qualification.

## Fast engineering checks

`make check` passes **288 tests**, with no selections or exclusions. Three fresh processes reported pytest times of **4.29, 4.23 and 4.27 s**; complete Make/lint/format/test wall times were **4.782, 4.731 and 4.750 s**. The background robustness job was paused for these measurements. After the final clock guard review, another full run passed in **4.669 s** (pytest **4.22 s**), with no background jobs. Evidence: `runs/test-timing.json`. The three Gymnasium warnings concern intentionally unbounded observations and checking its sensor wrapper.

The suite was reduced from 400 tests / 11.77 s by removing redundant CLI routing and invalid-argument permutations, replacing metadata-only full-character rollouts with a 36-step physical approach/draw/lift/settle fixture, consolidating native sensor/contact/reset checks, and combining real FFmpeg timing/padding coverage. The native cantilever check retains mass, discrete beam agreement, recovery and two timesteps on a three-segment fixture that settles within 0.3 s. Full character quality remains in `make validate`; broader rod/bundle refinement remains in `make mechanics`. No slow pytest group was hidden or moved behind an exclusion.

Repeated environment construction now copies private compiled model templates. A local 20-sample probe measured median construction stages of 46.9 ms for XML/mesh compilation versus 19.7 ms for a model copy. Isolation tests mutate friction, geometry groups, textures, body mass and joint state in one instance and verify another is unchanged. Fresh processes still compile their own initial templates.

`make build` produces the wheel, including the B601 SHA-256 manifest. The source pins are retained and the core uv lockfile is tracked.

## Setup audit

`make prepare` passes **8,385 / 8,385** candidate poses: every authored sample in 一二三十木大人永水日山 and a 125-point command-volume grid. The grid has no arm/handle collisions, maximum FK residual below 0.001 mm, minimum joint margin 0.224 rad, and maximum static gravity demand 73.8% of the configured rated torque. The arm-only published mass sums to 5.208 kg; the provisional handle adds 0.040 kg.

Evidence: `runs/preparation/preparation.json` and `workspace.npz`. This is a discrete geometric/static audit, not executed motion, a continuous swept-volume proof or dynamic-load qualification. Public inertia tensors and collision convex hulls remain unverified.

## Fixed held-out comparison

Default CPU BC: seed 7, 28 episodes, 35 epochs, 17,920 privileged teacher labels from the seven training characters. Training material variation is ±20%; evaluation uses nominal materials. No training episode truncated. The teacher and BC pass **123 checks** across 永水日山, including physical tracking, ink, force, orientation, contact/lifts, torque, joint speed/margins, collisions, conservation and actual-motion rendering.

| Character | Teacher RMSE mm | BC RMSE mm | BC ink RMSE mm | BC peak force N | Zero RMSE mm |
|---|---:|---:|---:|---:|---:|
| 永 | 1.569 | 1.572 | 1.787 | 0.698 | 89.998 |
| 水 | 1.552 | 1.555 | 1.724 | 0.688 | 82.407 |
| 日 | 1.422 | 1.424 | 1.477 | 0.675 | 86.221 |
| 山 | 1.513 | 1.514 | 1.617 | 0.686 | 99.317 |

Both teacher and BC complete 4/4 episodes, with 100% requested drawing contact and 100% clear high lifts. BC peak brush force stays below 0.70 N, orientation RMSE below 0.14°, peak joint speed below 0.43 rad/s, and torque demand below 68.4% of the configured rated limits. There are no arm/handle collisions. Zero action completes 4/4 stationary episodes and deposits no ink; missing-ink values remain null rather than being counted as accurate drawing.

Canonical artifacts: `runs/validation.json`, `runs/evaluation-learned.json`, and `runs/validation-rollout.{mp4,json,npz,png}` plus the scene PNG. The inspected H.264 recording shows the actual B601 motion and coupled brush/ink state. Named NPZ columns include measured joints/velocities, filtered joint targets/velocities and actual actuator torque. Affine actuator inputs are not reported as torque.

## Sensor policy and robustness

A separate 156-input sensor-history BC was trained on the same 28 episodes / 35 epochs / seed 7, with 17,920 labels. Its four 39-feature slices exclude contact-center, bristle-deflection, contact-fraction and ink-state truth. The seventh joint slot remains an explicit zero. The force channel is a synthetic compensated contact-force proxy.

The paired audit completed **448/448 episodes**, with zero truncations: 14 cases × four controllers × four held-out characters × seeds 7 and 17. Each controller has 112 completed episodes. Zero action has 112 missing-ink results; all other controllers have 112 valid ink measurements. Every case/controller cell retains eight trials and paired nominal deltas in `runs/sensor-bc/robustness.json`.

| Controller | Nominal mean ink RMSE mm | Combined-error mean ink RMSE mm | Combined drawing contact | Clear high lifts across all cases |
|---|---:|---:|---:|---:|
| Measured-only classical | 5.624 | 3.579 | 99.02% | 100% |
| Privileged oracle | 1.644 | 1.644 | 100% | 100% |
| Sensor-history BC | 1.634 | 3.351 | 98.52% | 100% |
| Zero | null | null | 0% | 100% |

Sensor BC case-mean ink error ranges from 1.493 mm under latency to 3.628 mm under positive paper XY registration error. Positive height error reduces its mean drawing contact to 97.22%; negative height error raises its mean episode peak force to 0.882 N. Completion therefore does not imply unchanged writing quality. Some perturbations reduce an existing bias, so lower error under a stress case is not a general robustness improvement. The oracle's identical sensor-case results are expected because it accesses privileged true state. Paper registration errors do not physically move the paper.

This exploratory run started before the final model-template cache and command-clock refinements; its recorded source hashes identify that version. The final fixed validation was rerun after those refinements. No acceptance threshold, hardware safety score or real-sensor qualification is inferred from this audit.

## Portability and scope

Actual privileged and sensor checkpoints were loaded on CPU and Apple Metal and compared on 128 observations each along a held-out trajectory. Maximum absolute normalized action differences were 8.94e-8 and 1.19e-7 respectively. Evidence: `runs/portability.json` and `portability.npz`. This checks sampled inference portability, not full cross-device rollout equivalence or training equivalence.

Optional SmolVLA has focused regression tests for causal cameras, normalization, frozen bases, artifact integrity, warm-start ancestry, separate recovery labels and live-state-safe caching. No B601 SmolVLA training, closed-loop quality or real-time inference qualification is claimed. Native rods retain integration/constitutive checks; no earlier Panda-to-native policy-transfer claim applies to this setup.

[DEPLOYMENT.md](DEPLOYMENT.md) records the remaining work: delivered URDF verification, encoder/mount/paper calibration, measured inertia/friction/backlash, real force sensing, brush/material identification, transport scheduling and independently tested hold/stop behavior. A simulation-time command lease cannot establish a wall-clock hardware watchdog.
