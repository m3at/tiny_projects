# Shodo arm laboratory

A headless Python 3.13 laboratory for the **reBot Arm B601-RS**, with a brush rigidly attached to the wrist and held vertically. Published six-joint geometry drives MuJoCo dynamics; elastic brush bundles couple contact forces back into the arm, and a conservative water/pigment grid records the executed ink. The stack stays small: uv, Make, MuJoCo, NumPy/SciPy, Gymnasium and optional neural policies. No ROS installation or hardware connection is required.

[DEPLOYMENT.md](DEPLOYMENT.md) records the public hardware evidence, provisional setup, calibration procedure and remaining hardware work. [PHYSICS.md](PHYSICS.md) explains the models; [VALIDATION.md](VALIDATION.md) records the current results. Tracking these authored paths is not evidence of calligraphy mastery or sim-to-real readiness.

## Run

Install uv and FFmpeg with libx264 for recordings, then:

```sh
make setup
make data
make prepare
make check
make validate
make demo CHARS=永水
```

`make data` verifies pinned KanjiVG SVGs and 25 B601 source/license/mesh files against SHA-256 manifests. Downloads live in ignored `data/`; the core and optional integration lockfiles are tracked. `make prepare` writes `runs/preparation/preparation.json` and full workspace/load arrays. `make validate` trains a missing BC checkpoint on 一二三十木大人 and compares teacher, learned and zero-action controllers on 永水日山. It renders actual executed motion to `runs/validation-rollout.mp4`, with named motion arrays, scene/paper images and JSON metrics. Numerical training/evaluation do not need FFmpeg.

Neural work defaults to CPU; `DEVICE=auto` opts into CUDA, then Apple Metal, then CPU. Small policies were previously measured faster on this host's CPU; physics and ink always run there. Use separate descriptive `RUN_DIR` values for experiments. New training datasets refuse overwrites. Ordinary evaluation reports and demos replace their named outputs.

## Robot and controls

The source is Seeed's pinned B601-RS URDF, converted directly to MJCF with its origins, axes, link masses, full inertia tensors and visual meshes. The gripper and its descendants are removed. The provisional tool is a 40 g, 130 mm handle with 30 mm exposed bristles; its pose is fixed to `link6`. The arm base is at world `(0.20, 0, -0.005)` m, and the 210 mm paper/pad is centered at `(0.50, 0, 0)` m. These are installation assumptions to measure, not manufacturer dimensions.

A 50 Hz Cartesian policy feeds damped IK and a bounded joint target filter. At each physics step, the arm applies MIT-style position/velocity impedance plus model gravity/Coriolis compensation. Total actuator torque is limited to 11 N·m for J1–J3 and 5 N·m for J4–J6, the published motor rated torques. Experiment target limits are 0.8 rad/s and 4 rad/s². The governor rejects stale/nonmonotonic commands and latches faults. This host-side mechanism has no hardware stop authority.

The action array retains six slots for existing learning/data code. Only XYZ increments act; rotation requests are recorded but their effective increments are zero. The wrist physically regulates the vertical brush orientation; the simulator never overwrites its pose after reset. Observed orientation errors and pen-lift failures therefore remain measurable.

The privileged input layout remains 40 features, contract 3. Sensor history remains 39 features per slice, contract 1. In both, the former seventh joint position/velocity slots are reserved zeros, including with native brush joints and sensor noise. A separate B601 robot contract and action contract 2 reject old Panda checkpoints/datasets/adapters. BC/PPO execution also checks the recorded robot configuration.

Configuration lives in immutable dataclasses, with TOML overrides. For example:

```toml
[robot]
base_xyz = [0.20, 0.0, -0.005]
mount_xyz = [0.0, 0.0, 0.0]
mount_rpy = [0.0, 0.0, 0.0]
handle_mass = 0.04
joint_speed = 0.8
joint_acceleration = 4.0
```

Run `make prepare CONFIG=your-setup.toml` before generating new training data for a changed placement, mount or controller. The importer is intentionally specific to this six-joint chain; replacing the public URDF requires updating its pin/manifest and checking FK, limits, inertias and meshes against the delivered description.

## Experiment

```sh
make train RUN_DIR=runs/bc-experiment EPISODES=28 EPOCHS=35
make evaluate RUN_DIR=runs/bc-experiment
make train OBSERVATION=sensor RUN_DIR=runs/sensor-bc
make evaluate RUN_DIR=runs/sensor-bc
make record POLICY=oracle RUN_DIR=runs/sensor-data CHARS=一二三十木大人 EPISODES=7 CAMERA_EVERY=5
make robustness POLICY=classical RUN_DIR=runs/sensor-baselines SEEDS="7 17 27"
make benchmark CHARS=永
```

Sensor actors receive synthetic measured pose/joints/velocity/force, authored references, commands, acquisition age and freshness. Contact-center, bristle-deflection, contact-fraction and ink-state truth are excluded. Privileged labels and separate diagnostics may supervise training. Force is a synthetic compensated contact-force proxy, not a calibrated wrist sensor. The measured-only classical baseline uses 0.0008 m/N force correction; it has no contact-offset compensation. Delayed/noisy measurements remain explicit robustness cases, not claims about the physical device.

`dataset.py` stores `T+1` observations and decision timestamps, `T` requested/effective commands, rewards, termination flags, unnormalized measured channels and separately named privileged diagnostics. Camera frames have their own acquisition times and no overlays or inset. Dataset metadata includes source attribution, configuration and hashes. Applied actions mean bounded Cartesian command increments, not executed displacement; named joint command, velocity and actual torque columns expose the subsequent joint-control behavior.

Optional PPO/residual training remains available through `make ppo` and `make residual`. Native cable rods remain available through `CONFIG=experiments/cable.toml`; they need much smaller physics steps and additional beam/contact/refinement validation. Existing exploratory presets are not qualified B601 controllers. The fast reduced brush is the default for iteration.

[SmolVLA](integrations/smolvla/README.md) remains an isolated optional integration with its own lockfile and `VLA_DEVICE=auto`. Record fresh B601 episodes before preparing/training adapters. Pretrained revisions remain pinned, base weights frozen during LoRA training, and camera selection causal. Synchronous inference timing and minibatch loss do not establish a 20 ms control deadline or policy quality. Nothing uploads automatically.

## Source attribution and code

[KanjiVG r20250816](https://github.com/KanjiVG/kanjivg/releases/tag/r20250816), by Ulrich Apel and contributors, supplies ordered centerlines under CC BY-SA 3.0. Preserve attribution and share-alike terms for derived trajectories. Pressure and timing are procedural; orientation is fixed vertical. [Seeed's B601 description](https://github.com/Seeed-Projects/reBot-DevArm/tree/ce074041cd1c26f67ce74c2ca6fca9af22f8aee5/Rebot_Arm_description/RS) and its CERN-OHL-W-2.0 license are retained beside downloaded meshes. Conversion changes and source limitations are documented in DEPLOYMENT.md.

`rebot.py` fetches/verifies/converts the arm; `robot.py` drives dynamics; `actuation.py` implements the shared joint governor and explicit encoder mapping; `preparation.py` audits candidate poses; `contracts.py` defines policy semantics; `runtime.py` adapts sensors and executes transitions. `brush.py`, `cable.py`, and `ink.py` model contact/deposition; `learning.py`, `rl.py`, and `validation.py` train and compare policies. No CAN driver, motor enable command or real-time hardware runner is included.
