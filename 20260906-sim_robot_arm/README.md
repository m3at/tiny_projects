# Shodo arm laboratory

A headless Python 3.13 laboratory for target-conditioned brush control: a torque-driven seven-joint Panda, elastic/frictional bristle bundles, water/pigment transport, KanjiVG stroke paths, imitation learning, and residual reinforcement learning. Neural training and inference support CPU, CUDA and Apple Metal; no GUI is required. Offscreen recordings show the executed arm and actual ink.

The simulation audit includes 5,460 cohort episodes across 260 glyphs and reduced/native numerical checks. [VALIDATION.md](VALIDATION.md) records results and controller tradeoffs; [PHYSICS.md](PHYSICS.md) explains models and sources. [WORK_LOG.md](WORK_LOG.md) summarizes engineering priorities. These are uncalibrated simulations, not demonstrated sim-to-real shodo.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and FFmpeg with the `libx264` encoder (for example, `brew install ffmpeg` on macOS), then:

```sh
make setup
make data
make check
make train
make validate
make demo CHARS=永水
```

Data acquisition verifies the pinned KanjiVG archive and every pinned Panda mesh. `make setup` creates the local `uv.lock` if absent and honors it with `--locked` thereafter. Artifacts default to `runs/`; data and checkpoints are local and ignored by git. Use separate `--run-dir` directories for independent experiments: rerunning a command replaces its corresponding outputs. Git tracks source history; versioned source archives are unnecessary.

`runs/validation-rollout.mp4` shows the arm and paper. Recordings use H.264 at CRF 18, compatible `yuv420p` color, and a front-loaded playback index (`faststart`). The 50 fps timebase preserves the controller's 20 ms boundaries: sampled views are held until the next simulation timestamp, with no motion interpolation. Views are sampled every five control steps (10 Hz), plus reset and the final state. A separate 100 ms hold displays the final state without extending simulated time. The PNG is final ink, NPZ stores named trajectory/force/joint columns, and JSON contains full metrics, timestamps, encoding settings and provenance.

`make demo CHARS=自在` prints per-character progress, simulated duration, ink error, peak force, elapsed time and output paths; detailed arrays stay in JSON. FFmpeg receives raw RGB frames directly, without temporary image files. Failed encoding does not replace an existing MP4, and missing FFmpeg is reported before a demo starts. Numerical training/evaluation do not require FFmpeg.

macOS uses native offscreen OpenGL. Linux may need an EGL driver with `MUJOCO_GL=egl make demo`, or OSMesa with `MUJOCO_GL=osmesa`; Linux rendering has not been tested on this host. No interactive viewer is launched.

## Learning and evaluation

```sh
uv run shodo train --episodes 56 --epochs 60 --seed 7
uv run shodo evaluate --chars 永水日山

# Frozen imitation controller plus learned residual correction.
uv run shodo ppo --residual --base-policy runs/bc.pt \
  --steps 1000000 --run-dir runs/residual-seed7 --seed 7
uv run shodo evaluate --policy ppo --run-dir runs/residual-seed7
uv run shodo validate --policy ppo --run-dir runs/residual-seed7
uv run shodo demo --policy ppo --run-dir runs/residual-seed7 --chars 永

# Omit --residual to train PPO from scratch.
uv run shodo ppo --steps 1000000 --run-dir runs/ppo-seed7 --seed 7
```

Training records seeds, configuration, package versions, source hashes, and source snapshots. Locks prevent concurrent trainers from writing one directory. PPO writes periodic atomic checkpoints. `--resume` continues the checkpoint/optimizer with matching configuration, objective, seed, training characters and residual base; it starts a new environment episode, not a bitwise replay of interrupted simulator/RNG state. Evaluation, demo and validation reuse the learned checkpoint's configuration with nominal materials by default; an explicit `--config` overrides it. For a resumed nondefault training run, supply the same original `--config` and residual base.

Default training characters are **一二三十木大人**, held out **永水日山**. `--chars` overrides the command's training/evaluation/demo set. Validation rejects training overlap. Held-out glyphs test control on new supplied paths, not recognition or autonomous glyph generation.

`ppo --ink-objective` optionally adds a loaded-ink accuracy bonus during drawing; it changes training rewards only. This separate experiment addresses the observed case where longer training improved air tracking but worsened actual ink accuracy. Its audit shows an ink/pressure tradeoff; it is not the default objective. Resume such a run with the same flag. See [PHYSICS.md](PHYSICS.md) for the exact reward and limitations.

### Compute device

BC/PPO training and learned-policy inference default to `--device cpu`: the small neural workloads measured on this Mac are faster on CPU than Apple Metal. `--device auto` opts into availability-based selection: CUDA first, then Apple Metal (`mps`), then CPU. Explicit `--device cpu`, `cuda` or `mps` selects that backend; an unavailable explicit accelerator raises an error instead of silently using CPU. Makefile commands expose the same choice through `DEVICE`:

```sh
make train DEVICE=cpu
make train DEVICE=auto
make evaluate DEVICE=cpu
uv run shodo ppo --device mps --steps 32768 --run-dir runs/metal-smoke
```

MuJoCo dynamics, brush contact and NumPy/SciPy ink transport remain on CPU. GPU execution applies only to neural tensor computation, not the whole simulator. Warm microbenchmarks on this host, with one PyTorch CPU thread, measured CPU/Metal single-observation inference at 16.90/378.32 µs and batch-256 Adam updates at 333.83/566.55 µs; these are component timings, not end-to-end training rates or CUDA results. Inference includes NumPy transfers; updates use resident batches. The report and reproducer are in `runs/device-smoke/`. Device selection is explicit so different hardware can be measured on its own merits.

Training and rollout metadata record the requested and resolved device. Checkpoints can be loaded on a different available device, and PPO resume permits changing devices; results across devices are not promised to be bitwise identical. Apple Metal BC training, residual PPO training and continuation of that PPO checkpoint on CPU have been exercised. The dated numerical comparisons in VALIDATION.md used CPU.

## Simulation contract

The pinned Apache-2.0 Menagerie Panda has no hand and carries a 40 g brush handle. Six Cartesian increments command XYZ and rotation-vector changes. Damped inverse kinematics feeds torque-limited inverse-dynamics joint servos. Only reset sets joint positions directly. Default control is 50 Hz, physics 500 Hz; action scales are 4 mm / 0.03 rad per control step.

The 40 float32 observations contain contact-center tracking error, command error, target preview, seven joint positions/velocities, brush force/deflection, target force, contact fraction, and tip height. Checkpoint observation contract is 3. Reward combines tracking, orientation, force and action effort.

The fast brush has 19 elastic/frictional bundles, persistent sticking/sliding contacts, pressure-dependent spread, and force/moment feedback into the robot. Ink deposits at **actual loaded contacts**, including accidental marks; it is never copied from the target or gated by the target's stroke state. Conservative water/mobile/fixed-pigment grids model spreading, adsorption and drying. Supply is continuously fed; dipping, finite reservoirs and paper handling are absent.

Reports separate handle-tip error from ink-center error: the handle intentionally offsets to compensate brush deformation. Visible ink coverage/spill complement tracking and contact metrics; missing ink is not successful drawing.

## Sensor policies, training data and robustness

The privileged 40-input benchmark remains available and existing checkpoints keep observation contract 3. Opt-in sensor policies use a separate `sensor-history` contract (version 1): 39 features per time slice, four slices by default. Inputs contain measured tool-pose tracking/command errors, reference preview, joint position/velocity, a world-frame force measurement, authored target force/drawing state, sample age and freshness. They exclude contact-center position, bristle deflection, contact fraction and ink state. History runs oldest to newest; reset repeats the first sample without importing a previous episode. The student is feedforward over this history, not recurrent.

`SensorEnv` adapts simulated pose/joints/contact force into those channels. Force is an ideal compensated contact-force proxy, not a simulated wrist transducer: sensor inertia, gravity compensation error and hardware filtering are not modeled. Pose noise is applied to a synthetic pose-estimator channel rather than recomputed from noisy joint encoders. Sensor age exposes delay/dropout; packets hold their acquisition timestamp. Reset bootstraps one current sample even with latency/dropout configured. Cameras are optional recorded data, not inputs to the current BC/PPO networks. This is a sensor-realistic interface experiment, not hardware readiness.

```sh
# Train a history-conditioned student from privileged teacher labels.
make train OBSERVATION=sensor RUN_DIR=runs/sensor-bc EPISODES=56 EPOCHS=60
make evaluate RUN_DIR=runs/sensor-bc

# Optional explicit synthetic noise/delay profile; separate from physical TOML settings.
make train SENSOR_CONFIG=experiments/sensors.json RUN_DIR=runs/sensor-noisy

# PPO and residual PPO also accept the sensor contract.
make residual OBSERVATION=sensor RUN_DIR=runs/sensor-residual \
  BASE_POLICY=runs/sensor-bc/bc.pt STEPS=32768

# Lossless training episodes, not demo videos. Use a new destination for each collection.
make record POLICY=classical RUN_DIR=runs/sensor-data CHARS=一永 EPISODES=2 CAMERA_EVERY=5
make record POLICY=learned RUN_DIR=runs/sensor-bc CHARS=永 EPISODES=1

# Paired classical/oracle/zero baselines, optionally with a learned sensor checkpoint.
make robustness POLICY=classical RUN_DIR=runs/sensor-baselines SEEDS="7 17 27"
make robustness RUN_DIR=runs/sensor-bc SEEDS="7 17 27"
```

In sensor mode `expert` means the measured-pose/normal-force classical baseline; `oracle` explicitly accesses privileged contact state and the true reference. BC uses privileged teacher actions as supervised labels but never includes that state in student observations. The classical controller uses proportional pose feedback and a specified 0.02 m/N normal-force correction; it is not claimed to be optimally tuned. It has a known native-brush force-limit failure; reduced-trained sensor BC also has poor native ink accuracy (see VALIDATION.md). Neither is a universally safe controller. Learned checkpoint loaders validate mode/history; PPO resume and residual bases additionally enforce identical training sensor settings. Evaluation can override corruption settings while retaining the input schema. Sensor checkpoints use `evaluate`/`robustness`; canonical `validate` remains a privileged benchmark. Custom sensor-policy callables declare a `sensor_config` attribute and may implement `reset()`; the episode runner calls reset once, then supplies copied observation arrays.

`contracts.py` defines the standalone sensor/reference schemas, feature encoding and action transform. `runtime.py` contains the synthetic sensor adapter and synchronous execution transaction. Requested actions are stored before clipping; applied actions are effective workspace-clipped Cartesian command increments, **not measured robot motion**. XYZ is world-frame meters; rotation uses additive world-axis rotation-vector coordinates with `R = Exp(rotvec) @ DOWN`, not body-frame twist increments. Sensor BC/PPO policies receive observation arrays; the optional SmolVLA actor additionally receives raw camera observations through its camera callback. Neither receives an environment object. This runtime does not provide a real-time watchdog, hardware driver or safety certification.

`record` writes one atomic, overwrite-protected `episodes/episode-000000.npz` per episode. Each contains `T+1` observations/decision timestamps and unnormalized `input_*` measured/reference/command channels, `T` requested/applied actions, rewards, termination/truncation flags, separately named privileged histories, embedded contracts/configuration/source and checkpoint provenance/attribution, and optional raw 640×480 perspective RGB frames with their own acquisition timestamps. Raw channels are sufficient to reconstruct the complete normalized policy history without privileged diagnostics. Frames have no diagnostic text, paper inset, interpolation or presentation holds; synthetic pinhole intrinsics and world-to-camera matrices accompany them. Decision time is not sensor acquisition time; both are recorded. Camera cadence is independent of policy observations. FFmpeg is unnecessary. Entire episodes are buffered in memory, so raw-camera collections should remain bounded; this is the native dataset format, not LeRobot compatibility.

```python
from shodo.dataset import load_episode

episode = load_episode("runs/sensor-data/episodes/episode-000000.npz")
batch = episode.window(start=0, length=16)  # 16 transitions and 17 observations
```

The robustness suite freezes named registration, tool-calibration, force-bias, noise, latency, dropout and combined cases. Paper XYZ/yaw cases change the **estimated reference frame**, not physical paper geometry. The same glyph/material seed is paired across cases/controllers; reports retain raw metrics, missing-ink/incomplete/truncation counts and each controller's difference from its own nominal result. Oracle results should be invariant to sensing-only changes. Default severities are engineering probes, not calibrated distributions or acceptance limits. Fixed cases and inspected glyphs are exploratory validation, not a blinded final test. Optional plant material cases are available through the Python API. Physical paper motion, camera corruption and hardware trials remain outside this implementation.

### Optional SmolVLA LoRA

[SmolVLA integration](integrations/smolvla/README.md) adds a pinned pretrained vision-language-action policy with adapter-only LoRA training over sensor state, raw causal camera frames, and effective Cartesian command increments. Run `make smolvla-setup`, then follow the recording, preparation, training, reload-verification, and held-out evaluation workflow there. Its separate locked environment and `VLA_DEVICE=auto` setting do not change ordinary BC/PPO dependencies or CPU defaults. The default 100-update run is a development check, not evidence of policy quality; inference is synchronous simulation, not real-time hardware control.

The guide also covers oracle recovery demonstrations (`make record POLICY=oracle EXPERT_NOISE=0.08`), separate one-step expert targets, opt-in inference caching/LoRA merging, and a causal replay latency audit. Keep explicitly marked training-character diagnostics separate from held-out evaluation reports.

## Native rods and settings

```sh
uv run shodo demo --policy expert --config experiments/quality.toml --chars 永
uv run shodo demo --policy expert --config experiments/cable.toml \
  --run-dir runs/native-demo --chars 永
uv run python -m shodo.mechanics
uv run python scripts/contact_sweep.py
```

TOML settings configure materials, bundle count, paper grid/transport and timestep. `experiments/quality.toml` uses 37 bundles and a 512² grid. The optional native MuJoCo cable backend adds rod inertia, bending/torsion and native contact. The curved, slow-touchdown preset passes four-glyph numerical and reduced-to-native BC transfer checks; completed refinement studies show remaining geometric sensitivity. Straight, axially rigid rods can generate touchdown spikes hidden by low-rate recordings. The audit measures microstep force peaks, penetration, and timestep/geometric sensitivity. A smooth recording and no solver warnings are not sufficient validation. Native simulation is substantially slower than the reduced model.

`free_hair_bundle()` preserves summed bending rigidity and cylindrical hair mass across bundle/segment counts. Its wet-hair material analogue is not calibration of a particular brush. [PHYSICS.md](PHYSICS.md) gives equations and primary references.

`experiments/cable-fast.toml` is an optional native speed/accuracy tradeoff: 2 mm reference curvature and 5 mm/s touchdown. Teacher and transferred BC pass the same 107 checks. On 永, the teacher takes 26.48 rather than 50.48 simulated seconds, but ink error rises from 2.14 to 2.31 mm and peak load from 0.44 to 0.66 N. It does not replace the slower reference preset.

`experiments/pressure.toml` separately tests stronger force feedback, a tighter pressure reward and ±50% material variation. It improves pressure robustness in the completed teacher/BC material grid; it does not silently change the default model:

```sh
make train CONFIG=experiments/pressure.toml RUN_DIR=runs/pressure \
  EPISODES=56 EPOCHS=60
make evaluate RUN_DIR=runs/pressure

# Optional pressure-focused residual, then a separate ink-bonus combination.
uv run shodo ppo --residual --base-policy runs/pressure/bc.pt \
  --config experiments/pressure.toml --steps 1000000 --seed 7 \
  --run-dir runs/pressure-residual
uv run shodo ppo --residual --base-policy runs/pressure/bc.pt \
  --config experiments/pressure.toml --steps 1000000 --seed 7 \
  --ink-objective --run-dir runs/pressure-ink-residual
```

The combination improves material-grid ink accuracy relative to pressure-only PPO, with similar mean force error but a worse maximum force error. Neither replaces the fixed default comparison; see the full tradeoff table in VALIDATION.md.

## Reproducing the longer audits

After training the named checkpoints above, these scripts reproduce the fixed geometry, numerical and material studies. The broad cohort and fine native rods can take hours on a CPU; they are not part of the quick test suite.

```sh
uv run python scripts/dataset_audit.py
uv run python scripts/reduced_audit.py
uv run python scripts/material_sweep.py
uv run python scripts/heldout_sweep.py --count 256
uv run python scripts/heldout_sweep.py --count 256 \
  --policies pressure-ink-residual --output runs/heldout-pressure-ink
uv run python scripts/material_sweep.py --policies pressure-ink-residual \
  --output runs/material-pressure-ink-residual
uv run python scripts/native_audit.py --chars 一永 \
  --cases dt-0.0001 bundles-7-segments-12 bundles-19-segments-6 bundles-19-segments-12
make benchmark CHARS=永
uv run python scripts/ik_benchmark.py
```

Run throughput measurements after competing jobs stop. The end-to-end benchmark includes warm teacher/control/dynamics/ink execution, excluding reset, loading and rendering. `--seed` selects the environment seed and defaults to 7. This teacher benchmark has no neural model, so `--device` does not move its work to GPU. The separate IK benchmark measures only equivalent prerequisite stages and Jacobians, not a whole-simulator speedup. See [VALIDATION.md](VALIDATION.md) for results and the exploratory status of additional pressure-controller comparisons.

## Data and attribution

[KanjiVG r20250816](https://github.com/KanjiVG/kanjivg/releases/tag/r20250816) supplies 6,702 SVGs with SHA-256 `69a2944ec1183086fdee5ba9c1f48bc306b867480a95b2f337f3203bf50689a3`. Ulrich Apel and contributors license it under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Original notices and provenance are retained. Preserve attribution and share-alike terms when distributing derived trajectories. The separate Panda asset license is retained beside its meshes.

KanjiVG supplies ordered schoolbook centerlines, **not measured pressure, tilt, timing or force**. Paths are resampled into a 180 mm square; lifts, pressure envelopes, tilt and timing are procedural. There is no learned visual/style objective.

## Code map

`data.py` prepares paths; `robot.py` constructs/drives the Panda; `brush.py` and `cable.py` implement brushes; `ink.py` transports water/pigment; `env.py` provides Gymnasium control; `learning.py` and `rl.py` train policies; `mechanics.py` and `validation.py` check physical/numerical and end-to-end behavior. Keep held-out baselines, test physical invariants, and profile before optimizing.
