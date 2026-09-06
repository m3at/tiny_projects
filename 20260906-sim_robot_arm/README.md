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
