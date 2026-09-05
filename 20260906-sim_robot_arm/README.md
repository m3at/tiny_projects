# Shodo arm laboratory

A headless Python 3.13 / MuJoCo environment for learning to move a brush along
Japanese stroke trajectories. Includes downloaded KanjiVG data, a Gymnasium API,
an analytic teacher, behavior cloning, PPO, quantitative evaluation, and offscreen
animation of the physical arm alongside its actual ink trace.

The environment has been trained and validated locally; see [VALIDATION.md](VALIDATION.md)
for measured tracking accuracy, performance, checked commands and limitations.

## Run

Install [uv](https://docs.astral.sh/uv/), then:

```sh
make setup
make data          # checksum-verified, pinned KanjiVG release (~12 MB)
make check
make train         # collect teacher demonstrations and train CPU MLP
make validate      # held-out baselines + API + offscreen rendering gates
make demo CHARS=永水
```

`runs/validation-rollout.gif` shows the arm and paper; `runs/validation-rollout.png` is the
final ink. Evaluation metrics are in `runs/evaluation.json`. Checkpoints and data
stay local and are ignored by git. First trajectory preparation is cached in memory
and takes longer than subsequent episodes. Training needs no display or GPU.
Offscreen rendering uses MuJoCo's OpenGL backend; Linux servers may require
`MUJOCO_GL=egl make demo` and an EGL driver, or `MUJOCO_GL=osmesa` with OSMesa.
macOS uses its native offscreen OpenGL context. No interactive viewer is launched.

For reinforcement learning experiments:

```sh
make ppo STEPS=32768
uv run shodo evaluate --policy ppo
uv run shodo demo --policy ppo --chars 永
```

PPO starts from scratch; this budget is a pipeline exercise, not a quality guarantee.
The supplied behavior-cloning baseline is the fast route to a working controller.
`uv run shodo demo --policy expert --chars 永` replays the analytic teacher.

Keep separate experiments with `--run-dir runs/my-experiment`. For example,
`uv run shodo train --run-dir runs/seed42 --seed 42 --episodes 56 --epochs 50`, then
`uv run shodo evaluate --run-dir runs/seed42`. Every training run records its seed,
dataset identity, package versions and source hashes. Rollout NPZ files contain
actual/target XYZ, joint positions and stroke IDs, with a `columns` array naming fields.
Rollout JSON also retains KanjiVG attribution. Reusing a run directory replaces its
artifacts; use a new directory for a comparison you want to preserve.

## Environment contract

The arm is a purpose-built 3-DOF SCARA: two horizontal revolute joints (270 and
250 mm links), plus a vertical brush slide. The brush is attached directly; there
is no hand, ink dipping, or paper handling. Its orientation is fixed vertical.
Position actuators have finite stiffness, damping and force limits; MuJoCo integrates
joint dynamics at 500 Hz. A controller step integrates ten physics steps (50 Hz).
Inverse kinematics drives actuator targets; only reset sets the initial joint pose.

Actions are three Cartesian increments in [-1, 1], each scaled to 4 mm per control
step, then clipped to the reachable paper workspace. This is a high-level tracking
task with an existing joint servo, not raw torque learning. The 12 float32 observations
are target-minus-tip / 4 mm, command-minus-tip / 4 mm, three joint velocities / 5,
and tip position relative to (0.32, 0, 0) / 0.1 m. The target is supplied by the
stroke planner; the policy does not invent glyphs from an image or character code.
Reward is `exp(-(3D_error / 8mm)^2) - 0.002 * sum(action^2)`.
Episodes end after the complete reference trajectory; numerical instability truncates.
`reset(seed=..., options={"char": "永"})` selects a reproducible glyph and cleans paper.

SVG paths are sampled by arc length, transformed from 109×109 coordinates onto a
180 mm square, with Y flipped into world coordinates. Explicit lifts to 25 mm
separate strokes. Drawing speed is at most 40 mm/s, air travel at most 60 mm/s.
Stroke order and direction come from upstream SVG document order. A -1 mm drawing
height and height-dependent circular footprint approximate brush compression. Ink
is deposited from the **actual simulated tip** each physics step, including accidental
marks; it is never copied from the target or gated by the target's pen state.

The nib and paper have no hard contact constraint: negative height represents soft
brush compression. This deliberate low-cost model does not simulate bristles,
contact force, ink fluid flow, absorption, brush tilt, or paper friction. It is useful
for trajectory/pen-lift controller experiments, not validated sim-to-real shodo.

## Data and attribution

`make data` downloads the [KanjiVG r20250816 main release](https://github.com/KanjiVG/kanjivg/releases/tag/r20250816)
with SHA-256 `69a2944ec1183086fdee5ba9c1f48bc306b867480a95b2f337f3203bf50689a3`.
The full main collection is extracted locally, with provenance JSON and original
SVG notices. KanjiVG is by **Ulrich Apel and contributors**, licensed
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Preserve attribution,
identify resampling/coordinate/height modifications, and apply share-alike terms
when distributing derived trajectory datasets. This dataset license is separate
from the parent repository's code license.

KanjiVG provides schoolbook centerlines and stroke order, **not expert brush motion,
pressure, tilt or timing**. Those quantities in this project are synthetic. See the
[official format](https://kanjivg.tagaini.net/svg-format.html) and
[download documentation](https://kanjivg.tagaini.net/files.html).

Default training glyphs are 一二三十木大人; evaluation holds out 永水日山. Holding out
glyphs tests a target-conditioned controller on new paths, not character recognition.
The collection includes other characters; use any available Unicode glyph with `--chars`.

Further research: [AnimCJK](https://github.com/parsimonhi/animCJK) adds Japanese glyph
outlines and medians, potentially useful as raster targets; its assets have mixed
Arphic/LGPL licensing and are not bundled here. [Wang et al.](https://arxiv.org/abs/1911.08002)
study dynamic brush models, while [Jia and Manocha](https://arxiv.org/abs/2309.08457)
use behavior cloning and reinforcement learning for brush manipulation. These are
extension directions, not evidence that this simplified simulator transfers to hardware.

## Extend

`src/shodo/data.py` owns acquisition and trajectory generation; `arm.xml` owns
mechanics; `env.py` owns control, observation/reward and ink; `learning.py` owns the
baseline; `cli.py` owns commands. Start with trajectory speed, brush depth, or reward
experiments. For pressure learning, add a calibrated compliant contact model and
force observations. For style learning, obtain licensed brush demonstrations or
silhouette targets and replace the target-conditioned objective. Keep held-out glyphs
separate and compare against both teacher and stationary baselines.
