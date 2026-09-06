# Shodo arm laboratory

Use Python 3.13 with uv. Makefile is the command entry point. Keep all work in this
directory; the parent is an unrelated multi-project repository. Never commit downloaded
data, virtual environments, or bulky training runs. Preserve source attribution in
derived datasets. Run `make check` and `make validate` after simulation changes.

Architecture: pinned KanjiVG SVGs -> pressure/tilt/contact trajectories -> Gymnasium
environment -> torque-driven Menagerie Panda -> coupled brush -> conservative ink grid.
The fast brush uses elastic/frictional bundles; optional native rods use MuJoCo's cable
plugin. Six Cartesian pose increments go through damped IK and inverse dynamics.
See PHYSICS.md for equations, source attribution and limits. Never teleport outside reset.

Default train characters: 一二三十木大人. Held-out characters: 永水日山.
Never claim calligraphy mastery or sim-to-real readiness from trajectory tracking.
Validation must compare teacher, learned, and zero-action baselines on held-out paths,
check physical tracking and pen lifts, and render actual executed motion.
Online research may be delegated as requested by the user. Prefer official primary
sources and document limitations. Keep tests focused on integration and invariants.

Current validation and known approximations are recorded in VALIDATION.md. Canonical
outputs use `runs/v2/validation-rollout.*`; `runs/initial-pass/` contains obsolete early
renders. Training seeds/settings, package versions and source hashes are saved with
artifacts. `make ppo` and `shodo evaluate --policy ppo` have also been exercised.
Avoid opening a GUI: default operation and rendering must remain headless.

The extended v2 audit is complete; WORK_LOG.md preserves its chronological record.
VALIDATION.md separates the fixed comparison from exploratory controller extensions
and documents remaining physical limitations. New work lives in runs/v2; checkpoint
observation contract is version 3 because tracking includes brush contact offset.
Preserve old recordings and checkpoints in archives. Native cable checks include
beam response, mass, recovery, penetration, force and timestep/geometric sensitivity;
absence of solver warnings alone remains insufficient evidence of physical accuracy.
