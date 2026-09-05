# Shodo arm laboratory

Use Python 3.13 with uv. Makefile is the command entry point. Keep all work in this
directory; the parent is an unrelated multi-project repository. Never commit downloaded
data, virtual environments, or bulky training runs. Preserve source attribution in
derived datasets. Run `make check` and `make validate` after simulation changes.

Architecture: pinned KanjiVG SVGs -> arc-length sampled 3D brush trajectories ->
Gymnasium environment -> MuJoCo SCARA position servos -> actual-tip ink raster.
Learning uses Cartesian velocity commands through analytic inverse kinematics, not
direct pose teleportation. The articulated arm is simulated dynamically. The brush
is an approximate compliant ink deposition model, not a bristle or fluid solver.

Default train characters: 一二三十木大人. Held-out characters: 永水日山.
Never claim calligraphy mastery or sim-to-real readiness from trajectory tracking.
Validation must compare teacher, learned, and zero-action baselines on held-out paths,
check physical tracking and pen lifts, and render actual executed motion.
Online research may be delegated as requested by the user. Prefer official primary
sources and document limitations. Keep tests focused on integration and invariants.

Current validation and known approximations are recorded in VALIDATION.md. Canonical
outputs use `runs/validation-rollout.*`; `runs/initial-pass/` contains obsolete early
renders. Training seeds/settings, package versions and source hashes are saved with
artifacts. `make ppo` and `shodo evaluate --policy ppo` have also been exercised.
Avoid opening a GUI: default operation and rendering must remain headless.
