# Engineering notes

## Working model

The headless laboratory couples a torque-driven seven-joint Panda to either elastic/frictional bristle bundles or native MuJoCo cable rods, with conservative water/mobile/fixed-pigment transport. Policies control six Cartesian pose increments. The privileged benchmark uses 40 observations (contract 3); opt-in sensor policies use histories of 39 measured/reference features (sensor contract 1). Only reset sets joint positions directly.

## Evidence and reproduction

[VALIDATION.md](VALIDATION.md) holds the dated numerical evidence, protocols, performance measurements and controller tradeoffs. [PHYSICS.md](PHYSICS.md) holds equations, source attribution and physical limitations. [README.md](README.md) holds setup and reproduction commands. Keep measurements in those documents rather than duplicating tables or process transcripts here.

The completed audit covers 5,460 broader-cohort episodes, 800 plant-only material perturbations, multi-seed training, native beam/contact sensitivity and reduced ink/timestep checks. The fixed four-controller comparison and exploratory pressure/ink objectives must remain distinct. Glyph-bootstrap intervals are conditional on the fixed checkpoint and three material draws, not training or hardware uncertainty.

## Engineering priorities

- Keep world coordinates, brush contacts, ink raster, scoring and 3D texture placement consistent. Qualitative recordings must show actual executed deposition, including unintended marks.
- Keep source and documentation focused on the current model. Git provides source history; `runs/` contains descriptive experiment outputs, not versioned source trees.
- Run `make check` and `make validate` after simulation changes. Add focused invariant or integration tests for corrections; numerical agreement alone cannot validate visual representation.
- Record configuration, seeds, dependencies and source/checkpoint hashes with scientific results. Do not imply that a changed renderer or controller has rerun the broad numerical study unless it has.
- Keep sensor actors separate from privileged teacher/evaluation channels. Record aligned policy episodes independently of demonstration movies, and use paired calibration/sensing stress cases without implying measured hardware distributions.
- Default to headless operation. Measure throughput without competing jobs and distinguish component microbenchmarks from whole-simulator performance.
- Keep SmolVLA's optional locked stack independent of the small-policy baseline. Reuse native recordings for causal, memory-mapped training inputs; keep counterfactual recovery targets separate from executed actions, preserve warm-start split lineage and adapter reload checks, and distinguish recorded-state accuracy, closed-loop drawing quality and real-time deadlines.

## Physical limits

The brush, ink, paper and robot setup are uncalibrated. Reduced bundles omit distributed inertia; native rods omit inter-hair locking and capillary clumping. Ink supply is continuous, and pressure, tilt and timing are procedural. Better tracking does not establish calligraphy mastery, aesthetic quality or hardware readiness.
