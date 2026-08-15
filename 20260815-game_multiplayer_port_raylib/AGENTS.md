# Broadside working notes

Broadside is a C++20 raylib desktop game. Keep dependencies pointed in one direction:

- `native/sim`: deterministic simulation only;
- `native/game`: shipyard, bots, economy, and match rules;
- `native/net`: typed commands/messages, authority, transports, and client replay;
- `native/presentation`: controller, layout, and adaptive-quality policy;
- `native/render`, desktop audio, and `native/main.cpp`: raylib presentation.

Simulation and gameplay code must not depend on rendering, sockets, filesystem, or wall-clock APIs.
The desktop must send commands through `GameClient`; it must not call `RoomAuthority` or mutate an
authoritative `Battle`.

## Required checks

After gameplay, authority, or replay changes:

```sh
cmake --preset headless
cmake --build --preset headless
ctest --test-dir build/headless --output-on-failure
./build/headless/bin/broadside_tool bench 1000
./build/headless/bin/broadside_tool golden
./build/headless/bin/broadside_tool reference
./build/headless/bin/broadside_tool netcheck
```

Run `make sanitizer` for memory-sensitive changes. After presentation changes, also run:

```sh
make format-check
make rendercheck
make audio-check
make capture
```

`make capture` compares perceptual hashes recorded on the current reference backend. Use
`VERIFY_BASELINES=0 make capture` to establish captures on a different backend, then inspect every
scene rather than treating a new hash as proof of correctness.

## Determinism

The simulation uses seeded randomness, fixed 60 Hz ticks, double-precision state, float32-collapsed
`sin`, `cos`, and `atan2`, stamped ammunition inputs, and strict native checksums. Preserve operation
ordering and disabled fused contraction. Broadside arcs, shared orbit sense, holes, stable manning,
hull damage pacing, magazine chains, and the melee target margin are load-bearing.

`native/reference/js_golden.txt` contains 900 semantic reference battles. Native generation must
match every winner, finish within one tick, and match final structure within 0.001. Before changing a
protected mechanic, measure the current behavior and update fixtures only when the new behavior is
intentional and explained.

## Local and network play

`--players` means total ships. The menu retains a human seat, but CLI `--bots-only` and
`--bots == --players` run unattended spectator matches; `--loop` repeats them after an eight-second
verdict hold. Desktop `--scenario` modes stage deterministic capture screens and are not full matches.

The authority owns validation, offers, purses, timers, phase changes, and verdicts. Immediate and
virtual transports exercise the same typed protocol. Real ENet serialization, lobby hosting, NAT
handling, and online smoke tests remain deferred; do not route around the transport boundary while
they are incomplete.

## Documentation

Explain why a boundary or protected rule exists before describing its mechanics. Describe the
current system directly and distinguish shipped behavior from planned work. Keep user commands in
`README.md`, mechanics in `GAME_DESIGN.md`, and presentation work in `PRESENTATION_PLAN.md`.

## Graphics work

The renderer uses reference-derived visual grammar with native MSAA/FXAA, world-mapped directional
water, constructed deck cells, fixed battle pools, native-resolution UI, and twelve deterministic
captures. Preserve the shared theme and screen hierarchy when extending it. Do not accept a feature
checklist as visual proof; inspect every changed capture at both acceptance resolutions and keep
canonical particle overflow at zero.
