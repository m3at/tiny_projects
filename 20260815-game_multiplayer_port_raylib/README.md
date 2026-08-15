# Broadside

Broadside is a local two-to-four-ship age-of-sail autobattler. Human captains spend scrap fitting a
shared sequence of larger hulls, then choose round or grape shot while the ships fight autonomously.
Damage persists between rounds. The first captain to three wins; round five ends the match even if
there is no unique leader.

The game runs entirely in native C++20 and raylib. JavaScript is retained only as an external
behavioral and visual reference; it is not part of the runtime.

## Build and play

The desktop build fetches a pinned raylib revision:

```sh
make build
make run
```

The menu chooses total ships and how many of them are bots, while retaining at least one human
captain. The command line also exposes unattended spectator matches for visual testing.

Command-line configuration skips the menu:

```sh
make run ARGS="--players 4 --bots 3 --seed 0xb0ad51de --speed 2"
```

- `--players N` sets total ships, not human players.
- `--bots N` replaces the last N seats with bots and may equal the total ship count on the CLI.
- `--bots-only` is shorthand for making every configured seat a bot.
- `--loop` waits eight seconds on a bots-only match verdict, then starts another match.
- `--seed N` makes the match reproducible.
- `--speed N` scales local authority time from 0.25× to 8×.
- `--width N --height N` sets the initial window size, with a 640×360 minimum.

For a complete visual four-ship game with no input required:

```sh
make spectate
```

Override the defaults with `PLAYERS=2 SPEED=4 make spectate`, or use the equivalent raw form:

```sh
make run ARGS="--players 4 --bots-only --speed 2"
```

For a continuous visual soak test:

```sh
make soak
```

The match runs through all shipyards, battles, and round results and stops on the match verdict unless
`--loop` is present. `--scenario duel` and `--scenario four-way` remain deterministic staged-screen
fixtures for captures; they do not run complete matches.

## Controls

Shipyard:

- Click an offer, then left-click an empty hull cell to place it.
- Press `1`–`5` to select offer cards without a mouse.
- Right-click a fitted cell to remove it, or press `X` to toggle explicit break-up mode.
- `R` rerolls, `F` refits, and `G` auto-fits through the normal typed command path.
- Space locks the active build. A private handoff screen separates consecutive human captains.

Battle:

- `A`, `L`, `Q`, and `P` toggle ammunition for human seats one through four.
- Each human captain also has Round and Grape controls in their own status panel.
- `M` mutes sound, `F3` shows diagnostics, `F11` toggles fullscreen, and Escape returns to the menu.

Results:

- Enter continues to the next round or starts a rematch.

## Fast verification

A headless build has no raylib or window-system dependency:

```sh
cmake --preset headless
cmake --build --preset headless
ctest --test-dir build/headless --output-on-failure
```

Useful entry points:

| Command | Purpose |
| --- | --- |
| `make spectate` | Watch one complete four-bot match at 2× speed. |
| `make soak` | Repeat complete bots-only matches for visual/performance testing. |
| `make quick` | Build headless and run the small `quick`-labelled tests. |
| `make test` | Build the configured desktop tree and run every CTest suite. |
| `make reference` | Generate and compare all 900 reference battles. |
| `make netcheck` | Run complete matches over immediate and virtual transports, including repair and reconnect. |
| `make audio-check` | Render isolated cues and dense mixes and enforce the audio envelope. |
| `make playthrough` | Cover a two-human match and an unattended four-bot match. |
| `make melee` | Measure three- and four-ship decisiveness and seat fairness. |
| `make ablate` | Compare canonical targeting with diagnostic rule variants. |
| `make rendercheck` | Check responsive hitbox bounds and adaptive-quality transitions. |
| `make capture` | Capture six deterministic scenes at 1280×720 and 960×540. |
| `make sanitizer` | Run the headless suite under ASan and UBSan. |
| `make format-check` | Verify native C++ formatting. |

`make capture` writes to `build/captures` by default and verifies the stored same-backend perceptual
hashes. Set `VERIFY_BASELINES=0` when establishing captures on a different graphics backend.

The desktop fixture scenarios are:

- `menu`: unopened local menu.
- `shipyard`: the first active shipyard turn.
- `duel`: a staged two-ship battle, advanced six seconds.
- `four-way`: a staged four-ship battle, advanced six seconds.
- `result`: the first completed round result.
- `match-end`: a precomputed match-complete result.

To leave a staged four-way battle open for inspection, the earlier command remains valid:

```sh
make run ARGS="--scenario four-way --speed 2"
```

For one image:

```sh
make capture-one SCENARIO=duel CAPTURE=build/duel.png ARGS="--width 1280 --height 720"
```

## Architecture

- `native/sim` is deterministic simulation: seeded RNG, fixed 60 Hz ticks, float32-collapsed
  trigonometry, ships, projectiles, damage, effects, results, replay, and checksums. It has no
  rendering, sockets, filesystem, or wall-clock dependencies.
- `native/game` owns shipyard rules, autobuilders, ammunition bots, economy, and match progression.
- `native/net` owns typed commands/messages, connection-based authorization, the authority, client
  replicas, and immediate or deterministic virtual transports.
- `native/presentation` owns testable layout, controller, and adaptive-quality policy.
- `native/render` and the desktop half of `native/audio` own raylib resources and presentation.
- `native/tools/main.cpp` is the headless diagnostics executable.

The desktop sends actions through `GameClient` and the transport boundary. It does not call
`RoomAuthority` or mutate an authoritative battle directly.

## Reference and presentation

`native/reference/js_golden.txt` contains 900 semantic reference battles. `broadside_tool reference`
rebuilds every native archetype and requires every winner to match, finish time within one tick, and
each final structure fraction within 0.001. Native repeatability uses the strict checksum fixture in
`native/reference/native_golden.txt`.

The native presentation keeps the reference's illustrated chart-table direction while taking
advantage of the desktop renderer. Ships use a stable tapered under-deck, inset deck plates, an
unambiguous velocity-side prow and bowsprit, aft flags, damage holes, and low-relief role-specific
part marks. The sea is a world-mapped directional wave field with etched lines, foam, an arena
boundary, up to 1.5× supersampling, adaptive resolution, multisampled geometry, and FXAA.

UI is rendered at native resolution over the scaled world. Small text uses filtered Inter with a
12.5-pixel raster floor; the decorative face is reserved for headings large enough to remain legible.
One chart-like visual grammar covers the menu, private captain handoff, shipyard, battle status, and
results. Checksums, render scale, and pool counters stay behind F3 instead of competing with play
state. Implementation and review gates are in [PRESENTATION_PLAN.md](PRESENTATION_PLAN.md).

## Networking status

Local play already exercises the protocol boundary intended for online play. The virtual wire covers
latency, jitter, reordering, disconnect, checksum repair, and replay-based reconnect. Real ENet packet
serialization, lobby hosting, NAT traversal, and online smoke tests remain deferred; the incomplete
adapter is disabled by default and is not used by the desktop runtime.
