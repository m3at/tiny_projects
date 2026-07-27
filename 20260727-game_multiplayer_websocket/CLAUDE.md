# Working notes

Broadside: a two-player hot-seat game. Players fit out an age-of-sail ship during a timed
build phase, then watch it fight itself. Five rounds, first to three. See GAME_DESIGN.md for
the design and the reasoning behind it.

No build step, no dependencies, no package.json. three.js is vendored in `vendor/`.

## Run and test

```
./tools/dev.sh              static server on 8123 + headless Chrome with CDP on 9222
./tools/dev.sh stop         tear both down -- always do this when finished
open http://127.0.0.1:8123/index.html
```

The game needs http for ES modules; opening the file directly will not work.

```
node tools/balance.js       archetype matchups per hull: win rates, length, decisiveness
node tools/match.js 40      full 5-round matches: economy, fill rates, sweeps, invariants
node tools/ablate.js        disables one mechanic at a time, reports what actually changes
node tools/events.js        confirms detonations / severings / dismastings fire
node tools/shot.js out.png "800 ;; ovBtn() ;; 400"    screenshot + console capture over CDP
```

Run the headless tools after any change to `sim/`, `config.js` or `data/`. They are fast
(seconds) and they catch balance and cross-round regressions that no amount of clicking will.

## Measure, do not guess

This is the working practice that matters most here. Every significant claim about this game
came from a tool, and several confident guesses were wrong:

- Wind looked like scenery. It is one of the four load-bearing mechanics (22% of outcomes).
- Wind *direction*, however, is scenery: sweeping it through 24 points barely moves a winner.
- The economy was inverted; ships got holier every round. Only a per-round fill-rate table
  showed it.
- A pre-fire grace period and a crew-affects-sail rule were doing nothing, or doing something
  invisible. Both removed after measuring.

Before removing or "simplifying" a mechanic, ablate it. Before optimising, measure.
`__game.perf.snapshot(__game.sceneCtl.renderer)` gives frame timings and draw calls.

## What is load-bearing

From `tools/ablate.js`. Do not casually change these four:

1. Shot passes through destroyed/empty cells. Disabling it makes 98% of battles time out.
2. Broadside arcs plus orbiting steering.
3. The wind's speed penalty (`WIND_MIN`).
4. The grape/round-shot toggle, the only live input.

Severing, magazine detonation and the magazine-required-to-fire rule are balance-neutral by
measurement. They are kept for drama or because they justify a part existing. That is a
deliberate decision, not an oversight.

## Architecture invariants

```
src/config.js      every gameplay and feel constant. Tools read the same file
src/theme.js       every colour. Keep PLAYER[] in step with --p1/--p2 in styles.css
src/match.js       scores, purses, hull progression, intel. Pure: no DOM, no three.js
src/sim/           deterministic battle core: rng, ship, battle
src/autobuild.js   greedy ship builder: bot opponent and the dev Fill button
src/dev.js         URL-driven dev harness, inert without ?dev
src/perf.js        rolling frame stats, always on
src/main.js        presentation and flow: phases, input, render loop
src/render/        three.js scene, ship meshes, particles, glyph textures
src/ui/            build-phase panel, HUD chrome
src/data/          parts, hull shapes (ASCII art)
tools/             headless harnesses and the CDP driver
```

Hold these:

- **`sim/` is pure and deterministic.** Seeded RNG only, no `Math.random`, no renderer or DOM
  imports, fixed 60Hz ticks, inputs applied between ticks. Same seed plus same input stream
  must produce the same battle; that property is what makes the WebSocket version a transport
  change rather than a rewrite. `autobuild.js` sits outside `sim/` for this reason.
- **Numbers live in `config.js`,** part statistics in `data/parts.js`, hull shapes in
  `data/hulls.js`, colours in `theme.js`. Do not scatter magic numbers into logic.
- **`render/shipView.js` `buildLayer()` is the asset seam.** One instanced layer per part
  type; swapping the box for a loaded mesh per type is the whole migration to real 3D.
- **`main.js` is the only file that knows about both the DOM and the simulation.**

## Rendering notes

Ships batch as one instanced deck plus one instanced layer per part type present. Per-instance
colour carries part identity *and* damage tint, so a single material serves every box on every
ship. Ship movement is a transform on the group, so instance matrices are only rewritten when
a cell's condition actually changes.

Particles are three instanced meshes. Everything lies flat on the water and blends additively,
which means the flat rotation bakes into the geometry (instance matrices are translate+scale
only) and opacity rides in `instanceColor`, since fading additively is the same as darkening.

Anything called per frame must be dirty-tracked. `hud.js` guards every DOM write; `scene.js`
only resizes the canvas and rebuilds the projection when something changed.

## Gotchas already paid for

- A `<canvas>` is a replaced element: `position: fixed; inset: 0` leaves it at its intrinsic
  300x150. State `width`/`height` explicitly.
- `THREE.Fog` plus an orthographic camera 300 units back flattens the scene to one colour.
- Chrome caches aggressively between tool runs. `tools/shot.js` disables it via CDP.
- `tools/shot.js` steps: a bare number waits ms, `@file.js` evaluates a file in the page
  (avoids fighting shell escaping). It navigates to about:blank when done, because a live
  WebGL page keeps a core busy.
- Headless rendering is software-rasterised and roughly 3x slower than real time. Do not use
  wall-clock waits to reach a simulation state; drive it directly
  (`for (...) __game.battle.advance(0.05)`), or the capture will race.
- Dev autoplay stops after one match on purpose. `&loop=1` opts back in. A forgotten looping
  tab once pinned a CPU core.
- In `ui/build.js`, `renderAll()` deliberately does not touch the hint text; callers own it.
  It used to, which silently ate every feedback message.

## Style

Plain modern JS, ES modules, 2-space indent, ~100 columns, semicolons. No TypeScript, no
framework, no bundler; keep it that way unless there is a reason.

Comments explain *why*, especially where a value was tuned or a mechanic was measured. Skip
comments that restate the code. Match the surrounding density.

Prose in docs and comments: plain, no bold or italics, minimal ceremony. Say what was
measured and what it showed.

## Open items

- Round 1 (sloop, 34 scrap, ~2 guns) is the least interesting round; the gun deck is simply
  the best pick and 31% of round-1 battles go to the bell on structure.
- ~45% of matches are 3-0 sweeps despite the comeback bonus. Persistent damage snowballs.
- An all-carronade ship loses at every scale in the bot's hands. Untested with mixed builds by
  a real player.
- No networking yet. The sim is ready for it; `main.js` is the part that changes.
