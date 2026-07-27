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
node tools/watch.js         is a battle worth watching: empty air, dead stretches, orbiting
node tools/balance.js       matchups per hull, graded, worst pairing named
node tools/parts.js         per part across hundreds of random builds: dominant, even, or a trap
node tools/tune.js          sweep one constant; prints watchability and fairness side by side
node tools/match.js 40      full 5-round matches: economy, fill rates, sweeps, invariants
node tools/ablate.js        disables one mechanic at a time, reports what actually changes
node tools/events.js        confirms detonations / severings / dismastings fire
node tools/bench.js         simulation throughput; every other tool is built out of this number
node tools/shot.js out.png "800 ;; ovBtn() ;; 400" "?dev=brawler,crusher&round=5"
```

Run the headless tools after any change to `sim/`, `config.js` or `data/`. They take seconds and
they catch regressions no amount of clicking will.

Which tool answers which question:

- "Is this fun to watch?" -> `watch.js`. Empty air, longest dead stretch, whether the pair
  actually orbits. This is the one the last gameplay pass was aimed at.
- "What should this number be?" -> `tune.js`. Sweeps a constant over a grid.
- "Does this mechanic matter at all?" -> `ablate.js`. Deletes it and counts winner flips.
- "Is any part a no-brainer or a trap?" -> `parts.js`.
- "Is any matchup hopeless?" -> `balance.js`, and read the worst cell, not the average.

## Measure, do not guess

This is the working practice that matters most here. Every significant claim about this game
came from a tool, and several confident guesses were wrong:

- Wind looked like scenery. It is load-bearing, though less so than it was (8% of outcomes now).
- Wind *direction*, however, is scenery: sweeping it through 24 points barely moves a winner.
- The economy was inverted; ships got holier every round. Only a per-round fill-rate table
  showed it.
- A pre-fire grace period and a crew-affects-sail rule were doing nothing, or doing something
  invisible. Both removed after measuring.
- "Ships fire one volley then sail to the edge" was reported as a feel problem and turned out to
  be a one-line steering bug: the two ships orbited in opposite senses, which is a parallel
  course. They held their range perfectly and marched off the map together.
- Battles were assumed to be too long. They were too *empty*: 86% of the time nothing was in the
  air. At 13-17s they are at the low end of the 15-25s that shipped autobattlers use.
- Sweeps were assumed to prove a snowball. 17% of matches are 3-0, and pure chance produces 25%
  in a first-to-three, so there was nothing to fix.
- A carronade ship with twice the raw damage of its opponent lost 100% of battles. Not a stats
  problem: a single grape volley was killing its entire crew three seconds in.

Two habits worth keeping. First, measure *feel* separately from *fairness* — giving both ships
the same orbit sense flips only 26% of winners, so `ablate.js` calls it middling, while
`watch.js` shows it taking empty air from 86% to 30%. Second, when a number looks absurd, trace
one battle before touching the part table; the last four balance bugs were all mechanism bugs
wearing a balance costume. `/tmp` throwaway scripts driving `createBattle` directly are the
fastest way in.

Before removing or "simplifying" a mechanic, ablate it. Before optimising, measure.
`__game.perf.snapshot(__game.sceneCtl.renderer)` gives frame timings and draw calls.

## What is load-bearing

From `tools/ablate.js`, in order. Do not casually change these:

1. Broadside arcs plus orbiting steering (58% of winners flip without it).
2. Shot passes through destroyed/empty cells (38%, and decisive endings collapse to 13%).
3. Both ships orbiting the *same* sense (26%). Opposite senses is the parallel-course bug.
4. Broadsides firing out their own flank only (23%). Either-beam is better to watch and worse
   as a game: it deletes the decision of where the battery goes.
5. Ships not running when crowded, `ORBIT_RETREAT = 0` (20%). Any higher and kiting returns.
6. `HULL_DAMAGE`, the per-hull damage pacing (14%). Without it round 5 is a nine-second coin
   flip and round 1 drags.
7. `GRAPE_CREW_SCALE` (12%) and the grape/round toggle itself (15%), the only live input.

Severing, magazine detonation, the magazine-required-to-fire rule and heavy-timber soak are all
balance-neutral by measurement (0-1%). They are kept for drama or because they justify a part
existing. That is a deliberate decision, not an oversight.

A note on `ORBIT_SENSE`: which beam the fight turns to is drawn per battle from the seeded rng
and deliberately not shown during the build phase. Any predictable geometry gives a sheltered
flank to hide the crew and powder behind, and a build that did so won 100% of 800 battles at
every hull size. `autobuild.js` keeps a `massed` archetype to hold that door shut — it should
measure near 50% against `brawler`, and if it ever climbs, that rule has been undone.

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
tools/lib.js       shared harness code: measureBattle, playBattle, the bot's ammo choice
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
- **The sim's hot path is performance-sensitive on purpose.** Every question here is answered by
  running thousands of battles, so throughput compounds across a session; `tools/bench.js`
  reports it, currently ~1600 battles/sec. The wins already taken: an integer cell grid
  (`gridIndex`) instead of `"dx,dz"` string keys, a maintained `aliveCells` count instead of
  filtering arrays per tick, in-place projectile compaction, trigonometry hoisted out of the gun
  loop, and `len()` instead of `Math.hypot`. Do not reintroduce per-tick allocation.
- **Whoever consumes `battle.effects` must drain it.** Nothing in `sim/` clears it.

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
- `tools/shot.js` takes the dev query as its *third* argument. It used to only read a `URL`
  environment variable, so a query passed as an argument was silently ignored and every "why is
  autoplay not running" session started there.
- The bot's ammunition heuristic in `tools/lib.js` keys off gun damage numbers. Rescaling the
  part table silently turned it all-grape, which made a whole archetype measure as hopeless.
  Re-check it after touching damage values.
- `tools/ablate.js` and `tools/tune.js` patch source by exact string match and throw when the
  target is missing. That is deliberate, but it means editing `config.js` or `parts.js` breaks
  them loudly; fix the patch strings in the same commit.

## Style

Plain modern JS, ES modules, 2-space indent, ~100 columns, semicolons. No TypeScript, no
framework, no bundler; keep it that way unless there is a reason.

Comments explain *why*, especially where a value was tuned or a mechanic was measured. Skip
comments that restate the code. Match the surrounding density.

Prose in docs and comments: plain, no bold or italics, minimal ceremony. Say what was
measured and what it showed.

## Balance, and how much of it to want

Perfect balance is not the goal and is not achievable here; the absence of cheats, no-brainers
and boring dominant plays is. Working numbers, from published practice:

- A matchup at 6-4 is fine and 7-3 is a counterpick; 8-2 is near unwinnable. Grade the *worst*
  cell — averaging win rates hides everything, since a roster where every matchup is 8-2 still
  averages 50%. `balance.js` prints the count past 7-3 and names the worst.
- A part whose "more of it wins" rate sits inside 47-53% is neutral; past 60% it wins games on
  its own; under 40% it is a trap. Both extremes are bugs — a part nobody takes is as broken as
  one everybody must. `parts.js` prints this.
- Distrust extremes that come from the bots. `autobuild.js` is greedy and builds *pure* ships,
  which the draft can never offer (five part types out of nine per round). Its pure-build grid
  comes out bimodal — near 50% or near 100%, little between — because damage compounds here.
  Check `parts.js` before nerfing a part on the strength of one lopsided pairing.
- A different best gun per hull size is good design, not a failure. What matters is that each
  round offers more than one defensible pick.

## Open items

- Several pure-build matchups are still past 7-3, mostly the carronade at large hull sizes and
  the swivel at small ones. See the caveats above before acting.
- Long guns are hard to fit. Restricting them to bow cells was right — a gun that always bears
  cannot also be spammable — but few builds carry any, so `parts.js` cannot read them.
- Nothing in the interface explains why a battle was lost. The genre answer is a post-battle
  per-part damage summary; a log says what happened, a summary says which decision was wrong.
- The random engaged beam is only a fair gamble if the player is told the odds. They are not
  told at all.
- No networking yet. The sim is ready for it; `main.js` is the part that changes.
