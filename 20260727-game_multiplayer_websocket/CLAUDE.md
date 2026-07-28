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
node tools/audio.js         renders every sound offline: clipping, DC offset, onset clicks
node tools/mix.js           what the mixer hears across real battles: drops, bursts, peak level
node tools/frames.js        frame times as a distribution, per phase; stutter lives in the tail
node tools/profile.js       browser CPU profile of the real page, software rasterising excluded
node tools/fill.js          what each layer of the scene costs to draw, and how it scales
node tools/playtest.js      plays a whole match through the real interface and complains
node tools/golden.js        fingerprint 900 battles; diff it across a refactor
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
- "Why does it hitch?" -> `frames.js` first, for which phase and how bad the tail is, then
  `profile.js` to name the function. Read the js line, not the wall line: headless wall times swing
  by 2x between identical runs, and the JavaScript numbers do not.
- "Can you actually hear the battle?" -> `mix.js`. Sounds offered against sounds played.
- "What is expensive to draw?" -> `fill.js`. Prices one layer at a time by hiding it. Its noise
  floor is a few ms at 1080p, so trust the big numbers and treat anything near zero -- or negative,
  which happens -- as unmeasured rather than free.
- "Did I break the game?" -> `playtest.js`. Every other tool drives the simulation; this one drives
  the interface, so it is the only thing that would notice if locking in stopped working. It also
  fails the run on any console output, which is how the suspended-context audio bug surfaced.

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
- A ship with crew for only half its guns could put every one of those hands on the flank the fight
  was not on, and spend the whole battle unable to fire. Found by hand-building exactly that ship
  in a probe and watching it do nothing for forty seconds; the fix is that hands go to the battery
  that will bear (`createBattle` sorts the guns before manning them).
- Masts have always cost crew in the build readout and never cost it in the battle, so ships fought
  with more guns manned than the panel said they could. Now the sails take their hands first.
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
src/autobuild.js   greedy ship builder: bot opponent and the dev Fill button
src/dev.js         URL-driven dev harness, inert without ?dev
src/perf.js        frame stats, always on: an EMA for scale, percentiles for the tail
src/main.js        presentation and flow: phases, input, render loop
src/render/        three.js scene, ship meshes, particles, glyph textures
  sea.js           the sea, the wind and the arena ring, as one full-screen triangle
  quality.js       the adaptive resolution control loop, kept apart from the scene
src/audio/sfx.js   every sound, synthesised. Takes a context, so it renders offline for testing
src/audio/play.js  when to make a noise: voice spacing, stereo placement, mute, the gesture unlock
src/ui/            build-phase panel, HUD chrome
src/data/          parts, hull shapes (ASCII art)

src/sim/           the deterministic battle core, one file per question
  rng.js           seeded generator
  geometry.js      len, wrapAngle, ship-local to world
  ship.js          the persistent design, and the runtime state of one ship in one battle
  steering.js      how the two ships sail and hold station
  gunnery.js       firing, shot in flight, and where a ball lands
  damage.js        what a ball does on arrival: structure, crew, magazines, severing
  battle.js        the clock over all of it, and how a round ends

tools/             headless harnesses and the CDP driver
  cdp.js           one DevTools client for every tool that drives the real browser
  harness.js       playBattle (outcomes) and measureBattle (what it looked like)
  bot.js           the stand-in player: the ammunition decision, and its reaction interval
  variant.js       run the sim with src/ patched in a temp dir, for ablate.js and tune.js
  golden.js        determinism guard; record before refactoring, diff after
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
  running thousands of battles, so throughput compounds across a session. `tools/bench.js` reports
  it: ~3200 battles/sec, about 45,000 simulated seconds per real second. Do not reintroduce
  per-tick allocation or per-tick trigonometry.
- **Refactor against `tools/golden.js`.** Record it, change the code, diff it. Anything meant to be
  structural must leave it byte-identical. It has already caught one refactor that changed
  behaviour and one that did not but looked like it might.
- **Whoever consumes `battle.effects` must drain it.** Nothing in `sim/` clears it.

## What made the simulation fast

From 1131 to ~3200 battles/sec, all of it from the profiler (`node --cpu-prof`) and none of it from
guessing. Worth knowing which lever is which, because the shape repeats:

- The cell grid is a *dense* array of nulls, not a sparse one. This was the single largest win, and
  the least obvious: a JavaScript array left full of holes can fall out of V8's fast element kinds,
  and this is the busiest read in the game. Same story for indexed `for` loops over `guns`/`cells`
  in the three hottest functions.
- Reload is an absolute deadline (`gun.readyAt`), not a countdown. A countdown means writing to
  every gun on every tick whether or not anything is happening.
- Range and arc tests compare squared lengths, so no square root, and the arc test is a dot
  product against a precomputed cosine rather than an `atan2` and an angle wrap.
- A ball outside the hull and travelling away from it is dropped immediately: it can never hit,
  since every shot flies faster than any ship sails. It also splashes alongside instead of off in
  the distance, which looks better.
- Heading sine and cosine are cached on the ship per tick; `refreshSystems` maintains `mass`,
  `sail` and `canFire` so `steer` and `checkEnd` do not recompute them.
- `severDisconnected` marks reachability with a per-ship stamp instead of a `Set` of strings, and
  skips the flood fill when the lost cell had at most one live neighbour -- but only after the
  first full sweep, because a build can start out with a section not joined to the helm.

Two things measured as *not* worth doing, which is equally useful to know: pooling the `effects`
objects (no measurable cost at all) and shortening projectile lifetimes (already handled by the
receding-ball test).

## Audio notes

Synthesised, no files: `src/audio/sfx.js` builds everything from oscillators and one shared buffer of
white noise. It takes an `AudioContext` rather than creating one, which is what lets `tools/audio.js`
render each sound through an `OfflineAudioContext` and measure it — offline contexts are exempt from
the autoplay gesture rule, so this works headlessly.

Run `node tools/audio.js` after touching a sound. It catches the three faults that are invisible to
clicking around: samples over 1.0, a DC offset, and a step at onset. All three have already been
caught this way.

Rules worth not relearning:

- Envelopes anchor at zero, ramp up over ~2ms, decay exponentially to a whisker above zero, then hard
  zero. `exponentialRampToValueAtTime(0)` throws; ramping *from* zero silently becomes a step.
  An instant gain step on noise measures a 0.8 jump at onset and clicks audibly.
- Noise must be bipolar. `Math.random()` alone is a 0.2 DC offset that thumps on every start and stop.
- Voices are fire-and-forget. A cannon costs ~50us to build; pooling gain nodes buys nothing and
  reintroduces clicks from stale automation. What is long-lived is the context, the noise buffer and
  the master bus.
- A suspended context's `resume()` returns a promise that never settles until the user has
  interacted, and its clock does not advance. Never await it, and emit nothing unless
  `ctx.state === 'running'`, or everything scheduled meanwhile fires at once when it finally starts.
- Sixteen guns is not sixteen sounds, but it is not one either. The first rule was a hard minimum
  gap per kind with everything inside it discarded, and `tools/mix.js` showed the bill: 25% of
  gunfire and 47% of hits thrown away, worst exactly when the most was happening, so a ship of the
  line sounded like a sloop. `play.js` now queues instead — events of a kind are laid end to end at
  a fixed cadence, and only a backlog past `lead` is dropped. Everything is heard, and the *peak
  level falls*, because level rather than events is what gets spent.
- Spend level, not events. Each kind carries a decaying count of how much of it is already ringing,
  and new voices scale by 1/sqrt(1 + duck * that). Twelve guns are louder than three without being
  four times louder. Peak level over 150 battles: 11.2 under the old rule at 75% heard, 8.9 now at
  100%.
- Loudness alone does not make a volley sound bigger. `sfx.cannon` takes the same density number as
  `weight` and ducks the crack hard, the body barely, and runs the tail longer and darker — mass
  gunfire is bassier than one gun. Without it every busy moment came out identical.
- Grape and round shot must sound different on arrival. Which one is loaded is the only decision a
  player makes during a battle, so grape is a patter of small bright strikes and round shot is a
  modal thud on one of three timbers. One recipe with pitch jitter reads as one sound stuttering,
  and hits are the most frequent event in the game at 7.7 a second.
- Hits have been too quiet twice: 20 dB below a cannon on the first pass, 13 dB on the second. They
  are the only confirmation that a shot connected. They sit about 8 dB down now.
- Interface sounds are layered recipes, not one tock with the pitch moved. The shape is borrowed
  from cuelume (github.com/Danilaa1/cuelume); the frequencies are not, because theirs are glassy and
  this game wants wood. Only buttons that commit make a sound — a tick on every card and every cell
  is a clock, which is what the first pass sounded like.
- The echo on `confirm` is explicit taps, the layers rendered again quieter and duller, not a
  feedback delay. Oscillators stop themselves; a `DelayNode` does not, so a feedback loop needs
  cleanup timers and lingers in the graph between presses.
- Beware what you measure: the first version of the onset check flagged the detonation, because high
  frequencies have large sample-to-sample deltas by nature. Measuring the step *out of silence* is
  the thing that distinguishes a click from a bright sound.
- `tools/audio.js` renders offline and will happily measure a stale module. It disables the HTTP
  cache over CDP now; before that a "why is my new sound missing" session started here.

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

Never read layout from the frame path, and never from a pointer handler. This was the largest
render finding by a distance, and it does not look like a rendering problem at all:

- `scene.js frame()` called `resize()` every frame, which read `canvas.clientWidth`. That getter
  cannot answer without flushing style and layout for the whole page, so the cost of a frame
  depended on whether the HUD had just written to the DOM. It was the top JavaScript entry in both
  phases. A `ResizeObserver` reports the same numbers off layout the browser has already done, and
  only when they change.
- `ui/build.js pick()` called `getBoundingClientRect()` on every `pointermove`, forcing a reflow of
  the build panel at mouse-report rate. The canvas is fixed to the viewport, so the rectangle is
  cached and re-read on resize and scroll.
- Measured on `tools/frames.js`, JavaScript per frame: build p50 1.1ms to 0.5ms and p99 7.3ms to
  1.2ms, with frames costing over twice the median falling from 6.5% to 1.7%; battle p50 1.0ms to
  0.4ms. Wall-clock times headless swing 2x run to run and are not worth quoting.

The sea was 60% of everything drawn, and it was drawing a constant:

- It was a 1400-unit plane with a `MeshStandardMaterial`, so a full-screen physically based shader
  ran on every pixel of every frame. But the plane faces straight up and the camera is orthographic,
  so the normal and the view direction are both constant over the whole plane -- every pixel
  computed the same colour. Nine samples across the frame all read `172d3a`. Deleting it took
  1920x1080 from 58.4ms a frame to 19.8ms with the open water byte-identical in a screenshot diff.
- That is why nothing in `render/sea.js` is lit. Under this camera every view-dependent term --
  fresnel, specular, environment reflection -- is a constant, so the only kind of shading that
  survives is banding a scalar field, which is view independent.
- It all comes back the moment the camera becomes perspective.
- `tools/fill.js` also reports the floor: with every mesh hidden a frame costs 0.7ms, so the clear
  and the swap are free and anything left to win is in the scene.

`render/sea.js` then replaced both the flat colour and the 420 wind-streak quads with one
full-screen triangle. The quads were the largest layer left -- 46% of a 720p frame -- and they were
decoration; worse, they were alpha-blended and overlapping, which on a tile-based mobile GPU
switches off the hidden-surface removal the architecture is built around. Notes on it:

- The vertex shader has no matrix. An orthographic projection has no perspective divide, so screen
  to the y=0 plane is a scale and an offset; checked against a real unproject-and-raycast at 1e-13
  world units. The z term carries the 60-degree foreshortening.
- Wind falls out for free by squashing isotropic noise about 7:1 along the wind axis. The thing 420
  quads existed to convey is two noise lookups and a rotation.
- Threshold high. The first pass banded the whole range and the sea fought the ships for attention;
  only the crests catch light now. Legibility is the constraint on a game board, not prettiness.
- No `sin()` in the hash. `sin` is not bit-specified in GLSL ES, so the popular
  `fract(sin(dot(...)))` hash gives different water on different GPUs, and its argument overflows
  mediump. Value noise after iq (MIT) instead.
- Time is wrapped on the CPU and the uniform is `highp`. A plain seconds-since-load clock in
  mediump loses a frame of resolution by about 32 seconds and the water starts to judder.
- Aliasing is handled without `fwidth`: under this camera world-units-per-pixel is the same
  everywhere on screen, so it is one uniform computed when the zoom changes.
- Dither is two summed IGN taps for a triangular distribution, applied after the colour space
  conversion. One tap of uniform noise does not remove banding, and dithering before the conversion
  gets quantised away.

Unsettled, and it needs a real phone: the software rasteriser prices this shader *above* the quads
it replaced, because it penalises per-pixel arithmetic heavily and cannot show the blending penalty
that makes the quads bad on a tiler. The reasoning says the shader wins on real hardware; the
measurement here says it loses. That is why the second noise layer is the first thing the adaptive
controller drops -- decoration degrades before sharpness does.

Adaptive resolution is what actually promises 60fps, because frame cost is very nearly linear in
pixel count and no amount of tidying changes that. `scene.js` counts late frames over one-second
windows and steps the rendering scale down as soon as a quarter of them missed.

Stepping back *up* is the hard half, and hysteresis alone does not do it. With a fixed four-window
delay the controller pumped between 0.5 and 0.6 for as long as it was watched: a machine that lands
between two steps is comfortable at the lower one precisely *because* it is lower, so "comfortable
for a while" is not evidence that the higher step would hold. What works is making a failed
promotion expensive -- every step up that gets undone soon after doubles the number of good windows
the next one needs. Headless now descends 1 to 0.5 in eight seconds, probes upward exactly once,
and holds for as long as you care to watch. Anything with a real GPU should never leave 1.

Watch for this shape anywhere a controller reacts to its own effect. Also worth knowing: a window
needs a minimum number of frames to count, or returning from a backgrounded tab hands it a single
enormous frame, reads 100% late, and drops the resolution for no reason.

Material choice is the second lever, and the folklore about it is wrong:

- Everything solid is `MeshLambertMaterial`, not `MeshStandardMaterial`. Standard is physically
  based and costs roughly ten times the fragment arithmetic per light for a specular lobe that, at
  roughness 0.7 with no environment map, is invisible. It is worse than it looks: three.js emits
  `RE_IndirectSpecular` unconditionally for Standard, so every pixel runs the image-based specular
  approximation -- including an `exp2` -- purely to compute a slight diffuse darkening.
- The objection is that Lambert shades per vertex and would band the boxes. That has been false
  since r144. Lambert is per-fragment, and the three.js *manual* still says otherwise while its own
  API docs say per-fragment; the vendored r169 source settles it. Only the specular term is lost.
- Measured, ships alone at 1280x720: 4.08ms with Standard, 2.888ms with Lambert, 29% off that
  layer. Visually free -- swapping the materials on one frozen frame moved 0.3% of pixels and not
  one of them by more than 9 of 255.
- Keep `antialias: true`. Trading MSAA for resolution is backwards on a tile-based mobile GPU,
  where MSAA samples never leave tile memory: 4x MSAA measures about +23% on a Pixel 6, while 2x
  supersampling through the pixel ratio is +300% for comparable edges. Quality gives through the
  resolution scale, never through this.
- Do *not* set `alpha: false`, which looks like a saving and is not. MDN's WebGL best practices has
  a section headed "Avoid alpha:false, which can be expensive": an RGB back buffer often has to be
  emulated over an RGBA surface. `stencil: false` is a genuine saving and is set.
- Do not reach for `BatchedMesh`. It is for many distinct geometries sharing a material; these are
  one geometry repeated, which is `InstancedMesh`'s case. It also hard-requires `WEBGL_multi_draw`.
- The canvas has `webglcontextlost` / `webglcontextrestored` handlers. Mobile loses the context
  routinely, and without them the canvas stays black for the rest of the session.

Two smaller ones from the same pass:

- The ghost preview built a `MeshStandardMaterial` per hover and disposed the last. It is the only
  non-instanced standard material in the scene, so nothing else kept its compiled program alive.
  One cached material per part type, and one mesh whose geometry and material are swapped. Arc
  preview rings were likewise rebuilt per hover and are now cached by shape.
- Marking an instance attribute dirty re-uploads the *whole* array, and `fx.js` sizes its arrays for
  400 shot and 220 puffs. A typical battle has a handful, so it was sending about 50KB a frame to
  describe a dozen particles, and sending it when there were none. `addUpdateRange` bounds the
  upload to what was written, and an empty layer uploads nothing.
- Anything flat that ends up facing the camera after its `rotateX(-Math.PI / 2)` -- the impact
  rings, the arc bands, the prow -- is `FrontSide`, so back-face culling drops half its triangles.
  The stern flag is the one real `DoubleSide`: it turns with the ship.
- The arena boundary is drawn by the sea shader, not by a mesh. It was a 128-segment ring with a
  transparent material -- a blended draw call for a circle that is one `length()` from the origin
  once the shader already has world coordinates. It also antialiases properly now, where the
  tessellated version had visible facets. Battle draw calls 40 to 37, triangles 1467 to 965.

A measurement caution learned the hard way here: headless Chrome died partway through a run of
`tools/fill.js` and the next `tools/frames.js` reported a 2.6 second frame and 5fps. It looked
exactly like a catastrophic regression and was a dead browser. Restart `./tools/dev.sh` and
reproduce before believing a sudden cliff.

## Browser baseline

Current browsers only: last couple of years of Chrome, Safari, Firefox, and phones of the same
vintage. That is a deliberate decision and it is what licenses the following, so a compatibility
shim added back in without a reason is a regression:

- `new AudioContext()` directly. `webkitAudioContext` was for Safari before 14.1.
- `ResizeObserver` and `AbortController` unguarded. Both are years past universal.
- `ResizeObserver` observing `device-pixel-content-box`, which is the newest thing relied on here
  (Safari 16.4). It gives the exact integer backing-store size, so the true device ratio is
  `deviceBox / cssBox` rather than `devicePixelRatio` -- the two disagree under browser zoom and on
  a 125%-scaled display, and the disagreement is a faint moire over the whole picture.
- One `AbortController` per build phase owns every listener it registers, so teardown is a single
  `abort()` instead of a removal list that has to be kept in step with the registration list.
- `pointermove` is registered `passive: true`. `contextmenu` cannot be: it calls `preventDefault`.

Weighed and rejected, with the reason, so they are not relitigated for free:

- **WebGPU.** three.js has a WebGPU renderer, but it is a different vendored build and a different
  material pipeline, and there is nothing to win: 37 draw calls and 965 triangles are not a
  dispatch-overhead problem.
- **OffscreenCanvas in a worker.** Moves rendering off the main thread. The main thread spends
  0.3ms a frame; there is nothing to move.
- **BatchedMesh.** Wrong tool -- see the rendering notes.
- **A glyph atlas to halve the ship draw calls.** Real, and worth maybe 15 calls. Draw calls are
  not the bottleneck at this scale; fill is.
- **`mediump` in the sea shader.** The obvious mobile win, and it does not apply: the value-noise
  hash reaches intermediate values around 250,000, and the dither needs `gl_FragCoord` where a
  mediump ULP at 1080p is 1.0. The two expensive parts are exactly the parts that need `highp`.
- **`renderer.compileAsync()`.** Nine programs, all built during the first frames. Nothing compiles
  at a round boundary.
- **Reusing ship views between rounds instead of rebuilding them.** Measured: 0.2ms to create and
  0.1ms to dispose, so 0.6ms per round for both ships. Not worth the lifecycle complexity.

## Code shape

Two consolidations worth not undoing:

- **`tools/cdp.js` is the only DevTools client.** Five tools drive the real browser and each had
  grown its own copy of the same forty lines, plus three separate copies of "click through the
  overlays until the game reaches a phase". Anything new that talks to Chrome imports this. It is
  not fewer lines overall -- the module is about as long as what it removed -- but the gotchas are
  written down once instead of being rediscovered per tool, and two of the five had already
  forgotten to disable the cache.
- **`render/quality.js` holds the adaptive-resolution policy**, and `scene.js` only applies it.
  A control loop that reacts to its own effect is where the bugs live, and it was crowding out the
  file that is supposed to be about the scene.

## Gotchas already paid for

- A `<canvas>` is a replaced element: `position: fixed; inset: 0` leaves it at its intrinsic
  300x150. State `width`/`height` explicitly.
- `THREE.Fog` plus an orthographic camera 300 units back flattens the scene to one colour.
- Chrome caches aggressively between tool runs. `tools/shot.js`, `tools/audio.js`,
  `tools/frames.js` and `tools/profile.js` all disable it via CDP; anything new that imports from
  the page must do the same or it will measure the previous edit.
- Headless is software-rasterised, so a CPU profile of the page is 98% `(program)` with no stack.
  `tools/profile.js` excludes it and renormalises, which is the only way the JavaScript is legible.
- `perf.sample` used to be handed the *clamped* step, so on any machine slower than 20fps every
  frame recorded as exactly 50ms and the tail was invisible. It takes real elapsed time now.
- `tools/shot.js` steps: a bare number waits ms, `@file.js` evaluates a file in the page
  (avoids fighting shell escaping). It navigates to about:blank when done, because a live
  WebGL page keeps a core busy.
- Headless rendering is software-rasterised and roughly 3x slower than real time. Do not use
  wall-clock waits to reach a simulation state; drive it directly
  (`for (...) __game.battle.advance(0.05)`), or the capture will race.
- Dev autoplay stops after one match on purpose. `&loop=1` opts back in. A forgotten looping
  tab once pinned a CPU core.
- The build countdown uses real elapsed time, not the clamped simulation step, so a slow machine
  does not hand out a longer build phase. Both still stop when the tab is backgrounded.
- `tools/shot.js` prints a line per step *before* the screenshot line. Piping it through `tail -3`
  hides exactly the output you wanted; grep for `step` instead.
- In `ui/build.js`, `renderAll()` deliberately does not touch the hint text; callers own it.
  It used to, which silently ate every feedback message.
- Device metrics have to be set *after* the page loads. `Emulation.setDeviceMetricsOverride`
  applied before navigation leaves the WebGL surface out of captured frames entirely: the game
  runs, the draw calls happen, and the screenshot comes back as a correct HUD over empty water,
  which reads exactly like a rendering regression. `cdp.js` documents it and `shot.js` still got it
  wrong once after the comment was written.
- Backticks inside the shader source end the template literal. A comment mentioning a ratio in
  backticks turned into `SyntaxError: Unexpected identifier`. `cp file /tmp/x.mjs && node --check`
  finds these in a second; the browser only says the module failed to parse.
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

## Playtesting

The build phase is raycasted clicks on a WebGL canvas, which no DOM query can reach, so `?dev`
exposes `__dev` for the CDP driver:

```
__dev.clickCell(dx, dz)    project a hull cell to screen and dispatch real pointer events
__dev.pickCard('gun deck') select an offered part by name, prefix-matched
__dev.tool('btn-reroll')   press reroll / refit / remove / lock in
__dev.state()              phase, round, score, purse, offer, readout, warnings, hint, battle
```

`&seed=1234` pins the match seed. Without it every match is random, which means a bug found while
playing cannot be replayed -- that was the first thing playtesting needed.

Two habits that found real bugs:

- Probe the *edges* headlessly, not through the UI: a helm with no guns, guns with no crew, no
  magazine, a part placed with no path back to the helm. A scratch script driving `createBattle`
  directly runs in milliseconds and says exactly what happened. Every case should end in a sensible
  way and none should run to the bell.
- When a probe reports something absurd, suspect the probe first and check it, but *then* keep
  going. Two of the last three "probe bugs" were real game bugs wearing a probe's clothes.

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
