# Working notes

Broadside: two to four players fit out an age-of-sail ship during a timed build phase, then watch it
fight itself. Five rounds, first to three. Play on one keyboard or over a WebSocket. See
GAME_DESIGN.md for the design and the reasoning behind it.

No build step, no dependencies, no package.json -- including the server and its WebSocket
implementation. three.js is vendored in `vendor/`.

## Run and test

```
./tools/dev.sh              game server on 8123 + headless Chrome with CDP on 9222
./tools/dev.sh stop         tear both down -- always do this when finished
open http://127.0.0.1:8123/index.html
```

`server/main.js` serves the directory and hosts the rooms on the same port, so an online game needs
nothing else running. It replaced `python -m http.server`, which could only do the first half. The
game needs http for ES modules; opening the file directly will not work.

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

Networking and more than two ships:

```
node tools/netcheck.js      the authority and the replay, headless, over a virtual wire
node tools/netplay.js 4     four real browsers through a whole online match
node tools/wscheck.js       the WebSocket implementation: framing, fragments, limits, teardown
node tools/engines.js       the same battles under node and Safari's jsc, bit for bit
node tools/melee.js         three and four ships: length, seat fairness, builds, damage sweep
node tools/fingerprint.js   full-precision state dump, engine-agnostic; what engines.js diffs
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
- "Did I break the networking?" -> `netcheck.js` first: it plays whole matches against a real room
  over a virtual wire with latency, jitter and reordering, in milliseconds, and checks the client's
  replay against the authority tick for tick. Then `netplay.js` for the parts only real browsers and
  a real socket can show -- the lobby, the clock estimate, the join code.
- "Will this desync on somebody else's browser?" -> `engines.js`. It is the only tool that can
  answer, and the answer was no for a long time. See the determinism note below.
- "Is a three- or four-way worth playing?" -> `melee.js`. Length, empty air, seat fairness against
  binomial noise, whether the field size changes what to build, and the `parts.js` lens run at every
  field size. Read section 5 before section 3: section 3 fights pure archetypes and is bimodal by
  construction, and where the two disagree section 5 has been right both times.
  It takes about 20 seconds, which is the one tool here that is not instant.

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

- Three or four ships was assumed to need *less* incoming damage per gun, since everyone is shooting
  at you. Backwards: a duel ends when one ship sinks and a four-way when three do, so the guns have
  three times as much hull to get through. Uncorrected, a four-way ran 38.9s and only a third of them
  reached a verdict. `SHIP_COUNT_DAMAGE` goes *up*.
- A ring start was assumed to be fair by symmetry. It was 79% / 7% / 14% by seat, because on a ring
  every ship is exactly equidistant from its two neighbours and "nearest enemy" was decided by which
  of two identical distances came out smaller in float32 -- which is not symmetric. Two ships locked
  onto each other and left seat 0 alone.
- The simulation was assumed to be deterministic across machines because it has a seeded rng and no
  `Math.random`. It was not: 68% of sampled ship states differed between Node and Safari within one
  second. See the determinism note below.
- The simulation was assumed to run at a fixed 60Hz. It did not: `advance(dt)` ran whatever fraction
  of a tick was left over at the end, so `advance(0.25)` took sixteen ticks and the step size was
  really the caller's frame time.

- The carronade looked like a trap in a melee: 0.30 times even at three ships against 1.33 in a duel,
  from the pure-archetype grid. It is not. Under the random-build lens it reads 48% / 43% / 43% as the
  field grows -- weak, never a trap -- and on the sloop the melee makes it *less* bad. The same pure
  grid that says 0.30 says 1.33 for the duel, and the random lens contradicts both, which is the
  bimodality this file already warns about, caught in the act.
- The massed battery looked strong in a melee (1.46 at three ships, pure builds). Under random builds
  it is 45% / 49% / 47% -- dead even at every field size. The `ORBIT_SENSE` door is still shut with
  three and four ships on the water, which is what that regression guard exists to say.

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

## Networking

Two to four players, one authority, and a battle every client reproduces rather than receives.

`src/net/room.js` is the authority: one room is one match, and it owns the phases, the purses, the
offers, the validation and the battle that counts. It touches no DOM, no renderer, no socket and no
wall clock -- time arrives as an argument to `update(now)` and messages leave through `emit(target,
msg)`. That is what lets the same file be the server for four browsers, the in-process authority for a
game on one laptop, and a headless harness that plays a five-round match in a few milliseconds.

**Local play is the networked path with the wire taken out.** `src/net/local.js` runs the same
`createRoom` in the page and hands messages straight to the client. A hot-seat game is therefore not
a second implementation of the flow that happens to resemble the online one; it is the online one.
This is the single most valuable decision in the networking work, because it means playtesting
locally exercises the code that decides an online match. The one difference is `hotseat: true`, which
runs build phases one seat at a time instead of all at once.

**The battle is replayed, not streamed.** The server sends `{seed, hullIndex, windTo, designs,
startAt}` once, then relays each ammunition toggle stamped with a tick number. Clients build the same
battle from those numbers and run it. Nothing about a ship's position ever crosses the wire; there
would be no point, since both sides compute it from the same inputs. A whole battle is a handful of
messages of a few dozen bytes.

**Latency is absorbed by playing in the past.** The client runs its replay `net.delayMs` behind its
estimate of the server's tick, so an input stamped for tick N has already arrived by the time tick N
is simulated. That is the entire latency strategy, and it works because the only input in the game is
a toggle that already costs a 1.3s reload -- 100ms of input delay is not perceptible. Deterministic
lockstep was considered and rejected: its whole benefit is hiding latency for dense continuous input,
which this game does not have, and its failure mode is stalling the simulation to wait for a packet.

**A late input is repaired, not papered over.** If a toggle arrives stamped for a tick already run,
the replay is wrong by construction. The client rebuilds it from tick zero and replays the whole input
stream -- a few milliseconds at 3000 battles/sec -- and widens its delay so the next one is not late.
The delay only grows within a session, because relearning it every round means paying for the lesson
five times. Measured on `netcheck.js` with a clock estimate 120ms wrong: 39 late inputs per client per
match before the adaptation, 2 after.

**The server states a checksum twice a second** (`sim/checksum.js`, FNV-1a over quantised state) and
the client compares it at exactly the same tick. Both walk ticks through
`timeline.runToMarks`, because a checksum taken at tick 61 on one machine and 60 on the other compares
nothing. A mismatch triggers the same rebuild. The authority's verdict is what the result screen
shows, so a client that drifts is a cosmetic problem and never a lost match.

**The client is not trusted.** Every build action is a command -- place, remove, refit, reroll, lock
-- validated against the room's own state by `src/shipyard.js`. The client applies the same rules
first, so a click lands on the deck immediately rather than a round trip later; when the two disagree
the room says so and sends the design back, and the client's copy is replaced wholesale rather than
patched. `netcheck.js` has a section for exactly this.

**No seed derived from the match seed ever reaches a client.** `hashSeed` is a couple of
multiplications by an odd constant and is trivially invertible, so handing out an offer seed hands out
the match seed -- and the match seed decides which beam the battle turns to, which CLAUDE.md already
records as worth 100% of 800 battles to a player who knows it. So offers are drawn by the authority
and rerolls are a round trip, and the battle seed is published at the moment the guns start and not
one moment earlier.

A disconnected player keeps their seat for 30 seconds and the bot works their ammunition while they
are away, so a flaky connection is a stutter and not a forfeit. A reconnect is handed the stored
battle message plus the whole stamped input log and catches up by simulating.

### Determinism across engines, which is not free

`sim/` has a seeded rng, no `Math.random` and fixed ticks, and that is not enough. ECMA-262 calls
`sin`, `cos`, `atan2`, `pow`, `exp` and `log` implementation-approximated: an engine may return
anything within an unspecified tolerance. Measured with `tools/engines.js`, V8 and Safari's
JavaScriptCore disagree on 4% of `sin` arguments and 21% of `atan2` arguments by up to 3 ULP, and
that was enough to make **68% of sampled ship states differ between Node and Safari inside the first
second of a battle**. No winner ever flipped -- the steering controller is contractive and damage is
quantised -- but a desync detector that fires on two thirds of its checks detects nothing.

The fix is `fsin`, `fcos` and `fatan2` in `sim/geometry.js`: every transcendental in the simulation
rounds its result to float32 with `Math.fround`. A disagreement of a few double ULP collapses onto one
float32, and ordinary arithmetic and `sqrt` are exactly specified by IEEE 754, so the rest of the
chain follows. After it, `engines.js` reports the two engines bit-identical, and it cost nothing on
`bench.js` and not one line of `golden.js`.

Worth knowing:

- `Math.sqrt` needs no wrapper and never will. IEEE 754 *requires* correct rounding for square root
  and only *recommends* it for transcendentals, and a 2024 normative change to ECMA-262 removed
  `sqrt`'s implementation-approximated status. `geometry.js len()` is safe on every engine forever.
- It is not a proof. A result sitting exactly on a float32 rounding boundary still splits, about one
  call in 2^29. That residue is why the server's outcome is authoritative rather than merely agreed.
- `config.js` imports the wrappers from `sim/geometry.js`, which looks like a layering inversion and
  is the lesser evil: `startPositions` and `windFactor` are read by the simulation, so they have to
  give the same answer everywhere, and a second copy of the wrappers is worse.
- `golden.js` cannot see this class of bug at all -- it prints rounded fingerprints. `engines.js` and
  `fingerprint.js` exist because it cannot.

### The simulation advances in whole ticks

`battle.advance(dt)` carries the remainder and runs only whole ticks; `battle.advanceTicks(n)` runs
exactly n. It did not used to: it subdivided the caller's dt and ran whatever fraction was left over,
so `advance(0.25)` took sixteen ticks (fifteen full and one of 5e-17) and a browser drawing at an
uneven frame rate ran a different number of differently sized ticks than the harness. Invisible in a
game watched on one machine, fatal for one replayed on two, because a tick number is the only thing an
input can be stamped with. Fixing it moved 2 of 900 golden fingerprints -- both the same seed, neither
a different winner.

## Three and four ships

A melee is a duel with target selection added and the incoming damage repaced, and deliberately
nothing else. Every melee path reduces *exactly* to the two-ship code at two ships -- same
arithmetic, same order, same draws from the rng -- and `golden.js` is byte-identical across the whole
change, which is how that claim is checked rather than hoped for. `SHIP_COUNT_DAMAGE` and
`SHIP_COUNT_ARENA` are indexed by ship count and read 1 at index 2; `startPositions` branches to the
mirrored pair.

- Ships start evenly spaced on a ring, each pointing at the middle. Initial targets are a **round
  robin**, not the nearest enemy: on a ring the two neighbours are exactly equidistant and float32
  noise picked between them asymmetrically, which measured as 79% / 7% / 14% by seat. A rotation is
  the one assignment no seat can be favoured by, and at two ships it is still the other ship. After
  the fix every seat is inside binomial noise at three and four ships.
- `pickTarget` reconsiders every `TARGET_RECHECK` and only switches for a rival inside
  `TARGET_SWITCH_MARGIN` of the current range. Without the margin a ship between two enemies swaps
  every few ticks and sails down the middle with its guns bearing on nothing.
- A ship that strikes leaves the fight and stays on the board. Shot already on its way to her splashes
  instead of pounding a hulk, dropped in one pass when she strikes rather than tested per projectile
  per tick.
- `ship.foes` is the list of enemies still afloat, rebuilt only when someone strikes, so the busiest
  loop in the game has no liveness test in it.
- The economy pays comeback money by placing rather than by winning: `battle.placing` orders the field
  and `placeBonus` scales the loser's bonus by where you came in. At two players it is exactly the old
  `loserBonus` and nothing else.

Measured, at the tuned values: a three-way settles in 16.6s at 38% empty air and a four-way in 17.1s
at 34%, against a duel's 14.2s at 30%, and both are essentially always decisive. Length and empty air
trade against each other the whole length of the damage sweep, so the clock can always be bought and
the only question is the price.

## Hulls in contact

Two problems that looked cosmetic and one of them was.

**Separation could not be one number.** A ship of the line is ten cells long and five wide, so bow to
bow two of them need twice the room they need beam to beam. With a single floor, measured, hulls
overlapped by up to 5.2 world units -- a third of a frigate's length -- and a ship of the line spent 21%
of every duel inside its opponent. `steering.js separate()` now takes the greater of the old floor and
the elliptical support radius of the two decks along the line joining them, which drops overlap to zero
at every hull and field size.

Only ever the greater, and that is the load-bearing half of the rule: the fight settles with the enemy
abeam, which is the cheapest orientation, and a rule that could *lower* the floor there would move the
range two ships settle at. Ablated against the old floor over 6000 paired battles: 6 winners flipped
(0.1%), mean finishing time moved 0.11s, and 2 of 150 matchup cells moved -- one of them toward even.
So no gun's range needed retuning, which was the thing to check before touching this at all.

**Contact now costs something.** `damage.js grind()` puts a crunch into the cell nearest the point of
contact on both ships, once per pair per half second, scaled by how fast the two hulls are moving
relative to each other. A crunch rather than damage per tick because `damageCell` has a floor of one
point per call, so sixty calls a second would saw a timber in half -- and because a discrete crunch has
an impact, a sound and a log line, which is what makes it something a player sees happen rather than a
bar going down.

Measured the same way, and the split is the point: a duel flips 0.1% of winners and moves the mean
battle from 14.33s to 14.28s, because two ships orbit at their preferred range and rarely touch. A
four-way flips 5.8% and comes in 0.45s shorter, because four ships in one arena crowd. That is flavour
in the duel the part table was tuned around, and a real consideration in the melee -- and it means
ramming is available to a player who wants it without being a strategy that reads on the duel grid.
Dismastings went from 695 to 854 per 1080 battles, which is rigging tearing as hulls grind past.

## Architecture invariants

```
src/config.js      every gameplay and feel constant. Tools read the same file
src/theme.js       every colour. Keep PLAYER[] in step with --p1..--p4 in styles.css
src/match.js       scores, purses, hull progression, intel, the shop offer. Pure
src/shipyard.js    the rules of fitting out: cost, legality, refunds. Pure, and the authority's
src/autobuild.js   greedy ship builder: bot opponent and the dev Fill button
src/bot.js         the ammunition decision, for bots and for absent players
src/dev.js         URL-driven dev harness, inert without ?dev
src/perf.js        frame stats, always on: an EMA for scale, percentiles for the tail
src/main.js        presentation and flow: overlays, camera, input, render loop

src/net/           the networked game, and the local one
  protocol.js      every message name and every constant both sides must agree on
  room.js          the authority: one room is one match. No DOM, no socket, no wall clock
  client.js        what this browser believes, and the battle it draws
  local.js         transport for a game with no network: the authority runs in the page
  socket.js        transport over the wire: clock estimate, reconnect, heartbeat

server/            the host process. Node, no dependencies
  main.js          static files and rooms on one port
  ws.js            RFC 6455, by hand: handshake, framing, fragments, ping, close
src/render/        three.js scene, ship meshes, particles, glyph textures
  sea.js           the sea, the wind and the arena ring, as one full-screen triangle
  quality.js       the adaptive resolution control loop, kept apart from the scene
src/audio/sfx.js   every sound, synthesised. Takes a context, so it renders offline for testing
src/audio/play.js  when to make a noise: voice spacing, stereo placement, mute, the gesture unlock
src/ui/            build-phase panel, HUD chrome
src/data/          parts, hull shapes (ASCII art)

src/sim/           the deterministic battle core, one file per question
  rng.js           seeded generator
  geometry.js      len, wrapAngle, ship-local to world, and the float32 trigonometry
  ship.js          the persistent design, and the runtime state of one ship in one battle
  steering.js      how the ships sail, hold station, and choose who to fight
  gunnery.js       firing, shot in flight, and where a ball lands
  damage.js        what a ball does on arrival: structure, crew, magazines, severing
  battle.js        the clock over all of it, and how a round ends
  timeline.js      a battle plus its stamped input stream. Authority and replay share it
  checksum.js      a fingerprint of battle state, for catching a replay that has drifted

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
- **`main.js` is the only file that knows about both the DOM and the simulation,** and it no longer
  knows the rules of anything: no purses, no legality, no verdicts. It draws what the client says is
  true and turns clicks into commands. If a rule is being decided in `main.js` or in `ui/`, it is in
  the wrong file -- `net/room.js` decides, `shipyard.js` and `match.js` hold the rules it applies.
- **`net/room.js` is the only authority, and local play uses it too.** Anything that reimplements a
  rule for the local case will drift from the online one, and the drift will not be found by playing
  locally, which is how the game is mostly played while being built.
- **Nothing in `sim/` or `net/room.js` may touch a wall clock.** Time is an argument. That is what
  makes a five-round match testable in milliseconds and a room's behaviour reproducible.
- **Every transcendental in `sim/` goes through `fsin`/`fcos`/`fatan2`.** A bare `Math.sin` in the
  simulation is a cross-browser desync waiting for a threshold to cross. `tools/engines.js` catches it;
  `golden.js` cannot.
- **The sim's hot path is performance-sensitive on purpose.** Every question here is answered by
  running thousands of battles, so throughput compounds across a session. `tools/bench.js` reports
  it: about 3100 battles/sec, roughly 45,000 simulated seconds per real second. Do not reintroduce
  per-tick allocation or per-tick trigonometry. It was ~3200 before the melee, and the 3% went on
  target selection and the per-ship loops in `tick`; the float32 trigonometry cost nothing measurable.
  Measure `bench.js` on an idle machine -- it swings by a factor of three under load, which is easy to
  mistake for a regression.
- **Refactor against `tools/golden.js`.** Record it, change the code, diff it. Anything meant to be
  structural must leave it byte-identical. It has already caught one refactor that changed
  behaviour and one that did not but looked like it might.
- **Whoever consumes `battle.effects` must drain it.** Nothing in `sim/` clears it.

## What made the simulation fast

From 1131 to ~3200 battles/sec (about 3100 since the melee), all of it from the profiler
(`node --cpu-prof`) and none of it from guessing. Worth knowing which lever is which, because the shape repeats:

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

- Keep a per-tick function small enough for V8 to inline. Generalising `checkEnd` to four ships grew
  it past the inlining budget and the whole of it stopped being inlined -- 6% of throughput, showing up
  in the profile as a new `checkEnd` entry and a `tick` that had doubled. Splitting the hot test from
  the cold verdict, which now lives in two functions that run once per battle, got it back. The lesson
  generalises: in this simulation a function called every tick is either small or not inlined.
- `ship.foes` is a list of the enemies still afloat rather than a list of all of them with a liveness
  test. It changes at most three times in a battle and is read guns times foes times ticks.

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
- A four-way is the worst case the mixer has, and it holds. `node tools/mix.js 150 4` pins every
  battle to four ships, which is about twice a duel's gunfire: cannons stay at 100% heard and the peak
  level rises from 8.90 to 9.11, which is the level-spending rule working exactly as intended -- twice
  the events for 2% more level. What does give is the least important cue: `break` drops from 100% to
  84% and `splash` from 67% to 54%, because three ships coming apart clusters those events. Left alone
  deliberately; widening the queue for them would spend the peak level that was hard to win.
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

A ship that strikes her colours fades into the sea colour rather than turning transparent, and the
reason is the shared materials. One `MeshLambertMaterial` serves the deck plates of every ship, so
lowering its opacity would fade the whole fleet; the part glyphs and the flagpole share materials too.
So the wreck is darkened toward `SEA.water` through the per-instance colours it already owns, the
handful of materials that *are* per view are faded properly, and the two on shared materials -- the
flagpole and the glyphs -- are hidden instead, at a point in the descent where a mast slipping under is
what it looks like. The hull settles two and a half units at the same time, and the whole group is
hidden once it is within half a percent of the water's colour, since an invisible wreck still costs a
draw call per part layer.

This matters more than it sounds: in a melee the survivors sail straight over the wreck, so without it
two hulls occupy the same water and it reads as a bug rather than as a wreck being passed.

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

Networking and the melee added these:

- **Do not stop listening for the gesture until the context is actually running.** The audio unlock
  removed its own listeners as soon as it had *called* `resume()` once, which assumes the call worked.
  Any browser that refused it, came up suspended anyway, or suspended the context again later left the
  game permanently mute with nothing able to try again -- and the only thing that appeared to fix it was
  pressing the mute button twice, because `setMuted` is the one other place that writes the master
  gain. It now stays subscribed until `ctx.state === 'running'`, listens for five kinds of gesture
  (activation is not granted on the same event in every browser), rewrites the gain whenever the
  context reaches running, and resumes on `visibilitychange`. `__dev.state().audio` reports all four
  reasons there might be no sound, because from the outside they are indistinguishable.
- **`syncFromBattle` writes the ship's whole position every frame, so anything else that moves the
  group loses.** The sinking animation set `group.position.y` and the next frame set it back to zero:
  the wreck faded on the spot without ever settling. Every per-frame write is a place a per-event write
  can be silently undone.

- **An in-process authority answers synchronously.** In a local game `client.lock()` runs the whole
  handoff before it returns, so code written in the obvious order -- send the command, then update the
  panel -- lands on the *next* phase's panel. Autoplay locked in the first captain and then disabled
  the second captain's lock button, so every hot-seat build phase ran its full forty seconds and the
  room had to time it out. Every handler in `ui/build.js` now states its feedback *before* sending.
  Nothing about it is visible over a socket, where the reply is a round trip away, which is exactly
  what makes it worth a comment.
- **A hidden tab gets no `requestAnimationFrame` at all** -- stopped, not throttled -- so its frame
  loop does not run and its replay of the battle sits at tick zero. That is right for a real player,
  since there is nothing to draw, and it means a driver with four tabs open can only ever see one of
  them play. `--disable-renderer-backgrounding` and friends do not fix it. `__dev.pump(dt)` supplies
  the clock instead, which is what `netplay.js` uses and what every other headless tool here already
  does. The cost in coverage is that `netplay.js` does not prove the frame loop calls `update()`;
  `playtest.js` and `shot.js` run a visible tab and do.
- **The dev harness used to write designs directly and now has to obey the shop.** The offer is five
  part types out of nine, so filtering a planned build against it dropped every gun of an archetype
  whose gun was not on offer, and two such ships met and neither could fire. Rounds ended on the
  five-second stalemate rule and a whole autoplayed match was nothing but draws -- it read as a
  balance collapse and it was a harness bug. `dev.js planFor()` picks a gun the shop is *selling*.
- **`Math.fround` is load-bearing, and `golden.js` cannot see it.** See the determinism note.
- **A ship view holds a reference to `design.parts`.** When the authority corrects a design, the
  client replaces the contents of that object rather than swapping in a new one; swapping leaves the
  deck on screen showing the design that was just thrown away.
- **The RFC 6455 magic GUID is `258EAFA5-E914-47DA-95CA-C5AB0DC85B11`.** A wrong one passes any test a
  server writes for itself and is rejected by every real client, because the client verifies the
  digest -- the symptom is every connection closing before it opens, with nothing saying why.
- **`http.Server` builds its sockets with `allowHalfOpen`,** so an upgraded socket whose peer sends FIN
  and stops leaves our writable half open for ever: no `close` event, one leaked connection per
  departed player, and `server.close()` hanging for thirty seconds. `socket.on('end', () =>
  socket.end())`. Found by counting opens against closes; every other symptom of it looks like
  something else.
- **`tools/melee.js` patches `config.js` by exact string match** for its damage sweep, the same way
  `ablate.js` and `tune.js` do. Changing `SHIP_COUNT_DAMAGE` breaks it loudly; fix the patch string in
  the same commit. This bit once, immediately, exactly as documented below.
- **Chrome's `/json/new` needs PUT** and rejects GET, which is a change from older versions and the
  reason `cdp.js openTab()` looks the way it does.

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
__dev.proceed()            press whatever the overlay button currently says
__dev.ammo(seat, 'grape')  switch a seat's ammunition, for a driver that cannot press keys
__dev.pump(dt)             advance the client by hand, for a tab that is not the visible one
__dev.state()              phase, seat, roster, score, purse, offer, readout, hint, wire, battle
```

The URL says what to play, so no tool has to click through a menu:

```
?dev=1                     manual play on one keyboard, plus a Fill button
?dev=brawler,sniper        autoplay, one archetype per seat
?dev=draft&players=4       autoplay, four drafted ships
?dev=1&bots=2              you plus two bots, locally
?dev=1&net=1               open an online room on this origin
?dev=1&net=1&room=ABCD     join one
?dev=1&net=1&watch=1&room=ABCD   join as a spectator
&name=Anne                 the name other players see
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
- The gun deck reads past 60% in a melee under the random-build lens: 55% in a duel, 64% at three
  ships, 63% at four, and 70-72% on a ship of the line. That is "wins games on its own", and it is the
  one melee reading not explained away below. It is also not a per-ship-count constant that put it
  there — flattening `SHIP_COUNT_ARENA` or `SHIP_COUNT_DAMAGE` moves the gun edges by at most two
  points, and flattening the damage scale wrecks the clock. It is intrinsic to having three enemies:
  `far` goes 24% to 48% as the field grows, so reach is worth more and a short gun has to close on one
  ship while two others shoot it. The only lever left is the part table, which costs the duel, where
  the gun deck is in band. Left alone deliberately; if a melee ever becomes the default rather than a
  variant, this is the row to revisit, and the honest fix is the part table plus a duel re-measure.
- Spectators are supported by the protocol and the server and have no way in from the interface except
  `?dev=1&net=1&watch=1&room=ABCD`.
- Nothing rate-limits build commands per socket. The room refuses illegal ones, so the exposure is
  noise rather than cheating, but a client sending a thousand places a second would be served.
- Four-way rounds are decided on the last ship afloat. An alternative worth measuring is ending the
  round on the *first* strike, which would shorten a four-way to a duel's length without buying it
  with empty air; `melee.js` section 4 shows damage alone cannot.
