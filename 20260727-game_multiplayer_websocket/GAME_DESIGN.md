# Broadside

Two players. Age of sail. You never steer the ship — you build it, then watch it fight.

Casual and fast. Skill lives in the build phase, luck lives in what you get offered and
where the shot lands.

## Reference stack

- Galaxy Trucker: frantic real-time assembly, then your ship gets taken apart in front of you.
- Super Auto Pets: short build phase, auto-battle, escalating economy, best-of-N.
- Airships: Conquer the Skies: shots hit the first thing in their path, ships degrade functionally.

## Match loop

First to 3 round wins. Five rounds maximum.

```
ROUND N
  wind is rolled and shown
  BUILD   player 1, then player 2 (hot-seat), timed
  BATTLE  both ships sail themselves, overtime from 20s, hard stop at 40s
  RESULT  point awarded, damage kept
```

Your ship persists across rounds, damage included. Each round you move into a bigger hull
and carry your parts with you. So round 1 is a blank slate and rounds 2-5 are triage:
patch the hole in the bow, or leave it and add another gun?

| Round | Hull             | Cells | Scrap granted | Build time |
| ----- | ---------------- | ----- | ------------- | ---------- |
| 1     | Sloop            | 11    | 34            | 40s        |
| 2     | Brig             | 17    | 32            | 26s        |
| 3     | Frigate          | 23    | 42            | 26s        |
| 4     | Heavy frigate    | 29    | 46            | 28s        |
| 5     | Ship of the line | 38    | 56            | 30s        |

Grants have to outpace attrition. Roughly nine cells are destroyed per battle, so a flat
allowance leaves ships holier every round instead of grander.

The loser of a round gets 45% of that round's grant as comeback money. A flat bonus is worth
nothing by round 5, which is how you got 3-0 sweeps. Unspent scrap carries over.

That is as far as catch-up goes, and it is enough. Sweeps run at about 17% of matches, and two
evenly matched players in a first-to-three produce a 3-0 a quarter of the time by chance alone —
so there is no measurable snowball left to correct. Adding more comeback help would only make
early rounds matter less.

## The hull

Hulls are tapered, not rectangular. Cells are addressed by offset from the ship's centre,
so a part placed at offset (-1, 0) stays at (-1, 0) when you move up to a bigger hull.

```
              bow
             /---\
          +--+---+--+
          |# | ^ | #|      ^ mast        # heavy timbers
          +--+---+--+      @ helm        G gun deck
          |G | @ | G|      * magazine    c crew quarters
          +--+---+--+
          |G | * | G|
          +--+---+--+
          |G | c | G|
          +--+---+--+
             \---/
             stern
```

That shape creates the layout grammar on its own:

- Centre column is the spine. Masts, magazine, crew, helm.
- Flanks are the gun line. Side-firing guns can only be placed off-centre, and the side is
  inferred from which flank the cell is on.
- The bow has almost no cells, so a forward-firing long gun genuinely costs you something.

An empty hull cell is a hole, and shot passes straight through it to whatever is behind.
Plugging gaps with cheap timber is how you stop a lucky ball reaching your magazine. This
turns out to be the single most important rule in the game — see "What earns its keep".

## Parts

| Part            | Cost | HP | Crew | Gunnery                                       |
| --------------- | ---- | -- | ---- | --------------------------------------------- |
| Hull timber     | 1    | 9  | -    | Filler, and load-bearing                      |
| Heavy timbers   | 3    | 30 | -    | Soaks 2 off every incoming hit                |
| Crew quarters   | 5    | 14 | +3   | Supplies the crew pool                        |
| Mast            | 4    | 11 | 1    | Speed and turn rate, up to what the hull needs |
| Powder magazine | 4    | 8  | -    | Needed to fire at all. Detonates when destroyed |
| Swivel gun      | 4    | 11 | 1    | All round, range 26, 1x3, reload 1.0s         |
| Gun deck        | 8    | 17 | 2    | Own flank, range 38, 3x4, reload 1.8s         |
| Carronade       | 9    | 14 | 1    | Own flank, range 24, 2x9, reload 1.9s         |
| Long gun        | 9    | 14 | 2    | Bow cells only, range 48, 1x22, reload 2.2s, halves soak |
| Helm            | free | 20 | -    | Pre-placed at centre. Lose it and you strike  |

Reloads and damage are deliberately fine-grained: the same damage per second split across
twice as many volleys. A battery that fires one heavy clap every three seconds leaves nothing
on screen in between, which measured as two thirds of the battle being empty air. Muzzle
speeds are halved for the same reason and for a second one — a ball that takes a second to
cross the water can be watched and anticipated, where a fast one just appears as a hit.

A hull only carries so much sail: past `mastsWanted` (two, rising to four on a ship of the
line) a mast does nothing. The build readout states the number, because sampling random builds
showed the ship with more masts winning only 36% of the time. A cap the player cannot see is a
cap they pay for.

Guns need gunners. Crew quarters supply a pool; guns are manned in placement order and any
gun left without crew stays silent. No live magazine means no gun fires at all — one
magazine is a gamble, two is insurance.

The shop offers five part types per build phase and you may buy as many of each as you can
afford; buying 38 cells one card at a time would be tedious, and the interesting luck is in
which types you are shown. Rerolling costs 2. Cheap timber is always offered, and powder or
crew are added if you have none, so a hand is never unplayable.

Refit repairs every damaged part at once for half the cost of each, worst first. Clicking
damaged cells one at a time was the same decision wrapped in busywork.

## Combat

Both ships run the same steering logic: one continuous controller rather than a ladder of
range bands. Each holds the heading that keeps its guns bearing at its preferred range — a
quarter turn off the bearing for a broadside ship, straight at the enemy for a bow chaser —
and the range error bleeds that offset toward an approach when the enemy is far and past
abeam when it is close. At the preferred range no correction is left and the ship simply
circles, which is exactly where its flanks want the enemy.

Both ships take the same sense of rotation, so they orbit their common midpoint. Opposite
senses were the original bug behind the dullest thing this game did: each ship kept the other
abeam by sailing a parallel course, so the pair held its range perfectly and marched off the
map together, firing nothing until the arena hauled them back. A first volley, a long silence,
then the action resuming in the corner. Fixing that alone took empty air from 86% of the
battle to about 30% and edge-of-arena time from 10% to zero.

A ship does not break off when the enemy gets inside its range (`ORBIT_RETREAT` is 0). Letting
it run was kiting: two ships holding a range neither could shoot at for twenty seconds. Now
the fight settles at the shorter of the two preferred ranges, where both sides can shoot, and
the reach of a long gun buys free opening shots rather than the right to refuse an engagement.

Which beam the action turns to is drawn at random per battle and is not shown during the build
phase. That is the price of a fixed engagement geometry: any beam the player can count on
becomes a sheltered side to hide the crew and the powder behind. Measured, a build that massed
its whole battery on the predictable side won 100% of 800 battles at every hull size, because
damage here compounds and concentration beats spreading. The draw makes massing a bet — double
the broadside that bears, or half of it idle — instead of an answer. `autobuild.js` keeps a
`massed` profile purely to hold that door shut; it should measure near 50%.

Wind is rolled once per round. Sailing with it is fast, into it is slow, so one side of the
orbit is quicker than the other and the two ships end up contesting the weather gauge.

Note what wind is not: there is no wind strength, and rotating the direction through 24
points almost never changes who wins. It is the speed penalty that matters, not the bearing.
An earlier draft of this document claimed the wind created a build decision ("skip a mast in
a light wind") — measurement showed that was wishful thinking, and the claim is gone.

### Weapon roles

Each gun owns a distinct corner of (range, arc, damage per ball):

- Long gun reaches furthest and punches through heavy timbers, but it fires forward only and
  has to be worked from the bow, so a ship carries a few and not a battery. Unlike a broadside
  a bow gun bears all the time, which is why it is restricted: unrestricted, a ship of the
  line carrying fourteen of them beat a pure broadside build 10-0.
- Gun deck is the mid-range broadside workhorse and the baseline everything else is priced
  against.
- Carronade is brutal inside 24 and useless outside it, and needs only one hand to work.
- Swivel gun is cheap, fires all round, needs one hand, and does almost nothing to armour. It
  is the only gun that can sit on the spine, and it is a grape platform.

A broadside fires out the flank its cell sits on and nowhere else. Letting it answer to either
beam is measurably better to watch — every gun works, and empty air drops six points — but it
deletes the decision of where to put the battery, so it stays as it is. Half your guns looking
at open water is the cost of the other half being pointed the right way.

Balance is not the goal; the absence of cheats, no-brainers and boring dominant plays is.
Two lenses are used, because a single one misleads:

- `tools/balance.js` fights pure single-gun builds and reports the worst pairing, graded on
  the fighting-game scale (5-5 even, 6-4 winnable, 7-3 a counterpick, 8-2 near unwinnable).
  Averaging win rates hides everything: a roster where every matchup is 8-2 still averages
  50%. Several pairings are still past 7-3 and that is the standing balance item.
- `tools/parts.js` samples hundreds of random legal builds instead, and asks per part: when one
  ship carried more of this than the other, how often did it win? That catches the two failure
  modes a matchup grid cannot — a part nobody should take is as broken as one everybody must.

The pure-build grid comes out bimodal, near 50% or near 100% with little between, which is the
signature of a compounding advantage rather than of gun statistics: first blood opens holes,
and holes let shot through to the vitals. So a lopsided pure matchup is weak evidence about a
gun, and both tools are read together.

### The live decision

The one thing you touch during a battle is your ammunition, switchable at any time:

- Round shot smashes hull, brings down masts, sinks.
- Grape shot shreds crew and barely marks the timbers.

Because crew man the guns, grape silences a broadside without scratching the ship. So you
spend the battle reading the enemy: their gun deck is untouched but they are down to two
crew quarters, stay on grape and their guns go quiet. Switching costs a reload.

Player 1 presses A, player 2 presses L.

### Reading the ship

The four mechanics that decide battles were all invisible in the interface at first. Each now
shows up somewhere:

- Open holes lead the build readout, and a warning appears past 30% of the hull. They are the
  rule that decides most battles, and a "cells filled" ratio buried that.
- Hovering a gun draws its firing arc as a band just outside the hull, taking its side from
  the flank the cell sits on. It is a direction indicator, deliberately not a range one.
- The spine is tinted a shade lighter than the flanks, so the rule that broadsides go on the
  flanks and vitals down the spine is visible before it gets broken.
- Each battle panel shows crew and how many guns still have hands on them, going red when any
  gun falls silent. That is the read grape shot is played against.

Ownership is carried by the hull, not the cargo: both ships mount the same parts in the same
colours, so each sits on a flat ellipse in its player colour. A panel flashes when its ship is
hit, since the structure bar barely moves on a single ball.

### Chaos rules

Two rules do most of the storytelling, and neither is there for balance.

- Severed sections break away. After any cell dies, flood-fill from the helm. Anything no
  longer connected drifts off, guns and all.
- Magazines detonate, damaging every neighbouring cell, which can chain.

In practice a magazine goes up in about a quarter of battles and something is dismasted in
most of them.

### Ending a round

- Helm destroyed, or every cell gone: immediate loss.
- Overtime from 20s: gunnery gets steadily deadlier, up to 2.6x, until something breaks.
- Hard stop at 40s: the ship with more surviving structure wins; an exact tie is a draw.

The overtime ramp replaced a bare timeout, and the shape of it is the point. A flat damage
spike at a deadline ends rounds almost instantly and whoever fires first wins regardless of
what they built, which severs the connection between the build and the result; a gentle
per-second ramp keeps that connection and still forbids a stalemate. Draws are now under 1%.

Battles settle in 13-17 seconds, and 15-25 is where shipped autobattlers put their combats.
This game's problem was never length — it was that most of the length was empty.

## What earns its keep

`tools/ablate.js` replays an identical grid of 504 battles with one mechanic disabled at a
time and reports how often the winner changes. This is the evidence behind what is in the
game and what was taken out.

| Mechanic disabled | Winner flips | Effect |
| ----------------- | ------------ | ------ |
| Orbiting / broadside arcs | 58% | ships joust bow-on, battles run 22s |
| Holes let shot through | 38% | decisive endings collapse 99% -> 13%, battles run 38s |
| Same orbit sense for both ships | 26% | parallel courses: the original dead-air bug |
| Broadsides answer to either beam | 23% | the layout decision disappears |
| Ships run when crowded | 20% | kiting returns, battles run 18s |
| Per-hull damage pacing | 14% | round 5 collapses to a 9s coin flip |
| Grape kills a whole man per pellet | 12% | one volley silences a ship outright |
| Grape shot never used | 15% | the live decision genuinely carries the battle |
| Staggered battery | 6% | outcomes barely move, but the volleys clump |
| Heavy timber soak | 0% | without it heavy timbers is just costlier hull |
| Severing | 0% | balance-neutral, kept for drama |
| Magazine detonation | 0% | balance-neutral, kept for drama |
| Magazine required to fire | 0% | balance-neutral, but it is why the magazine part exists |

Two mechanics carry the game outright — broadside arcs with orbiting and shot passing
through holes — followed by the steering rules that make an engagement happen at all, and
the grape/round-shot toggle. Everything else is either flavour that pays for itself in
stories, or a rule that justifies a part.

Note how differently the two harnesses read the same change. Giving both ships the same orbit
sense flips only 26% of winners, so `ablate.js` calls it middling; `watch.js` shows it taking
empty air from 86% to 30% and arena-edge time from 10% to zero. Whether a battle is fair and
whether it is worth watching are separate measurements and neither substitutes for the
other.

Removed after measuring, all of it invisible or inert:

- A 1.2s grace period before guns could open fire. Zero effect on any measure.
- Crew shortages also slowing the ship. Shifted win rates 9% through a channel no player
  could see or reason about.
- Per-cell repair, replaced by one Refit button.
- A lock-in bonus of up to 3 scrap for building quickly. A rule to learn for a rounding error.
- A crew tiebreak on timeout, on top of the structure comparison.
- An `ammoLock` field that was written and decremented but never read.

## Feel

Small things, all cheap:

- The camera closes in as the ships close, so an engagement fills the screen.
- A magazine detonation shakes the camera.
- The killing blow eases into slow motion over the verdict delay before the result screen.
- Wind streaks drift downwind across the sea, which is what makes the wind direction legible
  without reading the dial.
- The battery starts out of step and each reload varies, so a broadside rolls down the side
  instead of clapping. Perfectly synchronised guns gave one event per reload cycle however many
  guns were aboard, which measured as most of the empty air.
- A status panel flashes only when something aboard actually breaks. It used to flash on every
  graze, which on a ship of the line meant permanently, which meant nothing.

The chrome is square: no rounded corners anywhere. Panels are framed with a double rule — an
outer hairline, a dark gutter, an inner hairline — the way a chart is ruled, and accent colour
arrives as a rule along one edge rather than as a soft-cornered pill. Type is three families and
six sizes and no others: a slab serif for every heading, label and number, a humanist sans for
running text, and a monospace for part glyphs only, since those are the characters the deck
itself draws.

## Running it

No build step and no dependencies. three.js is vendored in `vendor/`.

```
./tools/dev.sh                 static server on 8123 plus headless Chrome for screenshots
./tools/dev.sh stop            tear both down
open http://127.0.0.1:8123/index.html
```

Any static server works; the game needs http for ES modules.

### Dev harness

`?dev=` switches on a harness that plays the game itself, so a change can be watched without
clicking through two build phases per round.

| URL | Effect |
| --- | ------ |
| `?dev=1` | manual play, plus a Fill button in the build panel |
| `?dev=brawler,sniper` | both sides auto-built each round, overlays auto-advance |
| `&x=3` | run at 3x speed |
| `&round=5` | start at round 5, with the scrap the earlier rounds would have granted |
| `&stop=2` | autoplay round 1, then hold round 2's build phase open, purse unspent |
| `&hold=1` | autoplay, but stop on each result screen |
| `&loop=1` | keep starting fresh matches (off by default) |

Archetypes: `brawler`, `massed`, `sniper`, `harasser`, `crusher`, `mixed`. Autoplay stops after
one match — looping for ever pins a CPU core, which is exactly what a forgotten headless tab
once did.

### Tuning

Every number that affects play or feel is in `src/config.js`, part statistics in
`src/data/parts.js`, hull shapes in `src/data/hulls.js` (drawn as ASCII), and every colour in
`src/theme.js`. The tools read the same config, so a change is measurable in seconds:

```
node tools/watch.js         is a battle worth watching: empty air, dead stretches, orbiting
node tools/balance.js       matchups per hull, graded, worst pairing named
node tools/parts.js         per part, across hundreds of random builds: dominant or a trap
node tools/tune.js          sweep one constant and see what it costs on both counts
node tools/match.js 40      full 5-round matches: economy, hull fill rates, sweep frequency
node tools/ablate.js        disables one mechanic at a time and reports what changes
node tools/events.js        confirms detonations, severings and dismastings actually fire
node tools/bench.js         simulation throughput, which sets how fast every question is answered
node tools/shot.js out.png "1500 ;; ovBtn() ;; 800" "?dev=brawler,crusher"
```

`watch.js` is the one to reach for first on any question of how a battle feels. It reports the
fraction of the battle with nothing in the air, the longest dead stretch, how often a loaded gun
cannot reach or cannot bear, how far the pair drifts from the middle, and how many revolutions
they complete. Those numbers, not the win rates, are what the last pass was aimed at.

`tune.js` sweeps a single constant across a grid of values and prints watchability and fairness
side by side, so "what should this number be" gets an answer rather than an opinion. It is the
counterpart to `ablate.js`, which only answers "does this mechanic matter at all".

`tools/match.js` also asserts the cross-round invariants: destroyed cells are cleared from
the design, the helm is always restored, no part sits off-hull, scrap never goes negative.

`tools/shot.js` steps can be `@path/to/file.js` to evaluate a file in the page, which avoids
fighting shell escaping for anything non-trivial.

## Layout

```
src/config.js      every gameplay and feel constant
src/theme.js       every colour
src/match.js       scores, purses, hull progression, intel. Pure, no DOM
src/autobuild.js   greedy ship builder: bot opponent and Fill button
src/dev.js         URL-driven dev harness, inert without ?dev
src/main.js        presentation and flow: phases, input, render loop
src/sim/           deterministic battle core: rng, ship, battle. No renderer, no DOM
src/data/          parts and hull shapes
src/render/        three.js scene, ship meshes, particles, glyph textures
src/ui/            build-phase panel, HUD chrome
tools/             headless harnesses and the CDP screenshot driver
```

Rendering is three.js with an orthographic top-down camera and placeholder boxes.

Ships batch as one instanced deck plus one instanced layer per part type aboard, so the
draw-call count follows the variety of parts rather than the number of cells. Per-instance
colour carries part identity and damage tint together, which lets a single material serve
every box on every ship. `buildLayer()` in `render/shipView.js` is the seam a real asset
replaces: one loaded mesh per part type, instanced across the cells carrying it.

Particles are three more instanced meshes. Everything lies flat on the water and blends
additively, so the flat rotation bakes into the geometry and per-instance opacity rides in
`instanceColor` -- fading additively is the same as darkening.

That structure, plus dirty-tracking every per-frame DOM write and only resizing the canvas
when it changed, took the largest hull from 238 draw calls and 639 scene objects to 36 and 45,
render from 1.7ms to 0.7ms, and the worst frame from 33ms to under 2ms. `src/perf.js` keeps
rolling frame stats; read them with `__game.perf.snapshot(__game.sceneCtl.renderer)`.

Everything in `sim/` is pure and deterministic: seeded, no renderer dependency, fixed 60Hz
ticks, inputs applied between ticks. Two clients running the same seed and the same input
stream produce the same battle, which is what makes the WebSocket version a matter of
relaying ammunition toggles and lock-ins rather than a rewrite.

The simulation also has to be fast, because every question above is answered by running
thousands of battles and that cost compounds across a session. A pass over the hot path — an
integer cell grid in place of string keys, maintained live-cell counts instead of filtering
arrays per tick, in-place projectile compaction, hoisted trigonometry, and `Math.sqrt` in place
of `Math.hypot` — took throughput from 1050 battles a second to about 1600. `tools/bench.js`
reports it; at roughly 15,000 simulated seconds per real second the browser has three orders of
magnitude of headroom over the one battle it actually needs.

## Open items

- Several pure-build matchups are still past 7-3, mostly involving the carronade at large hull
  sizes and the swivel at small ones. Read alongside `tools/parts.js` before nerfing anything:
  the pure-build grid is bimodal by construction, and a greedy bot's blind spot looks exactly
  like an imbalance.
- Long guns are hard to fit. Restricting them to the bow was right — a gun that always bears
  cannot also be spammable — but between the magazine, the foremast and three rows of bow, few
  builds carry any, and `parts.js` cannot get a reading on a part nobody takes.
- Nothing in the interface explains why a battle was lost. The genre answer is a post-battle
  per-part damage summary rather than a running log; a log says what happened, a summary says
  which decision was wrong.
- The engaged beam being drawn at random is a legible gamble only if the player is told the odds.
  Right now they are not told at all.
- No networking yet. The sim is ready; `main.js` is the part that changes.
