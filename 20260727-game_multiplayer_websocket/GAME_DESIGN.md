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
  BATTLE  both ships sail themselves, 30s cap
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
nothing by round 5, which is how you get 3-0 sweeps. Unspent scrap carries over.

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
| Heavy timbers   | 3    | 30 | -    | Soaks 3 off every incoming hit                |
| Crew quarters   | 5    | 14 | +3   | Supplies the crew pool                        |
| Mast            | 4    | 11 | 1    | Speed and turn rate                           |
| Powder magazine | 4    | 8  | -    | Needed to fire at all. Detonates when destroyed |
| Swivel gun      | 4    | 11 | 1    | All-round arc, range 26, 1x3, reload 1.3s     |
| Gun deck        | 8    | 17 | 2    | Broadside arc, range 38, 3x7, reload 3.0s     |
| Carronade       | 8    | 14 | 1    | Broadside arc, range 24, 2x19, reload 3.2s    |
| Long gun        | 9    | 14 | 2    | Bow arc, range 60, 1x18, reload 4.4s, halves soak |
| Helm            | free | 20 | -    | Pre-placed at centre. Lose it and you strike  |

Guns need gunners. Crew quarters supply a pool; guns are manned in placement order and any
gun left without crew stays silent. No live magazine means no gun fires at all — one
magazine is a gamble, two is insurance.

The shop offers five part *types* per build phase and you may buy as many of each as you can
afford; buying 38 cells one card at a time would be tedious, and the interesting luck is in
which types you are shown. Rerolling costs 2. Cheap timber is always offered, and powder or
crew are added if you have none, so a hand is never unplayable.

Refit repairs every damaged part at once for half the cost of each, worst first. Clicking
damaged cells one at a time was the same decision wrapped in busywork.

## Combat

Both ships run the same steering logic. They close to their preferred range — derived from
their own weapon mix — then orbit each other. Facing changes constantly, so front, side and
rear armour all matter, and broadsides bear and then stop bearing.

A ship too close for comfort opens the range. A broadside ship only has to sheer off a
little to keep its guns bearing; a bow-gun ship has to actually run, and cannot shoot while
it does. That trade is what keeps long guns honest.

Wind is rolled once per round. Sailing with it is fast, into it is slow, so one side of the
orbit is quicker than the other and the two ships end up contesting the weather gauge.

Note what wind is *not*: there is no wind strength, and rotating the direction through 24
points almost never changes who wins. It is the speed penalty that matters, not the bearing.
An earlier draft of this document claimed the wind created a build decision ("skip a mast in
a light wind") — measurement showed that was wishful thinking, and the claim is gone.

### Weapon roles

Each gun owns a distinct corner of (range, arc, damage per ball):

- Long gun reaches furthest and punches through heavy timbers, but only fires forward, and a
  bow has few cells to mount it in.
- Gun deck is the mid-range broadside workhorse.
- Carronade is brutal inside 24 and useless outside it, and needs only one hand to work.
- Swivel gun is cheap, fires all round, and does almost nothing to armour. It is a grape
  platform.

Intended as soft counters. In practice the measured matchups are close to even at frigate
scale (48-55% across the board) but skew at the extremes, and an all-carronade ship is
weak at every scale in the bot's hands. Rounding that out is the top open balance item.

### The live decision

The one thing you touch during a battle is your ammunition, switchable at any time:

- Round shot smashes hull, brings down masts, sinks.
- Grape shot shreds crew and barely marks the timbers.

Because crew man the guns, grape silences a broadside without scratching the ship. So you
spend the battle reading the enemy: their gun deck is untouched but they are down to two
crew quarters, stay on grape and their guns go quiet. Switching costs a reload.

Player 1 presses A, player 2 presses L.

### Chaos rules

Two rules do most of the storytelling, and neither is there for balance.

- Severed sections break away. After any cell dies, flood-fill from the helm. Anything no
  longer connected drifts off, guns and all.
- Magazines detonate, damaging every neighbouring cell, which can chain.

In practice a magazine goes up in about a quarter of battles and something is dismasted in
most of them.

### Ending a round

- Helm destroyed, or every cell gone: immediate loss.
- Timeout at 30s: the ship with more surviving structure wins; an exact tie is a draw.

Battles settle in roughly 15-22 seconds across every hull size.

## What earns its keep

`tools/ablate.js` replays an identical grid of 504 battles with one mechanic disabled at a
time and reports how often the winner changes. This is the evidence behind what is in the
game and what was taken out.

| Mechanic disabled | Winner flips | Effect |
| ----------------- | ------------ | ------ |
| Holes let shot through | 32% | decisive endings collapse 77% -> 2%, every battle times out |
| Orbiting / broadside arcs | 22% | ships joust bow-on; 44% swing in archetype win rates |
| Wind speed penalty | 22% | 41% swing in archetype win rates |
| Grape shot never used | 27% | the live decision genuinely carries the battle |
| Heavy timber soak | 2% | small, but without it heavy timbers is just costlier hull |
| Severing | 2% | balance-neutral, kept for drama |
| Magazine required to fire | 1% | balance-neutral, but it is why the magazine part exists |
| Magazine detonation | 0% | balance-neutral, kept for drama |

Four mechanics carry the game: **shot passing through holes**, **broadside arcs with
orbiting**, **the wind's speed penalty**, and **the grape/round-shot toggle**. Everything
else is either flavour that pays for itself in stories, or a rule that justifies a part.

Removed after measuring, all of it invisible or inert:

- A 1.2s grace period before guns could open fire. Zero effect on any measure.
- Crew shortages also slowing the ship. Shifted win rates 9% through a channel no player
  could see or reason about.
- Per-cell repair, replaced by one Refit button.
- A lock-in bonus of up to 3 scrap for building quickly. A rule to learn for a rounding error.
- A crew tiebreak on timeout, on top of the structure comparison.
- An `ammoLock` field that was written and decremented but never read.

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

Archetypes: `brawler`, `sniper`, `harasser`, `crusher`. Autoplay stops after one match —
looping for ever pins a CPU core, which is exactly what a forgotten headless tab once did.

### Tuning

Every number that affects play or feel is in `src/config.js`, part statistics in
`src/data/parts.js`, hull shapes in `src/data/hulls.js` (drawn as ASCII), and every colour in
`src/theme.js`. The tools read the same config, so a change is measurable in seconds:

```
node tools/balance.js       archetype matchups per hull: win rates, battle length, decisiveness
node tools/match.js 40      full 5-round matches: economy, hull fill rates, sweep frequency
node tools/ablate.js        disables one mechanic at a time and reports what changes
node tools/events.js        confirms detonations, severings and dismastings actually fire
node tools/shot.js out.png "1500 ;; ovBtn() ;; 800"    screenshot and console capture over CDP
```

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

Rendering is three.js with an orthographic top-down camera and placeholder boxes. Each part
is one coloured box plus a glyph decal, all built in `buildPart()` inside
`render/shipView.js` — that one function is what a real 3D asset replaces, and the build
phase, damage shading and battle keep working around it.

Everything in `sim/` is pure and deterministic: seeded, no renderer dependency, fixed 60Hz
ticks, inputs applied between ticks. Two clients running the same seed and the same input
stream produce the same battle, which is what makes the WebSocket version a matter of
relaying ammunition toggles and lock-ins rather than a rewrite.
