# Broadside

A two-player hot-seat game about fitting out an age-of-sail ship and then watching it fight.

You spend a timed build phase spending scrap on guns, crew, powder, masts and timber, laid out cell by cell on a hull. Then both ships sail themselves, and the only thing you touch is your ammunition: round shot to smash hull, grape to kill the crew that works their guns.

Five rounds, first to three. Your ship carries its damage into the next round, and the hull grows each time, so round one is a blank slate and the rest are triage: patch the hole in the bow, or leave it and add another gun?

![A round-four engagement between two heavy frigates](screenshot.png)

## Play it

```
./tools/dev.sh                          static server on 8123, plus headless Chrome for the tools
open http://127.0.0.1:8123/index.html
./tools/dev.sh stop                     when you are done
```

It needs to be served over http, because it uses ES modules; opening the file directly will not work. Any static server does — `python3 -m http.server 8123` is enough to just play.

Player 1 presses A to switch ammunition, player 2 presses L. Space locks in a build. M mutes.

## What is in here

No build step, no dependencies, no package.json, and no assets — three.js is vendored, and every sound is synthesised at runtime. It is a directory of text you can read start to finish.

```
index.html  styles.css
src/sim/      the deterministic battle core: seeded, fixed 60Hz ticks, no DOM, no renderer
src/render/   three.js: an instanced ship, instanced particles, an orthographic plan view
src/audio/    every sound, built from oscillators and one buffer of noise
src/ui/       the build panel and the HUD
tools/        headless harnesses, and a CDP driver that plays the game in a real browser
```

The simulation is pure and seeded, so the same seed and the same inputs always replay the same battle. That is what makes the eventual networked version a matter of relaying ammunition toggles rather than a rewrite, and it is also what lets the whole game be measured instead of guessed at:

```
node tools/watch.js     is a battle worth watching: dead air, dead stretches, orbiting
node tools/balance.js   matchups per hull, graded, worst pairing named
node tools/parts.js     per part, across hundreds of random builds: dominant, even, or a trap
node tools/tune.js      sweep any constant and see what it costs on both counts
node tools/ablate.js    disable one mechanic at a time and count what actually changes
node tools/bench.js     simulation throughput, currently about 3,200 battles a second
node tools/audio.js     render every sound offline: clipping, DC offset, onset clicks
node tools/mix.js       what the mixer actually hears once a battle is shouting at it
node tools/frames.js    frame times as a distribution, because stutter lives in the tail
node tools/fill.js      what each layer of the scene costs to draw, one at a time
node tools/playtest.js  plays a whole match through the real interface and complains
```

Nearly every decision in the game was settled by one of those, and several confident guesses were wrong. The ships used to fire one volley and then sail off the map together — that turned out to be a one-line steering bug, not a tuning problem. Grape shot was silently killing an entire crew in a single volley. A ship with crew for half its guns could put every hand on the wrong side and never fire a shot.

## More

- [GAME_DESIGN.md](GAME_DESIGN.md) — the design, the parts, and the reasoning behind both.
- [CLAUDE.md](CLAUDE.md) — working notes: what is load-bearing, what is flavour, and the gotchas
  already paid for.
