// Running battles headlessly. Two functions, because there are two questions.
//
//   playBattle    who won, how long, how much was left. What balance and ablation need.
//   measureBattle what the battle looked like while it was being decided. What watch.js needs.
//
// Both take a `mods` bundle rather than importing src/ directly, so tools/variant.js can substitute
// a patched copy of the source tree. Build one with variant.js, or hand in the real modules.

import { makeBot, REACTION } from '../src/bot.js';

export const pct = (n) => `${(n * 100).toFixed(0)}%`;

// Every harness derives a battle's wind from its seed the same way.
export function windForSeed(seed) {
  return (seed % 360) * (Math.PI / 180);
}

export function budgetFor(rounds, hullIndex) {
  return rounds.slice(0, hullIndex + 1).reduce((sum, r) => sum + r.scrap, 0);
}

const TAU = Math.PI * 2;

function wrap(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

function start(mods, designs, hullIndex, seed, opts) {
  const battle = mods.battle.createBattle({
    designs,
    hullIndex,
    seed,
    windTo: opts.windTo ?? windForSeed(seed),
  });
  return { battle, bot: makeBot(battle, { mode: opts.grape === false ? 'round' : 'grape' }) };
}

// Run to a conclusion. Steps in the bot's reaction interval rather than tick by tick: advance()
// subdivides into fixed ticks itself, so the battle is identical, and there is nothing to do in
// between. Effects are drained as we go, because nothing in sim/ clears them and a long battle
// otherwise accumulates a few thousand objects nobody reads.
export function playBattle(mods, designs, hullIndex, seed, opts = {}) {
  const { config, ship } = mods;
  const { battle, bot } = start(mods, designs, hullIndex, seed, opts);
  let guard = 0;
  while (!battle.over && guard++ < 60 / REACTION) {
    bot.update(REACTION);
    battle.advance(REACTION);
    battle.effects.length = 0;
  }
  return {
    battle,
    winner: battle.winner,
    time: battle.time,
    reason: battle.reason,
    decisive: battle.time < config.BATTLE_CAP - 0.1,
    // One entry per seat, so a melee reads the same way a duel does.
    struct: battle.ships.map((s) => ship.structureFraction(s)),
    // Best first. Null unless the battle actually concluded, which is what the guard above allows.
    placing: battle.placing,
  };
}

// Why is a ship not shooting? Only interesting once it has something loaded: 'far' means no loaded
// gun reaches, 'arc' means one would reach but points the wrong way. The two want completely
// different fixes. A broadside ship always has an idle off-side battery, so the test is whether
// nothing can bear, not whether anything cannot.
//
// The arc test is the simulation's own, from mods.gunnery, rather than a copy: this file used to
// carry its own version, which silently went stale when guns stopped answering to either beam.
function whyQuiet(bears, me, foe, now) {
  const dx = foe.x - me.x;
  const dz = foe.z - me.z;
  const distSq = dx * dx + dz * dz;
  let loaded = false;
  let inRange = false;
  for (const gun of me.guns) {
    if (!gun.manned || now < gun.readyAt) continue;
    loaded = true;
    if (distSq > gun.rangeSq) continue;
    inRange = true;
    if (bears(gun, me, dx, dz, distSq)) return 'bears';
  }
  if (!loaded) return 'reloading';
  return inRange ? 'arc' : 'far';
}

// A battle watched rather than scored. See tools/watch.js for what each number means and why.
//
// Two, three or four ships. Every pairwise number -- par, range, revs -- is a mean over the
// unordered pairs, and every per-ship number is a mean over the seats, so with two ships each one
// reduces arithmetically to the single pair or the single opposed couple it used to be. That is not
// a hope: `node tools/watch.js` and `node tools/balance.js` are byte-identical across the change.
export function measureBattle(mods, designs, hullIndex, seed, opts = {}) {
  const { ship, config, gunnery } = mods;
  const { battle, bot } = start(mods, designs, hullIndex, seed, opts);
  const bears = gunnery.bears;
  const ships = battle.ships;
  const n = ships.length;

  // Every unordered pair, once, with its own bearing history: revs is about how far the pair has
  // swung about each other, which is not a quantity a centroid can carry.
  const pairs = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ships[i];
      const b = ships[j];
      pairs.push({ a, b, last: Math.atan2(b.x - a.x, -(b.z - a.z)), turned: 0 });
    }
  }

  let ticks = 0;
  let dryTicks = 0;
  let gap = 0;
  let currentGap = 0;
  let edgeTicks = 0;
  let farTicks = 0;
  let arcTicks = 0;
  let driftSum = 0;
  let parSum = 0;
  let rangeSum = 0;
  let volleys = 0;
  let open = null;
  const live = [];
  const timeline = [];

  let guard = 0;
  while (!battle.over && guard++ < 90 / config.TICK) {
    bot.update(config.TICK);
    battle.advance(config.TICK);
    ticks++;

    let fired = 0;
    for (const e of battle.effects) if (e.type === 'muzzle') fired++;
    battle.effects.length = 0;
    volleys += fired;
    if (open === null && fired > 0) open = battle.time;

    // Why each ship is quiet, judged against the ship it is actually fighting. A melee ship's guns
    // can bear on someone it is not steering at, which counts: what is being measured is whether
    // the battery has anything to do, not whether it is obeying orders.
    for (let i = 0; i < n; i++) {
      const foe = ships[i].target;
      if (!foe) continue;
      const why = whyQuiet(bears, ships[i], foe, battle.time);
      if (why === 'far') farTicks++;
      else if (why === 'arc') arcTicks++;
    }

    // Dead time: nothing in the air and nothing leaving a barrel.
    if (battle.projectiles.length === 0 && fired === 0) {
      dryTicks++;
      currentGap += config.TICK;
      if (currentGap > gap) gap = currentGap;
    } else {
      currentGap = 0;
    }

    // Where the fight is happening: whoever is still in it. A struck ship stops sailing where she
    // died, so leaving her in drags the centroid and the mean range toward a hulk. Below two afloat
    // there is no fight left to locate -- the tick the battle ends on -- so the whole field stands
    // in, and that is also what keeps these numbers exactly what a duel used to report.
    live.length = 0;
    for (let i = 0; i < n; i++) if (!ships[i].out) live.push(ships[i]);
    if (live.length < 2) {
      live.length = 0;
      for (let i = 0; i < n; i++) live.push(ships[i]);
    }

    let cx = 0;
    let cz = 0;
    for (let i = 0; i < live.length; i++) {
      cx += live[i].x;
      cz += live[i].z;
    }
    const drift = Math.hypot(cx / live.length, cz / live.length);
    driftSum += drift / battle.arenaRadius;

    // The arena hauls a hulk back as readily as a fighting ship, and either one out here is the
    // camera's problem, so this counts every seat.
    for (let i = 0; i < n; i++) {
      if (Math.hypot(ships[i].x, ships[i].z) > battle.edge) {
        edgeTicks++;
        break;
      }
    }

    let parTick = 0;
    let rangeTick = 0;
    let livePairs = 0;
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        const a = live[i];
        const b = live[j];
        parTick += a.cos * b.cos + a.sin * b.sin; // cos of the angle between the two headings
        rangeTick += Math.hypot(b.x - a.x, b.z - a.z);
        livePairs++;
      }
    }
    const par = parTick / livePairs;
    const range = rangeTick / livePairs;
    parSum += par;
    rangeSum += range;

    for (let i = 0; i < pairs.length; i++) {
      const p = pairs[i];
      const bearing = Math.atan2(p.b.x - p.a.x, -(p.b.z - p.a.z));
      p.turned += Math.abs(wrap(bearing - p.last));
      p.last = bearing;
    }

    if (opts.trace && ticks % 30 === 0) {
      timeline.push(
        `    ${battle.time.toFixed(1)}s  range ${range.toFixed(0).padStart(3)}  ` +
          `midpoint ${drift.toFixed(0).padStart(3)}  ` +
          `par ${par.toFixed(2).padStart(5)}  ` +
          `shot ${String(battle.projectiles.length).padStart(2)}  ` +
          `struct ${ships.map((s) => pct(ship.structureFraction(s))).join('/')}`,
      );
    }
  }

  let turned = 0;
  for (let i = 0; i < pairs.length; i++) turned += pairs[i].turned;

  return {
    time: battle.time,
    winner: battle.winner,
    reason: battle.reason,
    decisive: battle.time < config.BATTLE_CAP - 0.1,
    placing: battle.placing,
    open: open ?? battle.time,
    dry: dryTicks / ticks,
    far: farTicks / (ticks * n),
    arc: arcTicks / (ticks * n),
    gap,
    edge: edgeTicks / ticks,
    drift: driftSum / ticks,
    par: parSum / ticks,
    range: rangeSum / ticks,
    revs: turned / pairs.length / TAU,
    volleys,
    timeline,
  };
}
