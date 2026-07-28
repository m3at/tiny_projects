// Running battles headlessly. Two functions, because there are two questions.
//
//   playBattle    who won, how long, how much was left. What balance and ablation need.
//   measureBattle what the battle looked like while it was being decided. What watch.js needs.
//
// Both take a `mods` bundle rather than importing src/ directly, so tools/variant.js can substitute
// a patched copy of the source tree. Build one with variant.js, or hand in the real modules.

import { makeBot, REACTION } from './bot.js';

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
  return { battle, bot: makeBot(battle, opts.grape === false ? 'round' : 'grape') };
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
    struct: [ship.structureFraction(battle.ships[0]), ship.structureFraction(battle.ships[1])],
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
export function measureBattle(mods, designs, hullIndex, seed, opts = {}) {
  const { ship, config, gunnery } = mods;
  const { battle, bot } = start(mods, designs, hullIndex, seed, opts);
  const bears = gunnery.bears;
  const [a, b] = battle.ships;
  const edge = config.ARENA_RADIUS * 0.8;

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
  let turned = 0;
  let open = null;
  let lastBearing = Math.atan2(b.x - a.x, -(b.z - a.z));
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

    const whyA = whyQuiet(bears, a, b, battle.time);
    if (whyA === 'far') farTicks++;
    else if (whyA === 'arc') arcTicks++;
    const whyB = whyQuiet(bears, b, a, battle.time);
    if (whyB === 'far') farTicks++;
    else if (whyB === 'arc') arcTicks++;

    // Dead time: nothing in the air and nothing leaving a barrel.
    if (battle.projectiles.length === 0 && fired === 0) {
      dryTicks++;
      currentGap += config.TICK;
      if (currentGap > gap) gap = currentGap;
    } else {
      currentGap = 0;
    }

    const mx = (a.x + b.x) / 2;
    const mz = (a.z + b.z) / 2;
    const drift = Math.hypot(mx, mz);
    driftSum += drift / config.ARENA_RADIUS;
    if (Math.hypot(a.x, a.z) > edge || Math.hypot(b.x, b.z) > edge) edgeTicks++;
    parSum += a.cos * b.cos + a.sin * b.sin; // cos of the angle between the two headings
    const range = Math.hypot(b.x - a.x, b.z - a.z);
    rangeSum += range;

    const bearing = Math.atan2(b.x - a.x, -(b.z - a.z));
    turned += Math.abs(wrap(bearing - lastBearing));
    lastBearing = bearing;

    if (opts.trace && ticks % 30 === 0) {
      timeline.push(
        `    ${battle.time.toFixed(1)}s  range ${range.toFixed(0).padStart(3)}  ` +
          `midpoint ${drift.toFixed(0).padStart(3)}  ` +
          `par ${(a.cos * b.cos + a.sin * b.sin).toFixed(2).padStart(5)}  ` +
          `shot ${String(battle.projectiles.length).padStart(2)}  ` +
          `struct ${pct(ship.structureFraction(a))}/${pct(ship.structureFraction(b))}`,
      );
    }
  }

  return {
    time: battle.time,
    winner: battle.winner,
    reason: battle.reason,
    decisive: battle.time < config.BATTLE_CAP - 0.1,
    open: open ?? battle.time,
    dry: dryTicks / ticks,
    far: farTicks / (ticks * 2),
    arc: arcTicks / (ticks * 2),
    gap,
    edge: edgeTicks / ticks,
    drift: driftSum / ticks,
    par: parSum / ticks,
    range: rangeSum / ticks,
    revs: turned / TAU,
    volleys,
    timeline,
  };
}
