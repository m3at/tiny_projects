// Shared bits of the headless harnesses. Deliberately dependency-free of src/ so that
// tools/ablate.js can apply them to patched copies of the source tree.

export const pct = (n) => `${(n * 100).toFixed(0)}%`;

// Every harness derives a battle's wind from its seed the same way.
export function windForSeed(seed) {
  return (seed % 360) * (Math.PI / 180);
}

export function budgetFor(rounds, hullIndex) {
  return rounds.slice(0, hullIndex + 1).reduce((sum, r) => sum + r.scrap, 0);
}

// The stand-in for a player's ammunition decision, used everywhere a bot needs one.
//
// Grape only pays while there is a crew left to break and guns for them to leave silent. Once
// the enemy deck is quiet, or its crew is too deep to shoot away, round shot is what sinks a
// ship: the win goes to whoever takes the helm, and grape barely scratches timber. An earlier
// version keyed off raw damage numbers and quietly went all-grape for any ship with light guns,
// which meant the swivel archetype spent every battle doing no structural damage at all.
export function chooseAmmo(me, enemy) {
  const manned = enemy.guns.reduce((n, g) => n + (g.cell.alive && g.manned ? 1 : 0), 0);
  if (enemy.crew <= 0 || manned === 0) return 'round';
  const best = me.guns.reduce((m, g) => Math.max(m, g.spec.round.damage), 0);
  return enemy.crew <= 6 || best <= 2 ? 'grape' : 'round';
}

export function applyBotAmmo(battle, me, enemy) {
  battle.setAmmo(me.index, chooseAmmo(me, enemy));
}

const TAU = Math.PI * 2;

function wrap(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

// Why is a ship not shooting? Only interesting once it has something loaded: 'far' means no
// loaded gun reaches, 'arc' means one would reach but is pointing the wrong way. The two
// want completely different fixes. A broadside ship always has an idle off-side battery, so
// the test is whether nothing can bear, not whether anything cannot.
function whyQuiet(me, foe) {
  const dist = Math.hypot(foe.x - me.x, foe.z - me.z);
  const bearing = Math.atan2(foe.x - me.x, -(foe.z - me.z));
  let loaded = false;
  let inRange = false;
  for (const gun of me.guns) {
    if (!gun.cell.alive || !gun.manned || gun.reloadLeft > 0) continue;
    loaded = true;
    if (dist > gun.spec.range) continue;
    inRange = true;
    for (const centre of gun.arcs) {
      if (Math.abs(wrap(bearing - (me.heading + centre))) <= gun.halfArc) return 'bears';
    }
  }
  if (!loaded) return 'reloading';
  return inRange ? 'arc' : 'far';
}

// A battle watched rather than scored: what does the round look like while it is being
// decided? See tools/watch.js for what each number means and why it is here.
export function measureBattle(mods, designs, hullIndex, seed, opts = {}) {
  const { ship, battle: battleMod, config } = mods;
  const battle = battleMod.createBattle({
    designs,
    hullIndex,
    seed,
    windTo: opts.windTo ?? windForSeed(seed),
  });
  const [a, b] = battle.ships;

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
    applyBotAmmo(battle, a, b);
    applyBotAmmo(battle, b, a);
    battle.advance(config.TICK);
    ticks++;

    let fired = 0;
    for (const e of battle.effects) if (e.type === 'muzzle') fired++;
    battle.effects.length = 0;
    volleys += fired;
    if (open === null && fired > 0) open = battle.time;

    for (const [me, foe] of [
      [a, b],
      [b, a],
    ]) {
      const w = whyQuiet(me, foe);
      if (w === 'far') farTicks++;
      else if (w === 'arc') arcTicks++;
    }

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
    const edge = config.ARENA_RADIUS * 0.8;
    if (Math.hypot(a.x, a.z) > edge || Math.hypot(b.x, b.z) > edge) edgeTicks++;
    parSum += Math.cos(a.heading - b.heading);
    const range = Math.hypot(b.x - a.x, b.z - a.z);
    rangeSum += range;

    const bearing = Math.atan2(b.x - a.x, -(b.z - a.z));
    turned += Math.abs(wrap(bearing - lastBearing));
    lastBearing = bearing;

    if (opts.trace && ticks % 30 === 0) {
      timeline.push(
        `    ${battle.time.toFixed(1)}s  range ${range.toFixed(0).padStart(3)}  ` +
          `midpoint ${drift.toFixed(0).padStart(3)}  ` +
          `par ${Math.cos(a.heading - b.heading).toFixed(2).padStart(5)}  ` +
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

// Run a battle to its conclusion with bots working the ammunition. `mods` supplies the
// modules so a patched source tree can be substituted.
export function playBattle({ ship, battle: battleMod, config }, designs, hullIndex, seed, opts = {}) {
  const battle = battleMod.createBattle({
    designs,
    hullIndex,
    seed,
    windTo: opts.windTo ?? windForSeed(seed),
  });
  let guard = 0;
  while (!battle.over && guard++ < 60 / config.TICK) {
    if (opts.grape === false) {
      battle.setAmmo(0, 'round');
      battle.setAmmo(1, 'round');
    } else {
      applyBotAmmo(battle, battle.ships[0], battle.ships[1]);
      applyBotAmmo(battle, battle.ships[1], battle.ships[0]);
    }
    battle.advance(config.TICK);
    // The consumer owns draining these. Leaving them to pile up is a slow leak: a long battle
    // accumulates a few thousand effect objects nobody reads.
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
