// Which mechanics actually change the game?
//
// Each variant is a patched copy of src/ in a temp dir, so production code stays clean.
// For every variant we replay the exact same matchups and seeds as the baseline and report
// how often the winner changed. A mechanic that never flips an outcome is not pulling its
// weight on balance -- though it may still be worth keeping for drama, which this cannot
// measure.
//
//   node tools/ablate.js

import { pct, budgetFor, playBattle } from './harness.js';
import { variant, cleanupVariants, checkPatches } from './variant.js';

// name -> [file, find, replace][]
const VARIANTS = {
  baseline: [],
  // The wind's speed penalty, flattened away.
  'no wind': [['config.js', 'export const WIND_MIN = 0.35;', 'export const WIND_MIN = 1;']],
  // Heavy timbers stop soaking, becoming just expensive hull.
  'no armour soak': [['data/parts.js', 'soak: 2,', 'soak: 0,']],
  // Destroyed cells block shot instead of letting it through to the spine.
  'holes block shot': [
    ['sim/gunnery.js', 'return cell !== null && cell.alive ? cell : null;', 'return cell;'],
  ],
  // Ships always point their bow at the enemy instead of presenting a flank.
  'no orbiting': [['sim/steering.js', "const bias = spec.arc === 'bow' ? 0 : 90;", 'const bias = 0;']],
  // Both ships circle the same way. Opposite senses put them on parallel courses, which is
  // what the game used to do: perfect range-keeping, no shooting, straight off the map.
  'opposed orbits': [
    ['sim/steering.js', 'const hold = bias * battle.sense;', 'const hold = bias * (ship.index === 0 ? 1 : -1);'],
  ],
  // Broadsides answer to either beam, so which flank a gun sits on stops being a decision.
  // Answering to either beam needs two windows, so the single-arc test becomes a pair. The gun
  // keeps its own arc and gains the mirrored one.
  'either-beam broadsides': [
    [
      'sim/gunnery.js',
      '  const dot = dx * ax + dz * az;\n  if (dot <= 0) return false;\n  return dot * dot >= gun.cosHalfArcSq * distSq;',
      '  const dot = Math.abs(dx * ax + dz * az);\n  return dot * dot >= gun.cosHalfArcSq * distSq;',
    ],
  ],
  // A ship driven inside its preferred range turns tail instead of holding the circle.
  'full retreat': [['config.js', 'export const ORBIT_RETREAT = 0;', 'export const ORBIT_RETREAT = 1;']],
  // Every hull takes damage at the same rate, so the big ones die in five seconds.
  'flat hull damage': [
    ['config.js', 'export const HULL_DAMAGE = [1.05, 0.5, 0.36, 0.24, 0.2];', 'export const HULL_DAMAGE = [1, 1, 1, 1, 1];'],
  ],
  // The battery fires as one clap instead of rolling down the side.
  'synchronised battery': [
    ['config.js', 'export const RELOAD_STAGGER = 0.35;', 'export const RELOAD_STAGGER = 0;'],
    ['config.js', 'export const RELOAD_JITTER = 0.35;', 'export const RELOAD_JITTER = 0;'],
  ],
  // Grape kills a whole man per pellet again.
  'unscaled grape': [
    ['config.js', 'export const GRAPE_CREW_SCALE = 0.15;', 'export const GRAPE_CREW_SCALE = 1;'],
  ],
  // Cut-off sections stay attached.
  'no severing': [
    ['sim/ship.js', 'const helm = ship.helm;', 'return [];\n  const helm = ship.helm;'],
  ],
  // Powder goes up quietly.
  'no detonation': [
    ['data/parts.js', 'detonate: { damage: 15, radius: 1 }', 'detonate: { damage: 0, radius: 0 }'],
  ],
  // Guns work with no powder aboard.
  'no magazine rule': [
    ['sim/gunnery.js', 'if (ship.magazines === 0) return;', ''],
  ],
};

// One battle between two archetypes under the given (possibly patched) modules.
function runBattle(mod, hullIndex, aType, bType, seed, opts = {}) {
  const budget = budgetFor(mod.config.ROUNDS, hullIndex);
  const designs = [mod.ship.createDesign(), mod.ship.createDesign()];
  mod.autobuild.autoBuild(designs[0], hullIndex, budget, mod.autobuild.ARCHETYPES[aType]);
  mod.autobuild.autoBuild(designs[1], hullIndex, budget, mod.autobuild.ARCHETYPES[bType]);
  return playBattle(mod, designs, hullIndex, seed, opts);
}

// The fixed grid every variant replays.
const HULLS = [0, 2, 4];
const SEEDS = 14;
function grid(mod, opts) {
  const names = Object.keys(mod.autobuild.ARCHETYPES);
  const rows = [];
  for (const hullIndex of HULLS) {
    for (const a of names) {
      for (const b of names) {
        if (a === b) continue;
        for (let s = 0; s < SEEDS; s++) {
          rows.push({ key: `${hullIndex}:${a}:${b}:${s}`, ...runBattle(mod, hullIndex, a, b, s * 7919 + hullIndex, opts) });
        }
      }
    }
  }
  return rows;
}

// Fail in milliseconds rather than minutes if a patch string has gone stale.
checkPatches(VARIANTS);

console.log('Replaying an identical grid of matchups under each variant.\n');
console.log('variant              battles  meanTime  decisive  winnerFlips  |Δ win rate|');
console.log('-'.repeat(78));

let base = null;
for (const [name, patches] of Object.entries(VARIANTS)) {
  const mod = await variant(name, patches);
  const rows = grid(mod, {});
  const meanTime = rows.reduce((s, r) => s + r.time, 0) / rows.length;
  const decisive = rows.filter((r) => r.decisive).length / rows.length;

  let flips = '-';
  let winShift = '-';
  if (!base) {
    base = rows;
  } else {
    const byKey = new Map(base.map((r) => [r.key, r]));
    flips = pct(rows.filter((r) => byKey.get(r.key).winner !== r.winner).length / rows.length);
    // Largest change in any single archetype's overall win rate.
    const rate = (set) => {
      const m = {};
      for (const r of set) {
        const [, a] = r.key.split(':');
        m[a] = m[a] || { w: 0, n: 0 };
        m[a].n++;
        if (r.winner === 0) m[a].w++;
      }
      return m;
    };
    const ra = rate(base);
    const rb = rate(rows);
    winShift = pct(Math.max(...Object.keys(ra).map((k) => Math.abs(ra[k].w / ra[k].n - rb[k].w / rb[k].n))));
  }
  console.log(
    `${name.padEnd(20)} ${String(rows.length).padStart(7)}  ${meanTime.toFixed(1).padStart(7)}s  ` +
      `${pct(decisive).padStart(8)}  ${String(flips).padStart(11)}  ${String(winShift).padStart(12)}`,
  );
}

// Wind gets a second, sharper test: hold everything else fixed and sweep the direction.
// If the winner never changes, the wind is scenery.
{
  const mod = await variant('wind-sweep', []);
  console.log('\nWind sweep: same designs and seed, wind direction rotated through 24 points.');
  console.log('hull              matchup             distinct outcomes  time spread');
  console.log('-'.repeat(74));
  for (const hullIndex of [0, 2, 4]) {
    for (const [a, b] of [['brawler', 'sniper'], ['harasser', 'crusher']]) {
      const results = [];
      for (let i = 0; i < 24; i++) {
        results.push(runBattle(mod, hullIndex, a, b, 4242 + hullIndex, { windTo: (i / 24) * Math.PI * 2 }));
      }
      const winners = new Set(results.map((r) => String(r.winner)));
      const times = results.map((r) => r.time);
      console.log(
        `  hull ${hullIndex}${' '.repeat(11)}${`${a} vs ${b}`.padEnd(22)}` +
          `${[...winners].join('/').padStart(9)}${' '.repeat(9)}` +
          `${(Math.max(...times) - Math.min(...times)).toFixed(1)}s`,
      );
    }
  }
}

// Grape shot: does having a second ammunition type change anything?
{
  const mod = await variant('grape', []);
  const withGrape = grid(mod, { grape: true });
  const without = grid(mod, { grape: false });
  const byKey = new Map(withGrape.map((r) => [r.key, r]));
  const flips = without.filter((r) => byKey.get(r.key).winner !== r.winner).length / without.length;
  console.log(`\nGrape shot never used: ${pct(flips)} of outcomes change.`);
}

cleanupVariants();
