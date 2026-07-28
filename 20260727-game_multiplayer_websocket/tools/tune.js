// Sweep one number and watch what it does to both the balance and the watchability.
//
//   node tools/tune.js                      list the sweeps
//   node tools/tune.js orbit-retreat        run one
//   node tools/tune.js all                  run all of them
//
// This is the tool for "does this constant matter, and which value is best" questions.
// ablate.js answers "does this mechanic matter at all" by deleting it; this answers "what
// should it be" by trying the alternatives on an identical grid of battles. Same patched-copy
// trick, so production code stays clean.
//
// Read the columns as a pair: `dry`/`gap`/`far`/`arc` say whether the battle is worth
// watching, `spread` and `sweep` say whether it is still fair. A value that empties the dry
// time and flattens the archetypes into a coin flip has won nothing.

import { pct, budgetFor, measureBattle } from './harness.js';
import { variant, cleanupVariants, checkPatches } from './variant.js';

// name -> { file, find, values: [[label, replacement], ...] }
// `find` must appear verbatim in the file, so a stale sweep fails loudly.
const SWEEPS = {
  'orbit-retreat': {
    file: 'config.js',
    find: 'export const ORBIT_RETREAT = 0;',
    values: [0, 0.25, 0.5, 0.75, 1].map((v) => [String(v), `export const ORBIT_RETREAT = ${v};`]),
  },
  // Does a broadside answer to either beam, or only to the flank it sits on? One flank keeps a
  // real build decision (mass for punch, split for consistency, against a randomly drawn
  // engaged side); either beam removes that axis but keeps every gun working.
  'broadside-arcs': {
    // Same patch as tools/ablate.js "either-beam broadsides": taking the absolute value of the dot
    // product mirrors the arc onto the other beam.
    file: 'sim/gunnery.js',
    find: '  const dot = dx * ax + dz * az;\n  if (dot <= 0) return false;\n  return dot * dot >= gun.cosHalfArcSq * distSq;',
    values: [
      [
        'one-flank',
        '  const dot = dx * ax + dz * az;\n  if (dot <= 0) return false;\n  return dot * dot >= gun.cosHalfArcSq * distSq;',
      ],
      [
        'either',
        '  const dot = Math.abs(dx * ax + dz * az);\n  return dot * dot >= gun.cosHalfArcSq * distSq;',
      ],
    ],
  },
  'orbit-close': {
    file: 'config.js',
    find: 'export const ORBIT_CLOSE = 0.55;',
    values: [0.35, 0.55, 0.75, 1].map((v) => [String(v), `export const ORBIT_CLOSE = ${v};`]),
  },
  'orbit-tolerance': {
    file: 'config.js',
    find: 'export const ORBIT_TOLERANCE = 0.6;',
    values: [0.25, 0.4, 0.6, 0.9, 1.3].map((v) => [
      String(v),
      `export const ORBIT_TOLERANCE = ${v};`,
    ]),
  },
  'preferred-range': {
    file: 'config.js',
    find: 'export const PREFERRED_RANGE_FRACTION = 0.85;',
    values: [0.55, 0.7, 0.85, 1].map((v) => [
      String(v),
      `export const PREFERRED_RANGE_FRACTION = ${v};`,
    ]),
  },
  'start-offset': {
    file: 'config.js',
    find: 'export const START_OFFSET = { x: 9, z: 24 };',
    values: [12, 18, 24, 32, 40].map((z) => [
      `z=${z}`,
      `export const START_OFFSET = { x: 9, z: ${z} };`,
    ]),
  },
  'battle-cap': {
    file: 'config.js',
    find: 'export const BATTLE_CAP = 40; // hard stop',
    values: [25, 30, 40, 60].map((v) => [
      String(v),
      `export const BATTLE_CAP = ${v}; // hard stop`,
    ]),
  },
  overtime: {
    file: 'config.js',
    find: 'export const OVERTIME_AT = 20;',
    values: [12, 16, 20, 28, 999].map((v) => [String(v), `export const OVERTIME_AT = ${v};`]),
  },
  wind: {
    file: 'config.js',
    find: 'export const WIND_MIN = 0.35;',
    values: [0.2, 0.35, 0.5, 0.7, 1].map((v) => [String(v), `export const WIND_MIN = ${v};`]),
  },
  separation: {
    file: 'config.js',
    find: 'export const MIN_SEPARATION = 9;',
    values: [7, 9, 13, 18].map((v) => [String(v), `export const MIN_SEPARATION = ${v};`]),
  },
  'reload-stagger': {
    file: 'config.js',
    find: 'export const RELOAD_STAGGER = 0.35;',
    values: [0, 0.35, 0.7, 1].map((v) => [String(v), `export const RELOAD_STAGGER = ${v};`]),
  },
  'reload-jitter': {
    file: 'config.js',
    find: 'export const RELOAD_JITTER = 0.35;',
    values: [0.08, 0.22, 0.35, 0.5].map((v) => [String(v), `export const RELOAD_JITTER = ${v};`]),
  },
  'grape-crew': {
    file: 'config.js',
    find: 'export const GRAPE_CREW_SCALE = 0.15;',
    values: [0.08, 0.15, 0.25, 0.4, 1].map((v) => [String(v), `export const GRAPE_CREW_SCALE = ${v};`]),
  },
  // Battle length per round. The array is per hull, so a sweep scales the whole thing and
  // tools/watch.js reports what it did to each round in turn.
  'hull-damage': {
    file: 'config.js',
    find: 'export const HULL_DAMAGE = [1.05, 0.5, 0.36, 0.24, 0.2];',
    values: [0.6, 0.8, 1, 1.3].map((k) => [
      `x${k}`,
      `export const HULL_DAMAGE = [${[1.05, 0.5, 0.36, 0.24, 0.2]
        .map((v) => Math.round(v * k * 100) / 100)
        .join(', ')}];`,
    ]),
  },
};

// The fixed grid every value replays. Sides swap on odd seeds so the orbit sense, which is
// the same for both ships, cannot favour one seat.
const HULLS = [0, 2, 4];
const SEEDS = 12;

function grid(mod) {
  const names = Object.keys(mod.autobuild.ARCHETYPES);
  const rows = [];
  for (const hullIndex of HULLS) {
    const budget = budgetFor(mod.config.ROUNDS, hullIndex);
    for (const a of names) {
      for (const b of names) {
        if (a === b) continue;
        for (let s = 0; s < SEEDS; s++) {
          const flip = s % 2 === 1;
          const types = flip ? [b, a] : [a, b];
          const seed = s * 7919 + hullIndex;
          const designs = types.map((t) => {
            const d = mod.ship.createDesign();
            mod.autobuild.autoBuild(d, hullIndex, budget, mod.autobuild.ARCHETYPES[t]);
            return d;
          });
          const m = measureBattle(mod, designs, hullIndex, seed);
          const seatWon = m.winner === null ? null : m.winner === 0;
          rows.push({ a, b, won: seatWon === null ? null : seatWon !== flip, ...m });
        }
      }
    }
  }
  return rows;
}

function summarise(rows) {
  const mean = (k) => rows.reduce((s, r) => s + r[k], 0) / rows.length;
  // Win rate per archetype, from its own point of view.
  const rate = {};
  for (const r of rows) {
    rate[r.a] = rate[r.a] || { w: 0, n: 0 };
    rate[r.a].n++;
    if (r.won) rate[r.a].w++;
  }
  const rates = Object.values(rate).map((v) => v.w / v.n);
  return {
    time: mean('time'),
    open: mean('open'),
    dry: mean('dry'),
    far: mean('far'),
    arc: mean('arc'),
    gap: mean('gap'),
    edge: mean('edge'),
    revs: mean('revs'),
    decisive: rows.filter((r) => r.decisive).length / rows.length,
    draws: rows.filter((r) => r.winner === null).length / rows.length,
    // How far apart the best and worst archetype are. 0 would be a coin flip for everyone.
    spread: Math.max(...rates) - Math.min(...rates),
  };
}

const HEAD =
  '  value      time   open   dry   far   arc    gap   edge  revs  decisive  draws  spread';

async function runSweep(name) {
  const sweep = SWEEPS[name];
  if (!sweep) throw new Error(`unknown sweep "${name}", expected one of ${Object.keys(SWEEPS).join(', ')}`);
  console.log(`\n=== ${name}  (${sweep.file}: ${sweep.find})`);
  console.log(HEAD);
  for (const [label, replacement] of sweep.values) {
    const mod = await variant(label, [[sweep.file, sweep.find, replacement]]);
    const s = summarise(grid(mod));
    console.log(
      `  ${label.padEnd(9)} ${s.time.toFixed(1).padStart(4)}s  ${s.open.toFixed(1).padStart(4)}s  ` +
        `${pct(s.dry).padStart(4)}  ${pct(s.far).padStart(4)}  ${pct(s.arc).padStart(4)}  ` +
        `${s.gap.toFixed(1).padStart(4)}s  ${pct(s.edge).padStart(4)}  ${s.revs.toFixed(1).padStart(4)}  ` +
        `${pct(s.decisive).padStart(8)}  ${pct(s.draws).padStart(5)}  ${pct(s.spread).padStart(6)}`,
    );
  }
}

// Every sweep's find-string, as checkPatches wants it.
const asPatches = Object.fromEntries(
  Object.entries(SWEEPS).map(([name, s]) => [name, [[s.file, s.find]]]),
);

const arg = process.argv[2];
if (!arg) {
  // Listing doubles as a health check: a sweep whose target has moved is reported here rather than
  // discovered halfway through a run.
  checkPatches(asPatches);
  console.log('sweeps (all patch targets resolve):');
  for (const [name, s] of Object.entries(SWEEPS)) {
    console.log(`  ${name.padEnd(18)} ${s.values.map(([l]) => l).join(' ')}`);
  }
} else {
  checkPatches({ [arg]: asPatches[arg] ? [[SWEEPS[arg].file, SWEEPS[arg].find]] : [] });
  for (const name of arg === 'all' ? Object.keys(SWEEPS) : [arg]) await runSweep(name);
}

cleanupVariants();
