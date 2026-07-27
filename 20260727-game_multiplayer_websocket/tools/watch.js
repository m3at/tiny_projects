// Is a battle worth watching? Every other harness measures who wins; this one measures what
// the round looks like while it is being decided.
//
//   node tools/watch.js            all archetype pairs, hulls 0/2/4
//   node tools/watch.js 2          just hull index 2
//   node tools/watch.js 2 trace    second-by-second trace of one battle
//
// The columns, and why each one is here:
//
//   open   when the first gun goes off. The opening run has nothing to watch, so it should
//          be short.
//   dry    fraction of the battle with nothing in the air and no muzzle flash. This is the
//          dead time the player sits through. The single number to drive down.
//   far    fraction of ship-ticks with a gun loaded but the enemy out of every gun's reach.
//          High means the two ships disagree about the range and one of them is being kited.
//   arc    loaded, in range, and still pointing the wrong way. Steering's fault rather than
//          the gunner's; a broadside ship's idle off-side battery does not count against it.
//   gap    longest dry stretch, in seconds. A 40% dry battle made of half-second lulls is
//          fine; one 10-second lull is the thing that feels broken.
//   edge   fraction of ticks where a ship is far enough out that the arena is hauling it
//          back. High means the fight is happening in the corner, not on stage.
//   drift  how far the midpoint between the ships wanders from the centre, as a fraction of
//          the arena radius. A real circling engagement holds near 0.
//   par    mean cos of the angle between the two headings. +1 is both ships sailing the same
//          course side by side, which looks like a parade and drifts off the map. -1 is a
//          mutual orbit, each keeping the other abeam. 0 is crossing.
//   revs   revolutions the pair completes about their midpoint.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { pct, budgetFor, measureBattle } from './lib.js';

const mods = { ship, battle: battleMod, config };
const names = Object.keys(ARCHETYPES);

function buildFor(hullIndex, archetype) {
  const design = ship.createDesign();
  autoBuild(design, hullIndex, budgetFor(config.ROUNDS, hullIndex), ARCHETYPES[archetype]);
  return design;
}

const KEYS = ['time', 'open', 'dry', 'far', 'arc', 'gap', 'edge', 'drift', 'par', 'revs', 'volleys'];

function row(label, m) {
  return (
    `  ${label.padEnd(22)} ${m.time.toFixed(1).padStart(4)}s  ${m.open.toFixed(1).padStart(4)}s  ` +
    `${pct(m.dry).padStart(4)}  ${pct(m.far).padStart(4)}  ${pct(m.arc).padStart(4)}  ` +
    `${m.gap.toFixed(1).padStart(4)}s  ${pct(m.edge).padStart(4)}  ${m.drift.toFixed(2).padStart(5)}  ` +
    `${m.par.toFixed(2).padStart(5)}  ${m.revs.toFixed(1).padStart(4)}  ${m.volleys.toFixed(0).padStart(4)}`
  );
}

const argHull = process.argv[2] !== undefined ? Number(process.argv[2]) : null;
const hulls = argHull !== null ? [argHull] : [0, 2, 4];
const SEEDS = 24;

if (process.argv[3] === 'trace') {
  const h = hulls[0];
  const r = measureBattle(mods, [buildFor(h, 'brawler'), buildFor(h, 'sniper')], h, 4242, {
    trace: true,
  });
  console.log(`brawler vs sniper on the ${HULLS[h].name}: ${r.reason} at ${r.time.toFixed(1)}s`);
  for (const line of r.timeline) console.log(line);
  console.log(
    `  open ${r.open.toFixed(1)}s  dry ${pct(r.dry)}  far ${pct(r.far)}  arc ${pct(r.arc)}  ` +
      `gap ${r.gap.toFixed(1)}s  edge ${pct(r.edge)}  drift ${r.drift.toFixed(2)}  ` +
      `par ${r.par.toFixed(2)}  revs ${r.revs.toFixed(1)}`,
  );
  process.exit(0);
}

console.log(
  '  matchup                 time   open   dry   far   arc    gap   edge  drift    par  revs  volleys',
);
for (const hullIndex of hulls) {
  console.log(`\n=== ${HULLS[hullIndex].name} ===`);
  const totals = Object.fromEntries(KEYS.map((k) => [k, 0]));
  let pairs = 0;

  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const acc = Object.fromEntries(KEYS.map((k) => [k, 0]));
      for (let s = 0; s < SEEDS; s++) {
        const seed = s * 7919 + hullIndex;
        const m = measureBattle(
          mods,
          [buildFor(hullIndex, names[i]), buildFor(hullIndex, names[j])],
          hullIndex,
          seed,
        );
        for (const k of KEYS) acc[k] += m[k];
      }
      for (const k of KEYS) {
        acc[k] /= SEEDS;
        totals[k] += acc[k];
      }
      pairs++;
      console.log(row(`${names[i]} vs ${names[j]}`, acc));
    }
  }

  for (const k of KEYS) totals[k] /= pairs;
  console.log(row('ALL', totals));
}
