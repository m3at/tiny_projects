// Determinism guard. Plays a fixed grid of battles and prints a fingerprint of every outcome.
//
//   node tools/golden.js > golden.txt     record
//   node tools/golden.js | diff golden.txt -    check nothing moved
//
// The point is refactoring. `sim/` promises that the same seed and the same input stream produce
// the same battle, which is the property the eventual networked version rests on -- and it is
// also what lets the simulation be rearranged with confidence. Any change that is meant to be
// purely structural must leave this output byte-identical; any change that is meant to alter
// behaviour should change it in a way you can read.
//
// It records more than the winner on purpose. A refactor that flips one cell of damage will not
// usually flip a winner, so the fingerprint carries the finishing time and both ships' structure.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import * as gunnery from '../src/sim/gunnery.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { budgetFor, playBattle } from './harness.js';

const mods = { ship, battle: battleMod, gunnery, config };
const names = Object.keys(ARCHETYPES);

let lines = 0;
for (let hullIndex = 0; hullIndex < HULLS.length; hullIndex++) {
  const budget = budgetFor(config.ROUNDS, hullIndex);
  for (const a of names) {
    for (const b of names) {
      if (a === b) continue;
      for (let s = 0; s < 6; s++) {
        const designs = [a, b].map((t) => {
          const d = ship.createDesign();
          autoBuild(d, hullIndex, budget, ARCHETYPES[t]);
          return d;
        });
        const seed = s * 7919 + hullIndex;
        const r = playBattle(mods, designs, hullIndex, seed);
        console.log(
          `${hullIndex} ${a.padEnd(9)} ${b.padEnd(9)} ${String(seed).padStart(6)}  ` +
            `w=${r.winner === null ? '-' : r.winner} t=${r.time.toFixed(3)} ` +
            `s=${r.struct[0].toFixed(4)}/${r.struct[1].toFixed(4)}`,
        );
        lines++;
      }
    }
  }
}
console.log(`# ${lines} battles`);
