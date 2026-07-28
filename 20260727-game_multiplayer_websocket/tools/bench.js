// How fast does the simulation run? Every other harness is built out of this number, so it
// sets how long a question takes to answer.
//
//   node tools/bench.js          all three hull sizes
//   node tools/bench.js 4        just the ship of the line
//
// Reports simulated seconds per real second, which is also the headroom the browser has: the
// render loop asks for one battle at 1x, so 1000x here is 1000 frames of slack.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import * as gunnery from '../src/sim/gunnery.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { budgetFor, playBattle } from './harness.js';

const mods = { ship, battle: battleMod, gunnery, config };
const names = Object.keys(ARCHETYPES);

function buildFor(hullIndex, archetype) {
  const design = ship.createDesign();
  autoBuild(design, hullIndex, budgetFor(config.ROUNDS, hullIndex), ARCHETYPES[archetype]);
  return design;
}

const hulls = process.argv[2] !== undefined ? [Number(process.argv[2])] : [0, 2, 4];
const REPS = 60;

console.log('  hull                 battles   sim time   real time      x realtime   per battle');
let totalBattles = 0;
let totalReal = 0;

for (const hullIndex of hulls) {
  // Designs are built once and reused: this measures the battle core, not the builder.
  const designs = names.map((n) => buildFor(hullIndex, n));
  let simSeconds = 0;
  let battles = 0;

  const t0 = performance.now();
  for (let r = 0; r < REPS; r++) {
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        // Fresh designs each battle, since a battle writes damage back into them.
        const pair = [ship.cloneDesign(designs[i]), ship.cloneDesign(designs[j])];
        const out = playBattle(mods, pair, hullIndex, r * 7919 + i * 31 + j);
        simSeconds += out.time;
        battles++;
      }
    }
  }
  const real = (performance.now() - t0) / 1000;
  totalBattles += battles;
  totalReal += real;

  console.log(
    `  ${HULLS[hullIndex].name.padEnd(20)} ${String(battles).padStart(6)}  ` +
      `${simSeconds.toFixed(0).padStart(7)}s  ${real.toFixed(2).padStart(8)}s  ` +
      `${(simSeconds / real).toFixed(0).padStart(13)}x  ${((real / battles) * 1000).toFixed(2).padStart(8)}ms`,
  );
}

console.log(
  `\n  ${totalBattles} battles in ${totalReal.toFixed(2)}s  ` +
    `(${(totalBattles / totalReal).toFixed(0)} battles/sec)`,
);
