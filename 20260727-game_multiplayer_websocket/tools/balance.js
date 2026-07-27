// Headless balance harness. Runs archetype matchups across many seeds and reports how
// long battles last, how often they end decisively, and whether the counter triangle holds.
//
//   node tools/balance.js            all matchups, hull 0 and 2
//   node tools/balance.js 2          just hull index 2
//   node tools/balance.js 0 verbose  print one blow-by-blow log

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { pct, budgetFor, playBattle } from './lib.js';

const mods = { ship, battle: battleMod, config };
const { ROUNDS } = config;

const names = Object.keys(ARCHETYPES);

function buildFor(hullIndex, archetype) {
  const design = ship.createDesign();
  const left = autoBuild(design, hullIndex, budgetFor(ROUNDS, hullIndex), ARCHETYPES[archetype]);
  return { design, left };
}

function runOne(hullIndex, aType, bType, seed) {
  const a = buildFor(hullIndex, aType);
  const b = buildFor(hullIndex, bType);
  const r = playBattle(mods, [a.design, b.design], hullIndex, seed);
  return { ...r, leftover: [a.left, b.left] };
}

const argHull = process.argv[2] !== undefined ? Number(process.argv[2]) : null;
const verbose = process.argv[3] === 'verbose';
const hulls = argHull !== null ? [argHull] : [0, 2, 4];
const SEEDS = 40;

for (const hullIndex of hulls) {
  const budget = budgetFor(ROUNDS, hullIndex);
  console.log(
    `\n=== ${HULLS[hullIndex].name}  (${HULLS[hullIndex].cells.length} cells, ${budget} scrap cumulative) ===`,
  );

  // Sanity: what does each archetype actually manage to build here?
  for (const n of names) {
    const { design, left } = buildFor(hullIndex, n);
    const s = ship.designStats(design, hullIndex);
    console.log(
      `  ${ARCHETYPES[n].label.padEnd(20)} guns:${String(s.gunCount).padStart(2)} masts:${s.masts} ` +
        `crew:${s.crewSupply}/${s.crewNeeded} mag:${s.magazines} unmanned:${s.unmanned.length} ` +
        `cells:${s.cellsUsed}/${s.cellsTotal} leftover:${left}`,
    );
  }

  const wins = {};
  let totalTime = 0;
  let decisive = 0;
  let games = 0;
  let draws = 0;

  for (const aType of names) {
    for (const bType of names) {
      if (aType === bType) continue;
      let aWins = 0;
      let n = 0;
      for (let s = 0; s < SEEDS; s++) {
        // Swap sides every other seed so the port/starboard orbit bias cancels out.
        const flip = s % 2 === 1;
        const r = runOne(hullIndex, flip ? bType : aType, flip ? aType : bType, s * 7919 + hullIndex);
        const winnerType = r.winner === null ? null : (r.winner === 0) !== flip ? aType : bType;
        if (winnerType === aType) aWins++;
        if (winnerType === null) draws++;
        n++;
        totalTime += r.time;
        if (r.decisive) decisive++;
        games++;
      }
      wins[`${aType} vs ${bType}`] = aWins / n;
    }
  }

  console.log(
    `  --- ${games} battles: mean ${(totalTime / games).toFixed(1)}s, decisive ${pct(decisive / games)}, draws ${pct(draws / games)}`,
  );
  const seen = new Set();
  for (const key of Object.keys(wins)) {
    const [a, b] = key.split(' vs ');
    if (seen.has(`${b} vs ${a}`)) continue;
    seen.add(key);
    console.log(`  ${a.padEnd(10)} vs ${b.padEnd(10)} ${pct(wins[key])}`);
  }

  if (verbose) {
    const r = runOne(hullIndex, 'brawler', 'sniper', 12345);
    console.log(`\n  sample battle: ${r.reason} at ${r.time.toFixed(1)}s`);
    console.log(`  structure left: ${pct(r.struct[0])} / ${pct(r.struct[1])}`);
    for (const e of r.log) console.log(`    ${e.t.toFixed(1)}s  ${e.text}`);
  }
}
