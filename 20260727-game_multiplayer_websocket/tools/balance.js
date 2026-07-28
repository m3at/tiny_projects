// Headless balance harness. Runs archetype matchups across many seeds and reports how long
// battles last, how often they end decisively, and how lopsided the worst pairing is.
//
//   node tools/balance.js            all matchups, hulls 0, 2 and 4
//   node tools/balance.js 2          just hull index 2
//   node tools/balance.js 0 verbose  print one blow-by-blow log
//
// Read the worst cell, not the average. Fighting-game practice grades a matchup on a ten-point
// scale: 5-5 even, 6-4 noticeable but winnable, 7-3 a counterpick, 8-2 near unwinnable. Sirlin
// shipped Fantasy Strike with nothing worse than 7-3, and notes that averaging win rates hides
// the problem completely -- a roster where every matchup is 8-2 still averages out to 50%. So
// this prints the count of pairings past 7-3 and names the worst one. An archetype average near
// 50% means nothing on its own.
//
// One caution the same sources raise: these are greedy bots, and a bot's blind spot looks
// exactly like an imbalance. Before nerfing a part because of a lopsided cell, check whether a
// differently built ship answers it.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import * as gunnery from '../src/sim/gunnery.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { pct, budgetFor, playBattle } from './harness.js';

const mods = { ship, battle: battleMod, gunnery, config };
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
  const pairs = [];
  for (const key of Object.keys(wins)) {
    const [a, b] = key.split(' vs ');
    if (seen.has(`${b} vs ${a}`)) continue;
    seen.add(key);
    pairs.push([a, b, wins[key]]);
  }
  // Distance from even, so 6-4 and 4-6 are the same severity.
  const skew = (r) => Math.abs(r - 0.5) * 2;
  pairs.sort((x, y) => skew(y[2]) - skew(x[2]));
  for (const [a, b, rate] of pairs) {
    const tenths = Math.round(rate * 10);
    const grade = skew(rate) >= 0.6 ? '  <-- past 7-3' : '';
    console.log(`  ${a.padEnd(10)} vs ${b.padEnd(10)} ${pct(rate).padStart(4)}  ${tenths}-${10 - tenths}${grade}`);
  }
  const bad = pairs.filter(([, , r]) => skew(r) >= 0.6);
  const worst = pairs[0];
  console.log(
    `  worst pairing ${worst[0]} vs ${worst[1]} at ${pct(worst[2])};  ` +
      `${bad.length}/${pairs.length} pairings past 7-3` +
      (bad.length ? `: ${bad.map(([a, b]) => `${a}/${b}`).join(', ')}` : ''),
  );

  if (verbose) {
    const r = runOne(hullIndex, 'brawler', 'sniper', 12345);
    console.log(`\n  sample battle: ${r.reason} at ${r.time.toFixed(1)}s`);
    console.log(`  structure left: ${pct(r.struct[0])} / ${pct(r.struct[1])}`);
    for (const e of r.log) console.log(`    ${e.t.toFixed(1)}s  ${e.text}`);
  }
}
