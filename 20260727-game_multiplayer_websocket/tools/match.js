// Plays complete 5-round matches headlessly, exercising the things a single battle never
// touches: hull upgrades between rounds, damage carrying over, scrap accounting, repairs.
//
//   node tools/match.js            10 matches, summary
//   node tools/match.js 1 verbose  one match, round by round

import { createDesign, fitDesignToHull, designStats, structureFraction } from '../src/sim/ship.js';
import { createBattle } from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { PARTS, repairCost } from '../src/data/parts.js';
import { HULLS } from '../src/data/hulls.js';
import { ROUNDS, TICK, POINTS_TO_WIN, loserBonus } from '../src/config.js';
import { makeRng, hashSeed } from '../src/sim/rng.js';
import { applyBotAmmo } from './lib.js';

const VERBOSE = process.argv[3] === 'verbose';
const MATCHES = Number(process.argv[2] || 10);

// Stand-in for a player: repair the worst damage, then spend what's left like the archetype.
function takeTurn(design, hullIndex, scrap, archetype) {
  let left = scrap;
  const damaged = Object.entries(design.parts)
    .filter(([, s]) => s.hp < PARTS[s.id].hp)
    .sort((a, b) => a[1].hp / PARTS[a[1].id].hp - b[1].hp / PARTS[b[1].id].hp);
  for (const [, slot] of damaged) {
    const cost = repairCost(slot.id);
    if (left < cost) break;
    // Only worth repairing things that do a job; timber is cheaper to replace.
    if (slot.id === 'timber') continue;
    left -= cost;
    slot.hp = PARTS[slot.id].hp;
  }
  return autoBuild(design, hullIndex, left, ARCHETYPES[archetype]);
}

function playMatch(seed, types) {
  const designs = [createDesign(), createDesign()];
  const scrap = [0, 0];
  const scores = [0, 0];
  let lastLoser = null;
  const rounds = [];

  for (let r = 0; r < ROUNDS.length; r++) {
    const hullIndex = ROUNDS[r].hull;
    for (let i = 0; i < 2; i++) {
      fitDesignToHull(designs[i], hullIndex);
      scrap[i] += ROUNDS[r].scrap + (lastLoser === i ? loserBonus(r) : 0);
      scrap[i] = takeTurn(designs[i], hullIndex, scrap[i], types[i]);
    }

    const stats = [designStats(designs[0], hullIndex), designStats(designs[1], hullIndex)];
    const battle = createBattle({
      designs,
      hullIndex,
      seed: hashSeed(seed, r, 4242),
      windTo: makeRng(hashSeed(seed, r, 77)).next() * Math.PI * 2,
    });

    let guard = 0;
    while (!battle.over && guard++ < 60 / TICK) {
      applyBotAmmo(battle, battle.ships[0], battle.ships[1]);
      applyBotAmmo(battle, battle.ships[1], battle.ships[0]);
      battle.advance(TICK);
    }
    const fracs = [structureFraction(battle.ships[0]), structureFraction(battle.ships[1])];
    battle.finish();

    // Invariant: the design must never keep a cell the battle destroyed, and the helm
    // must always come back so the next round is playable.
    for (let i = 0; i < 2; i++) {
      for (const cell of battle.ships[i].cells) {
        if (!cell.alive && designs[i].parts[cell.key] && cell.key !== '0,0') {
          throw new Error(`round ${r + 1}: destroyed cell ${cell.key} survived commitDamage`);
        }
      }
      if (!designs[i].parts['0,0']) throw new Error(`round ${r + 1}: player ${i + 1} lost its helm slot`);
      const allowed = new Set(HULLS[hullIndex].cells.map((c) => `${c.dx},${c.dz}`));
      for (const key of Object.keys(designs[i].parts)) {
        if (!allowed.has(key)) throw new Error(`round ${r + 1}: part at ${key} is off-hull`);
      }
      if (scrap[i] < 0) throw new Error(`round ${r + 1}: player ${i + 1} has negative scrap`);
    }

    if (battle.winner !== null) {
      scores[battle.winner]++;
      lastLoser = 1 - battle.winner;
    } else lastLoser = null;

    rounds.push({
      r: r + 1,
      hull: HULLS[hullIndex].name,
      time: battle.time,
      winner: battle.winner,
      reason: battle.reason,
      fracs,
      stats,
      scrapLeft: [...scrap],
      cells: [Object.keys(designs[0].parts).length, Object.keys(designs[1].parts).length],
    });

    if (scores[0] >= POINTS_TO_WIN || scores[1] >= POINTS_TO_WIN) break;
  }
  return { scores, rounds };
}

const names = Object.keys(ARCHETYPES);
// Per-round aggregates: is the ship actually getting grander, or just holier?
const perRound = ROUNDS.map(() => ({ n: 0, fill: 0, guns: 0, scrapLeft: 0, time: 0 }));
let totalRounds = 0;
let sweeps = 0;
let decidedEarly = 0;
const lengths = {};

for (let m = 0; m < MATCHES; m++) {
  const types = [names[m % names.length], names[(m + 1 + Math.floor(m / names.length)) % names.length]];
  const { scores, rounds } = playMatch(m * 1013 + 5, types);
  totalRounds += rounds.length;
  for (const r of rounds) {
    const agg = perRound[r.r - 1];
    const total = HULLS[ROUNDS[r.r - 1].hull].cells.length;
    agg.n += 2;
    agg.fill += r.cells[0] / total + r.cells[1] / total;
    agg.guns += r.stats[0].gunCount + r.stats[1].gunCount;
    agg.scrapLeft += r.scrapLeft[0] + r.scrapLeft[1];
    agg.time += r.time * 2;
  }
  lengths[rounds.length] = (lengths[rounds.length] || 0) + 1;
  if (Math.abs(scores[0] - scores[1]) >= 3) sweeps++;
  if (rounds.length < 5) decidedEarly++;

  if (VERBOSE) {
    console.log(`\n=== ${types[0]} vs ${types[1]}  final ${scores[0]}-${scores[1]} ===`);
    for (const r of rounds) {
      const s = r.stats;
      console.log(
        `  R${r.r} ${r.hull.padEnd(17)} ${r.time.toFixed(1)}s  ` +
          `winner ${r.winner === null ? 'draw' : r.winner + 1}  ` +
          `hull ${(r.fracs[0] * 100).toFixed(0)}%/${(r.fracs[1] * 100).toFixed(0)}%  ` +
          `guns ${s[0].gunCount}/${s[1].gunCount}  cells ${r.cells[0]}/${r.cells[1]}  ` +
          `scrapLeft ${r.scrapLeft[0]}/${r.scrapLeft[1]}`,
      );
      console.log(`       ${r.reason}`);
    }
  }
}

console.log(
  `\n${MATCHES} matches: mean ${(totalRounds / MATCHES).toFixed(1)} rounds, ` +
    `${sweeps} sweeps (3-0), ${decidedEarly} decided before round 5`,
);
console.log('round-count distribution:', lengths);
console.log('\nround  hull                cells  fill%  guns  scrapLeft  meanTime');
perRound.forEach((a, i) => {
  if (!a.n) return;
  const hull = HULLS[ROUNDS[i].hull];
  console.log(
    `  R${i + 1}   ${hull.name.padEnd(18)} ${String(hull.cells.length).padStart(3)}   ` +
      `${((a.fill / a.n) * 100).toFixed(0).padStart(4)}%  ${(a.guns / a.n).toFixed(1).padStart(4)}  ` +
      `${(a.scrapLeft / a.n).toFixed(1).padStart(8)}  ${(a.time / a.n).toFixed(1).padStart(7)}s`,
  );
});
console.log('invariants held: destroyed cells cleared, helm preserved, parts on-hull, scrap non-negative');
