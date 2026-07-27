// Which parts are worth taking? Sampled across the builds a player could actually arrive at,
// rather than across four hand-written pure archetypes.
//
//   node tools/parts.js          all hulls
//   node tools/parts.js 2 400    one hull, 400 builds
//
// Why this exists. tools/balance.js pits one pure build against another and the results come out
// bimodal: near 50% between similar builds, near 100% between dissimilar ones, with almost
// nothing in between. That is the signature of a compounding advantage -- first blood opens
// holes, holes let shot through to the vitals -- and it means a pure-archetype grid cannot tell a
// slightly better gun from a decisively better one. It also measures builds the draft can never
// offer, since a round shows five part types out of nine.
//
// So: draw a lot of random legal builds, fight them against each other, and ask two questions
// per part.
//
//   edge   of the battles where one ship carried strictly more of this part than the other, how
//          often did that ship win. 50% is a part that neither wins nor loses games. This is the
//          dominance test: something at 70% is being taken for the wrong reason.
//   taken  how often a build that could afford this part actually got value from it, as the share
//          of sampled builds carrying at least one.
//
// Sid Meier's test for a dead decision cuts both ways, so both extremes are bugs: a part nobody
// would take is as broken as one everybody must.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import { autoBuild } from '../src/autobuild.js';
import { PARTS, BUYABLE } from '../src/data/parts.js';
import { HULLS } from '../src/data/hulls.js';
import * as config from '../src/config.js';
import { makeRng } from '../src/sim/rng.js';
import { pct, budgetFor, playBattle } from './lib.js';

const mods = { ship, battle: battleMod, config };
const GUNS = BUYABLE.filter((id) => PARTS[id].gun);
// A bow chaser can only be worked from the bow, so it can never be a ship's main armament. Using
// it as one would measure "picked a gun that cannot fill a hull", not "this gun is weak".
const PRIMARY = GUNS.filter((id) => PARTS[id].gun.arc !== 'bow');
const ARMOUR = ['timber', 'heavy'];

// A random but plausible ship: some primary gun, sometimes a second kind, some sail, some armour.
// The point is spread, not quality -- a bad build is a useful data point.
function randomBuild(hullIndex, budget, rng) {
  const design = ship.createDesign();
  const profile = {
    gun: PRIMARY[rng.int(0, PRIMARY.length - 1)],
    second: rng.range(0, 1) < 0.55 ? GUNS[rng.int(0, GUNS.length - 1)] : null,
    gunCount: rng.int(1, 8),
    // Up to one past what the hull can use: the build readout states the number, so a player
    // over-rigging by five is not a build the game encourages.
    mastCount: rng.int(1, config.mastsWanted(HULLS[hullIndex].cells.length) + 1),
    armour: ARMOUR[rng.int(0, ARMOUR.length - 1)],
    massed: rng.range(0, 1) < 0.35,
  };
  autoBuild(design, hullIndex, budget, profile, rng.range(0, 1) < 0.5 ? 'port' : 'starboard');
  return design;
}

function countParts(design) {
  const counts = {};
  for (const slot of Object.values(design.parts)) counts[slot.id] = (counts[slot.id] || 0) + 1;
  return counts;
}

const argHull = process.argv[2] !== undefined ? Number(process.argv[2]) : null;
const BUILDS = Number(process.argv[3] || 240);
const hulls = argHull !== null ? [argHull] : [0, 2, 4];

for (const hullIndex of hulls) {
  const budget = budgetFor(config.ROUNDS, hullIndex);
  const rng = makeRng(20260728 + hullIndex);
  const builds = [];
  for (let i = 0; i < BUILDS; i++) {
    const design = randomBuild(hullIndex, budget, rng);
    builds.push({ design, counts: countParts(design) });
  }

  // Each build fights the next few in the list, so every ship meets several different opponents.
  const stat = {};
  for (const id of BUYABLE) stat[id] = { more: 0, moreWon: 0, carried: 0 };
  for (const b of builds) for (const id of BUYABLE) if (b.counts[id]) stat[id].carried++;

  let battles = 0;
  let draws = 0;
  let time = 0;
  const OPPONENTS = 5;
  for (let i = 0; i < builds.length; i++) {
    for (let k = 1; k <= OPPONENTS; k++) {
      const j = (i + k * 7) % builds.length;
      if (i === j) continue;
      const pair = [structuredClone(builds[i].design), structuredClone(builds[j].design)];
      const out = playBattle(mods, pair, hullIndex, i * 131 + k);
      battles++;
      time += out.time;
      if (out.winner === null) {
        draws++;
        continue;
      }
      const winner = out.winner === 0 ? i : j;
      for (const id of BUYABLE) {
        const a = builds[i].counts[id] || 0;
        const b = builds[j].counts[id] || 0;
        if (a === b) continue;
        const holder = a > b ? i : j;
        stat[id].more++;
        if (holder === winner) stat[id].moreWon++;
      }
    }
  }

  console.log(
    `\n=== ${HULLS[hullIndex].name}  ${BUILDS} builds, ${battles} battles, ` +
      `mean ${(time / battles).toFixed(1)}s, draws ${pct(draws / battles)} ===`,
  );
  console.log('  part            cost   edge   taken   verdict');
  const rows = BUYABLE.map((id) => ({
    id,
    edge: stat[id].more ? stat[id].moreWon / stat[id].more : 0.5,
    taken: stat[id].carried / BUILDS,
    n: stat[id].more,
  })).sort((a, b) => b.edge - a.edge);
  for (const r of rows) {
    // Bands from balance practice: inside 47-53% is noise, past 60% is a part that wins games on
    // its own, under 40% is a trap.
    const verdict =
      r.edge >= 0.6 ? 'dominant' : r.edge <= 0.4 ? 'trap' : r.edge >= 0.53 || r.edge <= 0.47 ? 'strong/weak' : 'even';
    console.log(
      `  ${r.id.padEnd(12)} ${String(PARTS[r.id].cost).padStart(5)}  ${pct(r.edge).padStart(5)}  ` +
        `${pct(r.taken).padStart(5)}   ${verdict}${r.n < 40 ? `  (only ${r.n} samples)` : ''}`,
    );
  }
}
