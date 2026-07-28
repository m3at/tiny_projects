// Three and four ships. Everything else in tools/ measures a duel, and a melee raises four
// questions a duel cannot answer at all.
//
//   node tools/melee.js
//
//   1. is it worth watching        the watch.js numbers per field size and per hull, so a
//                                  three-way can be read against the duel's 13-17s and 30% dry
//   2. is the ring start fair      identical ships in every seat, so the only thing that differs
//                                  is where the seat stands on the circle. A seat bias is the most
//                                  likely bug in a ring start and nothing else here would notice it
//   3. does the field change a build   one archetype against a field of a reference archetype at
//                                  two, three and four ships. A part table tuned for duels can
//                                  break in a melee, and this is where that shows
//   4. what SHIP_COUNT_DAMAGE should be   a sweep of the incoming-damage scale for three and for
//                                  four ships, against the duel's own battle length
//   5. is a part a trap in a melee  the tools/parts.js lens -- hundreds of random legal builds,
//                                  scored per part -- run at two, three and four ships. Section 3
//                                  reads pure archetypes, which the draft cannot offer; this is the
//                                  one to believe about a part
//
// Sections 1 and 4 print watch.js's columns; see that file for what each one means. The ones only a
// melee needs:
//
//   x-even  win rate as a multiple of an even share, so a 50% duel and a 25% four-way both read
//           1.00 and the three field sizes can be compared along one row.
//   place   mean finishing place, 0 for the winner and 1 for last. 0.50 is an even result. A seat
//           or a build can be consistently unlucky without ever losing outright, which a win rate
//           on its own hides.
//   dev     distance from even in standard deviations of the binomial. Under 2 is noise; a real
//           bias in a ring, with a few hundred battles behind it, lands nowhere near that.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import * as gunnery from '../src/sim/gunnery.js';
import * as autobuild from '../src/autobuild.js';
import * as config from '../src/config.js';
import { HULLS } from '../src/data/hulls.js';
import { PARTS, BUYABLE } from '../src/data/parts.js';
import { makeRng } from '../src/sim/rng.js';
import { pct, budgetFor, playBattle, measureBattle } from './harness.js';
import { variant, cleanupVariants, checkPatches } from './variant.js';

const real = { ship, battle: battleMod, gunnery, config, autobuild };
const names = Object.keys(autobuild.ARCHETYPES);
const HULL_SET = [0, 2, 4];
const COUNTS = [2, 3, 4];
const REFERENCE = 'brawler'; // the plain broadside every other build is measured against

// Designs are read-only once a battle is created (makeBattleShip copies every cell) and autoBuild
// is deterministic for a fixed archetype, so one design per hull per archetype serves every battle.
// It is also what lets the seat-fairness run hand literally the same object to all four seats.
const cache = new Map();
function designFor(mod, hullIndex, archetype) {
  if (!cache.has(mod)) cache.set(mod, new Map());
  const table = cache.get(mod);
  const key = `${hullIndex}/${archetype}`;
  if (table.has(key)) return table.get(key);
  const design = mod.ship.createDesign();
  const budget = budgetFor(mod.config.ROUNDS, hullIndex);
  mod.autobuild.autoBuild(design, hullIndex, budget, mod.autobuild.ARCHETYPES[archetype]);
  table.set(key, design);
  return design;
}

// Every field of distinct archetypes of a given size. At two ships this is exactly the fifteen
// pairs watch.js runs, which is what makes the duel rows comparable with that tool's output.
function fields(list, k) {
  if (k === 0) return [[]];
  const out = [];
  for (let i = 0; i <= list.length - k; i++) {
    for (const rest of fields(list.slice(i + 1), k - 1)) out.push([list[i], ...rest]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The watch.js table, for any field size
// ---------------------------------------------------------------------------

const COLS = [['time', 5], ['open', 5], ['dry', 4], ['far', 4], ['arc', 4], ['gap', 5],
  ['edge', 4], ['drift', 5], ['par', 5], ['revs', 4], ['volleys', 7], ['decisive', 8]];
const KEYS = COLS.map(([k]) => k);

const num = (v, w, d) => v.toFixed(d).padStart(w);

function head(label, w) {
  return `  ${label.padEnd(w)} ` + COLS.map(([k, cw]) => k.padStart(cw)).join('  ');
}

function row(label, w, m) {
  return (
    `  ${label.padEnd(w)} ${num(m.time, 4, 1)}s  ${num(m.open, 4, 1)}s  ` +
    `${pct(m.dry).padStart(4)}  ${pct(m.far).padStart(4)}  ${pct(m.arc).padStart(4)}  ` +
    `${num(m.gap, 4, 1)}s  ${pct(m.edge).padStart(4)}  ${num(m.drift, 5, 2)}  ` +
    `${num(m.par, 5, 2)}  ${num(m.revs, 4, 1)}  ${m.volleys.toFixed(0).padStart(7)}  ` +
    `${pct(m.decisive).padStart(8)}`
  );
}

function measureField(mod, count, hullIndex, seeds) {
  const acc = Object.fromEntries(KEYS.map((k) => [k, 0]));
  let n = 0;
  for (const field of fields(names, count)) {
    const designs = field.map((t) => designFor(mod, hullIndex, t));
    for (let s = 0; s < seeds; s++) {
      const m = measureBattle(mod, designs, hullIndex, s * 7919 + hullIndex);
      for (const k of KEYS) acc[k] += Number(m[k]);
      n++;
    }
  }
  for (const k of KEYS) acc[k] /= n;
  acc.battles = n;
  return acc;
}

// Every hull carries the same number of battles, so this is a plain mean, but weight it anyway.
function pool(rows) {
  const total = rows.reduce((s, r) => s + r.battles, 0);
  const acc = Object.fromEntries(KEYS.map((k) => [k, 0]));
  for (const r of rows) for (const k of KEYS) acc[k] += r[k] * r.battles;
  for (const k of KEYS) acc[k] /= total;
  acc.battles = total;
  return acc;
}

function overHulls(mod, count, hulls, seeds) {
  return pool(hulls.map((h) => measureField(mod, count, h, seeds)));
}

// ---------------------------------------------------------------------------
// 1. is a melee worth watching
// ---------------------------------------------------------------------------

const WATCH_SEEDS = 4;
const FIELD_W = 25;

console.log('=== 1. worth watching, by field size ===');
console.log(head('field    hull', FIELD_W));
for (const count of COUNTS) {
  const rows = [];
  for (const hullIndex of HULL_SET) {
    const m = measureField(real, count, hullIndex, WATCH_SEEDS);
    rows.push(m);
    console.log(row(`${count} ships`.padEnd(9) + HULLS[hullIndex].name, FIELD_W, m));
  }
  console.log(row(`${count} ships`.padEnd(9) + 'all', FIELD_W, pool(rows)));
  console.log('');
}

// ---------------------------------------------------------------------------
// 2. is the ring start fair
// ---------------------------------------------------------------------------
//
// Identical designs in every seat, so the wind, the orbit sense and the starting position are all
// that differ, and the first two are drawn from the seed and average out over hundreds of battles.
// What is left is the seat.
//
// The ring is rotationally symmetric and the wind is swept over the whole compass, so anything the
// seat index can still be seen through has to be an index-order tie-break inside the simulation.
// That is why this section exists at all, and why it prints a chi-square rather than eyeballing.

const SEAT_SEEDS = 300;
const SEAT_BUILDS = [REFERENCE, 'crusher'];
const SEAT_HULL = 2;
// Chi-square 95% critical values for one, two and three degrees of freedom. Enough of a table for a
// ring that holds at most four ships, and it keeps the tool free of dependencies.
const CHI95 = { 1: 3.84, 2: 5.99, 3: 7.81 };

function seatRun(count, archetype, seeds) {
  const design = designFor(real, SEAT_HULL, archetype);
  const field = new Array(count).fill(design);
  const wins = new Array(count).fill(0);
  const places = new Array(count).fill(0);
  let decided = 0;
  let draws = 0;
  let placed = 0;
  for (let s = 0; s < seeds; s++) {
    // A stride coprime with 360, so windForSeed walks every point of the compass.
    const r = playBattle(real, field, SEAT_HULL, s * 104729 + count);
    if (r.winner === null) draws++;
    else {
      wins[r.winner]++;
      decided++;
    }
    if (r.placing) {
      for (let place = 0; place < r.placing.length; place++) places[r.placing[place]] += place;
      placed++;
    }
  }
  return { wins, places, decided, draws, placed };
}

console.log('=== 2. is the ring start fair ===');
console.log(
  `  identical ${HULLS[SEAT_HULL].name.toLowerCase()}s in every seat, ${SEAT_SEEDS} battles per ` +
    'configuration; only the seat, the wind and the orbit sense differ',
);
console.log('  field     build      seat  battles   win%   even%     dev   place    dev');
for (const count of COUNTS) {
  for (const archetype of SEAT_BUILDS) {
    const r = seatRun(count, archetype, SEAT_SEEDS);
    const even = 1 / count;
    const sd = Math.sqrt((even * (1 - even)) / r.decided);
    // Place is uniform on 0..count-1 under a fair ring, so its spread is the discrete uniform's.
    const placeSd = Math.sqrt((count * count - 1) / 12 / r.placed);
    let chi = 0;
    let worst = 0;
    for (let seat = 0; seat < count; seat++) {
      const rate = r.wins[seat] / r.decided;
      const dev = (rate - even) / sd;
      const place = r.places[seat] / r.placed;
      chi += (r.wins[seat] - r.decided * even) ** 2 / (r.decided * even);
      if (Math.abs(dev) > Math.abs(worst)) worst = dev;
      console.log(
        `  ${`${count} ships`.padEnd(9)} ${archetype.padEnd(10)} ${String(seat).padStart(3)}  ` +
          `${String(r.decided).padStart(7)}  ${pct(rate).padStart(4)}  ${pct(even).padStart(6)}  ` +
          `${num(dev, 5, 1)}sd  ${num(place / (count - 1), 5, 2)}  ` +
          `${num((place - (count - 1) / 2) / placeSd, 5, 1)}sd`,
      );
    }
    const df = count - 1;
    console.log(
      `  ${''.padEnd(9)} ${archetype.padEnd(10)} widest seat ${num(worst, 5, 1)}sd, ` +
        `chi2 ${chi.toFixed(1)} on ${df} df against ${CHI95[df]} at 95%: ` +
        `${chi <= CHI95[df] ? 'inside binomial noise' : 'OUTSIDE binomial noise'}` +
        (r.draws ? `, ${pct(r.draws / SEAT_SEEDS)} draws` : ''),
    );
  }
}
console.log('');

// ---------------------------------------------------------------------------
// 3. does the field size change how a ship should be built
// ---------------------------------------------------------------------------
//
// One ship of the archetype under test against a field of the reference archetype, its seat rotated
// through the ring, so section 2's seat bias cannot be read as a build result. Pooled over three
// hulls, because a part that only breaks on a ship of the line is still broken.

const FIELD_SEEDS = 40; // per hull, so three times this per cell

function fieldRun(count, archetype, hulls, seeds) {
  let wins = 0;
  let decided = 0;
  let placeSum = 0;
  let placed = 0;
  for (const hullIndex of hulls) {
    for (let s = 0; s < seeds; s++) {
      const seat = s % count;
      const field = [];
      for (let i = 0; i < count; i++) {
        field.push(designFor(real, hullIndex, i === seat ? archetype : REFERENCE));
      }
      const r = playBattle(real, field, hullIndex, s * 7919 + hullIndex);
      if (r.winner !== null) {
        decided++;
        if (r.winner === seat) wins++;
      }
      if (r.placing) {
        placeSum += r.placing.indexOf(seat) / (count - 1);
        placed++;
      }
    }
  }
  return { rate: wins / decided, place: placeSum / placed };
}

console.log(`=== 3. one build against a field of ${REFERENCE}s ===`);
console.log(
  "  the test ship's seat rotates through the ring; pooled over " +
    `${HULL_SET.length} hulls, ${FIELD_SEEDS * HULL_SET.length} battles per cell`,
);
console.log('  archetype    2 win  x-even  place   3 win  x-even  place   4 win  x-even  place');
const cells = names.map((archetype) => ({
  archetype,
  by: COUNTS.map((count) => fieldRun(count, archetype, HULL_SET, FIELD_SEEDS)),
}));
for (const r of cells) {
  console.log(
    `  ${r.archetype.padEnd(10)} ` +
      r.by
        .map((c, i) => `${pct(c.rate).padStart(5)}  ${num(c.rate * COUNTS[i], 6, 2)}  ` +
          `${num(c.place, 5, 2)}`)
        .join('   '),
  );
}
for (let i = 0; i < COUNTS.length; i++) {
  const even = cells.map((r) => r.by[i].rate * COUNTS[i]);
  const hi = Math.max(...even);
  const lo = Math.min(...even);
  console.log(
    `  ${COUNTS[i]} ships: spread ${(hi - lo).toFixed(2)} x-even, ` +
      `best ${cells[even.indexOf(hi)].archetype} ${hi.toFixed(2)}, ` +
      `worst ${cells[even.indexOf(lo)].archetype} ${lo.toFixed(2)}`,
  );
}
console.log('');

// ---------------------------------------------------------------------------
// 4. what SHIP_COUNT_DAMAGE should be
// ---------------------------------------------------------------------------
//
// The constant was guessed downward, on the reasoning that a ship in a four-way takes fire from
// three directions and the round would therefore be over in a fraction of the duel's time. It runs
// the other way: a melee ends only when one ship is left, so a three-way sinks two ships and a
// four-way three, where a duel sinks one. Holding a duel's damage rate per ship therefore gives a
// round two or three times as long, and pulling the damage back lengthens it further. Every value
// under 1 measures worse than no correction at all.
//
// Index 3 only touches three-ship battles and index 4 only four-ship ones, so the two sweep
// independently and each is measured on its own field size.
//
// Checked against section 2: running the same grid with the seat bias patched out moves every mean
// length by under a second, so the sweep does not have to wait on that fix.

const FIND = 'export const SHIP_COUNT_DAMAGE = [1, 1, 1, 2, 3];';
const CURRENT = [1, 1, 1, 2, 3];
const GRID = {
  3: [1, 1.5, 2, 2.5, 3, 3.5, 4],
  4: [0.46, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
};
const SWEEP_SEEDS = 3;
// The window shipped autobattlers sit in, and the one the duel was tuned into. Damage buys length
// back and every step of it also buys empty air, since a battery that has been shot away is not
// firing, so the grid is read twice: closest length, and cheapest in dry inside the window.
const BAND = [13, 17];

checkPatches({ 'ship-count-damage': [['config.js', FIND]] });

console.log('=== 4. SHIP_COUNT_DAMAGE, swept against the duel it is meant to match ===');
const duel = overHulls(real, 2, HULL_SET, SWEEP_SEEDS);
console.log(
  `  duel baseline over the same grid: time ${duel.time.toFixed(1)}s  dry ${pct(duel.dry)}  ` +
    `gap ${duel.gap.toFixed(1)}s  decisive ${pct(duel.decisive)}  (${duel.battles} battles)`,
);
const picks = {};
for (const count of COUNTS.slice(1)) {
  console.log(`\n  index ${count}, measured on ${count}-ship battles`);
  console.log(`${head('value', 6)}  d-duel`);
  let closest = null;
  let inBand = null;
  for (const v of GRID[count]) {
    const table = CURRENT.slice();
    table[count] = v;
    const mod = await variant(`damage-${count}-${v}`, [
      ['config.js', FIND, `export const SHIP_COUNT_DAMAGE = [${table.join(', ')}];`],
    ]);
    const m = overHulls(mod, count, HULL_SET, SWEEP_SEEDS);
    const off = Math.abs(m.time - duel.time);
    const cand = { v, off, m };
    if (closest === null || off < closest.off) closest = cand;
    if (m.time >= BAND[0] && m.time <= BAND[1] && (inBand === null || m.dry < inBand.m.dry)) {
      inBand = cand;
    }
    console.log(
      `${row(String(v), 6, m)}  ${num(m.time - duel.time, 5, 1)}s` +
        (v === CURRENT[count] ? '  <- current' : ''),
    );
  }
  picks[count] = { closest, inBand };
}

console.log(`\n  d-duel is the gap to the duel's ${duel.time.toFixed(1)}s. Read it against dry:`);
console.log('  time and empty air move in opposite directions the whole length of the grid, so');
console.log('  the clock can always be bought and the only question is the price.');
for (const count of COUNTS.slice(1)) {
  const { closest, inBand } = picks[count];
  console.log(
    `  index ${count}: closest length ${closest.v} at ${closest.m.time.toFixed(1)}s ` +
      `(${closest.off.toFixed(1)}s off), ${pct(closest.m.dry)} dry vs the duel's ${pct(duel.dry)}`,
  );
  console.log(
    `           least dry inside ${BAND[0]}-${BAND[1]}s: ${inBand.v} at ` +
      `${inBand.m.time.toFixed(1)}s, ${pct(inBand.m.dry)} dry, ${pct(inBand.m.decisive)} decisive`,
  );
}
console.log(
  `  recommended SHIP_COUNT_DAMAGE = [1, 1, 1, ${picks[3].inBand.v}, ${picks[4].inBand.v}], ` +
    'the least-dry value inside the band',
);
console.log(
  `           current [1, 1, 1, ${CURRENT[3]}, ${CURRENT[4]}]; matching the duel's clock exactly ` +
    `wants [1, 1, 1, ${picks[3].closest.v}, ${picks[4].closest.v}]`,
);
console.log(
  '           and about fifteen more points of empty air, which is the wrong way round: the ' +
    'last\n           gameplay pass was aimed at dry, not at the clock.',
);

cleanupVariants();

// ---------------------------------------------------------------------------
// 5. the parts.js lens, at 2, 3 and 4 ships
// ---------------------------------------------------------------------------
//
// Section 3 fights whole pure archetypes, and CLAUDE.md is explicit that this is weak evidence about
// a part: autobuild builds pure ships the five-of-nine draft can never offer, and the grid comes out
// bimodal because damage compounds. So run the tools/parts.js lens instead -- hundreds of random
// legal builds from randomProfile, fought against each other, scored per part -- at each field size,
// with the same thresholds so the numbers can be read against `node tools/parts.js` directly.
//
// The two-ship statistic is "of the battles where one ship carried strictly more of this part than
// the other, how often did that ship win". That does not carry over to four ships on its own: with
// three rivals, "the ship with more of it" is not a pair, and an outright win is worth 25% by chance
// rather than 50%. Two statistics, defined exactly:
//
//   edge   the pairwise one, and the one the verdict is taken from. For every concluded battle and
//          every unordered pair of seats whose counts of this part differ, one comparison: the seat
//          carrying strictly more scores a win if it finished ahead of the other in battle.placing
//          (best first; survival outranks structure). Seats with equal counts contribute nothing, as
//          in parts.js. Even is 50% at every field size, because every comparison has exactly one
//          winner, so the 47-53 / 60 / 40 bands transfer unchanged. At two ships there is one pair
//          and finishing ahead is winning, so this is parts.js's number, and it reproduces it to
//          within a point or two.
//   x-ev   the field one, for reading against section 3. Of the battles where exactly one seat
//          carried strictly the most of this part -- ties at the top drop out, which is why a part
//          every build carries one of, like the magazine, has no samples at all -- how often that
//          seat won outright, as a multiple of an even share (rate x ship count). 1.00 is even. At
//          two ships it is exactly 2 x edge by construction; past that it is a different question,
//          since winning a four-way is three wins and placing second is not half of one.
//
// Draws are skipped, as in parts.js. `massed` is not a purchase but a placement pattern -- the whole
// battery on one flank, which randomProfile draws for about a third of builds -- and it is scored
// here as a pseudo-part with a count of one, because section 3 reports it as an archetype and the
// question of whether it is strong at three ships is the same question.
//
// The build pool is drawn once per hull and shared by all three field sizes, so `taken` does not
// depend on the field and the three edge columns differ only in the battles, not in the ships.
//
// An optional argument re-draws the pool: `node tools/melee.js 1` is the same measurement on a
// different sample. Run-to-run the tool is deterministic and prints byte-identical output, so that
// argument is the only honest stability check. Across samples the edges move by one to three points
// and no reading that this section is read for changes side.

// Per hull. The build sample, not the battle count, is what limits precision here: four times the
// battles per build tightens nothing, and twice the builds does.
const PARTS_BUILDS = 600;
const PARTS_FIELDS = 5; // fields per build per field size, so 3000 battles per hull per field size
const PARTS_SAMPLE = Number(process.argv[2] || 0);
const MASSED = 'massed';
const LENS_IDS = [...BUYABLE, MASSED];

// Same distribution as tools/parts.js: randomProfile lives in autobuild.js precisely so that the
// spread of builds is one definition in one place. A bad build is a useful data point here.
function randomBuild(hullIndex, budget, rng) {
  const design = ship.createDesign();
  const side = rng.range(0, 1) < 0.5 ? 'port' : 'starboard';
  const profile = autobuild.randomProfile(rng, hullIndex);
  autobuild.autoBuild(design, hullIndex, budget, profile, side);
  const counts = {};
  for (const slot of Object.values(design.parts)) counts[slot.id] = (counts[slot.id] || 0) + 1;
  if (profile.massed) counts[MASSED] = 1;
  return { design, counts };
}

function blankStat() {
  const stat = {};
  for (const id of LENS_IDS) stat[id] = { pairs: 0, pairsWon: 0, top: 0, topWon: 0 };
  return stat;
}

function scoreBattle(stat, seats, out) {
  const n = seats.length;
  // placing is best first; invert it once so "finished ahead" is a comparison and not a search.
  const rank = new Array(n);
  for (let p = 0; p < out.placing.length; p++) rank[out.placing[p]] = p;
  for (const id of LENS_IDS) {
    let best = -1;
    let bestSeat = -1;
    let tiedAtTop = false;
    for (let a = 0; a < n; a++) {
      const na = seats[a].counts[id] || 0;
      if (na > best) {
        best = na;
        bestSeat = a;
        tiedAtTop = false;
      } else if (na === best) {
        tiedAtTop = true;
      }
      for (let b = a + 1; b < n; b++) {
        const nb = seats[b].counts[id] || 0;
        if (na === nb) continue;
        const holder = na > nb ? a : b;
        const other = na > nb ? b : a;
        stat[id].pairs++;
        if (rank[holder] < rank[other]) stat[id].pairsWon++;
      }
    }
    if (best > 0 && !tiedAtTop) {
      stat[id].top++;
      if (out.winner === bestSeat) stat[id].topWon++;
    }
  }
}

function addInto(into, from) {
  for (const id of LENS_IDS) {
    into[id].pairs += from[id].pairs;
    into[id].pairsWon += from[id].pairsWon;
    into[id].top += from[id].top;
    into[id].topWon += from[id].topWon;
  }
}

// parts.js's bands, abbreviated because there are three per row: inside 47-53 is noise, past 60
// wins games on its own, under 40 is a trap.
function lensVerdict(edge) {
  if (edge >= 0.6) return 'dom';
  if (edge <= 0.4) return 'trap';
  if (edge >= 0.53) return 'str';
  if (edge <= 0.47) return 'weak';
  return 'even';
}

function lensRows(stats, carried, builds) {
  return LENS_IDS.map((id) => ({
    id,
    cost: id === MASSED ? '-' : String(PARTS[id].cost),
    taken: carried[id] / builds,
    by: COUNTS.map((count) => {
      const s = stats[count][id];
      return {
        edge: s.pairs ? s.pairsWon / s.pairs : 0.5,
        xeven: s.top ? (s.topWon / s.top) * count : 1,
        n: Math.min(s.pairs, s.top),
      };
    }),
  })).sort((a, b) => b.by[0].edge - a.by[0].edge);
}

const LENS_HEAD =
  '  part          cost  taken    2 edge  x-ev    3 edge  x-ev    4 edge  x-ev   verdict 2/3/4';

// Too little to read: parts.js's own threshold of 40 comparisons, plus a floor on how many builds
// carried the part at all. A part in one build in thirty can clear 40 comparisons and still be
// twenty ships measured five times each, which is the long gun's standing blind spot.
function isThin(r) {
  return r.by.some((c) => c.n < 40) || r.taken < 0.05;
}

function lensTable(rows) {
  console.log(LENS_HEAD);
  for (const r of rows) {
    const thin = isThin(r);
    console.log(
      `  ${r.id.padEnd(12)} ${r.cost.padStart(4)}  ${pct(r.taken).padStart(5)}  ` +
        r.by.map((c) => `${pct(c.edge).padStart(6)}  ${num(c.xeven, 4, 2)}`).join('  ') +
        `   ${r.by.map((c) => lensVerdict(c.edge)).join('/')}${thin ? '   thin' : ''}`,
    );
  }
}

console.log('\n=== 5. the parts.js lens, at 2, 3 and 4 ships ===');
console.log(
  `  ${PARTS_BUILDS} random legal builds per hull from randomProfile, each in ` +
    `${PARTS_FIELDS} fields at every field size` +
    (PARTS_SAMPLE ? `, sample ${PARTS_SAMPLE}` : ''),
);
console.log(
  '  edge is pairwise on placing and 50% is even at every field size; x-ev is an outright win' +
    '\n  as a multiple of an even share.' +
    ' verdicts are parts.js bands on edge: dom 60+, str 53+, even,' +
    '\n  weak 47-, trap 40-. thin means under 40 comparisons somewhere in the row, or the part in' +
    '\n  fewer than one build in twenty; a thin row is left out of the two summaries below.',
);

const lensPooled = Object.fromEntries(COUNTS.map((c) => [c, blankStat()]));
const lensCarried = Object.fromEntries(LENS_IDS.map((id) => [id, 0]));
const perHull = {};
let lensBuilds = 0;

for (const hullIndex of HULL_SET) {
  const budget = budgetFor(config.ROUNDS, hullIndex);
  const rng = makeRng(20260728 + hullIndex + PARTS_SAMPLE * 977);
  const builds = [];
  for (let i = 0; i < PARTS_BUILDS; i++) builds.push(randomBuild(hullIndex, budget, rng));
  const carried = Object.fromEntries(LENS_IDS.map((id) => [id, 0]));
  for (const b of builds) for (const id of LENS_IDS) if (b.counts[id]) carried[id]++;
  for (const id of LENS_IDS) lensCarried[id] += carried[id];
  lensBuilds += PARTS_BUILDS;

  const stats = {};
  const meta = {};
  for (const count of COUNTS) {
    const stat = blankStat();
    let battles = 0;
    let draws = 0;
    let time = 0;
    for (let i = 0; i < builds.length; i++) {
      for (let k = 1; k <= PARTS_FIELDS; k++) {
        // Strides coprime enough with the pool size that the seats are distinct builds, as in
        // parts.js: each build meets several different opponents rather than its neighbour five
        // times.
        const seats = [];
        for (let j = 0; j < count; j++) seats.push(builds[(i + j * k * 7) % builds.length]);
        const seed = (i * 131 + k) * 7 + count;
        // Section 2 finds a real seat bias in the ring, so the seat a build sits in is shuffled:
        // otherwise a build's position in the pool decides its position on the circle and the bias
        // reads as a part result.
        const shuffle = makeRng(seed * 31 + 5);
        for (let a = seats.length - 1; a > 0; a--) {
          const b = shuffle.int(0, a);
          const swap = seats[a];
          seats[a] = seats[b];
          seats[b] = swap;
        }
        const out = playBattle(real, seats.map((s) => ship.cloneDesign(s.design)), hullIndex, seed);
        battles++;
        time += out.time;
        if (out.winner === null) {
          draws++;
          continue;
        }
        scoreBattle(stat, seats, out);
      }
    }
    stats[count] = stat;
    meta[count] = { battles, draws, time: time / battles };
    addInto(lensPooled[count], stat);
  }
  perHull[hullIndex] = { stats, carried, meta };

  console.log(`\n  --- ${HULLS[hullIndex].name} ---`);
  console.log(
    `  2/3/4 ships: ${meta[2].battles} battles each, mean ` +
      `${COUNTS.map((c) => meta[c].time.toFixed(1)).join('/')}s, draws ` +
      `${COUNTS.map((c) => ((100 * meta[c].draws) / meta[c].battles).toFixed(0)).join('/')}%`,
  );
  lensTable(lensRows(stats, carried, PARTS_BUILDS));
}

console.log('\n  --- all hulls ---');
const pooledRows = lensRows(lensPooled, lensCarried, lensBuilds);
lensTable(pooledRows);

// Which readings the field size actually creates. A part that is already a trap in a duel on this
// hull is not a melee problem, and CLAUDE.md's rule that a different best gun per hull is good
// design applies to field size the same way -- so the useful output is not the list of extremes but
// the list of extremes the duel does not already have.
console.log(
  '\n  --- every dom or trap reading at 3 or 4 ships, against the duel on the same hull ---',
);
console.log('  hull              part          2 ships  3 ships  4 ships   made by the melee');
const dir = (edge) => (edge >= 0.6 ? 1 : edge <= 0.4 ? -1 : 0);
for (const hullIndex of HULL_SET) {
  const rows = lensRows(perHull[hullIndex].stats, perHull[hullIndex].carried, PARTS_BUILDS);
  for (const r of rows) {
    const melee = [r.by[1], r.by[2]];
    if (!melee.some((c) => dir(c.edge) !== 0)) continue;
    if (isThin(r)) continue; // the long gun and the magazine have nothing to read
    const worst = melee.reduce((a, b) => (Math.abs(b.edge - 0.5) > Math.abs(a.edge - 0.5) ? b : a));
    const made =
      dir(r.by[0].edge) === 0
        ? 'yes'
        : dir(r.by[0].edge) === dir(worst.edge)
          ? 'no, the duel reads the same way'
          : 'yes, and it reverses the duel';
    console.log(
      `  ${HULLS[hullIndex].name.padEnd(17)} ${r.id.padEnd(12)} ` +
        r.by.map((c) => pct(c.edge).padStart(7)).join('  ') +
        `   ${made}`,
    );
  }
}

console.log('\n  --- how far the field size moves each part, pooled over hulls ---');
for (const r of [...pooledRows].sort(
  (a, b) => Math.abs(b.by[2].edge - b.by[0].edge) - Math.abs(a.by[2].edge - a.by[0].edge),
)) {
  if (isThin(r)) continue;
  const shift = (r.by[2].edge - r.by[0].edge) * 100;
  console.log(
    `  ${r.id.padEnd(12)} ${r.by.map((c) => pct(c.edge).padStart(4)).join(' -> ')}   ` +
      `${shift >= 0 ? '+' : ''}${shift.toFixed(0)} pts from the duel to a four-way`,
  );
}
