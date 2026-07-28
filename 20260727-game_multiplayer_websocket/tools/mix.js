// Does the mixer actually hear the battle?
//
//   node tools/mix.js
//
// tools/audio.js measures one sound in isolation. Nothing measured what happens when a battle
// hands the mixer several hundred events, which is the case that decides whether a broadside
// sounds like a broadside. This runs real battles at frame cadence -- the same cadence main.js
// drains effects at -- and reports, per policy:
//
//   heard    share of events that reach a voice rather than being dropped as too close together
//   burst    the most events of that kind inside 250ms, which is what a big volley feels like
//   voices   peak overlapping voices, using each sound's real duration. The polyphony bill.
//   load     peak sum of voice levels, a proxy for how hard the limiter is working
//
// The point of the drop rule is that sixteen guns is not sixteen sounds; the point of measuring it
// is that the first rule threw away a third of the gunfire and half the hits, and did it hardest
// exactly when the most was happening. A quiet ship and a ship of the line came out the same.

import * as ship from '../src/sim/ship.js';
import * as battleMod from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import * as config from '../src/config.js';
import { makeBot } from '../src/bot.js';
import { budgetFor, windForSeed } from './harness.js';

// Effect type to mixer kind, and roughly how long each voice rings for. The durations are read off
// sfx.js; they only need to be close, since they are counting overlap and not rendering anything.
const KIND = {
  muzzle: 'cannon',
  impact: 'impact',
  splash: 'splash',
  destroy: 'break',
  sever: 'break',
  detonate: 'blast',
};
const DUR = { cannon: 0.45, impact: 0.22, splash: 0.32, break: 1.1, blast: 2.2 };
const LEVEL = { cannon: 1, impact: 0.6, splash: 0.45, break: 0.6, blast: 1.1 };

// Each policy takes (kind, now) and returns [when, gain]; a negative time means dropped.

// The rule shipped first: a hard minimum gap, and anything arriving inside it is discarded.
function gapDrop(spacing) {
  const last = {};
  return (kind, now) => {
    if (last[kind] !== undefined && now - last[kind] < spacing[kind]) return [-1, 0];
    last[kind] = now;
    return [now, 1];
  };
}

// The replacement: a scheduling cursor per kind. An event is placed at the cursor, or at now if the
// cursor has fallen behind, and the cursor then advances by gap. A clump comes out as a rolling
// burst rather than a single sound, and only a backlog longer than lead is dropped.
//
// With `duck`, each kind also carries a decaying count of how much of it is already sounding, and
// new voices are scaled by 1/sqrt(1 + duck * that). Twelve guns then come out louder than three but
// nothing like four times louder, which is both what a broadside sounds like and what keeps the
// limiter off. Without it the peak level is 17 and every busy moment is squashed to the same
// loudness, which is the complaint that started this.
function queued(voice, duck = 0) {
  const cursor = {};
  const energy = {};
  const seen = {};
  return (kind, now) => {
    const v = voice[kind];
    const at = Math.max(now, cursor[kind] || 0);
    if (at - now > v.lead) return [-1, 0];
    cursor[kind] = at + v.gap;
    if (!duck) return [at, 1];
    // Exponential decay towards zero over the kind's own ring-out time.
    const e = (energy[kind] || 0) * Math.exp(-Math.max(0, at - (seen[kind] ?? at)) / DUR[kind]);
    energy[kind] = e + 1;
    seen[kind] = at;
    return [at, 1 / Math.sqrt(1 + duck * e)];
  };
}

const VOICE = {
  cannon: { gap: 0.028, lead: 0.3 },
  impact: { gap: 0.022, lead: 0.22 },
  splash: { gap: 0.05, lead: 0.15 },
  break: { gap: 0.14, lead: 0.3 },
  blast: { gap: 0.2, lead: 0.4 },
};

const POLICIES = {
  'gap-drop (old)': () => gapDrop({ cannon: 0.05, impact: 0.04, splash: 0.1, break: 0.22, blast: 0.25 }),
  queued: () => queued(VOICE),
  'queued + duck 0.5': () => queued(VOICE, 0.5),
  'queued + duck 1.0': () => queued(VOICE, 1),
};

const names = Object.keys(ARCHETYPES);
const SEEDS = Number(process.argv[2] || 150);
const FRAME = config.TICK;

function buildFor(hullIndex, archetype) {
  const design = ship.createDesign();
  autoBuild(design, hullIndex, budgetFor(config.ROUNDS, hullIndex), ARCHETYPES[archetype]);
  return design;
}

// One pass over every battle, feeding each policy the same event stream.
const stats = {};
for (const name of Object.keys(POLICIES)) {
  stats[name] = { heard: {}, total: {}, burst: {}, voices: 0, load: 0 };
}
let seconds = 0;

for (let seed = 1; seed <= SEEDS; seed++) {
  const hullIndex = seed % 5;
  const designs = [
    buildFor(hullIndex, names[seed % names.length]),
    buildFor(hullIndex, names[(seed + 3) % names.length]),
  ];
  const battle = battleMod.createBattle({
    designs,
    hullIndex,
    seed,
    windTo: windForSeed(seed),
  });
  const bot = makeBot(battle, { mode: 'grape' });

  const takers = Object.fromEntries(Object.entries(POLICIES).map(([n, f]) => [n, f()]));
  // Scheduled voices per policy, as (kind, start) pairs, pruned as they expire.
  const live = Object.fromEntries(Object.keys(POLICIES).map((n) => [n, []]));
  const recent = Object.fromEntries(Object.keys(POLICIES).map((n) => [n, []]));

  let now = 0;
  let guard = 0;
  while (!battle.over && guard++ < 90 / FRAME) {
    bot.update(FRAME);
    battle.advance(FRAME);
    now += FRAME;

    for (const e of battle.effects) {
      const kind = KIND[e.type];
      if (!kind) continue;
      for (const [name, take] of Object.entries(takers)) {
        const s = stats[name];
        s.total[kind] = (s.total[kind] || 0) + 1;
        const [at, gain] = take(kind, now);
        if (at < 0) continue;
        s.heard[kind] = (s.heard[kind] || 0) + 1;
        if (kind === 'cannon' && gain < (s.quietest ?? 1)) s.quietest = gain;
        live[name].push([kind, at, gain]);
        recent[name].push([kind, at]);
      }
    }
    battle.effects.length = 0;

    // Peak overlap and peak level, evaluated at this instant.
    for (const name of Object.keys(POLICIES)) {
      const s = stats[name];
      const l = live[name];
      let keep = 0;
      let load = 0;
      for (let i = 0; i < l.length; i++) {
        const [kind, at, gain] = l[i];
        if (at + DUR[kind] < now) continue;
        l[keep++] = l[i];
        if (at <= now) load += LEVEL[kind] * gain;
      }
      l.length = keep;
      let active = 0;
      for (let i = 0; i < l.length; i++) if (l[i][1] <= now) active++;
      if (active > s.voices) s.voices = active;
      if (load > s.load) s.load = load;

      // Events of each kind inside the last 250ms.
      const r = recent[name];
      let rk = 0;
      const counts = {};
      for (let i = 0; i < r.length; i++) {
        if (r[i][1] < now - 0.25) continue;
        r[rk++] = r[i];
        counts[r[i][0]] = (counts[r[i][0]] || 0) + 1;
      }
      r.length = rk;
      for (const [kind, c] of Object.entries(counts)) {
        if (c > (s.burst[kind] || 0)) s.burst[kind] = c;
      }
    }
  }
  seconds += battle.time;
}

console.log(`\n  ${SEEDS} battles, ${seconds.toFixed(0)}s of fighting\n`);
const kinds = ['cannon', 'impact', 'splash', 'break', 'blast'];
for (const [name, s] of Object.entries(stats)) {
  console.log(`  ${name}`);
  console.log('    kind     offered   heard        per sec   burst/250ms');
  for (const kind of kinds) {
    const t = s.total[kind] || 0;
    const h = s.heard[kind] || 0;
    if (!t) continue;
    console.log(
      `    ${kind.padEnd(7)} ${String(t).padStart(7)} ${String(h).padStart(7)} ` +
        `${`${((100 * h) / t).toFixed(0)}%`.padStart(6)} ${(h / seconds).toFixed(1).padStart(9)} ` +
        `${String(s.burst[kind] || 0).padStart(9)}`,
    );
  }
  console.log(
    `    peak voices ${s.voices}, peak level ${s.load.toFixed(2)}` +
      `, quietest cannon ${(s.quietest ?? 1).toFixed(2)} of full\n`,
  );
}
