// Full-precision state dump of a fixed grid of battles, one line per sampled tick.
//
// This is not golden.js. golden.js prints a rounded fingerprint and answers "did this refactor
// change the game"; it cannot see a difference of 1e-14, which is exactly the size of difference
// that matters when two different JavaScript engines run the same simulation. Every double here is
// printed as its raw 64 bits, so any difference at all shows up.
//
// Engine-agnostic on purpose: no node: imports, no console assumptions. It runs under `node` and
// under Safari's `jsc`, which is how tools/engines.js compares the two.

import * as ship from '../src/sim/ship.js';
import { createBattle } from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { HULLS } from '../src/data/hulls.js';
import { ROUNDS, TICK } from '../src/config.js';
import { makeBot, REACTION } from '../src/bot.js';

const say = typeof print === 'function' ? print : console.log;

// Raw bits, because a decimal rendering of a double is a lossy way to ask "are these the same
// number" and toString(16) does not exist for doubles.
const view = new DataView(new ArrayBuffer(8));
function bits(x) {
  view.setFloat64(0, x);
  return (
    view.getUint32(0).toString(16).padStart(8, '0') + view.getUint32(4).toString(16).padStart(8, '0')
  );
}

const SAMPLE_TICKS = 15; // every quarter second of simulated time
const PAIRS = [
  ['brawler', 'sniper'],
  ['massed', 'crusher'],
  ['harasser', 'mixed'],
];

function budget(hullIndex) {
  return ROUNDS.slice(0, hullIndex + 1).reduce((s, r) => s + r.scrap, 0);
}

let battles = 0;
for (let hullIndex = 0; hullIndex < HULLS.length; hullIndex++) {
  for (const pair of PAIRS) {
    for (let s = 0; s < 2; s++) {
      const designs = pair.map((t) => {
        const d = ship.createDesign();
        autoBuild(d, hullIndex, budget(hullIndex), ARCHETYPES[t]);
        return d;
      });
      const seed = s * 7919 + hullIndex;
      const battle = createBattle({
        designs,
        hullIndex,
        seed,
        windTo: (seed % 360) * (Math.PI / 180),
      });
      const bot = makeBot(battle, {});
      const tag = `${battles}`;
      let ticks = 0;
      let guard = 0;
      while (!battle.over && guard++ < 90 / REACTION) {
        bot.update(REACTION);
        // Stepped in the bot's interval, then sampled between: advance() subdivides internally, so
        // the battle is identical either way.
        battle.advance(REACTION);
        battle.effects.length = 0;
        ticks += Math.round(REACTION / TICK);
        if (ticks % SAMPLE_TICKS !== 0) continue;
        const cols = [tag, String(ticks)];
        for (const sh of battle.ships) {
          cols.push(bits(sh.x), bits(sh.z), bits(sh.heading), bits(sh.speed));
          cols.push(bits(sh.crewLost), String(sh.crew), String(sh.aliveCells));
        }
        say(cols.join(' '));
      }
      say(
        `= ${tag} winner=${battle.winner} t=${bits(battle.time)} ` +
          battle.ships.map((sh) => bits(ship.structureFraction(sh))).join(' '),
      );
      battles++;
    }
  }
}
say(`# ${battles} battles`);
