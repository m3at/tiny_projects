// Sanity check: do the systems that carry the drama actually trigger in real battles?
import { createDesign } from '../src/sim/ship.js';
import { createBattle } from '../src/sim/battle.js';
import { autoBuild, ARCHETYPES } from '../src/autobuild.js';
import { ROUNDS, TICK } from '../src/config.js';

const names = Object.keys(ARCHETYPES);
const tally = {};
const logs = {};
let battles = 0, grapeUsed = 0;

for (const hullIndex of [0, 2, 4]) {
  let budget = 0;
  for (let i = 0; i <= hullIndex; i++) budget += ROUNDS[i].scrap;
  for (const a of names) for (const b of names) {
    if (a === b) continue;
    for (let s = 0; s < 12; s++) {
      const da = createDesign(), db = createDesign();
      autoBuild(da, hullIndex, budget, ARCHETYPES[a]);
      autoBuild(db, hullIndex, budget, ARCHETYPES[b]);
      const battle = createBattle({ designs: [da, db], hullIndex, seed: s * 131 + hullIndex, windTo: s });
      battles++;
      let usedGrape = false;
      while (!battle.over) {
        battle.setAmmo(0, battle.time > 6 ? 'grape' : 'round');
        battle.advance(TICK);
        for (const e of battle.effects) tally[e.type] = (tally[e.type] || 0) + 1;
        if (battle.ships[0].ammo === 'grape') usedGrape = true;
        battle.effects.length = 0;
      }
      if (usedGrape) grapeUsed++;
      for (const l of battle.log) {
        const kind = l.text.includes('detonates') ? 'detonation' : l.text.includes('break away') ? 'sever' : 'dismast';
        logs[kind] = (logs[kind] || 0) + 1;
      }
    }
  }
}
console.log(`${battles} battles`);
console.log('effects:', Object.fromEntries(Object.entries(tally).map(([k, v]) => [k, Math.round(v / battles * 10) / 10])), '(per battle)');
console.log('notable events total:', logs, ` | grape used in ${grapeUsed}/${battles}`);
