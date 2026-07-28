// The battle: a deterministic clock over two ships. Takes two designs, a seed and a wind
// direction, and advances in fixed ticks. Inputs (the ammunition toggle) are applied between ticks,
// so the same seed plus the same input stream always replays the same battle. No renderer
// dependency, no DOM, no Math.random.
//
// This file only orchestrates. Sailing is in steering.js, firing and shot in gunnery.js, and what a
// ball does on arrival in damage.js.

import { HULLS } from '../data/hulls.js';
import { makeRng } from './rng.js';
import { makeBattleShip, refreshSystems, commitDamage, structureFraction, shipName } from './ship.js';
import { fightingProfile, steer, separate } from './steering.js';
import { fireGuns, stepProjectiles } from './gunnery.js';
import {
  TICK,
  BATTLE_CAP,
  minSeparation,
  drawOrbitSense,
  RELOAD_STAGGER,
  AMMO_SWITCH_RELOAD,
  START_OFFSET,
} from '../config.js';

export function createBattle({ designs, hullIndex, seed, windTo }) {
  const rng = makeRng(seed);
  const ships = [
    makeBattleShip(designs[0], hullIndex, 0, { x: -START_OFFSET.x, z: START_OFFSET.z }, 0),
    makeBattleShip(designs[1], hullIndex, 1, { x: START_OFFSET.x, z: -START_OFFSET.z }, Math.PI),
  ];
  // Which beam the fight settles on. Drawn here rather than fixed or shown, because a predictable
  // engaged side is a sheltered side to hide the crew and powder behind.
  const sense = drawOrbitSense(rng);
  // The enemy sits off the port beam when the circle runs one way and off the starboard beam when
  // it runs the other.
  const engagedArc = sense > 0 ? -Math.PI / 2 : Math.PI / 2;

  for (const ship of ships) {
    ship.profile = fightingProfile(ship);
    // Hands go to the battery that will bear. Manning in cell order instead meant a ship with crew
    // for half its guns could put every one of them on the disengaged flank and spend the whole
    // battle unable to fire a shot -- found by building exactly that ship and watching it do
    // nothing for forty seconds.
    ship.guns.sort((x, y) => {
      const bearsX = x.arc === 0 || x.arc === engagedArc ? 0 : 1;
      const bearsY = y.arc === 0 || y.arc === engagedArc ? 0 : 1;
      if (bearsX !== bearsY) return bearsX - bearsY;
      return x.cell.key < y.cell.key ? -1 : 1;
    });
    refreshSystems(ship);
    // Start the battery out of step, so it rolls down the side instead of clapping.
    for (const gun of ship.guns) {
      gun.readyAt = rng.range(0, gun.spec.reload * RELOAD_STAGGER);
    }
  }

  const noted = new Set();
  const battle = {
    time: 0,
    sense,
    minSeparation: minSeparation(HULLS[hullIndex].length),
    ships,
    projectiles: [],
    // The consumer owns draining this. Nothing in here clears it.
    effects: [],
    log: [],
    windTo,
    windCos: Math.cos(windTo),
    windSin: Math.sin(windTo),
    over: false,
    winner: null,
    reason: '',
    rng,

    note(text) {
      battle.log.push({ t: battle.time, text });
    },
    // For events that would otherwise repeat every time another mast goes over the side.
    noteOnce(key, text) {
      if (noted.has(key)) return;
      noted.add(key);
      battle.note(text);
    },

    setAmmo(index, ammo) {
      const ship = ships[index];
      if (battle.over || ship.ammo === ammo) return;
      ship.ammo = ammo;
      // Switching costs you the guns that were already loaded, which is what makes *when* you
      // switch the decision rather than *whether*.
      for (const gun of ship.guns) {
        gun.readyAt = Math.max(gun.readyAt, battle.time + AMMO_SWITCH_RELOAD);
      }
      battle.effects.push({ type: 'ammo', ship: index, ammo });
    },

    advance(dt) {
      let remaining = Math.min(dt, 0.25); // never simulate a huge catch-up jump
      while (remaining > 0 && !battle.over) {
        const step = Math.min(TICK, remaining);
        tick(battle, step);
        remaining -= step;
      }
    },

    finish() {
      for (const ship of ships) commitDamage(ship);
    },
  };

  return battle;
}

function tick(battle, dt) {
  battle.time += dt;
  const [a, b] = battle.ships;

  steer(battle, a, b, dt);
  steer(battle, b, a, dt);
  separate(a, b, battle.minSeparation);

  fireGuns(battle, a, b);
  fireGuns(battle, b, a);
  stepProjectiles(battle, dt);

  checkEnd(battle);
}

const isDead = (ship) => !ship.helm.alive || ship.aliveCells === 0;

function checkEnd(battle) {
  const [a, b] = battle.ships;
  const aDead = isDead(a);
  const bDead = isDead(b);

  if (aDead || bDead) {
    battle.over = true;
    if (aDead && bDead) {
      battle.winner = null;
      battle.reason = 'Both ships strike their colours';
    } else {
      battle.winner = aDead ? 1 : 0;
      battle.reason = `${shipName(aDead ? a : b)} strikes colours`;
    }
    return;
  }

  // Overtime makes gunnery steadily deadlier from OVERTIME_AT, so this hard stop is a backstop
  // rather than the usual ending. Draws run under 1%.
  const stalemate = !a.canFire && !b.canFire;
  if (battle.time >= BATTLE_CAP || (stalemate && battle.time > 5)) {
    battle.over = true;
    const fa = structureFraction(a);
    const fb = structureFraction(b);
    if (Math.abs(fa - fb) < 0.01) {
      battle.winner = null;
      battle.reason = 'Both ships break off, evenly mauled';
    } else {
      battle.winner = fa > fb ? 0 : 1;
      battle.reason = stalemate
        ? 'Neither ship can fire; the day goes to the sounder hull'
        : `${shipName(battle.ships[fa > fb ? 0 : 1])} is the sounder ship at the bell`;
    }
  }
}
