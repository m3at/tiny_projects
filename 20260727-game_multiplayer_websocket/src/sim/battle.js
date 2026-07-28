// The battle: a deterministic clock over two to four ships. Takes the designs, a seed and a wind
// direction, and advances in fixed ticks. Inputs (the ammunition toggle) are applied between ticks,
// so the same seed plus the same input stream always replays the same battle. No renderer
// dependency, no DOM, no Math.random. That property is what makes the networked version a relay of
// toggles rather than a stream of positions, and tools/golden.js is what holds it.
//
// Two ships is a duel and three or four is a melee, and the difference is deliberately narrow: the
// melee adds target selection and pulls incoming damage back for the extra guns pointed at you.
// Everything below reduces exactly to the two-ship code when there are two ships -- same
// arithmetic, same order, same draws from the rng -- which is checked, not hoped for.
//
// This file only orchestrates. Sailing and target choice are in steering.js, firing and shot in
// gunnery.js, and what a ball does on arrival in damage.js.

import { HULLS } from '../data/hulls.js';
import { fcos, fsin } from './geometry.js';
import { makeRng } from './rng.js';
import { makeBattleShip, refreshSystems, commitDamage, structureFraction, shipName } from './ship.js';
import { fightingProfile, steer, separate, pickTarget } from './steering.js';
import { fireGuns, stepProjectiles } from './gunnery.js';
import { grind } from './damage.js';
import {
  TICK,
  BATTLE_CAP,
  minSeparation,
  drawOrbitSense,
  RELOAD_STAGGER,
  AMMO_SWITCH_RELOAD,
  SHIP_COUNT_DAMAGE,
  TARGET_RECHECK,
  arenaRadius,
  startPositions,
} from '../config.js';

// The most simulated time one advance() call will run, however long the caller was away. A tab that
// was backgrounded for a minute should resume, not fast-forward the whole battle in one frame.
const MAX_CATCHUP = 0.25;

export function createBattle({ designs, hullIndex, seed, windTo }) {
  const rng = makeRng(seed);
  const count = designs.length;
  const spots = startPositions(count);
  const ships = designs.map((design, i) =>
    makeBattleShip(design, hullIndex, i, spots[i], spots[i].heading),
  );
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

  const arena = arenaRadius(count);
  const edge = arena * 0.8;

  const noted = new Set();
  const battle = {
    time: 0,
    // The tick number is the clock every networked input is stamped against, and the only thing two
    // machines replaying this battle can agree on. `carry` is the fraction of a tick left over from
    // the last advance().
    tickCount: 0,
    carry: 0,
    sense,
    minSeparation: minSeparation(HULLS[hullIndex].length),
    ships,
    shipCount: count,
    afloat: count,
    // The arena grows with the field. steering.js reads edge/edgeSq every tick, so both are
    // precomputed here rather than derived from the radius per ship per tick.
    arenaRadius: arena,
    edge,
    edgeSq: edge * edge,
    // Three or four ships means two or three batteries pointed at you instead of one, so incoming
    // damage is scaled back to hold the round length the duel was tuned to. Exactly 1 at two ships.
    damageScale: SHIP_COUNT_DAMAGE[count] ?? 1,
    retargetAt: TARGET_RECHECK,
    // When each pair of hulls may next grind on each other. Part of the battle's state, so it
    // replays identically; a flat array because there are at most six pairs.
    contactAt: new Float64Array(count * count),
    // Best first, filled in when the battle ends. The economy pays comeback money by placing, so a
    // four-way needs an order and not just a winner.
    placing: null,
    projectiles: [],
    // The consumer owns draining this. Nothing in here clears it.
    effects: [],
    log: [],
    windTo,
    windCos: fcos(windTo),
    windSin: fsin(windTo),
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

    // Whole ticks and nothing else, which is what lets an input be stamped with a tick number.
    //
    // This used to take the caller's dt, subdivide it, and run whatever fraction of a tick was left
    // over at the end. Measured: advance(0.25) ran sixteen ticks, fifteen full and one of 5e-17, and
    // advance(0.05) ran four. So the step size was really the caller's frame time, and a browser
    // drawing at an uneven rate ran a different number of differently sized ticks than the harness
    // or than another browser. Invisible in a game watched on one machine, fatal for one replayed on
    // two. The remainder is carried instead.
    advance(dt) {
      battle.carry += Math.min(dt, MAX_CATCHUP); // never simulate a huge catch-up jump
      const n = Math.floor(battle.carry / TICK);
      if (n <= 0) return;
      battle.carry -= n * TICK;
      battle.advanceTicks(n);
    },

    advanceTicks(n) {
      for (let i = 0; i < n && !battle.over; i++) {
        tick(battle, TICK);
        battle.tickCount++;
      }
    },

    finish() {
      for (const ship of ships) commitDamage(ship);
    },
  };

  refreshFoes(battle);
  // Round robin, and not the nearest enemy. On a ring every ship is exactly equidistant from its two
  // neighbours, so "nearest" is decided by which of two identical distances came out a bit smaller in
  // float32 -- and it does not come out symmetrically. Measured with tools/melee.js: seats won 79%,
  // 7% and 14% of three-ship battles, because ships 1 and 2 both picked each other, locked on through
  // TARGET_SWITCH_MARGIN, and fought a private duel while seat 0 was left alone. A rotation is the one
  // assignment no seat can be favoured by, and at two ships it is still the other ship.
  for (const ship of ships) {
    ship.target = ships[(ship.index + 1) % count];
  }

  return battle;
}

// Who each ship may shoot at: everyone else still afloat. fireGuns walks this list for every gun on
// every tick, so it must not carry the dead -- and it changes only when someone strikes, which is at
// most three times in a battle.
function refreshFoes(battle) {
  for (const ship of battle.ships) {
    ship.foes = battle.ships.filter((other) => other !== ship && !other.out);
  }
}

function tick(battle, dt) {
  battle.time += dt;
  const ships = battle.ships;
  const n = ships.length;

  // Only a melee reconsiders. In a duel there is one enemy and the answer was settled at creation.
  if (n > 2 && battle.time >= battle.retargetAt) {
    battle.retargetAt = battle.time + TARGET_RECHECK;
    for (let i = 0; i < n; i++) {
      if (!ships[i].out) ships[i].target = pickTarget(battle, ships[i]);
    }
  }

  for (let i = 0; i < n; i++) {
    const ship = ships[i];
    if (ship.out) continue;
    // A target that has just struck is dropped at once rather than at the next recheck, or a ship
    // spends up to TARGET_RECHECK circling a hulk.
    if (ship.target === null || ship.target.out) ship.target = pickTarget(battle, ship);
    if (ship.target !== null) steer(battle, ship, ship.target, dt);
  }

  for (let i = 0; i < n; i++) {
    if (ships[i].out) continue;
    for (let j = i + 1; j < n; j++) {
      if (ships[j].out) continue;
      // separate() reports how deeply the two were inside each other before it pushed them apart.
      if (separate(ships[i], ships[j], battle.minSeparation) > 0) grind(battle, ships[i], ships[j]);
    }
  }

  for (let i = 0; i < n; i++) {
    if (!ships[i].out) fireGuns(battle, ships[i], ships[i].foes);
  }
  stepProjectiles(battle, dt);

  checkEnd(battle);
}

const isDead = (ship) => !ship.helm.alive || ship.aliveCells === 0;

// A ship that has struck leaves the fight, so shot already on its way to her falls in the water
// instead of pounding a hulk. Doing it here rather than testing `out` per projectile per tick keeps
// the check out of the busiest loop in the simulation.
function dropShotAt(battle, ship) {
  const list = battle.projectiles;
  let keep = 0;
  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    if (p.target === ship.index) battle.effects.push({ type: 'splash', x: p.x, z: p.z });
    else list[keep++] = p;
  }
  list.length = keep;
}

// Best first: whoever is still afloat, soundest first, then the ones that struck, latest first.
// Survival outranks structure -- a hulk that stayed afloat placed above one that did not.
function placings(battle) {
  return battle.ships
    .slice()
    .sort((x, y) => {
      if (x.out !== y.out) return x.out ? 1 : -1;
      if (x.out) return y.outAt - x.outAt;
      return structureFraction(y) - structureFraction(x);
    })
    .map((s) => s.index);
}

// Runs every tick, so it stays small enough for V8 to inline into tick(). Writing the verdict here
// as well cost 6% of simulation throughput -- the function grew past the inlining budget and the
// whole of it stopped being inlined, which the profiler showed as a new `checkEnd` entry and a tick
// that had doubled. Everything cold lives in the two functions below, which run once per battle.
function checkEnd(battle) {
  const ships = battle.ships;
  let struck = false;
  let stalemate = true;
  for (let i = 0; i < ships.length; i++) {
    const ship = ships[i];
    if (ship.out) continue;
    if (!ship.helm.alive || ship.aliveCells === 0) {
      struck = true;
      continue;
    }
    if (ship.canFire) stalemate = false;
  }

  if (struck) strikeColours(battle);
  if (battle.afloat <= 1) return settleByStrike(battle);
  // Overtime makes gunnery steadily deadlier from OVERTIME_AT, so this hard stop is a backstop
  // rather than the usual ending. Draws run under 1%.
  if (battle.time >= BATTLE_CAP || (stalemate && battle.time > 5)) settleAtBell(battle, stalemate);
}

function strikeColours(battle) {
  for (const ship of battle.ships) {
    if (ship.out || !isDead(ship)) continue;
    ship.out = true;
    ship.outAt = battle.time;
    battle.afloat--;
    dropShotAt(battle, ship);
    // A duel says so in the verdict instead; a melee needs the running commentary, since the fight
    // carries on without her.
    if (battle.shipCount > 2) battle.note(`${shipName(ship)} strikes colours`);
  }
  refreshFoes(battle);
}

function settleByStrike(battle) {
  const ships = battle.ships;
  battle.over = true;
  const survivor = ships.find((s) => !s.out) ?? null;
  if (battle.shipCount === 2) {
    if (survivor === null) {
      battle.winner = null;
      battle.reason = 'Both ships strike their colours';
    } else {
      battle.winner = survivor.index;
      battle.reason = `${shipName(ships[0].out ? ships[0] : ships[1])} strikes colours`;
    }
  } else if (survivor !== null) {
    battle.winner = survivor.index;
    battle.reason = `${shipName(survivor)} is the last afloat`;
  } else {
    battle.winner = null;
    battle.reason = 'Every ship strikes her colours';
  }
  battle.placing = placings(battle);
}

// Decided on structure among whoever is still afloat. A gap under a hundredth at the top is a draw,
// which is under 1% of battles.
function settleAtBell(battle, stalemate) {
  const duel = battle.shipCount === 2;
  battle.over = true;
  const live = battle.ships.filter((s) => !s.out);
  live.sort((x, y) => structureFraction(y) - structureFraction(x));
  const best = live[0];
  if (structureFraction(best) - structureFraction(live[1]) < 0.01) {
    battle.winner = null;
    battle.reason = duel ? 'Both ships break off, evenly mauled' : 'The field breaks off, evenly mauled';
  } else {
    battle.winner = best.index;
    battle.reason = stalemate
      ? duel
        ? 'Neither ship can fire; the day goes to the sounder hull'
        : 'No gun left in the fight; the day goes to the sounder hull'
      : `${shipName(best)} is the sounder ship at the bell`;
  }
  battle.placing = placings(battle);
}
