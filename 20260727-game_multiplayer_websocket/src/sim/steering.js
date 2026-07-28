// How the ships sail, and who each one is fighting. Nobody steers: each ship derives what it wants
// from the guns it carries and holds station on its target.
//
// This file is where the game's dullest failure lived, so the reasoning is worth keeping. The
// ships used to circle in opposite senses, which is a parallel course: each kept the other abeam
// by sailing alongside it, so the pair held its range perfectly and marched off the map together,
// firing nothing, until the arena hauled them back. That read as "one volley, a long silence, then
// the action resumes in the corner". Both ships now take the same sense, so they orbit their
// common midpoint.

import { len, wrapAngle, fatan2, fcos, fsin } from './geometry.js';
import {
  BASE_SPEED,
  BASE_TURN,
  ORBIT_CLOSE,
  ORBIT_TOLERANCE,
  ORBIT_RETREAT,
  PREFERRED_RANGE_FRACTION,
  TARGET_SWITCH_MARGIN,
  windFactorFrom,
} from '../config.js';

const sqDist = (a, b) => (b.x - a.x) * (b.x - a.x) + (b.z - a.z) * (b.z - a.z);

// Who a ship is fighting. In a duel there is only one answer and it never changes. In a melee it is
// the nearest enemy still afloat, revisited every TARGET_RECHECK seconds and only given up for a
// rival that is clearly closer: a ship equidistant between two enemies otherwise swaps every few
// ticks and sails straight down the middle with its guns bearing on nothing.
export function pickTarget(battle, ship) {
  const ships = battle.ships;
  if (ships.length === 2) return ships[1 - ship.index];
  let best = ship.target && !ship.target.out ? ship.target : null;
  // Squared, so the margin squares with it. A rival has to be inside TARGET_SWITCH_MARGIN of the
  // current range to be worth turning for.
  let bestSq = best ? sqDist(ship, best) * TARGET_SWITCH_MARGIN * TARGET_SWITCH_MARGIN : Infinity;
  for (let i = 0; i < ships.length; i++) {
    const foe = ships[i];
    if (foe === ship || foe.out) continue;
    const d = sqDist(ship, foe);
    if (d < bestSq) {
      best = foe;
      bestSq = d;
    }
  }
  return best;
}

// How a ship wants to fight, derived from the guns it actually carries, weighted by how much
// damage each contributes. A long-gun ship charges bow-on and stays far; a broadside ship presents
// its flank and closes.
export function fightingProfile(ship) {
  let wSum = 0;
  let range = 0;
  let arcBias = 0;
  for (const gun of ship.guns) {
    const spec = gun.spec;
    const w = (spec.shots * spec.round.damage) / spec.reload;
    const bias = spec.arc === 'bow' ? 0 : 90;
    wSum += w;
    range += w * spec.range;
    arcBias += w * bias;
  }
  if (wSum === 0) return { range: 34, arcBias: 90 };
  return {
    range: Math.min(88, Math.max(16, range / wSum)) * PREFERRED_RANGE_FRACTION,
    arcBias: arcBias / wSum,
  };
}

export function steer(battle, ship, enemy, dt) {
  const dx = enemy.x - ship.x;
  const dz = enemy.z - ship.z;
  const d = len(dx, dz);
  const bearing = fatan2(dx, -dz);

  const R = ship.profile.range;
  const bias = (ship.profile.arcBias * Math.PI) / 180;

  // One continuous controller, not a ladder of range bands. `hold` is the heading offset that
  // keeps the guns bearing at the preferred range: a quarter turn off the bearing for a broadside
  // ship, straight at the enemy for a bow chaser. `err` says whether the range wants closing or
  // opening, and bleeds that offset toward an approach (err > 0) or carries it past abeam into a
  // retreat (err < 0). At the preferred range no correction is left and the ship simply circles,
  // which is exactly where its flanks want the enemy.
  const err = Math.max(-1, Math.min(1, (d - R) / (R * ORBIT_TOLERANCE)));
  const hold = bias * battle.sense;
  const away = Math.PI * battle.sense;
  // Closing is oblique while the enemy is nearly in reach, so the guns keep bearing, and turns
  // into a straight charge as the range opens, because a beam arc is worth nothing at a range no
  // gun can shoot at. Holding the oblique angle all the way out let a long-ranged ship dictate the
  // range and leave a carronade ship trailing behind it, unable to fire a shot.
  const close = ORBIT_CLOSE + (1 - ORBIT_CLOSE) * err;
  const alpha = err > 0 ? hold * (1 - err * close) : hold + (away - hold) * -err * ORBIT_RETREAT;
  let desired = bearing + alpha;

  // Keep the fight on stage. Blends away from the tactical heading rather than from the current
  // one, so a ship being turned back still fights while it comes about. Compared squared, because
  // with the steering fixed this branch is now taken essentially never.
  const edge = battle.edge;
  if (ship.x * ship.x + ship.z * ship.z > battle.edgeSq) {
    const fromCentre = len(ship.x, ship.z);
    const inward = fatan2(-ship.x, ship.z);
    const pull = Math.min(1, (fromCentre - edge) / (battle.arenaRadius * 0.25));
    desired += wrapAngle(inward - desired) * pull;
  }

  // mass and sail are maintained by refreshSystems, which runs whenever either could change.
  const drive = ship.sail * ship.mass;
  const turnRate = BASE_TURN * drive;
  const diff = wrapAngle(desired - ship.heading);
  ship.heading = wrapAngle(ship.heading + Math.max(-turnRate * dt, Math.min(turnRate * dt, diff)));
  ship.cos = fcos(ship.heading);
  ship.sin = fsin(ship.heading);

  ship.windMult = windFactorFrom(ship.cos, ship.sin, battle.windCos, battle.windSin);
  const target = BASE_SPEED * ship.windMult * drive;
  ship.speed += (target - ship.speed) * Math.min(1, dt * 1.4);
  ship.x += ship.sin * ship.speed * dt;
  ship.z += -ship.cos * ship.speed * dt;
}

// How far a hull reaches in a given world direction, treating the deck as an ellipse: half its length
// along the ship, half its width across. The standard support radius, r = ab / sqrt((b cos)^2 +
// (a sin)^2), with the direction resolved onto the ship's own axes so no angle is ever formed.
function reachAlong(ship, nx, nz) {
  const fwd = nx * ship.sin - nz * ship.cos;
  const side = nx * ship.cos + nz * ship.sin;
  const bf = ship.semiWide * fwd;
  const as = ship.semiLong * side;
  return (ship.semiLong * ship.semiWide) / Math.sqrt(bf * bf + as * as);
}

// Hulls must not interpenetrate, and a single distance cannot express that: a ship of the line is ten
// cells long and five wide, so bow to bow two of them need twice the room they need beam to beam.
// With one number for both, measured, hulls overlapped by up to 5.2 world units -- a third of a
// frigate's length -- and a ship of the line spent a fifth of every duel inside its opponent.
//
// So the floor is the greater of what it always was and what the two hulls actually need along the
// line joining them. Only ever the greater: the fight settles with the enemy abeam, which is the
// cheapest orientation, and letting the new rule *lower* the floor there would move the range two
// ships settle at -- the one thing this is not allowed to touch.
export function separate(a, b, floor) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const d = len(dx, dz) || 0.001;
  const nx = dx / d;
  const nz = dz / d;
  const need = reachAlong(a, nx, nz) + reachAlong(b, nx, nz);
  const limit = need > floor ? need : floor;
  if (d >= limit) return 0;
  const push = (limit - d) / 2;
  a.x -= nx * push;
  a.z -= nz * push;
  b.x += nx * push;
  b.z += nz * push;
  return limit - d;
}
