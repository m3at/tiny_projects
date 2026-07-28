// How the two ships sail. Nobody steers: each ship derives what it wants from the guns it
// carries and holds station on the other.
//
// This file is where the game's dullest failure lived, so the reasoning is worth keeping. The
// ships used to circle in opposite senses, which is a parallel course: each kept the other abeam
// by sailing alongside it, so the pair held its range perfectly and marched off the map together,
// firing nothing, until the arena hauled them back. That read as "one volley, a long silence, then
// the action resumes in the corner". Both ships now take the same sense, so they orbit their
// common midpoint.

import { len, wrapAngle } from './geometry.js';
import {
  ARENA_RADIUS,
  BASE_SPEED,
  BASE_TURN,
  ORBIT_CLOSE,
  ORBIT_TOLERANCE,
  ORBIT_RETREAT,
  PREFERRED_RANGE_FRACTION,
  windFactorFrom,
} from '../config.js';

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
  const bearing = Math.atan2(dx, -dz);

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
  const edge = ARENA_RADIUS * 0.8;
  if (ship.x * ship.x + ship.z * ship.z > edge * edge) {
    const fromCentre = len(ship.x, ship.z);
    const inward = Math.atan2(-ship.x, ship.z);
    const pull = Math.min(1, (fromCentre - edge) / (ARENA_RADIUS * 0.25));
    desired += wrapAngle(inward - desired) * pull;
  }

  // mass and sail are maintained by refreshSystems, which runs whenever either could change.
  const drive = ship.sail * ship.mass;
  const turnRate = BASE_TURN * drive;
  const diff = wrapAngle(desired - ship.heading);
  ship.heading = wrapAngle(ship.heading + Math.max(-turnRate * dt, Math.min(turnRate * dt, diff)));
  ship.cos = Math.cos(ship.heading);
  ship.sin = Math.sin(ship.heading);

  ship.windMult = windFactorFrom(ship.cos, ship.sin, battle.windCos, battle.windSin);
  const target = BASE_SPEED * ship.windMult * drive;
  ship.speed += (target - ship.speed) * Math.min(1, dt * 1.4);
  ship.x += ship.sin * ship.speed * dt;
  ship.z += -ship.cos * ship.speed * dt;
}

// Hulls must not interpenetrate. The floor scales with hull length in config.minSeparation; the
// preferred range holds the ships much further apart than this, so it only ever matters visually.
export function separate(a, b, floor) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const d = len(dx, dz) || 0.001;
  if (d >= floor) return;
  const push = (floor - d) / 2;
  const nx = dx / d;
  const nz = dz / d;
  a.x -= nx * push;
  a.z -= nz * push;
  b.x += nx * push;
  b.z += nz * push;
}
