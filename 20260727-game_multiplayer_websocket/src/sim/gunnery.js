// Firing, shot in flight, and where a ball lands.
//
// Two things in here are shaped by the profiler rather than by the design, and both are exact
// rather than approximate:
//
//   The arc test is a dot product, not an angle comparison. Asking whether a bearing falls inside
//   an arc used an atan2 and a wrap per gun per tick; comparing the cosine of the angle against
//   the cosine of the arc's half-width is the same question with no trigonometry at all.
//
//   The hit test rejects on a squared distance before transforming anything into ship space. Most
//   shot in the air is nowhere near its target, and that test was a quarter of all simulation time.

import { CELL, GRAPE_EXTRA_SHOTS, GRAPE_SPREAD_SCALE, GRAPE_CREW_SCALE, RELOAD_JITTER } from '../config.js';
import { gridIndex } from './ship.js';
import { resolveHit } from './damage.js';

// Does this gun bear on a target in direction (dx, dz), `distSq` away squared? The arc's centre
// direction is the ship's heading rotated by the gun's mounting, built from the heading's cached
// sine and cosine and the gun's precomputed rotation.
//
// The test is cos(angle) >= cos(halfArc), in the form dot >= cos(halfArc) * dist and then squared so
// no square root is needed. Squaring is only valid with both sides positive, hence the dot <= 0
// rejection -- correct because every directional arc here is under a quarter turn.
export function bears(gun, ship, dx, dz, distSq) {
  if (gun.allRound) return true;
  const ax = ship.sin * gun.arcCos + ship.cos * gun.arcSin;
  const az = -ship.cos * gun.arcCos + ship.sin * gun.arcSin;
  const dot = dx * ax + dz * az;
  if (dot <= 0) return false;
  return dot * dot >= gun.cosHalfArcSq * distSq;
}

export function fireGuns(battle, ship, enemy) {
  if (ship.magazines === 0) return;
  const now = battle.time;
  const guns = ship.guns;
  for (let g = 0; g < guns.length; g++) {
    const gun = guns[g];
    // `manned` is false whenever the cell is dead, so it covers both.
    if (now < gun.readyAt || !gun.manned) continue;

    // Ranges and arcs are measured from the muzzle, not the ship's centre: on a ship of the line
    // that is a difference of several cells.
    const mx = ship.x + gun.cell.lx * ship.cos - gun.cell.lz * ship.sin;
    const mz = ship.z + gun.cell.lx * ship.sin + gun.cell.lz * ship.cos;
    const dx = enemy.x - mx;
    const dz = enemy.z - mz;
    const distSq = dx * dx + dz * dz;
    if (distSq > gun.rangeSq) continue;
    if (!bears(gun, ship, dx, dz, distSq)) continue;

    // Lead the target, then scatter. Only guns that actually fire pay for a square root.
    const spec = gun.spec;
    const flight = Math.sqrt(distSq) / spec.speed;
    const aimX = enemy.x + enemy.sin * enemy.speed * flight;
    const aimZ = enemy.z - enemy.cos * enemy.speed * flight;
    const aimBearing = Math.atan2(aimX - mx, -(aimZ - mz));

    const grape = ship.ammo === 'grape';
    const shot = grape ? spec.grape : spec.round;
    const count = grape ? spec.shots + GRAPE_EXTRA_SHOTS : spec.shots;
    // Grape's wider pattern folds into the spread up front rather than scaling each pellet.
    const spreadRad = ((spec.spread * Math.PI) / 180) * (grape ? GRAPE_SPREAD_SCALE : 1);
    const ttl = (spec.range / spec.speed) * 1.35;
    const crew = (shot.crew || 0) * GRAPE_CREW_SCALE;
    for (let i = 0; i < count; i++) {
      const ang = aimBearing + battle.rng.range(-spreadRad, spreadRad);
      const speed = spec.speed * battle.rng.range(0.94, 1.06);
      battle.projectiles.push({
        x: mx,
        z: mz,
        vx: Math.sin(ang) * speed,
        vz: -Math.cos(ang) * speed,
        owner: ship.index,
        target: enemy.index,
        damage: shot.damage,
        crew,
        pierce: !!spec.pierce,
        kind: ship.ammo,
        ttl,
      });
    }
    battle.effects.push({
      type: 'muzzle',
      x: mx,
      z: mz,
      heading: aimBearing,
      ship: ship.index,
      big: spec.shots > 1,
    });
    gun.readyAt = now + spec.reload * battle.rng.range(1 - RELOAD_JITTER, 1 + RELOAD_JITTER);
  }
}

// Compacts in place rather than rebuilding the array, which at a few hundred shot a second was a
// fresh array every tick.
export function stepProjectiles(battle, dt) {
  const list = battle.projectiles;
  let keep = 0;
  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    p.ttl -= dt;
    p.x += p.vx * dt;
    p.z += p.vz * dt;
    if (p.ttl <= 0) {
      battle.effects.push({ type: 'splash', x: p.x, z: p.z });
      continue;
    }
    // Broad phase inlined: most shot in the air is nowhere near its target, and this loop is the
    // busiest in the simulation. Only balls close enough to possibly land pay for the transform.
    const target = battle.ships[p.target];
    const ex = p.x - target.x;
    const ez = p.z - target.z;
    const gap = ex * ex + ez * ez;
    if (gap <= target.hitRadiusSq) {
      const hitCell = cellAt(target, ex, ez);
      if (hitCell !== null) {
        resolveHit(battle, target, hitCell, p);
        continue;
      }
    } else if (p.vx * ex + p.vz * ez >= 0) {
      // Outside the ship and travelling away from it: this ball has missed, and it cannot come
      // back, because every shot flies faster than any ship sails. It used to keep going until its
      // time to live ran out, which was a third of all the projectile-ticks the simulation spent --
      // and it splashed somewhere off in the distance instead of alongside, where the miss happened.
      battle.effects.push({ type: 'splash', x: p.x, z: p.z });
      continue;
    }
    list[keep++] = p;
  }
  list.length = keep;
}

// (dx, dz) is already relative to the ship's centre and already inside its hit radius.
function cellAt(ship, dx, dz) {
  const cx = Math.round((dx * ship.cos + dz * ship.sin) / CELL);
  const cz = Math.round((-dx * ship.sin + dz * ship.cos) / CELL);
  const cell = ship.grid[gridIndex(cx, cz)];
  // A destroyed cell is a hole: the ball keeps going and can reach the spine. Measurably the most
  // important rule in the game -- block it and 87% of battles run to the bell.
  return cell !== null && cell.alive ? cell : null;
}
