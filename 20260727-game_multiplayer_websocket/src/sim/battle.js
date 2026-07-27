// Deterministic battle simulation. No renderer dependency: it takes two designs, a seed
// and a wind direction, and advances in fixed ticks. Inputs (the ammunition toggle) are
// applied between ticks, so the same seed plus the same input stream always replays the
// same battle.

import { PARTS } from '../data/parts.js';
import { HULLS } from '../data/hulls.js';
import { makeRng } from './rng.js';
import {
  HELM_KEY,
  gridIndex,
  makeBattleShip,
  refreshSystems,
  severDisconnected,
  structureFraction,
  commitDamage,
} from './ship.js';
import {
  CELL,
  TICK,
  BATTLE_CAP,
  ARENA_RADIUS,
  BASE_SPEED,
  BASE_TURN,
  minSeparation,
  drawOrbitSense,
  ORBIT_CLOSE,
  ORBIT_TOLERANCE,
  ORBIT_RETREAT,
  HULL_DAMAGE,
  overtimeScale,
  RELOAD_STAGGER,
  RELOAD_JITTER,
  AMMO_SWITCH_RELOAD,
  START_OFFSET,
  PREFERRED_RANGE_FRACTION,
  GRAPE_EXTRA_SHOTS,
  GRAPE_SPREAD_SCALE,
  GRAPE_CREW_SCALE,
  MAGAZINE_BLAST_CREW,
  massFactor,
  sailFactor,
  windFactor,
} from '../config.js';

const TAU = Math.PI * 2;

// Math.hypot is dramatically slower than the arithmetic, and this runs a few thousand times a
// simulated second.
function len(x, z) {
  return Math.sqrt(x * x + z * z);
}

function wrapAngle(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

function toLocal(ship, wx, wz) {
  const dx = wx - ship.x;
  const dz = wz - ship.z;
  const c = Math.cos(ship.heading);
  const s = Math.sin(ship.heading);
  return { lx: dx * c + dz * s, lz: -dx * s + dz * c };
}

function toWorld(ship, lx, lz) {
  const c = Math.cos(ship.heading);
  const s = Math.sin(ship.heading);
  return { x: ship.x + lx * c - lz * s, z: ship.z + lx * s + lz * c };
}

// How a ship wants to fight, derived from the guns it actually carries. A long-gun ship
// charges bow-on and stays far; a broadside ship presents its flank and closes.
function fightingProfile(ship) {
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

export function createBattle({ designs, hullIndex, seed, windTo }) {
  const rng = makeRng(seed);
  const ships = [
    makeBattleShip(designs[0], hullIndex, 0, { x: -START_OFFSET.x, z: START_OFFSET.z }, 0),
    makeBattleShip(designs[1], hullIndex, 1, { x: START_OFFSET.x, z: -START_OFFSET.z }, Math.PI),
  ];
  for (const ship of ships) {
    ship.profile = fightingProfile(ship);
    // Start the battery out of step, so it rolls down the side instead of clapping.
    for (const gun of ship.guns) {
      gun.reloadLeft = rng.range(0, gun.spec.reload * RELOAD_STAGGER);
    }
  }

  const battle = {
    time: 0,
    sense: drawOrbitSense(rng),
    minSeparation: minSeparation(HULLS[hullIndex].length),
    ships,
    projectiles: [],
    effects: [],
    log: [],
    windTo,
    over: false,
    winner: null,
    reason: '',
    rng,
  };

  battle.setAmmo = (index, ammo) => {
    const ship = ships[index];
    if (battle.over || ship.ammo === ammo) return;
    ship.ammo = ammo;
    for (const gun of ship.guns) {
      gun.reloadLeft = Math.max(gun.reloadLeft, AMMO_SWITCH_RELOAD);
    }
    battle.effects.push({ type: 'ammo', ship: index, ammo });
  };

  battle.advance = (dt) => {
    let remaining = Math.min(dt, 0.25); // never simulate a huge catch-up jump
    while (remaining > 0 && !battle.over) {
      const step = Math.min(TICK, remaining);
      tick(battle, step);
      remaining -= step;
    }
  };

  battle.finish = () => {
    for (const ship of ships) commitDamage(ship);
  };

  return battle;
}

function tick(battle, dt) {
  battle.time += dt;
  const [a, b] = battle.ships;

  steer(battle, a, b, dt);
  steer(battle, b, a, dt);
  separate(a, b, battle.minSeparation);

  fireGuns(battle, a, b, dt);
  fireGuns(battle, b, a, dt);
  stepProjectiles(battle, dt);

  checkEnd(battle);
}

function steer(battle, ship, enemy, dt) {
  const dx = enemy.x - ship.x;
  const dz = enemy.z - ship.z;
  const d = len(dx, dz);
  const bearing = Math.atan2(dx, -dz);

  const R = ship.profile.range;
  const bias = (ship.profile.arcBias * Math.PI) / 180;

  // One continuous controller, not a ladder of range bands. `hold` is the heading offset
  // that keeps the guns bearing at the preferred range: a quarter turn off the bearing for
  // a broadside ship, straight at the enemy for a bow chaser. `err` says whether the range
  // wants closing or opening, and bleeds the offset toward an oblique approach (err > 0) or
  // carries it past abeam into a retreat (err < 0). At the preferred range there is no
  // correction left and the ship simply circles, which is exactly where its flanks want the
  // enemy.
  //
  // Both ships take the same sense of rotation, which is what makes it a circling engagement.
  // Opposite senses were the old bug: each ship kept the other abeam by sailing a parallel
  // course, so the pair held its range perfectly and marched off the map together, firing
  // nothing until the arena hauled them back.
  const err = Math.max(-1, Math.min(1, (d - R) / (R * ORBIT_TOLERANCE)));
  const hold = bias * battle.sense;
  const away = Math.PI * battle.sense;
  // Closing is oblique while the enemy is nearly in reach, so the guns keep bearing, and turns
  // into a straight charge as the range opens, because a beam arc is worth nothing at a range
  // no gun can shoot at. Holding the oblique angle all the way out let a long-ranged ship
  // dictate the range and leave a carronade ship trailing behind it, unable to fire a shot.
  const close = ORBIT_CLOSE + (1 - ORBIT_CLOSE) * err;
  const alpha = err > 0 ? hold * (1 - err * close) : hold + (away - hold) * -err * ORBIT_RETREAT;
  let desired = bearing + alpha;

  // Keep the fight on stage. Blends away from the tactical heading rather than from the
  // current one, so a ship being turned back still fights while it comes about.
  const fromCentre = len(ship.x, ship.z);
  if (fromCentre > ARENA_RADIUS * 0.8) {
    const inward = Math.atan2(-ship.x, ship.z);
    const pull = Math.min(1, (fromCentre - ARENA_RADIUS * 0.8) / (ARENA_RADIUS * 0.25));
    desired += wrapAngle(inward - desired) * pull;
  }

  const mass = massFactor(ship.aliveCells || 1);
  const sail = sailFactor(ship.masts, ship.sailWanted);
  const turnRate = BASE_TURN * sail * mass;
  const diff = wrapAngle(desired - ship.heading);
  ship.heading = wrapAngle(ship.heading + Math.max(-turnRate * dt, Math.min(turnRate * dt, diff)));

  ship.windMult = windFactor(ship.heading, battle.windTo);
  const target = BASE_SPEED * ship.windMult * sail * mass;
  ship.speed += (target - ship.speed) * Math.min(1, dt * 1.4);
  ship.x += Math.sin(ship.heading) * ship.speed * dt;
  ship.z += -Math.cos(ship.heading) * ship.speed * dt;
}

function separate(a, b, floor) {
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

function fireGuns(battle, ship, enemy, dt) {
  const canFire = ship.magazines > 0;
  const cos = Math.cos(ship.heading);
  const sin = Math.sin(ship.heading);
  for (const gun of ship.guns) {
    if (gun.reloadLeft > 0) gun.reloadLeft -= dt;
    if (!canFire || !gun.cell.alive || !gun.manned || gun.reloadLeft > 0) continue;

    const mx = ship.x + gun.cell.lx * cos - gun.cell.lz * sin;
    const mz = ship.z + gun.cell.lx * sin + gun.cell.lz * cos;
    const dx = enemy.x - mx;
    const dz = enemy.z - mz;
    const dist = len(dx, dz);
    if (dist > gun.spec.range) continue;

    const bearing = Math.atan2(dx, -dz);
    if (!bears(gun, ship.heading, bearing)) continue;

    // Lead the target, then scatter.
    const flight = dist / gun.spec.speed;
    const aimX = enemy.x + Math.sin(enemy.heading) * enemy.speed * flight;
    const aimZ = enemy.z - Math.cos(enemy.heading) * enemy.speed * flight;
    const aimBearing = Math.atan2(aimX - mx, -(aimZ - mz));

    const shot = ship.ammo === 'grape' ? gun.spec.grape : gun.spec.round;
    const count = ship.ammo === 'grape' ? gun.spec.shots + GRAPE_EXTRA_SHOTS : gun.spec.shots;
    for (let i = 0; i < count; i++) {
      const spreadRad = (gun.spec.spread * Math.PI) / 180;
      const jitter =
        battle.rng.range(-spreadRad, spreadRad) * (ship.ammo === 'grape' ? GRAPE_SPREAD_SCALE : 1);
      const ang = aimBearing + jitter;
      const speed = gun.spec.speed * battle.rng.range(0.94, 1.06);
      battle.projectiles.push({
        x: mx,
        z: mz,
        vx: Math.sin(ang) * speed,
        vz: -Math.cos(ang) * speed,
        owner: ship.index,
        target: enemy.index,
        damage: shot.damage,
        crew: (shot.crew || 0) * GRAPE_CREW_SCALE,
        pierce: !!gun.spec.pierce,
        kind: ship.ammo,
        ttl: (gun.spec.range / gun.spec.speed) * 1.35,
      });
    }
    battle.effects.push({
      type: 'muzzle',
      x: mx,
      z: mz,
      heading: aimBearing,
      ship: ship.index,
      big: gun.spec.shots > 1,
    });
    gun.reloadLeft = gun.spec.reload * battle.rng.range(1 - RELOAD_JITTER, 1 + RELOAD_JITTER);
  }
}

// A broadside gun deck ran the full width of the ship, so it answers to either beam: two
// windows at +/-90 with an eighth of a turn blind fore and aft. Which flank the cell sits on
// still decides where the damage lands, just not where the gun can shoot.
function bears(gun, heading, bearing) {
  for (const centre of gun.arcs) {
    if (Math.abs(wrapAngle(bearing - (heading + centre))) <= gun.halfArc) return true;
  }
  return false;
}

// Compacts in place rather than rebuilding the array, which at a few hundred shot a second was
// a fresh array every tick.
function stepProjectiles(battle, dt) {
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
    const target = battle.ships[p.target];
    const hitCell = cellAt(target, p.x, p.z);
    if (hitCell) {
      resolveHit(battle, target, hitCell, p);
      continue;
    }
    list[keep++] = p;
  }
  list.length = keep;
}

function cellAt(ship, wx, wz) {
  const dx = wx - ship.x;
  const dz = wz - ship.z;
  const c = Math.cos(ship.heading);
  const s = Math.sin(ship.heading);
  const cx = Math.round((dx * c + dz * s) / CELL);
  const cz = Math.round((-dx * s + dz * c) / CELL);
  const cell = ship.grid[gridIndex(cx, cz)];
  // A destroyed cell is a hole: the ball keeps going and can reach the spine.
  return cell !== undefined && cell.alive ? cell : null;
}

function resolveHit(battle, ship, cell, p) {
  battle.effects.push({
    type: 'impact',
    x: p.x,
    z: p.z,
    kind: p.kind,
    ship: ship.index,
  });
  if (p.crew > 0 && ship.crew > 0) {
    ship.crewLost += p.crew;
    battle.effects.push({ type: 'crew', x: p.x, z: p.z, ship: ship.index });
  }
  damageCell(battle, ship, cell, p.damage, p.pierce);
  refreshSystems(ship);
}

function damageCell(battle, ship, cell, amount, pierce, chain) {
  if (!cell.alive) return;
  const soak = pierce ? Math.floor(cell.soak / 2) : cell.soak;
  // Soak first, then the hull's pace factor. The other order let a big hull's factor drop a
  // ball below the soak line, which made heavy timbers immune rather than tough.
  const scale = (HULL_DAMAGE[ship.hullIndex] ?? 1) * overtimeScale(battle.time);
  cell.hp -= Math.max(1, amount - soak) * scale;
  if (cell.hp > 0) return;

  cell.hp = 0;
  cell.alive = false;
  ship.aliveCells--;
  const pos = toWorld(ship, cell.lx, cell.lz);
  battle.effects.push({ type: 'destroy', x: pos.x, z: pos.z, part: cell.id, ship: ship.index });

  if (cell.id === 'mast') {
    logOnce(battle, ship, 'mast', `${shipName(ship)} loses a mast`);
  }
  if (cell.magazine) {
    detonate(battle, ship, cell, chain || new Set());
  }
  const severed = severDisconnected(ship);
  for (const s of severed) {
    const sp = toWorld(ship, s.lx, s.lz);
    battle.effects.push({ type: 'sever', x: sp.x, z: sp.z, part: s.id, ship: ship.index });
  }
  if (severed.length >= 2) {
    battle.log.push({ t: battle.time, text: `${severed.length} sections break away from ${shipName(ship)}` });
  }
}

function detonate(battle, ship, cell, chain) {
  if (chain.has(cell.key)) return;
  chain.add(cell.key);
  const pos = toWorld(ship, cell.lx, cell.lz);
  battle.effects.push({ type: 'detonate', x: pos.x, z: pos.z, ship: ship.index });
  battle.log.push({ t: battle.time, text: `Powder magazine detonates aboard ${shipName(ship)}` });
  const spec = PARTS.magazine.detonate;
  for (let ox = -spec.radius; ox <= spec.radius; ox++) {
    for (let oz = -spec.radius; oz <= spec.radius; oz++) {
      if (ox === 0 && oz === 0) continue;
      const n = ship.grid[gridIndex(cell.dx + ox, cell.dz + oz)];
      if (n !== undefined && n.alive) damageCell(battle, ship, n, spec.damage, true, chain);
    }
  }
  ship.crewLost += MAGAZINE_BLAST_CREW;
}

function logOnce(battle, ship, tag, text) {
  const key = `${ship.index}:${tag}`;
  battle._logged = battle._logged || new Set();
  if (battle._logged.has(key)) return;
  battle._logged.add(key);
  battle.log.push({ t: battle.time, text });
}

function shipName(ship) {
  return ship.index === 0 ? 'Player 1' : 'Player 2';
}

function canEverFire(ship) {
  if (ship.magazines === 0) return false;
  return ship.guns.some((g) => g.cell.alive && g.manned);
}

function checkEnd(battle) {
  const [a, b] = battle.ships;
  const dead = (s) => !s.byKey.get(HELM_KEY)?.alive || s.aliveCells === 0;

  if (dead(a) || dead(b)) {
    const aDead = dead(a);
    const bDead = dead(b);
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

  const stalemate = !canEverFire(a) && !canEverFire(b);
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

export { structureFraction };
