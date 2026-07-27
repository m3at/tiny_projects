// Deterministic battle simulation. No renderer dependency: it takes two designs, a seed
// and a wind direction, and advances in fixed ticks. Inputs (the ammunition toggle) are
// applied between ticks, so the same seed plus the same input stream always replays the
// same battle.

import { PARTS } from '../data/parts.js';
import { makeRng } from './rng.js';
import {
  HELM_KEY,
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
  MIN_SEPARATION,
  AMMO_SWITCH_RELOAD,
  START_OFFSET,
  PREFERRED_RANGE_FRACTION,
  GRAPE_EXTRA_SHOTS,
  GRAPE_SPREAD_SCALE,
  MAGAZINE_BLAST_CREW,
  massFactor,
  sailFactor,
  windFactor,
} from '../config.js';

const TAU = Math.PI * 2;

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
  for (const ship of ships) ship.profile = fightingProfile(ship);

  const battle = {
    time: 0,
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
  separate(a, b);

  fireGuns(battle, a, b, dt);
  fireGuns(battle, b, a, dt);
  stepProjectiles(battle, dt);

  checkEnd(battle);
}

function steer(battle, ship, enemy, dt) {
  const dx = enemy.x - ship.x;
  const dz = enemy.z - ship.z;
  const d = Math.hypot(dx, dz);
  const bearing = Math.atan2(dx, -dz);

  const R = ship.profile.range;
  const bias = (ship.profile.arcBias * Math.PI) / 180;
  let alpha;
  if (d > R * 1.35) {
    alpha = bias * 0.18;
  } else if (d > R) {
    const t = (R * 1.35 - d) / (R * 0.35);
    alpha = bias * (0.18 + 0.82 * t);
  } else {
    // Too close for comfort, so open the range. A broadside ship only has to sheer off a
    // little to keep its guns bearing; a bow-gun ship has to actually run, and cannot
    // shoot while it does. That trade is what keeps long guns honest.
    const t = Math.min(1, (R - d) / (R * 0.45));
    const flee = bias > 0.8 ? bias + 0.55 : 2.5;
    alpha = bias + (flee - bias) * t;
  }

  const orbitSign = ship.index === 0 ? 1 : -1;
  let desired = bearing + alpha * orbitSign;

  // Keep the fight on stage.
  const fromCentre = Math.hypot(ship.x, ship.z);
  if (fromCentre > ARENA_RADIUS * 0.8) {
    const inward = Math.atan2(-ship.x, ship.z);
    const pull = Math.min(1, (fromCentre - ARENA_RADIUS * 0.8) / (ARENA_RADIUS * 0.25));
    desired = ship.heading + wrapAngle(inward - ship.heading) * pull;
  }

  const mass = massFactor(ship.cells.filter((c) => c.alive).length || 1);
  const sail = sailFactor(ship.masts, ship.cells.length);
  const turnRate = BASE_TURN * sail * mass;
  const diff = wrapAngle(desired - ship.heading);
  ship.heading = wrapAngle(ship.heading + Math.max(-turnRate * dt, Math.min(turnRate * dt, diff)));

  ship.windMult = windFactor(ship.heading, battle.windTo);
  const target = BASE_SPEED * ship.windMult * sail * mass;
  ship.speed += (target - ship.speed) * Math.min(1, dt * 1.4);
  ship.x += Math.sin(ship.heading) * ship.speed * dt;
  ship.z += -Math.cos(ship.heading) * ship.speed * dt;
}

function separate(a, b) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const d = Math.hypot(dx, dz) || 0.001;
  if (d >= MIN_SEPARATION) return;
  const push = (MIN_SEPARATION - d) / 2;
  const nx = dx / d;
  const nz = dz / d;
  a.x -= nx * push;
  a.z -= nz * push;
  b.x += nx * push;
  b.z += nz * push;
}

function fireGuns(battle, ship, enemy, dt) {
  const canFire = ship.magazines > 0;
  for (const gun of ship.guns) {
    if (gun.reloadLeft > 0) gun.reloadLeft -= dt;
    if (!canFire || !gun.cell.alive || !gun.manned || gun.reloadLeft > 0) continue;

    const muzzle = toWorld(ship, gun.cell.lx, gun.cell.lz);
    const dx = enemy.x - muzzle.x;
    const dz = enemy.z - muzzle.z;
    const dist = Math.hypot(dx, dz);
    if (dist > gun.spec.range) continue;

    const bearing = Math.atan2(dx, -dz);
    if (Math.abs(wrapAngle(bearing - (ship.heading + gun.arcCentre))) > gun.halfArc) continue;

    // Lead the target, then scatter.
    const flight = dist / gun.spec.speed;
    const aimX = enemy.x + Math.sin(enemy.heading) * enemy.speed * flight;
    const aimZ = enemy.z - Math.cos(enemy.heading) * enemy.speed * flight;
    const aimBearing = Math.atan2(aimX - muzzle.x, -(aimZ - muzzle.z));

    const shot = ship.ammo === 'grape' ? gun.spec.grape : gun.spec.round;
    const count = ship.ammo === 'grape' ? gun.spec.shots + GRAPE_EXTRA_SHOTS : gun.spec.shots;
    for (let i = 0; i < count; i++) {
      const spreadRad = (gun.spec.spread * Math.PI) / 180;
      const jitter =
        battle.rng.range(-spreadRad, spreadRad) * (ship.ammo === 'grape' ? GRAPE_SPREAD_SCALE : 1);
      const ang = aimBearing + jitter;
      const speed = gun.spec.speed * battle.rng.range(0.94, 1.06);
      battle.projectiles.push({
        x: muzzle.x,
        z: muzzle.z,
        vx: Math.sin(ang) * speed,
        vz: -Math.cos(ang) * speed,
        owner: ship.index,
        target: enemy.index,
        damage: shot.damage,
        crew: shot.crew || 0,
        pierce: !!gun.spec.pierce,
        kind: ship.ammo,
        ttl: (gun.spec.range / gun.spec.speed) * 1.35,
      });
    }
    battle.effects.push({
      type: 'muzzle',
      x: muzzle.x,
      z: muzzle.z,
      heading: aimBearing,
      ship: ship.index,
      big: gun.spec.shots > 1,
    });
    gun.reloadLeft = gun.spec.reload * battle.rng.range(0.92, 1.08);
  }
}

function stepProjectiles(battle, dt) {
  const alive = [];
  for (const p of battle.projectiles) {
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
    alive.push(p);
  }
  battle.projectiles = alive;
}

function cellAt(ship, wx, wz) {
  const { lx, lz } = toLocal(ship, wx, wz);
  const dx = Math.round(lx / CELL);
  const dz = Math.round(lz / CELL);
  const cell = ship.byKey.get(`${dx},${dz}`);
  // A destroyed cell is a hole: the ball keeps going and can reach the spine.
  return cell && cell.alive ? cell : null;
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
  const part = PARTS[cell.id];
  const soak = pierce ? Math.floor((part.soak || 0) / 2) : part.soak || 0;
  cell.hp -= Math.max(1, amount - soak);
  if (cell.hp > 0) return;

  cell.hp = 0;
  cell.alive = false;
  const pos = toWorld(ship, cell.lx, cell.lz);
  battle.effects.push({ type: 'destroy', x: pos.x, z: pos.z, part: cell.id, ship: ship.index });

  if (cell.id === 'mast') {
    logOnce(battle, ship, 'mast', `${shipName(ship)} loses a mast`);
  }
  if (part.magazine) {
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
      const n = ship.byKey.get(`${cell.dx + ox},${cell.dz + oz}`);
      if (n && n.alive) damageCell(battle, ship, n, spec.damage, true, chain);
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
  const dead = (s) => !s.byKey.get(HELM_KEY)?.alive || s.cells.every((c) => !c.alive);

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
    if (Math.abs(fa - fb) < 0.02) {
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
