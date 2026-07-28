// What a ball does when it arrives: structure, crew, magazines, and sections breaking away.

import { PARTS } from '../data/parts.js';
import { worldX, worldZ } from './geometry.js';
import { gridIndex, refreshSystems, severDisconnected, shipName } from './ship.js';
import {
  HULL_DAMAGE,
  MAGAZINE_BLAST_CREW,
  BASE_SPEED,
  COLLISION_DAMAGE,
  COLLISION_INTERVAL,
  overtimeScale,
} from '../config.js';

export function resolveHit(battle, ship, cell, p) {
  battle.effects.push({ type: 'impact', x: p.x, z: p.z, kind: p.kind, ship: ship.index });
  let changed = false;
  if (p.crew > 0 && ship.crew > 0) {
    ship.crewLost += p.crew;
    battle.effects.push({ type: 'crew', x: p.x, z: p.z, ship: ship.index });
    changed = true;
  }
  // Crew, masts, powder and which guns are manned only move when a cell dies or hands are lost. A
  // ball that merely dents a timber changes none of them, and most balls do exactly that.
  if (damageCell(battle, ship, cell, p.damage, p.pierce)) changed = true;
  if (changed) refreshSystems(ship);
}

// Returns whether the cell (or anything it took with it) died.
export function damageCell(battle, ship, cell, amount, pierce, chain) {
  if (!cell.alive) return false;
  const soak = pierce ? Math.floor(cell.soak / 2) : cell.soak;
  // Soak first, then the pace factors. The other order let a big hull's factor drop a ball below
  // the soak line, which made heavy timbers immune rather than tough.
  // battle.damageScale is 1 in a duel and pulls incoming fire back in a melee, where two or three
  // batteries are pointed at you instead of one.
  const scale = (HULL_DAMAGE[ship.hullIndex] ?? 1) * battle.damageScale * overtimeScale(battle.time);
  cell.hp -= Math.max(1, amount - soak) * scale;
  if (cell.hp > 0) return false;

  cell.hp = 0;
  cell.alive = false;
  ship.aliveCells--;
  battle.effects.push({
    type: 'destroy',
    x: worldX(ship, cell.lx, cell.lz),
    z: worldZ(ship, cell.lx, cell.lz),
    part: cell.id,
    ship: ship.index,
  });

  if (cell.id === 'mast') battle.noteOnce(`mast${ship.index}`, `${shipName(ship)} loses a mast`);
  if (cell.magazine) detonate(battle, ship, cell, chain || new Set());

  const severed = severDisconnected(ship, cell);
  for (const s of severed) {
    battle.effects.push({
      type: 'sever',
      x: worldX(ship, s.lx, s.lz),
      z: worldZ(ship, s.lx, s.lz),
      part: s.id,
      ship: ship.index,
    });
  }
  if (severed.length >= 2) {
    battle.note(`${severed.length} sections break away from ${shipName(ship)}`);
  }
  return true;
}

// Two hulls in contact. steering.js separate() has already pushed them apart and reports how deeply
// they were inside each other; this is what it costs them.
//
// One crunch per pair per COLLISION_INTERVAL, held on the battle so it is part of the deterministic
// state and replays identically. The cell that takes it is the one nearest the point of contact, which
// is what makes ramming a bow into somebody's magazine a thing a player can attempt on purpose.
export function grind(battle, a, b) {
  const pair = a.index * battle.shipCount + b.index;
  if (battle.time < battle.contactAt[pair]) return;
  battle.contactAt[pair] = battle.time + COLLISION_INTERVAL;

  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const d = Math.sqrt(dx * dx + dz * dz) || 1e-9;
  const nx = dx / d;
  const nz = dz / d;

  // How fast the two hulls are moving relative to each other. A scrape at speed tears timber; two
  // ships drifting into each other barely mark the paint.
  const rvx = b.sin * b.speed - a.sin * a.speed;
  const rvz = -b.cos * b.speed + a.cos * a.speed;
  const rel = Math.sqrt(rvx * rvx + rvz * rvz);
  const amount = COLLISION_DAMAGE * Math.min(1, rel / BASE_SPEED);
  if (amount < 0.5) return; // a nudge, not a collision

  // Halfway between the two centres is inside both hulls, since they were overlapping.
  const cx = a.x + nx * (d / 2);
  const cz = a.z + nz * (d / 2);
  crush(battle, a, cx, cz, amount);
  crush(battle, b, cx, cz, amount);
  // Once per pair per battle. Two hulls in contact stay in contact for a second or two, and a line
  // for every crunch would push everything else out of the log.
  battle.noteOnce(`hit${pair}`, `${shipName(a)} and ${shipName(b)} come together`);
}

// The live cell closest to a world point, and what a collision does to it. O(cells), run at most
// twice a second per pair, so it is nowhere near the hot path.
function crush(battle, ship, wx, wz, amount) {
  let best = null;
  let bestSq = Infinity;
  for (const cell of ship.cells) {
    if (!cell.alive) continue;
    const px = worldX(ship, cell.lx, cell.lz) - wx;
    const pz = worldZ(ship, cell.lx, cell.lz) - wz;
    const sq = px * px + pz * pz;
    if (sq < bestSq) {
      bestSq = sq;
      best = cell;
    }
  }
  if (best === null) return;
  battle.effects.push({ type: 'impact', x: wx, z: wz, kind: 'round', ship: ship.index });
  // Pierce: a hull grinding along your side does not care how heavy your scantlings are.
  if (damageCell(battle, ship, best, amount, true)) refreshSystems(ship);
}

// Balance-neutral by measurement, kept because it is the best story the game tells. `chain` stops
// a ring of magazines detonating each other for ever.
function detonate(battle, ship, cell, chain) {
  if (chain.has(cell.key)) return;
  chain.add(cell.key);
  battle.effects.push({
    type: 'detonate',
    x: worldX(ship, cell.lx, cell.lz),
    z: worldZ(ship, cell.lx, cell.lz),
    ship: ship.index,
  });
  battle.note(`Powder magazine detonates aboard ${shipName(ship)}`);
  const spec = PARTS.magazine.detonate;
  for (let ox = -spec.radius; ox <= spec.radius; ox++) {
    for (let oz = -spec.radius; oz <= spec.radius; oz++) {
      if (ox === 0 && oz === 0) continue;
      const n = ship.grid[gridIndex(cell.dx + ox, cell.dz + oz)];
      if (n !== null && n.alive) damageCell(battle, ship, n, spec.damage, true, chain);
    }
  }
  ship.crewLost += MAGAZINE_BLAST_CREW;
}
