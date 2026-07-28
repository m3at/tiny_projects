// What a ball does when it arrives: structure, crew, magazines, and sections breaking away.

import { PARTS } from '../data/parts.js';
import { worldX, worldZ } from './geometry.js';
import { gridIndex, refreshSystems, severDisconnected, shipName } from './ship.js';
import { HULL_DAMAGE, MAGAZINE_BLAST_CREW, overtimeScale } from '../config.js';

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
  const scale = (HULL_DAMAGE[ship.hullIndex] ?? 1) * overtimeScale(battle.time);
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
