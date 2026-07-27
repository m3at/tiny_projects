// Ship model: the persistent design a player edits between rounds, and the runtime state
// the battle simulation mutates.

import { PARTS } from '../data/parts.js';
import { HULLS, cellKey } from '../data/hulls.js';
import { CELL } from '../config.js';

export const HELM_KEY = '0,0';

export function createDesign() {
  return {
    parts: { [HELM_KEY]: { id: 'helm', hp: PARTS.helm.hp } },
  };
}

export function hullCellSet(hullIndex) {
  const set = new Set();
  for (const c of HULLS[hullIndex].cells) set.add(cellKey(c.dx, c.dz));
  return set;
}

// Parts that fell outside the previous hull can't happen (hulls only grow), but a part
// carried into a new hull still needs the cell to exist. Drop anything orphaned.
export function fitDesignToHull(design, hullIndex) {
  const allowed = hullCellSet(hullIndex);
  for (const key of Object.keys(design.parts)) {
    if (!allowed.has(key)) delete design.parts[key];
  }
  if (!design.parts[HELM_KEY]) design.parts[HELM_KEY] = { id: 'helm', hp: PARTS.helm.hp };
}

export function sideOfCell(dx) {
  if (dx < 0) return 'port';
  if (dx > 0) return 'starboard';
  return null;
}

// Can this part legally go in this cell?
export function placementError(design, hullIndex, dx, dz, partId) {
  const key = cellKey(dx, dz);
  if (!hullCellSet(hullIndex).has(key)) return 'Not part of the hull';
  if (design.parts[key]) return 'Cell already occupied';
  const part = PARTS[partId];
  if (part.gun && part.gun.arc === 'side' && sideOfCell(dx) === null) {
    return 'Broadside guns need a flank cell, not the spine';
  }
  return null;
}

export function designStats(design, hullIndex) {
  const hull = HULLS[hullIndex];
  let crewSupply = 0;
  let crewNeeded = 0;
  let magazines = 0;
  let masts = 0;
  const guns = [];
  const entries = Object.entries(design.parts);

  for (const [key, slot] of entries) {
    const part = PARTS[slot.id];
    if (part.crewSupply) crewSupply += part.crewSupply;
    if (part.crewCost) crewNeeded += part.crewCost;
    if (part.magazine) magazines++;
    if (slot.id === 'mast') masts++;
    if (part.gun) guns.push({ key, part });
  }

  // Guns are manned in a stable order so the "which gun goes silent" answer is
  // predictable rather than arbitrary.
  const sorted = [...entries].sort((a, b) => (a[0] < b[0] ? -1 : 1));
  let pool = crewSupply;
  const unmanned = [];
  for (const [key, slot] of sorted) {
    const part = PARTS[slot.id];
    if (!part.crewCost) continue;
    if (pool >= part.crewCost) pool -= part.crewCost;
    else unmanned.push(key);
  }

  const damaged = entries.filter(([, s]) => s.hp < PARTS[s.id].hp);

  return {
    hullName: hull.name,
    cellsUsed: entries.length,
    cellsTotal: hull.cells.length,
    crewSupply,
    crewNeeded,
    magazines,
    masts,
    gunCount: guns.length,
    unmanned,
    damaged,
  };
}

// Problems worth shouting about before the player locks in.
export function designWarnings(design, hullIndex) {
  const s = designStats(design, hullIndex);
  const out = [];
  if (s.gunCount === 0) out.push('No guns. Your ship cannot hurt anything.');
  else if (s.magazines === 0) out.push('No powder magazine. Not one gun will fire.');
  if (s.unmanned.length > 0) {
    out.push(`${s.unmanned.length} station(s) unmanned. Add crew quarters.`);
  }
  if (s.masts === 0) out.push('No masts. You will barely move.');
  const holes = s.cellsTotal - s.cellsUsed;
  if (holes > s.cellsTotal * 0.3) {
    out.push(`${holes} open holes. Shot passes through them to your spine.`);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Runtime state for one battle
// ---------------------------------------------------------------------------

export function makeBattleShip(design, hullIndex, index, startPos, startHeading) {
  const cells = [];
  const byKey = new Map();
  for (const [key, slot] of Object.entries(design.parts)) {
    const [dx, dz] = key.split(',').map(Number);
    const part = PARTS[slot.id];
    const cell = {
      key,
      dx,
      dz,
      id: slot.id,
      hp: slot.hp,
      maxHp: part.hp,
      alive: true,
      lx: dx * CELL,
      lz: dz * CELL,
    };
    cells.push(cell);
    byKey.set(key, cell);
  }

  const guns = [];
  for (const cell of cells) {
    const part = PARTS[cell.id];
    if (!part.gun) continue;
    const side = part.gun.arc === 'side' ? sideOfCell(cell.dx) : null;
    let arcCentre = 0;
    if (part.gun.arc === 'side') arcCentre = side === 'port' ? -Math.PI / 2 : Math.PI / 2;
    guns.push({
      cell,
      spec: part.gun,
      arcCentre,
      halfArc: (part.gun.halfArc * Math.PI) / 180,
      reloadLeft: part.gun.reload * 0.35, // stagger the opening volleys a little
      manned: false,
    });
  }
  guns.sort((a, b) => (a.cell.key < b.cell.key ? -1 : 1));

  const ship = {
    index,
    design,
    hullIndex,
    cells,
    byKey,
    guns,
    x: startPos.x,
    z: startPos.z,
    heading: startHeading,
    speed: 0,
    ammo: 'round',
    crewLost: 0,
    initialStructure: cells.reduce((a, c) => a + c.maxHp, 0),
  };
  refreshSystems(ship);
  return ship;
}

export function refreshSystems(ship) {
  let crewSupply = 0;
  let masts = 0;
  let magazines = 0;
  for (const cell of ship.cells) {
    if (!cell.alive) continue;
    const part = PARTS[cell.id];
    if (part.crewSupply) crewSupply += part.crewSupply;
    if (cell.id === 'mast') masts++;
    if (part.magazine) magazines++;
  }
  ship.crewSupply = crewSupply;
  ship.crew = Math.max(0, crewSupply - ship.crewLost);
  ship.masts = masts;
  ship.magazines = magazines;

  // Crew man the guns, and that is all crew do. An earlier version also let a thin crew
  // slow the ship down, which shifted win rates by 9% through a channel no player could
  // see or reason about.
  let pool = ship.crew;
  for (const gun of ship.guns) {
    const need = PARTS[gun.cell.id].crewCost || 0;
    gun.manned = gun.cell.alive && pool >= need;
    if (gun.manned) pool -= need;
  }
}

export function structureFraction(ship) {
  let left = 0;
  for (const cell of ship.cells) if (cell.alive) left += cell.hp;
  return ship.initialStructure === 0 ? 0 : left / ship.initialStructure;
}

// Anything no longer joined to the helm breaks away.
export function severDisconnected(ship) {
  const helm = ship.byKey.get(HELM_KEY);
  if (!helm || !helm.alive) return [];
  const seen = new Set([HELM_KEY]);
  const stack = [helm];
  while (stack.length) {
    const cur = stack.pop();
    const neighbours = [
      [cur.dx + 1, cur.dz],
      [cur.dx - 1, cur.dz],
      [cur.dx, cur.dz + 1],
      [cur.dx, cur.dz - 1],
    ];
    for (const [nx, nz] of neighbours) {
      const key = `${nx},${nz}`;
      if (seen.has(key)) continue;
      const cell = ship.byKey.get(key);
      if (!cell || !cell.alive) continue;
      seen.add(key);
      stack.push(cell);
    }
  }
  const severed = [];
  for (const cell of ship.cells) {
    if (cell.alive && !seen.has(cell.key)) {
      cell.alive = false;
      cell.hp = 0;
      severed.push(cell);
    }
  }
  return severed;
}

// Write the battle result back onto the persistent design.
export function commitDamage(ship) {
  for (const cell of ship.cells) {
    if (!cell.alive) delete ship.design.parts[cell.key];
    else ship.design.parts[cell.key].hp = cell.hp;
  }
  if (!ship.design.parts[HELM_KEY]) {
    // Helm was shot away. It comes back for free next round, otherwise the match is over
    // anyway; keeping it simple beats modelling a captured hulk.
    ship.design.parts[HELM_KEY] = { id: 'helm', hp: PARTS.helm.hp };
  }
}
