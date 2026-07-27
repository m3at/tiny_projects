// Ship model: the persistent design a player edits between rounds, and the runtime state
// the battle simulation mutates.

import { PARTS } from '../data/parts.js';
import { HULLS, cellKey, isBowCell } from '../data/hulls.js';
import { CELL, mastsWanted } from '../config.js';

export const HELM_KEY = '0,0';

// A flat integer index for a cell offset, for the one lookup that happens per projectile per
// tick. Building a "dx,dz" string there was the simulation's single hottest allocation.
const GRID_SPAN = 64;
export function gridIndex(dx, dz) {
  return (dx + 16) * GRID_SPAN + (dz + 32);
}

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
  if (part.gun && part.gun.arc === 'bow' && !isBowCell(hullIndex, dz)) {
    return 'A bow chaser has to be worked from the bow';
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
    mastsWanted: mastsWanted(hull.cells.length),
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
  else if (s.masts > s.mastsWanted) {
    out.push(`${s.masts - s.mastsWanted} mast(s) more than this hull can use.`);
  }
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
  const grid = [];
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
      soak: part.soak || 0,
      magazine: !!part.magazine,
      crewCost: part.crewCost || 0,
      crewSupply: part.crewSupply || 0,
    };
    cells.push(cell);
    byKey.set(key, cell);
    grid[gridIndex(dx, dz)] = cell;
  }

  const guns = [];
  for (const cell of cells) {
    const part = PARTS[cell.id];
    if (!part.gun) continue;
    // Where the gun can shoot, in ship-local radians. A broadside gun points out the flank it
    // sits on and nowhere else; a bow chaser only forward; a swivel everywhere, which one
    // 180-degree window covers.
    //
    // This is what makes the layout a decision. The engagement settles on one beam for the whole
    // battle and which one is drawn at random (config.drawOrbitSense), so massing the battery on
    // one flank doubles the broadside that bears or wastes it, while splitting it fights half a
    // battery every time and is never caught out. Letting broadsides answer to either beam
    // removes the exploit too, but it removes the decision with it.
    const arcs =
      part.gun.arc === 'side' ? [sideOfCell(cell.dx) === 'port' ? -Math.PI / 2 : Math.PI / 2] : [0];
    guns.push({
      cell,
      spec: part.gun,
      arcs,
      halfArc: (part.gun.halfArc * Math.PI) / 180,
      reloadLeft: 0, // createBattle staggers the battery with the seeded rng
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
    grid,
    guns,
    // Maintained rather than recounted: steer() and checkEnd() both want it every tick.
    aliveCells: cells.length,
    x: startPos.x,
    z: startPos.z,
    heading: startHeading,
    speed: 0,
    ammo: 'round',
    crewLost: 0,
    initialStructure: cells.reduce((a, c) => a + c.maxHp, 0),
    // sailFactor's target mast count never changes during a battle.
    sailWanted: mastsWanted(cells.length),
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
    crewSupply += cell.crewSupply;
    if (cell.id === 'mast') masts++;
    if (cell.magazine) magazines++;
  }
  ship.crewSupply = crewSupply;
  // Grape kills fractions of a man per pellet, so floor it: the number that mans the guns is
  // the number the panel shows.
  ship.crew = Math.max(0, Math.floor(crewSupply - ship.crewLost));
  ship.masts = masts;
  ship.magazines = magazines;

  // Crew man the guns, and that is all crew do. An earlier version also let a thin crew
  // slow the ship down, which shifted win rates by 9% through a channel no player could
  // see or reason about.
  let pool = ship.crew;
  for (const gun of ship.guns) {
    const need = gun.cell.crewCost;
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
      ship.aliveCells--;
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
