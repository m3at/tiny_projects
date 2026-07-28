// Ship model: the persistent design a player edits between rounds, and the runtime state
// the battle simulation mutates.

import { PARTS } from '../data/parts.js';
import { HULLS, cellKey, isBowCell } from '../data/hulls.js';
import { CELL, mastsWanted, massFactor, sailFactor } from '../config.js';
import { fcos, fsin } from './geometry.js';

export const HELM_KEY = '0,0';

// A flat integer index for a cell offset, for the lookup that happens per projectile per tick.
// Building a "dx,dz" string there was the simulation's single hottest allocation.
//
// The backing array is dense -- allocated full of nulls rather than left with holes -- because a
// sparse JavaScript array can fall out of V8's fast element kinds into dictionary mode, and this is
// the busiest read in the simulation. Span and offsets comfortably cover the largest hull, which is
// five cells wide and ten long.
const GRID_SPAN = 32;
const GRID_DX = 8;
const GRID_DZ = 16;
export const GRID_SIZE = GRID_SPAN * GRID_SPAN;

export function gridIndex(dx, dz) {
  return (dx + GRID_DX) * GRID_SPAN + (dz + GRID_DZ);
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

// Duplicate a design. Cheaper than structuredClone, which showed up in the profile of the
// harnesses -- they clone before every battle, because a battle writes damage back.
export function cloneDesign(design) {
  const parts = {};
  for (const key in design.parts) {
    const slot = design.parts[key];
    parts[key] = { id: slot.id, hp: slot.hp };
  }
  return { parts };
}

// Ships are named by seat, because ownership is carried by the hull colour and not the cargo.
// A room can rename a seat for the interface; the simulation's log stays impersonal.
export function shipName(ship) {
  return `Player ${ship.index + 1}`;
}

export function makeBattleShip(design, hullIndex, index, startPos, startHeading) {
  const cells = [];
  const grid = new Array(GRID_SIZE).fill(null);
  for (const key in design.parts) {
    const slot = design.parts[key];
    // Cheaper than key.split(',').map(Number), which allocated two arrays per cell per battle.
    const comma = key.indexOf(',');
    const dx = +key.slice(0, comma);
    const dz = +key.slice(comma + 1);
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
      hasGun: !!part.gun,
      reached: 0, // stamp used by severDisconnected
      crewCost: part.crewCost || 0,
      crewSupply: part.crewSupply || 0,
    };
    cells.push(cell);
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
    const arc =
      part.gun.arc === 'side' ? (sideOfCell(cell.dx) === 'port' ? -Math.PI / 2 : Math.PI / 2) : 0;
    const halfArc = (part.gun.halfArc * Math.PI) / 180;
    guns.push({
      cell,
      spec: part.gun,
      arc,
      halfArc,
      // The arc test is a dot product rather than an angle comparison, so it wants the arc's
      // direction as a rotation of the ship's heading and its width as a cosine. A gun that bears
      // all round gets halfArc >= 180, whose cosine is -1, and the test passes unconditionally
      // without needing a special case.
      arcCos: fcos(arc),
      arcSin: fsin(arc),
      // Squared, because the arc test compares squared lengths to keep a square root out of the
      // inner loop. Sound only because every directional arc here is under a quarter turn, so its
      // cosine is positive; a gun that bears all round skips the test entirely.
      allRound: halfArc >= Math.PI / 2,
      cosHalfArcSq: fcos(halfArc) * fcos(halfArc),
      rangeSq: part.gun.range * part.gun.range,
      // An absolute deadline rather than a countdown: a gun is ready when battle.time reaches it.
      // Counting down meant writing to every gun on every tick, whether anything was happening or
      // not. createBattle staggers the opening battery with the seeded rng.
      readyAt: 0,
      manned: false,
    });
  }
  guns.sort((a, b) => (a.cell.key < b.cell.key ? -1 : 1));

  // How far a ball can be from the ship's centre and still land on a cell: the outermost cell
  // centre, plus the half-cell the hit test rounds over on each axis. Squared, because it is only
  // ever compared against a squared distance -- a cheap reject before any trigonometry, and most
  // shot in the air at any moment is nowhere near its target.
  let farSq = 0;
  for (const cell of cells) {
    const d = cell.lx * cell.lx + cell.lz * cell.lz;
    if (d > farSq) farSq = d;
  }
  const far = Math.sqrt(farSq);
  const hitRadius = far + CELL * Math.SQRT1_2 + 1e-9;

  const ship = {
    index,
    design,
    hullIndex,
    cells,
    grid,
    guns,
    helm: grid[gridIndex(0, 0)],
    hitRadiusSq: hitRadius * hitRadius,
    // Heading is fixed for the duration of a tick, so its sine and cosine are cached rather than
    // recomputed per gun and per projectile. steer() refreshes them.
    cos: fcos(startHeading),
    sin: fsin(startHeading),
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
    // The hull as an ellipse, for keeping two of them out of each other. Half the deck's length along
    // the ship and half its width across, in world units -- the drawn silhouette is a little larger
    // still, but the deck plates are what a player sees overlapping.
    semiLong: (HULLS[hullIndex].length / 2) * CELL,
    semiWide: (HULLS[hullIndex].width / 2) * CELL,
    reachStamp: 0,
    swept: false,
    // Melee bookkeeping. `out` is set when the helm goes and the ship leaves the fight; `outAt` is
    // when, which is how a four-way is placed below the survivor. Both are inert in a duel, which
    // ends the moment either is set. `target` and `enemies` are filled in by createBattle.
    out: false,
    outAt: 0,
    target: null,
    enemies: null,
  };
  refreshSystems(ship);
  return ship;
}

export function refreshSystems(ship) {
  let crewSupply = 0;
  let masts = 0;
  let magazines = 0;
  const cells = ship.cells;
  for (let i = 0; i < cells.length; i++) {
    const cell = cells[i];
    if (!cell.alive) continue;
    crewSupply += cell.crewSupply;
    if (cell.id === 'mast') masts++;
    if (cell.magazine) magazines++;
  }
  ship.crewSupply = crewSupply;
  // Speed and turn rate only move when masts or cells are lost, and this function is exactly where
  // that is noticed. steer() reads them every tick and used to recompute both.
  ship.mass = massFactor(ship.aliveCells || 1);
  ship.sail = sailFactor(masts, ship.sailWanted);
  // Grape kills fractions of a man per pellet, so floor it: the number that mans the guns is
  // the number the panel shows.
  ship.crew = Math.max(0, Math.floor(crewSupply - ship.crewLost));
  ship.masts = masts;
  ship.magazines = magazines;

  // Crew man the guns, and that is all crew do. An earlier version also let a thin crew
  // slow the ship down, which shifted win rates by 9% through a channel no player could
  // see or reason about.
  // Hands work the sails before they work the guns. The build readout has always counted a mast's
  // crew against the total; the battle used to ignore it, so a ship fought with more guns manned
  // than the panel said it could.
  let sailHands = 0;
  for (let i = 0; i < cells.length; i++) {
    const cell = cells[i];
    if (cell.alive && !cell.hasGun) sailHands += cell.crewCost;
  }

  let pool = Math.max(0, ship.crew - sailHands);
  let anyManned = false;
  const guns = ship.guns;
  for (let i = 0; i < guns.length; i++) {
    const gun = guns[i];
    const need = gun.cell.crewCost;
    gun.manned = gun.cell.alive && pool >= need;
    if (gun.manned) {
      pool -= need;
      anyManned = true;
    }
  }
  // checkEnd asks this every tick and used to walk the gun list twice to answer it.
  ship.canFire = magazines > 0 && anyManned;
}

export function structureFraction(ship) {
  let left = 0;
  const cells = ship.cells;
  for (let i = 0; i < cells.length; i++) if (cells[i].alive) left += cells[i].hp;
  return ship.initialStructure === 0 ? 0 : left / ship.initialStructure;
}

function liveNeighbours(ship, dx, dz) {
  const g = ship.grid;
  const at = gridIndex(dx, dz);
  let n = 0;
  if (g[at + GRID_SPAN] !== null && g[at + GRID_SPAN].alive) n++;
  if (g[at - GRID_SPAN] !== null && g[at - GRID_SPAN].alive) n++;
  if (g[at + 1] !== null && g[at + 1].alive) n++;
  if (g[at - 1] !== null && g[at - 1].alive) n++;
  return n;
}

// Anything no longer joined to the helm breaks away.
//
// `lost` is the cell that just died. Removing a cell with one live neighbour or none cannot cut a
// connected hull in two, so the flood fill is skipped -- for the great majority of hits, since most
// shot lands on the outside of a ship. That shortcut is only sound once the hull is known to be
// connected in the first place, and it is not: nothing stops a player placing a part in a cell with
// no path back to the helm, and such a section rides along until the first hit of any kind sweeps
// it off. So the first sweep on a ship always runs in full, which is also what the original did on
// every single hit.
//
// Reachability is marked with a per-ship stamp on the cells rather than a Set of "dx,dz" strings.
// This runs on every cell destroyed and was 4% of simulation time.
export function severDisconnected(ship, lost = null) {
  const helm = ship.helm;
  if (!helm || !helm.alive) return [];
  if (lost && ship.swept && liveNeighbours(ship, lost.dx, lost.dz) <= 1) return [];

  const stamp = ++ship.reachStamp;
  const stack = [helm];
  helm.reached = stamp;
  while (stack.length) {
    const cur = stack.pop();
    for (let i = 0; i < 4; i++) {
      const nx = cur.dx + (i === 0 ? 1 : i === 1 ? -1 : 0);
      const nz = cur.dz + (i === 2 ? 1 : i === 3 ? -1 : 0);
      const cell = ship.grid[gridIndex(nx, nz)];
      if (cell === null || !cell.alive || cell.reached === stamp) continue;
      cell.reached = stamp;
      stack.push(cell);
    }
  }

  ship.swept = true;
  const severed = [];
  for (const cell of ship.cells) {
    if (cell.alive && cell.reached !== stamp) {
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
