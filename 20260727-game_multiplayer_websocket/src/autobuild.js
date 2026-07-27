// Greedy ship builder: the stand-in for a player in the headless harnesses, and behind
// the dev Fill button so playtesting does not mean clicking 38 cells by hand.
//
// Deliberately outside sim/, which is reserved for the deterministic battle core.

import { PARTS } from './data/parts.js';
import { HULLS, cellKey } from './data/hulls.js';
import { HELM_KEY, sideOfCell } from './sim/ship.js';

function freeCells(design, hullIndex) {
  return HULLS[hullIndex].cells.filter((c) => !design.parts[cellKey(c.dx, c.dz)]);
}

function spine(cells) {
  return cells
    .filter((c) => c.dx === 0)
    .sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz) || a.dz - b.dz);
}

// Amidships flank cells first, alternating port and starboard so broadsides stay even.
function flanks(cells) {
  const port = cells.filter((c) => c.dx < 0).sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz));
  const star = cells.filter((c) => c.dx > 0).sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz));
  const out = [];
  for (let i = 0; i < Math.max(port.length, star.length); i++) {
    if (port[i]) out.push(port[i]);
    if (star[i]) out.push(star[i]);
  }
  return out;
}

function place(design, cell, partId) {
  design.parts[cellKey(cell.dx, cell.dz)] = { id: partId, hp: PARTS[partId].hp };
}

export function autoBuild(design, hullIndex, budget, profile) {
  const gunId = profile.gun || 'gundeck';
  const gunCrew = PARTS[gunId].crewCost || 0;
  let wantGuns = profile.gunCount ?? 2;
  let wantMasts = profile.mastCount ?? 2;
  let scrap = budget;

  if (!design.parts[HELM_KEY]) place(design, { dx: 0, dz: 0 }, 'helm');

  const crewFor = (g, m) => Math.ceil((g * gunCrew + m) / 3);
  const costOf = (g, m) =>
    PARTS[gunId].cost * g + PARTS.mast.cost * m + PARTS.crew.cost * crewFor(g, m) + PARTS.magazine.cost;

  // Trim ambitions until the core fits: give up sail before giving up guns, but never
  // drop below a single mast.
  while (wantGuns > 0 && costOf(wantGuns, wantMasts) > scrap) {
    if (wantMasts > 1) wantMasts--;
    else wantGuns--;
  }
  if (wantGuns === 0) {
    wantMasts = Math.max(0, Math.min(wantMasts, Math.floor((scrap - 4) / PARTS.mast.cost)));
  }

  // ...and grow them again if the hull and the purse can take more, so a big budget
  // doesn't sit unspent on a ship-of-the-line.
  const hull = HULLS[hullIndex];
  const flankRoom = hull.cells.filter((c) => c.dx !== 0).length;
  const spineRoom = hull.cells.filter((c) => c.dx === 0).length - 1; // helm sits on the spine
  // Reserve a scrap per cell we haven't filled yet: an open hole lets a ball through to
  // the magazine, so plugging the hull is worth more than one extra gun.
  const holeReserve = (g, m) => hull.cells.length - (2 + g + m + crewFor(g, m));
  const affordable = (g, m) => costOf(g, m) + Math.max(0, holeReserve(g, m)) <= scrap;
  while (wantGuns + 1 <= flankRoom - 2 && affordable(wantGuns + 1, wantMasts) && wantGuns < 12) {
    wantGuns++;
  }
  while (wantMasts + 1 <= spineRoom - 2 && affordable(wantGuns, wantMasts + 1) && wantMasts < 5) {
    wantMasts++;
  }

  const spend = (partId) => {
    if (scrap < PARTS[partId].cost) return false;
    scrap -= PARTS[partId].cost;
    return true;
  };

  // 1. Magazine, tucked into the spine as far from the hull edge as we can manage.
  {
    const cells = spine(freeCells(design, hullIndex));
    const spot = cells[cells.length > 2 ? 1 : 0];
    if (spot && spend('magazine')) place(design, spot, 'magazine');
  }

  // 2. Masts forward on the spine.
  {
    const cells = spine(freeCells(design, hullIndex)).sort((a, b) => a.dz - b.dz);
    for (let i = 0; i < wantMasts && i < cells.length; i++) {
      if (spend('mast')) place(design, cells[i], 'mast');
    }
  }

  // 3. Guns on the flanks.
  {
    const cells = flanks(freeCells(design, hullIndex)).filter(
      (c) => PARTS[gunId].gun.arc !== 'side' || sideOfCell(c.dx) !== null,
    );
    for (let i = 0; i < wantGuns && i < cells.length; i++) {
      if (spend(gunId)) place(design, cells[i], gunId);
    }
  }

  // 4. Crew aft on the spine, enough to man what we bought.
  {
    let need = 0;
    for (const slot of Object.values(design.parts)) need += PARTS[slot.id].crewCost || 0;
    const quarters = Math.ceil(need / 3);
    // Aft on the spine by preference, but spill onto the flanks rather than leave guns
    // unmanned when the spine is full of masts.
    const cells = [
      ...spine(freeCells(design, hullIndex)).sort((a, b) => b.dz - a.dz),
      ...flanks(freeCells(design, hullIndex)).reverse(),
    ];
    for (let i = 0; i < quarters && i < cells.length; i++) {
      if (spend('crew')) place(design, cells[i], 'crew');
    }
  }

  // 5. Armour the outer flanks, then plug every remaining hole with cheap timber.
  {
    const armour = profile.armour || 'heavy';
    const cells = freeCells(design, hullIndex).sort(
      (a, b) => Math.abs(b.dx) - Math.abs(a.dx) || Math.abs(a.dz) - Math.abs(b.dz),
    );
    // Plugging every hole with cheap timber beats one thick plate and five open holes, so
    // only upgrade to armour with scrap surplus to what timbering the rest would cost.
    let holes = cells.length;
    for (const c of cells) {
      const upgrade = PARTS[armour].cost - PARTS.timber.cost;
      if (armour !== 'timber' && scrap - holes >= upgrade && scrap >= PARTS[armour].cost) {
        scrap -= PARTS[armour].cost;
        place(design, c, armour);
      } else if (scrap >= PARTS.timber.cost) {
        scrap -= PARTS.timber.cost;
        place(design, c, 'timber');
      } else break;
      holes--;
    }
  }

  return scrap;
}

export const ARCHETYPES = {
  brawler: { gun: 'gundeck', gunCount: 4, mastCount: 2, armour: 'heavy', label: 'Broadside brawler' },
  sniper: { gun: 'longgun', gunCount: 3, mastCount: 3, armour: 'timber', label: 'Long-gun sniper' },
  harasser: { gun: 'swivel', gunCount: 6, mastCount: 4, armour: 'timber', label: 'Grape harasser' },
  crusher: { gun: 'carronade', gunCount: 4, mastCount: 3, armour: 'heavy', label: 'Carronade crusher' },
};
