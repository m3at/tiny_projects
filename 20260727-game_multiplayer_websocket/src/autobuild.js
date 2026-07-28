// Greedy ship builder: the stand-in for a player in the headless harnesses, and behind
// the dev Fill button so playtesting does not mean clicking 38 cells by hand.
//
// Deliberately outside sim/, which is reserved for the deterministic battle core.

import { PARTS } from './data/parts.js';
import { HULLS, cellKey, isBowCell } from './data/hulls.js';
import { mastsWanted } from './config.js';
import { HELM_KEY, sideOfCell } from './sim/ship.js';

function freeCells(design, hullIndex) {
  return HULLS[hullIndex].cells.filter((c) => !design.parts[cellKey(c.dx, c.dz)]);
}

function spine(cells) {
  return cells
    .filter((c) => c.dx === 0)
    .sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz) || a.dz - b.dz);
}

// Amidships flank cells first. Without a preferred side they alternate, so broadsides stay
// even; with one, that side fills completely before the other gets a cell, which is how a
// player masses a battery on the flank the wind says will bear.
function flanks(cells, side = null) {
  const port = cells.filter((c) => c.dx < 0).sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz));
  const star = cells.filter((c) => c.dx > 0).sort((a, b) => Math.abs(a.dz) - Math.abs(b.dz));
  if (side === 'port') return [...port, ...star];
  if (side === 'starboard') return [...star, ...port];
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

// Cells a given gun is allowed to occupy, in the order the builder wants them.
function gunCells(design, hullIndex, gunId, massedSide) {
  const free = freeCells(design, hullIndex);
  const arc = PARTS[gunId].gun.arc;
  if (arc === 'side') return flanks(free, massedSide).filter((c) => sideOfCell(c.dx) !== null);
  const all = [...flanks(free, null), ...spine(free)];
  return arc === 'bow' ? all.filter((c) => isBowCell(hullIndex, c.dz)) : all;
}

// `profile.massed` crams the whole battery onto one flank. It stays as a regression guard: while
// the engaged beam was predictable, that build won 100% of 800 battles at every hull size. The
// beam is drawn per battle now, so it should measure as an even match with brawler; if it ever
// climbs again, that has been undone.
//
// `profile.second` is a supporting gun, filled in after the primary has taken what it can. Some
// guns cannot arm a whole ship on their own -- a bow chaser has only the bow to work from -- and
// pretending otherwise measured a pure long-gun ship as hopeless while a ship with long guns
// forward and a broadside amidships is a real build a player would arrive at.
export function autoBuild(design, hullIndex, budget, profile, side = 'port') {
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
  // How many cells this kind of gun may actually occupy: broadsides want flanks, a bow chaser
  // only the bow, and an all-round gun can use leftover spine as well.
  const gunArc = PARTS[gunId].gun.arc;
  const flankRoom =
    gunArc === 'bow'
      ? hull.cells.filter((c) => isBowCell(hullIndex, c.dz)).length
      : hull.cells.filter((c) => c.dx !== 0).length +
        (gunArc === 'side' ? 0 : Math.max(0, hull.cells.filter((c) => c.dx === 0).length - 4));
  const spineRoom = hull.cells.filter((c) => c.dx === 0).length - 1; // helm sits on the spine
  // Reserve a scrap per cell we haven't filled yet: an open hole lets a ball through to
  // the magazine, so plugging the hull is worth more than one extra gun.
  const holeReserve = (g, m) => hull.cells.length - (2 + g + m + crewFor(g, m));
  const affordable = (g, m) => costOf(g, m) + Math.max(0, holeReserve(g, m)) <= scrap;
  // Cells are the other budget, and the one that used to be missed: guns were grown until the
  // purse ran out, which on a big hull left no room for the crew quarters to man them.
  const roomFor = (g, m) => 2 + g + m + crewFor(g, m) <= hull.cells.length;
  while (
    wantGuns + 1 <= flankRoom - 2 &&
    affordable(wantGuns + 1, wantMasts) &&
    roomFor(wantGuns + 1, wantMasts) &&
    wantGuns < 24
  ) {
    wantGuns++;
  }
  while (
    wantMasts + 1 <= spineRoom - 2 &&
    affordable(wantGuns, wantMasts + 1) &&
    roomFor(wantGuns, wantMasts + 1) &&
    wantMasts < 5
  ) {
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

  // 2. Masts forward on the spine -- but never in the bow if this ship wants a chaser there.
  //    Masts used to take the stem first, which quietly meant a long-gun build placed no long
  //    guns at all: the only cells they are allowed were already stepped with masts.
  {
    const wantsBow = [gunId, profile.second].some((id) => id && PARTS[id].gun?.arc === 'bow');
    const cells = spine(freeCells(design, hullIndex))
      .filter((c) => !wantsBow || !isBowCell(hullIndex, c.dz))
      .sort((a, b) => a.dz - b.dz);
    for (let i = 0; i < wantMasts && i < cells.length; i++) {
      if (spend('mast')) place(design, cells[i], 'mast');
    }
  }

  // 3. Guns. The primary takes the cells its arc allows -- a broadside wants flanks, a bow
  //    chaser only the bow, an all-round gun will take leftover spine, which is the swivel's real
  //    niche since it is the only gun that can go down the middle.
  {
    const cells = gunCells(design, hullIndex, gunId, profile.massed ? side : null);
    for (let i = 0; i < wantGuns && i < cells.length; i++) {
      if (spend(gunId)) place(design, cells[i], gunId);
    }
  }

  // 3b. A supporting gun, if the profile has one and the primary left something over. Each
  //     addition has to leave room and scrap for the hands to work it and a plug for every
  //     remaining hole, otherwise the ship ends up bristling and silent.
  if (profile.second) {
    const secondId = profile.second;
    const cost = PARTS[secondId].cost;
    for (const cell of gunCells(design, hullIndex, secondId, profile.massed ? side : null)) {
      // Crew have not been bought yet, so the whole ship's requirement has to be reserved fresh
      // each time, in cells as well as scrap. Reserving only this gun's share left a ship of the
      // line carrying thirty guns and twenty-nine of them silent.
      let hands = PARTS[secondId].crewCost || 0;
      for (const slot of Object.values(design.parts)) hands += PARTS[slot.id].crewCost || 0;
      const quarters = Math.ceil(hands / 3);
      const cellsLeft = freeCells(design, hullIndex).length - 1;
      const plugs = Math.max(0, cellsLeft - quarters);
      if (quarters > cellsLeft) break;
      if (cost + quarters * PARTS.crew.cost + plugs > scrap) break;
      if (!spend(secondId)) break;
      place(design, cell, secondId);
    }
  }

  // 4. Crew aft on the spine, enough to man what we bought.
  //    Note it spills onto whichever flank is free, which used to be a free win: with a fixed
  //    orbit sense the enemy always sat on the same beam, so a lopsided build could tuck its
  //    crew and powder on the sheltered side. The sense is drawn per battle now.
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

// A plausible but unplanned ship, the way a player who takes what the draft offers ends up. Used by
// tools/parts.js to sample the space of buildable ships, and by the dev harness so an autoplayed
// match looks like a game rather than like a pure gun-deck ship meeting a pure carronade one.
//
// A bow chaser can never be a ship's main armament -- there is only the bow to work it from -- so it
// only ever appears as the supporting gun.
const PRIMARY_GUNS = ['swivel', 'gundeck', 'carronade'];
const SECOND_GUNS = ['swivel', 'gundeck', 'carronade', 'longgun'];

export function randomProfile(rng, hullIndex) {
  const wanted = mastsWanted(HULLS[hullIndex].cells.length);
  return {
    gun: PRIMARY_GUNS[rng.int(0, PRIMARY_GUNS.length - 1)],
    second: rng.range(0, 1) < 0.55 ? SECOND_GUNS[rng.int(0, SECOND_GUNS.length - 1)] : null,
    gunCount: rng.int(1, 8),
    // At most one mast past what the hull can use: the readout states the number, so a player
    // over-rigging by five is not a build the game encourages.
    mastCount: rng.int(1, wanted + 1),
    armour: rng.range(0, 1) < 0.5 ? 'heavy' : 'timber',
    massed: rng.range(0, 1) < 0.35,
    label: 'Drafted ship',
  };
}

export const ARCHETYPES = {
  brawler: { gun: 'gundeck', gunCount: 4, mastCount: 2, armour: 'heavy', label: 'Broadside brawler' },
  massed: {
    gun: 'gundeck',
    gunCount: 4,
    mastCount: 2,
    armour: 'heavy',
    massed: true,
    label: 'Massed battery',
  },
  // Long guns forward, broadside amidships: the bow can only work two or three chasers, so this
  // is what a long-gun ship actually looks like.
  sniper: {
    gun: 'longgun',
    gunCount: 4,
    mastCount: 3,
    second: 'gundeck',
    armour: 'timber',
    label: 'Long-gun sniper',
  },
  harasser: { gun: 'swivel', gunCount: 6, mastCount: 4, armour: 'timber', label: 'Grape harasser' },
  crusher: { gun: 'carronade', gunCount: 4, mastCount: 3, armour: 'heavy', label: 'Carronade crusher' },
  // Not an archetype so much as a control: a bit of everything, which is what a player who takes
  // what the draft offers ends up with. If the pure builds show hard counters but this one sits
  // near even against all of them, those counters are the bot's blind spot rather than the
  // game's, and nerfing parts to erase them would be a mistake.
  mixed: {
    gun: 'gundeck',
    gunCount: 3,
    mastCount: 3,
    second: 'swivel',
    armour: 'heavy',
    label: 'Mixed battery',
  },
};
