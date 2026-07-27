// Hull footprints, drawn bow-first as ASCII so they stay easy to edit.
//
// Cells are addressed by offset from the hull centre, (dx, dz), where dz is negative
// toward the bow. Because every hull is centred, a part at (-1, 0) stays at (-1, 0) when
// the player moves up to a bigger hull between rounds.

const SLOOP = [
  '.#.',
  '###',
  '###',
  '###',
  '.#.',
];

const BRIG = [
  '.#.',
  '###',
  '###',
  '###',
  '###',
  '###',
  '.#.',
];

const FRIGATE = [
  '..#..',
  '.###.',
  '#####',
  '#####',
  '#####',
  '.###.',
  '..#..',
];

const HEAVY_FRIGATE = [
  '..#..',
  '.###.',
  '.###.',
  '#####',
  '#####',
  '#####',
  '.###.',
  '.###.',
  '..#..',
];

const SHIP_OF_THE_LINE = [
  '..#..',
  '.###.',
  '#####',
  '#####',
  '#####',
  '#####',
  '#####',
  '#####',
  '.###.',
  '..#..',
];

// How far aft of the stem a bow chaser can be worked. Three rows leaves a real forward battery of
// a few guns without letting a ship of the line carry fourteen of them -- and unlike a broadside,
// every bow gun bears all the time, which measured as a 10-0 matchup. Two rows was too tight: with
// the magazine and the foremast also wanting spine cells up there, the bot could not place a single
// long gun and the part went untaken in every sampled build.
const BOW_ROWS = 3;

function build(name, art) {
  const h = art.length;
  const w = art[0].length;
  // Centre on the middle cell for odd sizes; for even row counts bias one cell aft of
  // amidships so the helm still sits on the spine.
  const cx = (w - 1) / 2;
  const cz = Math.floor((h - 1) / 2);
  const cells = [];
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      if (art[row][col] !== '#') continue;
      cells.push({ dx: col - cx, dz: row - cz });
    }
  }
  const bowZ = Math.min(...cells.map((c) => c.dz));
  return { name, cells, width: w, length: h, bowZ, bowLimit: bowZ + BOW_ROWS - 1 };
}

export const HULLS = [
  build('Sloop', SLOOP),
  build('Brig', BRIG),
  build('Frigate', FRIGATE),
  build('Heavy frigate', HEAVY_FRIGATE),
  build('Ship of the line', SHIP_OF_THE_LINE),
];


export function cellKey(dx, dz) {
  return `${dx},${dz}`;
}

// A bow chaser fires forward over the stem, so it has to be near it. Without this a ship of the
// line could carry fourteen of them, and unlike a broadside every one would bear all the time:
// measured, that pure long-gun build beat a pure broadside build 10-0 on the two biggest hulls.
export function isBowCell(hullIndex, dz) {
  return dz <= HULLS[hullIndex].bowLimit;
}
