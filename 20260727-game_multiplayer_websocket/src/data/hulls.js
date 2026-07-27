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
  return { name, cells, width: w, length: h };
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
