import { REPAIR_FRACTION } from '../config.js';

// Part definitions. Pure data, no renderer dependency.
//
// Reloads and damages are deliberately on the fine side: the same damage per second split into
// twice as many volleys. A battery that fires one heavy clap every three seconds leaves nothing
// on screen in between, which measured as two thirds of the battle being dead air.
//
// Muzzle speeds are half what they were for the same reason, and for a second one: a ball that
// takes a second to cross the water can be watched and anticipated, where a fast one simply
// appears as a hit. It costs accuracy against a moving target, which is paid for in damage.
//
// arc: where the gun can shoot, in ship-local degrees. 'bow' is one window centred forward,
//   'all' is one window wide enough to cover everything, and 'side' points out the flank the
//   cell sits on: 0 = bow, +90 = starboard, -90 = port.
// Ranges and speeds are in world units (one hull cell is CELL units wide, see config.js).

export const PARTS = {
  timber: {
    id: 'timber',
    name: 'Hull timber',
    glyph: '=',
    cost: 1,
    hp: 9,
    color: 0x8a6844,
    height: 0.3,
    blurb: 'Cheap filler. Stops a ball reaching the spine.',
  },
  heavy: {
    id: 'heavy',
    name: 'Heavy timbers',
    glyph: '#',
    cost: 3,
    hp: 30,
    soak: 2, // flat reduction per incoming hit
    color: 0x4d5a5e,
    height: 0.5,
    blurb: 'Soaks 2 off every hit. Small shot bounces.',
  },
  crew: {
    id: 'crew',
    name: 'Crew quarters',
    glyph: 'c',
    cost: 5,
    hp: 14,
    crewSupply: 3,
    color: 0xc9a227,
    height: 0.55,
    blurb: 'Supplies 3 crew. Guns without crew stay silent.',
  },
  mast: {
    id: 'mast',
    name: 'Mast',
    glyph: '^',
    cost: 4,
    hp: 11,
    crewCost: 1,
    color: 0xd8cbb0,
    height: 2.6,
    blurb: 'Speed and turn rate. Lets you work upwind.',
  },
  magazine: {
    id: 'magazine',
    name: 'Powder magazine',
    glyph: '*',
    cost: 4,
    hp: 8,
    magazine: true,
    detonate: { damage: 15, radius: 1 },
    color: 0xb03030,
    height: 0.55,
    blurb: 'Needed to fire at all. Detonates when destroyed.',
  },
  swivel: {
    id: 'swivel',
    name: 'Swivel gun',
    glyph: 'o',
    cost: 4,
    hp: 11,
    crewCost: 1,
    color: 0x7fa8c9,
    height: 0.7,
    gun: {
      arc: 'all',
      halfArc: 180,
      range: 26,
      reload: 1,
      shots: 1,
      spread: 5,
      speed: 26,
      round: { damage: 3 },
      grape: { damage: 1, crew: 1 },
    },
    blurb: 'Fires all round, fast, weak. Good grape platform.',
  },
  gundeck: {
    id: 'gundeck',
    name: 'Gun deck',
    glyph: 'G',
    cost: 8,
    hp: 17,
    crewCost: 2,
    color: 0x4a6fa5,
    height: 0.7,
    gun: {
      arc: 'side',
      halfArc: 50,
      range: 38,
      reload: 1.8,
      shots: 3,
      spread: 7,
      speed: 30,
      round: { damage: 4 },
      grape: { damage: 1, crew: 1 },
    },
    blurb: 'The gun line. Three-ball volley out its own flank.',
  },
  carronade: {
    id: 'carronade',
    name: 'Carronade',
    glyph: 'K',
    cost: 9,
    hp: 14,
    crewCost: 1, // light enough for a small gun crew: that is its real advantage
    color: 0x8b5fb0,
    height: 0.7,
    gun: {
      arc: 'side',
      halfArc: 55,
      range: 24,
      reload: 1.9,
      shots: 2,
      spread: 6,
      speed: 28,
      round: { damage: 9 },
      grape: { damage: 2, crew: 2 },
    },
    blurb: 'Smashes anything that gets close. Useless at range. Needs only one hand.',
  },
  longgun: {
    id: 'longgun',
    name: 'Long gun',
    glyph: 'L',
    cost: 9,
    hp: 14,
    crewCost: 2,
    color: 0x2f8f6f,
    height: 0.8,
    gun: {
      arc: 'bow',
      halfArc: 32,
      range: 48,
      reload: 2.2,
      shots: 1,
      spread: 3,
      speed: 39,
      pierce: true, // halves heavy timber soak
      round: { damage: 22 },
      grape: { damage: 2, crew: 2 },
    },
    blurb: 'Bow only, and few fit. Longest reach, and punches through heavy timbers.',
  },
  helm: {
    id: 'helm',
    name: 'Helm',
    glyph: '@',
    cost: 0,
    hp: 20,
    fixed: true,
    color: 0xe8e8e8,
    height: 0.9,
    blurb: 'Lose the helm and you strike your colours.',
  },
};

// Order shown in the shop / legend.
export const BUYABLE = [
  'timber',
  'heavy',
  'crew',
  'mast',
  'magazine',
  'swivel',
  'gundeck',
  'carronade',
  'longgun',
];

export function repairCost(partId) {
  return Math.max(1, Math.ceil(PARTS[partId].cost * REPAIR_FRACTION));
}
