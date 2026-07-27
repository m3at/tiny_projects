// Single place for every number that affects how the game plays or feels.
// Part statistics live in data/parts.js and hull shapes in data/hulls.js; everything else
// is here. tools/balance.js, tools/match.js and tools/events.js all read from this, so a
// change here is measurable in seconds.

// ---------------------------------------------------------------------------
// Space and time
// ---------------------------------------------------------------------------

export const CELL = 2.4; // world units per hull cell
export const TICK = 1 / 60;
export const BATTLE_CAP = 30; // seconds before the round is decided on damage
export const ARENA_RADIUS = 78;
export const START_OFFSET = { x: 10, z: 38 }; // ships start mirrored about the origin

// ---------------------------------------------------------------------------
// Sailing
// ---------------------------------------------------------------------------

export const BASE_SPEED = 13.5; // units/sec at full sail, running with the wind
export const BASE_TURN = 1.15; // rad/sec at full sail
export const MIN_SEPARATION = 9;

// Wind multiplier runs from WIND_MIN close-hauled to 1.0 running.
//
// This is the most load-bearing number in the file: tools/ablate.js shows that flattening
// it to 1 changes a quarter of all outcomes. It works by making one side of the orbit
// faster than the other, so the ships genuinely contest the weather gauge. Note the effect
// is the speed penalty, not the direction -- sweeping the direction through 24 points
// barely ever flips a winner.
export const WIND_MIN = 0.35;

export function massFactor(cellCount) {
  return 12 / (12 + cellCount * 0.42);
}

export function sailFactor(aliveMasts, hullCells) {
  const wanted = Math.max(2, Math.ceil(hullCells / 10));
  return 0.22 + 0.78 * Math.min(1, aliveMasts / wanted);
}

export function windFactor(heading, windTo) {
  return WIND_MIN + ((1 - WIND_MIN) * (Math.cos(heading - windTo) + 1)) / 2;
}

// How far out a ship wants to fight, as a fraction of its weapons' reach. Below 1 so guns
// stay comfortably in range rather than flickering at the edge of it.
export const PREFERRED_RANGE_FRACTION = 0.85;

// ---------------------------------------------------------------------------
// Gunnery
// ---------------------------------------------------------------------------

export const AMMO_SWITCH_RELOAD = 1.3; // switching ammunition costs you the loaded guns
export const GRAPE_EXTRA_SHOTS = 1; // grape throws this many more pellets per volley
export const GRAPE_SPREAD_SCALE = 1.8;
export const MAGAZINE_BLAST_CREW = 2; // crew lost when your own powder goes up

// ---------------------------------------------------------------------------
// Economy
// ---------------------------------------------------------------------------

// One entry per round: which hull, how much scrap is granted, how long to fit out.
// Grants have to outpace attrition or ships get holier every round instead of grander.
export const ROUNDS = [
  { hull: 0, scrap: 34, buildTime: 40 },
  { hull: 1, scrap: 32, buildTime: 26 },
  { hull: 2, scrap: 42, buildTime: 26 },
  { hull: 3, scrap: 46, buildTime: 28 },
  { hull: 4, scrap: 56, buildTime: 30 },
];

export const POINTS_TO_WIN = 3;

// Comeback money, as a fraction of the round's grant. A flat bonus is worth nothing by
// round 5, which is how you get 3-0 sweeps.
export const LOSER_BONUS_FRACTION = 0.45;

export function loserBonus(roundIndex) {
  return Math.round(ROUNDS[roundIndex].scrap * LOSER_BONUS_FRACTION);
}

export const REROLL_COST = 2;
export const OFFER_SIZE = 5;
export const REPAIR_FRACTION = 0.5;
