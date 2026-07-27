// Single place for every number that affects how the game plays or feels.
// Part statistics live in data/parts.js and hull shapes in data/hulls.js; everything else
// is here. tools/balance.js, tools/match.js and tools/events.js all read from this, so a
// change here is measurable in seconds.

// ---------------------------------------------------------------------------
// Space and time
// ---------------------------------------------------------------------------

export const CELL = 2.4; // world units per hull cell
export const TICK = 1 / 60;
export const BATTLE_CAP = 40; // hard stop; the round is decided on damage if it is reached

// Overtime. Rather than let a slow battle run to the bell and be awarded on structure, gunnery
// gets steadily deadlier from OVERTIME_AT until something breaks. The shape matters more than the
// numbers: Riot replaced a flat overtime damage spike in TFT with a stacking per-second ramp
// because the spike ended rounds "almost instantly" and whoever fired first won regardless of
// what they had built, which destroys the connection between the build and the result. A gentle
// continuous ramp keeps that connection and still forbids a stalemate.
export const OVERTIME_AT = 20;
export const OVERTIME_RAMP = 0.14; // extra damage per second past OVERTIME_AT
export const OVERTIME_MAX = 1.6; // ...up to this much extra

export function overtimeScale(time) {
  return 1 + Math.min(OVERTIME_MAX, Math.max(0, time - OVERTIME_AT) * OVERTIME_RAMP);
}
export const ARENA_RADIUS = 60;
// Ships start mirrored about the origin, roughly bow-on. The opening run has nothing to
// watch, so it is kept to a couple of seconds: this is 51 units apart, inside a long gun's
// reach and about two seconds' sailing from a carronade's.
export const START_OFFSET = { x: 9, z: 24 };

// ---------------------------------------------------------------------------
// Sailing
// ---------------------------------------------------------------------------

export const BASE_SPEED = 13.5; // units/sec at full sail, running with the wind
export const BASE_TURN = 1.15; // rad/sec at full sail
// Hulls must not interpenetrate, and a ship of the line is nearly nine cells long, so the
// floor scales with the hull. tools/tune.js separation shows this changes no outcomes at all --
// the preferred range holds the ships much further apart than this - so it is purely a matter of
// two grand ships not appearing to grind through each other.
export const MIN_SEPARATION = 9;
export function minSeparation(hullLengthCells) {
  return Math.max(MIN_SEPARATION, hullLengthCells * CELL * 0.55);
}

// Which way the fight circles. Both ships take the same sense, so they orbit their common
// midpoint instead of sailing a parallel course off the edge of the arena, and the sense
// decides which flank does the work: +1 holds the enemy off your port side, -1 off your
// starboard.
//
// Which way the fight circles. Both ships take the same sense, so they orbit their common
// midpoint instead of sailing a parallel course off the edge of the arena.
//
// Drawn from the battle's own rng, not fixed and not shown during the build phase, because
// whichever way it goes the enemy spends the whole battle on one beam. Any sense a player could
// count on became a sheltered side to hide the crew and the powder behind: measured, a lopsided
// build that did exactly that won 100% of 800 battles at every hull size. Reversing the circle
// part way through did not fix it and walked the engagement into the arena wall instead. So the
// engaged beam is luck, which is also the honest answer -- a captain does not get to choose
// which way the action turns.
export function drawOrbitSense(rng) {
  return rng.range(0, 1) < 0.5 ? 1 : -1;
}

// How far off the preferred range a ship tolerates before it stops circling and starts
// closing or opening, as a fraction of that range. Small values give a tight, twitchy
// circle; large ones let the range breathe and the guns fall silent.
export const ORBIT_TOLERANCE = 0.6;

// How much of the hold angle a ship gives up to close the range. At 1 it turns its bow
// straight at the enemy, which for a broadside ship means closing with every gun blind: a
// carronade ship, with twice the raw damage of a gun deck ship, lost 100% of battles because it
// spent the whole approach pointing the wrong way. Below 1 it converges obliquely instead,
// slower but still firing. Anything under about 0.6 keeps a 50-degree beam arc on target the
// whole way in. See tools/tune.js orbit-close.
export const ORBIT_CLOSE = 0.55;

// How hard a ship runs when the enemy is inside its preferred range, as a fraction of a full
// turn away. 1 is a ship that turns tail, which is what kiting looks like: two ships holding a
// range neither can shoot at for twenty seconds, which is how this game used to look. At 0 a
// ship never breaks off and simply keeps circling, so the fight settles at the shorter of the
// two preferred ranges and both sides can shoot. Measured, 0 gives the shortest dead stretches
// and the flattest archetype spread; every step upward widens both. See tools/tune.js.
export const ORBIT_RETREAT = 0;

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

// `wanted` is the mast count the hull needs for full sail, precomputed per ship as
// ship.sailWanted.
// Masts a hull needs for full sail. Beyond this they do nothing at all, which measured as a trap:
// sampling random builds, the ship carrying more masts won only 34% of the time. The number is now
// shown in the build readout, because a cap the player cannot see is a cap they will pay for.
export function mastsWanted(hullCells) {
  return Math.max(2, Math.ceil(hullCells / 10));
}

export function sailFactor(aliveMasts, wanted) {
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

// Incoming gun damage, per hull. Measured rather than chosen: a bigger hull carries more
// guns and presents a bigger target, so the damage a ship takes grows faster than its
// structure does. Left alone, round 5 collapsed into a five-second coin flip between two
// grand ships while round 1 dragged on for twenty seconds with two guns. Heavier scantlings
// are the in-world reason; holding every round to about the same length is the real one.
// These follow from the economy and the part table, so rerun tools/watch.js if either moves.
export const HULL_DAMAGE = [1.05, 0.5, 0.36, 0.24, 0.2];

// A battery whose pieces are perfectly in step gives one clap per reload cycle and silence
// in between, however many guns it has: measured, that was two thirds of the battle with
// nothing in the air. These two break the lockstep. STAGGER offsets each gun once, at the
// start, as a fraction of its reload; JITTER is how much each reload varies after that, and
// keeps the battery from drifting back into step.
export const RELOAD_STAGGER = 0.35;
export const RELOAD_JITTER = 0.35;

export const AMMO_SWITCH_RELOAD = 1.3; // switching ammunition costs you the loaded guns
export const GRAPE_EXTRA_SHOTS = 1; // grape throws this many more pellets per volley
export const GRAPE_SPREAD_SCALE = 1.8;

// Crew killed per pellet, against the whole-number crew a part supplies. A gun deck throws
// four pellets and a seven-gun broadside throws twenty-eight, so at one man per pellet grape
// deleted an entire crew in a single volley: measured, a carronade ship with twice the raw
// damage of its opponent lost 100% of battles without ever firing a shot, because it was
// silenced three seconds in. Grape is meant to suppress a gun deck over several volleys, not
// end the battle. See tools/tune.js grape-crew.
export const GRAPE_CREW_SCALE = 0.15;
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
