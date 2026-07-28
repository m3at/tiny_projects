// Single place for every number that affects how the game plays or feels.
// Part statistics live in data/parts.js and hull shapes in data/hulls.js; everything else
// is here. tools/balance.js, tools/match.js and tools/events.js all read from this, so a
// change here is measurable in seconds.

// The float32-collapsed trigonometry from the simulation, because the few functions here that use
// any are read by the simulation and have to give the same answer on every engine. See geometry.js.
import { fsin, fcos, fatan2 } from './sim/geometry.js';

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
// More than two ships
// ---------------------------------------------------------------------------
//
// A melee is not a duel with extra ships in it, and three numbers below carry that. They are all
// exactly neutral at two ships, so the duel the game was tuned around is untouched: SHIP_COUNT_*
// index by ship count and read 1 at index 2, and startPositions branches to the mirrored pair.

export const MAX_PLAYERS = 4;

// Incoming damage by how many ships are in the fight. Indexed by ship count, exactly 1 for a duel.
//
// The first version of this was 0.62 and 0.46 -- pulling damage *back*, on the reasoning that a ship
// in a four-way is under fire from three directions. That reasoning is wrong, and tools/melee.js said
// so: a duel ends when one ship sinks, a four-way when three do, so the extra guns are spread over
// three times as much hull to get through. Uncorrected, a three-way ran 24.6s and a four-way 38.9s
// against the duel's 14.2s, with only a third of four-ways reaching a verdict at all before the bell.
//
// So it goes up. These are the values with the least empty air inside the 13-17s band: a three-way
// settles in 17.0s at 38% dry and a four-way in 17.0s at 34%, against the duel's 14.2s at 30%, and
// four-ways go from 34% decisive to 100%. Matching the duel's clock exactly wants [4, 6] and costs
// about fifteen more points of empty air, which is the wrong way round -- the last gameplay pass was
// aimed at dead time, not at length. Length and empty air trade against each other the whole length of
// the sweep, so the clock can always be bought and the only question is the price. Rerun tools/melee.js
// after touching the part table or the economy.
export const SHIP_COUNT_DAMAGE = [1, 1, 1, 2, 3];

// The arena grows with the field, or four ships start inside each other's carronade range and the
// opening is a pile-up rather than an approach.
export const SHIP_COUNT_ARENA = [1, 1, 1, 1.18, 1.32];

export function arenaRadius(shipCount) {
  return ARENA_RADIUS * (SHIP_COUNT_ARENA[shipCount] ?? 1);
}

// Where ships start. Two ships keep the mirrored pair the duel was tuned on, byte for byte; more
// than two stand evenly around a ring, each pointing at the middle, so nobody starts with a free
// broadside on a ship that cannot answer.
export function startPositions(shipCount) {
  if (shipCount === 2) {
    return [
      { x: -START_OFFSET.x, z: START_OFFSET.z, heading: 0 },
      { x: START_OFFSET.x, z: -START_OFFSET.z, heading: Math.PI },
    ];
  }
  const radius = arenaRadius(shipCount) * 0.52;
  const out = [];
  for (let i = 0; i < shipCount; i++) {
    const theta = (Math.PI * 2 * i) / shipCount;
    const x = radius * fsin(theta);
    const z = -radius * fcos(theta);
    // Heading is the direction of travel, and (sin h, -cos h) points at the origin when h faces in.
    out.push({ x, z, heading: fatan2(-x, z) });
  }
  return out;
}

// How often a ship reconsiders who it is fighting, and how much closer a new candidate has to be
// before it is worth turning for. Without the margin a ship between two enemies at the same range
// swaps every few ticks and sails straight between them, guns bearing on nothing.
export const TARGET_RECHECK = 0.6;
export const TARGET_SWITCH_MARGIN = 0.8; // a rival must be inside this fraction of the current range

// ---------------------------------------------------------------------------
// Sailing
// ---------------------------------------------------------------------------

export const BASE_SPEED = 13.5; // units/sec at full sail, running with the wind
export const BASE_TURN = 1.15; // rad/sec at full sail
// The floor under how close two hulls may get, before geometry. It is only a floor now: steering.js
// separate() takes the greater of this and what the two hulls actually need along the line joining
// them, because one distance cannot express a shape that is twice as long as it is wide. This number
// still matters for small hulls, where the ellipses would otherwise let two sloops nearly touch.
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
  return WIND_MIN + ((1 - WIND_MIN) * (fcos(heading - windTo) + 1)) / 2;
}

// The same thing from cached sines and cosines, via cos(h - w) = cos h cos w + sin h sin w. The
// simulation already holds the heading's pair and computes the wind's once per battle, so this runs
// two multiplications instead of a cosine, every ship every tick.
export function windFactorFrom(cosH, sinH, cosW, sinW) {
  return WIND_MIN + ((1 - WIND_MIN) * (cosH * cosW + sinH * sinW + 1)) / 2;
}

// How far out a ship wants to fight, as a fraction of its weapons' reach. Below 1 so guns
// stay comfortably in range rather than flickering at the edge of it.
export const PREFERRED_RANGE_FRACTION = 0.85;

// ---------------------------------------------------------------------------
// Coming alongside
// ---------------------------------------------------------------------------

// Two hulls in contact grind on each other. A crunch every COLLISION_INTERVAL rather than damage
// every tick, because damageCell has a floor of one point per call and sixty calls a second would
// saw a timber in half instantly -- and because a discrete crunch, with its own impact and its own
// sound, is something a player can see happen.
//
// Scaled by how fast the two hulls are actually moving relative to each other, over BASE_SPEED. Two
// ships settling together barely mark each other; two crossing at speed tear cells out.
//
// Measured by paired ablation at these values: a duel is untouched -- 0.1% of winners flip and the mean
// battle moves from 14.33s to 14.28s -- because two ships orbit at their preferred range and rarely
// touch at all. A four-way flips 5.8% and comes in 0.45s shorter, because four ships in one arena
// crowd. That split is the reason to keep it: it is flavour in the duel the part table was tuned
// around, and a real consideration in the melee, where sailing through a crowd should cost something.
export const COLLISION_DAMAGE = 5;
export const COLLISION_INTERVAL = 0.5;

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

// Slow motion over the killing blow, before the result screen. The authority waits this out too,
// so a client is never cut off mid-explosion by a verdict that has already been decided.
export const VERDICT_DELAY = 1.8;

// A ship that has struck her colours goes under. Purely presentation -- the simulation took her out of
// the fight the moment her helm went -- but it has to happen, because in a melee the survivors sail
// straight over the wreck, and two hulls in the same water read as a bug rather than as a wreck being
// passed. She fades into the sea colour rather than turning transparent: the deck plates are one shared
// opaque material across every ship, so darkening toward the water is both cheaper than per-ship
// transparency and a better picture than a ghost.
export const SINK_TIME = 2.8;
export const SINK_DROP = 2.4; // world units the hull settles by

// Comeback money, as a fraction of the round's grant. A flat bonus is worth nothing by
// round 5, which is how you get 3-0 sweeps.
export const LOSER_BONUS_FRACTION = 0.45;

export function loserBonus(roundIndex) {
  return Math.round(ROUNDS[roundIndex].scrap * LOSER_BONUS_FRACTION);
}

// Comeback money by where you came in, 0 for the winner and the full loser's bonus for last. With
// two players this is exactly loserBonus and nothing else, so the duel's economy is untouched; with
// four it spreads, because paying every one of three losers a full bonus would make winning a round
// worth less than losing one.
export function placeBonus(roundIndex, place, playerCount) {
  if (place <= 0 || playerCount < 2) return 0;
  return Math.round(loserBonus(roundIndex) * (place / (playerCount - 1)));
}

export const REROLL_COST = 2;
export const OFFER_SIZE = 5;
export const REPAIR_FRACTION = 0.5;
