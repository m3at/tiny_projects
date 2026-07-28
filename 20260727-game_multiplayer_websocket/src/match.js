// Match bookkeeping: scores, purses, which hull we are on, what each player last saw of
// the other. Pure state with no DOM and no renderer, so the eventual server can own it and
// main.js is left doing only presentation and flow.

import { createDesign, fitDesignToHull, cloneDesign } from './sim/ship.js';
import { HULLS } from './data/hulls.js';
import { ROUNDS, POINTS_TO_WIN, loserBonus } from './config.js';
import { makeRng, hashSeed } from './sim/rng.js';

export function createMatch(seed, fromRound = 0) {
  const match = {
    seed,
    roundIndex: fromRound,
    scores: [0, 0],
    designs: [createDesign(), createDesign()],
    scrap: [0, 0],
    lastSeen: [null, null], // what each player saw of the other after the last battle
    lastLoser: null,
    windTo: 0,
  };
  // Starting at a later round has to hand over the scrap the skipped rounds would have
  // granted, or the big hulls turn up nearly empty.
  for (let r = 0; r < fromRound; r++) {
    match.scrap[0] += ROUNDS[r].scrap;
    match.scrap[1] += ROUNDS[r].scrap;
  }
  return match;
}

export const hullIndexOf = (match) => ROUNDS[match.roundIndex].hull;
export const hullOf = (match) => HULLS[hullIndexOf(match)];
export const roundOf = (match) => ROUNDS[match.roundIndex];

// Roll the wind, move both ships into this round's hull, and pay out.
export function beginRound(match) {
  const hullIndex = hullIndexOf(match);
  match.windTo = makeRng(hashSeed(match.seed, match.roundIndex, 77)).next() * Math.PI * 2;
  const bonus = match.lastLoser === null ? 0 : loserBonus(match.roundIndex);
  for (let i = 0; i < 2; i++) {
    fitDesignToHull(match.designs[i], hullIndex);
    match.scrap[i] += roundOf(match).scrap + (match.lastLoser === i ? bonus : 0);
  }
  return { hullIndex, bonus };
}

export function battleSeed(match) {
  return hashSeed(match.seed, match.roundIndex, 4242);
}

// What became of a ship, for the result screen. A running log says what happened; this says which
// decision was wrong, which is the thing a player watching a battle they cannot steer actually needs.
// Reads only the battle's own state, so it stays pure.
export function roundSummary(ship) {
  let mastsHad = 0;
  let mastsLeft = 0;
  let handsHad = 0;
  for (const cell of ship.cells) {
    handsHad += cell.crewSupply;
    if (cell.id === 'mast') {
      mastsHad++;
      if (cell.alive) mastsLeft++;
    }
  }
  let gunsLeft = 0;
  let firing = 0;
  for (const gun of ship.guns) {
    if (gun.cell.alive) gunsLeft++;
    if (gun.manned) firing++;
  }
  return {
    cellsHad: ship.cells.length,
    cellsLeft: ship.aliveCells,
    gunsHad: ship.guns.length,
    gunsLeft,
    firing,
    mastsHad,
    mastsLeft,
    handsHad,
    hands: ship.crew,
    powder: ship.magazines,
  };
}

export function offerSeed(match, player) {
  return hashSeed(match.seed, match.roundIndex, player, 913);
}

// Award the point and let both players see what they were up against.
export function recordResult(match, winner) {
  if (winner === null) {
    match.lastLoser = null;
  } else {
    match.scores[winner]++;
    match.lastLoser = 1 - winner;
  }
  const hullIndex = hullIndexOf(match);
  for (let i = 0; i < 2; i++) {
    match.lastSeen[i] = { design: cloneDesign(match.designs[i]), hullIndex };
  }
}

export function intelFor(match, player) {
  const seen = match.lastSeen[1 - player];
  return seen ? { design: seen.design, hullIndex: seen.hullIndex } : null;
}

export function isMatchOver(match) {
  return (
    match.scores[0] >= POINTS_TO_WIN ||
    match.scores[1] >= POINTS_TO_WIN ||
    match.roundIndex >= ROUNDS.length - 1
  );
}
