// Match bookkeeping: scores, purses, which hull we are on, what each player last saw of the
// others. Pure state with no DOM and no renderer, which is what lets the server own it and leaves
// main.js doing only presentation and flow. Two to four players; a two-player match behaves exactly
// as it did when that was the only kind.

import { createDesign, fitDesignToHull, cloneDesign, designStats } from './sim/ship.js';
import { HULLS } from './data/hulls.js';
import { BUYABLE } from './data/parts.js';
import { ROUNDS, POINTS_TO_WIN, MAX_PLAYERS, OFFER_SIZE, placeBonus } from './config.js';
import { makeRng, hashSeed } from './sim/rng.js';

export function createMatch(seed, fromRound = 0, playerCount = 2) {
  const n = Math.max(2, Math.min(MAX_PLAYERS, playerCount));
  const match = {
    seed,
    playerCount: n,
    roundIndex: fromRound,
    scores: new Array(n).fill(0),
    designs: Array.from({ length: n }, () => createDesign()),
    scrap: new Array(n).fill(0),
    lastSeen: new Array(n).fill(null), // what the others saw of each player after the last battle
    // Where each player came in last round, 0 for the winner. Drives comeback money, and is null
    // before the first battle.
    lastPlace: null,
    windTo: 0,
  };
  // Starting at a later round has to hand over the scrap the skipped rounds would have
  // granted, or the big hulls turn up nearly empty.
  for (let r = 0; r < fromRound; r++) {
    for (let i = 0; i < n; i++) match.scrap[i] += ROUNDS[r].scrap;
  }
  return match;
}

export const hullIndexOf = (match) => ROUNDS[match.roundIndex].hull;
export const hullOf = (match) => HULLS[hullIndexOf(match)];
export const roundOf = (match) => ROUNDS[match.roundIndex];

// Roll the wind, move every ship into this round's hull, and pay out. Returns the comeback money
// each player was given, in seat order, so the round intro can name it.
export function beginRound(match) {
  const hullIndex = hullIndexOf(match);
  match.windTo = makeRng(hashSeed(match.seed, match.roundIndex, 77)).next() * Math.PI * 2;
  const bonuses = new Array(match.playerCount).fill(0);
  for (let i = 0; i < match.playerCount; i++) {
    fitDesignToHull(match.designs[i], hullIndex);
    if (match.lastPlace !== null) {
      bonuses[i] = placeBonus(match.roundIndex, match.lastPlace[i], match.playerCount);
    }
    match.scrap[i] += roundOf(match).scrap + bonuses[i];
  }
  return { hullIndex, bonuses };
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

// What the shop shows. The offer is a set of part *types*, and you may buy as many of each as you can
// afford: filling 38 cells one card at a time would be tedious, and the interesting luck is in which
// types you are shown, not how many.
//
// Drawn by the authority and sent to the one player it belongs to. Neither the seed nor the offer of
// anyone else ever reaches a client -- hashSeed is a couple of multiplications and trivially
// invertible, so handing out any seed derived from the match seed hands out the match seed, and that
// decides which beam the battle turns to.
export function makeOffer(rng, design, hullIndex) {
  const stats = designStats(design, hullIndex);
  // Guarantees that stop a hand being unplayable: something cheap to plug holes with, powder if you
  // have none, and hands if you have none.
  const guaranteed = ['timber'];
  if (stats.magazines === 0) guaranteed.push('magazine');
  if (stats.crewSupply === 0) guaranteed.push('crew');

  const pool = BUYABLE.filter((id) => !guaranteed.includes(id));
  for (let i = pool.length - 1; i > 0; i--) {
    const j = rng.int(0, i);
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const picked = pool.slice(0, Math.max(0, OFFER_SIZE - guaranteed.length));
  const all = [...guaranteed, ...picked];
  return BUYABLE.filter((id) => all.includes(id));
}

// Award the point, record the placing that comeback money is paid on, and let everyone see what they
// were up against. `placing` is battle.placing: seat indices, best first.
export function recordResult(match, winner, placing) {
  if (winner !== null) match.scores[winner]++;

  const order = placing ?? match.designs.map((_, i) => i);
  const place = new Array(match.playerCount).fill(0);
  for (let p = 0; p < order.length; p++) place[order[p]] = p;
  if (winner === null) {
    // Nobody won, so nobody was beaten: shifting everyone up one collapses the drawn leaders onto
    // the winner's place and pays them nothing, which is what a drawn duel has always done.
    for (let i = 0; i < place.length; i++) place[i] = Math.max(0, place[i] - 1);
  }
  match.lastPlace = place;

  const hullIndex = hullIndexOf(match);
  for (let i = 0; i < match.playerCount; i++) {
    match.lastSeen[i] = { design: cloneDesign(match.designs[i]), hullIndex };
  }
}

// Everything `player` is allowed to know about the others: their ships as they stood at the end of
// the last battle. Nothing about what they are building right now, which is the whole reason this
// function exists rather than the panel reading match.designs.
export function intelFor(match, player) {
  const out = [];
  for (let i = 0; i < match.playerCount; i++) {
    if (i === player || match.lastSeen[i] === null) continue;
    out.push({ player: i, design: match.lastSeen[i].design, hullIndex: match.lastSeen[i].hullIndex });
  }
  return out;
}

export function isMatchOver(match) {
  return (
    match.scores.some((s) => s >= POINTS_TO_WIN) || match.roundIndex >= ROUNDS.length - 1
  );
}

// Who took the day, or null if the top score is shared.
export function matchWinner(match) {
  const best = Math.max(...match.scores);
  const leaders = match.scores.reduce((a, s, i) => (s === best ? [...a, i] : a), []);
  return leaders.length === 1 ? leaders[0] : null;
}
