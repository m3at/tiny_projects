// The rules of fitting out a ship: what a part costs, where it may go, what a refit repairs, what
// comes back when you break something up. Pure -- no DOM, no renderer, no rng.
//
// This used to live inside ui/build.js, mixed in with the panels that draw it. It is out here
// because the authority has to apply exactly these rules to a command that arrived over a socket,
// and the interface has to apply exactly these rules to a click, and a second implementation of
// "does this player have the scrap for this" is a second implementation of the game's economy. The
// panel now asks for a change and draws the result; the yard decides.
//
// Every method returns either { ok: true, ... } or { ok: false, why } where `why` is the sentence
// shown to the player. The interface says it in the hint line; the server sends it back as a deny.

import { PARTS, repairCost } from './data/parts.js';
import { REROLL_COST } from './config.js';
import { placementError } from './sim/ship.js';

export function createYard({ design, hullIndex, scrap, offer = [] }) {
  // Parts bought during this build phase refund in full; anything paid for in an earlier round is
  // already spent, so breaking it up returns nothing. Without this a player could sell last round's
  // ship back at full price every round.
  const placedThisPhase = new Set();

  const yard = {
    design,
    hullIndex,
    scrap,
    offer,

    setOffer(next) {
      yard.offer = next;
    },

    // Worst damage first, so a partial purse buys back the most broken parts.
    damaged() {
      return Object.entries(design.parts)
        .filter(([, slot]) => slot.hp < PARTS[slot.id].hp)
        .sort((a, b) => a[1].hp / PARTS[a[1].id].hp - b[1].hp / PARTS[b[1].id].hp);
    },

    refitCost() {
      return yard.damaged().reduce((sum, [, slot]) => sum + repairCost(slot.id), 0);
    },

    place(key, partId) {
      if (!yard.offer.includes(partId)) return { ok: false, why: 'That part is not on offer.' };
      const part = PARTS[partId];
      if (part.cost > yard.scrap) {
        return { ok: false, why: `Not enough scrap for a ${part.name.toLowerCase()}.` };
      }
      const comma = String(key).indexOf(',');
      if (comma < 0) return { ok: false, why: 'Not part of the hull' };
      const dx = +key.slice(0, comma);
      const dz = +key.slice(comma + 1);
      if (!Number.isFinite(dx) || !Number.isFinite(dz)) {
        return { ok: false, why: 'Not part of the hull' };
      }
      const err = placementError(design, hullIndex, dx, dz, partId);
      if (err) return { ok: false, why: err };

      design.parts[key] = { id: partId, hp: part.hp };
      placedThisPhase.add(key);
      yard.scrap -= part.cost;
      return { ok: true };
    },

    remove(key) {
      const slot = design.parts[key];
      if (!slot) return { ok: false, why: 'Nothing there to remove.' };
      if (PARTS[slot.id].fixed) return { ok: false, why: 'The helm stays where it is.' };
      delete design.parts[key];
      if (placedThisPhase.has(key)) {
        placedThisPhase.delete(key);
        yard.scrap += PARTS[slot.id].cost;
        return { ok: true, refund: PARTS[slot.id].cost };
      }
      return { ok: true, refund: 0 };
    },

    // One button repairs the whole ship, worst first, for as far as the purse reaches. Clicking
    // damaged cells one at a time was the same decision wrapped in busywork.
    refit() {
      let repaired = 0;
      for (const [, slot] of yard.damaged()) {
        const cost = repairCost(slot.id);
        if (yard.scrap < cost) break;
        yard.scrap -= cost;
        slot.hp = PARTS[slot.id].hp;
        repaired++;
      }
      if (repaired === 0) return { ok: false, why: 'Not enough scrap to repair anything.' };
      return { ok: true, repaired };
    },

    // The new offer comes from outside, because it is drawn from the match seed and the client is
    // never given that seed: the seed also decides which beam the battle turns to, and a player who
    // knew that in advance would build a sheltered flank and win every time.
    payForReroll() {
      if (yard.scrap < REROLL_COST) return { ok: false, why: 'Not enough scrap to reroll.' };
      yard.scrap -= REROLL_COST;
      return { ok: true };
    },
  };

  return yard;
}
