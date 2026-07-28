// The wire protocol, in one file that both sides import. The client fetches it over http and the
// server imports it off disk, so there is exactly one definition of every message name and every
// constant that has to match.
//
// JSON, not a binary encoding. The whole traffic of a battle is a handful of messages of a few dozen
// bytes; a hand-rolled binary format would save maybe twenty bytes each and cost a codec to keep in
// step with itself. JSON.stringify is native, and measured against MessagePack at these sizes it is
// faster. The one thing never sent over it is a float the far side has to re-simulate from: seeds,
// ticks and part names cross the wire, positions do not.

export const PROTOCOL_VERSION = 3;

// Application close codes. Never renumber these: a client too old to understand a new message is
// still able to read a close code, which is the only way it can tell the player why it was refused.
export const CLOSE = {
  PROTOCOL: 4001, // version mismatch
  RATE: 4002, // flooding
  IDLE: 4003, // no heartbeat
  NO_ROOM: 4004, // join code unknown, or the room is gone
  FULL: 4005, // room already has its four
  SERVER: 4006, // shutting down
};

// ---------------------------------------------------------------------------
// Client to server
// ---------------------------------------------------------------------------

export const C = {
  HELLO: 'hello', // { v, name, code?, token?, spectate? }
  READY: 'ready', // { on }
  ADDBOT: 'addbot', // fill an empty berth with a bot, in the lobby only
  REMATCH: 'rematch', // same roster, new seed, back to the lobby
  PLACE: 'place', // { key, part }
  REMOVE: 'remove', // { key }
  REFIT: 'refit',
  REROLL: 'reroll',
  LOCK: 'lock',
  AMMO: 'ammo', // { ammo }
  CONTINUE: 'continue', // dismissed an overlay; the room advances when everyone has
  PING: 'ping', // { c } -- the client's own clock, echoed back untouched
  RENAME: 'rename', // { name }
};

// ---------------------------------------------------------------------------
// Server to client
// ---------------------------------------------------------------------------

export const S = {
  WELCOME: 'welcome', // { v, seat, code, token, room }
  ROOM: 'room', // { room } -- roster, scores, phase; sent whenever any of it moves
  ROUND: 'round', // { round, hullIndex, windTo, buildMs, scrap, bonuses, design, offer, intel }
  PURSE: 'purse', // { scrap, design? } -- an ack, or a correction after a refused edit
  OFFER: 'offer', // { offer, scrap }
  LOCKED: 'locked', // { seat }
  BATTLE: 'battle', // { seed, hullIndex, windTo, designs, startAt, tick }
  AMMO: 'ammo', // { seat, ammo, tick } -- an input, stamped onto a tick
  SYNC: 'sync', // { tick, sum } -- authoritative tick and state checksum
  RESULT: 'result', // { winner, reason, scores, placing, lines, over }
  PONG: 'pong', // { c, s }
  DENY: 'deny', // { why } -- a refused action, with something to show the player
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// How far ahead of the current tick an input is stamped. Two ticks is 33ms of slack for the server
// to get the message out before the tick it belongs to comes due on a client that is running
// exactly in step. It costs nothing a player can feel: switching ammunition already costs a reload
// of 1.3s, and battle.setAmmo takes the later of the gun's deadline and now, so a toggle landing a
// frame late lands in the same reload.
export const INPUT_DELAY_TICKS = 2;

// How often the server states its tick and checksum. Every half second: often enough that a client
// which has drifted resyncs before anyone reads the panels, rare enough to be free.
export const SYNC_EVERY_TICKS = 30;

// How far behind the server's tick a client plays the battle. This is the whole latency strategy:
// the client is a replay, so it stays this far in the past and every input for a tick has already
// arrived by the time that tick is simulated. Measured RTT jitter moves it between these bounds.
export const RENDER_DELAY_MIN_MS = 55;
export const RENDER_DELAY_MAX_MS = 260;

// Clock sync. Eight probes on joining, then one every ten seconds. The offset is taken from the
// single lowest-RTT sample rather than an average: a sample delayed by queueing carries that delay
// into the offset asymmetrically, so averaging imports the error instead of cancelling it.
export const PING_BURST = 8;
export const PING_BURST_GAP_MS = 220;
export const PING_IDLE_MS = 10000;

// A seat is held this long for a player whose socket dropped, so a flaky connection is a stutter and
// not a forfeit. Their ammunition is worked by the bot while they are away.
export const SEAT_HOLD_MS = 30000;

// Overlays advance on their own if somebody has wandered off. A room cannot wait for an absent
// player for ever, and the alternative -- no timeout -- is a match hostage to one closed laptop.
export const INTRO_MS = 6000;
export const RESULT_MS = 11000;
// Grace after the build deadline before the room locks a player in on whatever they have placed.
export const LOCK_GRACE_MS = 900;

// ---------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------

// No O, 0, I, 1: a join code gets read out loud.
export const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
export const CODE_LENGTH = 4;
export const MAX_NAME = 14;

export function makeCode(rng) {
  let out = '';
  for (let i = 0; i < CODE_LENGTH; i++) out += CODE_ALPHABET[rng.int(0, CODE_ALPHABET.length - 1)];
  return out;
}

// Names arrive from strangers and end up in the DOM of three other people's browsers, so: printable
// ASCII only, no markup characters, trimmed, and capped.
//
// The angle brackets and the ampersand are the ones that matter. The result screen builds its lines as
// HTML -- it wants the player's name in bold next to what became of their ship -- and a name of
// `<img src=x onerror=...>` is then script running in every other player's page. Escaping at the point
// of use is the real defence and hud.js does it; this is the other half, because a name has no
// legitimate need for a tag in it and defence in depth is cheap here.
export function cleanName(raw, fallback = 'Captain') {
  const text = String(raw ?? '')
    .replace(/[^\x20-\x7e]/g, '')
    .replace(/[<>&"'`]/g, '')
    .trim()
    .slice(0, MAX_NAME);
  return text || fallback;
}
