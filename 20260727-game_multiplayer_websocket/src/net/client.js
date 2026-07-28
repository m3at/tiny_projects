// The client half of the protocol: what this browser believes about the room, and the battle it
// draws.
//
// The battle on screen is a replay. The authority runs the real one and stamps every input with a
// tick; this reproduces it from the seed, the designs and that stamped input stream, and deliberately
// stays a fraction of a second behind so that an input for tick N has always arrived before tick N is
// simulated. Nothing about a ship's position is ever sent -- there would be no point, since both
// sides compute it from the same numbers.
//
// A local game skips all of that. The authority is in the same page, so the battle it is running is
// the battle this draws: no replica, no delay, no checksum, nothing to diverge. That is what
// `transport.localRoom` means below, and it is the reason the two modes share this file instead of
// only resembling each other.
//
// Divergence is handled rather than prevented. sim/geometry.js makes the arithmetic bit-identical
// across engines, but that is a very high probability rather than a proof, so the server states a
// checksum twice a second and a mismatch rebuilds the replay from tick zero and replays the whole
// input stream into it. At three thousand battles a second that costs a few milliseconds, which is
// why this can afford to be the honest fix rather than a papered-over snap.

import { createBattle } from '../sim/battle.js';
import { createTimeline } from '../sim/timeline.js';
import { checksum } from '../sim/checksum.js';
import { cloneDesign } from '../sim/ship.js';
import { createYard } from '../shipyard.js';
import { TICK } from '../config.js';
import * as P from './protocol.js';

// How much of the battle one frame may catch up on. A tab that was hidden for a while comes back
// behind, and simulating four seconds costs about two milliseconds, so it can be generous.
const MAX_CATCHUP_TICKS = 240;

export function createClient({ transport }) {
  const listeners = new Map();

  const state = {
    status: 'connecting',
    statusText: '',
    room: null,
    // Which seats this browser may act for: one over a socket, all of the human ones on a hot seat.
    mySeats: [],
    seat: 0, // whose panel the build phase is showing
    phase: 'menu',
    round: 0,
    hullIndex: 0,
    windTo: 0,
    buildUntil: 0,
    buildTime: 0,
    scrap: 0,
    offer: [],
    intel: [],
    bonuses: [],
    scraps: [],
    result: null,
    hint: '',
  };

  // What the authority reports about itself, and what we make of the wire. Shown by the dev overlay
  // and by tools/netcheck.js; the only thing the game reads is `delayMs`.
  const net = {
    rtt: 0,
    jitter: 0,
    offset: 0,
    // What the wire measures, and what the replay actually uses. The second is never below the
    // first and grows when an input arrives late; see widenDelay.
    measuredDelayMs: 0,
    // What lateness has taught us this session. It only grows, and it survives a round boundary:
    // the wire is the same wire next round, so relearning it battle by battle would mean paying for
    // the lesson five times.
    learnedDelayMs: 0,
    delayMs: P.RENDER_DELAY_MIN_MS,
    tickLag: 0,
    desyncs: 0,
    resyncs: 0,
    lateInputs: 0,
    checked: 0,
  };

  let yard = null;
  let replica = null;
  let timeline = null;
  let battleInit = null;
  let battleStartAt = 0;
  const marks = new Map();
  const syncs = [];

  function emit(event, payload) {
    for (const fn of listeners.get(event) ?? []) fn(payload);
  }

  function on(event, fn) {
    if (!listeners.has(event)) listeners.set(event, []);
    listeners.get(event).push(fn);
  }

  const localRoom = () => transport.localRoom ?? null;

  // ---------------------------------------------------------------------------
  // incoming
  // ---------------------------------------------------------------------------

  function handle(msg) {
    switch (msg.t) {
      case P.S.WELCOME:
        state.mySeats = msg.mySeats ?? [msg.seat];
        state.seat = state.mySeats[0] ?? 0;
        state.room = msg.room;
        state.phase = msg.room.phase;
        emit('room', state.room);
        return;

      case P.S.ROOM: {
        const was = state.phase;
        state.room = msg.room;
        state.phase = msg.room.phase;
        if (msg.room.phase === 'lobby') state.result = null;
        emit('room', state.room);
        if (was !== state.phase) emit('phase', state.phase);
        return;
      }

      case P.S.ROUND:
        state.round = msg.round;
        state.hullIndex = msg.hullIndex;
        state.windTo = msg.windTo;
        if (msg.phase === 'intro') {
          state.scraps = msg.scrap ?? [];
          state.bonuses = msg.bonuses ?? [];
          state.buildTime = msg.buildTime ?? 0;
          state.phase = 'intro';
          emit('intro', msg);
          return;
        }
        // A build phase. Everything private to this seat arrives here and nowhere else.
        state.phase = 'build';
        state.seat = msg.seat ?? state.seat;
        state.buildUntil = msg.until;
        state.scrap = msg.scrap ?? 0;
        state.offer = msg.offer ?? [];
        state.intel = msg.intel ?? [];
        yard =
          msg.design === undefined
            ? null
            : createYard({
                design: cloneDesign(msg.design),
                hullIndex: msg.hullIndex,
                scrap: msg.scrap,
                offer: state.offer,
              });
        emit('build', { ...msg, yard });
        return;

      case P.S.PURSE:
        // The purse is the authority's. A design comes with it only when something was refused, in
        // which case our optimistic copy is thrown away rather than patched.
        if (yard) {
          yard.scrap = msg.scrap;
          // Replaced in place, not swapped: the ship view holds a reference to this object and
          // rebuilds its instanced layers from it. Handing it a new object leaves the deck on screen
          // showing the design that was just thrown away.
          if (msg.design) {
            const parts = yard.design.parts;
            for (const key of Object.keys(parts)) delete parts[key];
            for (const key in msg.design.parts) parts[key] = { ...msg.design.parts[key] };
          }
        }
        state.scrap = msg.scrap;
        emit('yard', { reset: !!msg.design });
        return;

      case P.S.OFFER:
        state.offer = msg.offer;
        state.scrap = msg.scrap;
        if (yard) {
          yard.setOffer(msg.offer);
          yard.scrap = msg.scrap;
        }
        emit('yard', { reset: false });
        return;

      case P.S.LOCKED:
        emit('locked', msg.seat);
        return;

      case P.S.BATTLE:
        startBattle(msg);
        return;

      case P.S.AMMO:
        if (timeline) {
          // Stamped for a tick this replay has already run. The replay is now wrong by construction,
          // and there is no need to wait for a checksum to say so: add() has put the input in its
          // proper place in the stream, so rebuilding from tick zero applies it where it belongs.
          // Widen the delay too, or the next one is late for the same reason.
          const late = msg.tick < (replica?.tickCount ?? 0);
          timeline.add({ tick: msg.tick, seat: msg.seat, ammo: msg.ammo });
          if (late) {
            net.lateInputs++;
            widenDelay();
            resync();
          }
        }
        emit('ammo', msg);
        return;

      case P.S.SYNC:
        if (msg.serverNow !== undefined) transport.observeServerClock?.(msg.serverNow);
        syncs.push({ tick: msg.tick, sum: msg.sum });
        if (syncs.length > 24) syncs.shift();
        compareSyncs();
        return;

      case P.S.RESULT:
        state.result = msg;
        state.phase = msg.over ? 'over' : 'result';
        emit('result', msg);
        return;

      case P.S.DENY:
        state.hint = msg.why;
        emit('deny', msg.why);
        return;

      default:
        return;
    }
  }

  // ---------------------------------------------------------------------------
  // the battle
  // ---------------------------------------------------------------------------

  function startBattle(msg) {
    state.phase = 'battle';
    state.hullIndex = msg.hullIndex;
    state.windTo = msg.windTo;
    battleStartAt = msg.startAt;
    net.delayMs = Math.max(P.RENDER_DELAY_MIN_MS, net.measuredDelayMs, net.learnedDelayMs);
    marks.clear();
    syncs.length = 0;
    if (msg.serverNow !== undefined) transport.observeServerClock?.(msg.serverNow);

    if (localRoom()) {
      // The authority is in this page. Nothing to replay.
      replica = null;
      timeline = null;
      battleInit = null;
    } else {
      battleInit = {
        seed: msg.seed,
        hullIndex: msg.hullIndex,
        windTo: msg.windTo,
        designs: msg.designs,
      };
      buildReplica();
      // A reconnect mid-battle arrives with everything already stamped and catches up by simulating.
      for (const input of msg.inputs ?? []) timeline.add(input);
    }
    emit('battle', { hullIndex: msg.hullIndex, windTo: msg.windTo, designs: msg.designs });
  }

  // The render delay is estimated from round trips, and an estimate can be wrong -- a clock offset
  // taken from a queued sample, a route that got worse, a laptop that went to sleep. An input arriving
  // for a tick already played is the direct evidence of being wrong, so the delay grows on it. It
  // never shrinks during a battle: a battle is fifteen seconds, and hunting for a tighter delay inside
  // one buys nothing a player could notice and risks doing this again.
  function widenDelay() {
    net.delayMs = Math.min(P.RENDER_DELAY_MAX_MS, net.delayMs + 40);
    net.learnedDelayMs = Math.max(net.learnedDelayMs, net.delayMs);
  }

  function buildReplica() {
    replica = createBattle({
      designs: battleInit.designs.map(cloneDesign),
      hullIndex: battleInit.hullIndex,
      seed: battleInit.seed,
      windTo: battleInit.windTo,
    });
    timeline = createTimeline(replica);
  }

  // Throw the replay away and build it again from the seed and the whole input stream. This is the
  // only repair, because it is the only one that is guaranteed to land on the authority's state
  // rather than near it.
  function resync() {
    if (!battleInit) return;
    const inputs = timeline ? timeline.inputs.slice() : [];
    const target = replica.tickCount;
    buildReplica();
    for (const input of inputs) timeline.add(input);
    marks.clear();
    timeline.runToMarks(target, P.SYNC_EVERY_TICKS, (tick, b) => marks.set(tick, checksum(b)));
    net.resyncs++;
    emit('resync', { tick: target });
  }

  function compareSyncs() {
    if (!replica) return;
    for (let i = syncs.length - 1; i >= 0; i--) {
      const sync = syncs[i];
      const mine = marks.get(sync.tick);
      if (mine === undefined) continue;
      syncs.splice(i, 1);
      net.checked++;
      if (mine === sync.sum) continue;
      net.desyncs++;
      resync();
      return;
    }
  }

  function advanceReplica() {
    if (!replica || replica.over) return;
    const serverMs = transport.serverNow();
    // The one line that is the latency strategy: play the battle net.delayMs in the past.
    const delay = Math.max(net.delayMs, net.measuredDelayMs, net.learnedDelayMs);
    const want = Math.floor((serverMs - delay - battleStartAt) / (TICK * 1000));
    net.tickLag = want - replica.tickCount;
    if (want <= replica.tickCount) return;
    const target = Math.min(want, replica.tickCount + MAX_CATCHUP_TICKS);
    timeline.runToMarks(target, P.SYNC_EVERY_TICKS, (tick, b) => {
      marks.set(tick, checksum(b));
      if (marks.size > 40) marks.delete(marks.keys().next().value);
    });
    compareSyncs();
  }

  // ---------------------------------------------------------------------------
  // outgoing
  // ---------------------------------------------------------------------------

  // A build command is applied here first and sent second, so a click lands on the deck at once
  // rather than a round trip later. The authority applies the same rules to the same command, so the
  // two agree; when they do not, it says so and sends the design back, and PURSE above replaces ours
  // wholesale. Optimism, with a correction that cannot be argued with.
  function build(kind, extra = {}) {
    if (!yard) return { ok: false, why: 'Not building.' };
    let res = { ok: true };
    if (kind === P.C.PLACE) res = yard.place(extra.key, extra.part);
    else if (kind === P.C.REMOVE) res = yard.remove(extra.key);
    else if (kind === P.C.REFIT) res = yard.refit();
    if (!res.ok) return res;
    state.scrap = yard.scrap;
    send({ t: kind, ...extra });
    return res;
  }

  function send(msg) {
    // On a hot seat one connection speaks for several captains, so every command says which.
    transport.send(state.mySeats.length > 1 ? { seat: state.seat, ...msg } : msg);
  }

  const client = {
    state,
    net,
    on,

    get battle() {
      const room = localRoom();
      return room ? room.battle : replica;
    },
    get yard() {
      return yard;
    },
    get isLocal() {
      return !!localRoom();
    },
    // In a local game the authority's clock is the one the battle is on; over a socket we only have
    // an estimate of it. Either way this is "how far into the battle are we".
    get battleTime() {
      const battle = client.battle;
      return battle ? battle.time : 0;
    },

    controls(seat) {
      return state.mySeats.includes(seat);
    },

    // The build panel is showing this seat. Hot seat only; over a socket it is always yours.
    focus(seat) {
      if (state.mySeats.includes(seat)) state.seat = seat;
    },

    place: (key, part) => build(P.C.PLACE, { key, part }),
    remove: (key) => build(P.C.REMOVE, { key }),
    refit: () => build(P.C.REFIT),
    reroll: () => send({ t: P.C.REROLL }),
    lock: () => send({ t: P.C.LOCK }),
    ready: (on) => send({ t: P.C.READY, on }),
    addBot: () => send({ t: P.C.ADDBOT }),
    rematch: () => send({ t: P.C.REMATCH }),
    proceed: () => send({ t: P.C.CONTINUE }),

    setAmmo(seat, ammo) {
      if (!state.mySeats.includes(seat)) return false;
      transport.send({ t: P.C.AMMO, ammo, seat });
      return true;
    },

    update(dt) {
      transport.update(dt);
      if (localRoom()) return;
      advanceReplica();
    },

    // Seconds left in the build phase, on the authority's clock rather than ours.
    buildLeft() {
      if (!state.buildUntil) return null;
      return Math.max(0, (state.buildUntil - transport.serverNow()) / 1000);
    },

    disconnect() {
      transport.close?.();
    },
  };

  transport.onMessage = handle;
  transport.onStatus = (status, text = '') => {
    state.status = status;
    state.statusText = text;
    emit('status', { status, text });
  };
  transport.net = net;

  return client;
}
