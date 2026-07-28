// The authority. One room is one match: a roster of two to four seats, the phase clock, the purses,
// the offers, and the battle itself.
//
// This is the only place the game is actually decided. Clients replay the battle to draw it, but the
// room's copy is the one that awards the point, so a client that drifts is a cosmetic problem and
// never a lost match. Nothing in here touches the DOM, the renderer, a socket or a wall clock: time
// arrives as an argument to update(), and messages leave through emit(). That is what lets the same
// file be the server for four browsers, the in-process authority for a hot-seat game on one laptop,
// and a headless harness that plays a whole match in ten milliseconds.
//
// Trust boundary: a client may ask for anything, and every ask is checked here. Placement legality,
// the purse, what is on offer, whose turn it is to be locked in -- all of it is decided from the
// room's own state. The client's copy is a mirror kept for the feel of an instant click.

import {
  createMatch,
  beginRound,
  battleSeed,
  offerSeed,
  hullIndexOf,
  roundOf,
  recordResult,
  intelFor,
  isMatchOver,
  matchWinner,
  roundSummary,
  makeOffer,
} from '../match.js';
import { createBattle } from '../sim/battle.js';
import { createTimeline } from '../sim/timeline.js';
import { checksum } from '../sim/checksum.js';
import { structureFraction, cloneDesign } from '../sim/ship.js';
import { makeRng } from '../sim/rng.js';
import { createYard } from '../shipyard.js';
import { autoBuild, randomProfile } from '../autobuild.js';
import { makeBot } from '../bot.js';
import { TICK, MAX_PLAYERS, POINTS_TO_WIN, VERDICT_DELAY } from '../config.js';
import * as P from './protocol.js';

// The most ticks one update() will run. A server that was blocked for a second should catch up over
// the next few updates, not simulate a second of battle inside one of them and hand every client a
// tick number it cannot reach.
const MAX_CATCHUP_TICKS = 20;

// `hotseat` is one keyboard and several captains taking turns, which is how this game was played
// before it had a socket. It changes exactly one thing: build phases run one seat at a time, each
// with its own countdown, instead of everybody fitting out at once. Everything else -- the offers,
// the validation, the battle, the economy -- is the same authority doing the same work, which is the
// point of running it in-process rather than writing a second game loop for local play.
export function createRoom({
  code = 'LOCAL',
  seed,
  now = 0,
  emit = () => {},
  hotseat = false,
  keepEffects = false,
} = {}) {
  const room = {
    code,
    seed,
    hotseat,
    keepEffects,
    phase: 'lobby',
    buildSeat: 0,
    bonuses: [],
    desired: [],
    players: [],
    spectators: 0,
    match: null,
    battle: null,
    timeline: null,
    // Where the battle's tick 0 sits on the room's clock. Every input stamp and every sync message
    // is measured from here.
    battleStartAt: 0,
    hullIndex: 0,
    verdictAt: 0,
    phaseUntil: 0,
    buildEndsAt: 0,
    yards: [],
    bot: null,
    botSeatList: [],
    result: null,
  };

  let clock = now;
  const rng = makeRng((seed ^ 0x9e3779b9) >>> 0);

  // ---------------------------------------------------------------------------
  // roster
  // ---------------------------------------------------------------------------

  const seatOf = (id) => room.players.find((p) => p.id === id) ?? null;

  // Which seats the room works the ammunition for: the bots, and anyone whose socket has dropped.
  // A disconnected player's ship keeps fighting rather than standing there loaded with the wrong
  // shot, which is the difference between a stutter and a forfeit.
  //
  // The list is live rather than a snapshot, because a socket drops mid-battle far more often than
  // between rounds: makeBot holds this exact array, so pushing a seat into it puts the bot on that
  // wheel within its next reaction interval instead of at the start of the next round.
  const botSeats = () => room.players.filter((p) => p.bot || !p.connected).map((p) => p.seat);

  function refreshBotSeats() {
    const wanted = botSeats();
    room.botSeatList.length = 0;
    for (const seat of wanted) room.botSeatList.push(seat);
  }

  function publicState() {
    return {
      code: room.code,
      phase: room.phase,
      round: room.match ? room.match.roundIndex : 0,
      hullIndex: room.match ? hullIndexOf(room.match) : 0,
      windTo: room.match ? room.match.windTo : 0,
      pointsToWin: POINTS_TO_WIN,
      spectators: room.spectators,
      players: room.players.map((p) => ({
        seat: p.seat,
        name: p.name,
        bot: p.bot,
        connected: p.connected,
        ready: p.ready,
        locked: p.locked,
        score: room.match ? room.match.scores[p.seat] : 0,
      })),
    };
  }

  function announce() {
    emit('all', { t: P.S.ROOM, room: publicState() });
  }

  // A seat is claimed for the whole match. Joining mid-match is a spectator's business, not a
  // player's: there is no ship to hand over and no purse to invent.
  function join({ id, name, bot = false }) {
    if (room.phase !== 'lobby') return { ok: false, why: 'That match has already sailed.' };
    if (room.players.length >= MAX_PLAYERS) return { ok: false, why: 'That room is full.' };
    const player = {
      id,
      seat: room.players.length,
      name,
      bot,
      connected: true,
      ready: bot,
      locked: false,
      continued: false,
      dropAt: 0,
    };
    room.players.push(player);
    announce();
    return { ok: true, player };
  }

  function addBot() {
    return join({ id: `bot${room.players.length}`, name: `Bot ${room.players.length + 1}`, bot: true });
  }

  // Leaving the lobby frees the seat and renumbers whoever is behind you, which is only safe before
  // a match exists. Once it does, the seat is held and the bot takes the wheel.
  function leave(id) {
    const player = seatOf(id);
    if (!player) return;
    if (room.phase === 'lobby') {
      room.players = room.players.filter((p) => p !== player);
      room.players.forEach((p, i) => (p.seat = i));
    } else {
      player.connected = false;
      player.dropAt = clock;
      refreshBotSeats();
    }
    announce();
  }

  function reconnect(id) {
    const player = seatOf(id);
    if (!player) return null;
    player.connected = true;
    player.dropAt = 0;
    refreshBotSeats();
    announce();
    return player;
  }

  function setReady(id, on) {
    const player = seatOf(id);
    if (!player || room.phase !== 'lobby') return;
    player.ready = !!on;
    announce();
    maybeStart();
  }

  function maybeStart() {
    if (room.phase !== 'lobby') return;
    const crew = room.players;
    if (crew.length < 2) return;
    if (!crew.every((p) => p.ready)) return;
    start();
  }

  // ---------------------------------------------------------------------------
  // match flow
  // ---------------------------------------------------------------------------

  function start(fromRound = 0) {
    room.match = createMatch(room.seed, fromRound, room.players.length);
    for (const p of room.players) {
      p.ready = false;
      p.locked = false;
    }
    startRound();
  }

  function startRound() {
    const { bonuses } = beginRound(room.match);
    room.hullIndex = hullIndexOf(room.match);
    room.bonuses = bonuses;
    room.phase = 'intro';
    room.phaseUntil = clock + P.INTRO_MS;
    for (const p of room.players) p.continued = p.bot;
    announce();
    emit('all', {
      t: P.S.ROUND,
      phase: 'intro',
      round: room.match.roundIndex,
      hullIndex: room.hullIndex,
      windTo: room.match.windTo,
      scrap: room.match.scrap.slice(),
      bonuses: bonuses.slice(),
      buildTime: roundOf(room.match).buildTime,
      until: room.phaseUntil,
    });
  }

  function startBuild() {
    room.phase = 'build';
    room.yards = room.players.map((p) => {
      const offerRng = makeRng(offerSeed(room.match, p.seat));
      const design = room.match.designs[p.seat];
      const yard = createYard({
        design,
        hullIndex: room.hullIndex,
        scrap: room.match.scrap[p.seat],
        offer: makeOffer(offerRng, design, room.hullIndex),
      });
      yard.rng = offerRng;
      return yard;
    });

    for (const p of room.players) {
      p.locked = false;
      // A bot spends its purse the moment the phase opens. Nothing is watching it think.
      if (p.bot) {
        const yard = room.yards[p.seat];
        const profile = randomProfile(makeRng(offerSeed(room.match, p.seat) ^ 0x5150), room.hullIndex);
        yard.scrap = autoBuild(yard.design, room.hullIndex, yard.scrap, profile);
        p.locked = true;
      }
    }

    if (room.hotseat) return openSeatBuild();

    room.buildEndsAt = clock + roundOf(room.match).buildTime * 1000;
    announce();
    for (const p of room.players) {
      if (p.bot) continue;
      emit(p.seat, buildMessageFor(p.seat));
    }
    // Spectators see the countdown but nothing that is anybody's hand.
    emit('spectators', {
      t: P.S.ROUND,
      phase: 'build',
      round: room.match.roundIndex,
      hullIndex: room.hullIndex,
      windTo: room.match.windTo,
      until: room.buildEndsAt,
    });
    checkAllLocked();
  }

  // Hot seat only: hand the keyboard to the next captain who has not locked in, and give them the
  // full countdown to themselves.
  function openSeatBuild() {
    const next = room.players.find((p) => !p.locked);
    if (!next) return startBattle();
    room.buildSeat = next.seat;
    room.buildEndsAt = clock + roundOf(room.match).buildTime * 1000;
    announce();
    emit(next.seat, buildMessageFor(next.seat));
  }

  function buildMessageFor(seat) {
    const yard = room.yards[seat];
    return {
      t: P.S.ROUND,
      phase: 'build',
      seat,
      round: room.match.roundIndex,
      hullIndex: room.hullIndex,
      windTo: room.match.windTo,
      until: room.buildEndsAt,
      scrap: yard.scrap,
      design: cloneDesign(yard.design),
      offer: yard.offer.slice(),
      // Only what the last battle showed. Never what anyone is building now.
      intel: intelFor(room.match, seat),
    };
  }

  function checkAllLocked() {
    if (room.phase !== 'build') return;
    if (!room.players.every((p) => p.locked)) return;
    startBattle();
  }

  function lockSeat(seat) {
    const player = room.players[seat];
    if (!player || player.locked) return;
    player.locked = true;
    room.match.scrap[seat] = room.yards[seat].scrap;
    emit('all', { t: P.S.LOCKED, seat });
    if (room.hotseat) return openSeatBuild();
    announce();
    checkAllLocked();
  }

  function startBattle() {
    for (let i = 0; i < room.players.length; i++) {
      if (!room.players[i].locked) room.match.scrap[i] = room.yards[i].scrap;
      room.players[i].locked = true;
    }

    const seed = battleSeed(room.match);
    room.battle = createBattle({
      designs: room.match.designs,
      hullIndex: room.hullIndex,
      seed,
      windTo: room.match.windTo,
    });
    room.timeline = createTimeline(room.battle);
    room.phase = 'battle';
    room.battleStartAt = clock;
    room.verdictAt = 0;
    room.desired = room.players.map(() => 'round');
    refreshBotSeats();
    room.bot = makeBot(room.battle, {
      seats: room.botSeatList,
      apply: (seat, ammo) => queueAmmo(seat, ammo),
    });

    // Kept, so a player who reconnects mid-battle can be handed the same message and rebuild the
    // same replay. With the stamped inputs alongside it, catching up is a few milliseconds of
    // simulation rather than a lost round.
    room.battleMessage = {
      t: P.S.BATTLE,
      seed,
      hullIndex: room.hullIndex,
      windTo: room.match.windTo,
      // The designs become public exactly here, and not one moment earlier. So does the seed, which
      // is what decides which beam the fight turns to.
      designs: room.match.designs.map(cloneDesign),
      startAt: room.battleStartAt,
    };
    announce();
    emit('all', { ...room.battleMessage, serverNow: clock });
  }

  const serverTick = () => Math.floor((clock - room.battleStartAt) / (TICK * 1000));

  // Stamp an input onto a tick in the near future and tell everyone. Far enough ahead that a client
  // running exactly in step still has it before that tick comes due; near enough that the player
  // cannot feel it.
  function queueAmmo(seat, ammo) {
    if (room.phase !== 'battle' || !room.battle || room.battle.over) return;
    if (ammo !== 'round' && ammo !== 'grape') return;
    if (room.desired[seat] === ammo) return; // no-op, and the bot re-states its choice every 250ms
    room.desired[seat] = ammo;
    const tick = Math.max(room.battle.tickCount, serverTick()) + P.INPUT_DELAY_TICKS;
    const input = { tick, seat, ammo };
    room.timeline.add(input);
    emit('all', { t: P.S.AMMO, ...input });
  }

  function stepBattle() {
    const battle = room.battle;
    if (battle.over) {
      if (room.verdictAt === 0) room.verdictAt = clock;
      // Let the killing blow land on every client before the result screen replaces it.
      if (clock - room.verdictAt >= VERDICT_DELAY * 1000) finishBattle();
      return;
    }

    if (room.bot) room.bot.update((clock - room.lastClock) / 1000);

    const target = Math.min(serverTick(), battle.tickCount + MAX_CATCHUP_TICKS);
    room.timeline.runToMarks(target, P.SYNC_EVERY_TICKS, (tick, b) => {
      emit('all', { t: P.S.SYNC, tick, sum: checksum(b), serverNow: clock });
    });
    // Nothing on the authority draws or listens, and nothing in sim/ clears this. The in-process
    // authority of a local game is the exception: there the renderer is reading this very battle.
    if (!room.keepEffects) battle.effects.length = 0;
  }

  function finishBattle() {
    const battle = room.battle;
    const summaries = battle.ships.map((ship) => ({
      seat: ship.index,
      structure: structureFraction(ship),
      ...roundSummary(ship),
    }));
    battle.finish();
    recordResult(room.match, battle.winner, battle.placing);

    const over = isMatchOver(room.match);
    room.result = {
      t: P.S.RESULT,
      winner: battle.winner,
      reason: battle.reason,
      placing: battle.placing,
      scores: room.match.scores.slice(),
      round: room.match.roundIndex,
      log: battle.log.slice(-4).map((l) => ({ t: l.t, text: l.text })),
      summaries,
      over,
      matchWinner: over ? matchWinner(room.match) : null,
      until: clock + P.RESULT_MS,
    };
    room.battle = null;
    room.timeline = null;
    room.bot = null;
    room.phase = over ? 'over' : 'result';
    room.phaseUntil = clock + P.RESULT_MS;
    for (const p of room.players) p.continued = p.bot;
    announce();
    emit('all', room.result);
  }

  function nextRound() {
    if (room.phase !== 'result') return;
    room.match.roundIndex++;
    startRound();
  }

  // Everyone still at the keyboard has dismissed the overlay. A player who has wandered off is not
  // allowed to hold the match up: the phase clock expires regardless.
  function allContinued() {
    // One keyboard means one person to press the button, whatever the roster says.
    if (room.hotseat) return room.players.some((p) => p.continued);
    return room.players.every((p) => p.continued || !p.connected);
  }

  // ---------------------------------------------------------------------------
  // commands
  // ---------------------------------------------------------------------------

  function deny(seat, why) {
    emit(seat, { t: P.S.DENY, why });
  }

  // Answers with the room's own idea of the purse and, when a command was refused, the design as the
  // room holds it -- so a client whose optimistic copy has run ahead is put straight in one message.
  function ackYard(seat, { correct = false } = {}) {
    const yard = room.yards[seat];
    emit(seat, {
      t: P.S.PURSE,
      scrap: yard.scrap,
      design: correct ? cloneDesign(yard.design) : undefined,
    });
  }

  function command(id, msg) {
    const owner = seatOf(id);
    if (!owner) return;
    // One keyboard means one connection acting for several captains, so the interface says which one
    // is at the wheel. Over a socket the seat comes from the connection and a client saying otherwise
    // is ignored -- that is the whole trust boundary, in one line.
    const seat =
      room.hotseat && Number.isInteger(msg.seat) && room.players[msg.seat] ? msg.seat : owner.seat;

    switch (msg.t) {
      case P.C.READY:
        setReady(id, msg.on);
        return;

      case P.C.REMATCH:
        rematch();
        return;

      case P.C.ADDBOT:
        // Anyone already seated may fill an empty berth. A room of strangers is not worth an
        // ownership model; the worst a stranger can do is add a fourth ship.
        if (room.phase === 'lobby') {
          addBot();
          maybeStart();
        }
        return;

      case P.C.CONTINUE:
        owner.continued = true;
        if (room.phase === 'intro' && allContinued()) startBuild();
        else if (room.phase === 'result' && allContinued()) nextRound();
        return;

      case P.C.AMMO:
        queueAmmo(seat, msg.ammo);
        return;

      case P.C.PLACE:
      case P.C.REMOVE:
      case P.C.REFIT:
      case P.C.REROLL:
      case P.C.LOCK:
        buildCommand(seat, msg);
        return;

      default:
        return;
    }
  }

  function buildCommand(seat, msg) {
    const player = room.players[seat];
    if (room.phase !== 'build' || player.locked) return deny(seat, 'The build phase is closed.');
    if (room.hotseat && seat !== room.buildSeat) return deny(seat, 'It is not your turn to build.');
    const yard = room.yards[seat];

    if (msg.t === P.C.LOCK) {
      lockSeat(seat);
      return;
    }

    let res;
    if (msg.t === P.C.PLACE) res = yard.place(msg.key, msg.part);
    else if (msg.t === P.C.REMOVE) res = yard.remove(msg.key);
    else if (msg.t === P.C.REFIT) res = yard.refit();
    else if (msg.t === P.C.REROLL) {
      res = yard.payForReroll();
      if (res.ok) {
        yard.setOffer(makeOffer(yard.rng, yard.design, room.hullIndex));
        emit(seat, { t: P.S.OFFER, offer: yard.offer.slice(), scrap: yard.scrap });
        return;
      }
    }

    if (!res.ok) {
      deny(seat, res.why);
      // A refusal means the two copies may already disagree, so send the whole hand back.
      ackYard(seat, { correct: true });
      return;
    }
    ackYard(seat);
  }

  // ---------------------------------------------------------------------------
  // the clock
  // ---------------------------------------------------------------------------

  room.lastClock = clock;

  function update(nowMs) {
    clock = nowMs;
    switch (room.phase) {
      case 'lobby':
        // Release a seat nobody came back for, so a room is not held open by a closed tab.
        for (const p of [...room.players]) {
          if (!p.connected && p.dropAt && clock - p.dropAt > P.SEAT_HOLD_MS) leave(p.id);
        }
        break;
      case 'intro':
        if (clock >= room.phaseUntil || allContinued()) startBuild();
        break;
      case 'build':
        // The deadline is the promise, and the grace is for the last packet in flight.
        if (clock > room.buildEndsAt + P.LOCK_GRACE_MS) {
          if (room.hotseat) lockSeat(room.buildSeat);
          else for (const p of room.players) if (!p.locked) lockSeat(p.seat);
        }
        break;
      case 'battle':
        stepBattle();
        break;
      case 'result':
        if (clock >= room.phaseUntil || allContinued()) nextRound();
        break;
      default:
        break;
    }
    room.lastClock = clock;
  }

  // A fresh match with the same roster. The seed moves, or everyone replays the same five rounds.
  function rematch() {
    if (room.phase !== 'over' && room.phase !== 'lobby') return;
    room.seed = (rng.next() * 0xffffffff) >>> 0;
    room.battle = null;
    room.result = null;
    room.match = null;
    room.phase = 'lobby';
    for (const p of room.players) p.ready = p.bot;
    announce();
  }

  Object.assign(room, {
    join,
    addBot,
    leave,
    reconnect,
    command,
    setReady,
    update,
    start,
    rematch,
    publicState,
    buildMessageFor,
    get clock() {
      return clock;
    },
  });
  return room;
}
