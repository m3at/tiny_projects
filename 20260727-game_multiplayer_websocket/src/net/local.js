// The transport for a game with no network: the authority runs in this page.
//
// It is the same createRoom the server runs, handed a clock that this file advances and an emit that
// hands messages straight to the client. So a hot-seat game is not a second implementation of the
// game loop that happens to resemble the networked one -- it is the networked one, with the wire
// taken out. Every rule, every purse, every offer and every verdict comes from the same code that
// decides an online match, which is why playtesting locally is worth anything at all.
//
// It also carries the one thing the networked path cannot: `localRoom`, which tells the client not to
// bother replaying a battle it can simply watch.

import { createRoom } from './room.js';
import * as P from './protocol.js';

export function createLocalTransport({ seed, players = 2, bots = 0, hotseat = true, speed = 1 }) {
  // The room's clock is ours to run, which is what lets ?x=3 play a match at three times speed and a
  // headless harness play one as fast as it can loop.
  let clock = 0;

  const transport = {
    localRoom: null,
    onMessage: () => {},
    onStatus: () => {},

    send(msg) {
      room.command('local', msg);
    },

    // There is no wire and therefore no clock to estimate: the authority's clock is right here.
    serverNow() {
      return clock;
    },

    update(dt) {
      clock += dt * 1000 * speed;
      room.update(clock);
    },
  };

  const room = createRoom({
    code: 'LOCAL',
    seed,
    now: clock,
    hotseat,
    // The renderer is reading the authority's own battle, so the authority must not drain the effects
    // out from under it.
    keepEffects: true,
    emit: (target, msg) => {
      // One connection, so everything addressed to a seat or to everyone is for us. Only the
      // spectator feed is not, and there are no spectators in a local game.
      if (target === 'spectators') return;
      transport.onMessage(msg);
    },
  });

  transport.localRoom = room;

  const humanSeats = [];
  for (let i = 0; i < players; i++) {
    room.join({ id: 'local', name: `Player ${i + 1}` });
    humanSeats.push(i);
  }
  for (let i = 0; i < bots; i++) room.addBot();

  // Every human seat belongs to this keyboard. The room knows them all by the same connection id,
  // which is exactly what a hot seat is.
  transport.start = ({ fromRound = 0 } = {}) => {
    transport.onStatus('open');
    transport.onMessage({
      t: P.S.WELCOME,
      v: P.PROTOCOL_VERSION,
      seat: 0,
      mySeats: humanSeats,
      code: room.code,
      room: room.publicState(),
    });
    room.start(fromRound);
  };

  return transport;
}
