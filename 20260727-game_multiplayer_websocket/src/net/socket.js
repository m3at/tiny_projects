// The transport for a game over the wire: one WebSocket, an estimate of the server's clock, and a
// reconnect that gets the same seat back.
//
// Clock estimation is the only part with any subtlety. The client has to answer "what tick is the
// server on" to know which tick to play, and it has nothing but round trips to answer it with. The
// method is NTP's: send our clock, get it back alongside theirs, and take the offset from the sample
// with the shortest round trip rather than an average of them. A sample delayed by a queue somewhere
// carries that delay into the offset asymmetrically, so averaging imports the error instead of
// cancelling it -- the fastest exchange is the least contaminated one, and that is the one to keep.
//
// Two smaller things that matter as much:
//
//   The offset is slewed, never stepped. A better sample arriving mid-battle would otherwise jump the
//   playback clock, and a jump of 20ms is a visible stutter in a battle being watched.
//
//   Every message the server stamps with its own clock is a free one-way sample, and while it cannot
//   tell us the offset it can put a floor under it: a message stamped S that arrives at local time L
//   proves the offset is at least S - L, because it cannot have arrived before it was sent. An offset
//   estimated too high is the dangerous direction -- it makes the client play ahead of the inputs it
//   has been sent -- so the floor is worth keeping even though the ceiling is not.

import * as P from './protocol.js';

const clamp = (lo, x, hi) => Math.max(lo, Math.min(hi, x));

export function createSocketTransport({ url, name, code = null, spectate = false }) {
  const transport = {
    localRoom: null, // there is nothing local about this one
    onMessage: () => {},
    onStatus: () => {},
    net: null,
  };

  let ws = null;
  let token = null;
  let joinCode = code;
  let closedForGood = false;
  let attempt = 0;
  let retryTimer = 0;

  // Clock estimation state.
  let offset = 0; // server ms minus local ms
  let target = 0; // where the offset is slewing to
  let offsetFloor = -Infinity;
  let haveOffset = false;
  let bestRtt = Infinity;
  const rtts = [];
  let pingsLeft = 0;
  let nextPingAt = 0;
  let localClock = 0;

  const nowLocal = () => performance.now();

  function setStatus(status, text) {
    transport.onStatus(status, text);
  }

  function connect() {
    setStatus('connecting', attempt > 0 ? `Reconnecting (${attempt})` : 'Connecting');
    ws = new WebSocket(url);

    ws.onopen = () => {
      attempt = 0;
      ws.send(
        JSON.stringify({
          t: P.C.HELLO,
          v: P.PROTOCOL_VERSION,
          name,
          code: joinCode,
          token,
          spectate,
        }),
      );
      // A burst on joining, then one every ten seconds. Eight is enough to find a clean sample on a
      // wire with ordinary jitter and takes under two seconds.
      pingsLeft = P.PING_BURST;
      nextPingAt = 0;
      bestRtt = Infinity;
      rtts.length = 0;
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return; // a frame we cannot read is a frame we cannot act on
      }
      if (msg.t === P.S.PONG) return onPong(msg);
      if (msg.t === P.S.WELCOME) {
        token = msg.token ?? token;
        joinCode = msg.code ?? joinCode;
        setStatus('open', '');
      }
      if (msg.serverNow !== undefined) observeServerClock(msg.serverNow);
      transport.onMessage(msg);
    };

    ws.onclose = (ev) => {
      ws = null;
      if (closedForGood) return;
      // A refusal is final: retrying a wrong join code or a version mismatch only repeats it.
      const fatal =
        ev.code === P.CLOSE.PROTOCOL ||
        ev.code === P.CLOSE.NO_ROOM ||
        ev.code === P.CLOSE.FULL;
      if (fatal) {
        closedForGood = true;
        setStatus('refused', closeReason(ev.code));
        return;
      }
      scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose always follows, and it is the one carrying a code worth reporting.
    };
  }

  function closeReason(code) {
    if (code === P.CLOSE.PROTOCOL) return 'This page is a different version from the server. Reload.';
    if (code === P.CLOSE.NO_ROOM) return 'No room with that code.';
    if (code === P.CLOSE.FULL) return 'That room already has four captains.';
    if (code === P.CLOSE.SERVER) return 'The server is going down.';
    return 'Disconnected.';
  }

  // Exponential backoff with full jitter: without the jitter, four browsers dropped by the same
  // network blip all come back at the same instant, which is the moment the server is least able to
  // take them.
  function scheduleReconnect() {
    attempt++;
    const capped = Math.min(30000, 500 * 2 ** Math.min(attempt, 6));
    const wait = capped / 2 + Math.random() * (capped / 2);
    setStatus('waiting', `Connection lost. Retrying in ${Math.round(wait / 1000)}s`);
    retryTimer = localClock + wait;
  }

  function onPong(msg) {
    const local = nowLocal();
    const rtt = local - msg.c;
    rtts.push(rtt);
    if (rtts.length > 32) rtts.shift();
    // offset = server - local, from the midpoint of the exchange.
    const sample = msg.s - (msg.c + local) / 2;
    if (rtt < bestRtt) {
      bestRtt = rtt;
      target = sample;
      if (!haveOffset) {
        offset = sample; // the first sample is not slewed, there is nothing to slew from
        haveOffset = true;
      }
    }
    if (transport.net) {
      const sorted = [...rtts].sort((a, b) => a - b);
      const min = sorted[0];
      const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
      transport.net.rtt = Math.round(min);
      transport.net.jitter = Math.round(p95 - min);
      // Half the shortest round trip is the one-way estimate; the jitter is what has to be absorbed
      // on top of it, plus a frame of slack.
      // What the wire says the delay should be. The client owns the delay actually in use and may
      // only widen it during a battle, so this cannot undo an adaptation made because an input
      // turned up late.
      transport.net.measuredDelayMs = clamp(
        P.RENDER_DELAY_MIN_MS,
        Math.round(min / 2 + (p95 - min) + 25),
        P.RENDER_DELAY_MAX_MS,
      );
      transport.net.offset = Math.round(offset);
    }
  }

  function observeServerClock(serverMs) {
    const floor = serverMs - nowLocal();
    if (floor > offsetFloor) offsetFloor = floor;
    if (!haveOffset) {
      offset = floor;
      target = floor;
      haveOffset = true;
    }
  }

  transport.observeServerClock = observeServerClock;

  transport.serverNow = () => Math.max(nowLocal() + offset, nowLocal() + offsetFloor);

  transport.send = (msg) => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg));
  };

  transport.update = (dt) => {
    localClock += dt * 1000;

    if (!ws && !closedForGood && retryTimer && localClock >= retryTimer) {
      retryTimer = 0;
      connect();
    }
    if (!ws || ws.readyState !== 1) return;

    const local = nowLocal();
    if (local >= nextPingAt) {
      const burst = pingsLeft > 0;
      if (burst) pingsLeft--;
      nextPingAt = local + (burst ? P.PING_BURST_GAP_MS : P.PING_IDLE_MS);
      transport.send({ t: P.C.PING, c: local });
    }

    // Slew: at most two milliseconds a frame, so a correction is invisible.
    if (haveOffset && offset !== target) {
      const step = clamp(-2, target - offset, 2);
      offset += step;
      if (Math.abs(target - offset) < 0.05) offset = target;
    }
  };

  transport.close = () => {
    closedForGood = true;
    if (ws) ws.close(1000);
    ws = null;
  };

  transport.start = () => connect();

  return transport;
}
