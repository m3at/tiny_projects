// The server: static files on one port and the game on the same one, over a WebSocket.
//
//   node server/main.js [port]        default 8123
//
// It serves the game directory as it stands -- no build step, so the files it hands out are the files
// on disk -- and hosts the rooms. There is no database, no session store and no state that survives a
// restart: a match is four sockets and an object, and if the process dies the match is over. That is
// the right shape for a game that lasts five rounds.
//
// The rooms are src/net/room.js, the same module a browser imports to play locally. This file is only
// plumbing: sockets in, seats out, and one interval driving every room's clock.

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { attachWebSocket, frameText } from './ws.js';
import { createRoom } from '../src/net/room.js';
import { makeRng } from '../src/sim/rng.js';
import * as P from '../src/net/protocol.js';

const ROOT = resolve(import.meta.dirname, '..');
const PORT = Number(process.argv[2] || process.env.PORT || 8123);

// One tick of the room clock. The battle simulation runs at 60Hz inside the room whatever this is --
// the room asks the clock what tick it should be on and catches up -- so this is only how often it
// looks. 20ms keeps a stamped input on the wire within a frame of the player pressing the key.
const ROOM_INTERVAL_MS = 20;

// Rooms with nobody in them are swept, or a long-running server accumulates one per abandoned match.
const EMPTY_ROOM_GRACE_MS = 60000;

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

// ---------------------------------------------------------------------------
// static files
// ---------------------------------------------------------------------------

async function serveStatic(req, res) {
  const url = new URL(req.url, 'http://localhost');
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';

  // normalize collapses any ".." before it is joined, so a request cannot climb out of the game
  // directory. The resolved path is checked against the root as well, because one guard for this is
  // one too few.
  const target = join(ROOT, normalize(pathname).replace(/^(\.\.[/\\])+/, ''));
  if (!resolve(target).startsWith(ROOT)) {
    res.writeHead(403).end('forbidden');
    return;
  }

  try {
    const info = await stat(target);
    if (info.isDirectory()) {
      res.writeHead(404).end('not found');
      return;
    }
    const body = await readFile(target);
    res.writeHead(200, {
      'content-type': TYPES[extname(target)] ?? 'application/octet-stream',
      // The game is edited and reloaded constantly and served from disk every time. A cache here is
      // a way to spend an afternoon debugging the previous version of a file.
      'cache-control': 'no-cache',
      'content-length': body.length,
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}

// ---------------------------------------------------------------------------
// rooms
// ---------------------------------------------------------------------------

const rooms = new Map(); // code -> { room, sockets: Map(id -> sock), spectators: Set, emptyAt }
const rng = makeRng((Date.now() ^ 0x5bf03635) >>> 0);

function makeRoom() {
  let code = P.makeCode(rng);
  let guard = 0;
  while (rooms.has(code) && guard++ < 64) code = P.makeCode(rng);

  const entry = {
    code,
    sockets: new Map(),
    spectators: new Set(),
    emptyAt: 0,
    room: null,
  };

  entry.room = createRoom({
    code,
    seed: (rng.next() * 0xffffffff) >>> 0,
    now: Date.now(),
    emit: (target, msg) => deliver(entry, target, msg),
  });

  rooms.set(code, entry);
  log(`room ${code} opened (${rooms.size} open)`);
  return entry;
}

// Serialize once, write the same bytes to every socket. A four-player room broadcasts a sync twice a
// second and an input whenever anyone presses a key; framing that separately per recipient is work
// for nothing.
function deliver(entry, target, msg) {
  const text = JSON.stringify(msg);
  if (target === 'all' || target === 'spectators') {
    const bytes = frameText(text);
    if (target === 'all') {
      for (const sock of entry.sockets.values()) sock.sendRaw(bytes);
    }
    for (const sock of entry.spectators) sock.sendRaw(bytes);
    return;
  }
  // A seat number. The room addresses a player by seat; we hold sockets by connection id.
  const player = entry.room.players[target];
  if (!player) return;
  const sock = entry.sockets.get(player.id);
  if (sock) sock.send(text);
}

function sweepRooms(now) {
  for (const entry of [...rooms.values()]) {
    const occupied = entry.sockets.size > 0 || entry.spectators.size > 0;
    if (occupied) {
      entry.emptyAt = 0;
      continue;
    }
    if (entry.emptyAt === 0) entry.emptyAt = now;
    else if (now - entry.emptyAt > EMPTY_ROOM_GRACE_MS) {
      rooms.delete(entry.code);
      log(`room ${entry.code} closed (${rooms.size} open)`);
    }
  }
}

// ---------------------------------------------------------------------------
// connections
// ---------------------------------------------------------------------------

const log = (text) => console.log(`${new Date().toISOString().slice(11, 19)} ${text}`);

// A join code is four characters from a 32-letter alphabet, so a determined stranger could walk the
// space. One handshake per connection and a short window per address is enough to make that dull;
// the rooms are ephemeral and there is nothing behind them worth taking.
const attempts = new Map(); // address -> { count, until }
const JOIN_WINDOW_MS = 10000;
const JOIN_LIMIT = 12;

function rateLimited(address, now) {
  const seen = attempts.get(address);
  if (!seen || now > seen.until) {
    attempts.set(address, { count: 1, until: now + JOIN_WINDOW_MS });
    return false;
  }
  seen.count++;
  return seen.count > JOIN_LIMIT;
}

function onOpen(sock) {
  let entry = null;
  let id = null;
  let spectating = false;
  let greeted = false;

  sock.on('message', (data) => {
    let msg;
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }

    if (msg.t === P.C.PING) {
      // Echo the client's own clock back untouched alongside ours. It needs both to work out the
      // offset, and it needs its own value to have been altered by nothing.
      sock.send(JSON.stringify({ t: P.S.PONG, c: msg.c, s: Date.now() }));
      return;
    }

    if (msg.t === P.C.HELLO) {
      if (greeted) return;
      greeted = true;
      hello(msg);
      return;
    }

    if (!entry || !id || spectating) return;
    entry.room.command(id, msg);
  });

  function hello(msg) {
    const now = Date.now();
    if (msg.v !== P.PROTOCOL_VERSION) {
      sock.close(P.CLOSE.PROTOCOL, 'version');
      return;
    }
    if (rateLimited(sock.remoteAddress ?? '?', now)) {
      sock.close(P.CLOSE.RATE, 'slow down');
      return;
    }

    const name = P.cleanName(msg.name);

    if (msg.code) {
      entry = rooms.get(String(msg.code).toUpperCase()) ?? null;
      if (!entry) {
        sock.close(P.CLOSE.NO_ROOM, 'no such room');
        return;
      }
    } else if (msg.spectate) {
      sock.close(P.CLOSE.NO_ROOM, 'nothing to watch');
      return;
    } else {
      entry = makeRoom();
    }

    // A reconnect claims its old seat back with the token it was given. This is the whole of session
    // handling: a dropped socket is a stutter rather than a forfeit, and the bot works the
    // ammunition in the meantime.
    if (msg.token && entry.room.players.some((p) => p.id === msg.token)) {
      id = msg.token;
      entry.sockets.set(id, sock);
      const player = entry.room.reconnect(id);
      log(`room ${entry.code}: seat ${player.seat} back`);
      greet(player.seat);
      // Put them straight back into whatever is happening.
      resume(player.seat);
      return;
    }

    if (msg.spectate) {
      spectating = true;
      entry.spectators.add(sock);
      entry.room.spectators = entry.spectators.size;
      greet(null);
      resume(null);
      return;
    }

    id = randomUUID();
    const res = entry.room.join({ id, name });
    if (!res.ok) {
      // A full or already-sailed room is still worth watching.
      spectating = true;
      id = null;
      entry.spectators.add(sock);
      entry.room.spectators = entry.spectators.size;
      greet(null);
      resume(null);
      return;
    }
    entry.sockets.set(id, sock);
    log(`room ${entry.code}: ${name} takes seat ${res.player.seat}`);
    greet(res.player.seat);
  }

  function greet(seat) {
    sock.send(
      JSON.stringify({
        t: P.S.WELCOME,
        v: P.PROTOCOL_VERSION,
        seat: seat ?? -1,
        mySeats: seat === null ? [] : [seat],
        code: entry.code,
        token: id,
        spectating,
        room: entry.room.publicState(),
        serverNow: Date.now(),
      }),
    );
  }

  // Bring a late or returning connection up to the present. Without this a reconnect sits on a blank
  // screen until the next round starts.
  function resume(seat) {
    const room = entry.room;
    if (room.phase === 'build' && seat !== null && room.yards[seat]) {
      sock.send(JSON.stringify(room.buildMessageFor(seat)));
    }
    if (room.phase === 'battle' && room.battle && room.battleMessage) {
      sock.send(
        JSON.stringify({
          ...room.battleMessage,
          serverNow: Date.now(),
          // Every input already stamped, so the replay is rebuilt from scratch and caught up in a few
          // milliseconds rather than the round being lost.
          inputs: room.timeline.inputs,
        }),
      );
    }
    if ((room.phase === 'result' || room.phase === 'over') && room.result) {
      sock.send(JSON.stringify(room.result));
    }
  }

  sock.on('close', () => {
    if (!entry) return;
    if (spectating) {
      entry.spectators.delete(sock);
      entry.room.spectators = entry.spectators.size;
      return;
    }
    if (!id) return;
    entry.sockets.delete(id);
    entry.room.leave(id);
    log(`room ${entry.code}: a captain left`);
  });

  sock.on('error', () => {
    // ws.js has already closed it; 'close' does the bookkeeping.
  });
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

const server = createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, rooms: rooms.size, v: P.PROTOCOL_VERSION }));
    return;
  }
  serveStatic(req, res).catch(() => {
    if (!res.headersSent) res.writeHead(500).end('server error');
  });
});

attachWebSocket(server, { path: '/ws', onOpen });

const timer = setInterval(() => {
  const now = Date.now();
  for (const entry of rooms.values()) {
    try {
      entry.room.update(now);
    } catch (err) {
      // One room throwing must not take the others down with it.
      console.error(`room ${entry.code} update failed`, err);
    }
  }
  sweepRooms(now);
}, ROOM_INTERVAL_MS);

server.listen(PORT, () => {
  console.log(`broadside on http://127.0.0.1:${PORT}/  (protocol v${P.PROTOCOL_VERSION})`);
});

function shutdown() {
  clearInterval(timer);
  for (const entry of rooms.values()) {
    for (const sock of entry.sockets.values()) sock.close(P.CLOSE.SERVER, 'shutting down');
    for (const sock of entry.spectators) sock.close(P.CLOSE.SERVER, 'shutting down');
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1500).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
