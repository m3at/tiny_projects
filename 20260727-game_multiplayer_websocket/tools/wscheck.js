// Does server/ws.js actually speak RFC 6455? Node's built-in WebSocket covers the happy paths, but
// a well-behaved client cannot produce the frames that break a hand-written parser, so the
// adversarial half of this writes the bytes onto a raw socket itself: unmasked frames, 64-bit
// lengths, three frames in one write, one frame across three writes.
//
// The last check is the process itself. If ws.js leaks a timer or a socket this tool hangs rather
// than failing, so a hang is the failure: nothing here calls process.exit on success.

import { createServer } from 'node:http';
import { connect } from 'node:net';
import { attachWebSocket, frameText } from '../server/ws.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let passed = 0, failed = 0;
const ok = (name, good, note = '') => {
  console.log(`  ${good ? 'ok  ' : 'FAIL'} ${name}${note ? `  ${note}` : ''}`);
  good ? passed++ : failed++;
};

// An echo endpoint with three magic words, which is all the application logic the checks need.
// Text comes back as text; binary comes back as text describing itself, which proves the opcode
// arrived intact without a second client code path.
function serve(limits) {
  const http = createServer((_req, res) => res.writeHead(404).end());
  const seen = { opens: 0, closes: 0 };
  const live = new Set();
  attachWebSocket(http, {
    path: '/ws',
    limits,
    onOpen(sock) {
      seen.opens++;
      live.add(sock);
      sock.on('message', (data, binary) => {
        if (binary) return sock.send(`binary:${data.length}:${data[0]}`);
        if (data === 'broadcast') {
          // The point of sendRaw: frame once, write the same bytes to everyone.
          const bytes = frameText(`fanout:${live.size}`);
          for (const s of live) s.sendRaw(bytes);
          return;
        }
        if (data === 'flood') {
          // Shout at a client that is not reading. sock.alive going false is the server deciding
          // the write queue has grown past what it will hold.
          const block = 'z'.repeat(60 * 1024);
          for (let i = 0; i < 400 && sock.alive; i++) sock.send(block);
          return;
        }
        if (data === 'bye') {
          sock.close(4321, 'as you were');
          sock.send('should never arrive');   // must be dropped, not thrown and not sent
          return;
        }
        sock.send(data);
      });
      sock.on('close', () => {
        seen.closes++;
        live.delete(sock);
      });
      sock.on('error', () => {});
    },
  });
  return new Promise((resolve) => {
    http.listen(0, '127.0.0.1', () =>
      resolve({ http, seen, port: http.address().port }));
  });
}

// ---------------------------------------------------------------- raw client

// Masked client frame. The mask is required in this direction, so every raw write below uses this.
function clientFrame(opcode, payload, { fin = true, mask = true } = {}) {
  const body = Buffer.isBuffer(payload) ? payload : Buffer.from(String(payload), 'utf8');
  const len = body.length;
  const lenBytes = len < 126 ? 0 : len < 65536 ? 2 : 8;
  const out = Buffer.allocUnsafe(2 + lenBytes + (mask ? 4 : 0) + len);
  out[0] = (fin ? 0x80 : 0) | opcode;
  out[1] = (mask ? 0x80 : 0) | (lenBytes === 0 ? len : lenBytes === 2 ? 126 : 127);
  if (lenBytes === 2) out.writeUInt16BE(len, 2);
  if (lenBytes === 8) {
    out.writeUInt32BE(0, 2);
    out.writeUInt32BE(len, 6);
  }
  let at = 2 + lenBytes;
  const key = Buffer.from([0x37, 0xfa, 0x21, 0x3d]);
  if (mask) {
    key.copy(out, at);
    at += 4;
  }
  body.copy(out, at);
  if (mask) for (let i = 0; i < len; i++) out[at + i] ^= key[i & 3];
  return out;
}

// Just enough of a parser to read what the server sends back. Same accumulate-then-consume shape as
// the server's, because the server splits its writes too once a payload gets large.
function rawClient(port, { path = '/ws' } = {}) {
  const socket = connect(port, '127.0.0.1');
  const state = {
    socket,
    frames: [],          // { opcode, payload }
    status: null,        // http status line, once seen
    headers: '',
    closed: false,
  };
  let buf = Buffer.alloc(0);
  let handshook = false;

  socket.on('error', () => {});
  socket.on('close', () => { state.closed = true; });
  socket.on('data', (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    if (!handshook) {
      const end = buf.indexOf('\r\n\r\n');
      if (end < 0) return;
      state.headers = buf.subarray(0, end).toString('latin1');
      state.status = Number(state.headers.split(' ')[1]);
      buf = buf.subarray(end + 4);
      handshook = true;
    }
    for (;;) {
      if (buf.length < 2) return;
      let len = buf[1] & 0x7f;
      let off = 2;
      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2);
        off = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        len = buf.readUInt32BE(6);
        off = 10;
      }
      if (buf.length < off + len) return;
      state.frames.push({ opcode: buf[0] & 0x0f, payload: buf.subarray(off, off + len) });
      buf = buf.subarray(off + len);
    }
  });

  state.handshake = async (version = '13') => {
    await new Promise((r) => socket.once('connect', r));
    socket.write(
      `GET ${path} HTTP/1.1\r\n` +
        `Host: 127.0.0.1:${port}\r\n` +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n' +
        `Sec-WebSocket-Version: ${version}\r\n\r\n`,
    );
    return state.await(() => state.status !== null || state.closed);
  };
  // Poll rather than promise per frame: several checks care about what did *not* arrive as well.
  state.await = async (cond, limitMs = 2000) => {
    const deadline = Date.now() + limitMs;
    while (Date.now() < deadline) {
      if (cond()) return true;
      await sleep(5);
    }
    return false;
  };
  state.frame = (opcode) => state.frames.find((f) => f.opcode === opcode);
  state.closeCode = () => {
    const f = state.frame(0x8);
    return f && f.payload.length >= 2 ? f.payload.readUInt16BE(0) : null;
  };
  return state;
}

// ------------------------------------------------------------------- checks

async function handshakeChecks(port) {
  console.log('handshake');

  const good = rawClient(port);
  await good.handshake();
  const accept = /sec-websocket-accept: (.+)/i.exec(good.headers)?.[1].trim();
  // The magic GUID hashed with this fixed key is the one worked example in the RFC.
  ok('accept 101 with the RFC 6455 example digest',
    good.status === 101 && accept === 's3pPLMBiTxaQ9kYGzzhZRbK+xOo=', accept || '');
  good.socket.destroy();

  const oldVersion = rawClient(port);
  await oldVersion.handshake('8');
  ok('reject Sec-WebSocket-Version: 8 with 400',
    oldVersion.status === 400 && /Sec-WebSocket-Version: 13/.test(oldVersion.headers),
    String(oldVersion.status));
  oldVersion.socket.destroy();

  const wrongPath = rawClient(port, { path: '/nope' });
  await wrongPath.handshake();
  ok('reject upgrade on the wrong path with 400', wrongPath.status === 400,
    String(wrongPath.status));
  wrongPath.socket.destroy();
}

// One helper for the built-in-WebSocket paths: connect, send, collect, close.
function client(port) {
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const got = [];
  const state = { ws, got, closeCode: null };
  ws.binaryType = 'arraybuffer';
  ws.addEventListener('message', (ev) => got.push(ev.data));
  ws.addEventListener('close', (ev) => { state.closeCode = ev.code; });
  ws.addEventListener('error', () => {});
  state.open = () => new Promise((r, j) => {
    ws.addEventListener('open', r, { once: true });
    ws.addEventListener('close', () => j(new Error('closed before open')), { once: true });
  });
  state.await = async (cond, limitMs = 4000) => {
    const deadline = Date.now() + limitMs;
    while (Date.now() < deadline) {
      if (cond()) return true;
      await sleep(5);
    }
    return false;
  };
  return state;
}

async function echoChecks(port) {
  console.log('echo');

  const c = client(port);
  await c.open();

  c.ws.send('hello');
  await c.await(() => c.got.length >= 1);
  ok('text round trip', c.got[0] === 'hello', JSON.stringify(c.got[0]));

  // Multi-byte on purpose: a length taken in characters rather than bytes passes an ASCII test and
  // truncates this one, and a mask applied per character corrupts it.
  const wide = 'broadside — 舷側斉射 — ⚓️ 100% ✅';
  c.ws.send(wide);
  await c.await(() => c.got.length >= 2);
  ok('utf-8 round trip byte-exact', c.got[1] === wide,
    `${Buffer.byteLength(wide)} bytes, ${wide.length} chars`);

  c.ws.send(new Uint8Array([0x42, 1, 2, 3]));
  await c.await(() => c.got.length >= 3);
  ok('binary frame arrives as bytes', c.got[2] === 'binary:4:66', String(c.got[2]));

  c.ws.close(1000, 'done');
  await c.await(() => c.closeCode !== null);
  ok('clean close echoes 1000', c.closeCode === 1000, String(c.closeCode));
}

async function sendChecks(port) {
  console.log('send');

  // Two connections, one frame, two writes. If sendRaw mangled the shared buffer -- by masking it
  // in place, say -- the second client would get rubbish.
  const a = client(port), b = client(port);
  await a.open();
  await b.open();
  a.ws.send('broadcast');
  await a.await(() => a.got.length >= 1 && b.got.length >= 1);
  ok('sendRaw fans one serialised frame out to both sockets',
    a.got[0] === 'fanout:2' && b.got[0] === 'fanout:2', `${a.got[0]} / ${b.got[0]}`);
  b.ws.close();

  // The server closes with an application code and then tries to send. The close must carry the
  // code, and the send after it must be a no-op rather than a frame after the close frame.
  a.ws.send('bye');
  await a.await(() => a.closeCode !== null);
  await sleep(50);
  ok('server-initiated close carries its code', a.closeCode === 4321, String(a.closeCode));
  ok('send after close is a no-op', a.got.length === 1, `${a.got.length} message(s)`);

  const bytes = frameText('fanout');
  ok('frameText is an unmasked final text frame',
    bytes[0] === 0x81 && (bytes[1] & 0x80) === 0 && bytes[1] === 6, `${bytes.length} bytes`);

  // A client that asks for a lot and then stops reading. The write queue is the same unbounded
  // buffer as an oversized inbound message, so it gets the same treatment.
  const deaf = rawClient(port);
  await deaf.handshake();
  deaf.socket.pause();
  deaf.socket.write(clientFrame(0x1, 'flood'));
  await sleep(300);
  deaf.socket.resume();
  const shut = await deaf.await(() => deaf.closeCode() !== null, 4000);
  ok('client that stops reading is closed 1013', shut && deaf.closeCode() === 1013,
    String(deaf.closeCode()));
  deaf.socket.destroy();
}

async function sizeChecks(port, bigPort) {
  console.log('size');

  const c = client(port);
  await c.open();
  const sixty = 'x'.repeat(60 * 1024);
  c.ws.send(sixty);
  await c.await(() => c.got.length >= 1);
  ok('60KB round trip (16-bit length both ways)', c.got[0] === sixty,
    `${c.got[0]?.length ?? 0} back`);
  c.ws.close();

  // The same message against a server whose limit allows it, so the 64-bit form is exercised in
  // both directions rather than only being rejected.
  const big = client(bigPort);
  await big.open();
  const seventy = 'y'.repeat(70 * 1024);
  big.ws.send(seventy);
  await big.await(() => big.got.length >= 1);
  ok('70KB round trip under a raised limit (64-bit length)', big.got[0] === seventy,
    `${big.got[0]?.length ?? 0} back`);
  big.ws.close();

  const over = client(port);
  await over.open();
  over.ws.send(seventy);
  await over.await(() => over.closeCode !== null);
  ok('70KB past maxMessageBytes closes 1009', over.closeCode === 1009, String(over.closeCode));
}

async function framingChecks(port) {
  console.log('framing');

  // Three complete frames in one write. A parser that handles one frame per read loses two.
  const batch = rawClient(port);
  await batch.handshake();
  batch.socket.write(Buffer.concat([
    clientFrame(0x1, 'one'),
    clientFrame(0x1, 'two'),
    clientFrame(0x1, 'three'),
  ]));
  const three = await batch.await(() => batch.frames.filter((f) => f.opcode === 0x1).length >= 3);
  const texts = batch.frames.filter((f) => f.opcode === 0x1).map((f) => f.payload.toString());
  ok('three frames in one write, all three in order',
    three && texts.join(',') === 'one,two,three', texts.join(','));
  batch.socket.destroy();

  // One frame across three writes, with a gap, so each write lands as its own read.
  const split = rawClient(port);
  await split.handshake();
  const whole = clientFrame(0x1, 'split across reads');
  for (const [from, to] of [[0, 1], [1, 7], [7, whole.length]]) {
    split.socket.write(whole.subarray(from, to));
    await sleep(30);
  }
  const arrived = await split.await(() => split.frame(0x1));
  ok('one frame split across three writes',
    arrived && split.frame(0x1).payload.toString() === 'split across reads',
    split.frame(0x1)?.payload.toString() ?? 'nothing');
  split.socket.destroy();

  // Fragmented message: text with FIN clear, then a continuation with FIN set.
  const frag = rawClient(port);
  await frag.handshake();
  frag.socket.write(clientFrame(0x1, 'half a ', { fin: false }));
  await sleep(20);
  frag.socket.write(clientFrame(0x0, 'message'));
  const joined = await frag.await(() => frag.frame(0x1));
  ok('fragmented message reassembled',
    joined && frag.frame(0x1).payload.toString() === 'half a message',
    frag.frame(0x1)?.payload.toString() ?? 'nothing');
  frag.socket.destroy();

  // A 64-bit length carrying a tiny payload. Legal, non-minimal, and the shape a fuzzer sends.
  const wide = rawClient(port);
  await wide.handshake();
  const long = clientFrame(0x1, 'tiny');
  // Rebuild the same frame with the 8-byte extended length in place of the 7-bit one.
  const forced = Buffer.allocUnsafe(long.length + 8);
  forced[0] = long[0];
  forced[1] = 0x80 | 127;
  forced.writeUInt32BE(0, 2);
  forced.writeUInt32BE(4, 6);
  long.copy(forced, 10, 2);
  wide.socket.write(forced);
  const tiny = await wide.await(() => wide.frame(0x1));
  ok('64-bit length on a 4-byte payload',
    tiny && wide.frame(0x1).payload.toString() === 'tiny',
    wide.frame(0x1)?.payload.toString() ?? 'nothing');
  wide.socket.destroy();
}

async function protocolChecks(port) {
  console.log('protocol');

  const bare = rawClient(port);
  await bare.handshake();
  bare.socket.write(clientFrame(0x1, 'unmasked', { mask: false }));
  await bare.await(() => bare.closeCode() !== null);
  ok('unmasked client frame closes 1002', bare.closeCode() === 1002, String(bare.closeCode()));
  bare.socket.destroy();

  const fatControl = rawClient(port);
  await fatControl.handshake();
  fatControl.socket.write(clientFrame(0x9, Buffer.alloc(130, 0x61)));
  await fatControl.await(() => fatControl.closeCode() !== null);
  ok('control frame over 125 bytes closes 1002', fatControl.closeCode() === 1002,
    String(fatControl.closeCode()));
  fatControl.socket.destroy();

  const splitControl = rawClient(port);
  await splitControl.handshake();
  splitControl.socket.write(clientFrame(0x9, 'ping', { fin: false }));
  await splitControl.await(() => splitControl.closeCode() !== null);
  ok('fragmented control frame closes 1002', splitControl.closeCode() === 1002,
    String(splitControl.closeCode()));
  splitControl.socket.destroy();

  // A text frame that is not valid UTF-8 is 1007, not a message with a replacement character in it.
  const junk = rawClient(port);
  await junk.handshake();
  junk.socket.write(clientFrame(0x1, Buffer.from([0x48, 0xc3, 0x28, 0x69])));
  await junk.await(() => junk.closeCode() !== null);
  ok('invalid utf-8 in a text frame closes 1007', junk.closeCode() === 1007,
    String(junk.closeCode()));
  junk.socket.destroy();

  const orphan = rawClient(port);
  await orphan.handshake();
  orphan.socket.write(clientFrame(0x0, 'nothing started this'));
  await orphan.await(() => orphan.closeCode() !== null);
  ok('continuation with no start closes 1002', orphan.closeCode() === 1002,
    String(orphan.closeCode()));
  orphan.socket.destroy();
}

async function heartbeatChecks(port) {
  console.log('heartbeat');

  const p = rawClient(port);
  await p.handshake();
  p.socket.write(clientFrame(0x9, 'are you there'));
  const pong = await p.await(() => p.frames.some((f) => f.opcode === 0xa && f.payload.length));
  ok('client ping answered with the same payload',
    pong && p.frame(0xa).payload.toString() === 'are you there',
    p.frame(0xa)?.payload.toString() ?? 'nothing');

  // An unsolicited pong is legal and must not upset anything; the echo still works after it.
  p.socket.write(clientFrame(0xa, 'unsolicited'));
  await sleep(30);
  p.socket.write(clientFrame(0x1, 'still here'));
  const alive = await p.await(() => p.frame(0x1));
  ok('unsolicited pong accepted, connection unaffected',
    alive && p.frame(0x1).payload.toString() === 'still here',
    p.frame(0x1)?.payload.toString() ?? 'nothing');
  p.socket.destroy();

  // A server with a hurried heartbeat, and a client that answers nothing.
  const fast = await serve({ pingMs: 50, idleMs: 180 });
  const mute = rawClient(fast.port);
  await mute.handshake();
  const pinged = await mute.await(() => mute.frames.some((f) => f.opcode === 0x9));
  ok('server pings on its own', pinged);
  const dropped = await mute.await(() => mute.closeCode() !== null, 3000);
  ok('silent client closed 1001 after idleMs', dropped && mute.closeCode() === 1001,
    String(mute.closeCode()));
  mute.socket.destroy();
  await new Promise((r) => fast.http.close(r));
}

async function teardownChecks(port, seen) {
  console.log('teardown');

  // Abrupt reset: no close frame, no FIN, just gone. This is a backgrounded browser tab.
  const before = seen.closes;
  const reset = rawClient(port);
  await reset.handshake();
  reset.socket.write(clientFrame(0x1, 'then vanish'));
  await reset.await(() => reset.frame(0x1));
  reset.socket.resetAndDestroy();
  const noticed = await reset.await(() => seen.closes === before + 1);
  ok('abrupt TCP reset fires close once and does not throw', noticed,
    `${seen.closes - before} close event(s)`);

  ok('every connection that opened also closed',
    seen.opens > 0 && seen.opens === seen.closes, `${seen.opens} open / ${seen.closes} closed`);
}

// -------------------------------------------------------------------- run

const main = await serve();
const big = await serve({ maxMessageBytes: 200000 });

await handshakeChecks(main.port);
await echoChecks(main.port);
await sendChecks(main.port);
await sizeChecks(main.port, big.port);
await framingChecks(main.port);
await protocolChecks(main.port);
await heartbeatChecks(main.port);
// Sockets from the checks above are destroyed but their close events are one turn behind.
await sleep(120);
await teardownChecks(main.port, main.seen);

// close() waits for every connection to end, so it hangs if a socket was left half-open. That was a
// real bug: http.Server sockets allow half-open, and a peer that sent FIN and stopped never closed.
await new Promise((r) => main.http.close(r));
await new Promise((r) => big.http.close(r));

// The other half of "no leaks": nothing ref'd should be left holding the loop open. A ping interval
// that was not cleared shows up here as a Timeout, a socket as a TCPWrap. The listening sockets are
// closed but stay in the list for one more turn of the loop, hence the sleep.
// Only sockets and timers are interesting: stdio is always in this list, as a TTYWrap on a terminal
// and a PipeWrap the moment the output is piped anywhere.
await sleep(20);
const held = process.getActiveResourcesInfo()
  .filter((r) => r === 'Timeout' || r.startsWith('TCP'));
ok('no ref\'d handles left after the servers close', held.length === 0, held.join(', ') || 'none');

console.log(`\n${passed} ok, ${failed} failed`);
// No process.exit on success. If ws.js left a timer or a socket behind, this tool hangs here rather
// than printing, and the hang is the report.
if (failed) process.exit(1);
