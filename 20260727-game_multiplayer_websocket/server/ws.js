// One WebSocket endpoint, RFC 6455, bolted onto the http server the game is already served from.
// No dependencies: the protocol is a handshake hash and a frame header with five cases.
//
// The parts that are easy to get wrong, and are paid for here:
//
//   TCP is a byte stream, not a message stream. One read can carry three frames and one frame can
//   arrive over three reads, so everything is accumulated and the parser only ever consumes whole
//   frames. This is the classic bug in a hand-written server and it hides until a message crosses
//   a segment boundary, which for this game means "until someone sends a real build".
//
//   Client-to-server frames are always masked, server-to-client frames never are. An unmasked
//   client frame has to be rejected rather than read: the mask is what stops an attacker using a
//   browser to make a cache or proxy see attacker-chosen bytes as a request of its own.
//
//   The payload length is checked before the payload is buffered, so an oversized message costs a
//   header and a close frame rather than memory.
//
// Nothing here negotiates an extension. We never echo Sec-WebSocket-Extensions, so permessage-
// deflate stays off however loudly the client asks for it, and that is what licenses treating any
// RSV bit as a protocol error.

import { EventEmitter } from 'node:events';
import { createHash } from 'node:crypto';

// RFC 6455 section 1.3, and the one constant here worth checking against the spec rather than
// against memory: a plausible-looking wrong GUID passes every test the server writes itself and is
// rejected by every real client, since browsers verify the digest.
const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

const OP_CONT = 0x0, OP_TEXT = 0x1, OP_BIN = 0x2;
const OP_CLOSE = 0x8, OP_PING = 0x9, OP_PONG = 0xa;

const MAX_MESSAGE_BYTES = 65536;   // a fitted-out ship as JSON is a couple of KB; this is slack
const IDLE_MS = 40000;
const PING_MS = 15000;
const CLOSE_GRACE_MS = 2000;       // how long to wait for the peer's close before destroying
// A client that stops reading but keeps the connection open would otherwise grow the kernel-side
// write queue without limit, which is the same unbounded buffer as an oversized message wearing
// different clothes. At a few KB per broadcast, a megabyte queued means it has not read in minutes.
const MAX_QUEUED_BYTES = 1 << 20;

// Decoding is fatal on purpose: a text frame that is not valid UTF-8 is a 1007, not a mojibake
// message handed to the game.
const utf8 = new TextDecoder('utf-8', { fatal: true });

// Build an unmasked frame. Callers outside this module only want text, hence frameText below.
function frame(opcode, payload) {
  const len = payload.length;
  const head = len < 126 ? 2 : len < 65536 ? 4 : 10;
  const out = Buffer.allocUnsafe(head + len);
  out[0] = 0x80 | opcode;
  if (len < 126) {
    out[1] = len;
  } else if (len < 65536) {
    out[1] = 126;
    out.writeUInt16BE(len, 2);
  } else {
    out[1] = 127;
    // 64-bit length, high word first. Node caps buffers well below 2^32 so the high word is zero.
    out.writeUInt32BE(0, 2);
    out.writeUInt32BE(len, 6);
  }
  payload.copy(out, head);
  return out;
}

// Serialise once, write to many. A broadcast to both players plus any spectators is the same bytes
// for everyone, and framing is the only per-recipient work a naive send() would repeat.
export function frameText(text) {
  return frame(OP_TEXT, Buffer.from(text, 'utf8'));
}

function frameClose(code, reason = '') {
  const body = Buffer.from(reason, 'utf8');
  const payload = Buffer.allocUnsafe(2 + body.length);
  payload.writeUInt16BE(code, 0);
  body.copy(payload, 2);
  return frame(OP_CLOSE, payload);
}

function reject(socket, status, text, extra = '') {
  const body = Buffer.from(text + '\n', 'utf8');
  socket.write(
    `HTTP/1.1 ${status} ${text}\r\n` +
      'Connection: close\r\n' +
      'Content-Type: text/plain\r\n' +
      `Content-Length: ${body.length}\r\n` +
      extra +
      '\r\n',
  );
  socket.end(body);
  // end() is enough for a well-behaved client; destroy after the flush in case it is not.
  const t = setTimeout(() => socket.destroy(), CLOSE_GRACE_MS);
  t.unref();
}

// Attach a WebSocket endpoint to an existing node:http server.
//   path      only upgrade requests to this pathname are accepted (e.g. '/ws')
//   onOpen(sock, req)   called once per accepted connection
//   limits    { maxMessageBytes, idleMs, pingMs }  optional; pingMs exists so tests can hurry
export function attachWebSocket(httpServer, { path = '/ws', onOpen, limits = {} } = {}) {
  const maxMessageBytes = limits.maxMessageBytes ?? MAX_MESSAGE_BYTES;
  const idleMs = limits.idleMs ?? IDLE_MS;
  const pingMs = limits.pingMs ?? PING_MS;

  httpServer.on('upgrade', (req, socket, head) => {
    // A rejected upgrade still has to be answered on a socket the http server has already stopped
    // managing, so every path out of here either writes a response or destroys.
    socket.on('error', () => socket.destroy());

    if (String(req.headers.upgrade || '').toLowerCase() !== 'websocket') {
      return reject(socket, 400, 'Bad Request');
    }
    // The pathname only; a query string is the caller's business.
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname !== path) return reject(socket, 400, 'Bad Request');

    if (req.headers['sec-websocket-version'] !== '13') {
      return reject(socket, 400, 'Bad Request', 'Sec-WebSocket-Version: 13\r\n');
    }
    const key = req.headers['sec-websocket-key'];
    // 16 random bytes, base64: anything else is not a WebSocket client, whatever it claims.
    if (!key || Buffer.from(key, 'base64').length !== 16) {
      return reject(socket, 400, 'Bad Request');
    }

    const accept = createHash('sha1').update(key + GUID).digest('base64');
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );

    adopt(socket, head, { maxMessageBytes, idleMs, pingMs }, (sock) => onOpen?.(sock, req));
  });
}

// onReady runs before the first byte is parsed, so a caller that attaches its listeners inside it
// cannot miss a message that arrived in the same packet as the handshake.
function adopt(socket, head, { maxMessageBytes, idleMs, pingMs }, onReady) {
  const sock = new EventEmitter();
  sock.remoteAddress = socket.remoteAddress;
  sock.alive = true;

  // Small messages at a fixed cadence is exactly the traffic Nagle delays.
  socket.setNoDelay(true);

  let buf = Buffer.alloc(0);
  let fragOp = 0;             // opcode of the message being reassembled, 0 when not fragmenting
  let fragParts = [];
  let fragBytes = 0;
  let closing = false;        // a close frame has been sent; stop reading and stop writing
  let done = false;           // 'close' has been emitted
  let lastRx = Date.now();

  const write = (bytes) => {
    if (closing || socket.destroyed || !socket.writable) return false;
    if (socket.writableLength > MAX_QUEUED_BYTES) {
      shutdown(1013, 'too slow');
      return false;
    }
    return socket.write(bytes);
  };

  const finish = () => {
    if (done) return;
    done = true;
    sock.alive = false;
    closing = true;
    clearInterval(beat);
    buf = Buffer.alloc(0);
    fragParts = [];
    socket.removeAllListeners('data');
    sock.emit('close');
  };

  const shutdown = (code, reason = '') => {
    if (closing) return;
    // Flags first, and the close frame goes straight to the socket rather than through write():
    // write() can itself decide to shut down, and going back through it from here is a loop.
    closing = true;
    sock.alive = false;
    if (!socket.destroyed && socket.writable) socket.write(frameClose(code, reason));
    socket.end();
    // If the peer never answers the close, stop waiting. Unref'd so it cannot hold the process.
    const t = setTimeout(() => socket.destroy(), CLOSE_GRACE_MS);
    t.unref();
  };

  // One timer does both jobs: liveness detection is only ever as fine as the ping interval, and a
  // second timer would buy resolution nobody can perceive at the price of another handle to leak.
  const beat = setInterval(() => {
    if (closing) return;
    if (Date.now() - lastRx > idleMs) return shutdown(1001, 'idle');
    write(frame(OP_PING, Buffer.alloc(0)));
  }, pingMs);
  beat.unref();

  sock.send = (text) => {
    if (!sock.alive) return;
    write(frameText(text));
  };
  sock.sendRaw = (bytes) => {
    if (!sock.alive) return;
    write(bytes);
  };
  sock.close = (code = 1000, reason = '') => shutdown(code, reason);

  const deliver = (opcode, payload) => {
    if (opcode === OP_BIN) return sock.emit('message', payload, true);
    try {
      sock.emit('message', utf8.decode(payload), false);
    } catch {
      shutdown(1007, 'invalid utf-8');
    }
  };

  // Consume as many whole frames as the buffer holds. Returns having left any partial frame in
  // place for the next read.
  const feed = (chunk) => {
    if (closing) return;
    lastRx = Date.now();
    buf = buf.length ? Buffer.concat([buf, chunk]) : chunk;

    for (;;) {
      if (buf.length < 2) return;
      const b0 = buf[0], b1 = buf[1];
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f;
      let off = 2;

      if (b0 & 0x70) return shutdown(1002, 'rsv set');          // no extension was negotiated
      if (opcode > 2 && opcode < 8) return shutdown(1002, 'bad opcode');
      if (opcode > 0xa) return shutdown(1002, 'bad opcode');

      const control = opcode >= 8;
      // Control frames carry status, not data: they must fit in one frame and stay small so they
      // can be interleaved into a fragmented message without waiting on it.
      if (control && (!fin || len > 125)) return shutdown(1002, 'bad control frame');

      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2);
        off = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        const high = buf.readUInt32BE(2);
        // The most significant bit of a 64-bit length must be zero, and anything with a high word
        // at all is orders of magnitude past what this endpoint will accept.
        if (high !== 0) return shutdown(1009, 'too big');
        len = buf.readUInt32BE(6);
        off = 10;
      }

      if (!masked) return shutdown(1002, 'unmasked client frame');

      // Decided on the header, before a byte of payload is kept. Continuations count against the
      // total for the whole message, or the limit is only a limit per frame.
      if (!control && fragBytes + len > maxMessageBytes) return shutdown(1009, 'message too big');

      const total = off + 4 + len;
      if (buf.length < total) return;

      const mask = buf.subarray(off, off + 4);
      // Copy out: the payload has to outlive the buffer, and unmasking in place would corrupt the
      // read buffer we are still parsing out of.
      const payload = Buffer.allocUnsafe(len);
      buf.copy(payload, 0, off + 4, total);
      for (let i = 0; i < len; i++) payload[i] ^= mask[i & 3];

      // subarray is a view on the same memory, which is what makes this cheap; the parent chunk is
      // bounded by the size limit above so holding it until the next read costs nothing.
      buf = buf.subarray(total);

      if (control) {
        if (opcode === OP_PING) {
          write(frame(OP_PONG, payload));
        } else if (opcode === OP_CLOSE) {
          if (len === 1) return shutdown(1002, 'bad close payload');
          // Echo the peer's code back, which is what tells it we agreed rather than crashed.
          const code = len >= 2 ? payload.readUInt16BE(0) : 1000;
          shutdown(code >= 1000 && code <= 4999 ? code : 1002, '');
          return;
        }
        // OP_PONG needs nothing: lastRx was stamped when the bytes arrived.
        continue;
      }

      if (opcode === OP_CONT) {
        if (!fragOp) return shutdown(1002, 'continuation without start');
        fragParts.push(payload);
        fragBytes += len;
        if (!fin) continue;
        const whole = Buffer.concat(fragParts);
        const op = fragOp;
        fragOp = 0;
        fragParts = [];
        fragBytes = 0;
        deliver(op, whole);
        continue;
      }

      if (fragOp) return shutdown(1002, 'nested fragment');
      if (fin) {
        deliver(opcode, payload);
      } else {
        fragOp = opcode;
        fragParts = [payload];
        fragBytes = len;
      }
    }
  };

  socket.on('data', feed);
  socket.on('close', finish);
  // http.Server creates its sockets with allowHalfOpen, so a handler can still write a response
  // after the request body has ended, and an upgraded socket inherits that. The consequence is
  // that a peer which sends FIN and stops leaves our writable half open forever: no 'close' on the
  // socket, no 'close' for the caller, and one leaked connection per departed player. Half-open is
  // meaningless for a WebSocket, so end our side as soon as theirs ends. Found by counting opens
  // against closes; every other symptom of it looks like something else.
  socket.on('end', () => socket.end());
  socket.on('error', (err) => {
    // An abrupt reset is normal for a browser tab that went away, and EventEmitter turns an
    // unhandled 'error' into a thrown exception, so only pass it on if someone asked for it.
    sock.alive = false;
    if (sock.listenerCount('error')) sock.emit('error', err);
    socket.destroy();
  });

  onReady(sock);

  // Bytes can already have arrived alongside the request. They are the first read, not a lost read.
  if (head && head.length) feed(head);

  return sock;
}
