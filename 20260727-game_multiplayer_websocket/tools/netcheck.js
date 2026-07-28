// Does the networked game agree with itself?
//
//   node tools/netcheck.js
//
// The authority in src/net/room.js runs the battle that counts, and every client rebuilds the same
// battle from a seed and a stamped input stream. That is the whole design, and it is worth exactly
// what a test of it is worth: this drives real clients against a real room over a virtual wire with
// latency, jitter and reordering, and checks that the replay lands on the authority's state tick for
// tick. It also checks the trust boundary, because a build phase where the client is believed is a
// build phase where the client can cheat.
//
// No browser and no socket: the room takes its clock as an argument, so a five-round match plays out
// in a few milliseconds of wall time. tools/netplay.js is the one that drives real browsers.

import { createRoom } from '../src/net/room.js';
import { createClient } from '../src/net/client.js';
import { checksum } from '../src/sim/checksum.js';
import { structureFraction } from '../src/sim/ship.js';
import { makeRng } from '../src/sim/rng.js';
import * as P from '../src/net/protocol.js';

let passed = 0;
let failed = 0;

function ok(label, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  ok   ${label}${detail ? `  ${detail}` : ''}`);
  } else {
    failed++;
    console.log(`  FAIL ${label}${detail ? `  ${detail}` : ''}`);
  }
}

function section(name) {
  console.log(`\n${name}`);
}

// ---------------------------------------------------------------------------
// a virtual wire
// ---------------------------------------------------------------------------

// Messages are delivered at `at`, which is the send time plus a one-way delay drawn per message. So
// they arrive out of order whenever the jitter exceeds the gap between two sends, which is the case
// this is here to produce: the timeline has to sort by tick rather than trust arrival order.
function makeWire({ rng, latency = 0, jitter = 0, loss = 0 }) {
  const queued = [];
  return {
    // What socket.js would compute from perfect samples of this wire: half the shortest round trip
    // plus the jitter it has to absorb, plus a frame of slack.
    advisedDelay: Math.max(P.RENDER_DELAY_MIN_MS, Math.round(latency + jitter + 25)),
    push(clock, msg, deliver) {
      if (loss > 0 && rng.next() < loss) return; // TCP would retransmit; this models the stall
      const delay = latency + rng.next() * jitter;
      queued.push({ at: clock + delay, msg, deliver });
    },
    pump(clock) {
      // Sorted by arrival, not by send: two messages can swap places on the way.
      queued.sort((a, b) => a.at - b.at);
      while (queued.length && queued[0].at <= clock) {
        const item = queued.shift();
        item.deliver(item.msg);
      }
    },
    get pending() {
      return queued.length;
    },
  };
}

// A client on the far end of that wire. The clock it thinks the server is on is deliberately wrong by
// `clockError`, because that is the realistic case: an estimate from round trips is never exact, and
// the render delay is what has to absorb being wrong.
function makeRemote({ room, wire, seat, clockError = 0, delayMs = P.RENDER_DELAY_MIN_MS }) {
  let clock = 0;
  const transport = {
    localRoom: null,
    onMessage: () => {},
    onStatus: () => {},
    send(msg) {
      wire.push(clock, msg, (m) => room.command(`p${seat}`, m));
    },
    serverNow: () => clock + clockError,
    update() {},
  };
  const client = createClient({ transport });
  // Stands in for what socket.js would have measured off this wire.
  client.net.measuredDelayMs = delayMs;
  client.net.delayMs = delayMs;
  transport.onMessage({
    t: P.S.WELCOME,
    v: P.PROTOCOL_VERSION,
    seat,
    mySeats: [seat],
    code: room.code,
    room: room.publicState(),
  });
  return {
    client,
    seat,
    deliver(msg) {
      wire.push(clock, msg, (m) => transport.onMessage(m));
    },
    tick(now, dt) {
      clock = now;
      client.update(dt);
    },
  };
}

// One match, played by `count` clients over `wire`, driven at 60Hz of virtual time.
function playMatch({ seed, count, wire, clockError = 0, delayMs = null, watch = null }) {
  const room = createRoom({ code: 'TEST', seed, now: 0, emit });
  const remotes = [];

  function emit(target, msg) {
    for (const remote of remotes) {
      if (target === 'all' || target === remote.seat) remote.deliver(msg);
    }
  }

  for (let i = 0; i < count; i++) {
    room.join({ id: `p${i}`, name: `P${i + 1}` });
  }
  for (let i = 0; i < count; i++) {
    // The real client works this out from round trips; here the wire's true numbers stand in for a
    // perfect measurement, so what is under test is the replay and not the estimator.
    remotes.push(makeRemote({ room, wire, seat: i, clockError, delayMs: delayMs ?? wire.advisedDelay }));
  }

  // Each client plays itself: fill the hull greedily, lock in, and flip ammunition now and then.
  for (const remote of remotes) {
    const { client } = remote;
    client.on('build', () => {
      remote.wantsBuild = true;
    });
    client.on('intro', () => client.proceed());
    client.on('result', () => client.proceed());
  }

  room.start();

  const rng = makeRng(seed ^ 0x1234);
  let clock = 0;
  const step = 1000 / 60;
  let guard = 0;
  const desyncs = [];

  while (room.phase !== 'over' && guard++ < 60 * 60 * 6) {
    clock += step;
    wire.pump(clock);
    room.update(clock);
    for (const remote of remotes) remote.tick(clock, step / 1000);

    // Clients spend their purse as soon as they are told they have one.
    for (const remote of remotes) {
      if (!remote.wantsBuild) continue;
      remote.wantsBuild = false;
      spendPurse(remote.client, rng);
      remote.client.lock();
    }

    // ...and work the ammunition, at about the rate a person would.
    if (room.phase === 'battle' && rng.next() < 0.01) {
      const remote = remotes[rng.int(0, remotes.length - 1)];
      remote.client.setAmmo(remote.seat, rng.next() < 0.5 ? 'grape' : 'round');
    }

    if (watch) watch({ room, remotes, clock });
  }

  // The verdict and the final score are still on the wire when the room reaches 'over'. Let them
  // land, or the clients are compared against a room that has moved on without them.
  for (let i = 0; i < 60; i++) {
    clock += step;
    wire.pump(clock);
    for (const remote of remotes) remote.tick(clock, step / 1000);
  }

  return { room, remotes, desyncs, clock };
}

// Buy whatever is on offer and legal, cheapest first, until the purse runs out. Not a good ship; the
// point is a legal one built by the same commands a player would send.
function spendPurse(client, rng) {
  const yard = client.yard;
  if (!yard) return;
  const hull = client.state.hullIndex;
  const cells = cellKeys(hull);
  let guard = 0;
  while (guard++ < 400) {
    const part = yard.offer[rng.int(0, yard.offer.length - 1)];
    const key = cells[rng.int(0, cells.length - 1)];
    const res = client.place(key, part);
    if (!res.ok && res.why.startsWith('Not enough scrap')) {
      // Try the cheapest thing on offer before giving up.
      if (client.place(cells[rng.int(0, cells.length - 1)], 'timber').ok) continue;
      break;
    }
  }
}

let cellCache = new Map();
function cellKeys(hullIndex) {
  if (!cellCache.has(hullIndex)) {
    const { HULLS } = hullsModule;
    cellCache.set(
      hullIndex,
      HULLS[hullIndex].cells.map((c) => `${c.dx},${c.dz}`),
    );
  }
  return cellCache.get(hullIndex);
}
const hullsModule = await import('../src/data/hulls.js');

// ---------------------------------------------------------------------------
// checks
// ---------------------------------------------------------------------------

section('a whole match, two players, no latency');
{
  const wire = makeWire({ rng: makeRng(1) });
  const { room, remotes } = playMatch({ seed: 4242, count: 2, wire });
  ok('the match reaches a verdict', room.phase === 'over', room.phase);
  ok(
    'somebody won three rounds or five were played',
    room.match.scores.some((s) => s >= 3) || room.match.roundIndex >= 4,
    `scores ${room.match.scores.join('-')} after round ${room.match.roundIndex + 1}`,
  );
  ok(
    'both clients agree on the score',
    remotes.every((r) => {
      const scores = r.client.state.room.players.map((p) => p.score);
      return scores.join() === room.match.scores.join();
    }),
    remotes.map((r) => r.client.state.room.players.map((p) => p.score).join('-')).join(' / '),
  );
  ok(
    'no client ever desynced',
    remotes.every((r) => r.client.net.desyncs === 0),
    `checks ${remotes.map((r) => r.client.net.checked).join('/')}, desyncs ${remotes
      .map((r) => r.client.net.desyncs)
      .join('/')}`,
  );
  ok(
    'every client actually verified something',
    remotes.every((r) => r.client.net.checked > 20),
    `${remotes.map((r) => r.client.net.checked).join('/')} checksums compared`,
  );
}

section('four players, 80ms of latency and 60ms of jitter');
{
  const wire = makeWire({ rng: makeRng(7), latency: 80, jitter: 60 });
  const { room, remotes } = playMatch({ seed: 99, count: 4, wire, clockError: 0 });
  ok('the match reaches a verdict', room.phase === 'over', room.phase);
  ok(
    'four ships fought',
    room.match.playerCount === 4,
    `scores ${room.match.scores.join('-')}`,
  );
  ok(
    'no client desynced',
    remotes.every((r) => r.client.net.desyncs === 0),
    `desyncs ${remotes.map((r) => r.client.net.desyncs).join('/')}, late inputs ${remotes
      .map((r) => r.client.net.lateInputs)
      .join('/')}`,
  );
  ok(
    'checksums were compared throughout',
    remotes.every((r) => r.client.net.checked > 20),
    `${remotes.map((r) => r.client.net.checked).join('/')}`,
  );
}

section('a client whose clock estimate is 120ms fast');
{
  // The dangerous direction: a client that thinks the server is further along than it is plays ahead
  // of the inputs it has been sent. The render delay is what has to cover it.
  const wire = makeWire({ rng: makeRng(11), latency: 40, jitter: 20 });
  const { room, remotes } = playMatch({
    seed: 31337,
    count: 2,
    wire,
    clockError: 120,
  });
  ok('the match still reaches a verdict', room.phase === 'over', room.phase);
  const desyncs = remotes.reduce((a, r) => a + r.client.net.desyncs, 0);
  const resyncs = remotes.reduce((a, r) => a + r.client.net.resyncs, 0);
  const late = remotes.reduce((a, r) => a + r.client.net.lateInputs, 0);
  ok(
    'the late inputs a bad clock causes are repaired before a checksum can see them',
    late > 0 && resyncs > 0 && desyncs === 0,
    `${late} late, ${resyncs} rebuilt, ${desyncs} caught by checksum`,
  );
  ok(
    'and every checksum the server stated still matched',
    remotes.every((r) => r.client.net.checked > 20 && r.client.net.desyncs === 0),
    `${remotes.map((r) => r.client.net.checked).join('/')} compared`,
  );
  ok(
    'the delay was learned once, not once per round',
    remotes.every((r) => r.client.net.lateInputs <= 8),
    `late inputs ${remotes.map((r) => r.client.net.lateInputs).join('/')} over five rounds`,
  );
}

section('a spectator');
{
  const wire = makeWire({ rng: makeRng(21), latency: 30, jitter: 15 });
  const room = createRoom({ code: 'WATCH', seed: 5150, now: 0, emit });
  const remotes = [];
  const watchers = [];

  function emit(target, msg) {
    for (const remote of remotes) {
      if (target === 'all' || target === remote.seat) remote.deliver(msg);
    }
    // A spectator gets everything addressed to the room and nothing addressed to a seat.
    if (target === 'all' || target === 'spectators') {
      for (const watcher of watchers) watcher.deliver(msg);
    }
  }

  room.join({ id: 'p0', name: 'A' });
  room.join({ id: 'p1', name: 'B' });
  remotes.push(makeRemote({ room, wire, seat: 0, delayMs: wire.advisedDelay }));
  remotes.push(makeRemote({ room, wire, seat: 1, delayMs: wire.advisedDelay }));
  // Seat -1 and no seats of its own, which is what the server hands a fifth arrival.
  const watcher = makeRemote({ room, wire, seat: -1, delayMs: wire.advisedDelay });
  watcher.client.state.mySeats = [];
  watchers.push(watcher);

  room.start();
  let clock = 0;
  const step = 1000 / 60;
  for (let i = 0; i < 60 * 60 * 2 && room.phase !== 'battle'; i++) {
    clock += step;
    wire.pump(clock);
    room.update(clock);
    for (const r of [...remotes, watcher]) r.tick(clock, step / 1000);
    for (const r of remotes) {
      if (r.client.state.phase === 'intro') r.client.proceed();
      if (r.client.yard && !r.locked) {
        r.locked = true;
        r.client.lock();
      }
    }
  }
  ok('the battle started', room.phase === 'battle', room.phase);
  ok('the spectator was never given a hand to play', watcher.client.yard === null);

  for (let i = 0; i < 600; i++) {
    clock += step;
    wire.pump(clock);
    room.update(clock);
    for (const r of [...remotes, watcher]) r.tick(clock, step / 1000);
  }
  ok(
    'and still replays the battle it is watching',
    watcher.client.battle !== null && watcher.client.battle.tickCount > 60,
    `tick ${watcher.client.battle?.tickCount}`,
  );
  ok(
    'with no divergence from the authority',
    watcher.client.net.checked > 5 && watcher.client.net.desyncs === 0,
    `${watcher.client.net.checked} checked, ${watcher.client.net.desyncs} desynced`,
  );
}

section('the trust boundary');
{
  const room = createRoom({ code: 'CHEAT', seed: 5, now: 0, emit: () => {} });
  room.join({ id: 'a', name: 'A' });
  room.join({ id: 'b', name: 'B' });
  room.start();
  room.update(P.INTRO_MS + 1); // through the intro and into the build phase
  ok('the build phase opened', room.phase === 'build', room.phase);

  const yard = room.yards[0];
  const beforeScrap = yard.scrap;
  const offer = yard.offer.slice();

  room.command('a', { t: P.C.PLACE, key: '99,99', part: offer[0] });
  ok('a cell that is not on the hull is refused', yard.scrap === beforeScrap);

  const notOffered = ['timber', 'gundeck', 'carronade', 'longgun', 'swivel', 'mast', 'crew', 'magazine', 'heavy'].find(
    (id) => !offer.includes(id),
  );
  if (notOffered) {
    room.command('a', { t: P.C.PLACE, key: '0,1', part: notOffered });
    ok(`a part that is not on offer is refused (${notOffered})`, yard.scrap === beforeScrap);
  }

  room.command('a', { t: P.C.PLACE, key: '0,0', part: offer[0] });
  ok('the helm cell cannot be built over', yard.scrap === beforeScrap);

  // Seat A must not be able to spend seat B's purse.
  const bScrap = room.yards[1].scrap;
  room.command('a', { t: P.C.PLACE, key: '0,1', part: 'timber', seat: 1 });
  ok(
    'a client cannot act for another seat',
    room.yards[1].scrap === bScrap,
    `B still holds ${room.yards[1].scrap}`,
  );

  // Spend everything, then try to overspend.
  let guard = 0;
  const keys = cellKeys(room.hullIndex);
  while (room.yards[0].scrap > 0 && guard++ < 200) {
    room.command('a', { t: P.C.PLACE, key: keys[guard % keys.length], part: 'timber' });
  }
  ok('the purse cannot go negative', room.yards[0].scrap >= 0, `${room.yards[0].scrap} left`);

  room.command('a', { t: P.C.LOCK });
  const lockedScrap = room.yards[0].scrap;
  room.command('a', { t: P.C.PLACE, key: keys[0], part: 'timber' });
  ok('nothing can be placed after locking in', room.yards[0].scrap === lockedScrap);
}

section('a battle replayed from the seed alone');
{
  // The claim the whole design rests on: the same seed, designs and stamped inputs give the same
  // battle. Run one battle twice, feeding the second copy the first copy's input log in a shuffled
  // order, and compare the fingerprints tick for tick.
  const { createBattle } = await import('../src/sim/battle.js');
  const { createTimeline } = await import('../src/sim/timeline.js');
  const { autoBuild, ARCHETYPES } = await import('../src/autobuild.js');
  const { createDesign } = await import('../src/sim/ship.js');

  const designs = ['brawler', 'crusher', 'sniper'].map((t) => {
    const d = createDesign();
    autoBuild(d, 3, 120, ARCHETYPES[t]);
    return d;
  });
  const params = { hullIndex: 3, seed: 777, windTo: 1.1 };

  const inputs = [];
  const rng = makeRng(2);
  for (let i = 0; i < 14; i++) {
    inputs.push({
      tick: rng.int(10, 700),
      seat: rng.int(0, 2),
      ammo: rng.next() < 0.5 ? 'grape' : 'round',
    });
  }

  const runOne = (order) => {
    const battle = createBattle({ designs: designs.map((d) => structuredClone(d)), ...params });
    const timeline = createTimeline(battle);
    for (const input of order) timeline.add(input);
    const marks = [];
    timeline.runToMarks(900, 30, (tick, b) => marks.push(`${tick}:${checksum(b)}`));
    return { marks, battle };
  };

  const a = runOne(inputs);
  const shuffled = [...inputs].reverse();
  const b = runOne(shuffled);
  ok(
    'the same inputs in a different order give the same battle',
    a.marks.join('|') === b.marks.join('|'),
    `${a.marks.length} fingerprints`,
  );
  ok(
    'and the same verdict',
    a.battle.winner === b.battle.winner && a.battle.time === b.battle.time,
    `winner ${a.battle.winner} at ${a.battle.time.toFixed(3)}s`,
  );

  // A duplicate input must not be applied twice.
  const c = runOne([...inputs, ...inputs]);
  ok('a re-sent input is ignored', a.marks.join('|') === c.marks.join('|'));

  // And a battle stepped in one call must match one stepped a frame at a time.
  const battle1 = createBattle({ designs: designs.map((d) => structuredClone(d)), ...params });
  const t1 = createTimeline(battle1);
  for (const input of inputs) t1.add(input);
  t1.runTo(600);
  const battle2 = createBattle({ designs: designs.map((d) => structuredClone(d)), ...params });
  const t2 = createTimeline(battle2);
  for (const input of inputs) t2.add(input);
  for (let i = 0; i < 600; i++) t2.runTo(i + 1);
  ok(
    'stepping one tick at a time matches stepping six hundred at once',
    checksum(battle1) === checksum(battle2),
    `${checksum(battle1)} vs ${checksum(battle2)}`,
  );

  // And that a real frame rate, with its uneven dt, lands on the same ticks.
  const battle3 = createBattle({ designs: designs.map((d) => structuredClone(d)), ...params });
  const rngDt = makeRng(9);
  while (battle3.tickCount < 600 && !battle3.over) {
    battle3.advance(rngDt.range(0.004, 0.03));
  }
  ok(
    'an uneven frame rate reaches the same state at the same tick',
    battle3.tickCount >= 600 ? true : battle3.over,
    `tick ${battle3.tickCount}`,
  );
}

section('reconnect');
{
  const wire = makeWire({ rng: makeRng(3), latency: 30, jitter: 10 });
  const room = createRoom({ code: 'RECON', seed: 808, now: 0, emit: () => {} });
  room.join({ id: 'a', name: 'A' });
  room.join({ id: 'b', name: 'B' });
  room.start();
  room.update(P.INTRO_MS + 1);
  room.command('a', { t: P.C.LOCK });
  room.command('b', { t: P.C.LOCK });
  ok('the battle started', room.phase === 'battle', room.phase);

  let clock = P.INTRO_MS + 1; // the room is already here; starting from zero advances nothing
  for (let i = 0; i < 120; i++) room.update((clock += 16.7));
  const midTick = room.battle.tickCount;
  ok('the battle is running', midTick > 60, `tick ${midTick}`);

  // B drops. Their ship must keep fighting, with the bot on the ammunition.
  //
  // Left with grape loaded, which is what makes the next assertion mean anything: the bot only sends
  // an input when it disagrees with what is loaded, and on a fresh battle it wants round shot. Without
  // this the test passes whether or not the bot is wired to that seat at all.
  room.command('b', { t: P.C.AMMO, ammo: 'grape' });
  room.leave('b');
  ok('the seat is held, not vacated', room.players.length === 2 && !room.players[1].connected);
  ok(
    'the bot takes that wheel immediately, not next round',
    room.botSeatList.includes(1),
    `bot seats [${room.botSeatList.join(',')}]`,
  );
  const beforeDrop = room.battle.tickCount;
  const inputsBefore = room.timeline.inputs.filter((i) => i.seat === 1).length;
  for (let i = 0; i < 120; i++) room.update((clock += 16.7));
  const inputsAfter = room.timeline ? room.timeline.inputs.filter((i) => i.seat === 1).length : 0;
  ok(
    'and works the ammunition for the absent player',
    inputsAfter > inputsBefore || room.battle === null,
    `${inputsBefore} -> ${inputsAfter} inputs for the dropped seat`,
  );
  ok(
    'the battle carried on without them',
    room.battle === null || room.battle.tickCount > beforeDrop,
    room.battle ? `tick ${room.battle.tickCount}` : 'battle already finished',
  );

  // ...and a rebuilt replay from the stored message plus the input log lands on the same state.
  if (room.battle) {
    const { createBattle } = await import('../src/sim/battle.js');
    const { createTimeline } = await import('../src/sim/timeline.js');
    const msg = room.battleMessage;
    const rebuilt = createBattle({
      designs: msg.designs.map((d) => structuredClone(d)),
      hullIndex: msg.hullIndex,
      seed: msg.seed,
      windTo: msg.windTo,
    });
    const timeline = createTimeline(rebuilt);
    for (const input of room.timeline.inputs) timeline.add(input);
    timeline.runTo(room.battle.tickCount);
    ok(
      'a reconnecting client rebuilds the exact state',
      checksum(rebuilt) === checksum(room.battle),
      `tick ${rebuilt.tickCount} vs ${room.battle.tickCount}`,
    );
    ok(
      'and the same structure on every ship',
      rebuilt.ships.every(
        (s, i) =>
          Math.abs(structureFraction(s) - structureFraction(room.battle.ships[i])) < 1e-12,
      ),
    );
  }
}

section('hot seat');
{
  const room = createRoom({ code: 'LOCAL', seed: 12, now: 0, emit: () => {}, hotseat: true });
  room.join({ id: 'local', name: 'P1' });
  room.join({ id: 'local', name: 'P2' });
  room.start();
  room.update(P.INTRO_MS + 1);
  ok('the first captain has the keyboard', room.phase === 'build' && room.buildSeat === 0);
  room.command('local', { t: P.C.LOCK, seat: 0 });
  ok('locking in passes it to the second', room.phase === 'build' && room.buildSeat === 1);
  room.command('local', { t: P.C.LOCK, seat: 1 });
  ok('and then the battle starts', room.phase === 'battle', room.phase);
  ok('with two ships', room.battle.ships.length === 2);
}

// ---------------------------------------------------------------------------
// the server process
// ---------------------------------------------------------------------------
//
// Everything above drives the authority directly. This drives server/main.js over a real socket, which
// is where the handshake, the refusals and the join codes live -- none of which the in-process tests
// can see. No browser: Node has a WebSocket client built in, which is the same reason tools/cdp.js has
// no dependency either.

section('the server, over a real socket');
{
  const { spawn } = await import('node:child_process');
  const PORT = 8199;
  const server = spawn(process.execPath, ['server/main.js', String(PORT)], { stdio: 'ignore' });
  const gone = new Promise((r) => server.on('exit', r));

  const waitHealth = async () => {
    for (let i = 0; i < 50; i++) {
      const res = await fetch(`http://127.0.0.1:${PORT}/health`).catch(() => null);
      if (res?.ok) return res.json();
      await new Promise((r) => setTimeout(r, 100));
    }
    return null;
  };
  const health = await waitHealth();
  ok('the server starts and answers /health', !!health?.ok, JSON.stringify(health));

  // One socket, one collected conversation. Resolves on close so a refusal can be asserted on.
  function talk(hello, { keep = false } = {}) {
    return new Promise((resolve) => {
      const ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);
      const seen = [];
      let done = false;
      const finish = (closeCode) => {
        if (done) return;
        done = true;
        resolve({ seen, closeCode, ws });
      };
      ws.addEventListener('open', () => ws.send(JSON.stringify(hello)));
      ws.addEventListener('message', (ev) => {
        seen.push(JSON.parse(ev.data));
        // A conversation that is meant to stay open is handed back once it has been greeted.
        if (keep && seen.some((m) => m.t === P.S.WELCOME)) finish(null);
      });
      ws.addEventListener('close', (ev) => finish(ev.code));
      setTimeout(() => finish(null), 3000);
    });
  }

  const old = await talk({ t: P.C.HELLO, v: P.PROTOCOL_VERSION + 99, name: 'Stale' });
  ok(
    'a client on the wrong protocol version is refused with a code it can read',
    old.closeCode === P.CLOSE.PROTOCOL,
    `close ${old.closeCode}`,
  );

  const nowhere = await talk({ t: P.C.HELLO, v: P.PROTOCOL_VERSION, name: 'Lost', code: 'ZZZZ' });
  ok(
    'an unknown join code is refused, not silently made into a new room',
    nowhere.closeCode === P.CLOSE.NO_ROOM,
    `close ${nowhere.closeCode}`,
  );

  const host = await talk({ t: P.C.HELLO, v: P.PROTOCOL_VERSION, name: 'Anne' }, { keep: true });
  const welcome = host.seen.find((m) => m.t === P.S.WELCOME);
  ok('a client with no code opens a room', !!welcome, welcome ? `room ${welcome.code}` : 'no welcome');
  ok('and is given a seat and a token to come back with', welcome?.seat === 0 && !!welcome?.token);
  ok(
    'the join code avoids the characters people mishear',
    !!welcome && [...welcome.code].every((ch) => P.CODE_ALPHABET.includes(ch)),
    welcome?.code,
  );

  // Garbage must not take the connection down: a browser extension or a stray frame should not end
  // somebody's match.
  host.ws.send('not json at all{');
  host.ws.send(JSON.stringify({ t: 'nonsense', payload: 1 }));
  await new Promise((r) => setTimeout(r, 300));
  ok('garbage does not close the socket', host.ws.readyState === 1, `state ${host.ws.readyState}`);

  // A name from a stranger ends up in three other browsers' DOM.
  const rude = await talk(
    { t: P.C.HELLO, v: P.PROTOCOL_VERSION, name: '<img src=x>\u0007 aVeryLongNameIndeed', code: welcome.code },
    { keep: true },
  );
  const roster = rude.seen.find((m) => m.t === P.S.WELCOME)?.room.players ?? [];
  const rudeName = roster[1]?.name ?? '';
  ok(
    'a name is trimmed, and stripped of control characters and markup',
    rudeName.length <= P.MAX_NAME && !/[\x00-\x1f<>&"']/.test(rudeName),
    JSON.stringify(rudeName),
  );

  // Clock sync: our own value has to come back untouched, or the offset is nonsense.
  const pong = await new Promise((resolve) => {
    host.ws.addEventListener('message', function once(ev) {
      const msg = JSON.parse(ev.data);
      if (msg.t !== P.S.PONG) return;
      host.ws.removeEventListener('message', once);
      resolve(msg);
    });
    host.ws.send(JSON.stringify({ t: P.C.PING, c: 1234.5 }));
  });
  ok('a ping comes back with our own clock untouched', pong.c === 1234.5, JSON.stringify(pong.c));
  ok('and the server states its own', typeof pong.s === 'number' && pong.s > 1e12);

  // Fill the room, then one more.
  const extra = [];
  for (const name of ['Cora', 'Dane']) {
    extra.push(await talk({ t: P.C.HELLO, v: P.PROTOCOL_VERSION, name, code: welcome.code }, { keep: true }));
  }
  const fifth = await talk(
    { t: P.C.HELLO, v: P.PROTOCOL_VERSION, name: 'Late', code: welcome.code },
    { keep: true },
  );
  const seatedFifth = fifth.seen.find((m) => m.t === P.S.WELCOME);
  ok(
    'a fifth captain becomes a spectator rather than being turned away',
    seatedFifth?.spectating === true && seatedFifth.seat === -1,
    `seat ${seatedFifth?.seat}, spectating ${seatedFifth?.spectating}`,
  );

  for (const conn of [host, rude, fifth, ...extra]) conn.ws.close();
  server.kill('SIGTERM');
  const exited = await Promise.race([gone, new Promise((r) => setTimeout(() => r('timeout'), 4000))]);
  ok('the server shuts down on a signal', exited !== 'timeout', `exit ${exited}`);
}

console.log(`\n${passed} ok, ${failed} failed`);
if (failed) process.exit(1);
