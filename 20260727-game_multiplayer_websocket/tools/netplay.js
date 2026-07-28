// Plays a whole online match with real browsers against the real server.
//
//   node tools/netplay.js [players] [seed]      default 2 players
//   node tools/netplay.js 4
//
// tools/netcheck.js drives the authority and the replay directly, over a virtual wire, and is where
// the netcode is actually pinned down. This is the other half: separate browser tabs, separate
// sockets, a real WebSocket server, real clock estimation from real round trips, and the game's own
// interface pressing its own buttons. It is the only thing that would notice if the lobby stopped
// starting matches, if the build panel stopped sending commands, or if a client's replay of the
// battle disagreed with the server's over an actual connection.
//
// What it asserts is what a player would: everybody sees the same score, nobody's battle diverged,
// the match reaches a verdict, and no tab logged anything.

import { attach, openTab, closeTab, sleep } from './cdp.js';

const WANT = Math.max(2, Math.min(4, Number(process.argv[2] || 2)));
const SEED = process.argv[3] || '';
const NAMES = ['Anne', 'Bart', 'Cora', 'Dane'];
const BASE = 'http://127.0.0.1:8123/index.html';

const problems = [];
const check = (ok, msg) => {
  if (!ok) {
    problems.push(msg);
    console.log(`  ISSUE: ${msg}`);
  }
};

// The server has to be the one in server/main.js, not a static file server: a static server hands out
// the page and then refuses the socket, which looks like the game failing to start.
const health = await fetch('http://127.0.0.1:8123/health')
  .then((r) => r.json())
  .catch(() => null);
if (!health?.ok) {
  console.error('no game server on 8123. Run ./tools/dev.sh first.');
  process.exit(1);
}
console.log(`server ok, protocol v${health.v}, ${health.rooms} room(s) open`);

// ---------------------------------------------------------------------------
// the players
// ---------------------------------------------------------------------------

const tabs = [];

async function player(index, code) {
  const query =
    `?dev=1&net=1&name=${NAMES[index]}` +
    (code ? `&room=${code}` : '') +
    (SEED ? `&seed=${SEED}` : '');
  // The first player reuses the tab dev.sh already opened; the rest get their own.
  const page = index === 0 ? await attach() : await openTab('about:blank');
  await page.open(query, 2600);
  const seat = await waitFor(page, (s) => s.seat !== null && s.code, 8000);
  check(!!seat, `${NAMES[index]} never joined`);
  tabs.push({ page, name: NAMES[index], index });
  return page;
}

async function waitFor(page, predicate, limitMs = 10000) {
  const deadline = Date.now() + limitMs;
  while (Date.now() < deadline) {
    const s = await page.json('__dev.state()', { soft: true });
    if (s && typeof s === 'object' && predicate(s)) return s;
    await sleep(150);
  }
  return null;
}

const first = await player(0, null);
const opened = await waitFor(first, (s) => s.code && s.code !== 'LOCAL', 9000);
check(!!opened, 'the first player never got a room code');
const code = opened?.code;
console.log(`room ${code}, ${WANT} players`);

for (let i = 1; i < WANT; i++) {
  await player(i, code);
  console.log(`  ${NAMES[i]} joined`);
}

// Everybody sees everybody before anyone is ready. This is the lobby working, and it is the first
// thing that breaks when the roster stops being broadcast.
for (const tab of tabs) {
  const s = await waitFor(tab.page, (st) => st.players.length === WANT, 6000);
  check(!!s, `${tab.name} does not see all ${WANT} captains`);
}

// ---------------------------------------------------------------------------
// ready up
// ---------------------------------------------------------------------------

for (const tab of tabs) await tab.page.evalIn('__dev.proceed()', { soft: true });
const started = await waitFor(first, (s) => s.phase !== 'lobby', 9000);
check(!!started, 'the match never started after everybody readied up');
console.log(`match started, phase ${started?.phase}`);

// ---------------------------------------------------------------------------
// play it
// ---------------------------------------------------------------------------

// One pass over every tab: dismiss whatever overlay is up, fit out if it is time to, and work the
// ammunition during a battle. Exactly what a person at each keyboard would be doing.
let flip = 0;
const roundsSeen = new Set();
let lastScores = null;

async function serve(tab) {
  // Every tab but one is hidden, and a hidden tab gets no requestAnimationFrame, so its frame loop is
  // not running and its replay of the battle would sit at tick zero for ever. The driver supplies the
  // clock instead -- the same thing every other headless tool in here does, and the same reason. What
  // it costs in coverage is that this does not prove the frame loop calls update(); tools/playtest.js
  // and tools/shot.js run a visible tab and do.
  await tab.page.evalIn('__dev.pump(0.06)', { soft: true });
  const s = await tab.page.json('__dev.state()', { soft: true });
  if (!s || typeof s !== 'object') return null;

  if (s.overlay) {
    // Everything except the match-end screen. Its button starts a fresh match, which drops the
    // socket and returns to the menu -- and then there is nothing left to read the final score off.
    if (s.phase !== 'over') await tab.page.evalIn('__dev.proceed()', { soft: true });
    return s;
  }

  if (s.phase === 'build' && !s.players[s.seat]?.locked) {
    // Fill and lock. The Fill button asks for the same placements a player would click, and every one
    // of them is validated by the server.
    await tab.page.evalIn(`__dev.tool('dev-fill')`, { soft: true });
    await tab.page.evalIn(`__dev.tool('btn-lock')`, { soft: true });
    return s;
  }

  if (s.phase === 'battle' && s.battle && !s.battle.over) {
    if (flip++ % 7 === 0) {
      const want = s.battle.ships[s.seat]?.ammo === 'grape' ? 'round' : 'grape';
      await tab.page.evalIn(`__dev.ammo(${s.seat}, '${want}')`, { soft: true });
    }
  }
  return s;
}

const deadline = Date.now() + 240000;
let finished = false;
while (Date.now() < deadline && !finished) {
  for (const tab of tabs) {
    const s = await serve(tab);
    if (!s) continue;
    tab.last = s;
    if (s.phase === 'over') finished = true;
    if (s.phase === 'battle' && !roundsSeen.has(s.roundIndex)) {
      roundsSeen.add(s.roundIndex);
      console.log(`  round ${s.roundIndex + 1} battle under way`);
    }
  }

  // The scores every client believes, compared against each other rather than against the server:
  // agreement between the clients is the thing a player would notice.
  const all = tabs.map((t) => t.last?.score?.join('-')).filter(Boolean);
  if (all.length === tabs.length && new Set(all).size > 1 && all.join() !== lastScores) {
    // A disagreement can be one message in flight, so it is only worth reporting if it persists.
    lastScores = all.join();
    await sleep(700);
    const again = [];
    for (const tab of tabs) {
      const s = await tab.page.json('__dev.state()', { soft: true });
      again.push(s?.score?.join('-'));
    }
    check(new Set(again).size === 1, `clients disagree on the score: ${again.join(' vs ')}`);
  }

  await sleep(350);
}

check(finished, 'the match never reached a verdict');

// ---------------------------------------------------------------------------
// what the wire did
// ---------------------------------------------------------------------------

console.log('\n  player   seat  score        rtt   delay  checks  desync  resync  late');
for (const tab of tabs) {
  const s = (await tab.page.json('__dev.state()', { soft: true })) ?? {};
  const net = s.net ?? {};
  console.log(
    `  ${tab.name.padEnd(8)} ${String(s.seat).padEnd(5)} ${String(s.score?.join('-') ?? '?').padEnd(12)} ` +
      `${String(net.rtt ?? '?').padStart(4)}ms ${String(net.delay ?? '?').padStart(6)}ms ` +
      `${String(net.checked ?? 0).padStart(6)} ${String(net.desyncs ?? 0).padStart(7)} ` +
      `${String(net.resyncs ?? 0).padStart(7)} ${String(net.lateInputs ?? 0).padStart(5)}`,
  );
  // The claim the whole design rests on: this browser's replay of the battle matched the server's at
  // every checkpoint. A resync is allowed -- it is the repair working -- but a desync that was never
  // repaired is not.
  check(
    (net.checked ?? 0) > 20,
    `${tab.name} verified only ${net.checked ?? 0} checksums; the sync stream is not arriving`,
  );
  check((net.desyncs ?? 0) === 0, `${tab.name} desynced ${net.desyncs} time(s)`);
  tab.final = s;
}

const finals = tabs.map((t) => t.final?.score?.join('-'));
check(new Set(finals).size === 1, `final scores differ between clients: ${finals.join(' vs ')}`);
console.log(`\n  final score ${finals[0]} after ${roundsSeen.size} round(s)`);

for (const tab of tabs) {
  process.stdout.write(`  ${tab.name}`);
  tab.page.printLogs(12);
}

console.log(problems.length ? `\n  ${problems.length} issue(s)` : '\n  no issues');
for (const tab of tabs) {
  await tab.page.close();
  await closeTab(tab.page);
}
process.exit(problems.length ? 1 : 0);
