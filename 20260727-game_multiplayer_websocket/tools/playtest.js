// Plays a whole match through the real interface, and complains about what a person would notice.
//
//   node tools/playtest.js [seed]
//
// Every other tool drives the simulation directly. This one drives the *game*: it picks cards,
// clicks hull cells, locks in, flips ammunition mid-battle, and dismisses overlays, all through
// the same events a player generates. That covers the half of the program the headless tools
// cannot see -- the build panel, the phase machine, the economy, the overlays -- and it is the
// only thing that would notice if locking in stopped working.
//
// It asserts what a player would notice, not what a unit test would: scrap never goes up while you
// are spending it or below zero, a round awards exactly one point or none, the score never passes
// three, and the match ends in a phase that exists.
import { attach, sleep } from './cdp.js';

const page = await attach();
const { evalIn } = page;
const problems = [];
const check = (ok, msg) => { if (!ok) { problems.push(msg); console.log('  ISSUE: ' + msg); } };

// The menu first, with no ?dev at all, because that is the one screen every other tool skips: they
// all arrive with a query that starts a game directly. A missing button here is invisible to
// everything else in tools/, and was.
await page.open('', 2200);
const menu = await page.json(`({
  title: document.getElementById('ov-title').textContent,
  button: document.getElementById('ov-btn').textContent,
  buttonShown: !document.getElementById('ov-btn').classList.contains('hidden'),
  modes: [...document.querySelectorAll('.menu-modes button')].map((b) => b.firstChild.textContent),
  counts: [...document.querySelectorAll('#ov-extra .field')].length,
})`);
console.log(`  menu: "${menu.title}" / "${menu.button}", modes [${menu.modes.join(', ')}]`);
check(menu.buttonShown, 'the menu has no button to press');
check(/sail/i.test(menu.button), `the menu button reads "${menu.button}"`);
check(menu.modes.length === 3, `expected three modes, saw ${menu.modes.length}`);
check(menu.counts === 2, 'the local mode should offer a captain count and a bot count');

// Press it, and check a real match starts from the menu alone.
await evalIn(`document.getElementById('ov-btn').click()`);
await sleep(900);
const afterMenu = await evalIn('__game.phase');
check(
  afterMenu !== 'menu',
  `pressing the menu button left the game in "${afterMenu}"`,
);
console.log(`  menu started a game: phase ${afterMenu}`);

await page.open('?dev=1&seed=' + (process.argv[2] || 2024));
await page.reachPhase('build');

const state = () => page.json('__dev.state()');

// Fill a ship by hand: guns on the flanks, crew and powder on the spine, timber in the gaps.
async function buildShip(label) {
  const s = await state();
  console.log(`\n  ${label}: ${s.round}, ${s.hull}, ${s.scrap} scrap, offer [${s.offer.join(', ')}]`);
  const before = Number(s.scrap);

  const cells = await page.json(`(async () => {
    const { HULLS } = await import('/src/data/hulls.js');
    return HULLS[globalThis.__game.client.state.hullIndex].cells.map((c) => [c.dx, c.dz]);
  })()`);

  // Buy in a sensible order: powder, crew, then guns, then plug holes.
  const wants = [
    ['powder magazine', cells.filter(([dx]) => dx === 0)],
    ['crew quarters', cells.filter(([dx]) => dx === 0)],
    ['gun deck', cells.filter(([dx]) => dx !== 0)],
    ['carronade', cells.filter(([dx]) => dx !== 0)],
    ['mast', cells.filter(([dx]) => dx === 0)],
    ['hull timber', cells],
  ];
  for (const [name, where] of wants) {
    const picked = await evalIn(`__dev.pickCard(${JSON.stringify(name)})`);
    if (picked.startsWith('no card')) continue;
    for (const [dx, dz] of where) {
      const scrapNow = Number((await state()).scrap);
      if (scrapNow <= 0) break;
      await evalIn(`__dev.clickCell(${dx}, ${dz})`);
    }
  }

  const after = await state();
  check(Number(after.scrap) <= before, `${label}: scrap went up during building (${before} -> ${after.scrap})`);
  check(Number(after.scrap) >= 0, `${label}: scrap went negative (${after.scrap})`);
  console.log(`    spent ${before - Number(after.scrap)}, left ${after.scrap}; readout ${JSON.stringify(after.readout)}`);
  if (after.warnings?.length) console.log(`    warnings: ${after.warnings.join(' | ')}`);
  return after;
}

// The build phase stays 'build' while the keyboard passes from one captain to the next, so waiting
// on the phase is not enough: wait for the seat the panel is showing to change.
async function reachSeat(seat, limitMs = 8000) {
  const deadline = Date.now() + limitMs;
  while (Date.now() < deadline) {
    const s = await state();
    if (s.phase === 'build' && s.seat === seat) return true;
    if (s.phase === 'battle') return false;
    await sleep(150);
  }
  return false;
}

let lastScore = [0, 0];
for (let round = 1; round <= 5; round++) {
  if ((await evalIn('__game.phase')) === 'over') break;
  if (!(await page.reachPhase('build'))) { check(false, `never reached build for round ${round}`); break; }

  check(await reachSeat(0), `round ${round}: player 1 never got the keyboard`);
  await buildShip(`round ${round} player 1`);
  await evalIn(`__dev.tool('btn-lock')`);
  await sleep(400);

  // The keyboard passes to player 2, still inside the same build phase.
  check(await reachSeat(1), `round ${round}: the keyboard never passed to player 2`);
  await buildShip(`round ${round} player 2`);
  await evalIn(`__dev.tool('btn-lock')`);
  await sleep(400);

  if (!(await page.reachPhase('battle'))) { check(false, `round ${round} never started`); break; }
  console.log(`    battle: watching, switching ammunition`);
  // Play the one live control: flip both sides' ammunition a few times.
  for (let i = 0; i < 8; i++) {
    await sleep(1200);
    if ((await evalIn('__game.phase')) !== 'battle') break;
    await evalIn(`document.querySelector('.ammo-btn[data-player="0"][data-ammo="${i % 2 ? 'round' : 'grape'}"]').click()`);
    await evalIn(`document.querySelector('.ammo-btn[data-player="1"][data-ammo="${i % 2 ? 'grape' : 'round'}"]').click()`);
  }
  // Let the round finish.
  for (let i = 0; i < 60 && (await evalIn('__game.phase')) === 'battle'; i++) await sleep(700);

  await sleep(600); // the authority holds the verdict back over the slow-motion
  const s = await state();
  const score = s.score.map(Number);
  const gained = score[0] + score[1] - (lastScore[0] + lastScore[1]);
  console.log(`    result: score ${score.join('-')}  (${gained === 1 ? 'one point awarded' : gained === 0 ? 'draw' : 'SUSPECT'})`);
  check(gained === 0 || gained === 1, `round ${round} awarded ${gained} points at once`);
  check(score[0] <= 3 && score[1] <= 3, `score past 3: ${score.join('-')}`);
  lastScore = score;
  // First to three. Stop here rather than pressing "New match" and counting round 1 of the next
  // one as round 6 -- which is what the first version of this did, and it reported the score
  // resetting to 0-0 as the game awarding minus three points.
  if (score[0] === 3 || score[1] === 3) {
    // `phase` stays 'result' for both a round result and the end of the match -- the difference is
    // the overlay, which is also the only part a player can see. Assert on that.
    const over = await page.json(`({
      title: document.getElementById('ov-title').textContent,
      button: document.getElementById('ov-btn').textContent,
    })`);
    check(/New match/i.test(over.button), `reached ${score.join('-')} but the button says "${over.button}"`);
    check(/takes the day|draw/i.test(over.title), `match-end title reads "${over.title}"`);
    console.log(`    match over: "${over.title}" / "${over.button}"`);
    break;
  }
  await evalIn(`(() => { const b = document.getElementById('ov-btn'); if (b) b.click(); })()`);
  await sleep(800);
}

const final = await evalIn('__game.phase');
console.log(`\n  final phase: ${final}, score ${(await state()).score.join('-')}`);
// 'intro' is where "New match" lands: the match-end button starts a fresh one.
// 'result' covers both the end of a round and the end of the match; 'intro' is a fresh match.
check(
  ['result', 'over', 'menu', 'intro', 'build', 'lobby'].includes(final),
  `ended in an odd phase: ${final}`,
);
console.log(problems.length ? `\n  ${problems.length} issue(s)` : '\n  no issues');
page.printLogs();
await page.close();
