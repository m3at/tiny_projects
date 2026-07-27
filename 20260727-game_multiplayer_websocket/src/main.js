// Presentation and flow for local hot-seat play: round intro, two build phases, battle,
// result. Match bookkeeping lives in match.js and the simulation in sim/, both free of the
// DOM; this file is the only part a networked version would replace.

import { createScene } from './render/scene.js';
import { createFx } from './render/fx.js';
import { createShipView } from './render/shipView.js';
import { structureFraction } from './sim/ship.js';
import { createBattle } from './sim/battle.js';
import { makeRng } from './sim/rng.js';
import { CELL, BATTLE_CAP } from './config.js';
import {
  createMatch,
  beginRound,
  battleSeed,
  offerSeed,
  hullIndexOf,
  hullOf,
  roundOf,
  recordResult,
  intelFor,
  isMatchOver,
} from './match.js';
import { startBuild } from './ui/build.js';
import { dev, devBuild } from './dev.js';
import {
  $,
  setVisible,
  setRound,
  setScore,
  setTimer,
  setWindLabel,
  windName,
  updateBattlePanels,
  setAmmoButtons,
  showOverlay as showOverlayRaw,
  hideOverlay,
} from './ui/hud.js';

const LOG_PILL_LIFE = 4.5;
const VERDICT_DELAY = 1.8; // let the last explosion play out before the result screen

const canvas = $('view');
const sceneCtl = createScene(canvas);
const fx = createFx(sceneCtl.scene);

let match = createMatch(randomSeed(), dev.fromRound);
let phase = 'menu';
let buildCtl = null;
let battle = null;
const views = [null, null];
let logPills = [];
let shownLogCount = 0;
let endTimer = 0;

function randomSeed() {
  return (Math.random() * 0xffffffff) >>> 0;
}

// In autoplay every overlay dismisses itself, so a whole match runs unattended. Results can
// be held for inspection, and the final screen always holds unless looping was asked for.
function showOverlay(opts) {
  showOverlayRaw(opts);
  const hold =
    (dev.holdResults && opts.kind === 'result') || (opts.kind === 'match-end' && !dev.loop);
  if (dev.autoplay && !hold) setTimeout(() => opts.onClick && opts.onClick(), 700 / dev.speed);
}

function clearViews() {
  for (let i = 0; i < 2; i++) {
    if (views[i]) {
      sceneCtl.scene.remove(views[i].group);
      views[i].dispose();
      views[i] = null;
    }
  }
}

function panels(which) {
  setVisible($('topbar'), which !== 'menu');
  setVisible($('build-ui'), which === 'build');
  setVisible($('battle-ui'), which === 'battle');
}

function frameBuild(snap) {
  sceneCtl.frame(0, 0, hullOf(match).length * CELL * 0.95, snap);
}

// Framing for the battle: closes in as the ships close, so the engagement fills the screen.
function midpoint() {
  const [a, b] = battle.ships;
  const d = Math.hypot(a.x - b.x, a.z - b.z);
  return {
    x: (a.x + b.x) / 2,
    z: (a.z + b.z) / 2,
    size: Math.max(24, Math.min(62, d * 0.62 + 13)),
  };
}

// ---------------------------------------------------------------------------
// phases
// ---------------------------------------------------------------------------

function startMatch() {
  match = createMatch(randomSeed(), dev.fromRound);
  setScore(0, 0);
  startRound();
}

function startRound() {
  const { bonus } = beginRound(match);
  sceneCtl.setWind(match.windTo);
  fx.setWind(match.windTo);
  setWindLabel(match.windTo);
  setRound(match.roundIndex, hullIndexOf(match));

  clearViews();
  phase = 'intro';
  panels('menu');
  setTimer(null);

  const bonusNote =
    match.lastLoser !== null
      ? `\nPlayer ${match.lastLoser + 1} takes ${bonus} extra scrap for losing the round.`
      : '';
  showOverlay({
    title: `Round ${match.roundIndex + 1} — ${hullOf(match).name}`,
    body:
      `The wind is ${windName(match.windTo).toLowerCase()}. ` +
      `You have ${roundOf(match).buildTime} seconds to fit out.` +
      `\nPlayer 1: ${match.scrap[0]} scrap. Player 2: ${match.scrap[1]} scrap.${bonusNote}`,
    button: 'Player 1, build',
    onClick: () => beginBuild(0),
  });
}

function beginBuild(player) {
  hideOverlay();
  phase = 'build';
  panels('build');
  clearViews();

  const hullIndex = hullIndexOf(match);
  const design = match.designs[player];
  const view = createShipView({ design, hullIndex, player, interactive: true });
  views[player] = view;
  sceneCtl.scene.add(view.group);
  frameBuild(true);

  // When dev is holding on this round, leave the purse unspent so it can be built by hand
  // on top of whatever the earlier rounds left behind.
  const holdHere = !!dev.stopAtRound && match.roundIndex + 1 >= dev.stopAtRound;
  if (dev.autoplay && !holdHere) {
    match.scrap[player] = devBuild(design, hullIndex, match.scrap[player], player);
    view.refresh();
  }

  buildCtl = startBuild({
    sceneCtl,
    view,
    design,
    hullIndex,
    player,
    roundIndex: match.roundIndex,
    scrap: match.scrap[player],
    rng: makeRng(offerSeed(match, player)),
    enemy: intelFor(match, player),
    onLockIn: (left) => {
      match.scrap[player] = left;
      buildCtl = null;
      if (player === 0) handOver();
      else beginBattle();
    },
  });

  if (dev.autoplay && !holdHere) {
    setTimeout(() => buildCtl && $('btn-lock').click(), 500 / dev.speed);
  }
}

function handOver() {
  clearViews();
  phase = 'handoff';
  panels('menu');
  setTimer(null);
  showOverlay({
    title: 'Pass the keyboard',
    body: 'Player 1 is fitted out.\nPlayer 2, your turn.',
    button: 'Player 2, build',
    onClick: () => beginBuild(1),
  });
}

function beginBattle() {
  hideOverlay();
  phase = 'battle';
  panels('battle');
  clearViews();
  fx.reset();
  logPills = [];
  shownLogCount = 0;
  $('battle-log').innerHTML = '';
  endTimer = 0;

  const hullIndex = hullIndexOf(match);
  battle = createBattle({
    designs: match.designs,
    hullIndex,
    seed: battleSeed(match),
    windTo: match.windTo,
  });

  for (let i = 0; i < 2; i++) {
    views[i] = createShipView({ design: match.designs[i], hullIndex, player: i });
    sceneCtl.scene.add(views[i].group);
    views[i].syncFromBattle(battle.ships[i]);
    setAmmoButtons(i, 'round');
  }
  updateBattlePanels(battle);
  const mid = midpoint();
  sceneCtl.frame(mid.x, mid.z, mid.size, true);
}

function pushLogPill(text) {
  const el = document.createElement('div');
  el.textContent = text;
  $('battle-log').appendChild(el);
  logPills.push({ el, t: 0 });
  while (logPills.length > 3) logPills.shift().el.remove();
}

function endRound() {
  const fracs = [structureFraction(battle.ships[0]), structureFraction(battle.ships[1])];
  const { winner, reason, log } = battle;
  battle.finish(); // writes damage back onto the persistent designs
  recordResult(match, winner);
  setScore(match.scores[0], match.scores[1]);

  const lines = log
    .slice(-5)
    .map((l) => `<b>${l.t.toFixed(0)}s</b> ${l.text}`)
    .concat([
      `<b>hull</b> Player 1 ${Math.round(fracs[0] * 100)}% sound, ` +
        `Player 2 ${Math.round(fracs[1] * 100)}% sound`,
    ]);

  phase = 'result';
  panels('menu');
  setTimer(null);
  battle = null;

  if (isMatchOver(match)) {
    const [a, b] = match.scores;
    showOverlay({
      kind: 'match-end',
      title: a === b ? 'A draw' : `Player ${a > b ? 1 : 2} takes the day`,
      body: `${reason}.\nFinal score ${a} - ${b}.`,
      log: lines,
      button: 'New match',
      onClick: () => {
        clearViews();
        startMatch();
      },
    });
  } else {
    showOverlay({
      kind: 'result',
      title: winner === null ? 'Round drawn' : `Round ${match.roundIndex + 1} to Player ${winner + 1}`,
      body:
        `${reason}.\nScore ${match.scores[0]} - ${match.scores[1]}. ` +
        'Damage carries into the next round.',
      log: lines,
      button: 'Next round',
      onClick: () => {
        match.roundIndex++;
        startRound();
      },
    });
  }
}

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

function setAmmo(player, ammo) {
  if (phase !== 'battle' || !battle || battle.over) return;
  battle.setAmmo(player, ammo);
  setAmmoButtons(player, ammo);
}

document.querySelectorAll('.ammo-btn').forEach((btn) => {
  btn.onclick = () => setAmmo(Number(btn.dataset.player), btn.dataset.ammo);
});

addEventListener('keydown', (e) => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  const flip = (p) => battle && setAmmo(p, battle.ships[p].ammo === 'round' ? 'grape' : 'round');
  if (k === 'a') flip(0);
  else if (k === 'l') flip(1);
  else if (k === 'enter' && !$('overlay').classList.contains('hidden')) $('ov-btn').click();
  else if (k === ' ' && phase === 'build') {
    e.preventDefault();
    $('btn-lock').click();
  }
});

addEventListener('resize', () => sceneCtl.resize());

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------

let last = performance.now();

function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  try {
    step(dt);
  } catch (err) {
    // One bad frame should not freeze the match.
    console.error('frame error', err);
  }
  requestAnimationFrame(loop);
}

function stepBuild(dt) {
  buildCtl.update(dt);
  // update() can lock in when the timer runs out, which clears buildCtl underneath us.
  if (buildCtl) setTimer(buildCtl.timeLeft);
  frameBuild(false);
}

function stepBattle(dt) {
  if (battle.over) {
    endTimer += dt;
    const mid = midpoint();
    sceneCtl.frame(mid.x, mid.z, Math.max(46, mid.size * 0.8));
    fx.update(dt, battle.projectiles);
    if (endTimer > VERDICT_DELAY) endRound();
    return;
  }

  battle.advance(dt * dev.speed);
  fx.consume(battle.effects);
  battle.effects.length = 0;
  setTimer(BATTLE_CAP - battle.time);
  for (let i = 0; i < 2; i++) views[i].syncFromBattle(battle.ships[i]);
  updateBattlePanels(battle);
  const mid = midpoint();
  sceneCtl.frame(mid.x, mid.z, mid.size);
  while (shownLogCount < battle.log.length) pushLogPill(battle.log[shownLogCount++].text);
  fx.update(dt, battle.projectiles);
}

function step(dt) {
  if (phase === 'build' && buildCtl) stepBuild(dt);
  else if (phase === 'battle' && battle) stepBattle(dt);

  for (const v of views) if (v) v.animate(dt);

  for (let i = logPills.length - 1; i >= 0; i--) {
    logPills[i].t += dt;
    if (logPills[i].t > LOG_PILL_LIFE) logPills.splice(i, 1)[0].el.remove();
  }

  sceneCtl.update(dt);
  sceneCtl.render();
}

// Test hook for the CDP harness in tools/.
globalThis.__game = {
  sceneCtl,
  get match() {
    return match;
  },
  get phase() {
    return phase;
  },
  get battle() {
    return battle;
  },
  get views() {
    return views;
  },
};

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

sceneCtl.frame(0, 0, 60, true);
sceneCtl.setWind(0.6);
fx.setWind(0.6);
panels('menu');
showOverlay({
  title: 'Broadside',
  body:
    'Two ships, five rounds, one keyboard.\n' +
    'You never steer. You fit the ship out, then read the battle and call the shot.\n\n' +
    'Round shot smashes hull. Grape shot kills the crew that works their guns.\n' +
    'Player 1 presses A to switch. Player 2 presses L.',
  button: 'Begin',
  onClick: startMatch,
});
requestAnimationFrame(loop);
