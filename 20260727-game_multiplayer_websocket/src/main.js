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
  roundSummary,
} from './match.js';
import { startBuild } from './ui/build.js';
import { dev, devBuild } from './dev.js';
import { makeBot } from './bot.js';
import { createPerf } from './perf.js';
import * as audio from './audio/play.js';
import {
  $,
  setVisible,
  setRound,
  setScore,
  setTimer,
  setWindLabel,
  windName,
  updateBattlePanels,
  resetBattlePanels,
  setAmmoButtons,
  showOverlay as showOverlayRaw,
  hideOverlay,
} from './ui/hud.js';

const LOG_PILL_LIFE = 4.5;
const VERDICT_DELAY = 1.8; // let the last explosion play out before the result screen
const SHAKE_ON_DETONATION = 3.5; // world units of camera jitter
const SHAKE_DECAY = 9;

const canvas = $('view');
const sceneCtl = createScene(canvas);
const fx = createFx(sceneCtl.scene);
const perf = createPerf();

let match = createMatch(dev.seed ?? randomSeed(), dev.fromRound);
let phase = 'menu';
let buildCtl = null;
let battle = null;
const views = [null, null];
let logPills = [];
// Only in dev autoplay: something has to work the ammunition, or the one live decision in the game
// never happens and an autoplayed match is not representative of a played one.
let bot = null;
let shownLogCount = 0;
let endTimer = 0;
let shake = 0;

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
  match = createMatch(dev.seed ?? randomSeed(), dev.fromRound);
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
    match.scrap[player] = devBuild(design, hullIndex, match.scrap[player], player, match.seed);
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
  audio.setAmbience(true);
  clearViews();
  fx.reset();
  logPills = [];
  shownLogCount = 0;
  $('battle-log').innerHTML = '';
  endTimer = 0;
  shake = 0;
  resetBattlePanels();

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
  bot = dev.autoplay ? makeBot(battle, { apply: setAmmo }) : null;
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

  // The timeline, then a line each on what state the ships ended in. The second part is the one
  // that explains the result: "one of six guns firing, two of eighteen hands" says the crew was too
  // thin far more clearly than a structure percentage does.
  const lines = log.slice(-4).map((l) => `<b>${l.t.toFixed(0)}s</b> ${l.text}`);
  for (let i = 0; i < 2; i++) {
    const s = roundSummary(battle.ships[i]);
    const bits = [`${Math.round(fracs[i] * 100)}% sound`];
    if (s.gunsHad > 0) bits.push(`${s.firing} of ${s.gunsHad} guns firing`);
    else bits.push('no guns');
    if (s.handsHad > 0) bits.push(`${s.hands} of ${s.handsHad} hands`);
    if (s.powder === 0) bits.push('powder gone');
    if (s.mastsHad > 0 && s.mastsLeft < s.mastsHad) {
      bits.push(`${s.mastsHad - s.mastsLeft} of ${s.mastsHad} masts down`);
    }
    lines.push(`<b>Player ${i + 1}</b> ${bits.join(', ')}`);
  }

  phase = 'result';
  panels('menu');
  setTimer(null);
  audio.setAmbience(false);
  bot = null;
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

function toggleMute() {
  const off = audio.setMuted(!audio.isMuted());
  const btn = $('btn-mute');
  btn.textContent = off ? 'Sound off' : 'Sound on';
  btn.classList.toggle('off', off);
}

function setAmmo(player, ammo) {
  if (phase !== 'battle' || !battle || battle.over) return;
  battle.setAmmo(player, ammo);
  setAmmoButtons(player, ammo);
  audio.ui('tick');
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
  else if (k === 'm') toggleMute();
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
  const elapsed = (now - last) / 1000;
  // The simulation and every animation get a clamped step, so one long frame cannot teleport a
  // ship across the arena. The build countdown gets the real elapsed time instead: it is a promise
  // to the player about seconds, and clamping made a laggy machine hand out a longer build phase
  // than a fast one. Both stop when the tab does, which is the behaviour you want when someone
  // switches away mid-build.
  const dt = Math.min(0.05, elapsed);
  last = now;
  const t0 = performance.now();
  try {
    update(dt, elapsed);
  } catch (err) {
    // One bad frame should not freeze the match.
    console.error('frame error', err);
  }
  const t1 = performance.now();
  sceneCtl.render();
  perf.sample(t1 - t0, performance.now() - t1, dt);
  requestAnimationFrame(loop);
}

function stepBuild(dt) {
  buildCtl.update(dt);
  // update() can lock in when the timer runs out, which clears buildCtl underneath us.
  if (buildCtl) setTimer(buildCtl.timeLeft);
  frameBuild(false);
}

// Camera framing plus a decaying shake offset, so a detonation is felt as well as seen.
function frameBattle(zoomScale = 1) {
  const mid = midpoint();
  let x = mid.x;
  let z = mid.z;
  if (shake > 0) {
    x += (Math.random() - 0.5) * shake;
    z += (Math.random() - 0.5) * shake;
  }
  sceneCtl.frame(x, z, mid.size * zoomScale);
}

function stepBattle(dt) {
  if (shake > 0) shake = Math.max(0, shake - dt * SHAKE_DECAY);

  if (battle.over) {
    endTimer += dt;
    // Ease into slow motion over the verdict delay: the killing blow gets to land.
    const slow = Math.max(0.15, 1 - endTimer / VERDICT_DELAY);
    frameBattle(0.86);
    fx.update(dt * slow, battle.projectiles);
    for (const v of views) if (v) v.animate(dt * slow);
    if (endTimer > VERDICT_DELAY) endRound();
    return;
  }

  if (bot) bot.update(dt * dev.speed);
  battle.advance(dt * dev.speed);
  for (const e of battle.effects) {
    if (e.type === 'detonate') shake = SHAKE_ON_DETONATION;
  }
  audio.consume(battle.effects);
  fx.consume(battle.effects);
  battle.effects.length = 0;
  setTimer(BATTLE_CAP - battle.time);
  for (let i = 0; i < 2; i++) views[i].syncFromBattle(battle.ships[i]);
  updateBattlePanels(battle, dt);
  frameBattle();
  while (shownLogCount < battle.log.length) pushLogPill(battle.log[shownLogCount++].text);
  fx.update(dt, battle.projectiles);
}

function update(dt, elapsed = dt) {
  if (phase === 'build' && buildCtl) stepBuild(elapsed);
  else if (phase === 'battle' && battle) stepBattle(dt);

  if (!(phase === 'battle' && battle && battle.over)) {
    for (const v of views) if (v) v.animate(dt);
  }

  for (let i = logPills.length - 1; i >= 0; i--) {
    logPills[i].t += dt;
    if (logPills[i].t > LOG_PILL_LIFE) logPills.splice(i, 1)[0].el.remove();
  }

  sceneCtl.update(dt);
}

// Test hook for the CDP harness in tools/.
globalThis.__game = {
  sceneCtl,
  perf,
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

$('btn-mute').onclick = toggleMute;

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
