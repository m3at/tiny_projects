// Presentation and flow. Draws whatever the client says is true and turns clicks into commands, and
// that is the whole of its job.
//
// It used to own the match as well -- the phases, the purses, the verdicts. All of that moved to
// src/net/room.js, which is the authority, and which runs either in this page for a local game or on
// a server for an online one. So this file no longer knows the rules of anything. It knows which
// overlay to show, where to point the camera, and which key belongs to which captain.
//
// The one thing worth understanding here: `client.battle` is the battle to draw, and it is either the
// authority's own (local play, nothing to replicate) or a replay reproduced from a seed and a stamped
// input stream (online). Both are the same shape, so nothing below has to care which.

import { createScene } from './render/scene.js';
import { createFx } from './render/fx.js';
import { createShipView } from './render/shipView.js';
import { createBattle } from './sim/battle.js';
import { CELL, BATTLE_CAP, VERDICT_DELAY, MAX_PLAYERS, SINK_TIME, arenaRadius } from './config.js';
import { HULLS } from './data/hulls.js';
import { startBuild } from './ui/build.js';
import { dev, devBuildCommands } from './dev.js';
import { makeBot } from './bot.js';
import { createPerf } from './perf.js';
import { createClient } from './net/client.js';
import { createLocalTransport } from './net/local.js';
import { createSocketTransport } from './net/socket.js';
import { buildMenu, buildLobby, savedName } from './ui/lobby.js';
import * as audio from './audio/play.js';
import {
  $,
  setVisible,
  setRound,
  setTimer,
  setWindLabel,
  windName,
  setupRoster,
  setScores,
  setRosterState,
  setLocked,
  updateBattlePanels,
  resetBattlePanels,
  setAmmoButtons,
  setNetBar,
  escapeHtml,
  showOverlay as showOverlayRaw,
  hideOverlay,
} from './ui/hud.js';

const LOG_PILL_LIFE = 4.5;
const SHAKE_ON_DETONATION = 3.5; // world units of camera jitter
const SHAKE_DECAY = 9;

// One key per seat. A duel keeps A and L, at opposite ends of the row, because two people share the
// keyboard; the third and fourth are only ever used by one person each in an online game.
const AMMO_KEYS = ['a', 'l', 'q', 'p'];

const canvas = $('view');
const sceneCtl = createScene(canvas);
const fx = createFx(sceneCtl.scene);
const perf = createPerf();

let client = null;
let phase = 'menu';
let buildCtl = null;
let views = [];
let logPills = [];
let bot = null; // dev autoplay only: something has to work the ammunition
let shownLogCount = 0;
let endTimer = 0;
let shake = 0;
let pendingResult = null;
let shownResult = null;
let names = [];
let lobbyView = null;

const randomSeed = () => (Math.random() * 0xffffffff) >>> 0;
const battleOf = () => (client ? client.battle : null);

// ---------------------------------------------------------------------------
// chrome
// ---------------------------------------------------------------------------

function showOverlay(opts) {
  showOverlayRaw(opts);
  // In autoplay every overlay dismisses itself, so a whole match runs unattended. Results can be
  // held for inspection, and the final screen always holds unless looping was asked for.
  const hold =
    (dev.holdResults && opts.kind === 'result') || (opts.kind === 'match-end' && !dev.loop);
  if (dev.autoplay && !hold) setTimeout(() => opts.onClick && opts.onClick(), 700 / dev.speed);
}

function panels(which) {
  setVisible($('topbar'), which !== 'menu');
  setVisible($('build-ui'), which === 'build');
  setVisible($('battle-ui'), which === 'battle');
}

function clearViews() {
  for (const view of views) {
    if (!view) continue;
    sceneCtl.scene.remove(view.group);
    view.dispose();
  }
  views = [];
}

function frameBuild(snap) {
  const hull = HULLS[client.state.hullIndex];
  sceneCtl.frame(0, 0, hull.length * CELL * 0.95, snap);
}

// Framing for the battle: the centre of whatever is still afloat, closing in as the ships close, so
// an engagement fills the screen. With four ships the spread is the widest gap between any two of
// them rather than the one distance a duel has.
function framing() {
  const battle = battleOf();
  const live = battle.ships.filter((s) => !s.out);
  const ships = live.length ? live : battle.ships;
  let x = 0;
  let z = 0;
  for (const ship of ships) {
    x += ship.x;
    z += ship.z;
  }
  x /= ships.length;
  z /= ships.length;
  let spread = 0;
  for (let i = 0; i < ships.length; i++) {
    for (let j = i + 1; j < ships.length; j++) {
      const d = Math.hypot(ships[i].x - ships[j].x, ships[i].z - ships[j].z);
      if (d > spread) spread = d;
    }
  }
  return { x, z, size: Math.max(24, Math.min(78, spread * 0.62 + 13)) };
}

function nameFor(seat) {
  return names[seat] ?? `Player ${seat + 1}`;
}

// ---------------------------------------------------------------------------
// starting a game
// ---------------------------------------------------------------------------

function attach(transport) {
  client = createClient({ transport });
  wire(client);
  transport.start();
}

function startLocal({ players, bots }) {
  attach(
    createLocalTransport({
      seed: dev.seed ?? randomSeed(),
      players,
      bots,
      hotseat: true,
      speed: dev.speed,
    }),
  );
  // Local play skips the lobby: the roster is settled by the menu.
  if (dev.fromRound) client.state.round = dev.fromRound;
}

function startOnline({ code, name, spectate }) {
  const base = dev.net && dev.net !== '1' ? dev.net : null;
  const url =
    base ??
    `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  attach(createSocketTransport({ url, name, code, spectate }));
}

function showMenu() {
  phase = 'menu';
  panels('menu');
  clearViews();
  setTimer(null);
  setNetBar(null);
  sceneCtl.frame(0, 0, 60, true);

  const menu = buildMenu({
    onLocal: (players, bots) => {
      hideOverlay();
      startLocal({ players, bots });
    },
    onCreate: (name) => {
      hideOverlay();
      startOnline({ code: null, name });
      waitingOverlay('Opening a room');
    },
    onJoin: (name, code) => {
      hideOverlay();
      startOnline({ code, name });
      waitingOverlay(`Joining ${code}`);
    },
  });

  showOverlay({
    title: 'Broadside',
    body:
      'Two to four ships, five rounds. You never steer: you fit the ship out, then read the ' +
      'battle and call the shot.\n' +
      'Round shot smashes hull. Grape shot kills the crew that works the guns.',
    node: menu.node,
    // The label changes with the mode -- the menu rewrites it through setOverlayButton -- but the
    // button has to exist, and passing null here hid it. Which nothing noticed, because every tool
    // in tools/ arrives with ?dev and skips this screen entirely. tools/playtest.js checks the menu
    // now for exactly that reason.
    button: menu.label(),
    onClick: () => menu.confirm(),
  });
}

function waitingOverlay(text) {
  showOverlay({ title: text, body: 'One moment.', button: null });
}

// ---------------------------------------------------------------------------
// client events
// ---------------------------------------------------------------------------

function wire(c) {
  c.on('status', ({ status, text }) => {
    if (status === 'refused') {
      showOverlay({
        title: 'Cannot join',
        body: text,
        button: 'Back',
        onClick: () => {
          client = null;
          showMenu();
        },
      });
      return;
    }
    updateNetBar();
  });

  c.on('room', (room) => {
    names = room.players.map((p) => p.name);
    if (room.phase === 'lobby') showLobby(room);
    if (room.players.length) {
      setRosterState(room.players);
      setScores(room.players.map((p) => p.score));
    }
  });

  c.on('intro', (msg) => introOverlay(msg));
  c.on('build', () => beginBuild());
  c.on('yard', ({ reset }) => buildCtl && buildCtl.refresh(reset));
  c.on('deny', (why) => buildCtl && buildCtl.deny(why));
  c.on('locked', (seat) => setLocked(seat, true));
  c.on('battle', () => beginBattle());
  c.on('ammo', (msg) => setAmmoButtons(msg.seat, msg.ammo));
  c.on('result', (msg) => {
    pendingResult = msg;
    // If the battle on screen has already finished, or there is none, show it at once.
    const battle = battleOf();
    if (!battle || (battle.over && endTimer > VERDICT_DELAY)) showResult();
  });
  c.on('resync', () => updateNetBar());
}

function showLobby(room) {
  if (client.state.result) return; // the match-over screen owns the screen until it is dismissed
  // A local game has no lobby: the menu settled the roster and the authority has already started.
  if (client.isLocal) return;
  phase = 'lobby';
  panels('menu');
  clearViews();
  lobbyView = buildLobby({
    room,
    mySeats: client.state.mySeats,
    onReady: (on) => client.ready(on),
    onAddBot: () => client.addBot(),
  });
  showOverlay({
    title: room.code === 'LOCAL' ? 'Ready' : `Room ${room.code}`,
    body:
      room.players.length < 2
        ? 'Waiting for a second captain. Send them the code.'
        : 'Everyone ready starts the match.',
    node: lobbyView.node,
    button: lobbyView.readyLabel,
    onClick: () => lobbyView.toggleReady(),
  });
}

function introOverlay(msg) {
  phase = 'intro';
  panels('menu');
  clearViews();
  setTimer(null);
  const hullIndex = msg.hullIndex;
  setRound(msg.round, hullIndex);
  sceneCtl.setWind(msg.windTo);
  fx.setWind(msg.windTo);
  setWindLabel(msg.windTo);
  sceneCtl.setArenaRadius(arenaRadius(client.state.room?.players.length ?? 2));

  const purses = (msg.scrap ?? [])
    .map((amount, seat) => `${nameFor(seat)}: ${amount}`)
    .join('. ');
  const paid = (msg.bonuses ?? [])
    .map((bonus, seat) => (bonus > 0 ? `${nameFor(seat)} takes ${bonus} extra for losing ground.` : null))
    .filter(Boolean);

  showOverlay({
    title: `Round ${msg.round + 1} — ${HULLS[hullIndex].name}`,
    body:
      `The wind is ${windName(msg.windTo).toLowerCase()}. ` +
      `${msg.buildTime} seconds to fit out.\n${purses}` +
      (paid.length ? `\n${paid.join(' ')}` : ''),
    button: 'Fit out',
    onClick: () => client.proceed(),
  });
}

function beginBuild() {
  // A spectator is told a build phase has started and is given nobody's hand to play, which is the
  // point of a spectator. There is nothing on the water to draw yet either, since a ship under
  // construction is its owner's business until the guns start.
  if (!client.yard) return watchBuild();

  hideOverlay();
  phase = 'build';
  panels('build');
  clearViews();
  if (buildCtl) buildCtl.destroy();

  const seat = client.state.seat;
  const hullIndex = client.state.hullIndex;
  setRound(client.state.round, hullIndex);

  // Only your own ship is on the water. What anybody else is fitting out is theirs until the guns
  // start, which is what the "last seen" column exists to stand in for.
  const view = createShipView({
    design: client.yard.design,
    hullIndex,
    player: seat,
    interactive: true,
  });
  views = [];
  views[seat] = view;
  sceneCtl.scene.add(view.group);
  frameBuild(true);

  buildCtl = startBuild({ sceneCtl, view, client, seat, names });

  if (dev.autoplay) {
    // Same commands a player would send, so autoplay is exercising the real path.
    const commands = devBuildCommands(
      client.yard.design,
      hullIndex,
      client.yard.scrap,
      seat,
      dev.seed ?? 0,
      client.state.offer,
    );
    for (const [key, part] of commands) client.place(key, part);
    view.refresh();
    buildCtl.refresh(false);
    setTimeout(() => buildCtl && $('btn-lock').click(), 400 / dev.speed);
  }
}

function beginBattle() {
  hideOverlay();
  phase = 'battle';
  panels('battle');
  audio.setAmbience(true);
  if (buildCtl) {
    buildCtl.destroy();
    buildCtl = null;
  }
  clearViews();
  fx.reset();
  logPills = [];
  shownLogCount = 0;
  $('battle-log').innerHTML = '';
  endTimer = 0;
  shake = 0;
  pendingResult = null;
  shownResult = null;
  sinking.clear();

  const battle = battleOf();
  const hullIndex = client.state.hullIndex;
  sceneCtl.setArenaRadius(battle.arenaRadius);

  setupRoster({
    players: client.state.room.players,
    mine: client.state.mySeats,
    onAmmo: (seat, ammo) => client.setAmmo(seat, ammo),
    keyFor: (seat) => AMMO_KEYS[seat],
  });
  setScores(client.state.room.players.map((p) => p.score));
  resetBattlePanels(battle.ships.length);

  for (let i = 0; i < battle.ships.length; i++) {
    const view = createShipView({ design: battle.ships[i].design, hullIndex, player: i });
    views[i] = view;
    sceneCtl.scene.add(view.group);
    view.syncFromBattle(battle.ships[i]);
    setAmmoButtons(i, 'round');
  }

  // Autoplay needs a hand on the ammunition, or the one live decision in the game never happens and
  // an autoplayed match is not representative of a played one. It goes through the client like any
  // other input, so it is stamped onto a tick and relayed exactly as a keypress would be.
  bot = dev.autoplay
    ? makeBot(battle, {
        seats: client.state.mySeats,
        apply: (seat, ammo) => client.setAmmo(seat, ammo),
      })
    : null;

  updateBattlePanels(battle);
  const mid = framing();
  sceneCtl.frame(mid.x, mid.z, mid.size, true);
}

function watchBuild() {
  phase = 'watching';
  panels('menu');
  clearViews();
  if (buildCtl) {
    buildCtl.destroy();
    buildCtl = null;
  }
  setRound(client.state.round, client.state.hullIndex);
  showOverlay({
    title: `Round ${client.state.round + 1} — fitting out`,
    body: 'The captains are spending their scrap. The battle starts when they have all locked in.',
    button: null,
  });
}

function pushLogPill(text) {
  const el = document.createElement('div');
  el.textContent = text;
  $('battle-log').appendChild(el);
  logPills.push({ el, t: 0 });
  while (logPills.length > 3) logPills.shift().el.remove();
}

// The authority's verdict, not ours. The replay on screen reaches the same end, but what is written
// on the result screen is what the room decided.
function showResult() {
  const msg = pendingResult;
  if (!msg || shownResult === msg) return;
  shownResult = msg;
  phase = msg.over ? 'over' : 'result';
  panels('menu');
  setTimer(null);
  audio.setAmbience(false);
  bot = null;
  setScores(msg.scores);

  // The battle log is written by the simulation from its own strings, but a seat name reaches it
  // through shipName, so it is escaped like everything else that becomes HTML here.
  const lines = msg.log.map((l) => `<b>${l.t.toFixed(0)}s</b> ${escapeHtml(l.text)}`);
  for (const s of msg.summaries) {
    const bits = [`${Math.round(s.structure * 100)}% sound`];
    if (s.gunsHad > 0) bits.push(`${s.firing} of ${s.gunsHad} guns firing`);
    else bits.push('no guns');
    if (s.handsHad > 0) bits.push(`${s.hands} of ${s.handsHad} hands`);
    if (s.powder === 0) bits.push('powder gone');
    if (s.mastsHad > 0 && s.mastsLeft < s.mastsHad) {
      // "1 of 1 masts down" is the arithmetic, not the sentence. There is a word for that.
      const lost = s.mastsHad - s.mastsLeft;
      bits.push(s.mastsLeft === 0 ? 'dismasted' : `${lost} of ${s.mastsHad} masts down`);
    }
    lines.push(`<b>${escapeHtml(nameFor(s.seat))}</b> ${bits.join(', ')}`);
  }

  if (msg.over) {
    const winner = msg.matchWinner;
    // Online, the same captains can go again without swapping codes: the room keeps its roster, draws
    // a new seed and goes back to the lobby. Locally there is nobody to wait for, so the menu is the
    // more useful place to land.
    const again = client.isLocal
      ? null
      : {
          label: 'Rematch',
          onClick: () => {
            client.state.result = null;
            client.rematch();
          },
        };
    showOverlay({
      kind: 'match-end',
      title: winner === null ? 'A draw' : `${nameFor(winner)} takes the day`,
      body: `${msg.reason}.\nFinal score ${msg.scores.join(' - ')}.`,
      log: lines,
      alt: again,
      button: 'New match',
      onClick: () => {
        clearViews();
        client.state.result = null;
        client.disconnect();
        client = null;
        showMenu();
      },
    });
    return;
  }

  showOverlay({
    kind: 'result',
    title:
      msg.winner === null
        ? 'Round drawn'
        : `Round ${msg.round + 1} to ${nameFor(msg.winner)}`,
    body: `${msg.reason}.\nScore ${msg.scores.join(' - ')}. Damage carries into the next round.`,
    log: lines,
    button: 'Next round',
    onClick: () => client.proceed(),
  });
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

addEventListener('keydown', (e) => {
  if (e.repeat) return;
  if (e.target instanceof HTMLInputElement) {
    if (e.key === 'Enter') $('ov-btn').click();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 'm') return toggleMute();
  if (k === 'enter' && !$('overlay').classList.contains('hidden')) return $('ov-btn').click();
  if (k === ' ' && phase === 'build') {
    e.preventDefault();
    $('btn-lock').click();
    return;
  }
  const seat = AMMO_KEYS.indexOf(k);
  const battle = battleOf();
  if (seat >= 0 && seat < MAX_PLAYERS && battle && client.controls(seat)) {
    const ship = battle.ships[seat];
    if (ship) client.setAmmo(seat, ship.ammo === 'round' ? 'grape' : 'round');
  }
});

addEventListener('resize', () => sceneCtl.resize());

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------

let last = performance.now();
let gpuSamples = [];

function loop(now) {
  const elapsed = (now - last) / 1000;
  // The simulation and every animation get a clamped step, so one long frame cannot teleport a
  // ship across the arena. Both stop when the tab does, which is the behaviour you want when
  // someone switches away mid-build; the build countdown is the authority's clock and carries on
  // regardless, which is the behaviour you want for a promise about seconds.
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
  // Adaptive quality can resize the drawing buffer. Do that before the draw which presents it:
  // resizing after render clears the buffer and leaves the compositor a chance to show that clear
  // (the brief black frame seen on a quality step) before the next requestAnimationFrame.
  sceneCtl.adapt(now, elapsed * 1000, gpuSamples);
  // Render tools temporarily replace render() with a no-op while they drive the renderer directly.
  // Treat a transport/tool wrapper with no timing result exactly like a frame whose query is pending.
  gpuSamples = sceneCtl.render() ?? [];
  for (const ms of gpuSamples) perf.sampleGpu(ms);
  // The real gap, not the clamped step. Passing dt here made every frame on a slow machine read as
  // exactly 50ms, which is the clamp rather than a measurement, and hid the tail completely.
  perf.sample(t1 - t0, performance.now() - t1, elapsed);
  requestAnimationFrame(loop);
}

function stepBuild() {
  setTimer(client.buildLeft());
  frameBuild(false);
}

// Camera framing plus a decaying shake offset, so a detonation is felt as well as seen.
function frameBattle(zoomScale = 1) {
  const mid = framing();
  let x = mid.x;
  let z = mid.z;
  if (shake > 0) {
    x += (Math.random() - 0.5) * shake;
    z += (Math.random() - 0.5) * shake;
  }
  sceneCtl.frame(x, z, mid.size * zoomScale);
}

function stepBattle(dt) {
  const battle = battleOf();
  if (!battle) {
    // The authority has moved on and we have not been told yet, or the round is over.
    if (pendingResult) showResult();
    return;
  }
  if (shake > 0) shake = Math.max(0, shake - dt * SHAKE_DECAY);

  if (battle.over) {
    endTimer += dt;
    // Ease into slow motion over the verdict delay: the killing blow gets to land.
    const slow = Math.max(0.15, 1 - endTimer / VERDICT_DELAY);
    frameBattle(0.86);
    updateSinking(battle, endTimer);
    drain(battle);
    fx.update(dt * slow, battle.projectiles);
    for (const view of views) if (view) view.animate(dt * slow);
    if (endTimer > VERDICT_DELAY && pendingResult) showResult();
    return;
  }

  if (bot) bot.update(dt * dev.speed);
  updateSinking(battle);
  drain(battle);
  setTimer(BATTLE_CAP - battle.time);
  for (let i = 0; i < battle.ships.length; i++) {
    if (views[i]) views[i].syncFromBattle(battle.ships[i]);
  }
  updateBattlePanels(battle, dt);
  frameBattle();
  fx.update(dt, battle.projectiles);
}

// A ship that struck her colours settles into the water over SINK_TIME. In a duel the battle ends the
// moment it happens, so it plays out under the slow motion; in a melee the survivors fight on over the
// wreck, which is the case this exists for.
//
// The splashes on the way down are pushed into the effect stream rather than drawn directly, so they
// get the splash sound as well as the ring. Nothing in sim/ reads effects -- the authority clears them
// unread -- so adding to the list from the renderer's side is presentation and stays presentation.
const sinking = new Set();

// `extra` is seconds to add on top of the battle clock, for the case where the battle clock has
// stopped: once the round is over the simulation is frozen, so the last ship down would settle by
// exactly zero and sit on the surface through the whole verdict delay.
function updateSinking(battle, extra = 0) {
  for (let i = 0; i < battle.ships.length; i++) {
    const ship = battle.ships[i];
    const view = views[i];
    if (!view || !ship.out) continue;
    if (!sinking.has(i)) {
      sinking.add(i);
      const rng = Math.random;
      for (let n = 0; n < 7; n++) {
        const a = (n / 7) * Math.PI * 2;
        const r = 3 + rng() * 4;
        battle.effects.push({
          type: 'splash',
          x: ship.x + Math.sin(a) * r,
          z: ship.z + Math.cos(a) * r,
        });
      }
    }
    view.setSunk((battle.time - ship.outAt + extra) / SINK_TIME);
  }
}

// Whoever consumes battle.effects must drain it; nothing in sim/ clears it.
function drain(battle) {
  for (const e of battle.effects) {
    if (e.type === 'detonate') shake = SHAKE_ON_DETONATION;
  }
  audio.consume(battle.effects);
  fx.consume(battle.effects);
  battle.effects.length = 0;
  while (shownLogCount < battle.log.length) pushLogPill(battle.log[shownLogCount++].text);
}

let netBarAt = 0;

function updateNetBar() {
  if (!client || client.isLocal) {
    setNetBar(null);
    return;
  }
  const net = client.net;
  const status = client.state.status;
  if (status !== 'open') {
    setNetBar(client.state.statusText || 'Connecting', true);
    return;
  }
  const bits = [`${net.rtt}ms`];
  if (net.jitter > 8) bits.push(`±${net.jitter}`);
  bits.push(`delay ${Math.round(Math.max(net.delayMs, net.measuredDelayMs))}ms`);
  if (net.resyncs) bits.push(`resync ${net.resyncs}`);
  setNetBar(bits.join('  '), net.desyncs > 0);
}

function update(dt, elapsed) {
  if (client) client.update(dt);

  if (phase === 'build' && client.yard) stepBuild();
  else if (phase === 'watching') setTimer(client.buildLeft());
  else if (phase === 'battle') stepBattle(dt);

  const battle = battleOf();
  if (!(phase === 'battle' && battle && battle.over)) {
    for (const view of views) if (view) view.animate(dt);
  }

  for (let i = logPills.length - 1; i >= 0; i--) {
    logPills[i].t += dt;
    if (logPills[i].t > LOG_PILL_LIFE) logPills.splice(i, 1)[0].el.remove();
  }

  // Once a second is often enough for a status line, and it writes to the DOM.
  netBarAt += dt;
  if (netBarAt > 1) {
    netBarAt = 0;
    updateNetBar();
  }

  sceneCtl.update(dt);
}

// Test hook for the CDP harness in tools/.
globalThis.__game = {
  sceneCtl,
  perf,
  createBattle,
  get client() {
    return client;
  },
  get room() {
    return client && client.isLocal ? client.state.room : null;
  },
  get phase() {
    return phase;
  },
  get battle() {
    return battleOf();
  },
  get views() {
    return views;
  },
  get net() {
    return client ? client.net : null;
  },
};

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

$('btn-mute').onclick = toggleMute;

// Only the buttons that commit to something make a sound. The build phase used to tick on every
// card and every cell, which thirty parts in ninety seconds turns into a clock. Registered after
// the handlers above so the mute toggle has already flipped: pressing it to unmute is audible,
// pressing it to mute is not.
$('ov-btn').addEventListener('click', () => audio.ui('confirm'));
$('btn-mute').addEventListener('click', () => audio.ui('press'));

sceneCtl.frame(0, 0, 60, true);
sceneCtl.setWind(0.6);
fx.setWind(0.6);
panels('menu');

// The dev harness skips the menu: it says in the URL what it wants to play.
if (dev.net) {
  startOnline({ code: dev.room, name: dev.name ?? savedName(), spectate: dev.watch });
  waitingOverlay(dev.room ? `Joining ${dev.room}` : 'Opening a room');
} else if (dev.enabled) {
  startLocal({ players: dev.players, bots: dev.bots });
} else {
  showMenu();
}

requestAnimationFrame(loop);
