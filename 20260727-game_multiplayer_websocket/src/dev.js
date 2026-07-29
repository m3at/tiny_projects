// Dev harness, off unless asked for by URL. Lets a whole match play itself so the battle
// can be watched (or screenshotted) without clicking through two build phases per round.
//
//   ?dev=brawler,sniper       both sides auto-built each round, overlays auto-advance
//   ?dev=draft                autoplay, but both ships drafted at random each round
//   ?dev=brawler,sniper&x=3   run at 3x speed
//   ?dev=1                    manual build, but adds a Fill button to the build panel
//   &seed=1234                pin the match seed, so a playthrough can be replayed exactly
//   ?dev=brawler,sniper&stop=2  autoplay round 1, then hold on round 2's build phase
//   ?dev=brawler,sniper&hold=1  autoplay, but stop on each result screen
//   &loop=1                   keep starting fresh matches (off by default)
//
// Archetype names come from autobuild.js. `draft` instead of a name gives each side a random
// plausible ship every round, drawn from the same distribution tools/parts.js samples -- which is
// much closer to what a real match looks like than a pure gun-deck ship meeting a pure carronade one.

import * as THREE from 'three';
import { autoBuild, randomProfile, ARCHETYPES } from './autobuild.js';
import { makeRng, hashSeed } from './sim/rng.js';
import { cloneDesign } from './sim/ship.js';
import { audioState } from './audio/play.js';
import { MAX_PLAYERS } from './config.js';

const params = new URLSearchParams(location.search);
const raw = params.get('dev');

const drafted = raw === 'draft';
const types = raw && raw !== '1' && !drafted ? raw.split(',').map((s) => s.trim()) : null;
const valid = drafted || (types && types.every((t) => ARCHETYPES[t]));

export const dev = {
  enabled: raw !== null,
  autoplay: !!valid,
  drafted,
  types: valid && types ? types : null,
  speed: Number(params.get('x') || 1),
  // Matches normally take a random seed. Pinning it is what makes a bug found while playing
  // reproducible, and the whole design rests on same-seed-same-match being true.
  seed: params.has('seed') ? Number(params.get('seed')) : null,
  // Skip ahead to a later round to inspect the big hulls without playing four rounds.
  fromRound: Math.max(0, Math.min(4, Number(params.get('round') || 1) - 1)),
  // Autoplay everything up to this round, then leave the build phase open to look at.
  stopAtRound: Number(params.get('stop') || 0),
  // Auto-advance the intro and handoff overlays, but stop on a round or match result.
  holdResults: params.get('hold') === '1',
  // Autoplay stops at the end of one match. Looping for ever pins a CPU core, which is
  // exactly what a forgotten headless tab did.
  loop: params.get('loop') === '1',
  // How many ships. Two is the duel the game was tuned around; three and four are a melee.
  // ?dev=brawler,sniper,crusher names one archetype per seat and sets this by itself.
  players: clampPlayers(params.get('players') ?? (types ? types.length : 2)),
  // Play against the machine without a second browser: ?bots=1 fills the other seats. Clamped on its
  // own and not through clampPlayers, whose floor of two turned "no bots" into one.
  bots: Math.max(0, Math.min(MAX_PLAYERS - 2, Number(params.get('bots') || 0))),
  // Connect to a server instead of running the authority in this page. ?net=1 uses this origin;
  // ?net=ws://host:port/ws points somewhere else. ?room=ABCD joins rather than creates.
  net: params.get('net'),
  room: params.get('room'),
  watch: params.get('watch') === '1',
  name: params.get('name'),
  // The official Three.js FPS/MS panel is on for interactive dev sessions. Render tools opt out so
  // profiling measures the game rather than the panel repainting its own canvas.
  stats: params.get('stats') !== '0',
};

function clampPlayers(n) {
  return Math.max(2, Math.min(MAX_PLAYERS, Number(n) || 2));
}

if (raw !== null && !valid && raw !== '1') {
  console.warn(
    `dev: unknown archetypes "${raw}", expected "draft" or two of ${Object.keys(ARCHETYPES).join(', ')}`,
  );
}

// Autoplay and the Fill button both plan a ship on a throwaway copy and then ask the authority for
// the placements, because the authority owns the ship and a dev button does not get to write one.
//
// The plan has to be made *from* the offer, not filtered against it afterwards. Filtering was the
// first version and it was quietly broken: the shop shows five part types out of nine, so an
// archetype whose gun was not among them had every one of its guns dropped and put to sea with none.
// Two of those meet and neither can fire, the round ends on the five-second stalemate rule, and a
// whole autoplayed match is nothing but draws. It looked like a balance collapse and it was a
// harness bug -- the old harness wrote the design directly and never had to obey the shop.
function planFor(profile, offer, hullIndex) {
  const adapted = { ...profile };
  const offered = (ids) => ids.filter((id) => offer.includes(id));
  if (!offer.includes(adapted.gun)) {
    // Any gun the shop is actually selling, else there is nothing to arm with and the readout will
    // say so.
    adapted.gun = offered(['gundeck', 'carronade', 'swivel', 'longgun'])[0] ?? null;
  }
  if (adapted.second && !offer.includes(adapted.second)) adapted.second = null;
  // Timber is always on offer, so this always resolves.
  if (!offer.includes(adapted.armour)) adapted.armour = 'timber';
  if (!adapted.gun) return null;
  return adapted;
}

// Placements for whatever profile the URL asked for. Seeded from the match, so an autoplayed game
// replays identically with the same &seed.
export function devBuildCommands(design, hullIndex, scrap, seat, seed, offer) {
  const rng = makeRng(hashSeed(seed, hullIndex, seat, 5150));
  const base =
    dev.drafted || !dev.types
      ? randomProfile(rng, hullIndex)
      : ARCHETYPES[dev.types[seat % dev.types.length]];
  const side = rng.range(0, 1) < 0.5 ? 'port' : 'starboard';
  return placements(design, hullIndex, scrap, planFor(base, offer, hullIndex), offer, side);
}

function placements(design, hullIndex, scrap, profile, offer, side = 'port') {
  if (!profile) return [];
  const plan = cloneDesign(design);
  autoBuild(plan, hullIndex, scrap, profile, side);
  const out = [];
  for (const key in plan.parts) {
    if (design.parts[key]) continue;
    const id = plan.parts[key].id;
    // A mast or a crew berth the shop is not selling this round simply does not get bought, which is
    // the same answer a player would get.
    if (offer.includes(id)) out.push([key, id]);
  }
  return out;
}

// The Fill button: a plausible broadside ship out of whatever is on offer.
export function devFillCommands(design, hullIndex, scrap, offer) {
  return placements(design, hullIndex, scrap, planFor(ARCHETYPES.brawler, offer, hullIndex), offer);
}

// A Fill button, so manual playtesting doesn't mean clicking 38 cells.
export function attachFillButton(container, onFill) {
  if (!dev.enabled) return;
  // startBuild runs once per build phase, so guard against stacking up buttons.
  const existing = container.querySelector('#dev-fill');
  if (existing) {
    existing.onclick = onFill;
    return;
  }
  const btn = document.createElement('button');
  btn.id = 'dev-fill';
  btn.className = 'tool';
  btn.textContent = 'Fill (dev)';
  btn.style.flex = '1';
  btn.onclick = onFill;
  container.appendChild(btn);
}

// ---------------------------------------------------------------------------
// Playtest hooks
// ---------------------------------------------------------------------------
//
// The build phase is raycasted clicks on a WebGL canvas, which a headless driver cannot reach by
// querying the DOM. These expose it: `__dev.clickCell(dx, dz)` projects a hull cell to screen and
// dispatches the same pointer events a player would generate, so tools/shot.js can play a real
// build phase rather than only pressing the Fill button.
//
// Off unless ?dev is present, and it reads everything from the __game hook rather than holding its
// own references, so main.js does not have to know about it.

function screenAt(dx, dz) {
  const { sceneCtl } = globalThis.__game;
  const canvas = sceneCtl.renderer.domElement;
  const camera = sceneCtl.camera;
  // Build-phase ships sit at the origin unrotated, so cell offsets are world offsets.
  const CELL_SIZE = 2.4;
  const p = new THREE.Vector3(dx * CELL_SIZE, 0.5, dz * CELL_SIZE).project(camera);
  const rect = canvas.getBoundingClientRect();
  return {
    x: rect.left + ((p.x + 1) / 2) * rect.width,
    y: rect.top + ((1 - p.y) / 2) * rect.height,
  };
}

function fire(target, type, x, y, extra = {}) {
  target.dispatchEvent(
    new PointerEvent(type, { clientX: x, clientY: y, bubbles: true, pointerId: 1, ...extra }),
  );
}

export const hooks = {
  // Screen position of a hull cell, for debugging a miss.
  where: screenAt,

  clickCell(dx, dz) {
    const canvas = globalThis.__game.sceneCtl.renderer.domElement;
    const { x, y } = screenAt(dx, dz);
    fire(canvas, 'pointermove', x, y);
    fire(canvas, 'pointerdown', x, y);
    fire(canvas, 'pointerup', x, y);
    canvas.dispatchEvent(new MouseEvent('click', { clientX: x, clientY: y, bubbles: true }));
    return `clicked ${dx},${dz}`;
  },

  // Select an offered part by name, case-insensitive and prefix-matched.
  pickCard(name) {
    const want = name.toLowerCase();
    for (const card of document.querySelectorAll('.card-part')) {
      const label = card.querySelector('.pname').textContent.toLowerCase();
      if (label.startsWith(want)) {
        card.click();
        return `selected ${label}`;
      }
    }
    return `no card matching "${name}" in [${[...document.querySelectorAll('.pname')]
      .map((e) => e.textContent)
      .join(', ')}]`;
  },

  tool(id) {
    const btn = document.getElementById(id);
    if (!btn || btn.disabled) return `${id} unavailable`;
    btn.click();
    return `clicked ${id}`;
  },

  // Everything a playtest wants to assert on, in one readable blob. Reads the client rather than the
  // DOM wherever the client knows better: the score lives in the room's state now, and the score row
  // in the top bar is built from the roster and does not exist until a match starts.
  state() {
    const text = (sel) => document.querySelector(sel)?.textContent?.trim() ?? null;
    const g = globalThis.__game;
    const client = g.client;
    const room = client?.state.room ?? null;
    return {
      phase: g.phase,
      audio: audioState(),
      round: text('#round-label'),
      roundIndex: client?.state.round ?? 0,
      hull: text('#hull-label'),
      code: room?.code ?? null,
      seat: client?.state.seat ?? null,
      mySeats: client?.state.mySeats ?? [],
      players: room ? room.players.map((p) => ({ name: p.name, bot: p.bot, locked: p.locked })) : [],
      score: room ? room.players.map((p) => p.score) : [],
      timer: text('#timer'),
      scrap: text('#scrap-value'),
      offer: [...document.querySelectorAll('.card-part .pname')].map((e) => e.textContent),
      readout: [...document.querySelectorAll('.stat-row')].map((r) =>
        r.textContent.replace(/\s+/g, ' '),
      ),
      warnings: [...document.querySelectorAll('.warn')].map((e) => e.textContent),
      hint: text('#hint'),
      overlay: document.getElementById('overlay').classList.contains('hidden')
        ? null
        : { title: text('#ov-title'), button: text('#ov-btn') },
      // The wire, for a networked playtest. Null in a local game, where there is none.
      net: g.net && client && !client.isLocal
        ? {
            status: client.state.status,
            rtt: g.net.rtt,
            delay: Math.round(Math.max(g.net.delayMs, g.net.measuredDelayMs)),
            desyncs: g.net.desyncs,
            resyncs: g.net.resyncs,
            lateInputs: g.net.lateInputs,
            checked: g.net.checked,
          }
        : null,
      battle: g.battle
        ? {
            t: +g.battle.time.toFixed(2),
            tick: g.battle.tickCount,
            over: g.battle.over,
            reason: g.battle.reason,
            shot: g.battle.projectiles.length,
            ships: g.battle.ships.map((s) => ({
              cells: s.aliveCells,
              crew: s.crew,
              guns: s.guns.filter((gun) => gun.manned).length,
              ammo: s.ammo,
              out: s.out,
            })),
          }
        : null,
    };
  },

  // Press the overlay button, whatever it currently says. Every phase change a player makes goes
  // through it, so a driver needs exactly this and nothing more.
  proceed() {
    const btn = document.getElementById('ov-btn');
    if (!btn || btn.disabled || document.getElementById('overlay').classList.contains('hidden')) {
      return 'no overlay';
    }
    btn.click();
    return 'clicked';
  },

  // Advance the client by hand, for a driver whose tab is not the visible one.
  //
  // A hidden tab gets no requestAnimationFrame at all -- not throttled, stopped -- so the frame loop
  // does not run and the replay of the battle does not advance. That is right for a real player,
  // since there is nothing to draw, and useless for a harness driving four browsers at once, only
  // one of which can be in front. This is the same call the frame loop makes, plus the drain the
  // frame loop would have done. Advancing is idempotent: the replay's target tick comes from the
  // server's clock, so pumping twice in a frame reaches the same tick as pumping once.
  pump(dt = 0.05) {
    const g = globalThis.__game;
    if (!g.client) return 'no client';
    g.client.update(dt);
    if (g.battle) g.battle.effects.length = 0;
    return g.battle ? g.battle.tickCount : 'no battle';
  },

  // Switch a seat's ammunition the way a keypress would, for a driver that cannot press keys.
  ammo(seat, kind) {
    const client = globalThis.__game.client;
    if (!client) return 'no client';
    return client.setAmmo(seat, kind) ? `${seat} -> ${kind}` : 'not your seat';
  },
};

if (dev.enabled) globalThis.__dev = hooks;
