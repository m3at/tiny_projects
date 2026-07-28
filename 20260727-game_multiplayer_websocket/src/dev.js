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

import { autoBuild, randomProfile, ARCHETYPES } from './autobuild.js';
import { makeRng, hashSeed } from './sim/rng.js';

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
};

if (raw !== null && !valid && raw !== '1') {
  console.warn(
    `dev: unknown archetypes "${raw}", expected "draft" or two of ${Object.keys(ARCHETYPES).join(', ')}`,
  );
}

// Spend a build phase's scrap the way the named archetype would, or as a random draft. Seeded from
// the match so an autoplayed game replays identically with the same &seed.
export function devBuild(design, hullIndex, scrap, player, seed = 0) {
  if (!dev.autoplay) return scrap;
  if (dev.drafted) {
    const rng = makeRng(hashSeed(seed, hullIndex, player, 5150));
    const side = rng.range(0, 1) < 0.5 ? 'port' : 'starboard';
    return autoBuild(design, hullIndex, scrap, randomProfile(rng, hullIndex), side);
  }
  return autoBuild(design, hullIndex, scrap, ARCHETYPES[dev.types[player]]);
}

// Spend a purse on a plausible ship, for the Fill button. Returns leftover scrap.
export function devFill(design, hullIndex, scrap) {
  return autoBuild(design, hullIndex, scrap, ARCHETYPES.brawler);
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
  // Vector3 is not a module global here; borrow the class off an existing vector.
  const V = camera.position.constructor;
  // Build-phase ships sit at the origin unrotated, so cell offsets are world offsets.
  const CELL_SIZE = 2.4;
  const p = new V(dx * CELL_SIZE, 0.5, dz * CELL_SIZE).project(camera);
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

  // Everything a playtest wants to assert on, in one readable blob.
  state() {
    const text = (sel) => document.querySelector(sel)?.textContent?.trim() ?? null;
    const g = globalThis.__game;
    return {
      phase: g.phase,
      round: text('#round-label'),
      hull: text('#hull-label'),
      score: [text('#score-p1'), text('#score-p2')],
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
      battle: g.battle
        ? {
            t: +g.battle.time.toFixed(2),
            over: g.battle.over,
            reason: g.battle.reason,
            shot: g.battle.projectiles.length,
            ships: g.battle.ships.map((s) => ({
              cells: s.aliveCells,
              crew: s.crew,
              guns: s.guns.filter((gun) => gun.manned).length,
              ammo: s.ammo,
            })),
          }
        : null,
    };
  },
};

if (dev.enabled) globalThis.__dev = hooks;
