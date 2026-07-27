// Dev harness, off unless asked for by URL. Lets a whole match play itself so the battle
// can be watched (or screenshotted) without clicking through two build phases per round.
//
//   ?dev=brawler,sniper       both sides auto-built each round, overlays auto-advance
//   ?dev=brawler,sniper&x=3   run at 3x speed
//   ?dev=1                    manual build, but adds a Fill button to the build panel
//   ?dev=brawler,sniper&stop=2  autoplay round 1, then hold on round 2's build phase
//   ?dev=brawler,sniper&hold=1  autoplay, but stop on each result screen
//   &loop=1                   keep starting fresh matches (off by default)
//
// Archetype names come from autobuild.js: brawler, sniper, harasser, crusher.

import { autoBuild, ARCHETYPES } from './autobuild.js';

const params = new URLSearchParams(location.search);
const raw = params.get('dev');

const types = raw && raw !== '1' ? raw.split(',').map((s) => s.trim()) : null;
const valid = types && types.every((t) => ARCHETYPES[t]);

export const dev = {
  enabled: raw !== null,
  autoplay: !!valid,
  types: valid ? types : null,
  speed: Number(params.get('x') || 1),
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
  console.warn(`dev: unknown archetypes "${raw}", expected two of ${Object.keys(ARCHETYPES).join(', ')}`);
}

// Spend a build phase's scrap the way the named archetype would.
export function devBuild(design, hullIndex, scrap, player) {
  if (!dev.autoplay) return scrap;
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
