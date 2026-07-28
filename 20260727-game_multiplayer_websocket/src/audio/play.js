// The game's side of the audio: when to make a noise, and how many at once.
//
// Synthesis is in sfx.js and knows nothing about the game. This file knows nothing about synthesis.
//
// Two problems worth stating, because both are audible when got wrong:
//
//   A broadside is one sound, not sixteen. A ship of the line fires sixteen guns and every hit
//   raises splinters; one voice per event is mud, and the compressor spends the whole battle
//   clamped. Each kind gets a minimum spacing, and anything arriving inside it is dropped -- these
//   are events nobody could pick out individually anyway.
//
//   Nothing exists before a gesture. A suspended context's resume() returns a promise that never
//   settles until the user has interacted, and its clock does not advance, so anything scheduled
//   meanwhile piles onto one instant and all goes off together the moment it starts. So: create on
//   the first input, and emit nothing at all unless the context is actually running.

import { ARENA_RADIUS } from '../config.js';
import { createSfx } from './sfx.js';

// Shortest gap between two sounds of a kind, in seconds. Set by ear against a ship of the line,
// which is the worst case by a wide margin.
const SPACING = {
  cannon: 0.05,
  impact: 0.04,
  splash: 0.1,
  break: 0.22,
  blast: 0.25,
};

// Relative loudness. Measured peaks: a cannon 0.65, a detonation 0.8, twelve cannons together 0.69.
const LEVEL = {
  cannonBig: 1,
  cannonSmall: 0.75,
  // A hit landing has to be audible under gunfire, which the first measured pass was not: it sat
  // nearly 20 dB below a cannon.
  impact: 0.6,
  splash: 0.45,
  break: 0.6,
  blast: 1.1,
};

let ctx = null;
let sfx = null;
let muted = false;
let wantAmbience = false;
const lastAt = {};

function build() {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) return;
  ctx = new Ctx();
  sfx = createSfx(ctx, { volume: muted ? 0 : 0.7 });
  if (wantAmbience) sfx.ambience(true);
}

// Called from the first pointer or key event anywhere in the page. Never awaits resume().
function unlock() {
  if (!ctx) build();
  if (ctx && ctx.state !== 'running') ctx.resume();
}

if (typeof addEventListener === 'function') {
  addEventListener('pointerdown', unlock, { once: true, capture: true });
  addEventListener('keydown', unlock, { once: true, capture: true });
}

const live = () => sfx && ctx.state === 'running' && !muted;

function spaced(kind) {
  const gap = SPACING[kind];
  const now = ctx.currentTime;
  if (lastAt[kind] !== undefined && now - lastAt[kind] < gap) return false;
  lastAt[kind] = now;
  return true;
}

// Gentle stereo placement from a world x coordinate.
const panOf = (x) => Math.max(-1, Math.min(1, ((x || 0) / ARENA_RADIUS) * 0.7));

export function setMuted(on) {
  muted = on;
  if (sfx) sfx.master.gain.setTargetAtTime(on ? 0 : 0.7, ctx.currentTime, 0.02);
  return muted;
}

export const isMuted = () => muted;

export function setAmbience(on) {
  wantAmbience = on;
  if (sfx) sfx.ambience(on);
}

export function ui(kind) {
  if (!live()) return;
  sfx.tick({ kind });
}

// Reads the same effect stream the renderer does, and is called just before main.js drains it.
export function consume(effects) {
  if (!live()) return;
  for (const e of effects) {
    switch (e.type) {
      case 'muzzle':
        if (spaced('cannon')) {
          sfx.cannon({ size: e.big ? LEVEL.cannonBig : LEVEL.cannonSmall, pan: panOf(e.x) });
        }
        break;
      case 'impact':
        if (spaced('impact')) sfx.impact({ size: LEVEL.impact, pan: panOf(e.x) });
        break;
      case 'splash':
        if (spaced('splash')) sfx.splash({ size: LEVEL.splash, pan: panOf(e.x) });
        break;
      case 'destroy':
      case 'sever':
        if (spaced('break')) sfx.timberBreak({ size: LEVEL.break, pan: panOf(e.x) });
        break;
      case 'detonate':
        if (spaced('blast')) sfx.detonation({ size: LEVEL.blast, pan: panOf(e.x) });
        break;
      default:
        break;
    }
  }
}
