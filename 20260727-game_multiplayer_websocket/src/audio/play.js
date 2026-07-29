// The game's side of the audio: when to make a noise, and how many at once.
//
// Synthesis is in sfx.js and knows nothing about the game. This file knows nothing about synthesis.
//
// Two problems worth stating, because both are audible when got wrong:
//
//   A broadside is not sixteen separate sounds, but it is not one either. The first rule here was a
//   hard minimum gap per kind with everything inside it discarded, and tools/mix.js showed what
//   that cost: a quarter of all gunfire and nearly half of all hits thrown away, and thrown away
//   hardest exactly when the most was happening. A ship of the line and a sloop came out the same.
//   The rule now is a queue -- see slot() -- which keeps every event and spends level instead.
//
//   Nothing exists before a gesture. A suspended context's resume() returns a promise that never
//   settles until the user has interacted, and its clock does not advance, so anything scheduled
//   meanwhile piles onto one instant and all goes off together the moment it starts. So: create on
//   the first input, and emit nothing at all unless the context is actually running.

import { ARENA_RADIUS } from '../config.js';
import { createSfx } from './sfx.js';

// Per kind: how far apart to space voices, how much backlog to tolerate before dropping, how hard
// to duck as the kind piles up, and how long one voice rings for. Set from tools/mix.js against
// 150 real battles -- these numbers pass 100% of events at a lower peak level than the old rule
// managed while dropping a quarter of them.
const VOICE = {
  cannon: { gap: 0.028, lead: 0.3, duck: 0.5, ring: 0.45 },
  impact: { gap: 0.022, lead: 0.22, duck: 0.5, ring: 0.22 },
  splash: { gap: 0.05, lead: 0.15, duck: 0.5, ring: 0.32 },
  break: { gap: 0.14, lead: 0.3, duck: 0.4, ring: 1.1 },
  blast: { gap: 0.2, lead: 0.4, duck: 0.3, ring: 2.2 },
};

// Relative loudness before ducking. Measured peaks: a cannon 0.65, a detonation 0.8.
const LEVEL = {
  cannonBig: 1,
  cannonSmall: 0.75,
  // A hit landing has to be audible under gunfire, and twice now it has not been: 20 dB below a
  // cannon on the first pass, still 13 dB below on the second. Hits are the most common event in
  // the game and the only one that tells you the shot connected, so they are worth the headroom.
  impact: 1,
  splash: 0.55,
  break: 0.6,
  splinter: 0.8,
  blast: 1.1,
};

let ctx = null;
let sfx = null;
let muted = false;
let wantAmbience = false;

// Per kind: the next free moment, how much of it is already sounding, and when that was last
// measured. Plain objects rather than a Map; there are five keys and this runs every frame.
const cursor = {};
const energy = {};
const seen = {};

export const VOLUME = 0.7;

// The sea bed is the one long-lived voice, so it is the one that has to be started at the right
// moment. A suspended context's clock does not advance, so nodes started against it are silent and
// only earn a console warning; statechange is when it becomes safe.
//
// The gain is written here as well as at construction, and that is not redundant. Every path that can
// leave the master gain somewhere other than where it belongs -- a context built while muted, an
// automation left half applied, a browser that resumed the graph in some state of its own -- ends up
// here when the context reaches running, and this puts it right. It is also exactly what pressing the
// mute button twice used to do, which is how the fault was reported: no sound until the player toggled
// the sound off and on again.
function onRunning() {
  if (!sfx) return;
  if (ctx.state === 'running') {
    sfx.master.gain.cancelScheduledValues(ctx.currentTime);
    sfx.master.gain.setValueAtTime(muted ? 0 : VOLUME, ctx.currentTime);
  }
  sfx.ambience(wantAmbience && ctx.state === 'running');
}

function build() {
  ctx = new AudioContext();
  ctx.addEventListener('statechange', onRunning);
  sfx = createSfx(ctx, { volume: muted ? 0 : VOLUME });
  onRunning();
}

const UNLOCK = { capture: true, passive: true };
const GESTURES = ['pointerdown', 'pointerup', 'keydown', 'touchend', 'click'];

// Called from every gesture until the context is genuinely running. Never awaits resume(): that
// promise does not settle until the browser is ready, and awaiting it means scheduling against a
// clock that is not moving.
//
// It keeps listening until `ctx.state === 'running'`, and that is the fix for a whole class of
// silence. The first version stopped listening as soon as it had *called* resume() once, which
// assumes the call worked -- so a resume that was refused, a context that came up suspended anyway,
// or one the browser suspended again later left the game permanently mute with nothing able to try
// again. Several gesture kinds, because activation is not granted on the same event everywhere:
// Safari has historically wanted a touchend or a click where Chrome is happy with a pointerdown.
function unlock() {
  // Synthetic events -- a dispatchEvent from the dev harness, or element.click() -- do not grant
  // activation, and building a context that cannot start leaves a suspended one lying around and
  // logs a warning for every node anything tries to play through it.
  if (navigator.userActivation && !navigator.userActivation.hasBeenActive) return;
  if (!ctx) build();
  if (ctx.state !== 'running') {
    // Chrome returns a promise; older Safari returns undefined. Either way, do not await it.
    const resumed = ctx.resume();
    if (resumed && resumed.catch) resumed.catch(() => {});
    return; // stay subscribed: whether it worked is not known yet
  }
  onRunning();
  for (const kind of GESTURES) removeEventListener(kind, unlock, UNLOCK);
}

for (const kind of GESTURES) addEventListener(kind, unlock, UNLOCK);

// Coming back to a backgrounded tab. Browsers suspend an idle context on their own, and a player who
// switched away mid-match and came back to silence has no reason to guess that the mute button is
// what fixes it.
addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' || !ctx) return;
  if (ctx.state === 'running') return;
  const resumed = ctx.resume();
  if (resumed && resumed.catch) resumed.catch(() => {});
});

const live = () => sfx && ctx.state === 'running' && !muted;

// Where and how loudly the next voice of a kind should go, or null if it should be dropped.
//
// Events of a kind are laid out end to end at least `gap` apart, starting from now. A dozen guns
// arriving inside one frame become a rolling broadside over a third of a second instead of one
// flam, which is both what it sounded like and what stops them summing into a clipped lump. Only a
// backlog longer than `lead` is discarded, and by then the event is late enough to be a lie.
//
// `weight` is how much of the kind is already ringing, on a 0..1 scale. Level falls off as the
// inverse square root of it, so twelve guns are louder than three without being four times louder,
// and sfx.js uses the same number to make a heavy volley bassier rather than merely bigger.
function slot(kind) {
  const v = VOICE[kind];
  const now = ctx.currentTime;
  const at = cursor[kind] > now ? cursor[kind] : now;
  if (at - now > v.lead) return null;
  cursor[kind] = at + v.gap;

  const e = (energy[kind] || 0) * Math.exp(-(at - (seen[kind] ?? at)) / v.ring);
  energy[kind] = e + 1;
  seen[kind] = at;
  return { at, gain: 1 / Math.sqrt(1 + v.duck * e), weight: e / (e + 3) };
}

// Gentle stereo placement from a world x coordinate.
const panOf = (x) => Math.max(-1, Math.min(1, ((x || 0) / ARENA_RADIUS) * 0.7));

export function setMuted(on) {
  muted = on;
  if (sfx) sfx.master.gain.setTargetAtTime(on ? 0 : VOLUME, ctx.currentTime, 0.02);
  // Unmuting is a gesture, and it is the gesture a player reaches for when they cannot hear anything,
  // so take it as one: if the context never started, this is the moment to try again.
  if (!on) unlock();
  return muted;
}

export const isMuted = () => muted;

// Why is there no sound? There are four separate reasons there might not be, and from the outside they
// are indistinguishable, which is how the gesture bug below survived. Read by __dev.state().
let emitted = 0;

export function audioState() {
  return {
    built: !!ctx,
    state: ctx ? ctx.state : 'none',
    muted,
    activated: navigator.userActivation ? navigator.userActivation.hasBeenActive : null,
    live: !!live(),
    // How many voices have actually been handed to the synthesiser. Everything else here can look
    // right while this stays at zero, which is the only way to tell "silent" from "not playing".
    emitted,
    gain: sfx ? sfx.master.gain.value : null,
    // A context can report 'running' and still have a clock that is not advancing, in which case
    // everything scheduled against it is silent. This is the only way to see that from outside.
    clock: ctx ? +ctx.currentTime.toFixed(3) : null,
  };
}

export function setAmbience(on) {
  wantAmbience = on;
  onRunning();
}

// Buttons and refusals. Unqueued: they answer a click, so they are never dense and must never be
// late. See UI in sfx.js for what each one is.
export function ui(name) {
  if (!live()) return;
  emitted++;
  sfx.ui(name);
}

// Reads the same effect stream the renderer does, and is called just before main.js drains it.
export function consume(effects) {
  if (!live()) return;
  for (const e of effects) {
    emitted++;
    switch (e.type) {
      case 'muzzle': {
        const s = slot('cannon');
        if (!s) break;
        const base = e.big ? LEVEL.cannonBig : LEVEL.cannonSmall;
        sfx.cannon({ when: s.at, size: base * s.gain, pan: panOf(e.x), weight: s.weight });
        break;
      }
      case 'impact': {
        const s = slot('impact');
        if (!s) break;
        sfx.impact({ when: s.at, size: LEVEL.impact * s.gain, pan: panOf(e.x), kind: e.kind });
        break;
      }
      case 'splash': {
        const s = slot('splash');
        if (s) sfx.splash({ when: s.at, size: LEVEL.splash * s.gain, pan: panOf(e.x) });
        break;
      }
      case 'sink': {
        const s = slot('splash');
        if (s) sfx.splash({ when: s.at, size: LEVEL.splash * 1.45 * s.gain, pan: panOf(e.x) });
        break;
      }
      // A mast going over the side is a second and a half of rigging and rope; anything else is a
      // cell coming apart, which is short. Using the long sound for both made every destroyed
      // timber sound like a dismasting, and there are twenty of those a battle.
      case 'destroy': {
        const mast = e.part === 'mast';
        const s = slot('break');
        if (!s) break;
        const opts = { when: s.at, size: (mast ? LEVEL.break : LEVEL.splinter) * s.gain, pan: panOf(e.x) };
        if (mast) sfx.timberBreak(opts);
        else sfx.splinter(opts);
        break;
      }
      case 'sever': {
        const s = slot('break');
        if (s) sfx.splinter({ when: s.at, size: LEVEL.splinter * s.gain, pan: panOf(e.x) });
        break;
      }
      case 'detonate': {
        const s = slot('blast');
        if (s) sfx.detonation({ when: s.at, size: LEVEL.blast * s.gain, pan: panOf(e.x) });
        break;
      }
      default:
        break;
    }
  }
}
