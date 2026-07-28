// Every sound in the game, synthesised. There are no audio files here and there should not be: the
// whole project stays a directory of text.
//
// Takes an AudioContext rather than making one, so the same code can be rendered through an
// OfflineAudioContext and measured. That is what tools/audio.js does, and it is the only way to
// catch the three faults you cannot hear by clicking around: samples over 1.0, a DC offset, and a
// step discontinuity at onset. Every parameter below was checked that way.
//
// The rules that matter, each one learned from a measurement:
//
//   Envelopes anchor at zero, ramp up over about 2ms, decay exponentially to a whisker above zero,
//   then hard-zero. An instant gain step on a noise source measures a 0.82 sample-to-sample jump at
//   onset, which is an audible click; 2ms brings it to 0.02 and still reads as instant.
//   exponentialRampToValueAtTime(0) throws, and ramping *from* zero silently becomes a step.
//
//   Noise is bipolar. Math.random() alone is 0..1, which is a DC offset of 0.2 that thumps on every
//   start and stop.
//
//   Voices are fire-and-forget. Building a cannon costs about 50 microseconds and the context
//   collects the chain when the source ends; pooling gain nodes buys nothing and reintroduces
//   clicks from stale automation.
//
//   Layers, not volume. A cannon is a crack, a boom and a tail; a detonation is not a louder cannon
//   but a longer, lower sweep with a second punch and debris. Wood sounds like wood because of three
//   inharmonic resonances -- harmonic ones sound like a marimba.

const EPS = 0.0001; // exponential ramps must target something above zero
const ATTACK = 0.002;
const NOISE_SEC = 2;

export function createSfx(ctx, { volume = 0.7 } = {}) {
  // Master gain into a compressor doing limiter duty. Measured: twelve cannons at once peak at 1.14
  // and clip 23 samples without it, 0.74 and none with it. At normal levels it is barely engaged,
  // so it is insurance rather than a crutch. There is no makeup gain, so level is set ahead of it.
  const master = ctx.createGain();
  master.gain.value = volume;
  const limiter = ctx.createDynamicsCompressor();
  limiter.threshold.value = -14;
  limiter.knee.value = 6;
  limiter.ratio.value = 12;
  limiter.attack.value = 0.003; // faster than this eats the cannon's crack
  limiter.release.value = 0.15;
  master.connect(limiter).connect(ctx.destination);

  const noiseBuf = ctx.createBuffer(1, ctx.sampleRate * NOISE_SEC, ctx.sampleRate);
  const data = noiseBuf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;

  const rnd = (a, b) => a + Math.random() * (b - a);

  // A random offset and a slight detune, so two overlapping shots never phase-lock into one flam.
  function noise(t, dur, rate = 1) {
    const src = ctx.createBufferSource();
    src.buffer = noiseBuf;
    src.playbackRate.value = rate;
    const offset = Math.random() * (NOISE_SEC - dur * rate - 0.01);
    src.start(t, Math.max(0, offset), dur * rate + 0.01);
    return src;
  }

  function env(t, peak, dur, attack = ATTACK) {
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + attack);
    g.gain.exponentialRampToValueAtTime(peak * EPS, t + dur);
    g.gain.setValueAtTime(0, t + dur);
    return g;
  }

  function filt(type, freq, q = 1) {
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    return f;
  }

  // A pitch-dropping body. Below about 60Hz a sine reads as pressure rather than as a note, which is
  // why every boom has to end low.
  function bodyTone(t, f0, f1, dur, peak, type = 'sine') {
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + dur * 0.6);
    const g = env(t, peak, dur, 0.004);
    o.connect(g);
    o.start(t);
    o.stop(t + dur + 0.02);
    return g;
  }

  // Where a voice lands in the stereo field. Gentle: the camera is close and a top-down view with
  // hard panning is disorienting. pan 0 skips the node entirely.
  function out(pan) {
    if (!pan) return master;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan));
    p.connect(master);
    return p;
  }

  const at = (when) => (when || ctx.currentTime) + 0.001;

  // ---- cannon: crack, boom, tail -------------------------------------------------------------
  function cannon({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);
    const j = rnd(0.88, 1.14); // per-shot pitch jitter, so a broadside is not one sound repeated
    const lvl = rnd(0.85, 1) * size;

    const crack = noise(t, 0.06, rnd(0.9, 1.1));
    crack.connect(filt('bandpass', 1900 * j, 0.8)).connect(env(t, 0.5 * lvl, 0.05, 0.0008)).connect(dest);

    bodyTone(t + 0.004, 150 * j, 42 * j, 0.34 * rnd(0.9, 1.1), 0.85 * lvl).connect(dest);

    // The lowpass sweeping down is what reads as distance and air absorption.
    const tail = noise(t, 0.5, rnd(0.95, 1.05));
    const lp = filt('lowpass', 1100 * j, 1.1);
    lp.frequency.setValueAtTime(1100 * j, t);
    lp.frequency.exponentialRampToValueAtTime(160, t + 0.4);
    tail.connect(lp).connect(env(t, 0.55 * lvl, 0.42, 0.004)).connect(dest);
  }

  // ---- ball into timber: modal synthesis -----------------------------------------------------
  function impact({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);
    const j = rnd(0.85, 1.2);
    const lvl = rnd(0.8, 1) * size;

    const exciter = noise(t, 0.03, 1);
    const click = env(t, 1, 0.025, 0.0006);
    exciter.connect(click);
    // Inharmonic ratios: this is the whole difference between a hull and a xylophone.
    for (const [f, q, a] of [
      [190, 11, 0.45],
      [365, 9, 0.3],
      [712, 7, 0.16],
    ]) {
      click.connect(filt('bandpass', f * j, q)).connect(env(t, a * lvl, rnd(0.1, 0.2), 0.0015)).connect(dest);
    }
    bodyTone(t, 110 * j, 65 * j, 0.1, 0.4 * lvl, 'triangle').connect(dest);
    const knock = noise(t, 0.1, 1);
    knock.connect(filt('lowpass', 620 * j, 0.9)).connect(env(t, 0.28 * lvl, 0.08, 0.001)).connect(dest);
  }

  // ---- a ball into the sea -------------------------------------------------------------------
  function splash({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);
    const lvl = rnd(0.75, 1) * size;

    const n = noise(t, 0.35, rnd(0.95, 1.1));
    const bp = filt('bandpass', 700, 0.7);
    bp.frequency.setValueAtTime(rnd(600, 850), t);
    bp.frequency.exponentialRampToValueAtTime(rnd(2200, 3200), t + 0.09);
    bp.frequency.exponentialRampToValueAtTime(900, t + 0.3); // water closing over
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.5 * lvl, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.12 * lvl, t + 0.07); // spray, then wash
    g.gain.exponentialRampToValueAtTime(0.5 * lvl * EPS, t + 0.3);
    g.gain.setValueAtTime(0, t + 0.3);
    n.connect(bp).connect(g).connect(dest);

    // A bubble's pitch rises as it collapses. A falling sweep here sounds like a gunshot instead.
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(rnd(170, 260), t + 0.02);
    o.frequency.exponentialRampToValueAtTime(rnd(520, 800), t + 0.1);
    o.connect(env(t + 0.02, 0.14 * lvl, 0.1, 0.006)).connect(dest);
    o.start(t + 0.02);
    o.stop(t + 0.14);
  }

  // ---- the magazine going up -----------------------------------------------------------------
  function detonation({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);

    const crack = noise(t, 0.12, 1);
    crack.connect(filt('highpass', 900, 0.7)).connect(env(t, 0.55 * size, 0.1, 0.0008)).connect(dest);

    bodyTone(t, 95, 24, 1.5, 0.95 * size).connect(dest);
    bodyTone(t + 0.07, 140, 38, 0.7, 0.4 * size, 'triangle').connect(dest); // second punch

    const rumble = noise(t, 2.2, 1);
    const lp = filt('lowpass', 2400, 1.2);
    lp.frequency.setValueAtTime(2400, t);
    lp.frequency.exponentialRampToValueAtTime(70, t + 1.2);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.8 * size, t + 0.006);
    g.gain.setTargetAtTime(0, t + 0.02, 0.45);
    g.gain.setValueAtTime(0, t + 2.15); // setTargetAtTime never arrives; force the end
    rumble.connect(lp).connect(g).connect(dest);

    for (let i = 0; i < 14; i++) {
      const dt = t + 0.1 + Math.random() * 1.1;
      const grain = noise(dt, 0.05, 1);
      grain
        .connect(filt('bandpass', rnd(400, 2600), 5))
        .connect(env(dt, rnd(0.04, 0.13) * size, rnd(0.03, 0.08), 0.001))
        .connect(dest);
    }
  }

  // ---- a mast going over the side ------------------------------------------------------------
  function timberBreak({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);

    // Stick-slip: grains spaced by the square of their index, so the creak accelerates.
    for (let i = 0; i < 18; i++) {
      const p = i / 17;
      const dt = t + p * p * 0.45;
      const grain = noise(dt, 0.04, 1);
      grain
        .connect(filt('bandpass', rnd(700, 2400) * (1 + p * 0.5), 9))
        .connect(env(dt, rnd(0.06, 0.2) * size, rnd(0.02, 0.05), 0.001))
        .connect(dest);
    }
    const groan = ctx.createOscillator();
    groan.type = 'sawtooth';
    groan.frequency.setValueAtTime(rnd(115, 150), t);
    groan.frequency.exponentialRampToValueAtTime(62, t + 0.5);
    groan.connect(filt('bandpass', 320, 4)).connect(env(t, 0.3 * size, 0.55, 0.03)).connect(dest);
    groan.start(t);
    groan.stop(t + 0.6);

    const snap = noise(t + 0.46, 0.14, 1);
    snap.connect(filt('bandpass', 1500, 1.4)).connect(env(t + 0.46, 0.5 * size, 0.12, 0.0008)).connect(dest);
    bodyTone(t + 0.46, 180, 55, 0.3, 0.5 * size, 'triangle').connect(dest);
    splash({ when: t + 0.62, size: 0.9 * size, pan });
  }

  // ---- interface ticks: almost inaudible on purpose ------------------------------------------
  function tick({ when = 0, kind = 'place' } = {}) {
    const t = at(when);
    const f = kind === 'select' ? rnd(1500, 1700) : kind === 'deny' ? 260 : rnd(1050, 1200);
    const o = ctx.createOscillator();
    o.type = kind === 'deny' ? 'square' : 'triangle';
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * 0.8, t + 0.03);
    o.connect(env(t, kind === 'deny' ? 0.045 : 0.055, 0.035, 0.0015)).connect(master);
    o.start(t);
    o.stop(t + 0.05);
    // A whisper of noise makes it a tock rather than a beep.
    const n = noise(t, 0.02, 1);
    n.connect(filt('highpass', 2500, 0.7)).connect(env(t, 0.03, 0.015, 0.0005)).connect(master);
  }

  // ---- sea and wind --------------------------------------------------------------------------
  // Two filtered beds off one looping source, moved by slow LFOs so it never sits still. It exists
  // so the quiet between volleys does not sound like the game has stopped.
  let amb = null;
  function ambience(on) {
    if (on && !amb) {
      const t = ctx.currentTime;
      const src = ctx.createBufferSource();
      src.buffer = noiseBuf;
      src.loop = true;
      const bed = ctx.createGain();
      bed.gain.setValueAtTime(0, t);
      bed.gain.linearRampToValueAtTime(1, t + 2); // never snap ambience on
      bed.connect(master);

      const windFilter = filt('lowpass', 380, 0.6);
      const windGain = ctx.createGain();
      windGain.gain.value = 0.035;
      const gust = ctx.createOscillator();
      gust.type = 'sine';
      gust.frequency.value = 0.07;
      const gustLevel = ctx.createGain();
      gustLevel.gain.value = 0.022;
      gust.connect(gustLevel).connect(windGain.gain); // an LFO drives a param, not an input
      const gustCut = ctx.createGain();
      gustCut.gain.value = 170;
      gust.connect(gustCut).connect(windFilter.frequency);
      src.connect(windFilter).connect(windGain).connect(bed);

      const seaFilter = filt('bandpass', 600, 0.5);
      const seaGain = ctx.createGain();
      seaGain.gain.value = 0.028;
      const swell = ctx.createOscillator();
      swell.type = 'sine';
      swell.frequency.value = 0.13;
      const swellLevel = ctx.createGain();
      swellLevel.gain.value = 0.014;
      swell.connect(swellLevel).connect(seaGain.gain);
      src.connect(seaFilter).connect(seaGain).connect(bed);

      src.start(t);
      gust.start(t);
      swell.start(t);
      amb = { src, gust, swell, bed };
    } else if (!on && amb) {
      const t = ctx.currentTime;
      const dying = amb;
      amb = null;
      dying.bed.gain.cancelScheduledValues(t);
      dying.bed.gain.setValueAtTime(dying.bed.gain.value, t);
      dying.bed.gain.linearRampToValueAtTime(0, t + 1.2);
      dying.src.stop(t + 1.3);
      dying.gust.stop(t + 1.3);
      dying.swell.stop(t + 1.3);
    }
  }

  return { master, cannon, impact, splash, detonation, timberBreak, tick, ambience };
}
