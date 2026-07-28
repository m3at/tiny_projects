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
  //
  // `weight` is how much gunfire is already sounding, 0 for an isolated shot and 1 in the middle of
  // a broadside. It does not make the shot louder -- play.js is turning the level *down* by then,
  // or twelve guns would clip. It changes the balance instead: the crack ducks hard, the body ducks
  // little, and the tail runs longer and darker. Mass gunfire is bassier and boomier than one gun,
  // so a full broadside now sounds different from three shots rather than merely more frequent.
  // Without this every busy moment came out the same, which was the original complaint.
  function cannon({ when = 0, size = 1, pan = 0, weight = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);
    const j = rnd(0.84, 1.18); // per-shot pitch jitter, so a broadside is not one sound repeated
    const lvl = rnd(0.85, 1) * size;

    const crack = noise(t, 0.06, rnd(0.9, 1.1));
    crack
      .connect(filt('bandpass', 1900 * j, 0.8))
      .connect(env(t, 0.5 * lvl * (1 - weight * 0.55), 0.05, 0.0008))
      .connect(dest);

    // Deeper as the volley builds: a wall of guns has no single identifiable pitch.
    const drop = 1 - weight * 0.15;
    bodyTone(t + 0.004, 150 * j * drop, 42 * j * drop, 0.34 * rnd(0.9, 1.1) * (1 + weight * 0.5),
      0.85 * lvl * (1 + weight * 0.15)).connect(dest);

    // The lowpass sweeping down is what reads as distance and air absorption.
    const len = 0.42 * (1 + weight * 0.8);
    const tail = noise(t, len + 0.08, rnd(0.95, 1.05));
    const lp = filt('lowpass', 1100 * j, 1.1);
    lp.frequency.setValueAtTime(1100 * j * (1 - weight * 0.35), t);
    lp.frequency.exponentialRampToValueAtTime(160 - weight * 60, t + len * 0.95);
    tail.connect(lp).connect(env(t, 0.55 * lvl, len, 0.004)).connect(dest);
  }

  // ---- ball into timber: modal synthesis -----------------------------------------------------
  //
  // Three timbers rather than one. Hits are the most frequent sound in the game by a wide margin --
  // tools/mix.js counts 7.7 a second, nearly twice the gunfire -- so a single recipe with pitch
  // jitter on top reads as one sound stuttering. Each set is inharmonic, which is the whole
  // difference between a hull and a xylophone; what varies between them is how far apart and how
  // long-ringing, which is the difference between a heavy frame and a thin plank.
  const TIMBERS = [
    [[190, 11, 0.45], [365, 9, 0.3], [712, 7, 0.16]], // deep and solid
    [[240, 14, 0.4], [521, 11, 0.26], [889, 8, 0.2]], // tighter, more of a knock
    [[152, 9, 0.5], [287, 8, 0.34], [566, 6, 0.14]], // dull and heavy
  ];

  function impact({ when = 0, size = 1, pan = 0, kind = 'round' } = {}) {
    const t = at(when);
    const dest = out(pan);
    const lvl = rnd(0.8, 1) * size;

    // Grape is a cloud of small shot, not one ball: a patter of little strikes over about 50ms,
    // bright and bodiless. It should be unmistakable from round shot, because which one is loaded
    // is the only decision the player makes during a battle.
    if (kind === 'grape') {
      const hits = 5 + Math.floor(Math.random() * 4);
      for (let i = 0; i < hits; i++) {
        const dt = t + Math.random() * 0.05;
        const g = noise(dt, 0.03, 1);
        g.connect(filt('bandpass', rnd(900, 2600), 6))
          .connect(env(dt, rnd(0.11, 0.28) * lvl, rnd(0.015, 0.04), 0.0008))
          .connect(dest);
      }
      const rattle = noise(t, 0.09, 1);
      rattle.connect(filt('highpass', 1400, 0.8)).connect(env(t, 0.3 * lvl, 0.07, 0.001)).connect(dest);
      // A little body, or a cloud of small shot reads as static rather than as metal on oak.
      bodyTone(t, 190, 120, 0.07, 0.16 * lvl, 'triangle').connect(dest);
      return;
    }

    const j = rnd(0.85, 1.2);
    const exciter = noise(t, 0.03, 1);
    const click = env(t, 1, 0.025, 0.0006);
    exciter.connect(click);
    const timber = TIMBERS[Math.floor(Math.random() * TIMBERS.length)];
    for (const [f, q, a] of timber) {
      click.connect(filt('bandpass', f * j, q)).connect(env(t, a * lvl, rnd(0.1, 0.2), 0.0015)).connect(dest);
    }
    bodyTone(t, 110 * j, 65 * j, 0.1, 0.4 * lvl, 'triangle').connect(dest);
    const knock = noise(t, 0.1, 1);
    knock.connect(filt('lowpass', 620 * j, 0.9)).connect(env(t, 0.28 * lvl, 0.08, 0.001)).connect(dest);

    // Now and then a ball glances off instead of biting, and whines away across the water. Rare on
    // purpose: it is the sound you notice, so it has to stay an event.
    if (Math.random() < 0.12) {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      const f0 = rnd(1300, 2100);
      o.frequency.setValueAtTime(f0, t + 0.01);
      o.frequency.exponentialRampToValueAtTime(f0 * rnd(0.3, 0.45), t + 0.34);
      o.connect(filt('bandpass', 1700, 7)).connect(env(t + 0.01, 0.11 * lvl, 0.34, 0.006)).connect(dest);
      o.start(t + 0.01);
      o.stop(t + 0.37);
    }
  }

  // ---- a cell coming apart -------------------------------------------------------------------
  // Distinct from timberBreak, which is a mast going over the side and takes a second and a half.
  // Most destroyed cells are a gun or a stack of timber, and they happen often enough that the long
  // sound was both wrong and in the way.
  function splinter({ when = 0, size = 1, pan = 0 } = {}) {
    const t = at(when);
    const dest = out(pan);
    for (let i = 0; i < 7; i++) {
      const dt = t + Math.random() * 0.13;
      const g = noise(dt, 0.05, 1);
      g.connect(filt('bandpass', rnd(600, 2200), 7))
        .connect(env(dt, rnd(0.08, 0.2) * size, rnd(0.03, 0.07), 0.001))
        .connect(dest);
    }
    const crunch = noise(t, 0.2, 1);
    crunch.connect(filt('lowpass', 900, 1)).connect(env(t, 0.32 * size, 0.16, 0.002)).connect(dest);
    bodyTone(t, 165, 72, 0.22, 0.3 * size, 'triangle').connect(dest);
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

  // ---- interface: layered recipes ------------------------------------------------------------
  //
  // Layered rather than one tock with the pitch moved, in the shape cuelume uses
  // (github.com/Danilaa1/cuelume): short layers with individual offsets, and optionally a soft echo
  // tail. The frequencies are not cuelume's -- theirs are bright and glassy, which is right for a
  // web interface and wrong next to cannon fire.
  //
  // The echo is explicit taps -- the layers rendered again, quieter and duller -- rather than a
  // feedback delay. Oscillators stop themselves; a DelayNode does not, so a feedback loop needs
  // cleanup timers and stays alive in the graph between presses. Taps also render identically
  // offline, which is what tools/audio.js needs.
  //
  // Everything here is short on purpose. These fire while a player is working quickly, and a sound
  // that outlasts the action it reports starts overlapping itself.
  const UI = {
    // Picking a card off the offer. Bright and tiny.
    select: {
      layers: [
        { tone: 'triangle', f: 1580, jitter: 0.06, glide: 0.82, peak: 0.072, dur: 0.03, attack: 0.0015 },
        { noise: 'highpass', f: 2600, q: 0.7, peak: 0.04, dur: 0.014, attack: 0.0005 },
      ],
    },
    // Setting a part into a cell. A shade lower than select, so a build phase has two notes in it
    // rather than one repeated.
    place: {
      layers: [
        { tone: 'triangle', f: 1120, jitter: 0.07, glide: 0.8, peak: 0.08, dur: 0.035, attack: 0.0015 },
        { noise: 'highpass', f: 2400, q: 0.7, peak: 0.043, dur: 0.015, attack: 0.0005 },
      ],
    },
    // A soft wooden knock for the secondary tools.
    press: {
      layers: [
        { noise: 'bandpass', f: 1600, q: 1.4, peak: 0.13, dur: 0.022 },
        { tone: 'triangle', f: 340, peak: 0.035, dur: 0.03, attack: 0.002 },
      ],
    },
    // The commit. Two notes, E then B, and done inside a fifth of a second: it marks the moment
    // rather than celebrating it. An earlier three-note version with a double echo ran half a
    // second and started to feel like an award.
    confirm: {
      layers: [
        { noise: 'bandpass', f: 2200, q: 1.6, peak: 0.085, dur: 0.014 },
        { tone: 'sine', f: 659.25, peak: 0.055, dur: 0.06, attack: 0.003 },
        { tone: 'sine', f: 987.77, offset: 0.042, peak: 0.055, dur: 0.12, attack: 0.003 },
      ],
      echo: { delay: 0.085, taps: 1, decay: 0.26, lowpass: 3000 },
    },
    // A refusal that is calm and recoverable: a dull knock, then two notes falling.
    deny: {
      layers: [
        { noise: 'bandpass', f: 780, q: 1.1, peak: 0.15, dur: 0.045 },
        { tone: 'triangle', f: 415.3, offset: 0.028, peak: 0.05, dur: 0.1, attack: 0.004 },
        { tone: 'triangle', f: 329.63, offset: 0.105, peak: 0.045, dur: 0.16, attack: 0.004 },
      ],
    },
  };

  function uiLayer(layer, t, dest, scale) {
    const start = t + (layer.offset || 0);
    const g = env(start, layer.peak * scale, layer.dur, layer.attack ?? 0.001);
    if (layer.tone) {
      const f = layer.jitter ? layer.f * rnd(1 - layer.jitter, 1 + layer.jitter) : layer.f;
      const o = ctx.createOscillator();
      o.type = layer.tone;
      o.frequency.setValueAtTime(f, start);
      // A small downward glide is most of what makes a tone read as a click rather than a beep.
      if (layer.glide) o.frequency.exponentialRampToValueAtTime(f * layer.glide, start + layer.dur);
      o.connect(g).connect(dest);
      o.start(start);
      o.stop(start + layer.dur + 0.02);
    } else {
      noise(start, layer.dur, 1)
        .connect(filt(layer.noise, layer.f, layer.q ?? 1))
        .connect(g)
        .connect(dest);
    }
  }

  function ui(name, { when = 0 } = {}) {
    const recipe = UI[name];
    if (!recipe) return;
    const t = at(when);
    for (const layer of recipe.layers) uiLayer(layer, t, master, 1);
    if (!recipe.echo) return;
    const { delay, taps, decay, lowpass } = recipe.echo;
    const wet = filt('lowpass', lowpass, 0.7);
    wet.connect(master);
    for (let i = 1; i <= taps; i++) {
      for (const layer of recipe.layers) uiLayer(layer, t + delay * i, wet, decay ** i);
    }
  }

  // ---- sea and wind --------------------------------------------------------------------------
  // Two filtered beds off one looping source, moved by slow LFOs so it never sits still. It exists
  // so the quiet between volleys does not sound like the game has stopped, which means it only has
  // to be present, not audible. Backed off a quarter from the first pass, where it sat close enough
  // to the front to be listened to rather than lived in.
  const AMBIENCE = 0.75;
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
      windGain.gain.value = 0.035 * AMBIENCE;
      const gust = ctx.createOscillator();
      gust.type = 'sine';
      gust.frequency.value = 0.07;
      const gustLevel = ctx.createGain();
      gustLevel.gain.value = 0.022 * AMBIENCE;
      gust.connect(gustLevel).connect(windGain.gain); // an LFO drives a param, not an input
      const gustCut = ctx.createGain();
      gustCut.gain.value = 170;
      gust.connect(gustCut).connect(windFilter.frequency);
      src.connect(windFilter).connect(windGain).connect(bed);

      const seaFilter = filt('bandpass', 600, 0.5);
      const seaGain = ctx.createGain();
      seaGain.gain.value = 0.028 * AMBIENCE;
      const swell = ctx.createOscillator();
      swell.type = 'sine';
      swell.frequency.value = 0.13;
      const swellLevel = ctx.createGain();
      swellLevel.gain.value = 0.014 * AMBIENCE;
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

  return { master, cannon, impact, splash, splinter, detonation, timberBreak, ui, ambience };
}
