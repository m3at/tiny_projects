// Renders every sound through an OfflineAudioContext and measures it. Needs the dev server and
// headless Chrome up (./tools/dev.sh), because Web Audio only exists in a browser.
//
//   node tools/audio.js
//
// This catches the three faults you cannot hear by clicking around, and which nothing else in the
// project would notice:
//
//   clipped   samples outside -1..1. Layered cannons and a magazine going up at the same moment is
//             the case that clips, so there is a deliberately abusive pile-up at the end.
//   dc        mean sample value. Anything far from zero means a noise buffer built from
//             Math.random() without centring, which thumps on every start and stop.
//   onset     how abruptly the sound leaves silence, as the first non-silent sample's jump divided
//             by the sound's own peak. An envelope with no attack ramp starts at full level, which
//             scores near 1 and clicks; a 2ms ramp starts at a fraction of a percent.
//
//             The obvious metric -- largest sample-to-sample jump near the start -- does not work,
//             and flagged the detonation on the first run here. High frequencies have large jumps
//             between adjacent samples by nature, so a bright sound scores high whether or not it
//             clicks. Measuring the step out of silence separates the two.
//
// Offline contexts are exempt from the autoplay gesture requirement, which is what makes this
// possible headlessly at all.

import { attach } from './cdp.js';

const page = await attach();
await page.open('');

// The measuring is done in the page so it can import the real module.
const script = `(async () => {
  const { createSfx } = await import('/src/audio/sfx.js');

  function analyse(buf) {
    const n = buf.length;
    const L = buf.getChannelData(0);
    const R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : L;
    let peak = 0, sum = 0, sq = 0, clipped = 0;
    for (let i = 0; i < n; i++) {
      for (const ch of [L, R]) {
        const v = ch[i];
        const a = Math.abs(v);
        if (a > peak) peak = a;
        if (a > 1) clipped++;
        sum += v;
        sq += v * v;
      }
    }
    // How hard the waveform leaves silence, relative to its own peak.
    let onset = 0;
    const floor = peak * 0.002;
    for (let i = 1; i < n; i++) {
      if (Math.abs(L[i]) > floor) {
        onset = peak > 0 ? Math.abs(L[i] - L[i - 1]) / peak : 0;
        break;
      }
    }
    return {
      peak: +peak.toFixed(3),
      rms: +Math.sqrt(sq / (n * 2)).toFixed(3),
      dc: +(sum / (n * 2)).toFixed(5),
      clipped,
      onset: +onset.toFixed(3),
    };
  }

  async function render(seconds, fn) {
    const ctx = new OfflineAudioContext(2, Math.ceil(44100 * seconds), 44100);
    const sfx = createSfx(ctx, { volume: 0.7 });
    fn(sfx, ctx);
    return analyse(await ctx.startRendering());
  }

  const rows = [];
  const add = async (name, seconds, fn) => rows.push([name, await render(seconds, fn)]);

  await add('cannon', 1.2, (s) => s.cannon({ when: 0.01 }));
  await add('cannon panned', 1.2, (s) => s.cannon({ when: 0.01, pan: -0.7 }));
  await add('cannon in volley', 1.6, (s) => s.cannon({ when: 0.01, size: 0.4, weight: 1 }));
  await add('impact round', 0.8, (s) => s.impact({ when: 0.01, size: 1 }));
  await add('impact grape', 0.8, (s) => s.impact({ when: 0.01, size: 1, kind: 'grape' }));
  await add('splinter', 0.8, (s) => s.splinter({ when: 0.01, size: 0.8 }));
  await add('splash', 0.8, (s) => s.splash({ when: 0.01, size: 0.55 }));
  await add('timber break', 2.2, (s) => s.timberBreak({ when: 0.01, size: 0.6 }));
  await add('detonation', 3, (s) => s.detonation({ when: 0.01, size: 1.1 }));
  await add('ui select', 0.3, (s) => s.ui('select', { when: 0.01 }));
  await add('ui place', 0.3, (s) => s.ui('place', { when: 0.01 }));
  await add('ui press', 0.3, (s) => s.ui('press', { when: 0.01 }));
  await add('ui confirm', 0.8, (s) => s.ui('confirm', { when: 0.01 }));
  await add('ui deny', 0.5, (s) => s.ui('deny', { when: 0.01 }));
  await add('ambience 4s', 4, (s) => s.ambience(true));

  // A rolling broadside from a ship of the line, spaced and ducked the way play.js would: sixteen
  // guns at the mixer's 28ms cadence, each one quieter and heavier than the last as the wall of
  // sound builds. This is the case the level policy exists for.
  await add('broadside x16', 3, (s) => {
    for (let i = 0; i < 16; i++) {
      const weight = i / (i + 3);
      s.cannon({ when: 0.01 + i * 0.028, size: (i % 3 ? 0.75 : 1) / Math.sqrt(1 + 0.5 * i), weight });
    }
  });

  // The worst moment the game can produce, at the density tools/mix.js actually measured: a burst
  // of ten guns and twenty-two hits inside 250ms, with a magazine going up underneath. The old
  // mixer could not reach this -- it discarded most of it -- so the level policy is what keeps the
  // peak down now, and this is the check that it does.
  await add('worst case', 3.5, (s) => {
    for (let i = 0; i < 10; i++) {
      s.cannon({ when: 0.01 + i * 0.028, size: 1 / Math.sqrt(1 + 0.5 * i), weight: i / (i + 3) });
    }
    for (let i = 0; i < 22; i++) {
      s.impact({ when: 0.2 + i * 0.022, size: 1 / Math.sqrt(1 + 0.5 * i), kind: i % 2 ? 'grape' : 'round' });
    }
    for (let i = 0; i < 5; i++) s.splash({ when: 0.3 + i * 0.05, size: 0.55 / Math.sqrt(1 + 0.5 * i) });
    s.detonation({ when: 0.4, size: 1.1 });
    s.timberBreak({ when: 0.5, size: 0.6 });
    s.splinter({ when: 0.62, size: 0.8 });
  });

  return JSON.stringify(rows);
})()`;

const rows = JSON.parse(await page.evalIn(script));
console.log('  sound             peak    rms       dc  clipped  onset  verdict');
let bad = 0;
for (const [name, m] of rows) {
  const notes = [];
  if (m.clipped > 0) notes.push(`CLIPS ${m.clipped}`);
  if (Math.abs(m.dc) > 0.01) notes.push('DC OFFSET');
  if (m.onset > 0.25) notes.push('ONSET CLICK');
  if (notes.length) bad++;
  console.log(
    `  ${name.padEnd(16)} ${String(m.peak).padStart(5)}  ${String(m.rms).padStart(5)}  ` +
      `${String(m.dc).padStart(8)}  ${String(m.clipped).padStart(7)}  ${String(m.onset).padStart(5)}  ` +
      (notes.length ? notes.join(', ') : 'ok'),
  );
}
console.log(bad === 0 ? '\n  all clean' : `\n  ${bad} sound(s) need attention`);

await page.close();
