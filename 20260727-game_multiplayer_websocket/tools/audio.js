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

const PORT = 9222;
const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((t) => t.type === 'page');
if (!page) {
  console.error('no page target; run ./tools/dev.sh first');
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 1;
const pending = new Map();
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) pending.get(msg.id)(msg);
});
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });

await send('Runtime.enable');
await send('Page.navigate', { url: 'http://127.0.0.1:8123/index.html' });
await new Promise((r) => setTimeout(r, 1200));

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
  await add('impact', 0.8, (s) => s.impact({ when: 0.01, size: 0.6 }));
  await add('splash', 0.8, (s) => s.splash({ when: 0.01, size: 0.45 }));
  await add('timber break', 2.2, (s) => s.timberBreak({ when: 0.01, size: 0.6 }));
  await add('detonation', 3, (s) => s.detonation({ when: 0.01, size: 1.1 }));
  await add('tick place', 0.3, (s) => s.tick({ when: 0.01, kind: 'place' }));
  await add('tick deny', 0.3, (s) => s.tick({ when: 0.01, kind: 'deny' }));
  await add('ambience 4s', 4, (s) => s.ambience(true));

  // A rolling broadside from a ship of the line: sixteen guns inside two seconds.
  await add('broadside x16', 3, (s) => {
    for (let i = 0; i < 16; i++) s.cannon({ when: 0.01 + i * 0.11, size: i % 3 ? 0.75 : 1 });
  });

  // The worst moment the game can produce: both batteries, hits, splashes and a magazine at once.
  await add('worst case', 3.5, (s) => {
    for (let i = 0; i < 14; i++) s.cannon({ when: 0.01 + i * 0.05, size: 1 });
    for (let i = 0; i < 10; i++) s.impact({ when: 0.2 + i * 0.05, size: 0.6 });
    for (let i = 0; i < 8; i++) s.splash({ when: 0.3 + i * 0.09, size: 0.45 });
    s.detonation({ when: 0.4, size: 1.1 });
    s.timberBreak({ when: 0.5, size: 0.6 });
  });

  return JSON.stringify(rows);
})()`;

const res = await send('Runtime.evaluate', {
  expression: script,
  awaitPromise: true,
  returnByValue: true,
});
if (res.result?.exceptionDetails) {
  console.error(res.result.exceptionDetails.exception?.description || res.result.exceptionDetails.text);
  process.exit(1);
}

const rows = JSON.parse(res.result.result.value);
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

await send('Page.navigate', { url: 'about:blank' });
ws.close();
