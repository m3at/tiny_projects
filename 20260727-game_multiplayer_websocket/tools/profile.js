// Where the frame time goes, from the browser's own sampling profiler.
//
//   node tools/profile.js              profile a battle, print the hot functions
//   node tools/profile.js 20 build     profile 20 seconds of the build phase instead
//
// The simulation is profiled with `node --cpu-prof`, which says nothing about the half of the
// program that only exists in a browser: three.js, the instanced rebuilds, the DOM writes, the
// audio graph. This drives the real page over CDP and reads Profiler.takePreciseCoverage's
// cousin, Profiler.stop, for self time per function.
//
// Two caveats, both important when reading the numbers:
//
//   Headless rendering is software-rasterised and roughly 3x slower than a real GPU, so the
//   absolute cost of the draw is meaningless here. What survives the difference is the JavaScript:
//   allocation, rebuild work, DOM writes. Those are what stutter is usually made of, and they are
//   measured honestly.
//
//   A sampling profiler reports where time was spent, not where it was *lost*. A garbage collection
//   pause shows up as (garbage collector) self time, which is a symptom; the cause is whoever
//   allocated. tools/frames.js measures the pauses themselves.

import { attach, sleep } from './cdp.js';

const SECONDS = Number(process.argv[2] || 15);
const PHASE = process.argv[3] || 'battle';

const page = await attach();
const { send, evalIn } = page;
await page.open('?dev=draft&seed=7777&loop=1');

await page.reachPhase(PHASE);
console.log(`  profiling ${SECONDS}s of the ${await evalIn('__dev.state().phase')} phase`);

// Hovering is only work when a pointer is actually moving, so a build-phase profile that does not
// move one measures an idle panel.
if (PHASE === 'build') {
  await evalIn(`__dev.pickCard('hull timber')`);
  await evalIn(`(() => {
    const cells = [];
    for (let dz = -5; dz <= 5; dz++) for (const dx of [-1, 0, 1]) cells.push([dx, dz]);
    let i = 0;
    globalThis.__sweep = setInterval(() => {
      const [dx, dz] = cells[i++ % cells.length];
      const c = __game.sceneCtl.renderer.domElement;
      const p = __dev.where(dx, dz);
      c.dispatchEvent(new PointerEvent('pointermove', {
        clientX: p.x, clientY: p.y, bubbles: true, pointerId: 1,
      }));
    }, 40);
  })()`);
}

await evalIn('__game.perf.reset()');
await send('Profiler.enable');
await send('Profiler.setSamplingInterval', { interval: 100 }); // microseconds
await send('Profiler.start');
await sleep(SECONDS * 1000);
const { result } = await send('Profiler.stop');
const perf = await evalIn('JSON.stringify(__game.perf.snapshot(__game.sceneCtl.renderer))');

// Self time per node, from the sample counts and the delta timestamps.
const { nodes, samples, timeDeltas } = result.profile;
const byId = new Map(nodes.map((n) => [n.id, n]));
const self = new Map();
for (let i = 0; i < samples.length; i++) {
  const dt = timeDeltas[i] || 0;
  self.set(samples[i], (self.get(samples[i]) || 0) + dt);
}

const rows = [];
let totalUs = 0;
let rasterUs = 0;
for (const [id, us] of self) {
  const node = byId.get(id);
  if (!node) continue;
  const f = node.callFrame;
  const name = f.functionName || '(anonymous)';
  // Headless draws on the CPU, and the rasteriser lands in (program) with no stack. It swamps
  // everything -- 98% of a battle profile -- and tells us nothing, since a real GPU does that work.
  if (name === '(program)' || name === '(idle)') {
    rasterUs += us;
    continue;
  }
  totalUs += us;
  const file = (f.url || '').split('/').slice(-1)[0] || '(native)';
  rows.push([`${name} — ${file}${f.lineNumber >= 0 ? `:${f.lineNumber + 1}` : ''}`, us]);
}
rows.sort((a, b) => b[1] - a[1]);

console.log(
  `\n  ${(totalUs / 1000).toFixed(0)}ms of JavaScript over ${SECONDS}s ` +
    `(${(rasterUs / 1000).toFixed(0)}ms of software rasterising excluded)\n`,
);
console.log('  self time  share  function');
for (const [label, us] of rows.slice(0, 28)) {
  const share = (100 * us) / totalUs;
  if (share < 0.4) break;
  console.log(
    `  ${`${(us / 1000).toFixed(1)}ms`.padStart(9)}  ${`${share.toFixed(1)}%`.padStart(5)}  ${label}`,
  );
}
console.log(`\n  perf.snapshot: ${perf}`);
if (PHASE === 'build') await evalIn('clearInterval(globalThis.__sweep)');

await send('Profiler.disable');
page.printLogs();
await page.close();
