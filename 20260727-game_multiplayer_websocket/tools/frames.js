// Frame times, as a distribution rather than an average.
//
//   node tools/frames.js            build phase and battle phase, one match
//   node tools/frames.js 12         12 seconds in each phase
//
// Stutter is a tail problem. An average frame time says nothing about it: 4ms mean with one frame
// in two hundred at 40ms reads as smooth on paper and hitches to the eye. This reports p50, p90,
// p99 and max per phase, counts frames costing more than twice the median, and prints the worst
// few with their update/render split so the cost can be attributed. When supported, asynchronous
// GPU timings name actual raster work separately instead of folding it into browser wall time.
//
// Headless rendering is software-rasterised and roughly 3x slower than a real GPU, so treat the
// absolute numbers as a ceiling and read the *shape*: a p99 near the median means smooth, and a
// p99 several times the median means something occasional and expensive, which is what to hunt.
// The update/render split survives the difference, because both are JavaScript.

import { attach, sleep } from './cdp.js';

const SECONDS = Number(process.argv[2] || 10);
const page = await attach();
await page.open('?dev=draft&seed=31337&loop=1');

function report(label, d, snap) {
  if (!d) return console.log(`  ${label}: no frames`);
  console.log(`\n  ${label}   ${d.frames} frames`);
  console.log(
    `    wall  p50 ${String(d.wall.p50).padStart(6)}ms  p90 ${String(d.wall.p90).padStart(6)}ms  ` +
      `p99 ${String(d.wall.p99).padStart(6)}ms  max ${String(d.wall.max).padStart(6)}ms`,
  );
  console.log(
    `    js    p50 ${String(d.js.p50).padStart(6)}ms  p90 ${String(d.js.p90).padStart(6)}ms  ` +
      `p99 ${String(d.js.p99).padStart(6)}ms  max ${String(d.js.max).padStart(6)}ms`,
  );
  if (d.gpu) {
    console.log(
      `    gpu   p50 ${String(d.gpu.p50).padStart(6)}ms  p90 ${String(d.gpu.p90).padStart(6)}ms  ` +
        `p99 ${String(d.gpu.p99).padStart(6)}ms  max ${String(d.gpu.max).padStart(6)}ms  ` +
        `(${d.gpuSamples} samples)`,
    );
  } else {
    console.log('    gpu   unavailable; adaptive quality is using wall time');
  }
  console.log(
    `    js spikes over 2x median: ${d.spikes} (${d.spikeShare}%)   ` +
      `p99/p50 ${(d.js.p99 / Math.max(d.js.p50, 0.01)).toFixed(1)}x`,
  );
  console.log(
    `    worst: ${d.worstFrames.map((f) => `${f.ms}ms (u${f.update} r${f.render})`).join('  ')}`,
  );
  const s = JSON.parse(snap);
  console.log(
    `    draw calls ${s.drawCalls}, triangles ${s.triangles}, programs ${s.programs}, ` +
      `geometries ${s.geometries}, textures ${s.textures}, ` +
      `scale ${s.quality.scale} (${s.quality.source})`,
  );
}

for (const phase of ['build', 'battle']) {
  if (!(await page.reachPhase(phase))) {
    console.log(`  never reached the ${phase} phase`);
    continue;
  }
  await page.evalIn('__game.perf.reset()');
  // During the build phase, sweep the pointer over the hull: hovering rewrites the ghost mesh and
  // the arc preview, and that work only happens when someone is actually moving a mouse.
  if (phase === 'build') {
    await page.evalIn(`__dev.pickCard('hull timber')`);
    const sweep = `(() => {
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
      return 'sweeping';
    })()`;
    await page.evalIn(sweep);
  }
  await sleep(SECONDS * 1000);
  const d = await page.evalIn('JSON.stringify(__game.perf.distribution())');
  const snap = await page.evalIn(`JSON.stringify({
    ...__game.perf.snapshot(__game.sceneCtl.renderer),
    quality: __game.sceneCtl.qualityState,
  })`);
  if (phase === 'build') await page.evalIn('clearInterval(globalThis.__sweep)');
  report(phase, JSON.parse(d), snap);
}

page.printLogs();
await page.close();
