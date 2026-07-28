// What each layer of the scene costs to draw, and how that scales with resolution.
//
//   node tools/fill.js                 price every layer at three resolutions
//   node tools/fill.js 1920 1080       one resolution
//
// The other render tools measure JavaScript. On anything phone-shaped the bottleneck is not
// JavaScript and not draw calls -- forty draw calls is nothing -- it is fill rate: how many pixels
// get shaded, how many times each, and how expensive the shader is per pixel. Nothing measured
// that, so nothing could say whether the sea or the wind streaks or the smoke was the problem.
//
// Method: render the same frame N times, then read one pixel back, and divide. Two traps had to be
// paid for here. gl.finish() does not do the job -- in Chrome the rasteriser lives in another
// process, and finish() returns once the commands are accepted, so every layer priced at 0.10ms
// whatever it was. A one-pixel readPixels does force the pipeline to drain. And performance.now()
// is clamped to 100us, so a single frame cannot be timed at all; batching past 20ms of work is
// what makes the number mean something. Then hide one named layer at a time and repeat: the
// difference is what that layer costs.
//
// Headless Chrome rasterises in software, which sounds like a problem and is actually the point: a
// software rasteriser is overwhelmingly fill-rate bound, which is the same thing that limits a
// phone GPU. The absolute milliseconds are far too pessimistic, so read the *shares* and the
// per-megapixel slope, not the totals. A layer that is 40% of the cost here will be roughly 40% of
// the cost on weak hardware too.

import { attach } from './cdp.js';

const argW = Number(process.argv[2] || 0);
const argH = Number(process.argv[3] || 0);
const SIZES = argW && argH ? [[argW, argH]] : [[854, 480], [1280, 720], [1920, 1080]];

const page = await attach();
const { evalIn } = page;
await page.open('?dev=draft&seed=5150&loop=1');
await page.reachPhase('battle');

// Wind the battle to a busy moment and freeze it, so every measurement draws the same frame. A
// live battle would change the particle count underneath the comparison and make the layers look
// like they cost whatever happened to be on screen.
const setup = await evalIn(`(() => {
  const g = globalThis.__game;
  if (!g.battle) return 'no battle';
  for (let i = 0; i < 240; i++) g.battle.advance(1 / 60);
  return 'projectiles ' + g.battle.projectiles.length;
})()`);
console.log(`  frozen at: ${setup}`);

// 'everything' hides the lot, which gives the floor: the clear, the swap and the browser's own
// compositing. Scene work can never get below it, so it says how much headroom is actually left.
const LAYERS = ['sea', 'shots', 'puffs', 'rings', 'ships', 'everything'];

// The whole sweep happens inside one page call, and this matters. An earlier version resized the
// renderer once per layer over separate CDP round trips, and the times climbed monotonically
// through the run until hiding a layer appeared to make the frame nearly three times *slower*.
// Repeated resizes reallocate the drawing buffer and the page never settles between them. Size is
// set once here, the game's own animation frame is parked so it is not competing for the GPU, and
// the baseline is measured again at the end -- if the two baselines disagree the whole table is
// suspect, and the tool says so rather than quietly reporting drift as a result.
const measure = `(async (w, h, layers, frames) => {
  const g = globalThis.__game;
  const r = g.sceneCtl.renderer, scene = g.sceneCtl.scene, camera = g.sceneCtl.camera;
  const gl = r.getContext();
  const px = new Uint8Array(4);
  const drain = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);

  // Stop the game drawing into the same canvas while we time it. Its animation frame keeps running
  // -- there is no hook to stop it and the simulation is cheap -- but its render becomes a no-op,
  // so the only thing reaching the GPU is the loop below. Freezing dev.speed stops the battle
  // advancing underneath, which would otherwise change the particle count mid-table.
  const { dev } = await import('/src/dev.js');
  const realSpeed = dev.speed;
  dev.speed = 0;
  const realRender = g.sceneCtl.render;
  g.sceneCtl.render = () => {};

  g.sceneCtl.setAdaptive(false); // or the controller rescales the thing being timed
  const oldRatio = r.getPixelRatio();
  const oldW = r.domElement.width / oldRatio, oldH = r.domElement.height / oldRatio;
  r.setPixelRatio(1);
  r.setSize(w, h, false);

  const shipGroups = g.views.filter(Boolean).map((v) => v.group);
  const meshes = [];
  scene.traverse((o) => { if (o.isMesh) meshes.push(o); });
  const pick = (name) => {
    if (name === 'everything') return meshes.concat(shipGroups);
    if (name === 'ships') return shipGroups;
    const o = scene.getObjectByName(name);
    return o ? [o] : [];
  };

  const time = () => {
    for (let i = 0; i < 8; i++) r.render(scene, camera);
    drain();
    const runs = [];
    for (let pass = 0; pass < 3; pass++) {
      const t0 = performance.now();
      for (let i = 0; i < frames; i++) r.render(scene, camera);
      drain();
      runs.push((performance.now() - t0) / frames);
    }
    runs.sort((a, b) => a - b);
    return runs[1];
  };

  // Paired, not sequential. Sustained software rasterising heats the machine and the baseline
  // drifted 88% from the top of the table to the bottom, which made later rows read as negative
  // costs -- hiding a layer appearing to make the frame slower. Timing the full scene immediately
  // before each layer and differencing the pair cancels any drift slower than one measurement.
  const first = time();
  const calls = r.info.render.calls;
  const out = {};
  const bases = [];
  for (const name of layers) {
    const objs = pick(name);
    const was = objs.map((o) => o.visible);
    const withAll = time();
    objs.forEach((o) => { o.visible = false; });
    const without = time();
    objs.forEach((o, i) => { o.visible = was[i]; });
    bases.push(withAll);
    out[name] = { without, cost: withAll - without, base: withAll };
  }
  const again = time();

  r.setPixelRatio(oldRatio);
  r.setSize(oldW, oldH, false);
  g.sceneCtl.setAdaptive(true);
  g.sceneCtl.render = realRender;
  dev.speed = realSpeed;
  return { first, again, calls, out };
})`;

const run = async (w, h, frames) =>
  JSON.parse(
    await evalIn(
      `(async () => JSON.stringify(await (${measure})(${w}, ${h}, ${JSON.stringify(LAYERS)}, ${frames})))()`,
    ),
  );

for (const [w, h] of SIZES) {
  const mp = (w * h) / 1e6;
  const { first, again, calls, out } = await run(w, h, w * h > 1.5e6 ? 30 : 60);
  const base = (first + again) / 2;
  const drift = Math.abs(first - again) / base;
  console.log(
    `\n  ${w}x${h}  (${mp.toFixed(2)} Mpx)   ${base.toFixed(2)}ms per frame, ` +
      `${(base / mp).toFixed(2)}ms per megapixel, ${calls} draw calls`,
  );
  if (drift > 0.15) {
    console.log(
      `    note: the machine drifted -- baseline ${first.toFixed(2)}ms at the start, ` +
        `${again.toFixed(2)}ms at the end. Each row below is still a paired comparison against a ` +
        `baseline taken beside it, so the costs hold; the totals do not.`,
    );
  }
  console.log('    layer      hidden    cost   share  (of its own paired baseline)');
  const rows = Object.entries(out).sort((a, b) => b[1].cost - a[1].cost);
  for (const [layer, m] of rows) {
    const share = (100 * m.cost) / m.base;
    console.log(
      `    ${layer.padEnd(9)} ${m.without.toFixed(2).padStart(7)}ms ${m.cost.toFixed(2).padStart(7)}ms ` +
        `${`${share.toFixed(0)}%`.padStart(6)}`,
    );
  }
}

page.printLogs();
await page.close();
