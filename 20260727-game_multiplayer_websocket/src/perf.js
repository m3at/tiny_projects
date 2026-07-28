// Rolling frame statistics. Cheap enough to leave on always -- two clock reads and three array
// writes per frame -- and it keeps the update and render costs separate, which is the only way to
// tell a simulation problem from a draw-call problem.
//
// Averages are the wrong instrument for stutter. A frame budget of 16.7ms with a mean of 4ms still
// hitches visibly if one frame in two hundred costs 40ms, and an exponential moving average hides
// exactly that. So the last few seconds of frames are kept in a ring buffer and read back as
// percentiles, with the worst offenders retained whole. p99 and max are the numbers that matter;
// the mean is there for scale.
//
// Read it from the console or the CDP harness via __game.perf.snapshot() and
// __game.perf.distribution().

const ALPHA = 0.06; // EMA weight; ~30 frames of memory
const CAP = 900; // ~15s at 60Hz, one Float32Array each, allocated once

export function createPerf() {
  let update = 0;
  let render = 0;
  let frame = 0;
  let worstUpdate = 0;
  let worstRender = 0;
  let frames = 0;

  const totalMs = new Float32Array(CAP);
  const updateMs = new Float32Array(CAP);
  const renderMs = new Float32Array(CAP);
  let head = 0;
  let filled = 0;
  const sorted = new Float32Array(CAP); // scratch for percentiles, so reading allocates nothing

  return {
    sample(u, r, dtSeconds) {
      frames++;
      // Skip the first few frames: module init and the first compile skew everything.
      if (frames < 10) return;
      update += (u - update) * ALPHA;
      render += (r - render) * ALPHA;
      frame += (dtSeconds * 1000 - frame) * ALPHA;
      if (u > worstUpdate) worstUpdate = u;
      if (r > worstRender) worstRender = r;

      totalMs[head] = dtSeconds * 1000;
      updateMs[head] = u;
      renderMs[head] = r;
      head = (head + 1) % CAP;
      if (filled < CAP) filled++;
    },

    reset() {
      worstUpdate = 0;
      worstRender = 0;
      head = 0;
      filled = 0;
    },

    // Percentiles over the ring, plus the worst few frames with their update/render split. A spike
    // is a frame costing more than twice the median, which is the thing an eye reads as a hitch.
    //
    // Two distributions, because they answer different questions. `wall` is the gap between
    // frames, which is what the player experiences but also includes whatever the GPU and the rest
    // of the machine were doing. `js` is our own update plus render cost, which is the part this
    // code can do anything about, and the only part that means the same thing on a software
    // rasteriser as on a real GPU.
    distribution() {
      if (filled === 0) return null;

      const quantiles = (src, scratchFrom) => {
        for (let i = 0; i < filled; i++) sorted[i] = scratchFrom(i);
        const view = sorted.subarray(0, filled);
        view.sort();
        const at = (q) => +view[Math.min(filled - 1, Math.floor(q * filled))].toFixed(2);
        return {
          p50: at(0.5),
          p90: at(0.9),
          p99: at(0.99),
          max: +view[filled - 1].toFixed(2),
          median: view[Math.floor(filled / 2)],
        };
      };

      const wall = quantiles(totalMs, (i) => totalMs[i]);
      const js = quantiles(null, (i) => updateMs[i] + renderMs[i]);

      let spikes = 0;
      for (let i = 0; i < filled; i++) if (updateMs[i] + renderMs[i] > js.median * 2) spikes++;

      // The five worst frames by JavaScript cost, without disturbing the ring.
      const worst = [];
      for (let i = 0; i < filled; i++) worst.push(i);
      worst.sort((a, b) => updateMs[b] + renderMs[b] - (updateMs[a] + renderMs[a]));
      return {
        frames: filled,
        wall,
        js,
        spikes,
        spikeShare: +((100 * spikes) / filled).toFixed(2),
        worstFrames: worst.slice(0, 5).map((i) => ({
          ms: +(updateMs[i] + renderMs[i]).toFixed(2),
          update: +updateMs[i].toFixed(2),
          render: +renderMs[i].toFixed(2),
        })),
      };
    },

    snapshot(renderer) {
      const info = renderer?.info;
      return {
        frames,
        fps: +(1000 / Math.max(frame, 0.001)).toFixed(1),
        updateMs: +update.toFixed(3),
        renderMs: +render.toFixed(3),
        worstUpdateMs: +worstUpdate.toFixed(2),
        worstRenderMs: +worstRender.toFixed(2),
        drawCalls: info?.render.calls ?? null,
        triangles: info?.render.triangles ?? null,
        geometries: info?.memory.geometries ?? null,
        textures: info?.memory.textures ?? null,
        programs: info?.programs?.length ?? null,
        pixelRatio: renderer ? +renderer.getPixelRatio().toFixed(2) : null,
      };
    },
  };
}
