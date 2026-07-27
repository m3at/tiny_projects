// Rolling frame statistics. Cheap enough to leave on always (two clock reads per frame),
// and it keeps the update and render costs separate, which is the only way to tell a
// simulation problem from a draw-call problem.
//
// Read it from the console or the CDP harness via __game.perf.snapshot().

const ALPHA = 0.06; // EMA weight; ~30 frames of memory

export function createPerf() {
  let update = 0;
  let render = 0;
  let frame = 0;
  let worstUpdate = 0;
  let worstRender = 0;
  let frames = 0;

  return {
    sample(updateMs, renderMs, dtSeconds) {
      frames++;
      // Skip the first few frames: module init and the first compile skew everything.
      if (frames < 10) return;
      update += (updateMs - update) * ALPHA;
      render += (renderMs - render) * ALPHA;
      frame += (dtSeconds * 1000 - frame) * ALPHA;
      worstUpdate = Math.max(worstUpdate, updateMs);
      worstRender = Math.max(worstRender, renderMs);
    },
    reset() {
      worstUpdate = 0;
      worstRender = 0;
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
      };
    },
  };
}
