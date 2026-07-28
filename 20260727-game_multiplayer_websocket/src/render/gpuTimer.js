// Asynchronous GPU frame timing.
//
// performance.now() around renderer.render() measures JavaScript command submission, not when the
// GPU finishes those commands. readPixels() gets the real answer by stalling the whole pipeline,
// which is useful in tools/fill.js and unacceptable in the game. Timer queries report the same
// elapsed GPU time a few frames later without making either processor wait for the other.
//
// Not every browser exposes the extension, so absence is an ordinary state. We sample one frame in
// four: enough evidence for the one-second quality windows, with negligible driver/query traffic.

const SAMPLE_INTERVAL = 4;
const MAX_PENDING = 8;
const NS_TO_MS = 1e-6;

export function createGpuTimer(context) {
  let gl = context;
  let ext = null;
  let active = null;
  let frame = 0;
  const pending = [];
  const results = [];

  function findExtension() {
    ext = gl?.getExtension('EXT_disjoint_timer_query_webgl2') ?? null;
  }

  function discardPending(canDelete = true) {
    if (canDelete && gl && !gl.isContextLost()) {
      if (active) gl.deleteQuery(active);
      for (const query of pending) gl.deleteQuery(query);
    }
    active = null;
    pending.length = 0;
    results.length = 0;
  }

  findExtension();

  return {
    get supported() {
      return !!ext;
    },

    begin() {
      frame++;
      if (!ext || active || pending.length >= MAX_PENDING || frame % SAMPLE_INTERVAL !== 0) return;
      const query = gl.createQuery();
      if (!query) return;
      gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
      active = query;
    },

    end() {
      if (!active || !ext) return;
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      pending.push(active);
      active = null;
    },

    // The returned array is reused. Consume it before the next poll rather than retaining it.
    poll() {
      results.length = 0;
      if (!ext || pending.length === 0) return results;

      // Availability is explicitly non-blocking. Do not ask for the result until it says yes.
      if (!gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) return results;

      // A frequency change, context event or similar discontinuity invalidates every outstanding
      // duration, not just the first. Drop the batch instead of teaching quality from bad evidence.
      if (gl.getParameter(ext.GPU_DISJOINT_EXT)) {
        discardPending();
        return results;
      }

      while (
        pending.length &&
        gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)
      ) {
        const query = pending.shift();
        const ms = gl.getQueryParameter(query, gl.QUERY_RESULT) * NS_TO_MS;
        gl.deleteQuery(query);
        if (Number.isFinite(ms) && ms >= 0) results.push(ms);
      }
      return results;
    },

    contextLost() {
      discardPending(false);
      ext = null;
    },

    contextRestored(context) {
      gl = context;
      discardPending(false);
      frame = 0;
      findExtension();
    },
  };
}
