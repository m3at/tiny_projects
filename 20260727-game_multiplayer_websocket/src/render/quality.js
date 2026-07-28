// Deciding how many pixels to draw.
//
// Frame cost is very nearly linear in pixel count -- tools/fill.js measures the slope -- so
// rendering scale is the one lever that always works, and it is the only way to promise 60fps on
// hardware nobody has tested on. This is the policy; scene.js owns the renderer and applies it.
//
// The important distinction is between a late frame and a GPU-bound frame. A browser can miss a
// requestAnimationFrame because of layout, another tab, the OS scheduler or JavaScript; lowering
// resolution fixes none of those. When asynchronous GPU timers are available, they are the signal
// this controller follows. Wall time remains the fallback on browsers which expose no timer.
//
// Kept apart from scene.js because it is a control loop rather than a scene, and control loops
// reacting to their own effect are where the bugs live. This one has three, all found by watching
// it run rather than by reading it:
//
//   Stepping down and stepping up are not symmetric. Down is cheap to get wrong, so it happens as
//   soon as a quarter of a second's frames were late. Up is expensive to get wrong, so it needs
//   several consecutive comfortable windows.
//
//   Hysteresis alone is not enough. With a fixed four-window delay it still pumped between 0.5 and
//   0.6 for as long as it was watched: a machine that lands between two steps is comfortable at the
//   lower one *because* it is lower, so "comfortable for a while" is not evidence that the higher
//   one would hold. Every promotion that gets undone soon after doubles what the next one costs, so
//   after a couple of attempts it settles and stops trying.
//
//   A window needs a minimum number of frames. Coming back from a backgrounded tab hands it a
//   single enormous frame, which is 100% late and says nothing at all about the hardware.

const WALL_LATE_MS = 20; // a frame slower than 50fps
const GPU_DROP_MS = 12; // leaves ~4.7ms at 60Hz for submission and the compositor
const GPU_RISE_MS = 8; // promotion needs substantially more room than merely avoiding a drop
const WINDOW_MS = 1000;
const DROP_SHARE = 0.25; // late frames in a window that force a step down
const RISE_SHARE = 0.05; // late frames under which a window counts as comfortable
const RISE_WINDOWS = 4; // consecutive comfortable windows before the first attempt to step up
const RISE_MAX = 240; // ceiling on the doubling, so it still re-probes every few minutes
const MIN_WINDOW_FRAMES = 20; // below this a window is not evidence of anything
const MIN_GPU_SAMPLES = 6; // gpuTimer samples one frame in four, or about 15 times a second
const GPU_EMPTY_WINDOWS = 2; // then assume a nominally present timer is not returning results
const WARMUP_MS = 1000; // module init, shader compiles, first upload of every buffer

export function createQuality({ steps, onChange }) {
  let step = 0;
  let enabled = true;

  let windowStart = 0;
  let frames = 0;
  let wallLate = 0;
  let gpuFrames = 0;
  let gpuLate = 0;
  let gpuBusy = 0;
  let emptyGpuWindows = 0;
  let goodWindows = 0;
  let windowIndex = 0;
  let roseAt = -1e9;
  let riseNeed = RISE_WINDOWS;
  let source = 'warming';
  let lastPressure = 0;

  function moveTo(next) {
    step = next;
    goodWindows = 0;
    onChange(steps[step]);
  }

  function resetWindow(now = 0) {
    windowStart = now;
    frames = 0;
    wallLate = 0;
    gpuFrames = 0;
    gpuLate = 0;
    gpuBusy = 0;
  }

  return {
    get step() {
      return step;
    },
    get scale() {
      return steps[step];
    },
    get state() {
      return {
        enabled,
        step,
        scale: steps[step],
        source,
        pressure: +lastPressure.toFixed(3),
      };
    },

    // Called once per frame with the real gap and any asynchronous GPU results which became
    // available on the previous draw. GPU samples describe older frames, but a one-second control
    // window cares about sustained load rather than matching one result to one requestAnimationFrame.
    sample(now, frameMs, gpuSamples = [], gpuSupported = false) {
      if (!enabled) return;
      if (windowStart === 0) {
        windowStart = now + WARMUP_MS;
        source = 'warming';
        return;
      }
      if (now < windowStart) return;

      frames++;
      if (frameMs > WALL_LATE_MS) wallLate++;
      for (const ms of gpuSamples) {
        gpuFrames++;
        if (ms > GPU_DROP_MS) gpuLate++;
        if (ms > GPU_RISE_MS) gpuBusy++;
      }
      if (now - windowStart < WINDOW_MS) return;

      if (frames >= MIN_WINDOW_FRAMES) {
        let pressure;
        let comfortable;

        if (gpuSupported && gpuFrames >= MIN_GPU_SAMPLES) {
          emptyGpuWindows = 0;
          source = 'gpu';
          pressure = gpuLate / gpuFrames;
          comfortable = gpuBusy / gpuFrames < RISE_SHARE;
        } else {
          if (gpuSupported) emptyGpuWindows++;
          // Give a supported timer two windows to return enough asynchronous results. If it never
          // does, retain adaptive resolution through the wall-time fallback rather than silently
          // pinning quality forever.
          const fallback = !gpuSupported || emptyGpuWindows >= GPU_EMPTY_WINDOWS;
          source = fallback ? 'wall' : 'waiting-gpu';
          pressure = fallback ? wallLate / frames : null;
          comfortable = fallback ? pressure < RISE_SHARE : false;
        }

        if (pressure !== null) {
          lastPressure = pressure;
          if (pressure > DROP_SHARE && step < steps.length - 1) {
            // A recent promotion caused this. Make the next one much harder to earn.
            if (windowIndex - roseAt <= riseNeed + 1) {
              riseNeed = Math.min(riseNeed * 2, RISE_MAX);
            }
            moveTo(step + 1);
          } else if (comfortable) {
            if (++goodWindows >= riseNeed && step > 0) {
              roseAt = windowIndex;
              moveTo(step - 1);
            }
          } else {
            goodWindows = 0;
          }
        }
        windowIndex++;
      }

      resetWindow(now);
    },

    // Tools that time the renderer themselves have to turn this off, or the thing being measured
    // moves while it is being measured.
    setEnabled(on) {
      if (enabled === on) return;
      enabled = on;
      source = on ? 'warming' : 'disabled';
      emptyGpuWindows = 0;
      resetWindow();
      if (!on && step !== 0) moveTo(0);
    },
  };
}
