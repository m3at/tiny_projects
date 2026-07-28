// Deciding how many pixels to draw.
//
// Frame cost is very nearly linear in pixel count -- tools/fill.js measures the slope -- so
// rendering scale is the one lever that always works, and it is the only way to promise 60fps on
// hardware nobody has tested on. This is the policy; scene.js owns the renderer and applies it.
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

const LATE_MS = 20; // a frame slower than 50fps
const WINDOW_MS = 1000;
const DROP_SHARE = 0.25; // late frames in a window that force a step down
const RISE_SHARE = 0.05; // late frames under which a window counts as comfortable
const RISE_WINDOWS = 4; // consecutive comfortable windows before the first attempt to step up
const RISE_MAX = 240; // ceiling on the doubling, so it still re-probes every few minutes
const MIN_WINDOW_FRAMES = 20; // below this a window is not evidence of anything
const WARMUP_MS = 1000; // module init, shader compiles, first upload of every buffer

export function createQuality({ steps, onChange }) {
  let step = 0;
  let enabled = true;

  let windowStart = 0;
  let frames = 0;
  let late = 0;
  let goodWindows = 0;
  let windowIndex = 0;
  let roseAt = -1e9;
  let riseNeed = RISE_WINDOWS;

  function moveTo(next) {
    step = next;
    goodWindows = 0;
    onChange(steps[step]);
  }

  return {
    get step() {
      return step;
    },
    get scale() {
      return steps[step];
    },

    // Called once per frame with the real gap since the last one.
    sample(now, frameMs) {
      if (!enabled) return;
      if (windowStart === 0) {
        windowStart = now + WARMUP_MS;
        return;
      }
      if (now < windowStart) return;

      frames++;
      if (frameMs > LATE_MS) late++;
      if (now - windowStart < WINDOW_MS) return;

      if (frames >= MIN_WINDOW_FRAMES) {
        const share = late / frames;
        if (share > DROP_SHARE && step < steps.length - 1) {
          // A recent promotion caused this. Make the next one much harder to earn.
          if (windowIndex - roseAt <= riseNeed + 1) riseNeed = Math.min(riseNeed * 2, RISE_MAX);
          moveTo(step + 1);
        } else if (share < RISE_SHARE) {
          if (++goodWindows >= riseNeed && step > 0) {
            roseAt = windowIndex;
            moveTo(step - 1);
          }
        } else {
          goodWindows = 0;
        }
        windowIndex++;
      }

      windowStart = now;
      frames = 0;
      late = 0;
    },

    // Tools that time the renderer themselves have to turn this off, or the thing being measured
    // moves while it is being measured.
    setEnabled(on) {
      enabled = on;
      if (!on && step !== 0) moveTo(0);
    },
  };
}
