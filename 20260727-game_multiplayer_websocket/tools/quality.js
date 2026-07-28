// Deterministic checks for the adaptive-resolution control loop.
//
//   node tools/quality.js
//
// Browser frame measurements are noisy by definition. These checks drive the policy with synthetic
// windows and hold the decisions that should not depend on whichever machine runs frames.js:
// GPU pressure lowers resolution, CPU/browser lateness does not, missing timers fall back, and a
// lower step eventually earns its way back up under sustained headroom.

import { createQuality } from '../src/render/quality.js';

const STEPS = [1, 0.85, 0.72];
const FRAME_MS = 1000 / 60;

function controller() {
  const changes = [];
  const quality = createQuality({
    steps: STEPS,
    onChange: (scale) => changes.push(scale),
  });
  return { quality, changes, now: 0, frame: 0 };
}

function drive(c, seconds, { wallMs, gpuMs, gpuSupported }) {
  const end = c.now + seconds * 1000;
  while (c.now < end) {
    c.now += FRAME_MS;
    c.frame++;
    const samples = gpuMs != null && c.frame % 4 === 0 ? [gpuMs] : [];
    c.quality.sample(c.now, wallMs, samples, gpuSupported);
  }
}

const failures = [];
const check = (ok, message) => {
  if (!ok) failures.push(message);
};

{
  const c = controller();
  drive(c, 3.2, { wallMs: 25, gpuMs: 2, gpuSupported: true });
  check(c.quality.scale === 1, 'CPU/browser lateness lowered resolution despite an idle GPU');
  check(c.quality.state.source === 'gpu', 'available GPU evidence was not selected');
}

{
  const c = controller();
  drive(c, 3.2, { wallMs: 16.7, gpuMs: 14, gpuSupported: true });
  check(c.quality.scale < 1, 'sustained GPU pressure did not lower resolution');
}

{
  const c = controller();
  drive(c, 3.2, { wallMs: 25, gpuMs: null, gpuSupported: false });
  check(c.quality.scale < 1, 'wall-time fallback did not lower resolution');
  check(c.quality.state.source === 'wall', 'unsupported GPU timer did not select wall time');
}

{
  const c = controller();
  drive(c, 2.2, { wallMs: 25, gpuMs: null, gpuSupported: true });
  check(c.quality.scale === 1, 'a supported timer fell back before its grace window elapsed');
  drive(c, 1.2, { wallMs: 25, gpuMs: null, gpuSupported: true });
  check(c.quality.scale < 1, 'a non-returning timer never fell back to wall time');
}

{
  const c = controller();
  drive(c, 3.2, { wallMs: 16.7, gpuMs: 14, gpuSupported: true });
  const lowered = c.quality.scale;
  drive(c, 5.2, { wallMs: 16.7, gpuMs: 2, gpuSupported: true });
  check(lowered < 1 && c.quality.scale > lowered, 'sustained GPU headroom did not restore quality');
}

if (failures.length) {
  for (const failure of failures) console.error(`  FAIL: ${failure}`);
  process.exitCode = 1;
} else {
  console.log('  gpu pressure, CPU isolation, fallback and recovery: pass');
}
