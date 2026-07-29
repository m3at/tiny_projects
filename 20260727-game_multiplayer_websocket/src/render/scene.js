// Scene, sea and camera. Orthographic and tilted: a plan view you can still read, with
// enough perspective that masts and gun boxes have height.
//
// Two things this file is careful about, because frame() runs every frame:
//   - the canvas is only resized and the projection only rebuilt when something actually
//     changed, rather than unconditionally;
//   - the camera's orientation is fixed for the whole game (constant tilt, constant offset
//     from the target), so it is computed once and only the position moves after that.

import * as THREE from 'three';
import { SEA, LIGHT } from '../theme.js';
import { createSea } from './sea.js';
import { createGpuTimer } from './gpuTimer.js';
import { createQuality } from './quality.js';

const TILT = (60 * Math.PI) / 180; // from horizontal
const CAMERA_DISTANCE = 300; // irrelevant to scale under orthographic; just clears the scene

// Rendering scale steps for the adaptive resolution controller. Below about half the frame is
// visibly soft, so that is the floor: past it, a device simply gets fewer than 60 frames.
const SCALES = [1, 0.85, 0.72, 0.6, 0.5];
const MAX_RATIO = 2; // past this the extra pixels buy nothing you can see on any current display
// Start within a 1080p-sized fill budget. Without this, a 4K screen or a Retina laptop begins at
// 5-8 million pixels and spends one quality window at each step discovering the obvious. The sea is
// fill-bound and fill.js shows near-linear cost, so CSS area is enough to avoid those first seconds
// of missed frames without a device database.
const MAX_PIXELS = 1920 * 1080;

export function createScene(canvas) {
  // Nothing draws to the stencil buffer, so the context does not have to allocate one. On a
  // tile-based mobile GPU that is tile memory saved on every frame, for free. Depth stays: the
  // scene is tilted boxes and needs it.
  //
  // `alpha: false` is deliberately NOT set, though it looks like the same kind of saving. It is
  // not: MDN's WebGL best practices has a section headed "Avoid alpha:false, which can be
  // expensive", because an RGB back buffer often has to be emulated on top of an RGBA surface. The
  // opaque pass already writes 1.0 to alpha, which is what that page recommends doing instead.
  //
  // No MSAA. The scene is already drawn at up to two device pixels per CSS pixel, while default
  // framebuffer MSAA multiplied its dominant full-screen pass as well as the few edges that need
  // smoothing. Measured through Metal at 1080p: 0.37ms with it, 0.067ms without it. One consistent
  // path is also easier to reason about than a context option that cannot be changed after creation.
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    stencil: false,
    powerPreference: 'high-performance',
  });
  const gpuTimer = createGpuTimer(renderer.getContext());

  // Mobile loses the GL context routinely -- backgrounding, memory pressure, a driver reset -- and
  // without these the canvas simply stays black for the rest of the session. Preventing the default
  // on loss is what allows a restore to be delivered at all. three.js rebuilds its own resources on
  // restore, so there is nothing else to do here.
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    gpuTimer.contextLost();
  });
  canvas.addEventListener('webglcontextrestored', () => {
    gpuTimer.contextRestored(renderer.getContext());
    renderer.setClearColor(SEA.water);
    lastWidth = 0; // forget the cached size so resize() actually reapplies it
    resize();
  });
  // Corrected to the compositor's real ratio by the ResizeObserver below; this is only the value
  // used for the very first frame, before layout has been observed once.
  let deviceRatio = Math.min(devicePixelRatio, MAX_RATIO);
  // The sea pass covers every pixel, so this is only what shows for the frame or two before it
  // first draws, and after a context restore.
  renderer.setClearColor(SEA.water);

  // No fog: the orthographic camera sits far back for the tilt, so distance-based fog would
  // flatten the whole scene to one colour.
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 1, CAMERA_DISTANCE * 2);
  const target = new THREE.Vector3();
  let viewSize = 70;

  scene.add(new THREE.HemisphereLight(LIGHT.sky, LIGHT.ground, 1.15));
  const sun = new THREE.DirectionalLight(LIGHT.sun, 1.5);
  sun.position.set(-45, 90, 38);
  scene.add(sun);

  // ---- sea, wind and the arena boundary ----
  // One unlit full-screen pass. render/sea.js explains why it is a shader rather than a lit
  // plane, 420 drifting quads and a ring mesh -- all three of which it replaced.
  const sea = createSea();
  const seaU = sea.material.uniforms;
  scene.add(sea);

  let seaTime = 0;

  // ---- camera ----

  let lastWidth = 0;
  let lastHeight = 0;
  let lastViewSize = -1;

  function updateProjection() {
    const aspect = lastWidth / lastHeight;
    camera.left = -viewSize * aspect;
    camera.right = viewSize * aspect;
    camera.top = viewSize;
    camera.bottom = -viewSize;
    camera.updateProjectionMatrix();
    lastViewSize = viewSize;
  }

  let appliedRatio = -1;

  function renderRatio(w, h, scale = quality.scale) {
    return Math.min(deviceRatio, Math.sqrt(MAX_PIXELS / (w * h))) * scale;
  }

  function applySize(w, h) {
    if (!w || !h) return;
    const ratio = renderRatio(w, h);
    if (w === lastWidth && h === lastHeight && ratio === appliedRatio) return;
    lastWidth = w;
    lastHeight = h;
    appliedRatio = ratio;
    // setPixelRatio() calls setSize() internally, so following it with setSize() reallocates and
    // clears the drawing buffer twice. setDrawingBufferSize() applies all three values in one pass.
    renderer.setDrawingBufferSize(w, h, ratio);
    updateProjection();
  }

  // Reading clientWidth forces the browser to flush style and layout before it can answer. frame()
  // used to call this every frame, which made the cost of a frame depend on whether the HUD had
  // just written to the DOM -- the profiler put it at the top of the JavaScript in both phases, and
  // it is the kind of stutter you will never find by looking at the renderer. A ResizeObserver
  // reports the same numbers off the layout the browser has already done, and only when they
  // change, so the frame path reads nothing.
  function resize() {
    const box = canvas.getBoundingClientRect();
    applySize(Math.round(box.width), Math.round(box.height));
  }

  // Observed in *device* pixels, not CSS pixels. devicePixelContentBoxSize is the exact integer
  // backing-store size the compositor wants, and dividing it by the CSS size gives the true ratio,
  // which is not always devicePixelRatio: at browser zoom or on a 125%-scaled display the two
  // disagree, the drawing buffer ends up a non-integer multiple of the displayed size, and the
  // result is a faint moire over the whole picture. Asking for the device box removes the guesswork.
  new ResizeObserver((entries) => {
    const e = entries[entries.length - 1];
    const css = e.contentBoxSize[0];
    const dev = e.devicePixelContentBoxSize[0];
    const w = Math.round(css.inlineSize);
    const h = Math.round(css.blockSize);
    if (w > 0) deviceRatio = Math.min(dev.inlineSize / w, MAX_RATIO);
    applySize(w, h);
  }).observe(canvas, { box: 'device-pixel-content-box' });

  // ---- adaptive resolution ----
  // The policy lives in quality.js; this is only how it is applied.
  const quality = createQuality({
    steps: SCALES,
    onChange: (scale) => {
      const ratio = renderRatio(lastWidth, lastHeight, scale);
      renderer.setDrawingBufferSize(lastWidth, lastHeight, ratio);
      appliedRatio = ratio;
    },
  });

  // The offset from target to camera never changes, so neither does the orientation.
  const offset = new THREE.Vector3(0, Math.sin(TILT), Math.cos(TILT)).multiplyScalar(
    CAMERA_DISTANCE,
  );
  resize();
  camera.position.copy(offset);
  camera.lookAt(0, 0, 0);
  const orientation = camera.quaternion.clone();

  function placeCamera() {
    camera.position.copy(target).add(offset);
    camera.quaternion.copy(orientation);
  }

  return {
    renderer,
    scene,
    camera,
    raycaster: new THREE.Raycaster(),

    setWind(w) {
      // The wind blows towards this bearing; 0 is -z, and x is starboard of it.
      seaU.uWind.value.set(Math.sin(w), -Math.cos(w));
    },

    // The arena grows with the number of ships in it, and the boundary is drawn by the sea shader
    // rather than by a mesh, so moving it is a uniform.
    setArenaRadius(radius) {
      const centre = radius - 0.45;
      seaU.uRing.value.set(centre * centre, 2 * centre * 0.45, 2 * centre);
    },

    // Ease toward a framing rather than snapping, so the battle camera glides.
    frame(cx, cz, size, snap = false) {
      if (snap) {
        target.set(cx, 0, cz);
        viewSize = size;
      } else {
        target.x += (cx - target.x) * 0.08;
        target.z += (cz - target.z) * 0.08;
        viewSize += (size - viewSize) * 0.05;
      }
      if (Math.abs(viewSize - lastViewSize) > 1e-4) updateProjection();
      placeCamera();

      // Screen to water, as a scale and an offset. Exact under an orthographic camera, which has
      // no perspective divide; the z term carries the 60-degree foreshortening.
      const aspect = lastWidth / Math.max(1, lastHeight);
      seaU.uMap.value.set(target.x, target.z, viewSize * aspect, -viewSize / Math.sin(TILT));

      // Fade the pattern out rather than letting it alias when a streak stops covering pixels.
      // Under this camera the world-units-per-pixel is the same everywhere on screen, so it is one
      // number computed here instead of fwidth() in the shader.
      const worldPerPixel =
        (2 * viewSize * aspect) / Math.max(1, lastWidth * renderer.getPixelRatio());
      seaU.uPx.value = worldPerPixel;
      seaU.uDetail.value = Math.max(0, Math.min(1, (0.55 - worldPerPixel) / 0.35));
    },

    update(dt) {
      // Wrapped, and highp in the shader. A plain seconds-since-load clock loses resolution fast
      // in mediump -- past about 32 seconds the step exceeds a frame and the water judders.
      seaTime = (seaTime + dt) % 3600;
      seaU.uTime.value = seaTime;
    },

    render() {
      gpuTimer.begin();
      try {
        renderer.render(scene, camera);
      } finally {
        gpuTimer.end();
      }
      return gpuTimer.poll();
    },

    adapt: (now, frameMs, gpuSamples) =>
      quality.sample(now, frameMs, gpuSamples, gpuTimer.supported),
    setAdaptive: (on) => quality.setEnabled(on),

    get renderScale() {
      return quality.scale;
    },

    get qualityState() {
      return quality.state;
    },

    resize,
  };
}
