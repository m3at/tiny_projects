// Scene, sea and camera. Orthographic and tilted: a plan view you can still read, with
// enough perspective that masts and gun boxes have height.
//
// Two things this file is careful about, because frame() runs every frame:
//   - the canvas is only resized and the projection only rebuilt when something actually
//     changed, rather than unconditionally;
//   - the camera's orientation is fixed for the whole game (constant tilt, constant offset
//     from the target), so it is computed once and only the position moves after that.

import * as THREE from 'three';
import { ARENA_RADIUS } from '../config.js';
import { SEA, LIGHT } from '../theme.js';

const TILT = (60 * Math.PI) / 180; // from horizontal
const CAMERA_DISTANCE = 300; // irrelevant to scale under orthographic; just clears the scene
const STREAK_COUNT = 420;
const STREAK_SPREAD = 1.9; // how far past the viewport edge streaks are scattered

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(SEA.background);

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

  const sea = new THREE.Mesh(
    new THREE.PlaneGeometry(1400, 1400).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: SEA.water, roughness: 0.55, metalness: 0.1 }),
  );
  sea.position.y = -0.4;
  scene.add(sea);

  // Marks the edge of the engagement area, so the containment nudge does not look like the
  // ships randomly changing their minds.
  const arena = new THREE.Mesh(
    new THREE.RingGeometry(ARENA_RADIUS - 0.9, ARENA_RADIUS, 128).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: SEA.arenaRing,
      transparent: true,
      // Faint: the fight settles near the middle now, so the ring is a hint about where the
      // stage ends rather than a boundary anyone touches.
      opacity: 0.22,
      side: THREE.DoubleSide,
    }),
  );
  arena.position.y = -0.3;
  scene.add(arena);

  // ---- wind streaks ----
  // They drift downwind, which is what makes the wind direction legible without reading a
  // dial. Positions are normalised around the camera target so they always fill the frame
  // however far the zoom has travelled; keeping world positions would strand them all in one
  // corner after a zoom-out.
  const streaks = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(1, 0.22).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: SEA.windStreak, transparent: true, opacity: 0.14 }),
    STREAK_COUNT,
  );
  streaks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  streaks.frustumCulled = false;
  scene.add(streaks);

  const streakData = [];
  for (let i = 0; i < STREAK_COUNT; i++) {
    streakData.push({
      ux: Math.random() * 2 - 1,
      uz: Math.random() * 2 - 1,
      len: 2.5 + Math.random() * 7,
      speed: 0.55 + Math.random() * 0.75,
    });
  }

  let windTo = 0;
  const streakRotation = new THREE.Matrix4();
  let streakRotationStale = true;

  const wrapUnit = (v) => ((((v + 1) % 2) + 2) % 2) - 1;

  // Each streak is a flat quad long in +x; rotating about y by (90deg - windTo) points it
  // downwind. The rotation is shared, so only the x column carries the per-streak length and
  // the matrices can be written straight into the buffer.
  function updateStreaks(dt) {
    if (streakRotationStale) {
      streakRotation.makeRotationY(Math.PI / 2 - windTo);
      streakRotationStale = false;
    }
    const e = streakRotation.elements;
    const arr = streaks.instanceMatrix.array;
    const radius = viewSize * STREAK_SPREAD;
    const vx = Math.sin(windTo);
    const vz = -Math.cos(windTo);
    const lengthScale = viewSize / 24;

    for (let i = 0; i < STREAK_COUNT; i++) {
      const s = streakData[i];
      s.ux = wrapUnit(s.ux + vx * s.speed * dt * 0.42);
      s.uz = wrapUnit(s.uz + vz * s.speed * dt * 0.42);
      const len = s.len * lengthScale;
      const o = i * 16;
      arr[o] = e[0] * len;
      arr[o + 1] = e[1] * len;
      arr[o + 2] = e[2] * len;
      arr[o + 4] = e[4];
      arr[o + 5] = e[5];
      arr[o + 6] = e[6];
      arr[o + 8] = e[8];
      arr[o + 9] = e[9];
      arr[o + 10] = e[10];
      arr[o + 12] = target.x + s.ux * radius;
      arr[o + 13] = -0.28;
      arr[o + 14] = target.z + s.uz * radius;
      arr[o + 15] = 1;
    }
    streaks.instanceMatrix.needsUpdate = true;
  }

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

  function resize() {
    const w = canvas.clientWidth || innerWidth;
    const h = canvas.clientHeight || innerHeight;
    if (w === lastWidth && h === lastHeight) return;
    lastWidth = w;
    lastHeight = h;
    renderer.setSize(w, h, false);
    updateProjection();
  }

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
      windTo = w;
      streakRotationStale = true;
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
      resize();
      if (Math.abs(viewSize - lastViewSize) > 1e-4) updateProjection();
      placeCamera();
    },

    update(dt) {
      updateStreaks(dt);
    },

    render() {
      renderer.render(scene, camera);
    },

    resize,
  };
}
