// Scene, sea and camera. Orthographic and tilted: a plan view you can still read, with
// enough perspective that masts and gun boxes have height.

import * as THREE from 'three';
import { ARENA_RADIUS } from '../config.js';
import { SEA, LIGHT } from '../theme.js';

const TILT = (60 * Math.PI) / 180; // from horizontal
const STREAK_COUNT = 420;

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(SEA.background);

  // No fog: the orthographic camera sits far back for the tilt, so distance-based fog
  // would flatten the whole scene to one colour.
  const scene = new THREE.Scene();

  const camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.1, 900);
  const target = new THREE.Vector3();
  let viewSize = 70;

  scene.add(new THREE.HemisphereLight(LIGHT.sky, LIGHT.ground, 1.15));
  const sun = new THREE.DirectionalLight(LIGHT.sun, 1.5);
  sun.position.set(-45, 90, 38);
  scene.add(sun);

  // Sea
  const sea = new THREE.Mesh(
    new THREE.PlaneGeometry(1400, 1400),
    new THREE.MeshStandardMaterial({ color: SEA.water, roughness: 0.55, metalness: 0.1 }),
  );
  sea.rotation.x = -Math.PI / 2;
  sea.position.y = -0.4;
  scene.add(sea);

  // A ring marking the edge of the engagement area, so the containment nudge doesn't look
  // like the ships randomly changing their minds.
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(ARENA_RADIUS - 0.9, ARENA_RADIUS, 128),
    new THREE.MeshBasicMaterial({
      color: SEA.arenaRing,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -0.3;
  scene.add(ring);

  // Wind streaks. These double as the wind indicator: they drift downwind, so the
  // direction is legible without reading a dial.
  const streakGeo = new THREE.PlaneGeometry(1, 0.22);
  const streakMat = new THREE.MeshBasicMaterial({
    color: SEA.windStreak,
    transparent: true,
    opacity: 0.22,
  });
  const streaks = new THREE.InstancedMesh(streakGeo, streakMat, STREAK_COUNT);
  streaks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  streaks.frustumCulled = false;
  scene.add(streaks);

  // Streaks live in normalised space around the camera target, so they always fill the
  // frame no matter how far the zoom has travelled. Keeping world positions instead lets a
  // zoom-out strand them all in one corner.
  const streakData = [];
  for (let i = 0; i < STREAK_COUNT; i++) {
    streakData.push({
      ux: Math.random() * 2 - 1,
      uz: Math.random() * 2 - 1,
      len: 2.5 + Math.random() * 7,
      speed: 0.55 + Math.random() * 0.75,
    });
  }
  const dummy = new THREE.Object3D();
  let windTo = 0;

  const wrapUnit = (v) => (((v + 1) % 2) + 2) % 2 - 1;

  function updateStreaks(dt) {
    const vx = Math.sin(windTo);
    const vz = -Math.cos(windTo);
    const R = viewSize * 1.9;
    for (let i = 0; i < STREAK_COUNT; i++) {
      const s = streakData[i];
      s.ux = wrapUnit(s.ux + vx * s.speed * dt * 0.42);
      s.uz = wrapUnit(s.uz + vz * s.speed * dt * 0.42);
      dummy.position.set(target.x + s.ux * R, -0.28, target.z + s.uz * R);
      dummy.rotation.set(-Math.PI / 2, 0, -windTo + Math.PI / 2);
      dummy.scale.set(s.len * (viewSize / 24), 1, 1);
      dummy.updateMatrix();
      streaks.setMatrixAt(i, dummy.matrix);
    }
    streaks.instanceMatrix.needsUpdate = true;
  }

  function resize() {
    const w = canvas.clientWidth || innerWidth;
    const h = canvas.clientHeight || innerHeight;
    renderer.setSize(w, h, false);
    const aspect = w / h;
    camera.left = -viewSize * aspect;
    camera.right = viewSize * aspect;
    camera.top = viewSize;
    camera.bottom = -viewSize;
    camera.updateProjectionMatrix();
  }

  function placeCamera() {
    const d = 300;
    camera.position.set(
      target.x,
      target.y + Math.sin(TILT) * d,
      target.z + Math.cos(TILT) * d,
    );
    camera.lookAt(target);
  }

  return {
    renderer,
    scene,
    camera,
    setWind(w) {
      windTo = w;
    },
    // Ease toward a framing rather than snapping, so the battle camera glides.
    frame(cx, cz, size, snap = false) {
      const k = snap ? 1 : 0.08;
      target.x += (cx - target.x) * k;
      target.z += (cz - target.z) * k;
      viewSize += (size - viewSize) * (snap ? 1 : 0.05);
      resize();
      placeCamera();
    },
    update(dt) {
      updateStreaks(dt);
    },
    render() {
      renderer.render(scene, camera);
    },
    resize,
    // Screen point -> point on the sea plane, for build-phase picking.
    raycaster: new THREE.Raycaster(),
  };
}
