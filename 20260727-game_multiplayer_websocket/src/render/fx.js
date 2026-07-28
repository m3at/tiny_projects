// Projectiles and particles, as three instanced draw calls total.
//
// Everything here lies flat on the water and is additively blended. Two consequences make
// instancing straightforward: the flat rotation is baked into the geometry, so an instance
// matrix is only a translation and a uniform scale; and under additive blending, fading a
// sprite out is the same as scaling its colour toward black, so per-instance opacity can
// ride along in instanceColor instead of needing one material per particle.

import * as THREE from 'three';
import { puffTexture } from './glyphs.js';
import { FX } from '../theme.js';

const MAX_SHOT = 400;
const MAX_PUFF = 220;
const MAX_RING = 60;

// Write a translation and uniform scale straight into an instanceMatrix buffer. Column
// major, and the off-diagonals stay zero because the geometry carries the flat rotation.
function writeTS(array, i, x, y, z, s) {
  const o = i * 16;
  array[o] = s;
  array[o + 5] = s;
  array[o + 10] = s;
  array[o + 12] = x;
  array[o + 13] = y;
  array[o + 14] = z;
  array[o + 15] = 1;
}

function makeInstanced(geometry, material, count) {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.instanceMatrix.array.fill(0); // once, so writeTS only touches what varies
  return mesh;
}

// Marking an attribute dirty re-uploads the whole array, and these arrays are sized for the worst
// case: 400 shot, 220 puffs, 60 rings. A battle typically has a handful of each, so the default
// behaviour was sending about 50KB a frame to describe a dozen live particles, and sending it even
// when there were none at all. An update range bounds the upload to what was written; no live
// instances means no upload.
function upload(mesh, live) {
  // count is already 0, so nothing is drawn from the buffer and nothing needs to reach the GPU.
  if (live === 0) return;
  mesh.instanceMatrix.clearUpdateRanges();
  mesh.instanceMatrix.addUpdateRange(0, live * 16);
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.clearUpdateRanges();
  mesh.instanceColor.addUpdateRange(0, live * 3);
  mesh.instanceColor.needsUpdate = true;
}

export function createFx(scene) {
  const shots = makeInstanced(
    new THREE.SphereGeometry(0.42, 8, 6),
    // Lambert like the ship parts. A ball in flight is a few pixels across; the specular
    // highlight that metalness bought was never resolvable.
    new THREE.MeshLambertMaterial(),
    MAX_SHOT,
  );
  shots.name = 'shots';
  scene.add(shots);

  const puffs = makeInstanced(
    new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      map: puffTexture(),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
    MAX_PUFF,
  );
  puffs.name = 'puffs';
  scene.add(puffs);

  const rings = makeInstanced(
    new THREE.RingGeometry(0.72, 1, 28).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      // Flat and face-up after the rotateX below, so the camera never sees the back. FrontSide
      // (the default) lets back-face culling drop half the triangles.
    }),
    MAX_RING,
  );
  rings.name = 'rings';
  scene.add(rings);

  // Index into these arrays is the instance index, so they are kept compact.
  const livePuffs = [];
  const liveRings = [];
  const colour = new THREE.Color();
  let windTo = 0;

  function puff(x, z, o) {
    if (livePuffs.length >= MAX_PUFF) return;
    livePuffs.push({
      x,
      y: o.y ?? 0.6,
      z,
      t: 0,
      life: o.life ?? 0.7,
      from: o.from ?? 2,
      to: o.to ?? 7,
      opacity: o.opacity ?? 0.8,
      rise: o.rise ?? 1.4,
      dx: o.dx ?? 0,
      dz: o.dz ?? 0,
      color: o.color ?? 0xffffff,
    });
  }

  function ring(x, z, o) {
    if (liveRings.length >= MAX_RING) return;
    liveRings.push({
      x,
      z,
      t: 0,
      life: o.life ?? 0.45,
      to: o.to ?? 6,
      opacity: o.opacity ?? 0.7,
      color: o.color ?? 0xffffff,
    });
  }

  // Translate one tick's worth of simulation effects into visuals.
  function consume(effects) {
    for (const e of effects) {
      switch (e.type) {
        case 'muzzle': {
          const dx = Math.sin(e.heading);
          const dz = -Math.cos(e.heading);
          puff(e.x + dx * 1.6, e.z + dz * 1.6, {
            color: FX.muzzleFlash,
            life: 0.16,
            from: e.big ? 5 : 3.2,
            to: e.big ? 9 : 6,
            opacity: 0.95,
            y: 0.9,
          });
          for (let i = 0; i < (e.big ? 3 : 2); i++) {
            puff(e.x + dx * (2 + i * 1.7), e.z + dz * (2 + i * 1.7), {
              color: FX.muzzleSmoke,
              life: 1.5 + i * 0.3,
              from: 2.5,
              to: 12,
              opacity: 0.3,
              rise: 1.1,
              dx: Math.sin(windTo) * 5,
              dz: -Math.cos(windTo) * 5,
            });
          }
          break;
        }
        case 'impact': {
          const grape = e.kind === 'grape';
          puff(e.x, e.z, {
            color: grape ? FX.impactGrape : FX.impactRound,
            life: 0.2,
            from: grape ? 1.4 : 2.4,
            to: grape ? 3 : 5.5,
            opacity: 0.85,
            y: 1,
          });
          if (!grape) ring(e.x, e.z, { color: FX.impactRing, life: 0.3, to: 4, opacity: 0.5 });
          break;
        }
        case 'crew':
          puff(e.x, e.z, { color: FX.crew, life: 0.4, from: 1.2, to: 3.4, opacity: 0.55, y: 1.2 });
          break;
        case 'destroy':
          puff(e.x, e.z, {
            color: FX.debris,
            life: 1.3,
            from: 3,
            to: 13,
            opacity: 0.5,
            rise: 2,
            dx: Math.sin(windTo) * 5,
            dz: -Math.cos(windTo) * 5,
          });
          ring(e.x, e.z, { color: FX.destroyRing, life: 0.4, to: 7, opacity: 0.5 });
          break;
        case 'sever':
          puff(e.x, e.z, {
            color: FX.splinters,
            life: 1.1,
            from: 3,
            to: 11,
            opacity: 0.42,
            rise: 1.5,
          });
          break;
        case 'detonate':
          puff(e.x, e.z, { color: FX.blastCore, life: 0.3, from: 8, to: 26, opacity: 1, y: 1.6 });
          puff(e.x, e.z, { color: FX.blastFire, life: 0.7, from: 6, to: 32, opacity: 0.8, y: 1.2 });
          ring(e.x, e.z, { color: FX.blastRing, life: 0.75, to: 26, opacity: 0.85 });
          for (let i = 0; i < 8; i++) {
            const a = (i / 8) * Math.PI * 2;
            puff(e.x + Math.cos(a) * 3, e.z + Math.sin(a) * 3, {
              color: FX.blastSmoke,
              life: 2.2,
              from: 4,
              to: 20,
              opacity: 0.45,
              rise: 3,
              dx: Math.cos(a) * 9,
              dz: Math.sin(a) * 9,
            });
          }
          break;
        case 'splash':
          ring(e.x, e.z, { color: FX.splash, life: 0.5, to: 3.6, opacity: 0.4 });
          break;
        default:
          break;
      }
    }
  }

  function writeColour(array, i, hex, scale) {
    colour.setHex(hex).multiplyScalar(scale);
    array[i * 3] = colour.r;
    array[i * 3 + 1] = colour.g;
    array[i * 3 + 2] = colour.b;
  }

  function update(dt, projectiles) {
    // ---- shot in flight ----
    let n = 0;
    for (const p of projectiles) {
      if (n >= MAX_SHOT) break;
      writeTS(shots.instanceMatrix.array, n, p.x, 1.1, p.z, p.kind === 'grape' ? 0.55 : 1);
      writeColour(shots.instanceColor.array, n, p.kind === 'grape' ? FX.grapeShot : FX.roundShot, 1);
      n++;
    }
    shots.count = n;
    upload(shots, n);

    // ---- puffs: advance, swap-remove the dead, then write the survivors ----
    for (let i = livePuffs.length - 1; i >= 0; i--) {
      const p = livePuffs[i];
      p.t += dt;
      if (p.t >= p.life) {
        livePuffs[i] = livePuffs[livePuffs.length - 1];
        livePuffs.pop();
        continue;
      }
      p.y += p.rise * dt;
      p.x += p.dx * dt;
      p.z += p.dz * dt;
    }
    for (let i = 0; i < livePuffs.length; i++) {
      const p = livePuffs[i];
      const k = p.t / p.life;
      writeTS(puffs.instanceMatrix.array, i, p.x, p.y, p.z, p.from + (p.to - p.from) * k);
      writeColour(puffs.instanceColor.array, i, p.color, p.opacity * (1 - k) * (1 - k));
    }
    puffs.count = livePuffs.length;
    upload(puffs, livePuffs.length);

    // ---- rings ----
    for (let i = liveRings.length - 1; i >= 0; i--) {
      const r = liveRings[i];
      r.t += dt;
      if (r.t >= r.life) {
        liveRings[i] = liveRings[liveRings.length - 1];
        liveRings.pop();
      }
    }
    for (let i = 0; i < liveRings.length; i++) {
      const r = liveRings[i];
      const k = r.t / r.life;
      writeTS(rings.instanceMatrix.array, i, r.x, 0.15, r.z, 0.5 + r.to * k);
      writeColour(rings.instanceColor.array, i, r.color, r.opacity * (1 - k));
    }
    rings.count = liveRings.length;
    upload(rings, liveRings.length);
  }

  return {
    setWind(w) {
      windTo = w;
    },
    consume,
    update,
    reset() {
      livePuffs.length = 0;
      liveRings.length = 0;
      shots.count = 0;
      puffs.count = 0;
      rings.count = 0;
    },
  };
}
