// Projectiles and particles. Flat quads lying on the water plane rather than true
// billboards: from a top-down camera they read the same and cost nothing.

import * as THREE from 'three';
import { puffTexture } from './glyphs.js';
import { FX } from '../theme.js';

const MAX_SHOT = 500;
const MAX_PUFF = 260;
const MAX_RING = 60;

export function createFx(scene) {
  // ---- shot in flight ----
  const shotGeo = new THREE.SphereGeometry(0.42, 8, 6);
  const shots = new THREE.InstancedMesh(
    shotGeo,
    new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.5 }),
    MAX_SHOT,
  );
  shots.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  shots.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SHOT * 3), 3);
  shots.frustumCulled = false;
  shots.count = 0;
  scene.add(shots);

  // ---- soft puffs (smoke, flash) ----
  const puffGeo = new THREE.PlaneGeometry(1, 1);
  const puffMat = new THREE.MeshBasicMaterial({
    map: puffTexture(),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const puffs = [];
  const puffPool = [];
  for (let i = 0; i < MAX_PUFF; i++) {
    const m = new THREE.Mesh(puffGeo, puffMat.clone());
    m.rotation.x = -Math.PI / 2;
    m.visible = false;
    scene.add(m);
    puffPool.push(m);
  }

  // ---- expanding rings (impacts, detonations, splashes) ----
  const ringGeo = new THREE.RingGeometry(0.72, 1, 28);
  const rings = [];
  const ringPool = [];
  for (let i = 0; i < MAX_RING; i++) {
    const m = new THREE.Mesh(
      ringGeo,
      new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide }),
    );
    m.rotation.x = -Math.PI / 2;
    m.visible = false;
    scene.add(m);
    ringPool.push(m);
  }

  const dummy = new THREE.Object3D();
  const tmpColor = new THREE.Color();

  function puff(x, z, opts) {
    const mesh = puffPool.pop();
    if (!mesh) return;
    mesh.visible = true;
    mesh.position.set(x, opts.y ?? 0.6, z);
    mesh.material.color.set(opts.color ?? 0xffffff);
    mesh.material.opacity = opts.opacity ?? 0.8;
    puffs.push({
      mesh,
      t: 0,
      life: opts.life ?? 0.7,
      from: opts.from ?? 2,
      to: opts.to ?? 7,
      op: opts.opacity ?? 0.8,
      rise: opts.rise ?? 1.4,
      drift: opts.drift ?? 0,
      dx: opts.dx ?? 0,
      dz: opts.dz ?? 0,
    });
  }

  function ring(x, z, opts) {
    const mesh = ringPool.pop();
    if (!mesh) return;
    mesh.visible = true;
    mesh.position.set(x, 0.15, z);
    mesh.material.color.set(opts.color ?? 0xffffff);
    rings.push({ mesh, t: 0, life: opts.life ?? 0.45, to: opts.to ?? 6, op: opts.opacity ?? 0.7 });
  }

  let windTo = 0;

  return {
    setWind(w) {
      windTo = w;
    },

    // Translate one tick's worth of sim effects into visuals.
    consume(effects) {
      for (const e of effects) {
        if (e.type === 'muzzle') {
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
        } else if (e.type === 'impact') {
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
        } else if (e.type === 'crew') {
          puff(e.x, e.z, { color: FX.crew, life: 0.4, from: 1.2, to: 3.4, opacity: 0.55, y: 1.2 });
        } else if (e.type === 'destroy') {
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
        } else if (e.type === 'sever') {
          puff(e.x, e.z, { color: FX.splinters, life: 1.1, from: 3, to: 11, opacity: 0.42, rise: 1.5 });
        } else if (e.type === 'detonate') {
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
        } else if (e.type === 'splash') {
          ring(e.x, e.z, { color: FX.splash, life: 0.5, to: 3.6, opacity: 0.4 });
        }
      }
    },

    update(dt, projectiles) {
      // shots
      let n = 0;
      for (const p of projectiles) {
        if (n >= MAX_SHOT) break;
        dummy.position.set(p.x, 1.1, p.z);
        const s = p.kind === 'grape' ? 0.55 : 1;
        dummy.scale.setScalar(s);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        shots.setMatrixAt(n, dummy.matrix);
        tmpColor.setHex(p.kind === 'grape' ? FX.grapeShot : FX.roundShot);
        shots.setColorAt(n, tmpColor);
        n++;
      }
      shots.count = n;
      shots.instanceMatrix.needsUpdate = true;
      if (shots.instanceColor) shots.instanceColor.needsUpdate = true;

      // puffs
      for (let i = puffs.length - 1; i >= 0; i--) {
        const p = puffs[i];
        p.t += dt;
        const k = p.t / p.life;
        if (k >= 1) {
          p.mesh.visible = false;
          puffPool.push(p.mesh);
          puffs.splice(i, 1);
          continue;
        }
        const size = p.from + (p.to - p.from) * k;
        p.mesh.scale.setScalar(size);
        p.mesh.position.y += p.rise * dt;
        p.mesh.position.x += p.dx * dt;
        p.mesh.position.z += p.dz * dt;
        p.mesh.material.opacity = p.op * (1 - k) * (1 - k);
      }

      // rings
      for (let i = rings.length - 1; i >= 0; i--) {
        const r = rings[i];
        r.t += dt;
        const k = r.t / r.life;
        if (k >= 1) {
          r.mesh.visible = false;
          ringPool.push(r.mesh);
          rings.splice(i, 1);
          continue;
        }
        r.mesh.scale.setScalar(0.5 + r.to * k);
        r.mesh.material.opacity = r.op * (1 - k);
      }
    },

    reset() {
      for (const p of puffs) {
        p.mesh.visible = false;
        puffPool.push(p.mesh);
      }
      puffs.length = 0;
      for (const r of rings) {
        r.mesh.visible = false;
        ringPool.push(r.mesh);
      }
      rings.length = 0;
      shots.count = 0;
    },
  };
}
