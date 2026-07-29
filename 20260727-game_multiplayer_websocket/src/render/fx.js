// Cannonballs and every transient effect, in two instanced draw calls.
//
// The old renderer was already well batched, but every event was either the same Gaussian circle
// or the same geometric ring. This keeps the batching and replaces the vocabulary: one tiny atlas
// and one sprite shader can draw smoke, flashes, timber, water crowns and foam.
// Sprites may face the camera or lie on the sea, so volume and water contact no longer look alike.

import * as THREE from 'three';
import { fxAtlasTexture, FX_TILE } from './glyphs.js';
import { FX } from '../theme.js';

const MAX_SHOT = 400;
const MAX_SPRITE = 720;

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

function makeShots() {
  const mesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.42, 8, 6),
    new THREE.MeshLambertMaterial(),
    MAX_SHOT,
  );
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SHOT * 3), 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceMatrix.array.fill(0);
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.name = 'shots';
  return mesh;
}

const SPRITE_VERT = /* glsl */ `
  attribute vec3 iPosition;
  attribute vec3 iScaleAngle;
  attribute vec4 iColorAlpha;
  attribute vec2 iTileFacing;
  varying vec2 vUv;
  varying vec4 vColorAlpha;
  varying float vTile;

  void main() {
    float c = cos(iScaleAngle.z);
    float s = sin(iScaleAngle.z);
    vec2 corner = position.xy * iScaleAngle.xy;
    corner = mat2(c, -s, s, c) * corner;
    vec4 centre = modelViewMatrix * vec4(iPosition, 1.0);
    vec4 mvPosition;
    if (iTileFacing.y > 0.5) {
      // Camera-facing: smoke and debris have height instead of lying painted on the water.
      mvPosition = centre + vec4(corner, 0.0, 0.0);
    } else {
      // Sea-facing: foam stays attached to the surface.
      // Negating y preserves the quad's front-face winding when XY is folded onto XZ. Without it
      // every sea-facing sprite was back-face culled; only the camera-facing half of the atlas drew.
      mvPosition = modelViewMatrix * vec4(iPosition + vec3(corner.x, 0.0, -corner.y), 1.0);
    }
    gl_Position = projectionMatrix * mvPosition;
    vUv = position.xy + 0.5;
    vColorAlpha = iColorAlpha;
    vTile = iTileFacing.x;
  }
`;

const SPRITE_FRAG = /* glsl */ `
  precision mediump float;
  uniform sampler2D uAtlas;
  varying vec2 vUv;
  varying vec4 vColorAlpha;
  varying float vTile;

  void main() {
    float tile = floor(vTile + 0.5);
    // Canvas textures are uploaded bottom-up, while the atlas authoring helpers count rows from
    // the canvas top. Mirror the two-row index here.
    vec2 cell = vec2(mod(tile, 4.0), 1.0 - floor(tile / 4.0));
    // A one-pixel transparent gutter keeps linear filtering inside the selected cell.
    vec2 localUv = mix(vec2(0.02), vec2(0.98), vUv);
    float mask = texture2D(uAtlas, (cell + localUv) / vec2(4.0, 2.0)).a;
    float alpha = mask * vColorAlpha.a;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(vColorAlpha.rgb, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    // Premultiplied alpha lets pale flashes glow while grey smoke actually veils what is behind it
    // instead of adding light, which was the old Gaussian-puff look.
    gl_FragColor = vec4(gl_FragColor.rgb * alpha, alpha);
  }
`;

function dynamicAttribute(size) {
  const attr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SPRITE * size), size);
  attr.setUsage(THREE.DynamicDrawUsage);
  return attr;
}

function makeSprites() {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      new Float32Array([
        -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0,
        -0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
      ]),
      3,
    ),
  );
  geometry.setAttribute('iPosition', dynamicAttribute(3));
  geometry.setAttribute('iScaleAngle', dynamicAttribute(3));
  geometry.setAttribute('iColorAlpha', dynamicAttribute(4));
  geometry.setAttribute('iTileFacing', dynamicAttribute(2));
  geometry.instanceCount = 0;

  const material = new THREE.ShaderMaterial({
    vertexShader: SPRITE_VERT,
    fragmentShader: SPRITE_FRAG,
    uniforms: { uAtlas: { value: fxAtlasTexture() } },
    transparent: true,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    premultipliedAlpha: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'sprites';
  mesh.frustumCulled = false;
  mesh.visible = false;
  return mesh;
}

function uploadAttribute(attr, live) {
  attr.clearUpdateRanges();
  attr.addUpdateRange(0, live * attr.itemSize);
  attr.needsUpdate = true;
}

export function createFx(scene) {
  const shots = makeShots();
  const sprites = makeSprites();
  scene.add(shots, sprites);

  const spriteGeo = sprites.geometry;
  const positions = spriteGeo.getAttribute('iPosition');
  const scales = spriteGeo.getAttribute('iScaleAngle');
  const colours = spriteGeo.getAttribute('iColorAlpha');
  const tiles = spriteGeo.getAttribute('iTileFacing');
  const colour = new THREE.Color();

  // Objects are retained and reused after swap-removal. Event recipes allocate no meshes,
  // geometries, materials or GPU resources, and after the high-water mark they allocate no particle
  // records either.
  const particles = new Array(MAX_SPRITE);
  let liveCount = 0;
  let windTo = 0;
  let entropy = 0x62b9d1a5;

  function random() {
    entropy ^= entropy << 13;
    entropy ^= entropy >>> 17;
    entropy ^= entropy << 5;
    return (entropy >>> 0) / 4294967296;
  }

  function between(a, b) {
    return a + (b - a) * random();
  }

  function spawn(
    x, y, z, tile, color, life,
    w0, h0, w1, h1,
    opacity = 1, facing = 1, angle = 0,
    vx = 0, vy = 0, vz = 0, spin = 0, gravity = 0, fade = 2,
  ) {
    if (liveCount >= MAX_SPRITE - 8) return;
    let p = particles[liveCount];
    if (!p) p = particles[liveCount] = {};
    liveCount++;
    p.x = x;
    p.y = y;
    p.z = z;
    p.tile = tile;
    p.color = color;
    p.life = life;
    p.w0 = w0;
    p.h0 = h0;
    p.w1 = w1;
    p.h1 = h1;
    p.opacity = opacity;
    p.facing = facing;
    p.angle = angle;
    p.vx = vx;
    p.vy = vy;
    p.vz = vz;
    p.spin = spin;
    p.gravity = gravity;
    p.fade = fade;
    p.t = 0;
  }

  function smoke(x, z, scale = 1, life = 1.5, opacity = 0.34) {
    spawn(
      x, 1.2, z, FX_TILE.smoke, FX.muzzleSmoke, life,
      3.2 * scale, 2.7 * scale, 10 * scale, 7.5 * scale,
      opacity, 1, between(-0.5, 0.5),
      Math.sin(windTo) * 4.5, 1.25, -Math.cos(windTo) * 4.5,
      between(-0.15, 0.15),
    );
  }

  function foam(x, z, size, life = 0.7, opacity = 0.58, angle = 0) {
    spawn(
      x, 0.35, z, FX_TILE.foam, FX.foam, life,
      size * 0.35, size * 0.22, size, size * 0.6,
      opacity, 0, angle, 0, 0, 0, between(-0.2, 0.2), 0, 1.4,
    );
  }

  function splinters(x, z, color, count, force = 1) {
    for (let i = 0; i < count; i++) {
      const a = between(-Math.PI, Math.PI);
      const speed = between(1.5, 5.5) * force;
      spawn(
        x + between(-0.5, 0.5), 1.1, z + between(-0.5, 0.5),
        FX_TILE.splinter, color, between(0.35, 0.7),
        between(1.1, 2.2), between(1.8, 3.8), 0.45, 1.1,
        between(0.55, 0.9), 1, a,
        Math.cos(a) * speed, between(2, 5) * force, Math.sin(a) * speed,
        between(-4, 4), 10,
      );
    }
  }

  function consume(effects) {
    for (const e of effects) {
      switch (e.type) {
        case 'muzzle': {
          const dx = Math.sin(e.heading);
          const dz = -Math.cos(e.heading);
          spawn(
            e.x + dx * 1.5, 1.1, e.z + dz * 1.5,
            FX_TILE.flash, FX.muzzleFlash, 0.14,
            e.big ? 8 : 5, e.big ? 5 : 3.5, e.big ? 11 : 7, 1.2,
            1, 1, between(-0.25, 0.25), dx * 2, 0.7, dz * 2, 0, 0, 1,
          );
          for (let i = 0; i < (e.big ? 3 : 2); i++) {
            smoke(e.x + dx * (2.5 + i * 1.4), e.z + dz * (2.5 + i * 1.4), 0.85 + i * 0.16);
          }
          break;
        }
        case 'impact': {
          const grape = e.kind === 'grape';
          spawn(
            e.x, 1.1, e.z, FX_TILE.flash,
            grape ? FX.impactGrape : FX.impactRound, grape ? 0.1 : 0.16,
            grape ? 2.5 : 4.2, grape ? 2.5 : 4.2,
            grape ? 3.2 : 6.5, grape ? 3.2 : 5,
            0.9, 1, between(-0.5, 0.5), 0, 0.8, 0, 0, 0, 1,
          );
          splinters(e.x, e.z, grape ? FX.impactGrape : FX.splinters, grape ? 2 : 4, grape ? 0.65 : 1);
          break;
        }
        case 'crew':
          splinters(e.x, e.z, FX.crew, 3, 0.55);
          break;
        case 'destroy':
          smoke(e.x, e.z, 0.8, 1.25, 0.38);
          splinters(e.x, e.z, e.part === 'mast' ? FX.splinters : FX.debris, e.part === 'mast' ? 7 : 5, 1.1);
          break;
        case 'sever':
          splinters(e.x, e.z, FX.splinters, 6, 1.25);
          foam(e.x, e.z, 5, 0.55, 0.35);
          break;
        case 'detonate': {
          spawn(
            e.x, 1.8, e.z, FX_TILE.core, FX.blastCore, 0.34,
            14, 12, 31, 25, 1, 1, between(-0.4, 0.4), 0, 4, 0, 0.4, 0, 1,
          );
          foam(e.x, e.z, 28, 0.8, 0.82);
          for (let i = 0; i < 6; i++) {
            const a = (i / 6) * Math.PI * 2 + between(-0.22, 0.22);
            spawn(
              e.x + Math.cos(a) * between(1, 4), 1.2, e.z + Math.sin(a) * between(1, 4),
              FX_TILE.core, FX.blastFire, between(0.45, 0.8),
              between(5, 9), between(5, 10), between(12, 19), between(15, 25),
              0.75, 1, a, Math.cos(a) * 7, between(2, 5), Math.sin(a) * 7,
              between(-0.5, 0.5),
            );
          }
          for (let i = 0; i < 8; i++) {
            const a = (i / 8) * Math.PI * 2 + between(-0.3, 0.3);
            spawn(
              e.x + Math.cos(a) * 3, 1.5, e.z + Math.sin(a) * 3,
              FX_TILE.smoke, FX.blastSmoke, between(1.6, 2.4),
              5, 4, between(15, 22), between(12, 19),
              0.48, 1, a, Math.cos(a) * 6, between(2, 4), Math.sin(a) * 6,
              between(-0.2, 0.2),
            );
          }
          splinters(e.x, e.z, FX.splinters, 10, 1.8);
          break;
        }
        case 'splash':
          spawn(
            e.x, 0.7, e.z, FX_TILE.splash, FX.splash, 0.48,
            2.3, 3.8, 4.8, 6.8, 0.72, 1, between(-0.3, 0.3),
            0, 2.5, 0, 0, 8,
          );
          break;
        default:
          break;
      }
    }
  }

  function writeSprite(i, x, y, z, w, h, angle, colorHex, alpha, tile, facing) {
    positions.setXYZ(i, x, y, z);
    scales.setXYZ(i, w, h, angle);
    colour.setHex(colorHex);
    colours.setXYZW(i, colour.r, colour.g, colour.b, alpha);
    tiles.setXY(i, tile, facing);
  }

  function update(dt, projectiles) {
    let shotCount = 0;
    for (const p of projectiles) {
      if (shotCount >= MAX_SHOT) break;
      writeTS(
        shots.instanceMatrix.array,
        shotCount,
        p.x,
        1.1,
        p.z,
        p.kind === 'grape' ? 0.55 : 1,
      );
      colour.setHex(p.kind === 'grape' ? FX.grapeShot : FX.roundShot);
      shots.instanceColor.setXYZ(shotCount, colour.r, colour.g, colour.b);
      shotCount++;
    }
    shots.count = shotCount;
    shots.visible = shotCount > 0;
    if (shotCount > 0) {
      shots.instanceMatrix.clearUpdateRanges();
      shots.instanceMatrix.addUpdateRange(0, shotCount * 16);
      shots.instanceMatrix.needsUpdate = true;
      shots.instanceColor.clearUpdateRanges();
      shots.instanceColor.addUpdateRange(0, shotCount * 3);
      shots.instanceColor.needsUpdate = true;
    }

    for (let i = liveCount - 1; i >= 0; i--) {
      const p = particles[i];
      p.t += dt;
      if (p.t >= p.life) {
        const dead = p;
        liveCount--;
        particles[i] = particles[liveCount];
        particles[liveCount] = dead;
        continue;
      }
      p.vy -= p.gravity * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      p.angle += p.spin * dt;
    }

    let n = 0;
    for (; n < liveCount && n < MAX_SPRITE; n++) {
      const p = particles[n];
      const k = p.t / p.life;
      writeSprite(
        n, p.x, p.y, p.z,
        p.w0 + (p.w1 - p.w0) * k,
        p.h0 + (p.h1 - p.h0) * k,
        p.angle, p.color, p.opacity * Math.pow(1 - k, p.fade), p.tile, p.facing,
      );
    }

    spriteGeo.instanceCount = n;
    sprites.visible = n > 0;
    if (n > 0) {
      uploadAttribute(positions, n);
      uploadAttribute(scales, n);
      uploadAttribute(colours, n);
      uploadAttribute(tiles, n);
    }
  }

  function sinkBurst(ship, stage) {
    const forwardX = ship.sin;
    const forwardZ = -ship.cos;
    const sideX = ship.cos;
    const sideZ = ship.sin;
    const along = stage === 0 ? 0 : stage % 2 ? -3.5 : 3.5;
    const side = stage === 0 ? 0 : (stage % 2 ? -1 : 1) * 2.8;
    const x = ship.x + forwardX * along + sideX * side;
    const z = ship.z + forwardZ * along + sideZ * side;
    if (stage < 3) {
      spawn(
        x, 0.8, z, FX_TILE.splash, FX.splash, 0.65,
        stage === 0 ? 7 : 4, stage === 0 ? 10 : 6,
        stage === 0 ? 12 : 7, stage === 0 ? 15 : 10,
        0.68, 1, between(-0.4, 0.4), 0, 2.2, 0, 0, 7,
      );
    }
  }

  return {
    setWind(w) {
      windTo = w;
    },
    consume,
    update,
    sinkBurst,
    reset() {
      liveCount = 0;
      entropy = 0x62b9d1a5;
      shots.count = 0;
      shots.visible = false;
      spriteGeo.instanceCount = 0;
      sprites.visible = false;
    },
  };
}
