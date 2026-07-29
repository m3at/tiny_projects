// The sea, wind and arena boundary, in one opaque full-screen pass.
//
// This is a deliberately flattened Gerstner ocean. Three Gerstner waves are evaluated on a small
// screen grid in the vertex shader, where their height and analytic slope become broad colour
// bands, lit faces and broken whitecaps. The surface is not physically displaced: ships, shots and
// picking all live on the simulation's y=0 plane, and making only the background heave underneath
// them would trade readability for a depth illusion the fixed orthographic camera cannot support.
//
// The useful shape of the reference approach survives -- directional waves, wavelength-derived
// amplitude, sharper crests and normals derived from the same function -- while its expensive parts
// do not. There is no 512-square water mesh, reflection render target, normal texture, lighting
// stack or transparent layer. Three warped wave trains and the soft five-colour palette run per
// vertex; the fragment shader only interpolates the result and draws the arena ring. One draw call,
// no textures or overdraw.

import * as THREE from 'three';
import { ARENA_RADIUS } from '../config.js';
import { SEA } from '../theme.js';

const VERT = /* glsl */ `
  uniform vec4 uMap;
  uniform vec2 uWind;
  uniform float uTime;
  uniform float uDetail;
  uniform vec3 uDeep;
  uniform vec3 uWater;
  uniform vec3 uSwell;
  uniform vec3 uGlint;
  uniform vec3 uFoam;

  varying vec2 vWorld;
  varying vec3 vSeaColor;

  const float TAU = 6.28318530718;

  // Accumulate the visible parts of a Gerstner wave. a = steepness / wave number is the same
  // wavelength-derived amplitude used by a displaced surface; the derivative a*k collapses back to
  // steepness, which gives the surface slope without sampling the function again.
  void wave(
    vec2 p,
    vec2 direction,
    float wavelength,
    float steepness,
    float speed,
    inout float height,
    inout vec2 slope,
    inout vec3 phases
  ) {
    float k = TAU / wavelength;
    float phase = k * dot(direction, p) - speed * uTime;
    float s = sin(phase);
    float c = cos(phase);
    height += (steepness / k) * s;
    slope += direction * steepness * c;
    phases = vec3(phases.yz, s);
  }

  void main() {
    // Orthographic projection makes screen-to-water an exact scale and offset. Keeping the ocean
    // screen filling avoids both a giant world mesh and the horizon/edge cases it brings with it.
    vWorld = uMap.xy + position.xy * uMap.zw;

    vec2 across = vec2(-uWind.y, uWind.x);
    vec2 d0 = uWind;
    vec2 d1 = normalize(uWind * 0.55 + across * 0.84);
    vec2 d2 = normalize(uWind * 0.68 - across * 0.73);

    // Bend the whole wave field slowly across the arena. Without this low-frequency domain warp,
    // three clean sine trains still resolve into a recognisable repeating interference tile when
    // viewed from above. These two sines run per vertex and move much more slowly than the waves.
    float warpA = sin(dot(vWorld, vec2(0.031, -0.019)) + uTime * 0.07);
    float warpB = sin(dot(vWorld, vec2(-0.017, 0.027)) - uTime * 0.05);
    vec2 samplePoint = vWorld + 3.6 * vec2(warpA, warpB);
    float broadPatch = 0.5 + 0.25 * (warpA + warpB);

    float height = 0.0;
    vec2 slope = vec2(0.0);
    vec3 phases = vec3(0.0);
    wave(samplePoint, d0, 16.0, 0.11, 1.15, height, slope, phases);
    wave(samplePoint, d1, 10.5, 0.10, 1.50, height, slope, phases);
    wave(samplePoint, d2, 7.2, 0.07, 1.90, height, slope, phases);

    // Normalise by the sum of the three amplitudes. This keeps the palette thresholds meaningful
    // if the wavelengths are tuned later.
    float waveHeight = height / 0.53;
    vec3 normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    float light = dot(normal, normalize(vec3(-0.42, 0.88, 0.22)));

    // White water belongs to the high, steep side of a swell. A second cross-crest rhythm breaks a
    // continuous contour into hand-painted dashes. Both signals are evaluated here, not per pixel.
    float crest = waveHeight + length(slope) * 0.18;
    float breakup = 0.5 +
      0.25 * sin(dot(vWorld, across) * 0.61 + phases.z * 1.3 - uTime * 0.18) +
      0.25 * sin(dot(vWorld, d1) * 0.37 - phases.x * 1.7 + uTime * 0.11);

    // Broad, low-contrast transitions interpolate cleanly across this grid. The earlier faceting
    // came from hard palette thresholds, not from vertex shading itself.
    float body = clamp(0.38 + broadPatch * 0.30 + waveHeight * 0.12, 0.0, 1.0);
    vSeaColor = mix(uDeep, uWater, body);
    vSeaColor = mix(
      vSeaColor,
      uSwell,
      smoothstep(0.12, 0.62, waveHeight) * 0.32
    );
    vSeaColor = mix(vSeaColor, uGlint, smoothstep(0.875, 0.97, light) * 0.14);

    // Wide ramps keep these sparse caps soft even when the smallest wave spans only a few grid
    // cells at the four-player framing.
    float cap = smoothstep(0.70, 0.84, crest) *
      (1.0 - smoothstep(0.90, 1.02, crest));
    float broken = smoothstep(0.64, 0.94, breakup);
    vSeaColor = mix(vSeaColor, uFoam, cap * broken * uDetail * 0.13);

    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const FRAG = /* glsl */ `
  precision highp float;

  uniform float uPx;
  // (radius squared, 2 * radius * half-thickness, 2 * radius). This lets the fragment shader draw
  // the circular boundary in squared-distance space, without a square root per pixel.
  uniform vec3 uRing;
  uniform vec3 uRingCol;

  varying vec2 vWorld;
  varying vec3 vSeaColor;

  void main() {
    vec3 col = vSeaColor;

    // Arena boundary in the opaque pass: one fewer transparent draw call, correct antialiasing at
    // every render scale, and no overdraw.
    float ringDelta = abs(dot(vWorld, vWorld) - uRing.x);
    float ring = 1.0 - smoothstep(uRing.y, uRing.y + uPx * 1.5 * uRing.z, ringDelta);
    col = mix(col, uRingCol, ring * 0.31);

    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`;

export function createSea() {
  // At 160 by 90 a 1440p capture interpolates across roughly nine pixels rather than sixteen, which
  // removes the visible triangular steps without approaching the reference demo's 263,169 vertices.
  // This is 14,651 vertices, and the expensive reflection/texture fragment work is still absent.
  const geometry = new THREE.PlaneGeometry(2, 2, 160, 90);
  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uMap: { value: new THREE.Vector4(0, 0, 100, -100) },
      uWind: { value: new THREE.Vector2(0, -1) },
      uTime: { value: 0 },
      uDetail: { value: 1 },
      uPx: { value: 0.2 },
      uRing: {
        value: new THREE.Vector3(
          (ARENA_RADIUS - 0.45) ** 2,
          2 * (ARENA_RADIUS - 0.45) * 0.45,
          2 * (ARENA_RADIUS - 0.45),
        ),
      },
      uRingCol: { value: new THREE.Color(SEA.arenaRing) },
      uDeep: { value: new THREE.Color(SEA.deep) },
      uWater: { value: new THREE.Color(SEA.water) },
      uSwell: { value: new THREE.Color(SEA.swell) },
      uGlint: { value: new THREE.Color(SEA.glint) },
      uFoam: { value: new THREE.Color(SEA.foam) },
    },
    depthTest: false,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'sea';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  return mesh;
}
