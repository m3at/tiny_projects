// The sea and the wind, as one full-screen triangle.
//
// This replaces two things: the flat clear colour, and the 420 translucent quads that used to drift
// downwind to show wind direction. tools/fill.js priced those quads at 46% of a 720p frame and 29%
// of a 1080p one -- by a wide margin the most expensive thing left in the scene, and they were
// decoration. They were also alpha-blended and overlapping, which is the worst case for a mobile
// tile-based GPU: blending switches off the hidden-surface removal that the whole architecture is
// built around, so every layer costs full price. One opaque triangle has no overdraw and no
// blending at all.
//
// Three things make this cheap enough to be worth doing.
//
// The projection is affine, so there is no matrix in the vertex shader. An orthographic camera has
// no perspective divide, so the map from screen to the y=0 plane is a scale and an offset -- two
// uniforms. Checked against a real unproject-and-raycast over the frame: agreement to 1e-13 world
// units, which is to say it is not an approximation.
//
// Nothing is lit. Under an orthographic camera looking at a flat plane, the surface normal and the
// view direction are both constant over the whole screen, so every view-dependent term -- fresnel,
// specular, environment reflection -- collapses to a constant. That is why the old MeshStandard sea
// plane was computing one colour two million times a frame. Banding a scalar field is view
// independent, so it is the shape of shading that actually survives here.
//
// The pattern is squashed along the wind axis. Stretching isotropic noise 7:1 turns blobs into
// wind-aligned streaks for free, which is the entire trick: the thing the 420 quads existed to
// convey now falls out of two noise lookups.

import * as THREE from 'three';
import { ARENA_RADIUS } from '../config.js';
import { SEA } from '../theme.js';

// How far the noise is stretched along the wind and across it. The ratio is what reads as wind;
// the absolute numbers set the size of a streak in world units (1 / value).
const ALONG = 0.15;
const ACROSS = 1.05;
const DRIFT = 3.4; // noise units a second, downwind

const VERT = /* glsl */ `
  // uMap is (target.x, target.z, halfWidthWorld, -halfHeightWorld / sin(tilt)). The tilt divisor is
  // there because a 60-degree camera foreshortens the z axis by exactly that much.
  uniform vec4 uMap;
  varying vec2 vWorld;
  void main() {
    vWorld = uMap.xy + position.xy * uMap.zw;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const FRAG = /* glsl */ `
  precision highp float;

  uniform vec2 uWind;    // unit vector the wind blows towards, in world xz
  uniform float uTime;   // seconds, wrapped on the CPU so it never grows large
  uniform float uDetail; // 0 when a streak is too small to resolve, 1 when it is comfortable
  uniform float uRich;   // 1 for both noise layers, 0 for one. A quality step, see scene.js.
  uniform float uPx;     // world units per device pixel; the camera makes this constant on screen
  uniform float uScale;  // REF_VIEW / viewSize, so a streak keeps its size on screen as we zoom
  uniform vec2 uRing;    // arena boundary: centreline radius, half thickness
  uniform vec3 uRingCol;
  uniform vec3 uDeep;
  uniform vec3 uFoam;
  varying vec2 vWorld;

  // Value noise after Inigo Quilez (iquilezles.org/articles/gradientnoise, MIT). Deliberately no
  // sin() in the hash: sin is not bit-specified in GLSL ES, so the popular fract(sin(dot(...)))
  // hash gives visibly different water on different GPUs, and its argument overflows mediump.
  float hash(vec2 p) {
    p = 50.0 * fract(p * 0.3183099 + vec2(0.71, 0.113));
    return -1.0 + 2.0 * fract(p.x * p.y * (p.x + p.y));
  }

  float vnoise(vec2 x) {
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0); // quintic: continuous second derivative
    float a = hash(p + vec2(0.0, 0.0));
    float b = hash(p + vec2(1.0, 0.0));
    float c = hash(p + vec2(0.0, 1.0));
    float d = hash(p + vec2(1.0, 1.0));
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
  }

  // Interleaved gradient noise, Jorge Jimenez, SIGGRAPH 2014. Takes pixel coordinates, not uvs, and
  // has to be highp: at 1080p gl_FragCoord reaches 2000, where a mediump ULP is 1.0 and the hash
  // returns nothing but zero. Two taps summed give a triangular distribution, which is what
  // actually removes banding -- one tap of uniform noise does not.
  float ign(vec2 fc) {
    return fract(52.9829189 * fract(dot(fc, vec2(0.06711056, 0.00583715))));
  }

  void main() {
    // Into the wind frame: x runs downwind, y across it. Scaled by the zoom, so a streak is about
    // the same size on screen whether the camera is on one ship in the build phase or on both of
    // them at arm's length. Purely world-space noise looked right in battle and turned into huge
    // smudges behind the build grid, because zooming in magnifies world-sized features. The old
    // quads did the same thing with a viewSize/24 length scale; this is that rule, kept.
    vec2 d = uWind;
    vec2 q = vec2(dot(vWorld, d), dot(vWorld, vec2(-d.y, d.x))) * uScale;

    float t = uTime * ${DRIFT.toFixed(2)};
    // Two layers at different stretches and speeds. One alone reads as a texture sliding past
    // rather than as water, so the second is worth its four extra hashes -- but it is the first
    // thing to go when the machine cannot keep up. The branch is on a uniform, so every fragment
    // in the draw takes the same side of it and there is no divergence cost.
    float a = vnoise(q * vec2(${ALONG.toFixed(3)}, ${ACROSS.toFixed(3)}) - vec2(t, 0.0));
    float h = a;
    if (uRich > 0.5) {
      float b = vnoise(q * vec2(0.062, 0.44) - vec2(t * 0.6, 0.0) + 31.4);
      h = a * 0.62 + b * 0.38;
    }

    // Only the crests catch light. Thresholding high leaves most of the surface open water and
    // picks out sparse wind-aligned lines, which is what the old quads drew and what keeps the sea
    // behind the ships rather than beside them. The first pass banded the whole range and the
    // result fought the ships for attention -- legibility is the constraint here, not prettiness.
    float crest = smoothstep(0.40, 0.86, h);
    float wash = smoothstep(-0.75, 0.75, h);
    vec3 col = mix(uDeep, uFoam, (crest * 0.13 + wash * 0.03) * uDetail);

    // The arena boundary, drawn here rather than as a mesh. It used to be a 128-segment ring with
    // a transparent material -- a whole blended draw call for a circle the shader can express as a
    // distance from the origin. Folding it in also antialiases it properly: the edge is one pixel
    // wide by construction, where the geometry version had hard tessellated edges.
    float ring = 1.0 - smoothstep(uRing.y, uRing.y + uPx * 1.5, abs(length(vWorld) - uRing.x));
    col = mix(col, uRingCol, ring * 0.22);

    // Dither last, after the colour space conversion three.js appends, or it is quantised away.
    float n = ign(gl_FragCoord.xy) + ign(gl_FragCoord.xy + 5.588238) - 1.0;
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
    gl_FragColor.rgb += n / 255.0;
  }
`;

export function createSea() {
  // One triangle covering the viewport, not two. A quad has a diagonal seam through the middle,
  // which shades a strip of pixels twice and puts a derivative discontinuity across the screen.
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    // ShaderMaterial, not RawShaderMaterial: the raw one is handed neither the #version line nor
    // the colorspace chunk included above, and silently fails to compile.
    uniforms: {
      uMap: { value: new THREE.Vector4(0, 0, 100, -100) },
      uWind: { value: new THREE.Vector2(0, -1) },
      uTime: { value: 0 },
      uDetail: { value: 1 },
      uRich: { value: 1 },
      uPx: { value: 0.2 },
      uScale: { value: 1 },
      uRing: { value: new THREE.Vector2(ARENA_RADIUS - 0.45, 0.45) },
      uRingCol: { value: new THREE.Color(SEA.arenaRing) },
      uDeep: { value: new THREE.Color(SEA.water) },
      uFoam: { value: new THREE.Color(SEA.windStreak) },
    },
    depthTest: false,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'sea';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1; // before everything: this is the background, and it replaces the clear
  return mesh;
}
