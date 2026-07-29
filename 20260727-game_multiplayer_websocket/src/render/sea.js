// The sea and the wind, as one opaque full-screen pass.
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
// Water and wind are deliberately different signals. Two slow crossing swells are evaluated at
// the vertices of a modest screen grid and interpolated over the pixels. The fragment shader only
// adds sparse, quick wind crests. This makes the water feel like water instead of a scrolling wind
// texture, while moving the new work from millions of fragments to a few thousand vertices.

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
  uniform float uTime;
  varying vec2 vWorld;
  varying float vSwell;
  void main() {
    vWorld = uMap.xy + position.xy * uMap.zw;
    float a = sin(dot(vWorld, vec2(0.083, 0.031)) + uTime * 0.32);
    float b = sin(dot(vWorld, vec2(-0.047, 0.071)) - uTime * 0.21);
    vSwell = 0.65 * a + 0.35 * b;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const FRAG = /* glsl */ `
  precision highp float;

  uniform vec2 uWind;    // unit vector the wind blows towards, in world xz
  uniform float uTime;   // seconds, wrapped on the CPU so it never grows large
  uniform float uDetail; // 0 when a streak is too small to resolve, 1 when it is comfortable
  uniform float uPx;     // world units per device pixel; the camera makes this constant on screen
  uniform float uScale;  // REF_VIEW / viewSize, so a streak keeps its size on screen as we zoom
  uniform vec2 uRing;    // arena boundary: centreline radius, half thickness
  uniform vec3 uRingCol;
  uniform vec3 uDeep;
  uniform vec3 uSwell;
  uniform vec3 uFoam;
  varying vec2 vWorld;
  varying float vSwell;

  // Deliberately no sin() in the hash: sin is not bit-specified in GLSL ES, so the popular
  // fract(sin(dot(...))) form gives visibly different water on different GPUs, and its argument
  // overflows mediump.
  float hash(vec2 p) {
    p = 50.0 * fract(p * 0.3183099 + vec2(0.71, 0.113));
    return -1.0 + 2.0 * fract(p.x * p.y * (p.x + p.y));
  }

  // One elongated cell is one patch of foam. A random offset per crosswind row breaks up the grid,
  // and soft edges make the cells join as streaks. This costs two hashes rather than value noise's
  // four hashes plus a quintic interpolation, while retaining the only information the field needs
  // to convey: direction and speed.
  float streakField(vec2 x) {
    float row = floor(x.y);
    // Irrational row stepping is enough to avoid aligned starts and costs a multiply plus fract;
    // hashing the row was a second full hash in every fragment for no visible benefit.
    x.x += fract(row * 0.61803398875) * 5.0;
    vec2 p = floor(x);
    vec2 f = fract(x);
    float sx = smoothstep(0.0, 0.12, f.x) * (1.0 - smoothstep(0.58, 1.0, f.x));
    float sy = smoothstep(0.0, 0.18, f.y) * (1.0 - smoothstep(0.82, 1.0, f.y));
    return hash(p) * sx * sy;
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
    // A second octave used to add another full noise evaluation to every pixel. Side-by-side
    // captures showed no legibility gain at play scale, while fill.js priced the sea as most of the
    // frame, so the single broad field is the intentional full-quality rendering now.
    float h = streakField(q * vec2(${ALONG.toFixed(3)}, ${ACROSS.toFixed(3)}) - vec2(t, 0.0));

    // Only the crests catch light. Thresholding high leaves most of the surface open water and
    // picks out sparse wind-aligned lines, which is what the old quads drew and what keeps the sea
    // behind the ships rather than beside them. The first pass banded the whole range and the
    // result fought the ships for attention -- legibility is the constraint here, not prettiness.
    float crest = smoothstep(0.66, 0.93, h);
    vec3 col = mix(uDeep, uSwell, (0.5 + 0.5 * vSwell) * 0.18);
    col = mix(col, uFoam, crest * 0.09 * uDetail);

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
  // A screen-space grid lets the slow swell live in the vertex shader. At 64 by 36 it is smooth at
  // every supported resolution (the wavelengths are hundreds of pixels) and is still only 2,405
  // vertices -- negligible beside the fragment work it avoids.
  const geometry = new THREE.PlaneGeometry(2, 2, 64, 36);

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
      uPx: { value: 0.2 },
      uScale: { value: 1 },
      uRing: { value: new THREE.Vector2(ARENA_RADIUS - 0.45, 0.45) },
      uRingCol: { value: new THREE.Color(SEA.arenaRing) },
      uDeep: { value: new THREE.Color(SEA.water) },
      uSwell: { value: new THREE.Color(SEA.swell) },
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
