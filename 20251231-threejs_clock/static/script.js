import * as THREE from 'three';
import { FontLoader } from 'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/jsm/geometries/TextGeometry.js';
import { HDRLoader } from 'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/jsm/loaders/HDRLoader.js';

// ============================================================================
// CONSTANTS
// ============================================================================
const IS_VALIDATION = false;
// const IS_VALIDATION = true;

const CLOCK_RADIUS = 5;
const CLOCK_DEPTH = 0.3;
const BASE_CAMERA_DISTANCE = 15;

// Roman numerals mapping
const ROMAN_NUMERALS = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI'];

// ============================================================================
// CONFIGURATION OPTIONS
// ============================================================================
const HDRI_FILES = IS_VALIDATION ? [
  'hdri/shanghai_bund.hdr',
  'hdri/spiaggia_di_mondello.hdr',
  'hdri/symmetrical_garden_02.hdr',
] : [
  'hdri/studio_small_03.hdr',
  'hdri/venice_sunset.hdr',
  'hdri/industrial_workshop_foundry.hdr',
  'hdri/brown_photostudio_02.hdr',
  'hdri/modern_buildings.hdr',
  'hdri/kloppenheim_02.hdr',
  'hdri/autumn_forest.hdr',
  'hdri/blue_photo_studio.hdr',
  'hdri/courtyard.hdr',
  'hdri/kiara_dawn.hdr',
];

const TEXTURE_FILES = IS_VALIDATION ? {
  metal: 'textures/corrugated_iron.jpg',
  wood: 'textures/kitchen_wood.jpg',
  leather: 'textures/bark.jpg',
  concrete: 'textures/rusty_metal.jpg',
} : {
  metal: 'textures/metal.jpg',
  wood: 'textures/wood.jpg',
  leather: 'textures/leather.jpg',
  concrete: 'textures/concrete.jpg',
};

const FONTS = IS_VALIDATION ? [
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/droid/droid_serif_regular.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/droid/droid_sans_regular.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/droid/droid_sans_mono_regular.typeface.json',
] : [
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/helvetiker_bold.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/helvetiker_regular.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/optimer_bold.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/optimer_regular.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/gentilis_bold.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/gentilis_regular.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/droid/droid_serif_bold.typeface.json',
  'https://cdn.jsdelivr.net/npm/three@0.182.0/examples/fonts/droid/droid_sans_bold.typeface.json',
];

const COLOR_PALETTES = IS_VALIDATION ? [
  { face: 0xfdf6e3, rim: 0x8b4513, hands: 0x5d3a1a, accent: 0xb8860b, bg: 0x2c1810 },
  { face: 0x383838, rim: 0xc9b037, hands: 0xc9b037, accent: 0xffd700, bg: 0x0a0a0a },
  { face: 0xe8e8e8, rim: 0x4682b4, hands: 0x2f4f4f, accent: 0x1e90ff, bg: 0x1a1a2e },
] : [
  { face: 0xf5f5f5, rim: 0x2c3e50, hands: 0x2c3e50, accent: 0xe74c3c, bg: 0x1a1a2e },
  { face: 0xffffff, rim: 0x1a1a1a, hands: 0x1a1a1a, accent: 0xff6b6b, bg: 0xf0f0f0 },
  { face: 0x434d50, rim: 0x00cec9, hands: 0x00cec9, accent: 0xfd79a8, bg: 0x0d0d0d },
  { face: 0xffeaa7, rim: 0xd63031, hands: 0xd63031, accent: 0x00b894, bg: 0x2d3436 },
  { face: 0xdfe6e9, rim: 0x6c5ce7, hands: 0x6c5ce7, accent: 0xfdcb6e, bg: 0x2d3436 },
  { face: 0x55efc4, rim: 0x00b894, hands: 0x2d3436, accent: 0xe17055, bg: 0x0d0d0d },
  { face: 0xfad390, rim: 0xe55039, hands: 0x4a4a4a, accent: 0x4a4a4a, bg: 0x1e3799 },
  { face: 0x303f4a, rim: 0xf5f6fa, hands: 0xf5f6fa, accent: 0xffc048, bg: 0x353b48 },
  { face: 0xecf0f1, rim: 0xc0392b, hands: 0x2c3e50, accent: 0xc0392b, bg: 0x34495e },
  { face: 0x2c3e50, rim: 0xf39c12, hands: 0xecf0f1, accent: 0xe74c3c, bg: 0x1a1a2e },
];

const HAND_STYLES = IS_VALIDATION
  ? ['breguet', 'leaf']
  : ['classic', 'sword', 'baton', 'spade', 'needle', 'dauphine'];
const NUMBER_DISPLAY_MODES = ['all', 'quarters', 'twelve_only', 'even', 'none'];
const NUMBER_STYLES = ['arabic', 'roman'];
const BEZEL_STYLES = ['simple', 'rounded', 'stepped', 'coin_edge', 'fluted'];

// ============================================================================
// STATE
// ============================================================================
let appearance = getDefaultAppearance();
let currentTime = { hours: 0, minutes: 0, seconds: 0 };

// ============================================================================
// THREE.JS SETUP
// ============================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(appearance.backgroundColor);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, BASE_CAMERA_DISTANCE);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('container').appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambientLight);

const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
fillLight.position.set(-5, -2, 5);
scene.add(fillLight);

// Sun light (casts shadows)
const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
sunLight.shadow.camera.near = 1;
sunLight.shadow.camera.far = 50;
sunLight.shadow.camera.left = -10;
sunLight.shadow.camera.right = 10;
sunLight.shadow.camera.top = 10;
sunLight.shadow.camera.bottom = -10;
sunLight.shadow.bias = -0.001;
scene.add(sunLight);

// Clock group
let clockGroup = new THREE.Group();
scene.add(clockGroup);

// Background sphere for HDRI
let backgroundSphere = null;

// References
let hourHand = null, minuteHand = null, secondHand = null;

// ============================================================================
// ASSET LOADING
// ============================================================================
const loadedFonts = {};
const loadedTextures = {};
const loadedHDRIs = {};

const textureLoader = new THREE.TextureLoader();
const hdrLoader = new HDRLoader();
const fontLoader = new FontLoader();

async function loadAllAssets() {
  const fontPromises = FONTS.map((url, i) =>
    new Promise((resolve) => {
      fontLoader.load(url, (font) => {
        loadedFonts[i] = font;
        resolve();
      }, undefined, () => resolve());
    })
  );

  const texturePromises = Object.entries(TEXTURE_FILES).map(([name, url]) =>
    new Promise((resolve) => {
      textureLoader.load(url, (texture) => {
        texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
        texture.colorSpace = THREE.SRGBColorSpace;
        loadedTextures[name] = texture;
        resolve();
      }, undefined, () => resolve());
    })
  );

  const hdriPromises = HDRI_FILES.map((url, i) =>
    new Promise((resolve) => {
      hdrLoader.load(url, (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping;
        loadedHDRIs[i] = texture;
        resolve();
      }, undefined, () => resolve());
    })
  );

  await Promise.all([...fontPromises, ...texturePromises, ...hdriPromises]);
}

// ============================================================================
// APPEARANCE HELPERS
// ============================================================================
function getDefaultAppearance() {
  return {
    // Colors
    faceColor: 0xf5f5f5,
    rimColor: 0x2c3e50,
    hourHandColor: 0x2c3e50,
    minuteHandColor: 0x34495e,
    secondHandColor: 0xe74c3c,
    markerColor: 0x2c3e50,
    numberColor: 0x2c3e50,
    backgroundColor: 0x1a1a2e,
    // Structure
    hasSecondHand: true,
    hasMinuteMarkers: true,
    rimThickness: 0.15,
    // Bezel
    bezelStyle: 'simple',
    bezelDepth: 0.2,
    // Hand style
    handStyle: 'breguet',
    hourHandLength: 2.2,
    minuteHandLength: 3.5,
    secondHandLength: 4.0,
    hourHandWidth: 0.3,
    minuteHandWidth: 0.2,
    secondHandWidth: 0.06,
    // Numbers
    numberDisplayMode: 'all',
    numberStyle: 'arabic',
    numberSize: 0.5,
    numberDistance: 1.1,
    fontIndex: 0,
    // Materials
    faceTexture: null,
    rimTexture: null,
    useMetallic: false,
    metallicRoughness: 0.3,
    metalness: 0.9,
    // Environment
    hdriIndex: null,
    useEnvironment: false,
    useHdriBackground: false,
    hdriBackgroundRotation: 0,
    hdriBackgroundBlur: 0.3,
    // Sun (for shadows)
    sunElevation: 45,  // degrees above horizon (5-90)
    sunAzimuth: 0,     // degrees around clock
    sunColor: 0xffffff,
    sunDistance: 25,
  };
}

function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function updateSun() {
  // Convert elevation and azimuth to position
  // Sun is "behind" the clock (negative Z) to cast shadows forward
  const elevationRad = appearance.sunElevation * (Math.PI / 180);
  const azimuthRad = appearance.sunAzimuth * (Math.PI / 180);
  const d = appearance.sunDistance;

  // At elevation 90, sun is directly above (y = d)
  // At elevation 0, sun is at horizon (y = 0)
  // Azimuth rotates around the Y axis in the XZ plane
  const y = d * Math.sin(elevationRad);
  const horizontalDist = d * Math.cos(elevationRad);
  const x = horizontalDist * Math.sin(azimuthRad);
  const z = -horizontalDist * Math.cos(azimuthRad); // negative Z = behind clock

  sunLight.position.set(x, y, z);
  sunLight.color.setHex(appearance.sunColor);

}

function randomizeSunColor() {
  // Colors from white to yellow to light orange
  const sunColors = [
    0xffffff,  // white
    0xfffef0,  // warm white
    0xfffacd,  // lemon chiffon
    0xffefd5,  // papaya whip
    0xffe4b5,  // moccasin
    0xffd699,  // light orange
  ];
  return randomChoice(sunColors);
}

// ============================================================================
// HAND CREATION
// ============================================================================
function createHandShape(style, length, width) {
  const shape = new THREE.Shape();

  switch (style) {
    case 'sword':
      shape.moveTo(0, -0.2);
      shape.lineTo(width / 2, length * 0.15);
      shape.lineTo(width / 4, length * 0.8);
      shape.lineTo(0, length);
      shape.lineTo(-width / 4, length * 0.8);
      shape.lineTo(-width / 2, length * 0.15);
      shape.closePath();
      break;

    case 'baton':
      const r = width / 2;
      shape.moveTo(-width / 2, 0);
      shape.lineTo(-width / 2, length - r);
      shape.quadraticCurveTo(-width / 2, length, 0, length);
      shape.quadraticCurveTo(width / 2, length, width / 2, length - r);
      shape.lineTo(width / 2, 0);
      shape.quadraticCurveTo(width / 2, -r, 0, -r);
      shape.quadraticCurveTo(-width / 2, -r, -width / 2, 0);
      break;

    case 'spade':
      shape.moveTo(0, -0.3);
      shape.lineTo(width / 2, 0);
      shape.lineTo(width / 3, length * 0.7);
      shape.lineTo(width / 2, length * 0.75);
      shape.lineTo(0, length);
      shape.lineTo(-width / 2, length * 0.75);
      shape.lineTo(-width / 3, length * 0.7);
      shape.lineTo(-width / 2, 0);
      shape.closePath();
      break;

    case 'needle':
      shape.moveTo(0, -0.15);
      shape.lineTo(width / 2, 0);
      shape.lineTo(width / 6, length * 0.9);
      shape.lineTo(0, length);
      shape.lineTo(-width / 6, length * 0.9);
      shape.lineTo(-width / 2, 0);
      shape.closePath();
      break;

    case 'dauphine':
      // Elegant tapered with hollow center appearance
      shape.moveTo(-width / 2, -0.2);
      shape.lineTo(width / 2, -0.2);
      shape.lineTo(width / 2, length * 0.1);
      shape.lineTo(width / 4, length * 0.95);
      shape.lineTo(0, length);
      shape.lineTo(-width / 4, length * 0.95);
      shape.lineTo(-width / 2, length * 0.1);
      shape.closePath();
      break;

    case 'breguet':
      // Moon-tip style - tapered hand with circular moon near tip
      const circleY = length * 0.72;
      const circleR = width * 0.55;

      shape.moveTo(-width / 2, -0.2);
      shape.lineTo(width / 2, -0.2);
      // Taper up to the moon
      shape.lineTo(width / 4, circleY - circleR);
      // Draw circular moon (centered on hand axis)
      shape.absarc(0, circleY, circleR, -Math.PI / 2, Math.PI * 1.5, false);
      // Continue down the other side
      shape.lineTo(-width / 4, circleY - circleR);
      shape.closePath();
      break;

    case 'leaf':
      // Leaf-shaped hand with curved edges
      shape.moveTo(0, -0.2);
      shape.quadraticCurveTo(width * 0.6, length * 0.2, width * 0.5, length * 0.5);
      shape.quadraticCurveTo(width * 0.3, length * 0.8, 0, length);
      shape.quadraticCurveTo(-width * 0.3, length * 0.8, -width * 0.5, length * 0.5);
      shape.quadraticCurveTo(-width * 0.6, length * 0.2, 0, -0.2);
      break;

    case 'classic':
    default:
      shape.moveTo(-width / 2, -0.3);
      shape.lineTo(width / 2, -0.3);
      shape.lineTo(width / 3, length);
      shape.lineTo(-width / 3, length);
      shape.closePath();
      break;
  }

  return shape;
}

function createHand(style, length, width, depth, color) {
  const shape = createHandShape(style, length, width);
  const extrudeSettings = { depth, bevelEnabled: false };
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);

  let material;
  if (appearance.useMetallic) {
    material = new THREE.MeshStandardMaterial({
      color,
      metalness: appearance.metalness,
      roughness: appearance.metallicRoughness,
      envMapIntensity: 1.5,
    });
  } else {
    material = new THREE.MeshPhongMaterial({ color, shininess: 80 });
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  return mesh;
}

// ============================================================================
// BEZEL CREATION
// ============================================================================
function createBezel() {
  const style = appearance.bezelStyle;
  const thickness = appearance.rimThickness;
  const depth = appearance.bezelDepth;

  let geometry;
  let material;

  const baseMaterial = () => {
    if (appearance.rimTexture && loadedTextures[appearance.rimTexture]) {
      const tex = loadedTextures[appearance.rimTexture].clone();
      tex.repeat.set(8, 1);
      return new THREE.MeshStandardMaterial({
        map: tex,
        color: appearance.rimColor,
        metalness: appearance.metalness,
        roughness: appearance.metallicRoughness,
        envMapIntensity: 2.0,
      });
    } else if (appearance.useMetallic) {
      return new THREE.MeshStandardMaterial({
        color: appearance.rimColor,
        metalness: appearance.metalness,
        roughness: appearance.metallicRoughness,
        envMapIntensity: 2.0,
      });
    } else {
      return new THREE.MeshPhongMaterial({ color: appearance.rimColor, shininess: 100 });
    }
  };

  switch (style) {
    case 'rounded':
      // Rounded/domed bezel
      const tubeRadius = thickness * 1.2;
      geometry = new THREE.TorusGeometry(CLOCK_RADIUS, tubeRadius, 24, 64);
      material = baseMaterial();
      break;

    case 'stepped':
      // Two-tier stepped bezel
      const group = new THREE.Group();
      const inner = new THREE.Mesh(
        new THREE.TorusGeometry(CLOCK_RADIUS - thickness * 0.3, thickness * 0.6, 16, 64),
        baseMaterial()
      );
      inner.position.z = CLOCK_DEPTH / 2 + depth * 0.3;
      inner.castShadow = true;
      group.add(inner);

      const outer = new THREE.Mesh(
        new THREE.TorusGeometry(CLOCK_RADIUS + thickness * 0.2, thickness * 0.8, 16, 64),
        baseMaterial()
      );
      outer.position.z = CLOCK_DEPTH / 2;
      outer.castShadow = true;
      group.add(outer);
      return group;

    case 'coin_edge':
      // Knurled/coin edge bezel
      const coinGroup = new THREE.Group();
      const mainRing = new THREE.Mesh(
        new THREE.TorusGeometry(CLOCK_RADIUS, thickness, 16, 64),
        baseMaterial()
      );
      mainRing.position.z = CLOCK_DEPTH / 2;
      mainRing.castShadow = true;
      coinGroup.add(mainRing);

      // Add notches
      const notchCount = 120;
      for (let i = 0; i < notchCount; i++) {
        const angle = (i / notchCount) * Math.PI * 2;
        const notch = new THREE.Mesh(
          new THREE.BoxGeometry(0.03, thickness * 2.2, depth * 0.8),
          baseMaterial()
        );
        notch.position.x = Math.cos(angle) * CLOCK_RADIUS;
        notch.position.y = Math.sin(angle) * CLOCK_RADIUS;
        notch.position.z = CLOCK_DEPTH / 2;
        notch.rotation.z = angle;
        notch.castShadow = true;
        coinGroup.add(notch);
      }
      return coinGroup;

    case 'fluted':
      // Fluted bezel (like Rolex)
      const flutedGroup = new THREE.Group();
      const baseRing = new THREE.Mesh(
        new THREE.TorusGeometry(CLOCK_RADIUS, thickness * 0.8, 16, 64),
        baseMaterial()
      );
      baseRing.position.z = CLOCK_DEPTH / 2;
      baseRing.castShadow = true;
      flutedGroup.add(baseRing);

      const fluteCount = 40;
      for (let i = 0; i < fluteCount; i++) {
        const angle = (i / fluteCount) * Math.PI * 2;
        const flute = new THREE.Mesh(
          new THREE.CylinderGeometry(thickness * 0.3, thickness * 0.3, depth * 1.5, 8),
          baseMaterial()
        );
        flute.position.x = Math.cos(angle) * CLOCK_RADIUS;
        flute.position.y = Math.sin(angle) * CLOCK_RADIUS;
        flute.position.z = CLOCK_DEPTH / 2;
        flute.rotation.x = Math.PI / 2;
        flute.castShadow = true;
        flutedGroup.add(flute);
      }
      return flutedGroup;

    case 'simple':
    default:
      geometry = new THREE.TorusGeometry(CLOCK_RADIUS, thickness, 16, 64);
      material = baseMaterial();
      break;
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.z = CLOCK_DEPTH / 2;
  mesh.castShadow = true;
  return mesh;
}

// ============================================================================
// CLOCK BUILDING
// ============================================================================
function buildClock() {
  // Clear existing
  scene.remove(clockGroup);
  clockGroup = new THREE.Group();
  scene.add(clockGroup);

  // Update environment
  if (appearance.useEnvironment && appearance.hdriIndex !== null && loadedHDRIs[appearance.hdriIndex]) {
    scene.environment = loadedHDRIs[appearance.hdriIndex];
  } else {
    scene.environment = null;
  }

  // Update background and sun
  updateBackground();
  updateSun();

  buildClockFace();
  const bezel = createBezel();
  clockGroup.add(bezel);
  buildMarkers();
  buildNumbers();
  buildCenterCap();
  buildHands();

  setTime(currentTime.hours, currentTime.minutes, currentTime.seconds);
}

function updateBackground() {
  // Remove existing background sphere
  if (backgroundSphere) {
    scene.remove(backgroundSphere);
    backgroundSphere = null;
  }

  if (appearance.useHdriBackground && appearance.hdriIndex !== null && loadedHDRIs[appearance.hdriIndex]) {
    // Use HDRI directly as scene background
    scene.background = loadedHDRIs[appearance.hdriIndex];
    scene.backgroundRotation = new THREE.Euler(0, appearance.hdriBackgroundRotation, 0);
    scene.backgroundBlurriness = appearance.hdriBackgroundBlur;
  } else {
    scene.background = new THREE.Color(appearance.backgroundColor);
    scene.backgroundRotation = new THREE.Euler(0, 0, 0);
    scene.backgroundBlurriness = 0;
  }
}

function buildClockFace() {
  const geometry = new THREE.CylinderGeometry(CLOCK_RADIUS - 0.1, CLOCK_RADIUS - 0.1, CLOCK_DEPTH, 64);

  let material;
  if (appearance.faceTexture && loadedTextures[appearance.faceTexture]) {
    const tex = loadedTextures[appearance.faceTexture].clone();
    tex.repeat.set(2, 2);
    material = new THREE.MeshStandardMaterial({
      map: tex,
      color: appearance.faceColor,
      roughness: 0.6,
      metalness: 0.1,
    });
  } else if (appearance.useMetallic) {
    material = new THREE.MeshStandardMaterial({
      color: appearance.faceColor,
      metalness: 0.15,
      roughness: 0.5,
      envMapIntensity: 0.5,
    });
  } else {
    material = new THREE.MeshPhongMaterial({ color: appearance.faceColor, shininess: 30 });
  }

  const face = new THREE.Mesh(geometry, material);
  face.rotation.x = Math.PI / 2;
  face.receiveShadow = true;
  clockGroup.add(face);
}

function buildMarkers() {
  const markerMaterial = () => appearance.useMetallic
    ? new THREE.MeshStandardMaterial({
        color: appearance.markerColor,
        metalness: appearance.metalness * 0.8,
        roughness: appearance.metallicRoughness + 0.1,
        envMapIntensity: 1.2,
      })
    : new THREE.MeshPhongMaterial({ color: appearance.markerColor, shininess: 60 });

  // Hour markers
  for (let i = 0; i < 12; i++) {
    const angle = Math.PI / 2 - (i / 12) * Math.PI * 2;
    const isMainHour = i % 3 === 0;

    const markerLength = isMainHour ? 0.4 : 0.25;
    const markerWidth = isMainHour ? 0.1 : 0.05;

    const geometry = new THREE.BoxGeometry(markerLength, markerWidth, 0.05);
    const marker = new THREE.Mesh(geometry, markerMaterial());

    const distance = CLOCK_RADIUS - 0.5;
    marker.position.set(
      Math.cos(angle) * distance,
      Math.sin(angle) * distance,
      CLOCK_DEPTH / 2 + 0.03
    );
    marker.rotation.z = angle;
    marker.castShadow = true;
    clockGroup.add(marker);
  }

  // Minute markers
  if (appearance.hasMinuteMarkers) {
    for (let i = 0; i < 60; i++) {
      if (i % 5 === 0) continue;

      const angle = Math.PI / 2 - (i / 60) * Math.PI * 2;
      const geometry = new THREE.CircleGeometry(0.04, 8);
      const material = new THREE.MeshPhongMaterial({ color: 0x7f8c8d });
      const marker = new THREE.Mesh(geometry, material);

      const distance = CLOCK_RADIUS - 0.4;
      marker.position.set(
        Math.cos(angle) * distance,
        Math.sin(angle) * distance,
        CLOCK_DEPTH / 2 + 0.02
      );
      clockGroup.add(marker);
    }
  }
}

function shouldShowNumber(i) {
  const hourNum = i === 0 ? 12 : i;
  switch (appearance.numberDisplayMode) {
    case 'quarters': return hourNum % 3 === 0;
    case 'twelve_only': return hourNum === 12;
    case 'even': return hourNum % 2 === 0;
    case 'none': return false;
    case 'all':
    default: return true;
  }
}

function getNumberText(i) {
  const hourNum = i === 0 ? 12 : i;
  if (appearance.numberStyle === 'roman') {
    return ROMAN_NUMERALS[i];
  }
  return String(hourNum);
}

function buildNumbers() {
  const font = loadedFonts[appearance.fontIndex] || loadedFonts[0];
  if (!font) return;

  const numberMaterial = () => appearance.useMetallic
    ? new THREE.MeshStandardMaterial({
        color: appearance.numberColor,
        metalness: 0.6,
        roughness: 0.3,
        envMapIntensity: 1.0,
      })
    : new THREE.MeshPhongMaterial({ color: appearance.numberColor, shininess: 50 });

  for (let i = 0; i < 12; i++) {
    if (!shouldShowNumber(i)) continue;

    const angle = Math.PI / 2 - (i / 12) * Math.PI * 2;
    const text = getNumberText(i);

    // Adjust size for roman numerals (they're wider)
    const size = appearance.numberStyle === 'roman'
      ? appearance.numberSize * 0.7
      : appearance.numberSize;

    const textGeometry = new TextGeometry(text, {
      font,
      size,
      depth: 0.06,
    });
    textGeometry.computeBoundingBox();
    const bb = textGeometry.boundingBox;
    const textWidth = bb.max.x - bb.min.x;
    const textHeight = bb.max.y - bb.min.y;

    const mesh = new THREE.Mesh(textGeometry, numberMaterial());

    const distance = CLOCK_RADIUS - appearance.numberDistance;
    mesh.position.set(
      Math.cos(angle) * distance - textWidth / 2,
      Math.sin(angle) * distance - textHeight / 2,
      CLOCK_DEPTH / 2 + 0.02
    );
    mesh.castShadow = true;
    clockGroup.add(mesh);
  }
}

function buildCenterCap() {
  const geometry = new THREE.CylinderGeometry(0.22, 0.22, 0.18, 32);
  const material = appearance.useMetallic
    ? new THREE.MeshStandardMaterial({
        color: appearance.rimColor,
        metalness: appearance.metalness,
        roughness: appearance.metallicRoughness * 0.5,
        envMapIntensity: 2.5,
      })
    : new THREE.MeshPhongMaterial({ color: appearance.rimColor, shininess: 120 });
  const cap = new THREE.Mesh(geometry, material);
  cap.rotation.x = Math.PI / 2;
  cap.position.z = CLOCK_DEPTH / 2 + 0.18;
  cap.castShadow = true;
  clockGroup.add(cap);
}

function buildHands() {
  const style = appearance.handStyle;

  hourHand = createHand(style, appearance.hourHandLength, appearance.hourHandWidth, 0.08, appearance.hourHandColor);
  hourHand.position.z = CLOCK_DEPTH / 2 + 0.05;
  clockGroup.add(hourHand);

  minuteHand = createHand(style, appearance.minuteHandLength, appearance.minuteHandWidth, 0.06, appearance.minuteHandColor);
  minuteHand.position.z = CLOCK_DEPTH / 2 + 0.12;
  clockGroup.add(minuteHand);

  if (appearance.hasSecondHand) {
    secondHand = createHand('needle', appearance.secondHandLength, appearance.secondHandWidth, 0.04, appearance.secondHandColor);
    secondHand.position.z = CLOCK_DEPTH / 2 + 0.18;
    clockGroup.add(secondHand);
  } else {
    secondHand = null;
  }
}

// ============================================================================
// TIME MANAGEMENT
// ============================================================================
function setTime(hours, minutes, seconds) {
  currentTime = { hours, minutes, seconds };

  const hourFraction = (hours % 12 + minutes / 60 + seconds / 3600) / 12;
  const minuteFraction = (minutes + seconds / 60) / 60;
  const secondFraction = seconds / 60;

  if (hourHand) hourHand.rotation.z = -hourFraction * Math.PI * 2;
  if (minuteHand) minuteHand.rotation.z = -minuteFraction * Math.PI * 2;
  if (secondHand) secondHand.rotation.z = -secondFraction * Math.PI * 2;
}

function randomizeTime() {
  const hours = Math.floor(Math.random() * 12);
  const minutes = Math.floor(Math.random() * 60);
  const seconds = Math.floor(Math.random() * 60);
  setTime(hours, minutes, seconds);
}

// ============================================================================
// CAMERA MANAGEMENT
// ============================================================================
function randomizeCamera() {
  const angleVariation = 45 * (Math.PI / 180);
  const xAngle = (Math.random() - 0.5) * 2 * angleVariation;
  const yAngle = (Math.random() - 0.5) * 2 * angleVariation;

  const zoomFactor = randomInRange(0.8, 1.8);
  const distance = BASE_CAMERA_DISTANCE * zoomFactor;

  camera.position.x = distance * Math.sin(yAngle);
  camera.position.y = distance * Math.sin(xAngle);
  camera.position.z = distance * Math.cos(xAngle) * Math.cos(yAngle);

  camera.fov = randomInRange(35, 70);
  camera.updateProjectionMatrix();

  camera.lookAt(0, 0, 0);
}

// ============================================================================
// APPEARANCE RANDOMIZATION
// ============================================================================
function randomizeBackground() {
  const hdriCount = Object.keys(loadedHDRIs).length;
  if (hdriCount === 0) return;

  // Environment for reflections (70% chance)
  appearance.useEnvironment = Math.random() > 0.3;
  appearance.hdriIndex = Math.floor(Math.random() * hdriCount);

  // HDRI Background (90% chance)
  appearance.useHdriBackground = Math.random() > 0.1;
  appearance.hdriBackgroundRotation = Math.random() * Math.PI * 2;
  appearance.hdriBackgroundBlur = randomInRange(0, 0.15);
}

function randomizeStyle() {
  const palette = randomChoice(COLOR_PALETTES);

  // Colors
  appearance.faceColor = palette.face;
  appearance.rimColor = palette.rim;
  appearance.hourHandColor = palette.hands;
  appearance.minuteHandColor = palette.hands;
  appearance.secondHandColor = palette.accent;
  appearance.markerColor = palette.hands;
  appearance.numberColor = palette.hands;
  appearance.backgroundColor = palette.bg;

  // Hand style and dimensions
  appearance.handStyle = randomChoice(HAND_STYLES);
  appearance.hourHandLength = randomInRange(1.8, 2.5);
  appearance.minuteHandLength = randomInRange(appearance.hourHandLength + 0.8, 3.8);
  appearance.secondHandLength = randomInRange(appearance.minuteHandLength + 0.2, 4.2);
  appearance.hourHandWidth = randomInRange(0.25, 0.45);
  appearance.minuteHandWidth = randomInRange(0.12, appearance.hourHandWidth - 0.05);
  appearance.secondHandWidth = randomInRange(0.04, 0.1);

  // Bezel
  appearance.bezelStyle = randomChoice(BEZEL_STYLES);
  appearance.bezelDepth = randomInRange(0.15, 0.35);
  appearance.rimThickness = randomInRange(0.1, 0.25);

  // Structure
  appearance.hasSecondHand = Math.random() > 0.15;
  appearance.hasMinuteMarkers = Math.random() > 0.35;

  // Numbers
  appearance.numberDisplayMode = randomChoice(NUMBER_DISPLAY_MODES);
  appearance.numberStyle = Math.random() > 0.65 ? 'roman' : 'arabic';
  appearance.numberSize = randomInRange(0.35, 0.65);
  appearance.numberDistance = randomInRange(0.9, 1.4);
  appearance.fontIndex = Math.floor(Math.random() * Object.keys(loadedFonts).length);

  // Materials - increase metallic chance and reflectivity
  appearance.useMetallic = Math.random() > 0.25;
  appearance.metallicRoughness = randomInRange(0.1, 0.4);
  appearance.metalness = randomInRange(0.8, 1.0);

  // Textures (25% chance for face, 35% for rim)
  const faceTextures = ['wood', 'leather', 'concrete', 'metal'];
  appearance.faceTexture = Math.random() > 0.75 ? randomChoice(faceTextures) : null;
  appearance.rimTexture = Math.random() > 0.65 ? 'metal' : null;

  // Sun position and color
  appearance.sunElevation = randomInRange(10, 75);
  appearance.sunAzimuth = randomInRange(-60, 60);
  appearance.sunColor = randomizeSunColor();
}

function randomizeAppearance() {
  randomizeBackground();
  randomizeStyle();
  scene.background = new THREE.Color(appearance.backgroundColor);
  buildClock();
}

// ============================================================================
// RENDER
// ============================================================================
function render() {
  renderer.render(scene, camera);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
document.getElementById('randomTime').addEventListener('click', () => {
  randomizeTime();
  render();
});

document.getElementById('randomCamera').addEventListener('click', () => {
  randomizeCamera();
  render();
});

document.getElementById('randomAppearance').addEventListener('click', () => {
  randomizeAppearance();
  render();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  render();
});

// ============================================================================
// API FOR DATASET GENERATION
// ============================================================================
function randomizeAll(includeBackground = true) {
  if (includeBackground) {
    randomizeBackground();
  }
  randomizeStyle();
  scene.background = new THREE.Color(appearance.backgroundColor);
  buildClock();
  randomizeCamera();
  randomizeTime();
  render();
}

function getCurrentTime() {
  return { ...currentTime };
}

// Expose API on window for external access
window.clockAPI = {
  randomizeAll,
  randomizeTime,
  randomizeCamera,
  randomizeAppearance,
  randomizeBackground,
  randomizeStyle,
  setTime,
  getCurrentTime,
  render,
};

// ============================================================================
// INITIALIZATION
// ============================================================================
async function init() {
  await loadAllAssets();

  buildClock();

  const now = new Date();
  setTime(now.getHours(), now.getMinutes(), now.getSeconds());

  render();
}

init();
