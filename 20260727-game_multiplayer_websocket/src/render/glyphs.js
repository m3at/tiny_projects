// Canvas-drawn glyph textures so placeholder boxes are still readable at a glance.
// Real part meshes will make these redundant, at which point this file goes away.

import * as THREE from 'three';

const cache = new Map();

function luminance(hex) {
  const r = (hex >> 16) & 255;
  const g = (hex >> 8) & 255;
  const b = hex & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

export function glyphTexture(glyph, color) {
  const key = `${glyph}:${color}`;
  if (cache.has(key)) return cache.get(key);

  const size = 96;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = luminance(color) > 0.5 ? 'rgba(12,16,22,0.86)' : 'rgba(240,236,226,0.86)';
  ctx.font = `700 ${size * 0.62}px ui-monospace, Menlo, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(glyph, size / 2, size * 0.54);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  cache.set(key, tex);
  return tex;
}

// Soft radial blob, used for muzzle smoke and explosion flashes.
export function puffTexture() {
  const key = 'puff';
  if (cache.has(key)) return cache.get(key);
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.55)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  cache.set(key, tex);
  return tex;
}
