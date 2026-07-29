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

// A small authored vocabulary for every transient effect. Keeping the silhouettes in one texture
// means smoke, flashes, timber and water can look unrelated while still sharing one material and
// one instanced draw. The atlas is generated once, is 256x128 RGBA (128 KiB), and has no mipmaps so
// neighbouring cells cannot bleed into one another.
export const FX_TILE = Object.freeze({
  smoke: 0,
  flash: 1,
  splinter: 2,
  splash: 3,
  foam: 4,
  core: 5,
  wake: 6,
  streak: 7,
});

export function fxAtlasTexture() {
  const key = 'fx-atlas';
  if (cache.has(key)) return cache.get(key);

  const cell = 64;
  const canvas = document.createElement('canvas');
  canvas.width = cell * 4;
  canvas.height = cell * 2;
  const ctx = canvas.getContext('2d');

  function inTile(index, draw) {
    ctx.save();
    ctx.translate((index % 4) * cell, Math.floor(index / 4) * cell);
    draw();
    ctx.restore();
  }

  function radial(x, y, r, inner = 0.85) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(255,255,255,${inner})`);
    g.addColorStop(0.56, 'rgba(255,255,255,0.42)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  inTile(FX_TILE.smoke, () => {
    radial(24, 36, 20);
    radial(39, 34, 18, 0.72);
    radial(31, 22, 17, 0.76);
    radial(17, 25, 13, 0.62);
    radial(47, 23, 11, 0.55);
  });

  inTile(FX_TILE.flash, () => {
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 28);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.22, 'rgba(255,255,255,.92)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    for (let i = 0; i < 24; i++) {
      const a = (i * Math.PI) / 12 - Math.PI / 2;
      const r = i % 2 ? 8 : i % 4 ? 19 : 30;
      const x = 32 + Math.cos(a) * r;
      const y = 32 + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  });

  inTile(FX_TILE.splinter, () => {
    ctx.fillStyle = '#fff';
    for (const [x, y, w, lean] of [[17, 47, 5, -5], [29, 50, 4, 2], [41, 46, 5, 6]]) {
      ctx.beginPath();
      ctx.moveTo(x - w, y);
      ctx.lineTo(x + lean, 10 + Math.abs(lean));
      ctx.lineTo(x + w, y);
      ctx.closePath();
      ctx.fill();
    }
  });

  inTile(FX_TILE.splash, () => {
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.ellipse(32, 49, 25, 7, 0, 0, Math.PI * 2);
    ctx.fill();
    for (const [x, top, w] of [[15, 24, 6], [25, 7, 7], [36, 15, 8], [48, 27, 5]]) {
      ctx.beginPath();
      ctx.moveTo(x - w, 49);
      ctx.quadraticCurveTo(x - w / 2, top + 9, x, top);
      ctx.quadraticCurveTo(x + w / 2, top + 9, x + w, 49);
      ctx.closePath();
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(48, 13, 3, 0, Math.PI * 2);
    ctx.arc(12, 16, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  inTile(FX_TILE.foam, () => {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    for (const [a, b] of [[0.12, 1.2], [1.55, 2.85], [3.25, 4.5], [4.9, 6.05]]) {
      ctx.beginPath();
      ctx.ellipse(32, 32, 25, 17, 0, a, b);
      ctx.stroke();
    }
  });

  inTile(FX_TILE.core, () => {
    radial(28, 34, 27, 1);
    radial(43, 25, 15, 0.82);
    radial(20, 18, 13, 0.74);
  });

  inTile(FX_TILE.wake, () => {
    ctx.strokeStyle = '#fff';
    ctx.lineCap = 'round';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(31, 8);
    ctx.quadraticCurveTo(21, 31, 6, 57);
    ctx.moveTo(33, 8);
    ctx.quadraticCurveTo(43, 31, 58, 57);
    ctx.stroke();
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 3;
    for (const y of [24, 35, 46]) {
      ctx.beginPath();
      ctx.moveTo(27, y);
      ctx.lineTo(37, y + 3);
      ctx.stroke();
    }
  });

  inTile(FX_TILE.streak, () => {
    const g = ctx.createLinearGradient(5, 32, 59, 32);
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(0.55, 'rgba(255,255,255,.55)');
    g.addColorStop(1, 'rgba(255,255,255,1)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(4, 29);
    ctx.lineTo(55, 24);
    ctx.lineTo(61, 32);
    ctx.lineTo(55, 40);
    ctx.lineTo(4, 35);
    ctx.closePath();
    ctx.fill();
  });

  const tex = new THREE.CanvasTexture(canvas);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  cache.set(key, tex);
  return tex;
}
