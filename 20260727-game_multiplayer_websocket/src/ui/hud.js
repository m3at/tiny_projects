// DOM chrome: top bar, wind dial, battle status panels, enemy schematic.

import { PARTS } from '../data/parts.js';
import { HULLS } from '../data/hulls.js';
import { structureFraction } from '../sim/ship.js';

export const $ = (id) => document.getElementById(id);

export function setVisible(el, on) {
  el.classList.toggle('hidden', !on);
}

export function setRound(roundIndex, hullIndex) {
  $('round-label').textContent = `Round ${roundIndex + 1}`;
  $('hull-label').textContent = HULLS[hullIndex].name;
}

export function setScore(a, b) {
  $('score-p1').textContent = a;
  $('score-p2').textContent = b;
}

export function setTimer(seconds) {
  if (seconds === null) {
    $('timer').textContent = '';
    return;
  }
  const s = Math.max(0, Math.ceil(seconds));
  $('timer').textContent = `${s}`;
  $('timer').classList.toggle('urgent', s <= 8);
}

// Wind blows toward windTo. World -z is up on screen, so angle 0 points up.
export function drawWindDial(windTo) {
  const canvas = $('wind-dial');
  const ctx = canvas.getContext('2d');
  const r = canvas.width / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = 'rgba(233,228,216,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(r, r, r - 3, 0, Math.PI * 2);
  ctx.stroke();

  const dx = Math.sin(windTo);
  const dy = -Math.cos(windTo);
  const len = r - 9;
  ctx.strokeStyle = '#9fd0e8';
  ctx.fillStyle = '#9fd0e8';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(r - dx * len, r - dy * len);
  ctx.lineTo(r + dx * len * 0.55, r + dy * len * 0.55);
  ctx.stroke();
  // Arrowhead at the downwind end.
  const hx = r + dx * len;
  const hy = r + dy * len;
  const px = -dy;
  const py = dx;
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx - dx * 8 + px * 5, hy - dy * 8 + py * 5);
  ctx.lineTo(hx - dx * 8 - px * 5, hy - dy * 8 - py * 5);
  ctx.closePath();
  ctx.fill();
}

const COMPASS = [
  'Northerly',
  'North-easterly',
  'Easterly',
  'South-easterly',
  'Southerly',
  'South-westerly',
  'Westerly',
  'North-westerly',
];

// A sailor names the wind by where it comes from, so report that rather than the vector
// the simulation actually uses.
export function windName(windTo) {
  const from = (windTo + Math.PI) % (Math.PI * 2);
  return COMPASS[Math.round((from / (Math.PI * 2)) * 8) % 8];
}

export function setWindLabel(windTo) {
  drawWindDial(windTo);
  $('wind-label').textContent = windName(windTo);
}

export function updateBattlePanels(battle) {
  for (let i = 0; i < 2; i++) {
    const ship = battle.ships[i];
    const frac = structureFraction(ship);
    const bar = $(`hp-p${i + 1}`);
    bar.style.width = `${Math.max(0, frac * 100).toFixed(1)}%`;
    bar.style.background = frac > 0.55 ? '#7ec98a' : frac > 0.28 ? '#e0c164' : '#e07a64';

    const manned = ship.guns.filter((g) => g.cell.alive && g.manned).length;
    const total = ship.guns.filter((g) => g.cell.alive).length;
    $(`crew-p${i + 1}`).textContent = `crew ${ship.crew}/${ship.crewSupply}`;
    $(`guns-p${i + 1}`).textContent =
      ship.magazines === 0 ? 'no powder' : `guns ${manned}/${total}`;
  }
}

export function setAmmoButtons(player, ammo) {
  document.querySelectorAll(`.ammo-btn[data-player="${player}"]`).forEach((b) => {
    b.classList.toggle('active', b.dataset.ammo === ammo);
  });
}

// Small plan-view schematic of a design, used for "enemy, last seen".
export function drawSchematic(canvas, design, hullIndex) {
  const ctx = canvas.getContext('2d');
  const hull = HULLS[hullIndex];
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!design) return;

  const size = Math.min(w / (hull.width + 1), h / (hull.length + 1));
  const ox = w / 2;
  const oy = h / 2;

  for (const c of hull.cells) {
    const x = ox + c.dx * size - size / 2;
    const y = oy + c.dz * size - size / 2;
    const slot = design.parts[`${c.dx},${c.dz}`];
    if (slot) {
      const part = PARTS[slot.id];
      ctx.fillStyle = `#${part.color.toString(16).padStart(6, '0')}`;
      ctx.fillRect(x, y, size - 1, size - 1);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.font = `700 ${size * 0.62}px ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(part.glyph, x + size / 2, y + size / 2);
    } else {
      // An empty cell is a hole worth noticing.
      ctx.strokeStyle = 'rgba(233,228,216,0.16)';
      ctx.strokeRect(x + 0.5, y + 0.5, size - 2, size - 2);
    }
  }
}

export function showOverlay({ title, body, log, button, onClick }) {
  $('ov-title').textContent = title;
  $('ov-body').textContent = body || '';
  const logEl = $('ov-log');
  logEl.innerHTML = '';
  for (const line of log || []) {
    const div = document.createElement('div');
    div.innerHTML = line;
    logEl.appendChild(div);
  }
  const btn = $('ov-btn');
  btn.textContent = button || 'Continue';
  btn.onclick = onClick;
  setVisible($('overlay'), true);
  btn.focus();
}

export function hideOverlay() {
  setVisible($('overlay'), false);
}
