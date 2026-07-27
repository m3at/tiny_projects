// DOM chrome: top bar, wind dial, battle status panels, enemy schematic.
//
// Everything here is dirty-tracked. These functions run every frame during a battle, and
// writing an unchanged textContent still invalidates layout, so each write is guarded.

import { PARTS } from '../data/parts.js';
import { HULLS } from '../data/hulls.js';
import { structureFraction } from '../sim/ship.js';

export const $ = (id) => document.getElementById(id);

// Last value written, per element and property.
const written = new WeakMap();

function once(el, key, value) {
  let map = written.get(el);
  if (!map) written.set(el, (map = new Map()));
  if (map.get(key) === value) return false;
  map.set(key, value);
  return true;
}

function text(el, value) {
  if (once(el, 'text', value)) el.textContent = value;
}

function style(el, prop, value) {
  if (once(el, `style:${prop}`, value)) el.style[prop] = value;
}

function toggle(el, cls, on) {
  if (once(el, `class:${cls}`, on)) el.classList.toggle(cls, on);
}

export function setVisible(el, on) {
  el.classList.toggle('hidden', !on);
}

export function setRound(roundIndex, hullIndex) {
  text($('round-label'), `Round ${roundIndex + 1}`);
  text($('hull-label'), HULLS[hullIndex].name);
}

export function setScore(a, b) {
  text($('score-p1'), String(a));
  text($('score-p2'), String(b));
}

export function setTimer(seconds) {
  if (seconds === null) {
    text($('timer'), '');
    return;
  }
  const s = Math.max(0, Math.ceil(seconds));
  text($('timer'), String(s));
  toggle($('timer'), 'urgent', s <= 8);
}

// Wind blows toward windTo. World -z is up on screen, so angle 0 points up.
export function drawWindDial(windTo) {
  const canvas = $('wind-dial');
  const ctx = canvas.getContext('2d');
  const r = canvas.width / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // A compass rose rather than a bare circle: eight ticks, north doubled, so the arrow can
  // be read against something.
  ctx.strokeStyle = 'rgba(233,228,216,0.22)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(r, r, r - 3, 0, Math.PI * 2);
  ctx.stroke();
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const inner = r - 3 - (i === 0 ? 7 : i % 2 === 0 ? 4 : 2.5);
    ctx.beginPath();
    ctx.moveTo(r + Math.sin(a) * inner, r - Math.cos(a) * inner);
    ctx.lineTo(r + Math.sin(a) * (r - 3), r - Math.cos(a) * (r - 3));
    ctx.stroke();
  }

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
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx - dx * 8 - dy * 5, hy - dy * 8 + dx * 5);
  ctx.lineTo(hx - dx * 8 + dy * 5, hy - dy * 8 - dx * 5);
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

// A sailor names the wind by where it comes from, so report that rather than the vector the
// simulation actually uses.
export function windName(windTo) {
  const from = (windTo + Math.PI) % (Math.PI * 2);
  return COMPASS[Math.round((from / (Math.PI * 2)) * 8) % 8];
}

export function setWindLabel(windTo) {
  drawWindDial(windTo);
  text($('wind-label'), windName(windTo));
}

// Flash a panel when its ship takes real damage, so it registers even though the bar barely
// moves. The threshold matters: a ship of the line is hit several times a second now, and
// flashing on every graze left the panel permanently lit and meaning nothing. One cell is worth
// about 3% of a hull, so this fires when something aboard actually breaks.
const FLASH_THRESHOLD = 0.012;
const FLASH_TIME = 0.18;
const lastStructure = [1, 1];
const hitFlash = [0, 0];

export function resetBattlePanels() {
  lastStructure[0] = 1;
  lastStructure[1] = 1;
  hitFlash[0] = 0;
  hitFlash[1] = 0;
}

export function updateBattlePanels(battle, dt = 0) {
  for (let i = 0; i < 2; i++) {
    const ship = battle.ships[i];
    const frac = structureFraction(ship);

    if (frac < lastStructure[i] - FLASH_THRESHOLD) hitFlash[i] = FLASH_TIME;
    lastStructure[i] = frac;
    if (hitFlash[i] > 0) hitFlash[i] -= dt;
    toggle($(`ship-p${i + 1}`), 'hit', hitFlash[i] > 0);

    const bar = $(`hp-p${i + 1}`);
    style(bar, 'width', `${Math.max(0, frac * 100).toFixed(1)}%`);
    toggle(bar, 'warn', frac <= 0.55 && frac > 0.28);
    toggle(bar, 'critical', frac <= 0.28);

    // The read that matters for grape shot: how many guns still have hands on them.
    const alive = ship.guns.filter((g) => g.cell.alive);
    const manned = alive.filter((g) => g.manned).length;
    text($(`crew-p${i + 1}`), `crew ${ship.crew}/${ship.crewSupply}`);
    text(
      $(`guns-p${i + 1}`),
      ship.magazines === 0 ? 'no powder' : `guns ${manned}/${alive.length}`,
    );
    toggle(
      $(`guns-p${i + 1}`),
      'alert',
      alive.length > 0 && (ship.magazines === 0 || manned < alive.length),
    );
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
