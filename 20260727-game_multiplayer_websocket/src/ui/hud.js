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

// The overlay's log lines are the only HTML this game builds from data rather than from literals: the
// result screen wants a player's name in bold beside what became of their ship. Names come from
// strangers over a socket, so anything interpolated into one goes through this first. protocol.js
// cleanName strips markup characters as well, and neither is a reason to skip the other.
export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
  });
}

export function setRound(roundIndex, hullIndex) {
  text($('round-label'), `Round ${roundIndex + 1}`);
  text($('hull-label'), HULLS[hullIndex].name);
}

// ---------------------------------------------------------------------------
// the roster: the score row and the battle panels
// ---------------------------------------------------------------------------
//
// Both are built from the roster rather than written into index.html, because the roster is two to
// four seats and is not known until a match starts. A duel keeps exactly the reading it always had --
// name, score, dash, score, name, each name pointing at its own side of the screen -- and three or
// four become a list, because "3 - 1 - 0 - 2" with a dash between every pair is not a score anyone
// can read.

let roster = [];
let mySeats = [];
const seatClass = (seat) => `p${seat + 1}`;

export function setupRoster({ players, mine = [], onAmmo = null, keyFor = null }) {
  roster = players;
  mySeats = mine;
  buildScoreRow();
  buildPanels(onAmmo, keyFor);
}

function buildScoreRow() {
  const el = $('score');
  el.innerHTML = '';
  written.delete(el);
  const duel = roster.length === 2;
  el.classList.toggle('many', !duel);

  roster.forEach((player, i) => {
    if (duel && i === 1) {
      const score = document.createElement('strong');
      score.id = 'score-1';
      el.appendChild(score);
    }
    const tag = document.createElement('span');
    tag.className = `tag ${seatClass(player.seat)}${duel && i === 1 ? ' edge-right' : ''}`;
    if (player.bot) tag.classList.add('bot');
    tag.id = `tag-${player.seat}`;
    tag.textContent = player.name;
    el.appendChild(tag);
    if (duel && i === 0) {
      const score = document.createElement('strong');
      score.id = 'score-0';
      el.appendChild(score);
      const dash = document.createElement('span');
      dash.className = 'dash';
      dash.textContent = '-';
      el.appendChild(dash);
    }
    if (!duel) {
      const score = document.createElement('strong');
      score.id = `score-${player.seat}`;
      el.appendChild(score);
    }
  });
}

// Seats alternate sides: 0 and 2 to the left, 1 and 3 to the right. So a duel is the left-against-
// right picture the game has always drawn, and a four-way is two pairs facing each other.
function buildPanels(onAmmo, keyFor) {
  const cols = [$('panels-left'), $('panels-right')];
  for (const col of cols) col.innerHTML = '';

  for (const player of roster) {
    const seat = player.seat;
    const mine = mySeats.includes(seat);
    const panel = document.createElement('div');
    panel.className = `ship-status framed ${seatClass(seat)}${mine ? ' mine' : ''}`;
    panel.id = `ship-${seat}`;

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = player.name;
    // Marked only when there is somebody else's panel to tell it apart from. On one keyboard every
    // ship is yours, and a "you" on all four says nothing.
    if (mine && mySeats.length === 1) {
      const you = document.createElement('span');
      you.className = 'you';
      you.textContent = 'you';
      name.appendChild(you);
    }
    panel.appendChild(name);

    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.id = `hp-${seat}`;
    bar.appendChild(fill);
    panel.appendChild(bar);

    const stats = document.createElement('div');
    stats.className = 'stats';
    stats.innerHTML = `<span id="crew-${seat}"></span><span id="guns-${seat}"></span>`;
    panel.appendChild(stats);

    // Only your own panel gets buttons. Somebody else's ammunition is not yours to load, and a row
    // of dead controls on three panels reads as a bug.
    if (mine && onAmmo) {
      const ammo = document.createElement('div');
      ammo.className = 'ammo';
      for (const kind of ['round', 'grape']) {
        const btn = document.createElement('button');
        btn.className = `ammo-btn${kind === 'round' ? ' active' : ''}`;
        btn.dataset.player = String(seat);
        btn.dataset.ammo = kind;
        btn.textContent = kind === 'round' ? 'Round shot' : 'Grape';
        btn.onclick = () => onAmmo(seat, kind);
        ammo.appendChild(btn);
      }
      const key = keyFor ? keyFor(seat) : null;
      if (key) {
        const hint = document.createElement('div');
        hint.className = 'key-hint';
        hint.textContent = `key: ${key.toUpperCase()}`;
        ammo.appendChild(hint);
      }
      panel.appendChild(ammo);
    }

    cols[seat % 2].appendChild(panel);
  }
}

export function setScores(scores) {
  for (const player of roster) {
    const el = $(`score-${player.seat}`);
    if (el) text(el, String(scores[player.seat] ?? 0));
  }
}

// A seat whose player has walked away, or one that was never a player. Shown in both places a name
// appears, because a battle against a bot should not look like a battle against a person.
export function setRosterState(players) {
  for (const player of players) {
    const tag = $(`tag-${player.seat}`);
    if (tag) {
      text(tag, player.name);
      toggle(tag, 'away', !player.connected && !player.bot);
    }
    const panel = $(`ship-${player.seat}`);
    if (panel) toggle(panel, 'away', !player.connected && !player.bot);
  }
}

export function setLocked(seat, on) {
  const tag = $(`tag-${seat}`);
  if (tag) toggle(tag, 'locked', on);
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
let lastStructure = [];
let hitFlash = [];

export function resetBattlePanels(count = roster.length) {
  lastStructure = new Array(count).fill(1);
  hitFlash = new Array(count).fill(0);
}

export function updateBattlePanels(battle, dt = 0) {
  for (let i = 0; i < battle.ships.length; i++) {
    const ship = battle.ships[i];
    const panel = $(`ship-${i}`);
    if (!panel) continue;
    const frac = structureFraction(ship);

    if (frac < lastStructure[i] - FLASH_THRESHOLD) hitFlash[i] = FLASH_TIME;
    lastStructure[i] = frac;
    if (hitFlash[i] > 0) hitFlash[i] -= dt;
    toggle(panel, 'hit', hitFlash[i] > 0);
    // In a melee a ship leaves the fight without the battle ending, so her panel has to say so.
    toggle(panel, 'struck', !!ship.out);

    const bar = $(`hp-${i}`);
    style(bar, 'width', `${Math.max(0, frac * 100).toFixed(1)}%`);
    toggle(bar, 'warn', frac <= 0.55 && frac > 0.28);
    toggle(bar, 'critical', frac <= 0.28);

    // The read that matters for grape shot: how many guns still have hands on them.
    let aliveGuns = 0;
    let manned = 0;
    for (const gun of ship.guns) {
      if (!gun.cell.alive) continue;
      aliveGuns++;
      if (gun.manned) manned++;
    }
    text($(`crew-${i}`), ship.out ? 'struck' : `crew ${ship.crew}/${ship.crewSupply}`);
    text($(`guns-${i}`), ship.magazines === 0 ? 'no powder' : `guns ${manned}/${aliveGuns}`);
    toggle(
      $(`guns-${i}`),
      'alert',
      !ship.out && aliveGuns > 0 && (ship.magazines === 0 || manned < aliveGuns),
    );
  }
}

export function setAmmoButtons(seat, ammo) {
  document.querySelectorAll(`.ammo-btn[data-player="${seat}"]`).forEach((b) => {
    b.classList.toggle('active', b.dataset.ammo === ammo);
  });
}

// The wire, for a networked game. Nothing here changes how the game plays; it is here because a
// player who is losing wants to know whether it is them or the connection.
export function setNetBar(line, trouble = false) {
  const el = $('netbar');
  if (!line) {
    setVisible(el, false);
    return;
  }
  setVisible(el, true);
  text(el, line);
  toggle(el, 'trouble', trouble);
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

// `node` is anything an overlay needs beyond prose -- the menu, a lobby roster, a join code -- and
// `alt` is a second button for the choice that is not the obvious one.
export function showOverlay({ title, body, log, node, button, onClick, alt }) {
  $('ov-title').textContent = title;
  $('ov-body').textContent = body || '';
  const logEl = $('ov-log');
  logEl.innerHTML = '';
  for (const line of log || []) {
    const div = document.createElement('div');
    div.innerHTML = line;
    logEl.appendChild(div);
  }
  const extra = $('ov-extra');
  extra.innerHTML = '';
  if (node) extra.appendChild(node);

  const altBtn = $('ov-alt');
  setVisible(altBtn, !!alt);
  if (alt) {
    altBtn.textContent = alt.label;
    altBtn.onclick = alt.onClick;
    altBtn.disabled = !!alt.disabled;
  }

  const btn = $('ov-btn');
  setVisible(btn, button !== null);
  btn.textContent = button || 'Continue';
  btn.onclick = onClick;
  btn.disabled = false;
  setVisible($('overlay'), true);
  // Focus the button unless the overlay put a field in front of it, in which case typing is what
  // the player came here to do.
  const field = node ? node.querySelector('input') : null;
  if (field) field.focus();
  else btn.focus();
}

export function setOverlayButton({ label, disabled }) {
  const btn = $('ov-btn');
  if (label !== undefined) btn.textContent = label;
  if (disabled !== undefined) btn.disabled = disabled;
}

export function hideOverlay() {
  setVisible($('overlay'), false);
}

// The "last seen" column: one schematic per opponent, captioned, because in a four-way the label
// "enemy" does not identify anybody. Sized to fit however many there are.
export function drawIntel(entries, names) {
  const list = $('enemy-list');
  list.innerHTML = '';
  list.classList.toggle('single', entries.length === 1);
  setVisible($('enemy-intel'), entries.length > 0);
  const wide = entries.length === 1;
  for (const entry of entries) {
    const figure = document.createElement('figure');
    const canvas = document.createElement('canvas');
    canvas.width = wide ? 180 : 84;
    canvas.height = wide ? 240 : 112;
    figure.appendChild(canvas);
    const caption = document.createElement('figcaption');
    caption.className = `p${entry.player + 1}`;
    caption.textContent = names[entry.player] ?? `Player ${entry.player + 1}`;
    figure.appendChild(caption);
    list.appendChild(figure);
    drawSchematic(canvas, entry.design, entry.hullIndex);
  }
}
