// Build phase. Click a card, click a cell. Also handles repair, removal and the reroll.

import * as THREE from 'three';
import { PARTS, BUYABLE, repairCost } from '../data/parts.js';
import { ROUNDS, REROLL_COST, OFFER_SIZE } from '../config.js';
import { designStats, designWarnings, placementError } from '../sim/ship.js';
import { $, drawSchematic, setVisible } from './hud.js';
import { attachFillButton, devFill } from '../dev.js';
import * as audio from '../audio/play.js';

// The offer is a set of part *types*, and you may buy as many of each as you can afford.
// Filling 38 cells one card at a time would be tedious, and the interesting luck is in
// which types you are shown, not how many.
function makeOffer(rng, design, hullIndex) {
  const stats = designStats(design, hullIndex);
  // Guarantees that stop a hand being unplayable: something cheap to plug holes with,
  // powder if you have none, and hands if you have none.
  const guaranteed = ['timber'];
  if (stats.magazines === 0) guaranteed.push('magazine');
  if (stats.crewSupply === 0) guaranteed.push('crew');

  const pool = BUYABLE.filter((id) => !guaranteed.includes(id));
  for (let i = pool.length - 1; i > 0; i--) {
    const j = rng.int(0, i);
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const picked = pool.slice(0, Math.max(0, OFFER_SIZE - guaranteed.length));
  const all = [...guaranteed, ...picked];
  return BUYABLE.filter((id) => all.includes(id));
}

export function startBuild({ sceneCtl, view, design, hullIndex, player, roundIndex, scrap, rng, enemy, onLockIn }) {
  let selected = null; // part id, or 'remove'
  let offer = makeOffer(rng, design, hullIndex);
  const placedThisPhase = new Set();
  let timeLeft = ROUNDS[roundIndex].buildTime;
  let done = false;
  let hoverKey = null;

  $('build-who').textContent = `Player ${player + 1}`;
  $('build-who').style.color = player === 0 ? 'var(--p1)' : 'var(--p2)';

  setVisible($('enemy-intel'), !!enemy);
  if (enemy) drawSchematic($('enemy-canvas'), enemy.design, enemy.hullIndex);

  // ---------- rendering the panels ----------

  function renderOffer() {
    const el = $('offer');
    el.innerHTML = '';
    for (const id of offer) {
      const part = PARTS[id];
      const card = document.createElement('div');
      card.className = 'card-part';
      if (selected === id) card.classList.add('selected');
      if (part.cost > scrap) card.classList.add('unaffordable');
      const hex = `#${part.color.toString(16).padStart(6, '0')}`;
      card.innerHTML =
        `<div class="swatch" style="background:${hex}">${part.glyph}</div>` +
        `<div><div class="pname">${part.name}</div><div class="pblurb">${part.blurb}</div></div>` +
        `<div class="pcost">${part.cost}</div>`;
      card.onclick = () => {
        selected = selected === id ? null : id;
        audio.ui('select');
        renderAllWithHint();
      };
      el.appendChild(card);
    }
    $('btn-reroll').disabled = scrap < REROLL_COST;
    $('btn-scrap').classList.toggle('selected', selected === 'remove');

    // Refit repairs the whole ship in one click. Hunting damaged cells one at a time was
    // the same decision wrapped in busywork.
    const damage = damagedParts();
    const total = damage.reduce((s, [, slot]) => s + repairCost(slot.id), 0);
    const refit = $('btn-refit');
    refit.innerHTML = total ? `Refit <span class="muted">${total}</span>` : 'Refit';
    refit.disabled = total === 0 || scrap < repairCost(damage[0][1].id);
    refit.title = total ? `Repair ${damage.length} damaged part(s)` : 'Nothing is damaged';
  }

  function damagedParts() {
    return Object.entries(design.parts)
      .filter(([, s]) => s.hp < PARTS[s.id].hp)
      .sort((a, b) => a[1].hp / PARTS[a[1].id].hp - b[1].hp / PARTS[b[1].id].hp);
  }

  function renderReadout() {
    const s = designStats(design, hullIndex);
    const holes = s.cellsTotal - s.cellsUsed;
    // Open holes lead: shot passes straight through them, which is the rule that decides
    // most battles, and a ratio of filled cells buried it.
    const rows = [
      ['Open holes', `${holes}`, holes > s.cellsTotal * 0.25],
      ['Guns', `${s.gunCount}`, s.gunCount === 0],
      ['Crew', `${s.crewSupply} of ${s.crewNeeded}`, s.crewSupply < s.crewNeeded],
      // The mast count the hull can use, not just the count carried: extra masts do nothing, and
      // that was invisible.
      ['Masts', `${s.masts} of ${s.mastsWanted}`, s.masts === 0],
      ['Powder', `${s.magazines}`, s.magazines === 0],
    ];
    if (s.damaged.length) rows.push(['Damaged parts', `${s.damaged.length}`, true]);
    $('readout').innerHTML = rows
      .map(
        ([k, v, bad]) =>
          `<div class="stat-row${bad ? ' bad' : ''}"><span class="muted">${k}</span><span class="v">${v}</span></div>`,
      )
      .join('');
    $('warnings').innerHTML = designWarnings(design, hullIndex)
      .map((w) => `<div class="warn">${w}</div>`)
      .join('');
  }

  function setHint(text) {
    $('hint').textContent = text;
  }

  // A refused action says so twice: in the hint, and with a knock and two falling notes.
  function deny(text) {
    audio.ui('deny');
    setHint(text);
  }

  function defaultHint() {
    if (selected === 'remove') return 'Click a part to remove it. Parts bought this round refund in full.';
    if (selected) return `Click a hull cell to place a ${PARTS[selected].name.toLowerCase()}.`;
    return 'Pick a part, then click a cell. Spine holds masts, powder, crew. Flanks hold broadsides.';
  }

  // Does not touch the hint: callers own that, otherwise a re-render eats the feedback
  // message that prompted it.
  function renderAll() {
    $('scrap-value').textContent = scrap;
    renderOffer();
    renderReadout();
    view.refresh();
  }

  function renderAllWithHint() {
    renderAll();
    setHint(defaultHint());
  }

  // ---------- interaction ----------

  const canvas = sceneCtl.renderer.domElement;
  const ndc = new THREE.Vector2();

  // getBoundingClientRect flushes style and layout before it can answer, and pick() runs on every
  // pointermove -- so moving the mouse across the hull was forcing a reflow of the whole build
  // panel at mouse-report rate. The canvas is fixed to the viewport, so its rectangle only moves
  // when the window does; read it then, and not in the hot path.
  let rect = canvas.getBoundingClientRect();
  const remeasure = () => {
    rect = canvas.getBoundingClientRect();
  };
  // One AbortController owns every listener this build phase registers, so tearing them all down
  // is a single abort() rather than a list that has to be kept in step with the list above it.
  const listeners = new AbortController();
  const on = { signal: listeners.signal };
  addEventListener('resize', remeasure, on);
  addEventListener('scroll', remeasure, { capture: true, signal: listeners.signal });

  function pick(event) {
    ndc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    sceneCtl.raycaster.setFromCamera(ndc, sceneCtl.camera);
    const hits = sceneCtl.raycaster.intersectObjects(view.pickTargets, false);
    return hits.length ? view.cellKeyForInstance(hits[0].instanceId) : null;
  }

  function onMove(event) {
    if (done) return;
    const key = pick(event);
    if (key === hoverKey) return;
    hoverKey = key;
    const partId = PARTS[selected] ? selected : null;
    view.setGhost(key, partId);
    view.setArcPreview(key, key && !design.parts[key] ? partId : null);
  }

  function act(key) {
    const slot = design.parts[key];

    if (selected === 'remove') {
      if (!slot) return deny('Nothing there to remove.');
      if (PARTS[slot.id].fixed) return deny('The helm stays where it is.');
      audio.ui('place');
      delete design.parts[key];
      if (placedThisPhase.has(key)) {
        scrap += PARTS[slot.id].cost;
        placedThisPhase.delete(key);
        setHint(`Removed, ${PARTS[slot.id].cost} scrap back.`);
      } else {
        setHint('Broken up for nothing. It was already paid for.');
      }
      renderAll();
      return;
    }

    if (!selected) return deny('Pick a part first.');

    const part = PARTS[selected];
    if (part.cost > scrap) return deny(`Not enough scrap for a ${part.name.toLowerCase()}.`);
    const err = placementError(design, hullIndex, ...key.split(',').map(Number), selected);
    if (err) return deny(err);

    audio.ui('place');
    design.parts[key] = { id: selected, hp: part.hp };
    placedThisPhase.add(key);
    scrap -= part.cost;
    renderAllWithHint();
    // Keep the part selected so filling a flank is a row of clicks, not a row of round trips.
    view.setGhost(key, selected);
    view.setArcPreview(null, null);
  }

  function onClick(event) {
    if (done) return;
    const key = pick(event);
    if (key) act(key);
  }

  function onContext(event) {
    if (done) return;
    event.preventDefault();
    const key = pick(event);
    if (!key) return;
    const prev = selected;
    selected = 'remove';
    act(key);
    selected = prev;
    renderAll();
  }

  // Passive: none of these call preventDefault, and saying so lets the browser dispatch them
  // without waiting to find out.
  canvas.addEventListener('pointermove', onMove, { passive: true, signal: listeners.signal });
  canvas.addEventListener('click', onClick, on);
  canvas.addEventListener('contextmenu', onContext, on);

  $('btn-reroll').onclick = () => {
    if (scrap < REROLL_COST) return;
    audio.ui('press');
    scrap -= REROLL_COST;
    offer = makeOffer(rng, design, hullIndex);
    renderAllWithHint();
  };
  $('btn-scrap').onclick = () => {
    audio.ui('press');
    selected = selected === 'remove' ? null : 'remove';
    renderAllWithHint();
    view.setGhost(hoverKey, null);
    view.setArcPreview(null, null);
  };

  // Worst damage first, so a partial purse still buys back the most broken parts.
  $('btn-refit').onclick = () => {
    audio.ui('press');
    let repaired = 0;
    for (const [, slot] of damagedParts()) {
      const cost = repairCost(slot.id);
      if (scrap < cost) break;
      scrap -= cost;
      slot.hp = PARTS[slot.id].hp;
      repaired++;
    }
    setHint(repaired ? `Refitted ${repaired} part(s).` : 'Not enough scrap to repair anything.');
    renderAll();
  };

  function finish() {
    if (done) return;
    done = true;
    listeners.abort();
    view.setGhost(null, null);
    view.setArcPreview(null, null);
    onLockIn(scrap);
  }

  // The commit gets the one rising three-note sound in the game. On the click only: the countdown
  // running out calls finish() too, and confirming there would claim a decision nobody made.
  $('btn-lock').onclick = () => {
    if (!done) audio.ui('confirm');
    finish();
  };

  attachFillButton(document.querySelector('#build-ui .tools'), () => {
    scrap = devFill(design, hullIndex, scrap);
    renderAllWithHint();
  });

  renderAllWithHint();

  return {
    update(dt) {
      if (done) return;
      timeLeft -= dt;
      if (timeLeft <= 0) finish();
    },
    get timeLeft() {
      return timeLeft;
    },
    destroy() {
      done = true;
      listeners.abort();
    },
  };
}
