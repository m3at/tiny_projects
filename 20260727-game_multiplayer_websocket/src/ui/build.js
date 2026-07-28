// The build phase, as an interface. Click a card, click a cell.
//
// It no longer changes anything itself. Every action is a command to the authority -- place, remove,
// refit, reroll, lock in -- which applies the rules in shipyard.js and answers with the purse. The
// client applies the same rules to the same command first so the deck responds to the click rather
// than to the round trip, and if the authority disagrees it sends the design back and this redraws.
// So there is one implementation of what a part costs and where it may go, and it is not this file.
//
// The countdown is the authority's too. A clock the player can see and a deadline the server
// enforces have to be the same clock, or the last second of a build phase is a lie.

import * as THREE from 'three';
import { PARTS } from '../data/parts.js';
import { REROLL_COST } from '../config.js';
import { designStats, designWarnings } from '../sim/ship.js';
import { $, drawIntel } from './hud.js';
import { attachFillButton, devFillCommands } from '../dev.js';
import * as audio from '../audio/play.js';

export function startBuild({ sceneCtl, view, client, seat, names }) {
  let selected = null; // part id, or 'remove'
  let done = false;
  let hoverKey = null;

  const yard = () => client.yard;
  const design = () => client.yard.design;
  const hullIndex = client.state.hullIndex;

  $('build-who').textContent = names[seat] ?? `Player ${seat + 1}`;
  $('build-who').className = `p${seat + 1}`;

  drawIntel(client.state.intel, names);

  // ---------- rendering the panels ----------

  function renderOffer() {
    const el = $('offer');
    el.innerHTML = '';
    const scrap = yard().scrap;
    for (const id of client.state.offer) {
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
    const total = yard().refitCost();
    const damage = yard().damaged();
    const refit = $('btn-refit');
    refit.innerHTML = total ? `Refit <span class="muted">${total}</span>` : 'Refit';
    refit.disabled = total === 0;
    refit.title = total ? `Repair ${damage.length} damaged part(s)` : 'Nothing is damaged';
  }

  function renderReadout() {
    const s = designStats(design(), hullIndex);
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
    $('warnings').innerHTML = designWarnings(design(), hullIndex)
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
    $('scrap-value').textContent = yard().scrap;
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
    view.setArcPreview(key, key && !design().parts[key] ? partId : null);
  }

  function act(key) {
    if (selected === 'remove') {
      const res = client.remove(key);
      if (!res.ok) return deny(res.why);
      audio.ui('place');
      setHint(
        res.refund > 0
          ? `Removed, ${res.refund} scrap back.`
          : 'Broken up for nothing. It was already paid for.',
      );
      renderAll();
      return;
    }

    if (!selected) return deny('Pick a part first.');

    const res = client.place(key, selected);
    if (!res.ok) return deny(res.why);

    audio.ui('place');
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

  // Every handler below states its own feedback *before* sending the command, never after.
  //
  // In a local game the authority is in this page, so a command runs to completion inside the call:
  // by the time client.lock() returns, the next captain's build phase has already been set up. Code
  // written in the obvious order -- send, then update the panel -- therefore lands on the *next*
  // phase's panel. That cost an afternoon: autoplay locked in the first captain and then disabled the
  // second captain's lock button, so every hot-seat build phase ran to its full forty seconds and the
  // room had to time it out. Nothing about it is visible over a socket, where the reply is a round
  // trip away, which is exactly what makes it worth a comment.
  $('btn-reroll').onclick = () => {
    if (yard().scrap < REROLL_COST) return;
    audio.ui('press');
    setHint('Rerolling...');
    // The new hand is drawn by the authority: the seed that draws it is the same seed that decides
    // which beam the battle turns to, so a client is never given it.
    client.reroll();
  };

  $('btn-scrap').onclick = () => {
    audio.ui('press');
    selected = selected === 'remove' ? null : 'remove';
    renderAllWithHint();
    view.setGhost(hoverKey, null);
    view.setArcPreview(null, null);
  };

  $('btn-refit').onclick = () => {
    audio.ui('press');
    const res = client.refit();
    if (!res.ok) return deny(res.why);
    setHint(`Refitted ${res.repaired} part(s).`);
    renderAll();
  };

  // The commit gets the one rising three-note sound in the game. On the click only: the countdown
  // running out locks in as well, and confirming there would claim a decision nobody made.
  $('btn-lock').onclick = () => {
    if (done) return;
    audio.ui('confirm');
    setHint('Locked in. Waiting for the others.');
    $('btn-lock').disabled = true;
    client.lock();
  };

  attachFillButton(document.querySelector('#build-ui .tools'), () => {
    for (const [key, part] of devFillCommands(design(), hullIndex, yard().scrap, client.state.offer)) {
      client.place(key, part);
    }
    renderAllWithHint();
  });

  renderAllWithHint();
  $('btn-lock').disabled = false;

  return {
    // The authority corrected us, or answered a reroll. Redraw from whatever it says is true.
    refresh(reset) {
      if (done) return;
      renderAll();
      if (reset) setHint('The harbour master disagreed. Your ship is as shown.');
    },
    deny(why) {
      if (!done) deny(why);
    },
    destroy() {
      done = true;
      listeners.abort();
      view.setGhost(null, null);
      view.setArcPreview(null, null);
    },
  };
}
