// Renders one ship as a plan-view grid of boxes.
//
// Batching: the deck is one instanced mesh, and each part *type* present on the ship gets
// one instanced layer for its boxes and one for its glyph decals. That keeps the draw-call
// count tied to the variety of parts aboard rather than the number of cells, and it is the
// same shape the renderer wants once real assets exist: one loaded mesh per part type,
// instanced across the cells that carry it. buildLayer() is the seam.
//
// Per-instance colour carries both part identity and damage tint, so a single material is
// shared by every box on every ship.
//
// Cost: the ship's own movement is a transform on the group, so instance matrices are only
// rewritten when a cell's condition actually changes, not every frame.

import * as THREE from 'three';
import { PARTS } from '../data/parts.js';
import { HULLS, cellKey } from '../data/hulls.js';
import { CELL, SINK_DROP } from '../config.js';
import { PLAYER, SEA, HOLE, FX } from '../theme.js';
import { glyphTexture } from './glyphs.js';
import { HELM_KEY } from '../sim/ship.js';

const DECK_HEIGHT = 0.24;
const DECK_TOP = 0.12;

// Shared across every ship and every view; never disposed.
const deckGeo = new THREE.BoxGeometry(CELL * 0.96, DECK_HEIGHT, CELL * 0.96);
const glyphGeo = new THREE.PlaneGeometry(CELL * 0.62, CELL * 0.62).rotateX(-Math.PI / 2);
const boxGeoCache = new Map();
const glyphMatCache = new Map();
const ghostMatCache = new Map();
const arcGeoCache = new Map();

// Colour comes entirely from instanceColor, so one material serves every part box.
//
// Lambert, not Standard. Standard is physically based, and against these lights that costs roughly
// ten times the fragment arithmetic for a specular lobe so broad at roughness 0.7 that it is not
// visible. Worse, three.js emits RE_IndirectSpecular unconditionally for Standard, so every pixel
// runs the image-based specular approximation -- including an exp2 -- to compute a slight diffuse
// darkening, even though there is no environment map for it to reflect.
//
// The usual objection is that Lambert shades per vertex and would band these boxes. That has been
// false since r144: Lambert has been per-fragment for years and the three.js *manual* still says
// otherwise, while its own API docs say per-fragment. Verified in the vendored r185 source, whose
// Lambert fragment shader runs RE_Direct_Lambert against the interpolated normal exactly as
// Standard does. The only thing given up is the specular term.
const partMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });

// The deck plates want exactly the same material as the part boxes -- white, lit, tinted entirely
// through instanceColor -- so they share one. It used to be built per ship view, which meant two
// fresh materials every round for a result indistinguishable from this one.
const deckMaterial = partMaterial;

// Bare timber: masts, flagpoles. One colour, no per-instance variation, so one material for every
// ship in the game rather than one per view.
const sparMaterial = new THREE.MeshLambertMaterial({ color: SEA.spar });

// Shared by every arc band on every ship: only the geometry differs.
const arcMaterial = new THREE.MeshBasicMaterial({
  color: FX.arc,
  transparent: true,
  opacity: 0.3,
  depthWrite: false,
});

function boxGeoFor(partId) {
  if (!boxGeoCache.has(partId)) {
    boxGeoCache.set(
      partId,
      new THREE.BoxGeometry(CELL * 0.78, PARTS[partId].height, CELL * 0.78),
    );
  }
  return boxGeoCache.get(partId);
}

// One translucent material per part type, cached and never disposed. Building one per hover and
// disposing the last was measurable: a fresh material has no compiled program attached, so three.js
// recomputes its cache key and looks it up again, and disposing the old one can delete the program
// outright -- the ghost is the only non-instanced standard material in the scene, so nothing else
// keeps it alive. Nine part types is nine materials, which is cheaper than one per mouse move.
function ghostMatFor(partId) {
  if (!ghostMatCache.has(partId)) {
    ghostMatCache.set(
      partId,
      new THREE.MeshLambertMaterial({
        color: PARTS[partId].color,
        transparent: true,
        opacity: 0.5,
      }),
    );
  }
  return ghostMatCache.get(partId);
}

// Arc bands take one of a handful of shapes -- three centres by a few half-widths by one radius per
// hull -- so they cache exactly. They were being built and disposed on every pointer move.
function arcGeoFor(radius, startDeg, half) {
  const key = `${radius.toFixed(2)}:${startDeg}:${half}`;
  if (!arcGeoCache.has(key)) {
    arcGeoCache.set(
      key,
      new THREE.RingGeometry(
        radius,
        radius + CELL * 0.85,
        48,
        1,
        (startDeg * Math.PI) / 180,
        (2 * half * Math.PI) / 180,
      ).rotateX(-Math.PI / 2),
    );
  }
  return arcGeoCache.get(key);
}

function glyphMatFor(partId) {
  if (!glyphMatCache.has(partId)) {
    const part = PARTS[partId];
    glyphMatCache.set(
      partId,
      new THREE.MeshBasicMaterial({
        map: glyphTexture(part.glyph, part.color),
        transparent: true,
        depthWrite: false,
      }),
    );
  }
  return glyphMatCache.get(partId);
}

const pos = new THREE.Vector3();
const quat = new THREE.Quaternion();
const scale = new THREE.Vector3();
const mat = new THREE.Matrix4();
const tint = new THREE.Color();
const ZERO = new THREE.Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);

export function createShipView({ design, hullIndex, player, interactive = false }) {
  const group = new THREE.Group();
  // Yaw first, then local pitch and roll. That lets a sinking hull heel in its own frame without
  // changing the battle heading that syncFromBattle writes every frame.
  group.rotation.order = 'YXZ';
  const hull = HULLS[hullIndex];
  const colours = PLAYER[player];
  const capacity = hull.cells.length;

  // ---- deck: one instanced mesh, one instance per hull cell ----
  const deck = new THREE.InstancedMesh(
    deckGeo,
    deckMaterial,
    capacity,
  );
  deck.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
  deck.frustumCulled = false;
  group.add(deck);

  // cellKey -> { dx, dz, deckIndex, partId, slot, soundness }
  const cells = new Map();
  const keyByInstance = []; // deck instanceId -> cellKey, for raycast hits
  hull.cells.forEach((c, i) => {
    const key = cellKey(c.dx, c.dz);
    pos.set(c.dx * CELL, 0, c.dz * CELL);
    mat.compose(pos, quat.identity(), scale.set(1, 1, 1));
    deck.setMatrixAt(i, mat);
    const base = c.dx === 0 ? colours.spine : colours.deck;
    deck.setColorAt(i, tint.setHex(base));
    cells.set(key, {
      key,
      dx: c.dx,
      dz: c.dz,
      deckIndex: i,
      base,
      partId: null,
      slot: -1,
      soundness: 1,
    });
    keyByInstance[i] = key;
  });
  deck.instanceMatrix.needsUpdate = true;
  deck.instanceColor.needsUpdate = true;

  // ---- part layers, created on demand per part type ----
  const layers = new Map(); // partId -> { box, glyph, count }

  function buildLayer(partId) {
    const box = new THREE.InstancedMesh(boxGeoFor(partId), partMaterial, capacity);
    box.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    box.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    box.frustumCulled = false;
    const glyph = new THREE.InstancedMesh(glyphGeo, glyphMatFor(partId), capacity);
    glyph.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    glyph.frustumCulled = false;
    group.add(box, glyph);
    const layer = { box, glyph, count: 0 };
    layers.set(partId, layer);
    return layer;
  }

  const layerFor = (partId) => layers.get(partId) ?? buildLayer(partId);

  // Write one cell's box and glyph transforms. `frac` is how sound the part is; `fall` is
  // the roll angle used when a mast goes over the side.
  function writeCell(cell, frac, fall = 0) {
    const layer = layers.get(cell.partId);
    if (!layer || cell.slot < 0) return;
    const height = PARTS[cell.partId].height;
    const sy = 0.55 + 0.45 * frac;

    quat.setFromAxisAngle({ x: 0, y: 0, z: 1 }, fall);
    pos.set(cell.dx * CELL, DECK_TOP + (height * sy) / 2 - (fall ? 0.6 : 0), cell.dz * CELL);
    mat.compose(pos, quat, scale.set(1, sy, 1));
    layer.box.setMatrixAt(cell.slot, mat);
    layer.box.setColorAt(
      cell.slot,
      sinkTint(tint.setHex(PARTS[cell.partId].color).multiplyScalar(0.35 + 0.65 * frac)),
    );

    pos.y = DECK_TOP + height * sy + 0.02 - (fall ? 0.6 : 0);
    mat.compose(pos, quat, scale.set(1, 1, 1));
    layer.glyph.setMatrixAt(cell.slot, mat);

    layer.box.instanceMatrix.needsUpdate = true;
    layer.box.instanceColor.needsUpdate = true;
    layer.glyph.instanceMatrix.needsUpdate = true;
    cell.soundness = frac;
  }

  function hideCell(cell) {
    const layer = layers.get(cell.partId);
    if (!layer || cell.slot < 0) return;
    layer.box.setMatrixAt(cell.slot, ZERO);
    layer.glyph.setMatrixAt(cell.slot, ZERO);
    layer.box.instanceMatrix.needsUpdate = true;
    layer.glyph.instanceMatrix.needsUpdate = true;
  }

  // Reassign every slot from the design. O(cells), so cheap enough to run on each edit.
  function refresh() {
    for (const layer of layers.values()) layer.count = 0;
    for (const cell of cells.values()) {
      cell.partId = null;
      cell.slot = -1;
    }
    for (const [key, slot] of Object.entries(design.parts)) {
      const cell = cells.get(key);
      if (!cell) continue;
      const layer = layerFor(slot.id);
      cell.partId = slot.id;
      cell.slot = layer.count++;
      writeCell(cell, slot.hp / PARTS[slot.id].hp);
    }
    for (const layer of layers.values()) {
      layer.box.count = layer.count;
      layer.glyph.count = layer.count;
      // An instanced mesh with zero instances still costs a draw call otherwise.
      layer.box.visible = layer.count > 0;
      layer.glyph.visible = layer.count > 0;
    }
    dead.clear();
  }

  // ---- hull silhouette, prow, flag: fixed decoration ----
  const bowZ = Math.min(...hull.cells.map((c) => c.dz));
  const sternZ = Math.max(...hull.cells.map((c) => c.dz));

  // Ownership marker. Both ships carry the same part palette, so which is which has to come
  // from the hull, not from the cargo.
  const silhouette = new THREE.Mesh(
    new THREE.CircleGeometry(1, 48).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: colours.flag,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    }),
  );
  silhouette.scale.set((hull.width / 2 + 0.55) * CELL, 1, (hull.length / 2 + 0.75) * CELL);
  silhouette.position.y = -0.2;
  // Only needed in battle. During the build phase whose ship it is was never in question, and
  // the disc just competes with the deck plates.
  silhouette.visible = !interactive;
  group.add(silhouette);

  // Unlit: a flat upward-facing surface under a bright sun blows out to near-white, which
  // read as a different colour on each ship.
  const prowShape = new THREE.Shape();
  prowShape.moveTo(-CELL * 0.48, 0);
  prowShape.lineTo(CELL * 0.48, 0);
  prowShape.lineTo(0, CELL * 1.25);
  const prow = new THREE.Mesh(
    new THREE.ShapeGeometry(prowShape),
    new THREE.MeshBasicMaterial({ color: colours.hull }),
  );
  prow.rotation.x = -Math.PI / 2; // flat, apex toward -z (forward)
  prow.position.set(0, 0.02, (bowZ - 0.5) * CELL);
  group.add(prow);

  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.1, 0.1, 5.5, 6),
    sparMaterial,
  );
  pole.position.set(0, 2.75, (sternZ + 0.55) * CELL);
  group.add(pole);

  const flag = new THREE.Mesh(
    new THREE.PlaneGeometry(2.1, 1.2),
    new THREE.MeshBasicMaterial({ color: colours.flag, side: THREE.DoubleSide }),
  );
  flag.position.set(1.05, 4.6, (sternZ + 0.55) * CELL);
  group.add(flag);

  // ---- build-phase affordances ----

  let ghost = null;
  let highlighted = null;

  function setDeckColour(cell, hex) {
    deck.setColorAt(cell.deckIndex, sinkTint(tint.setHex(hex)));
    deck.instanceColor.needsUpdate = true;
  }

  // One mesh for the life of the view, hidden when there is nothing to preview. Its geometry and
  // material both come from caches, so a hover is three property writes rather than an allocation,
  // a shader lookup and two disposals.
  function setGhost(key, partId) {
    if (ghost) ghost.visible = false;
    if (highlighted && highlighted !== key) {
      const prev = cells.get(highlighted);
      setDeckColour(prev, dead.has(highlighted) ? HOLE : prev.base);
      highlighted = null;
    }
    if (!key) return;
    const cell = cells.get(key);
    if (!cell) return;
    setDeckColour(cell, FX.ghost);
    highlighted = key;
    if (!partId || design.parts[key]) return;

    if (!ghost) {
      ghost = new THREE.Mesh(boxGeoFor(partId), ghostMatFor(partId));
      ghost.frustumCulled = false;
      group.add(ghost);
    }
    ghost.geometry = boxGeoFor(partId);
    ghost.material = ghostMatFor(partId);
    ghost.position.set(cell.dx * CELL, DECK_TOP + PARTS[partId].height / 2, cell.dz * CELL);
    ghost.visible = true;
  }

  // ---- firing-arc preview ----
  // Arcs decide whether a gun ever bears, and nothing else in the UI shows them. This is a
  // direction indicator sitting just outside the hull, deliberately not a range indicator.
  const hullRadius = (hull.width / 2 + 0.6) * CELL;
  const arcs = [];

  // Geometries are cached and shared, so they are never disposed here -- only detached.
  function clearArc() {
    for (const mesh of arcs) group.remove(mesh);
    arcs.length = 0;
  }

  function setArcPreview(key, partId) {
    clearArc();
    const gun = partId && PARTS[partId].gun;
    if (!gun) return;
    const cell = cells.get(key);
    if (!cell) return;

    // One band per firing window, matching sim/ship.js: a broadside points out its own flank, a
    // bow chaser forward, a swivel all round. Nothing to show for a broadside on the spine,
    // where it cannot go anyway.
    if (gun.arc === 'side' && cell.dx === 0) return;
    const centres = gun.arc === 'side' ? [cell.dx > 0 ? 90 : -90] : [0];
    const half = gun.arc === 'all' ? 180 : gun.halfArc;

    for (const centreDeg of centres) {
      // RingGeometry theta runs from +x; once laid flat, +x is starboard and +90deg is the bow,
      // which is the mirror of the arc convention (0 = bow, +90 = starboard).
      const startDeg = 90 - centreDeg - half;
      const mesh = new THREE.Mesh(arcGeoFor(hullRadius, startDeg, half), arcMaterial);
      mesh.position.set(0, 0.05, cell.dz * CELL);
      arcs.push(mesh);
      group.add(mesh);
    }
  }

  // ---- battle-phase sync ----

  const dead = new Set();
  const falling = [];

  function syncFromBattle(ship) {
    // y comes from how far under she is, not from zero: this runs every frame and setSunk does not,
    // so writing a flat zero here quietly undid the whole descent -- the ship faded on the spot
    // without ever settling.
    group.position.set(ship.x, -SINK_DROP * sunk, ship.z);
    group.rotation.set(sinkPitch, -ship.heading, sinkRoll);

    for (const cell of ship.cells) {
      const view = cells.get(cell.key);
      if (!view || view.slot < 0) continue;
      if (cell.alive) {
        const frac = cell.hp / cell.maxHp;
        // Only touch the buffers when the condition actually moved.
        if (Math.abs(frac - view.soundness) > 0.001) writeCell(view, frac);
      } else if (!dead.has(cell.key)) {
        dead.add(cell.key);
        setDeckColour(view, HOLE);
        if (cell.id === 'mast') {
          falling.push({ cell: view, t: 0, dir: cell.dx >= 0 ? 1 : -1 });
        } else {
          hideCell(view);
        }
      }
    }
    if (!ship.helm.alive) flag.visible = false;
  }

  // ---- going under ----
  //
  // Driven from outside, because how far under she is comes from the battle clock and this view does
  // not have one. Everything her own materials own fades; everything on a shared material is hidden
  // instead, since fading one ship's shared material would fade the rest of the fleet with it.
  //
  // A colour rewrite is O(cells) and would be wasteful every frame for two and a half seconds, so it
  // only happens when the amount has actually moved -- which for a fade nobody is looking at closely
  // is about fifty times over the whole descent.
  let sinkProgress = 0;
  let sunk = 0;
  let sunkDrawn = -1;
  let sinkRoll = 0;
  let sinkPitch = 0;
  let sinkBiasReady = false;
  let rollSign = player % 2 ? -1 : 1;
  let pitchSign = player % 3 ? 1 : -1;
  const seaTint = new THREE.Color(SEA.water);

  function sinkTint(colour) {
    return sunk > 0 ? colour.lerp(seaTint, sunk * 0.92) : colour;
  }

  function setSunk(amount) {
    const next = Math.max(0, Math.min(1, amount));
    if (next === sinkProgress) return;
    sinkProgress = next;
    if (!sinkBiasReady && next > 0) {
      // The presentation leans toward the side that actually came apart. It is computed once from
      // the already-known dead cells, so there is no per-frame scan and no simulation input.
      let dx = 0;
      let dz = 0;
      for (const key of dead) {
        const cell = cells.get(key);
        if (cell) {
          dx += cell.dx;
          dz += cell.dz;
        }
      }
      if (Math.abs(dx) > 0.5) rollSign = Math.sign(dx);
      if (Math.abs(dz) > 0.5) pitchSign = Math.sign(dz);
      sinkBiasReady = true;
    }
    const descentT = Math.max(0, Math.min(1, (next - 0.08) / 0.68));
    const descent = descentT * descentT * (3 - 2 * descentT);
    const heel = Math.min(1, next / 0.2);
    sunk = descent;
    sinkRoll = rollSign * (0.06 * heel + 0.22 * descent);
    sinkPitch = pitchSign * 0.1 * descent;
    group.position.y = -SINK_DROP * sunk;
    group.rotation.x = sinkPitch;
    group.rotation.z = sinkRoll;
    silhouette.material.opacity = 0.22 * (1 - sunk);
    for (const mesh of [prow, flag]) {
      mesh.material.transparent = true;
      mesh.material.opacity = 1 - sunk;
    }
    // sparMaterial is every ship's masts, so the flagpole cannot fade. It slips under instead, and
    // the part glyphs go with it: a bright white letter on a hull the colour of the sea reads as a
    // mistake rather than as a wreck.
    pole.visible = sunk < 0.8;
    // Once she is the colour of the water there is nothing left to look at, and an invisible wreck
    // still costs a draw call per part layer.
    group.visible = sunk < 0.995;
    for (const layer of layers.values()) {
      layer.glyph.visible = layer.count > 0 && sunk < 0.4;
    }
    if (Math.abs(sunk - sunkDrawn) < 0.02 && sunk < 1) return;
    sunkDrawn = sunk;
    for (const cell of cells.values()) {
      setDeckColour(cell, dead.has(cell.key) ? HOLE : cell.base);
      if (cell.slot >= 0 && cell.partId) writeCell(cell, cell.soundness);
    }
  }

  function animate(dt) {
    for (let i = falling.length - 1; i >= 0; i--) {
      const f = falling[i];
      f.t += dt;
      const k = Math.min(1, f.t / 0.8);
      writeCell(f.cell, f.cell.soundness, f.dir * k * Math.PI * 0.52);
      if (f.t > 1.6) {
        hideCell(f.cell);
        falling.splice(i, 1);
      }
    }
  }

  refresh();

  return {
    group,
    // Raycasting against the deck reports an instanceId, which maps back to a cell.
    pickTargets: interactive ? [deck] : [],
    cellKeyForInstance(instanceId) {
      return keyByInstance[instanceId] ?? null;
    },
    refresh,
    setGhost,
    setArcPreview,
    syncFromBattle,
    setSunk,
    animate,
    dispose() {
      clearArc();
      deck.dispose();
      for (const layer of layers.values()) {
        layer.box.dispose();
        layer.glyph.dispose();
      }
      // The ghost's geometry and material are both shared caches; only the mesh is ours.
      if (ghost) {
        group.remove(ghost);
        ghost = null;
      }
      // Geometry is per view and goes; materials are only ours where the colour is the player's.
      // The flagpole shares sparMaterial with every other ship, so disposing it here would blank
      // the masts on the opposing ship as well.
      for (const m of [silhouette, prow, pole, flag]) m.geometry.dispose();
      for (const m of [silhouette, prow, flag]) m.material.dispose();
    },
  };
}
