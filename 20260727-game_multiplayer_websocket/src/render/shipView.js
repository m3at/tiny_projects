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
import { CELL } from '../config.js';
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

// Colour comes entirely from instanceColor, so one material serves every part box.
const partMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7 });

function boxGeoFor(partId) {
  if (!boxGeoCache.has(partId)) {
    boxGeoCache.set(
      partId,
      new THREE.BoxGeometry(CELL * 0.78, PARTS[partId].height, CELL * 0.78),
    );
  }
  return boxGeoCache.get(partId);
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
  const hull = HULLS[hullIndex];
  const colours = PLAYER[player];
  const capacity = hull.cells.length;

  // ---- deck: one instanced mesh, one instance per hull cell ----
  const deck = new THREE.InstancedMesh(
    deckGeo,
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85 }),
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
    layer.box.setColorAt(cell.slot, tint.setHex(PARTS[cell.partId].color).multiplyScalar(0.35 + 0.65 * frac));

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
    new THREE.MeshBasicMaterial({ color: colours.hull, side: THREE.DoubleSide }),
  );
  prow.rotation.x = -Math.PI / 2; // flat, apex toward -z (forward)
  prow.position.set(0, 0.02, (bowZ - 0.5) * CELL);
  group.add(prow);

  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.1, 0.1, 5.5, 6),
    new THREE.MeshStandardMaterial({ color: SEA.spar }),
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
    deck.setColorAt(cell.deckIndex, tint.setHex(hex));
    deck.instanceColor.needsUpdate = true;
  }

  function setGhost(key, partId) {
    if (ghost) {
      group.remove(ghost);
      ghost.geometry.dispose();
      ghost.material.dispose();
      ghost = null;
    }
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

    const part = PARTS[partId];
    ghost = new THREE.Mesh(
      boxGeoFor(partId),
      new THREE.MeshStandardMaterial({
        color: part.color,
        roughness: 0.7,
        transparent: true,
        opacity: 0.5,
      }),
    );
    ghost.position.set(cell.dx * CELL, DECK_TOP + part.height / 2, cell.dz * CELL);
    group.add(ghost);
  }

  // ---- firing-arc preview ----
  // Arcs decide whether a gun ever bears, and nothing else in the UI shows them. This is a
  // direction indicator sitting just outside the hull, deliberately not a range indicator.
  const hullRadius = (hull.width / 2 + 0.6) * CELL;
  let arc = null;

  function clearArc() {
    if (!arc) return;
    group.remove(arc);
    arc.geometry.dispose();
    arc.material.dispose();
    arc = null;
  }

  function setArcPreview(key, partId) {
    clearArc();
    const gun = partId && PARTS[partId].gun;
    if (!gun) return;
    const cell = cells.get(key);
    if (!cell) return;

    // Broadside guns take their side from the flank they sit on; nothing to show on the spine.
    let centreDeg = 0;
    if (gun.arc === 'side') {
      if (cell.dx === 0) return;
      centreDeg = cell.dx > 0 ? 90 : -90;
    } else if (gun.arc === 'all') {
      centreDeg = 0;
    }
    const half = gun.arc === 'all' ? 180 : gun.halfArc;

    // RingGeometry theta runs from +x; once laid flat, +x is starboard and +90deg is the bow,
    // which is the mirror of the arc convention (0 = bow, +90 = starboard).
    const startDeg = 90 - centreDeg - half;
    arc = new THREE.Mesh(
      new THREE.RingGeometry(
        hullRadius,
        hullRadius + CELL * 0.85,
        48,
        1,
        (startDeg * Math.PI) / 180,
        (2 * half * Math.PI) / 180,
      ).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({
        color: FX.arc,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    arc.position.set(0, 0.05, cell.dz * CELL);
    group.add(arc);
  }

  // ---- battle-phase sync ----

  const dead = new Set();
  const falling = [];

  function syncFromBattle(ship) {
    group.position.set(ship.x, 0, ship.z);
    group.rotation.y = -ship.heading;

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
    if (!ship.byKey.get(HELM_KEY)?.alive) flag.visible = false;
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
    animate,
    dispose() {
      clearArc();
      deck.material.dispose();
      deck.dispose();
      for (const layer of layers.values()) {
        layer.box.dispose();
        layer.glyph.dispose();
      }
      if (ghost) {
        ghost.material.dispose();
        ghost = null;
      }
      for (const m of [silhouette, prow, pole, flag]) {
        m.geometry.dispose();
        m.material.dispose();
      }
    },
  };
}
