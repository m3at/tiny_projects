// Renders one ship as a plan-view grid of boxes.
//
// Every part is a coloured box plus a glyph decal, all of it built in buildPart(). That is
// the only function a real 3D asset needs to replace: swap the box for a loaded mesh and
// the build phase, the damage shading and the battle all keep working.

import * as THREE from 'three';
import { PARTS } from '../data/parts.js';
import { HULLS, cellKey } from '../data/hulls.js';
import { CELL } from '../config.js';
import { PLAYER, SEA, HOLE, FX } from '../theme.js';
import { glyphTexture } from './glyphs.js';
import { HELM_KEY } from '../sim/ship.js';

// Shared by every ship, so never disposed per-view.
const deckGeo = new THREE.BoxGeometry(CELL * 0.96, 0.24, CELL * 0.96);
const glyphGeo = new THREE.PlaneGeometry(CELL * 0.62, CELL * 0.62);

export function createShipView({ design, hullIndex, player, interactive = false }) {
  const group = new THREE.Group();
  const hull = HULLS[hullIndex];
  const colours = PLAYER[player];
  const cells = new Map(); // cellKey -> { deck, part, dx, dz }
  const pickTargets = [];
  const falling = [];

  // A deck plate for every hull cell, whether or not a part sits on it: an empty plate
  // reads as an open hole, which is exactly what it is.
  for (const c of hull.cells) {
    const deck = new THREE.Mesh(
      deckGeo,
      new THREE.MeshStandardMaterial({ color: colours.deck, roughness: 0.85 }),
    );
    deck.position.set(c.dx * CELL, 0, c.dz * CELL);
    deck.userData.cellKey = cellKey(c.dx, c.dz);
    group.add(deck);
    if (interactive) pickTargets.push(deck);
    cells.set(deck.userData.cellKey, { deck, part: null, dx: c.dx, dz: c.dz });
  }

  // Prow: a flat triangle on the deck plane. In plan view that reads as a bow far better
  // than anything with height, and it settles which way the ship is pointing.
  const bowZ = Math.min(...hull.cells.map((c) => c.dz));
  const prowShape = new THREE.Shape();
  prowShape.moveTo(-CELL * 0.48, 0);
  prowShape.lineTo(CELL * 0.48, 0);
  prowShape.lineTo(0, CELL * 1.25);
  const prow = new THREE.Mesh(
    new THREE.ShapeGeometry(prowShape),
    new THREE.MeshStandardMaterial({ color: colours.hull, roughness: 0.8, side: THREE.DoubleSide }),
  );
  prow.rotation.x = -Math.PI / 2; // lie flat, apex toward -z (forward)
  prow.position.set(0, 0.02, (bowZ - 0.5) * CELL);
  group.add(prow);

  // Stern colours: the only way to tell the two ships apart at a glance.
  const sternZ = Math.max(...hull.cells.map((c) => c.dz));
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

  // ---- parts ----

  function buildPart(partId) {
    const part = PARTS[partId];
    const g = new THREE.Group();
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(CELL * 0.78, part.height, CELL * 0.78),
      new THREE.MeshStandardMaterial({ color: part.color, roughness: 0.7 }),
    );
    g.add(box);

    const glyph = new THREE.Mesh(
      glyphGeo,
      new THREE.MeshBasicMaterial({
        map: glyphTexture(part.glyph, part.color),
        transparent: true,
        depthWrite: false,
      }),
    );
    glyph.rotation.x = -Math.PI / 2;
    g.add(glyph);

    g.userData = { box, glyph, height: part.height, baseColor: new THREE.Color(part.color) };
    setSoundness(g, 1);
    return g;
  }

  function disposePart(mesh) {
    group.remove(mesh);
    mesh.userData.box.geometry.dispose();
    mesh.userData.box.material.dispose();
    mesh.userData.glyph.material.dispose(); // shares glyphGeo, but the material is its own
  }

  // A battered part sits lower and darker. One function drives both the accumulated damage
  // shown during the build phase and live damage during a battle.
  function setSoundness(mesh, frac) {
    const { box, glyph, height, baseColor } = mesh.userData;
    const scale = 0.55 + 0.45 * frac;
    box.material.color.copy(baseColor).multiplyScalar(0.35 + 0.65 * frac);
    box.scale.y = scale;
    box.position.y = 0.12 + (height * scale) / 2;
    glyph.position.y = 0.12 + height * scale + 0.02;
  }

  // Rebuild every part mesh from the design. Cheap enough at these cell counts that
  // incremental updates are not worth the bookkeeping.
  function refresh() {
    for (const [key, cell] of cells) {
      if (cell.part) {
        disposePart(cell.part);
        cell.part = null;
      }
      const slot = design.parts[key];
      if (!slot) continue;
      const mesh = buildPart(slot.id);
      mesh.position.set(cell.dx * CELL, 0, cell.dz * CELL);
      group.add(mesh);
      cell.part = mesh;
      setSoundness(mesh, slot.hp / PARTS[slot.id].hp);
    }
  }

  // ---- build-phase affordances ----

  let ghost = null;
  function setGhost(key, partId) {
    if (ghost) {
      disposePart(ghost);
      ghost = null;
    }
    for (const cell of cells.values()) cell.deck.material.emissive.setHex(0x000000);
    if (!key) return;
    const cell = cells.get(key);
    if (!cell) return;
    cell.deck.material.emissive.setHex(FX.ghost);
    if (!partId || design.parts[key]) return;
    ghost = buildPart(partId);
    ghost.position.set(cell.dx * CELL, 0, cell.dz * CELL);
    ghost.userData.box.material.transparent = true;
    ghost.userData.box.material.opacity = 0.5;
    group.add(ghost);
  }

  // ---- battle-phase sync ----

  const dead = new Set();
  function syncFromBattle(ship) {
    group.position.set(ship.x, 0, ship.z);
    group.rotation.y = -ship.heading;

    for (const cell of ship.cells) {
      const view = cells.get(cell.key);
      if (!view || !view.part) continue;
      if (cell.alive) {
        setSoundness(view.part, cell.hp / cell.maxHp);
      } else if (!dead.has(cell.key)) {
        dead.add(cell.key);
        view.deck.material.color.setHex(HOLE);
        if (cell.id === 'mast') {
          // Masts go over the side rather than blinking out.
          falling.push({ mesh: view.part, t: 0, dir: cell.dx >= 0 ? 1 : -1 });
        } else {
          view.part.visible = false;
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
      f.mesh.rotation.z = f.dir * k * Math.PI * 0.52;
      f.mesh.position.y = -k * 0.6;
      if (f.t > 1.6) {
        f.mesh.visible = false;
        falling.splice(i, 1);
      }
    }
  }

  refresh();

  return {
    group,
    pickTargets,
    refresh,
    setGhost,
    syncFromBattle,
    animate,
    dispose() {
      for (const cell of cells.values()) {
        if (cell.part) disposePart(cell.part);
        cell.deck.material.dispose();
      }
      if (ghost) disposePart(ghost);
      for (const m of [prow, pole, flag]) {
        m.geometry.dispose();
        m.material.dispose();
      }
    },
  };
}
