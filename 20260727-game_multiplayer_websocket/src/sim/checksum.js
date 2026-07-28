// A fingerprint of a battle's state, so a client can tell whether its replay still matches the
// authority's.
//
// FNV-1a over quantised state rather than raw doubles. The simulation is bit-identical across
// engines (see geometry.js and tools/engines.js), so hashing the raw bits would work too, but it
// would need a DataView per number and it would report a mismatch for a difference far below what
// anyone could see. A twelve-bit fraction is about a quarter of a millimetre on a 120-unit arena:
// tight enough that a real divergence is caught within a second of starting, loose enough that the
// residual risk from float32 rounding boundaries does not throw false alarms.
//
// Cheap on purpose -- it runs on the server for every room every thirty ticks.

const Q = 4096;

export function checksum(battle) {
  let h = 2166136261 >>> 0;
  // Local rather than a closure, because this is called often enough for the allocation to matter.
  h ^= battle.tickCount >>> 0;
  h = Math.imul(h, 16777619);

  for (const ship of battle.ships) {
    h ^= Math.round(ship.x * Q) >>> 0;
    h = Math.imul(h, 16777619);
    h ^= Math.round(ship.z * Q) >>> 0;
    h = Math.imul(h, 16777619);
    h ^= Math.round(ship.heading * Q) >>> 0;
    h = Math.imul(h, 16777619);
    h ^= Math.round(ship.speed * Q) >>> 0;
    h = Math.imul(h, 16777619);
    // The state a player can actually read off the panels, which is the state a desync would be
    // noticed in: hands, hull, and whether she is still in the fight.
    h ^= (ship.crew + ship.aliveCells * 97 + (ship.out ? 1 : 0)) >>> 0;
    h = Math.imul(h, 16777619);
    for (const cell of ship.cells) {
      h ^= (Math.round(cell.hp * Q) + (cell.alive ? 0 : 0x5f5f)) >>> 0;
      h = Math.imul(h, 16777619);
    }
  }
  h ^= battle.projectiles.length >>> 0;
  h = Math.imul(h, 16777619);
  return h >>> 0;
}
