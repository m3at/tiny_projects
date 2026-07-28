// Small shared maths for the simulation. Every function here runs thousands of times a
// simulated second, so each one exists because the obvious standard-library call was slower.

export const TAU = Math.PI * 2;

// Math.hypot is dramatically slower than the arithmetic it stands for.
export function len(x, z) {
  return Math.sqrt(x * x + z * z);
}

// Fold an angle into (-pi, pi]. Almost every call is already in range -- differences between two
// headings, mostly -- so that case is tested first and the modulo skipped.
export function wrapAngle(a) {
  if (a >= -Math.PI && a <= Math.PI) return a;
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

// Ship-local to world. Uses the heading's cached sine and cosine, which steer() refreshes once
// per tick; computing them here per call was most of the cost of the projectile hit test.
export function worldX(ship, lx, lz) {
  return ship.x + lx * ship.cos - lz * ship.sin;
}

export function worldZ(ship, lx, lz) {
  return ship.z + lx * ship.sin + lz * ship.cos;
}
