// Small shared maths for the simulation. Every function here runs thousands of times a
// simulated second, so each one exists because the obvious standard-library call was slower.

export const TAU = Math.PI * 2;

// Every sine, cosine and arc tangent in the simulation goes through one of these three, and the
// reason is the whole networked game.
//
// ECMA-262 calls sin, cos, atan2, pow, exp and log "implementation-approximated": an engine may
// return any value within an unspecified tolerance. Measured with tools/engines.js, V8 and Safari's
// JavaScriptCore disagree on 4% of sin arguments and 21% of atan2 arguments, by up to 3 units in the
// last place. That was enough to make 68% of sampled ship states differ between Node and Safari
// within the first second of a battle -- bounded, never once changing a winner, and still fatal,
// because a desync detector that fires on two thirds of its checks detects nothing.
//
// Rounding the result to float32 collapses a disagreement of a few double ULP onto one value, and
// ordinary arithmetic (+ - * /) and sqrt are exactly specified by IEEE 754, so the rest of the chain
// follows. After this, tools/engines.js reports the two engines bit-identical.
//
// Not a proof: a result sitting exactly on a float32 rounding boundary still splits, which is about
// one call in 2^29. That residue is why the server's outcome is authoritative and a client
// disagreement is a cosmetic resync rather than a lost match. Cost, measured: nothing on
// tools/bench.js and not one line of tools/golden.js.
//
// IEEE 754 requires correct rounding for square root, so len() below needs no wrapper and never has.
export const fsin = (a) => Math.fround(Math.sin(a));
export const fcos = (a) => Math.fround(Math.cos(a));
export const fatan2 = (y, x) => Math.fround(Math.atan2(y, x));

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
