// Does the simulation produce the same battle on a different JavaScript engine?
//
// It has to. The server runs the sim under Node's V8 and decides the outcome; every client replays
// it to draw the battle, and one of those clients is Safari, which is JavaScriptCore. If the two
// disagree the desync detector fires constantly and means nothing.
//
// The answer is not obvious and it is not free. IEEE 754 requires correct rounding for sqrt, which
// is why geometry.js len() is safe on every engine forever, but only *recommends* it for sin, cos,
// atan2, pow and friends -- ECMA-262 calls them implementation-approximated. Measured here, V8 and
// JSC disagree on 4% of sin arguments and 21% of atan2 arguments, by up to 3 ULP.
//
// The fix is in sim/geometry.js: every transcendental in the simulation is rounded to float32 with
// Math.fround. Two results that differ in the last bit or three of a double collapse onto the same
// float32, and the ordinary arithmetic between them is exactly specified, so the whole chain becomes
// bit-identical. It is not a proof -- a double sitting exactly on a float32 rounding boundary still
// splits -- but it moves the hazard from "certain, constantly" to about 2^-29 per call, and it costs
// nothing measurable in either throughput or outcomes.
//
//   node tools/engines.js          compare node against jsc
//
// Needs Safari's jsc, which ships with macOS at the path below. Without it this prints a note and
// exits 0 rather than failing a run on a machine that cannot answer the question.

import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const JSC =
  '/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc';
const SCRIPT = 'tools/fingerprint.js';

function run(label, cmd, args) {
  const t0 = Date.now();
  const out = execFileSync(cmd, args, { encoding: 'utf8', maxBuffer: 1 << 28 });
  console.log(`  ${label.padEnd(6)} ${((Date.now() - t0) / 1000).toFixed(1)}s  ${out.split('\n').length} lines`);
  return out.split('\n');
}

if (!existsSync(JSC)) {
  console.log(`jsc not found at ${JSC}`);
  console.log('Cross-engine determinism unchecked. This is a macOS-only check.');
  process.exit(0);
}

console.log('running the same battles under two engines');
const a = run('node', process.execPath, [SCRIPT]);
const b = run('jsc', JSC, ['-m', SCRIPT]);

if (a.length !== b.length) {
  console.log(`\nFAIL different line counts: node ${a.length}, jsc ${b.length}`);
  process.exit(1);
}

let diffs = 0;
let firstDiff = null;
let outcomeDiffs = 0;
let samples = 0;
for (let i = 0; i < a.length; i++) {
  const isOutcome = a[i].startsWith('=');
  if (!isOutcome && a[i] !== '' && !a[i].startsWith('#')) samples++;
  if (a[i] === b[i]) continue;
  diffs++;
  if (isOutcome) outcomeDiffs++;
  if (firstDiff === null) firstDiff = i;
}

const pct = (n, d) => `${((100 * n) / (d || 1)).toFixed(2)}%`;
console.log(`\n  state samples      ${samples}`);
console.log(`  lines differing    ${diffs} (${pct(diffs, a.length)})`);
console.log(`  outcomes differing ${outcomeDiffs}`);
if (firstDiff !== null) {
  console.log(`\n  first difference at line ${firstDiff}:`);
  console.log(`    node ${a[firstDiff]}`);
  console.log(`    jsc  ${b[firstDiff]}`);
}

if (diffs === 0) {
  console.log('\n  bit-identical on both engines.');
  process.exit(0);
}
console.log(
  '\nFAIL the simulation is engine-dependent. Every transcendental in sim/ must go through the\n' +
    'Math.fround wrappers in sim/geometry.js -- see the note at the top of this file.',
);
process.exit(1);
