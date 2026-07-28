// Running the simulation with the source altered, without altering the source.
//
// Both tools/ablate.js ("does this mechanic matter at all") and tools/tune.js ("what should this
// number be") work by copying src/ to a temp directory, applying literal text patches, and
// importing the copy. This is the shared machinery.
//
// Patches match by exact string and throw when the target is missing. That is deliberate: a sweep
// that silently no-ops would quietly report that a constant does not matter. It does mean editing
// config.js or parts.js can break a patch string, and the fix belongs in the same change.

import { cpSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const SRC = new URL('../src/', import.meta.url).pathname;
const roots = [];

// `patches` is an array of [file, find, replace], relative to src/.
export function buildVariant(name, patches) {
  const dir = mkdtempSync(join(tmpdir(), 'variant-'));
  roots.push(dir);
  cpSync(SRC, join(dir, 'src'), { recursive: true });
  for (const [file, find, replace] of patches) {
    const path = join(dir, 'src', file);
    const text = readFileSync(path, 'utf8');
    if (!text.includes(find)) throw new Error(`${name}: patch target missing in ${file}: ${find}`);
    writeFileSync(path, text.replace(find, replace));
  }
  return dir;
}

// The bundle every harness function takes, so a patched tree can stand in for the real one.
export async function loadVariant(dir) {
  const [ship, battle, gunnery, autobuild, config] = await Promise.all([
    import(join(dir, 'src/sim/ship.js')),
    import(join(dir, 'src/sim/battle.js')),
    import(join(dir, 'src/sim/gunnery.js')),
    import(join(dir, 'src/autobuild.js')),
    import(join(dir, 'src/config.js')),
  ]);
  return { ship, battle, gunnery, autobuild, config };
}

export async function variant(name, patches) {
  return loadVariant(buildVariant(name, patches));
}

// Check every patch resolves before running anything. A stale find-string is the normal cost of
// editing config.js, and finding out after three minutes of battles is a waste of a run.
export function checkPatches(table) {
  const stale = [];
  for (const [name, patches] of Object.entries(table)) {
    for (const [file, find] of patches) {
      const text = readFileSync(join(SRC, file), 'utf8');
      if (!text.includes(find)) stale.push(`${name}: ${file} no longer contains: ${find.slice(0, 70)}`);
    }
  }
  if (stale.length) {
    throw new Error(`${stale.length} stale patch target(s):\n  ${stale.join('\n  ')}`);
  }
}

export function cleanupVariants() {
  for (const dir of roots) rmSync(dir, { recursive: true, force: true });
  roots.length = 0;
}
