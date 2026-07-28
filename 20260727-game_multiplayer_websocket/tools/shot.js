// Drives headless Chrome over CDP to catch console errors and grab screenshots.
// No dependencies: Node's built-in WebSocket talks to Chrome directly.
//
//   node tools/shot.js out.png "steps" [query]
//
// A step is either a number (wait that many ms), JS to evaluate in the page, or
// '@path/to/file.js' to evaluate that file's contents (no shell escaping to fight).
// Steps are separated by ';;'. Example:
//   node tools/shot.js b.png "800 ;; ovBtn() ;; 500 ;; ovBtn() ;; 1200"
//
// The third argument is appended to the page URL, for the dev harness in src/dev.js:
//   node tools/shot.js b.png "6000" "?dev=brawler,crusher&round=5&x=4"

import { readFileSync } from 'node:fs';

import { attach, sleep } from './cdp.js';

const out = process.argv[2] || 'shot.png';
const steps = (process.argv[3] || '1200').split(';;').map((s) => s.trim()).filter(Boolean);
const query = process.argv[4] || '';

const page = await attach();
const { send, evalIn } = page;

// Convenience helpers available to step scripts.
const PRELUDE = `
  globalThis.ovBtn = () => { const b = document.getElementById('ov-btn'); if (b && !document.getElementById('overlay').classList.contains('hidden')) { b.click(); return 'clicked ' + b.textContent; } return 'overlay hidden'; };
  globalThis.lock = () => { document.getElementById('btn-lock').click(); return 'locked'; };
  globalThis.fill = (which) => globalThis.__fill ? globalThis.__fill(which) : 'no fill hook';
  'ok'
`;

await page.open(query, 1400);
// After the page is up, never before: an override applied first leaves the WebGL surface out of
// the captured frame entirely, so the shot comes back as a correct HUD over empty water.
await page.resize(1440, 900);
await evalIn(PRELUDE);

for (const step of steps) {
  if (/^\d+$/.test(step)) {
    await sleep(Number(step));
    continue;
  }
  const expression = step.startsWith('@') ? readFileSync(step.slice(1), 'utf8') : step;
  const val = await evalIn(expression, { soft: true });
  console.log(`  step ${JSON.stringify(step.slice(0, 60))} -> ${JSON.stringify(val)}`);
}

console.log((await page.screenshot(out)) ? `wrote ${out}` : 'screenshot failed');
page.printLogs();
// Leave the tab idle: a live WebGL page keeps a core busy long after we stop looking.
await page.close({ keep: !!process.env.KEEP });
