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

import { writeFileSync, readFileSync } from 'node:fs';

const PORT = 9222;
const out = process.argv[2] || 'shot.png';
const steps = (process.argv[3] || '1200').split(';;').map((s) => s.trim()).filter(Boolean);
const query = process.argv[4] || '';
const url = (process.env.URL || 'http://127.0.0.1:8123/index.html') + query;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((t) => t.type === 'page');
if (!page) {
  console.error('no page target; is chrome running with --remote-debugging-port?');
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const logs = [];
const t0 = Date.now();
const stamp = () => String(Date.now() - t0).padStart(5) + 'ms';

ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
    return;
  }
  if (msg.method === 'Runtime.consoleAPICalled') {
    const text = msg.params.args.map((a) => a.value ?? a.description ?? a.type).join(' ');
    logs.push(`${stamp()} [${msg.params.type}] ${text}`);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    const d = msg.params.exceptionDetails;
    logs.push(`${stamp()} [EXCEPTION] ${d.exception?.description || d.text}`);
  } else if (msg.method === 'Log.entryAdded') {
    const e = msg.params.entry;
    if (e.level === 'error' || e.level === 'warning') logs.push(`${stamp()} [${e.level}] ${e.text} ${e.url || ''}`);
  }
});

const send = (method, params = {}) =>
  new Promise((resolve) => {
    const mid = ++id;
    pending.set(mid, resolve);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

await new Promise((r) => ws.addEventListener('open', r));

await send('Runtime.enable');
await send('Log.enable');
await send('Page.enable');
await send('Network.enable');
// Otherwise Chrome happily serves a stale stylesheet between iterations.
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440,
  height: 900,
  deviceScaleFactor: 1,
  mobile: false,
});

// Convenience helpers available to step scripts.
const PRELUDE = `
  globalThis.ovBtn = () => { const b = document.getElementById('ov-btn'); if (b && !document.getElementById('overlay').classList.contains('hidden')) { b.click(); return 'clicked ' + b.textContent; } return 'overlay hidden'; };
  globalThis.lock = () => { document.getElementById('btn-lock').click(); return 'locked'; };
  globalThis.fill = (which) => globalThis.__fill ? globalThis.__fill(which) : 'no fill hook';
  'ok'
`;

await send('Page.navigate', { url });
await sleep(1400);
await send('Runtime.evaluate', { expression: PRELUDE });

for (const step of steps) {
  if (/^\d+$/.test(step)) {
    await sleep(Number(step));
    continue;
  }
  const expression = step.startsWith('@') ? readFileSync(step.slice(1), 'utf8') : step;
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  const val = res.result?.result;
  const err = res.result?.exceptionDetails;
  console.log(`  step ${JSON.stringify(step.slice(0, 60))} -> ${err ? 'ERROR ' + (err.exception?.description || err.text) : JSON.stringify(val?.value)}`);
}

const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log(`wrote ${out}`);
} else {
  console.log('screenshot failed', JSON.stringify(shot).slice(0, 300));
}

// Leave the tab idle: a live WebGL page keeps a core busy long after we stop looking.
if (!process.env.KEEP) await send('Page.navigate', { url: 'about:blank' });

console.log('--- console ---');
if (logs.length === 0) console.log('(clean)');
for (const l of logs.slice(0, 40)) console.log(l);

ws.close();
