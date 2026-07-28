// Talking to headless Chrome. Node's built-in WebSocket speaks the DevTools protocol directly, so
// there is no dependency here either.
//
// Five tools drive the real browser -- shot, audio, frames, profile, fill -- and each of them had
// grown its own copy of the same forty lines: find the page target, open the socket, match replies
// to requests by id, disable the cache, wait, tear down. They also had three separate copies of
// "click through the overlays until the game reaches a phase". This is that code, once.
//
// Two things every caller wants and half of them had forgotten:
//
//   The cache is off. Chrome will otherwise serve the previous run's copy of a module, which shows
//   up as a sound that still has its old level, or a function that does not exist yet.
//
//   The page is left on about:blank. A live WebGL page keeps rendering after the tool exits and
//   will quietly hold a core busy until someone notices.

const PORT = 9222;
const BASE = process.env.URL || 'http://127.0.0.1:8123/index.html';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function attach() {
  const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json().catch(() => null);
  const target = targets?.find((t) => t.type === 'page');
  if (!target) {
    console.error('no page target on port 9222; run ./tools/dev.sh first');
    process.exit(1);
  }
  return connect(target);
}

// A second, third and fourth browser, for anything that needs more than one player. Each tab is a
// separate page with its own storage and its own socket, which is what makes it a real second client
// rather than a second view of the first one.
//
// Chrome's /json/new needs PUT and rejects GET, which is a change from older versions and the reason
// this is one line rather than a fetch anybody would have written.
export async function openTab(url = 'about:blank') {
  const res = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`, {
    method: 'PUT',
  });
  if (!res.ok) {
    console.error(`could not open a tab (${res.status}); run ./tools/dev.sh first`);
    process.exit(1);
  }
  const target = await res.json();
  const page = await connect(target);
  page.targetId = target.id;
  return page;
}

export async function closeTab(page) {
  if (!page.targetId) return;
  await fetch(`http://127.0.0.1:${PORT}/json/close/${page.targetId}`).catch(() => {});
}

async function connect(target) {
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));

  let nextId = 0;
  const pending = new Map();
  const logs = [];
  const started = Date.now();

  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
      return;
    }
    const at = String(Date.now() - started).padStart(5) + 'ms';
    if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? a.type).join(' ');
      logs.push({ at, level: msg.params.type, text });
    } else if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      logs.push({ at, level: 'exception', text: d.exception?.description || d.text });
    } else if (msg.method === 'Log.entryAdded') {
      const e = msg.params.entry;
      if (e.level === 'error' || e.level === 'warning') {
        logs.push({ at, level: e.level, text: `${e.text} ${e.url || ''}`.trim() });
      }
    }
  });

  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = ++nextId;
      pending.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params }));
    });

  // Returns the value, or throws with the page's own stack. Tools that would rather see the error
  // than die can pass { soft: true } and get the description back as a string.
  const evalIn = async (expression, { soft = false } = {}) => {
    const r = await send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    const bad = r.result?.exceptionDetails;
    if (bad) {
      const text = bad.exception?.description || bad.text || 'evaluation failed';
      if (soft) return `THREW: ${text}`;
      throw new Error(text);
    }
    return r.result?.result?.value;
  };

  // Wrapped in an async IIFE with an await, so this works whether the expression is a value or a
  // promise. Without the await, an async expression stringifies as "{}" -- a pending promise -- and
  // the caller gets a plausible-looking empty object rather than an error.
  const json = async (expression, opts) =>
    JSON.parse(await evalIn(`(async () => JSON.stringify(await (${expression})))()`, opts));

  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Network.enable');
  await send('Network.setCacheDisabled', { cacheDisabled: true });

  return {
    send,
    evalIn,
    json,
    logs,

    // Device metrics have to be applied *after* the page has loaded. Setting them before navigating
    // stops the game initialising at all, and the symptom is a screenshot of a correct HUD over an
    // empty canvas, which reads convincingly as a rendering bug.
    async resize(width, height) {
      await send('Emulation.setDeviceMetricsOverride', {
        width,
        height,
        deviceScaleFactor: 1,
        mobile: false,
      });
    },

    async open(query = '', settle = 2200) {
      await send('Page.navigate', { url: BASE + query });
      await sleep(settle);
    },

    // Dev autoplay dismisses its own overlays, but the opening menu waits for a person, and a match
    // that has already ended waits on the result screen.
    async reachPhase(phase, limitMs = 40000) {
      const deadline = Date.now() + limitMs;
      while (Date.now() < deadline) {
        if ((await evalIn('__dev.state().phase', { soft: true })) === phase) return true;
        await evalIn(
          `(() => { const o = document.getElementById('overlay'), b = document.getElementById('ov-btn');
             if (o && !o.classList.contains('hidden') && b) b.click(); })()`,
          { soft: true },
        );
        await sleep(300);
      }
      return false;
    },

    async screenshot(path) {
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      if (!shot.result?.data) return false;
      const { writeFileSync } = await import('node:fs');
      writeFileSync(path, Buffer.from(shot.result.data, 'base64'));
      return true;
    },

    printLogs(limit = 40) {
      // The GPU driver complains about readPixels stalls, which is a performance note about
      // something these tools do deliberately to force the pipeline to drain. It is not a fault in
      // the game and it drowns out anything that is.
      const real = logs.filter((l) => !/GL Driver Message .*Performance/.test(l.text));
      if (real.length === 0) return console.log('  console: clean');
      console.log('  console:');
      for (const l of real.slice(0, limit)) console.log(`    ${l.at} [${l.level}] ${l.text}`);
    },

    async close({ keep = false } = {}) {
      if (!keep) await send('Page.navigate', { url: 'about:blank' });
      ws.close();
    },
  };
}
