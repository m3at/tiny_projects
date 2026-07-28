// The menu and the lobby: choosing how to play, and waiting for the other captains.
//
// Both build a node that goes inside the overlay card, so there is one full-screen panel in the game
// and not three. Neither knows anything about the room beyond what it is handed.

import { MAX_PLAYERS } from '../config.js';
import { CODE_LENGTH, cleanName } from '../net/protocol.js';
import { $, setOverlayButton } from './hud.js';

const NAME_KEY = 'broadside.name';

export function savedName() {
  try {
    return localStorage.getItem(NAME_KEY) || 'Captain';
  } catch {
    // Private browsing, or storage turned off. A default name is not worth an error.
    return 'Captain';
  }
}

function remember(name) {
  try {
    localStorage.setItem(NAME_KEY, name);
  } catch {
    /* nothing to do about it */
  }
}

function button(label, sub) {
  const btn = document.createElement('button');
  btn.className = 'tool';
  btn.innerHTML = `${label}${sub ? `<span class="sub">${sub}</span>` : ''}`;
  return btn;
}

function field(label, { value = '', placeholder = '', cls = '', max = 24 } = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  const id = `f-${label.replace(/\W/g, '')}`;
  wrap.innerHTML = `<label for="${id}">${label}</label>`;
  const input = document.createElement('input');
  input.id = id;
  input.value = value;
  input.placeholder = placeholder;
  input.maxLength = max;
  if (cls) input.className = cls;
  wrap.appendChild(input);
  return { wrap, input };
}

// A row of numbers to pick from, which is a cleaner control than a stepper for a range of three.
function chooser(label, values, initial, onPick) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  wrap.innerHTML = `<label>${label}</label>`;
  const row = document.createElement('div');
  row.className = 'ov-buttons';
  row.style.justifyContent = 'flex-start';
  const buttons = new Map();
  for (const value of values) {
    const btn = document.createElement('button');
    btn.className = 'tool';
    btn.textContent = String(value);
    btn.style.minWidth = '38px';
    btn.onclick = () => {
      select(value);
      onPick(value);
    };
    buttons.set(value, btn);
    row.appendChild(btn);
  }
  function select(value) {
    for (const [v, btn] of buttons) btn.classList.toggle('selected', v === value);
  }
  function enable(allowed) {
    for (const [v, btn] of buttons) btn.disabled = !allowed.includes(v);
  }
  select(initial);
  wrap.appendChild(row);
  return { wrap, select, enable };
}

export function buildMenu({ onLocal, onCreate, onJoin }) {
  const node = document.createElement('div');

  const modes = document.createElement('div');
  modes.className = 'menu-modes';
  const bLocal = button('One keyboard', 'Two to four captains taking turns to build');
  const bCreate = button('Open a room', 'Play over the network. You get a code to share');
  const bJoin = button('Join a room', 'Somebody has given you a four-letter code');
  modes.append(bLocal, bCreate, bJoin);
  node.appendChild(modes);

  const extra = document.createElement('div');
  node.appendChild(extra);

  let mode = 'local';
  let captains = 2;
  let bots = 0;

  const nameField = field('Your name', { value: savedName(), max: 14 });
  const codeField = field('Room code', { placeholder: 'ABCD', cls: 'code', max: CODE_LENGTH });

  function render() {
    for (const [btn, name] of [
      [bLocal, 'local'],
      [bCreate, 'create'],
      [bJoin, 'join'],
    ]) {
      btn.classList.toggle('selected', mode === name);
    }
    extra.innerHTML = '';

    if (mode === 'local') {
      const row = document.createElement('div');
      row.className = 'field-row';
      // Captains and bots have to add up to four or fewer, and to at least two ships in the water.
      const botPick = chooser('Bots', [0, 1, 2], bots, (v) => {
        bots = v;
        sync();
      });
      const capPick = chooser('At this keyboard', [1, 2, 3, 4], captains, (v) => {
        captains = v;
        sync();
      });
      function sync() {
        capPick.enable([1, 2, 3, 4].filter((c) => c + bots <= MAX_PLAYERS && c + bots >= 2));
        botPick.enable([0, 1, 2].filter((b) => b + captains <= MAX_PLAYERS && b + captains >= 2));
        capPick.select(captains);
        botPick.select(bots);
        setOverlayButton({
          label: `Sail — ${captains + bots} ship${captains + bots > 1 ? 's' : ''}`,
        });
      }
      row.append(capPick.wrap, botPick.wrap);
      extra.appendChild(row);
      sync();
      return;
    }

    const row = document.createElement('div');
    row.className = 'field-row';
    row.appendChild(nameField.wrap);
    if (mode === 'join') row.appendChild(codeField.wrap);
    extra.appendChild(row);
    setOverlayButton({ label: mode === 'create' ? 'Open the room' : 'Join' });
  }

  bLocal.onclick = () => {
    mode = 'local';
    render();
  };
  bCreate.onclick = () => {
    mode = 'create';
    render();
    nameField.input.focus();
  };
  bJoin.onclick = () => {
    mode = 'join';
    render();
    codeField.input.focus();
  };

  const menu = {
    node,
    label: () => 'Sail — 2 ships',
    confirm() {
      if (mode === 'local') return onLocal(captains, bots);
      const name = cleanName(nameField.input.value);
      remember(name);
      if (mode === 'create') return onCreate(name);
      const code = codeField.input.value.trim().toUpperCase();
      if (code.length !== CODE_LENGTH) {
        codeField.input.focus();
        return;
      }
      onJoin(name, code);
    },
  };

  render();
  return menu;
}

export function buildLobby({ room, mySeats, onReady, onAddBot }) {
  const node = document.createElement('div');
  const me = mySeats[0];
  const mine = room.players.find((p) => p.seat === me) ?? null;
  const ready = !!(mine && mine.ready);

  if (room.code !== 'LOCAL') {
    const code = document.createElement('div');
    code.className = 'joincode';
    code.innerHTML = `<span class="muted">Room code</span><strong>${room.code}</strong>`;
    const copy = document.createElement('button');
    copy.className = 'tool';
    copy.textContent = 'Copy link';
    copy.onclick = async () => {
      const url = `${location.origin}${location.pathname}?dev=1&net=1&room=${room.code}`;
      try {
        await navigator.clipboard.writeText(url);
        copy.textContent = 'Copied';
      } catch {
        // Clipboard access needs a secure context and a permission. The code is on screen anyway.
        copy.textContent = 'Copy failed';
      }
    };
    code.appendChild(copy);
    node.appendChild(code);
  }

  const roster = document.createElement('div');
  roster.className = 'roster';
  for (let seat = 0; seat < MAX_PLAYERS; seat++) {
    const player = room.players.find((p) => p.seat === seat) ?? null;
    const row = document.createElement('div');
    row.className = `p${seat + 1}`;
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = player ? player.name : 'Empty berth';
    if (!player) who.style.color = 'var(--dim)';
    const state = document.createElement('span');
    state.className = 'state';
    if (!player) state.textContent = '';
    else if (player.bot) state.textContent = 'bot';
    else if (!player.connected) state.textContent = 'away';
    else if (player.ready) {
      state.textContent = 'ready';
      state.classList.add('on');
    } else state.textContent = 'fitting out';
    row.append(who, state);
    roster.appendChild(row);
  }
  node.appendChild(roster);

  if (onAddBot && room.players.length < MAX_PLAYERS) {
    const add = document.createElement('button');
    add.className = 'tool';
    add.style.marginTop = '10px';
    add.textContent = 'Add a bot';
    add.onclick = onAddBot;
    node.appendChild(add);
  }

  return {
    node,
    readyLabel: ready ? 'Not ready' : 'Ready',
    toggleReady() {
      onReady(!ready);
    },
  };
}
