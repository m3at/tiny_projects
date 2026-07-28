#!/usr/bin/env bash
# Starts the game server and a headless Chrome with CDP open, for tools/shot.js and friends.
# Usage:  ./tools/dev.sh [port]      start
#         ./tools/dev.sh stop        tear both down
#         open http://127.0.0.1:8123/index.html
#
# The server is server/main.js: it serves the directory as it stands and hosts the WebSocket rooms
# on the same port, so an online game needs nothing else running. It replaced python -m http.server,
# which could only do the first half.
set -euo pipefail

if [ "${1:-}" = "stop" ]; then
  pkill -f "remote-debugging-port=9222" 2>/dev/null || true
  pkill -f "server/main.js" 2>/dev/null || true
  pkill -f "http.server" 2>/dev/null || true
  echo "stopped server and headless chrome"
  exit 0
fi

PORT="${1:-8123}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${TMPDIR:-/tmp}/broadside-chrome"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME="$(command -v chromium || command -v google-chrome)"

pkill -f "server/main.js" 2>/dev/null || true
sleep 0.3
nohup node "$ROOT/server/main.js" "$PORT" >"${TMPDIR:-/tmp}/broadside-server.log" 2>&1 </dev/null &
disown || true

# Wait for it rather than sleeping a fixed second: the tools that follow will fail confusingly if
# the first request lands before the listener is up.
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.1
done

if ! curl -s "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then
  # The backgrounding flags matter for tools/netplay.js, which drives up to four tabs at once. Only
  # one tab can be foreground, and Chrome throttles requestAnimationFrame in the others to nothing --
  # so three of the four clients simply stop replaying the battle and report no checksums, which looks
  # exactly like the sync stream being broken.
  nohup "$CHROME" --headless=new --remote-debugging-port=9222 --disable-gpu \
    --use-gl=swiftshader --enable-unsafe-swiftshader --no-first-run \
    --disable-backgrounding-occluded-windows --disable-renderer-backgrounding \
    --disable-background-timer-throttling --disable-ipc-flooding-protection \
    --user-data-dir="$PROFILE" about:blank >/dev/null 2>&1 </dev/null &
  disown || true
  sleep 3
fi

cat <<EOF
serving  http://127.0.0.1:$PORT/index.html
health   $(curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 'NOT RESPONDING')
watch    http://127.0.0.1:$PORT/index.html?dev=brawler,sniper
melee    http://127.0.0.1:$PORT/index.html?dev=draft&players=4
online   http://127.0.0.1:$PORT/index.html?dev=1&net=1
devtools CDP on 9222
log      ${TMPDIR:-/tmp}/broadside-server.log

  node tools/netcheck.js         the authority and the replay, headless
  node tools/netplay.js          two real browsers through a whole online match
  node tools/melee.js            three and four ships: length, seat fairness, builds
  node tools/playtest.js         a local match through the real interface
  node tools/shot.js out.png "1500 ;; ovBtn() ;; 800"
EOF
