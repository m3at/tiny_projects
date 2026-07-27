#!/usr/bin/env bash
# Starts a static server and a headless Chrome with CDP open, for tools/shot.js.
# Usage:  ./tools/dev.sh [port]      start
#         ./tools/dev.sh stop        tear both down
#         open http://127.0.0.1:8123/index.html
set -euo pipefail

if [ "${1:-}" = "stop" ]; then
  pkill -f "remote-debugging-port=9222" 2>/dev/null || true
  pkill -f "http.server" 2>/dev/null || true
  echo "stopped server and headless chrome"
  exit 0
fi

PORT="${1:-8123}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${TMPDIR:-/tmp}/broadside-chrome"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME="$(command -v chromium || command -v google-chrome)"

pkill -f "http.server $PORT" 2>/dev/null || true
nohup python3 -m http.server "$PORT" --directory "$ROOT" >/dev/null 2>&1 </dev/null &
disown || true
sleep 1

if ! curl -s "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then
  nohup "$CHROME" --headless=new --remote-debugging-port=9222 --disable-gpu \
    --use-gl=swiftshader --enable-unsafe-swiftshader --no-first-run \
    --user-data-dir="$PROFILE" about:blank >/dev/null 2>&1 </dev/null &
  disown || true
  sleep 3
fi

cat <<EOF
serving  http://127.0.0.1:$PORT/index.html
play     http://127.0.0.1:$PORT/index.html
watch    http://127.0.0.1:$PORT/index.html?dev=brawler,sniper
devtools CDP on 9222

  node tools/balance.js          archetype matchups per hull
  node tools/match.js 40         full 5-round matches, economy and fill rates
  node tools/events.js           are detonations/severings/dismastings firing
  node tools/shot.js out.png "1500 ;; ovBtn() ;; 800"
EOF
