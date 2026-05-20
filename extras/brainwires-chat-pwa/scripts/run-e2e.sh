#!/usr/bin/env bash
# End-to-end harness driver:
#   1. Asserts the wasm bundle is built (web/pkg/*.wasm fresh).
#   2. Boots nginx in user mode (port 8090) in the background.
#   3. Runs the Playwright spec(s) against http://localhost:8090.
#   4. Tears nginx down on exit (success or failure).

set -euo pipefail

PWA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$PWA_DIR/web"
RUN_DIR="$PWA_DIR/.nginx-run"
NGINX_LOG="$RUN_DIR/nginx.start.log"

if [[ ! -f "$WEB_DIR/pkg/brainwires_chat_pwa_bg.wasm" ]]; then
    echo "error: web/pkg/ missing — run extras/brainwires-chat-pwa/web/build.sh first" >&2
    exit 1
fi

cleanup() {
    if [[ -f "$RUN_DIR/nginx.pid" ]]; then
        local pid
        pid=$(cat "$RUN_DIR/nginx.pid" 2>/dev/null || true)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "$pid" 2>/dev/null || true
        fi
        rm -f "$RUN_DIR/nginx.pid"
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$RUN_DIR"

# Sanity-check the rendered config first (catches typos without
# eating the real run).
"$PWA_DIR/scripts/nginx-local.sh" --check >"$NGINX_LOG" 2>&1

# Start nginx in the background. The wrapper has `daemon off;` baked
# into the conf, so we explicitly background here.
"$PWA_DIR/scripts/nginx-local.sh" >>"$NGINX_LOG" 2>&1 &
NGINX_BG=$!

# Wait for the listener to come up (or the process to die).
for _ in $(seq 1 30); do
    if curl -fsS -o /dev/null http://localhost:8090/index.html; then
        break
    fi
    if ! kill -0 "$NGINX_BG" 2>/dev/null; then
        echo "error: nginx exited before becoming ready — see $NGINX_LOG" >&2
        cat "$NGINX_LOG" >&2
        exit 1
    fi
    sleep 0.5
done
if ! curl -fsS -o /dev/null http://localhost:8090/index.html; then
    echo "error: nginx did not respond on :8090 after 15s" >&2
    cat "$NGINX_LOG" >&2
    exit 1
fi

echo "==> nginx ready on http://localhost:8090/ — running Playwright"
cd "$WEB_DIR"
exec npx playwright test \
    --config=tests/e2e/playwright.config.mjs \
    --reporter=list \
    "$@"
