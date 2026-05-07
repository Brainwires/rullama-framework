#!/usr/bin/env bash
# Run nginx in user mode against the local PWA build.
#
# Usage:
#   ./scripts/nginx-local.sh           # foreground, ^C to stop
#   ./scripts/nginx-local.sh --check   # validate config only, exit
#
# The Playwright e2e harness expects the PWA on http://localhost:8080
# with /ollama-registry/v2/library/gemma4/{manifests/e2b,blobs/...}
# served from ~/.ollama/models/. No sudo, no privileged ports, no
# system files touched.

set -euo pipefail

PWA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$PWA_DIR/web"
RUN_DIR="$PWA_DIR/.nginx-run"
CONF_TEMPLATE="$PWA_DIR/nginx.local.conf"
CONF_RENDERED="$RUN_DIR/nginx.conf"

NGINX_BIN="${NGINX_BIN:-/usr/local/bin/nginx}"
OLLAMA_CACHE="${NGINX_OLLAMA_CACHE:-$HOME/.ollama/models}"

# nginx ships its mime.types alongside the binary; locate it.
if [[ -f /usr/local/etc/nginx/mime.types ]]; then
    MIME_TYPES=/usr/local/etc/nginx/mime.types
elif [[ -f /opt/homebrew/etc/nginx/mime.types ]]; then
    MIME_TYPES=/opt/homebrew/etc/nginx/mime.types
elif [[ -f /etc/nginx/mime.types ]]; then
    MIME_TYPES=/etc/nginx/mime.types
else
    echo "error: cannot find nginx mime.types — set NGINX_MIME_TYPES" >&2
    exit 1
fi
MIME_TYPES="${NGINX_MIME_TYPES:-$MIME_TYPES}"

if [[ ! -x "$NGINX_BIN" ]]; then
    echo "error: nginx binary not at $NGINX_BIN — set NGINX_BIN" >&2
    exit 1
fi

if [[ ! -d "$WEB_DIR/pkg" ]]; then
    echo "error: web/pkg/ not built. Run extras/brainwires-chat-pwa/web/build.sh first." >&2
    exit 1
fi

if [[ ! -d "$OLLAMA_CACHE" ]]; then
    echo "warn: ollama cache dir not found at $OLLAMA_CACHE — model fetch will 404." >&2
fi

mkdir -p \
    "$RUN_DIR" \
    "$RUN_DIR/client_body_temp" \
    "$RUN_DIR/proxy_temp" \
    "$RUN_DIR/fastcgi_temp" \
    "$RUN_DIR/uwsgi_temp" \
    "$RUN_DIR/scgi_temp"

# Render the template. Awk-based substitution keeps the script free of
# bash-isms that would trip up paths with spaces.
sed \
    -e "s|{{NGINX_RUN_DIR}}|$RUN_DIR|g" \
    -e "s|{{NGINX_PWA_ROOT}}|$WEB_DIR|g" \
    -e "s|{{NGINX_OLLAMA_CACHE}}|$OLLAMA_CACHE|g" \
    -e "s|{{NGINX_MIME_TYPES}}|$MIME_TYPES|g" \
    "$CONF_TEMPLATE" > "$CONF_RENDERED"

if [[ "${1:-}" == "--check" ]]; then
    "$NGINX_BIN" -t -c "$CONF_RENDERED" -p "$RUN_DIR"
    exit $?
fi

echo "==> nginx -c $CONF_RENDERED"
echo "==> root: $WEB_DIR"
echo "==> ollama cache: $OLLAMA_CACHE"
echo "==> http://localhost:8080/  (^C to stop)"
exec "$NGINX_BIN" -c "$CONF_RENDERED" -p "$RUN_DIR"
