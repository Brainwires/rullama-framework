#!/usr/bin/env bash
# deploy.sh — Hot-swap production rebuild for brainwires-chat-pwa.
#
# Different beast from web/start.sh:
#
#   ./web/start.sh prod      Tears down → builds → brings up.
#                            Used for first-time bring-up. Down-during-build.
#   ./deploy.sh              Builds the new image WHILE the old container
#                            keeps serving, then swaps. Down only during
#                            the swap (~2–3s). Auto-rolls-back on health
#                            check failure.
#
# Strategy:
#   1. Tag the current `:latest` image as `:previous` for rollback.
#   2. `docker compose build` the new image (slow; old container still up).
#   3. `docker compose up -d --no-build` to swap (brief downtime).
#   4. Hit /manifest.json; if it 200s, success. Else roll back to :previous.
#
# Usage:
#   ./deploy.sh               # normal deploy (force prod mode)
#   ./deploy.sh --rollback    # restore :previous and bring it up
#
# This is a PROD-mode deploy. Forces DEV_MODE=false regardless of `.env`.
# For local live-edit dev, use ./web/start.sh dev instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"
SERVICE=chat-pwa
IMAGE=brainwires-chat-pwa
MAX_WAIT=30  # seconds — nginx + container restart should be fast

# Pull HOST_PORT from .env if present so the health check hits the
# actual published port. .env's defaults match docker-compose.yml's
# `${HOST_PORT:-8080}`. Don't error on a missing .env; the compose
# default of 8080 still applies.
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; . "$ENV_FILE"; set +a
fi
HOST_PORT="${HOST_PORT:-8080}"
HEALTH_URL="http://localhost:${HOST_PORT}/manifest.json"

# ── Colour helpers ────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ── Health check ──────────────────────────────────────────────────────
# Hitting /manifest.json verifies both that nginx is responding AND that
# the freshly built static assets are present. `/` would 200 even if the
# bundled JS or pkg/ output were missing.
wait_healthy() {
    yellow "Waiting for health check on ${HEALTH_URL} (up to ${MAX_WAIT}s)..."
    for i in $(seq 1 "$MAX_WAIT"); do
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            green "Healthy after ${i}s."
            return 0
        fi
        sleep 1
    done
    return 1
}

# ── Rollback ──────────────────────────────────────────────────────────
do_rollback() {
    yellow "Rolling back to previous image..."
    if docker image inspect "${IMAGE}:previous" >/dev/null 2>&1; then
        docker tag "${IMAGE}:previous" "${IMAGE}:latest"
        DEV_MODE=false docker compose -f "$COMPOSE_FILE" up -d --no-build "$SERVICE"
        if wait_healthy; then
            green "Rollback successful."
        else
            red "Rollback health check also failed. Manual intervention required."
            exit 1
        fi
    else
        red "No previous image found. Cannot roll back."
        exit 1
    fi
}

# ── Manual rollback flag ──────────────────────────────────────────────
if [ "${1:-}" = "--rollback" ]; then
    do_rollback
    exit 0
fi

# ── Pre-flight: don't trample a running dev session ───────────────────
DEV_MODE_FILE="$SCRIPT_DIR/web/.run/mode"
if [ -f "$DEV_MODE_FILE" ] && [ "$(cat "$DEV_MODE_FILE")" = "dev" ]; then
    red "Dev mode appears to be running (web/.run/mode=dev)."
    red "Stop it first:  ./web/start.sh stop"
    red "Otherwise this deploy will swap out the dev container."
    exit 1
fi

# ── Deploy ────────────────────────────────────────────────────────────

# Force prod-mode regardless of .env. DEV_MODE=true at build time would
# bake the wrong build-info.js into the image and disable the service
# worker for end users.
export DEV_MODE=false

# 1. Save current image as :previous (best-effort).
if docker image inspect "${IMAGE}:latest" >/dev/null 2>&1; then
    yellow "Tagging current image as ${IMAGE}:previous..."
    docker tag "${IMAGE}:latest" "${IMAGE}:previous"
fi

# 2. Build new image while old container keeps serving.
# `--no-cache` matches the reference deploy script — guarantees a clean
# build from current source. Costs ~3–5 extra minutes on this project
# because of the wasm-pack stage rebuilding candle. Drop the flag if
# you want layer-cache acceleration on routine deploys.
yellow "Building new image (old container keeps serving during build)..."
docker compose -f "$COMPOSE_FILE" build --no-cache "$SERVICE"

# 3. Swap containers (brief downtime starts here).
yellow "Swapping containers..."
docker compose -f "$COMPOSE_FILE" up -d --no-build "$SERVICE"

# 4. Wait for healthy; roll back on failure.
if wait_healthy; then
    green "Deploy complete."
    docker image prune -f >/dev/null 2>&1 || true
else
    red "New container failed health check."
    do_rollback
    exit 1
fi
