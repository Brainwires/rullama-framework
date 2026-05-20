#!/usr/bin/env bash
# brainwires-chat-pwa launcher — daemon-mode for both prod and dev.
#
# Usage:
#   ./web/start.sh                  # production (default)
#   ./web/start.sh prod
#   ./web/start.sh dev              # live-edit via bind-mount overlay
#   ./web/start.sh stop             # stop everything
#   ./web/start.sh status           # show running state
#   ./web/start.sh logs [WHICH]     # tail logs
#                                   #   WHICH = esbuild | cargo | container | all
#
# Each start always shuts down any existing instance first (containers +
# host watchers), so switching between prod and dev is seamless and
# idempotent.
#
# prod  → docker compose up -d --build
#         (just docker-compose.yml; no bind mount)
#
# dev   → docker compose -f docker-compose.yml -f docker-compose.dev.yml
#         up -d --build
#         The overlay bind-mounts ./web → /usr/share/nginx/html (ro), so
#         host-side edits to the bundled output reflect in the running
#         container immediately. Plus two background loops on the host:
#           1. esbuild --watch          (web/src → web/app.js, web/sw.js)
#           2. cargo-watch + wasm-pack  (wasm/   → web/pkg/)
#         With DEV_MODE=true, boot.js unregisters the service worker and
#         clears bw-chat-cache-v1; bw-models-v1 is preserved.
#
# State (PIDs + log files) lives in web/.run/ (gitignored). Every
# start command returns control to the shell as soon as the loops are
# launched.

set -euo pipefail

cd "$(dirname "$0")"
WEB_DIR="$(pwd)"
cd ..
PWA_DIR="$(pwd)"
WASM_CRATE_DIR="$PWA_DIR/wasm"
RUN_DIR="$WEB_DIR/.run"
COMPOSE_BASE="docker-compose.yml"
COMPOSE_DEV="docker-compose.dev.yml"

CMD="${1:-prod}"

# ── Helpers ────────────────────────────────────────────────────────────

# Compose invocation for the recorded mode (or current request).
# $1 = mode (prod|dev). Echoes the `-f` flag chain.
compose_files_for() {
    case "$1" in
        dev) echo "-f $COMPOSE_BASE -f $COMPOSE_DEV" ;;
        *)   echo "-f $COMPOSE_BASE" ;;
    esac
}

stop_pid_file() {
    local file=$1
    [ -f "$file" ] || return 0
    local pid
    pid=$(cat "$file" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
}

stop_all() {
    if [ -d "$RUN_DIR" ]; then
        for f in "$RUN_DIR"/*.pid; do
            [ -f "$f" ] || continue
            stop_pid_file "$f"
        done
    fi
    # Use the recorded mode's compose files if present; fall back to dev
    # overlay (a superset that also matches prod-only containers).
    local prior_mode="prod"
    if [ -f "$RUN_DIR/mode" ]; then
        prior_mode=$(cat "$RUN_DIR/mode")
    fi
    # shellcheck disable=SC2086
    ( cd "$PWA_DIR" && docker compose $(compose_files_for "$prior_mode") \
        down --remove-orphans ) >/dev/null 2>&1 || true
}

write_pid() {
    local name=$1 pid=$2
    mkdir -p "$RUN_DIR"
    echo "$pid" > "$RUN_DIR/$name.pid"
}

# ── Subcommands ────────────────────────────────────────────────────────

start_prod() {
    echo "==> stopping any existing chat-pwa instance"
    stop_all

    mkdir -p "$RUN_DIR"
    echo "prod" > "$RUN_DIR/mode"

    echo "==> starting chat-pwa (production, detached)"
    cd "$PWA_DIR"
    DEV_MODE=false docker compose -f "$COMPOSE_BASE" up -d --build

    echo
    echo "Containers detached. Open http://localhost:${HOST_PORT:-8080}"
    echo "  status:  ./web/start.sh status"
    echo "  logs:    ./web/start.sh logs"
    echo "  stop:    ./web/start.sh stop"
}

start_dev() {
    # Pre-flight (dev only)
    command -v wasm-pack >/dev/null \
        || { echo "wasm-pack missing — cargo install wasm-pack" >&2; exit 1; }
    command -v cargo-watch >/dev/null \
        || { echo "cargo-watch missing — cargo install cargo-watch --locked" >&2; exit 1; }
    if [ ! -d "$WEB_DIR/node_modules" ]; then
        echo "==> npm install"
        ( cd "$WEB_DIR" && npm install )
    fi

    echo "==> stopping any existing chat-pwa instance"
    stop_all

    mkdir -p "$RUN_DIR"
    echo "dev" > "$RUN_DIR/mode"

    echo "==> starting chat-pwa (dev, bind-mount overlay, detached)"

    # Watcher 1: esbuild — `setsid` detaches the subshell from the
    # controlling terminal so it survives parent-shell exit (the whole
    # point of daemon-mode dev). `< /dev/null` closes stdin so the
    # detached process doesn't compete for the tty.
    setsid bash -c "cd \"$WEB_DIR\" && exec node build.mjs --watch" \
        </dev/null >"$RUN_DIR/esbuild.log" 2>&1 &
    write_pid esbuild "$!"

    # Watcher 2: cargo-watch + wasm-pack — mirrors web/build.sh.
    setsid bash -c "exec cargo watch \
        --workdir \"$WASM_CRATE_DIR\" \
        -w \"$WASM_CRATE_DIR/src\" \
        -w \"$WASM_CRATE_DIR/Cargo.toml\" \
        -s 'wasm-pack build --target web --release --out-dir \"$WEB_DIR/pkg\" --out-name brainwires_chat_pwa \"$WASM_CRATE_DIR\"'" \
        </dev/null >"$RUN_DIR/cargo-watch.log" 2>&1 &
    write_pid cargo-watch "$!"

    # Containers — proper compose-managed daemon via the dev overlay.
    cd "$PWA_DIR"
    DEV_MODE=true docker compose -f "$COMPOSE_BASE" -f "$COMPOSE_DEV" up -d --build

    echo
    echo "All loops detached. State in: $RUN_DIR"
    echo "  open:    http://localhost:${HOST_PORT:-8080}"
    echo "  status:  ./web/start.sh status"
    echo "  logs:    ./web/start.sh logs        (combined)"
    echo "           ./web/start.sh logs esbuild | cargo | container"
    echo "  stop:    ./web/start.sh stop"
}

cmd_stop() {
    echo "==> stopping chat-pwa"
    stop_all
    rm -rf "$RUN_DIR"
    echo "Stopped."
}

cmd_status() {
    if [ ! -d "$RUN_DIR" ] || [ ! -f "$RUN_DIR/mode" ]; then
        echo "chat-pwa: not running"
        return 0
    fi
    local mode
    mode=$(cat "$RUN_DIR/mode")
    echo "chat-pwa: $mode"
    if [ "$mode" = "dev" ]; then
        for name in esbuild cargo-watch; do
            local f="$RUN_DIR/$name.pid"
            [ -f "$f" ] || continue
            local pid
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                printf '  %-15s pid %-7s running\n' "$name" "$pid"
            else
                printf '  %-15s pid %-7s DEAD\n'    "$name" "$pid"
            fi
        done
    fi
    echo
    # shellcheck disable=SC2086
    ( cd "$PWA_DIR" && docker compose $(compose_files_for "$mode") ps )
}

cmd_logs() {
    local what="${1:-all}"
    local mode="prod"
    [ -f "$RUN_DIR/mode" ] && mode=$(cat "$RUN_DIR/mode")

    case "$what" in
        esbuild)
            [ -f "$RUN_DIR/esbuild.log" ] || { echo "no esbuild log (is dev running?)" >&2; exit 1; }
            exec tail -F "$RUN_DIR/esbuild.log"
            ;;
        cargo|cargo-watch|wasm)
            [ -f "$RUN_DIR/cargo-watch.log" ] || { echo "no cargo-watch log (is dev running?)" >&2; exit 1; }
            exec tail -F "$RUN_DIR/cargo-watch.log"
            ;;
        container|cnt|nginx)
            cd "$PWA_DIR"
            # shellcheck disable=SC2086
            exec docker compose $(compose_files_for "$mode") logs -f
            ;;
        all|*)
            local logs=()
            if [ -d "$RUN_DIR" ]; then
                for f in "$RUN_DIR"/*.log; do
                    [ -f "$f" ] && logs+=("$f")
                done
            fi
            if [ ${#logs[@]} -gt 0 ]; then
                # Tail host-watcher logs alongside container logs by
                # backgrounding compose-logs and tailing files in
                # foreground; the trap cleans up compose-logs on Ctrl-C.
                cd "$PWA_DIR"
                # shellcheck disable=SC2086
                docker compose $(compose_files_for "$mode") logs -f --no-color &
                local logs_pid=$!
                trap 'kill '"$logs_pid"' 2>/dev/null || true' INT TERM EXIT
                tail -F "${logs[@]}"
            else
                cd "$PWA_DIR"
                # shellcheck disable=SC2086
                exec docker compose $(compose_files_for "$mode") logs -f
            fi
            ;;
    esac
}

print_help() {
    cat <<USAGE
Usage: $(basename "$0") <command> [args]

Commands:
  prod (default)      Start in production mode (containers detached).
  dev                 Start in dev mode with bind-mounted live editing
                      (containers detached + host watchers backgrounded).
  stop                Stop everything (containers + host watchers).
  status              Show running state.
  logs [WHICH]        Tail logs.
                      WHICH = esbuild | cargo | container | all (default).
  -h, --help, help    This message.

Each start command always shuts down any existing instance first, so
switching between prod and dev is seamless.
USAGE
}

# ── Dispatch ───────────────────────────────────────────────────────────

case "$CMD" in
    prod|production)    start_prod ;;
    dev|development)    start_dev ;;
    stop|down)          cmd_stop ;;
    status|ps)          cmd_status ;;
    logs|log)           cmd_logs "${2:-all}" ;;
    -h|--help|help)     print_help ;;
    *)
        echo "Unknown command: $CMD (try --help)" >&2
        exit 1
        ;;
esac
