#!/usr/bin/env bash
# bench-on-ios.sh — build, deploy, run rullama bench on the connected iPhone.
#
# End-to-end flow:
#   1. cargo build the static lib for aarch64-apple-ios
#   2. xcodebuild the Swift app, sign automatically with the personal team
#   3. install the .app on the device via xcrun devicectl
#   4. launch the app + stream stdout (via pymobiledevice3 syslog)
#
# Prereqs (one-time):
#   rustup target add aarch64-apple-ios
#   ~/.local/venvs/pymd3/bin/pymobiledevice3 (for syslog capture)
#   ideviceimagemounter auto-mount has been run (DDI on device)
#   Device in Developer Mode, paired & trusted with this Mac

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
BUNDLE_ID="com.nightness.rullamabench"
APP_NAME="BenchApp"
PYMD3="${PYMD3:-$HOME/.local/venvs/pymd3/bin/pymobiledevice3}"

log() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }

UDID="$(xcrun devicectl list devices 2>/dev/null \
    | awk '/connected[[:space:]]/ {print $(NF-2); exit}')"
if [[ -z "${UDID}" ]]; then
    echo "ERROR: no connected device found. Run: xcrun devicectl list devices" >&2
    exit 1
fi
log "device UDID = ${UDID}"

# ---- 1. Rust static lib ----
log "cargo build aarch64-apple-ios release"
cargo build --release \
    --target aarch64-apple-ios \
    --manifest-path "$ROOT/Cargo.toml"

# ---- 2. Xcode build + sign ----
log "xcodebuild BenchApp.app"
DERIVED="$ROOT/build-derived"
rm -rf "$DERIVED"
xcodebuild \
    -project "$ROOT/BenchApp.xcodeproj" \
    -scheme "$APP_NAME" \
    -configuration Release \
    -destination "generic/platform=iOS" \
    -derivedDataPath "$DERIVED" \
    -allowProvisioningUpdates \
    CODE_SIGN_STYLE=Automatic \
    DEVELOPMENT_TEAM=Y9UU9WUQD2 \
    build | tail -20

APP_PATH="$DERIVED/Build/Products/Release-iphoneos/${APP_NAME}.app"
if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: build did not produce ${APP_PATH}" >&2
    exit 1
fi
log "built ${APP_PATH}"

# ---- 3. install ----
log "installing on device"
xcrun devicectl device install app --device "$UDID" "$APP_PATH"

# ---- 4. launch + capture syslog ----
log "launching + capturing stdout"
if [[ ! -x "$PYMD3" ]]; then
    echo "WARN: pymobiledevice3 not found at $PYMD3 — skipping syslog capture" >&2
    xcrun devicectl device process launch --device "$UDID" "$BUNDLE_ID"
    echo "App launched; open Console.app to see output."
    exit 0
fi

# Background syslog stream filtered to our app's stdout.
"$PYMD3" syslog live --pid "$APP_NAME" 2>/dev/null &
SYSLOG_PID=$!
trap "kill $SYSLOG_PID 2>/dev/null || true" EXIT

sleep 1
xcrun devicectl device process launch --device "$UDID" "$BUNDLE_ID"

# Read stdout until the bench prints its done sentinel or times out.
sleep 30
kill $SYSLOG_PID 2>/dev/null || true
trap - EXIT
log "done"
