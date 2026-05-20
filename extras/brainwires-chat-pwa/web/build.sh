#!/usr/bin/env bash
# Build pipeline for brainwires-chat-pwa.
#
# 1. Compile wasm/ to wasm32 via wasm-pack → web/pkg/
# 2. Bundle JS via build.mjs (esbuild) and patch sw.js with SRI hashes.
#
# Run from anywhere; the script chdir's into its own directory first.
set -euo pipefail

cd "$(dirname "$0")"
WEB_DIR="$(pwd)"
WASM_CRATE_DIR="$WEB_DIR/../wasm"

if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "error: wasm-pack not found. Install with:" >&2
    echo "  cargo install wasm-pack --locked" >&2
    exit 1
fi

# web_sys_unstable_apis enables web-sys's WebGPU bindings (navigator.gpu etc.)
export RUSTFLAGS="${RUSTFLAGS:-} --cfg=web_sys_unstable_apis"

# NOTE: --release is required here for *functional correctness*, not
# perf. Candle's quantized + WGPU paths use u32 bit-arithmetic that
# wraps cleanly in release but TRAPS under Rust's debug overflow
# checks ("Uncaught RuntimeError: unreachable" the moment the model
# loads). --dev wasm physically cannot run the model. This is the
# ONE exception to the no-`--release` rule for this project — see
# memory/feedback_no_release_builds.md.
echo "==> wasm-pack build $WASM_CRATE_DIR"
wasm-pack build \
    --target web \
    --release \
    --out-dir "$WEB_DIR/pkg" \
    --out-name brainwires_chat_pwa \
    "$WASM_CRATE_DIR"

# wasm-pack drops a few files we don't want shipped with the static bundle.
rm -f "$WEB_DIR/pkg/.gitignore" \
      "$WEB_DIR/pkg/package.json" \
      "$WEB_DIR/pkg/README.md"

echo "==> bundling JS via esbuild"
node "$WEB_DIR/build.mjs"

echo ""
echo "build complete:"
echo "  wasm:   $WEB_DIR/pkg/brainwires_chat_pwa_bg.wasm"
echo "  js:     $WEB_DIR/app.js"
echo "  worker: $WEB_DIR/local-worker.js"
echo "  sw:     $WEB_DIR/sw.js"
