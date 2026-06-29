#!/usr/bin/env bash
# Build the Rust FFI cdylib and generate C# bindings via uniffi-bindgen-cs.
#
# Prerequisites:
#   cargo install uniffi-bindgen-cs --git https://github.com/aspect-build/uniffi-bindgen-cs
#
# Usage:
#   ./build-native.sh          # Debug build
#   ./build-native.sh release  # Release build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FFI_CRATE="audio-demo-ffi"
LIB_NAME="libaudio_demo_ffi"

PROFILE="${1:-debug}"
if [ "$PROFILE" = "release" ]; then
    CARGO_FLAGS="--release"
    TARGET_DIR="$WORKSPACE_ROOT/target/release"
else
    CARGO_FLAGS=""
    TARGET_DIR="$WORKSPACE_ROOT/target/debug"
fi

echo "==> Building $FFI_CRATE ($PROFILE)..."
cargo build -p "$FFI_CRATE" $CARGO_FLAGS

# Determine native library filename
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_FILE="$LIB_NAME.so"
    RUNTIME_DIR="linux-x64"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_FILE="$LIB_NAME.dylib"
    RUNTIME_DIR="osx-arm64"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    LIB_FILE="audio_demo_ffi.dll"
    RUNTIME_DIR="win-x64"
else
    echo "Unknown OS: $OSTYPE"
    exit 1
fi

NATIVE_LIB="$TARGET_DIR/$LIB_FILE"
if [ ! -f "$NATIVE_LIB" ]; then
    echo "ERROR: $NATIVE_LIB not found"
    exit 1
fi

echo "==> Generating C# bindings..."
GENERATED_DIR="$SCRIPT_DIR/BrainwiresAudio/Generated"
mkdir -p "$GENERATED_DIR"

if command -v uniffi-bindgen-cs &>/dev/null; then
    uniffi-bindgen-cs "$NATIVE_LIB" --out-dir "$GENERATED_DIR"
    echo "    -> $GENERATED_DIR/audio_demo_ffi.cs"
else
    echo "WARNING: uniffi-bindgen-cs not found. Skipping C# binding generation."
    echo "         Install: cargo install uniffi-bindgen-cs --git https://github.com/aspect-build/uniffi-bindgen-cs"
fi

echo "==> Copying native library to runtimes..."
DEST="$SCRIPT_DIR/BrainwiresAudio/runtimes/$RUNTIME_DIR/native"
mkdir -p "$DEST"
cp "$NATIVE_LIB" "$DEST/"
echo "    -> $DEST/$LIB_FILE"

echo "==> Done! Run 'dotnet build AudioDemo.sln' to build the Avalonia app."
