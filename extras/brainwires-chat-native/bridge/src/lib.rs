//! uniffi FFI bridge for the brainwires-chat-native React Native app.
//!
//! Compile to:
//!   - `.so` for Android (cargo-ndk, JNI loader on the Kotlin side)
//!   - `.xcframework` for iOS / macOS (lipo + Swift package)
//!   - `.dll` for Windows (loaded by react-native-windows C++/WinRT shim)
//!   - linked directly into the Tauri Linux build (Rust-host, no FFI hop)
//!
//! Bindings are generated via the `uniffi-bindgen` binary in this crate
//! (see `npm run bridge:bindings` from the project root).

uniffi::setup_scaffolding!();

/// Returns the framework version string. Smoke-test for the FFI loop —
/// call this from the JS side as soon as bindings are linked.
#[uniffi::export]
pub fn framework_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
