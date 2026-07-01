#![no_main]
//! Fuzz target stub for safetensors / GGUF model-header parsing.
//!
//! The framework's local-inference path (under `rullama-provider`'s
//! `llama-cpp-2` feature and any future safetensors loader in
//! `rullama-inference`) reads model files from disk. A malicious model
//! file must not be able to crash or pwn the loader.
//!
//! NOTE: this target is currently a stub. We deliberately do not wire it
//! to a specific in-tree loader yet because the framework's local-model
//! ingest is a thin shim over `llama-cpp-2` / external safetensors crates,
//! both of which have their own upstream fuzz suites. The harness target
//! will become meaningful when the framework grows its own header
//! validation logic; for now the stub asserts the byte-array entry-point
//! itself does not panic on arbitrary input.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Minimal probe — verify we can take a byte slice and walk it without
    // panicking. Replace with a call to the in-tree model-header validator
    // once it lands.
    let _len = data.len();
    let _first = data.first().copied();
});
