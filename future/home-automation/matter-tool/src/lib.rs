//! Library-side helpers for `matter-tool`, exposed so integration tests can
//! exercise pure-Rust logic without spawning the binary.
//!
//! The binary itself lives in `main.rs` and re-implements its CLI wiring; this
//! module hosts only the small, deterministic helpers that are worth testing
//! directly.

/// Decode a TLV payload supplied on the command line as a hex string.
///
/// Accepts an optional `0x` prefix. Returns a typed `hex::FromHexError` for
/// malformed input (odd length, non-hex characters) rather than panicking —
/// this is the contract the `invoke` subcommand relies on.
pub fn parse_tlv_hex(input: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(input.trim_start_matches("0x"))
}
