//! Smoke tests for `matter-tool` command helpers.
//!
//! The mDNS discovery path relies on real network advertisers and the
//! `mdns-sd` crate does not expose an in-process test harness, so
//! `discover::run` is out of scope here. Today this file covers the TLV hex
//! parser exclusively.

use matter_tool::parse_tlv_hex;

#[test]
fn parse_tlv_hex_rejects_malformed() {
    // Odd-length hex must return a typed error — NEVER panic.
    let err = parse_tlv_hex("0A1").expect_err("odd-length hex should fail");
    assert!(
        matches!(err, hex::FromHexError::OddLength),
        "expected OddLength, got {err:?}"
    );

    // Non-hex characters should likewise produce a typed error.
    let err = parse_tlv_hex("zz").expect_err("non-hex chars should fail");
    assert!(
        matches!(err, hex::FromHexError::InvalidHexCharacter { .. }),
        "expected InvalidHexCharacter, got {err:?}"
    );
}

#[test]
fn parse_tlv_hex_accepts_valid() {
    let bytes = parse_tlv_hex("0a1b2c").expect("valid hex");
    assert_eq!(bytes, vec![0x0a, 0x1b, 0x2c]);

    // `0x` prefix is tolerated.
    let bytes = parse_tlv_hex("0x0A1B2C").expect("valid hex w/ prefix");
    assert_eq!(bytes, vec![0x0a, 0x1b, 0x2c]);

    // Empty input decodes to empty bytes (matches the `None` payload path).
    let bytes = parse_tlv_hex("").expect("empty hex");
    assert!(bytes.is_empty());
}
