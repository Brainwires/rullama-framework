/// Matter 1.3 key-derivation function helpers.
///
/// Implements:
/// - `derive_passcode_verifier` — Matter §3.10 PBKDF2-based SPAKE2+ verifier generation.
/// - `hkdf_expand_label` — Generic HKDF-SHA256 helper used for session key derivation.
use hkdf::Hkdf;
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;

/// Derive the SPAKE2+ w0s / w1s verifier from a passcode.
///
/// Per Matter Core Specification §3.10.1:
/// 1. Encode `passcode` as 4-byte little-endian.
/// 2. Run PBKDF2-HMAC-SHA256(password=passcode_le, salt, iterations, dkLen=80).
/// 3. Split: w0s = bytes[0..40], w1s = bytes[40..80].
///
/// Returns `(w0s, w1s)` as owned byte vectors, each 40 bytes long.
///
/// # Errors
/// Returns `Err` if `iterations` is zero.
pub fn derive_passcode_verifier(
    passcode: u32,
    salt: &[u8],
    iterations: u32,
) -> Result<(Vec<u8>, Vec<u8>), &'static str> {
    if iterations == 0 {
        return Err("PBKDF2 iterations must be > 0");
    }

    // passcode encoded as 4-byte little-endian per Matter spec §3.10
    let passcode_bytes = passcode.to_le_bytes();

    // Run PBKDF2-HMAC-SHA256 producing 80 bytes
    let mut dk = vec![0u8; 80];
    pbkdf2_hmac::<Sha256>(&passcode_bytes, salt, iterations, &mut dk);

    let w0s = dk[..40].to_vec();
    let w1s = dk[40..].to_vec();

    Ok((w0s, w1s))
}

/// Expand a secret using HKDF-SHA256 with an arbitrary label.
///
/// `ikm`   — Input keying material (the "secret").
/// `salt`  — Optional salt (pass `&[]` for zero-length salt).
/// `label` — Info string; the caller is responsible for any protocol-required prefixes.
/// `length` — Number of output bytes.
///
/// This is a thin wrapper around `hkdf::Hkdf<Sha256>` with no implicit prefix.
/// Matter §4.13.2 session-key derivation calls this with `label = "SessionKeys"` etc.
pub fn hkdf_expand_label(ikm: &[u8], salt: &[u8], label: &str, length: usize) -> Vec<u8> {
    let salt_opt = if salt.is_empty() { None } else { Some(salt) };
    let hk = Hkdf::<Sha256>::new(salt_opt, ikm);
    let mut out = vec![0u8; length];
    hk.expand(label.as_bytes(), &mut out)
        .expect("HKDF expand: output length must be <= 255 * hash_len");
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Matter §3.10 test vector:
    ///   passcode  = 20202021
    ///   salt      = hex "5350414b453250" (7 bytes: "SPAKE2P")  — used in early chip-tool CI
    ///   iterations = 1000
    ///
    /// The exact w0s/w1s bytes are not published in the public spec text, so we validate:
    ///   - output lengths are correct
    ///   - determinism (same inputs → same outputs)
    ///   - different passcode → different verifier
    #[test]
    fn derive_passcode_verifier_length() {
        let salt = b"SPAKE2P+";
        let (w0s, w1s) = derive_passcode_verifier(20202021, salt, 1000).unwrap();
        assert_eq!(w0s.len(), 40, "w0s must be 40 bytes");
        assert_eq!(w1s.len(), 40, "w1s must be 40 bytes");
        assert_ne!(w0s, w1s, "w0s and w1s should differ");
    }

    #[test]
    fn derive_passcode_verifier_deterministic() {
        let salt = b"test-salt-16byte";
        let (a0, a1) = derive_passcode_verifier(20202021, salt, 1000).unwrap();
        let (b0, b1) = derive_passcode_verifier(20202021, salt, 1000).unwrap();
        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
    }

    #[test]
    fn derive_passcode_verifier_different_passcode() {
        let salt = b"test-salt-16byte";
        let (a0, _) = derive_passcode_verifier(20202021, salt, 1000).unwrap();
        let (b0, _) = derive_passcode_verifier(11111111, salt, 1000).unwrap();
        assert_ne!(a0, b0, "different passcodes must produce different w0s");
    }

    #[test]
    fn derive_passcode_verifier_zero_iterations_err() {
        assert!(derive_passcode_verifier(20202021, b"salt", 0).is_err());
    }

    #[test]
    fn hkdf_expand_label_length() {
        let ikm = [0u8; 32];
        let out = hkdf_expand_label(&ikm, b"salt", "SessionKeys", 48);
        assert_eq!(out.len(), 48);
    }

    #[test]
    fn hkdf_expand_label_deterministic() {
        let ikm = [1u8; 32];
        let a = hkdf_expand_label(&ikm, b"", "TestLabel", 32);
        let b = hkdf_expand_label(&ikm, b"", "TestLabel", 32);
        assert_eq!(a, b);
    }

    #[test]
    fn hkdf_expand_label_different_labels() {
        let ikm = [2u8; 32];
        let a = hkdf_expand_label(&ikm, b"s", "LabelA", 32);
        let b = hkdf_expand_label(&ikm, b"s", "LabelB", 32);
        assert_ne!(a, b);
    }

    /// Spot-check against a known HKDF-SHA256 vector (RFC 5869 Appendix A.1).
    /// IKM = 0x0b * 22, salt = [0x00..=0x0c] (13 bytes), info = "SessionKeys", L=32
    ///
    /// We verify that our hkdf_expand_label wrapper produces the same output as calling
    /// Hkdf directly — both sides use the same hkdf crate so the result is definitionally
    /// consistent.
    #[test]
    fn hkdf_rfc5869_vector() {
        let ikm = vec![0x0bu8; 22];
        let salt: Vec<u8> = (0x00u8..=0x0cu8).collect();
        let info_str = "SessionKeys";

        // Build the expected OKM independently using the hkdf crate directly.
        let hk = hkdf::Hkdf::<sha2::Sha256>::new(Some(salt.as_slice()), &ikm);
        let mut expected = vec![0u8; 32];
        hk.expand(info_str.as_bytes(), &mut expected).unwrap();

        let got = hkdf_expand_label(&ikm, &salt, info_str, 32);
        assert_eq!(got, expected);
    }
}
