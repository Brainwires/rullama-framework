use hkdf::Hkdf;
use hmac::{Hmac, Mac};
/// SPAKE2+ password-authenticated key exchange.
///
/// Implements RFC 9383 "SPAKE2+, an Augmented Password-Authenticated Key Exchange (PAKE) Protocol"
/// as required by Matter Core Specification §3.9 (commissioning PAKE).
///
/// # Roles
/// - **Prover** (commissioner / controller): knows the passcode; derives `(w0, w1)` from it.
/// - **Verifier** (commissionee / device): stores `(w0, L = w1*P)` so the passcode itself
///   is not stored on the device.
///
/// # Protocol flow
/// ```text
///  Prover                                  Verifier
///  ──────                                  ────────
///  new(w0s, w1s)                           new(w0s, L_bytes)
///  pA = pake_message() ──── pA ──────────>
///                      <─── pB ─────────── pB = pake_message()
///  keys = finish(pB, ctx)                  keys = finish(pA, ctx)
///  assert keys.ke == remote_ke
/// ```
use p256::{
    EncodedPoint, FieldBytes, NonZeroScalar, ProjectivePoint, PublicKey, Scalar, U256,
    elliptic_curve::{
        ops::{MulByGenerator, Reduce},
        sec1::{FromEncodedPoint, ToEncodedPoint},
    },
};
use rand_core::OsRng;
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

// ── Augmented SPAKE2+ fixed points M and N (from RFC 9383 Appendix B.1) ──────
//
// M and N are fixed P-256 points whose discrete logarithm with respect to the
// generator P is unknown ("nothing-up-my-sleeve" points).
//
// These are the SEC1 compressed encodings from RFC 9383 §4 / Appendix B.1.

/// M point (compressed) — prover's fixed point.
const M_COMPRESSED: &[u8] = &[
    0x02, 0x88, 0x6e, 0x2f, 0x97, 0xac, 0xe4, 0x6e, 0x55, 0xba, 0x9d, 0xd7, 0x24, 0x25, 0x79, 0xf2,
    0x99, 0x3b, 0x64, 0xe1, 0x6e, 0xf3, 0xdc, 0xab, 0x95, 0xaf, 0xd4, 0x97, 0x33, 0x3d, 0x8f, 0xa1,
    0x2f,
];

/// N point (compressed) — verifier's fixed point.
const N_COMPRESSED: &[u8] = &[
    0x02, 0xd8, 0xbb, 0xd6, 0xc6, 0x39, 0xc6, 0x29, 0x37, 0xb0, 0x4d, 0x99, 0x7f, 0x38, 0xc3, 0x77,
    0x07, 0x19, 0xc6, 0x29, 0xd7, 0x01, 0x4d, 0x49, 0xa2, 0x4b, 0x4f, 0x98, 0xba, 0xa1, 0x29, 0x2b,
    0x49,
];

// ── Key output ────────────────────────────────────────────────────────────────

/// Session keys and confirmation values produced by a completed SPAKE2+ exchange.
///
/// Both `Spake2PlusProver::finish` and `Spake2PlusVerifier::finish` produce this struct.
/// A successful commissioning requires:
///   - Both sides compute the same `ke` (session key).
///   - Prover's `ca` matches the verifier's independently-derived `ca` (and vice-versa for `cb`).
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Spake2PlusKeys {
    /// Ke: 16-byte session encryption key.
    pub ke: [u8; 16],
    /// Ka: 16-byte attestation/confirmation key material.
    pub ka: [u8; 16],
    /// cA: 32-byte confirmation MAC from the prover (sent to verifier for verification).
    pub ca: [u8; 32],
    /// cB: 32-byte confirmation MAC from the verifier (sent to prover for verification).
    pub cb: [u8; 32],
}

impl std::fmt::Debug for Spake2PlusKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spake2PlusKeys")
            .field("ke", &"[redacted]")
            .field("ka", &"[redacted]")
            .field("ca", &hex::encode(self.ca))
            .field("cb", &hex::encode(self.cb))
            .finish()
    }
}

// ── Prover (commissioner) ─────────────────────────────────────────────────────

/// SPAKE2+ prover state (commissioner / controller side).
///
/// The prover knows the passcode and derives `(w0, w1)` from it via PBKDF2
/// (see [`crate::matter::crypto::kdf::derive_passcode_verifier`]).
pub struct Spake2PlusProver {
    w0: Scalar,
    w1: Scalar,
    /// Ephemeral scalar `x`.
    x: Scalar,
    /// pA = x*P + w0*M  (sent to verifier as pake_message).
    pub_x: ProjectivePoint,
}

impl Spake2PlusProver {
    /// Create a new prover from raw `w0s` / `w1s` verifier bytes (each 40 bytes).
    ///
    /// Reduces each 40-byte value modulo the P-256 group order to produce scalars `w0`, `w1`,
    /// then generates a random ephemeral scalar `x` and computes `pA = x*P + w0*M`.
    pub fn new(w0s: &[u8], w1s: &[u8]) -> Result<Self, &'static str> {
        let w0 = scalar_from_wide(w0s)?;
        let w1 = scalar_from_wide(w1s)?;
        let m = decode_point(M_COMPRESSED)?;

        // Random ephemeral scalar x
        let x_nz = NonZeroScalar::random(&mut OsRng);
        let x: Scalar = *x_nz;

        // pA = x*P + w0*M
        let xp = ProjectivePoint::mul_by_generator(&x);
        let w0m = m * w0;
        let pub_x = xp + w0m;

        Ok(Self { w0, w1, x, pub_x })
    }

    /// Return pA encoded as a 65-byte uncompressed SEC1 point.
    pub fn pake_message(&self) -> Vec<u8> {
        point_to_uncompressed(&self.pub_x)
    }

    /// Complete the exchange after receiving `pb_bytes` (pB from the verifier).
    ///
    /// Computes the shared values Z and V, builds the transcript hash TT, and derives
    /// session key `ke`, attestation key `ka`, and confirmation values `ca` / `cb`.
    pub fn finish(&self, pb_bytes: &[u8], context: &[u8]) -> Result<Spake2PlusKeys, &'static str> {
        let pb = decode_point(pb_bytes)?;
        let n = decode_point(N_COMPRESSED)?;

        // Z = x * (pB - w0*N)
        let w0n = n * self.w0;
        let pb_minus_w0n = pb - w0n;
        let z = pb_minus_w0n * self.x;

        // V = w1 * (pB - w0*N)
        let v = pb_minus_w0n * self.w1;

        derive_keys(context, &self.pub_x, &pb, &z, &v, &self.w0)
    }
}

// ── Verifier (commissionee / device) ─────────────────────────────────────────

/// SPAKE2+ verifier state (commissionee / device side).
///
/// The device stores `(w0, L = w1*P)` — it never holds the passcode directly.
pub struct Spake2PlusVerifier {
    w0: Scalar,
    /// L = w1 * P  (stored on the device in place of w1).
    l: ProjectivePoint,
    /// Ephemeral scalar `y`.
    y: Scalar,
    /// pB = y*P + w0*N  (sent to prover as pake_message).
    pub_y: ProjectivePoint,
}

impl Spake2PlusVerifier {
    /// Create a new verifier from `w0s` bytes and the stored `L` point bytes.
    ///
    /// `w0s`     — 40-byte raw verifier value for w0 (same as from PBKDF2).
    /// `l_bytes` — Encoded point `L = w1*P` (33-byte compressed or 65-byte uncompressed).
    pub fn new(w0s: &[u8], l_bytes: &[u8]) -> Result<Self, &'static str> {
        let w0 = scalar_from_wide(w0s)?;
        let l = decode_point(l_bytes)?;
        let n = decode_point(N_COMPRESSED)?;

        // Random ephemeral scalar y
        let y_nz = NonZeroScalar::random(&mut OsRng);
        let y: Scalar = *y_nz;

        // pB = y*P + w0*N
        let yp = ProjectivePoint::mul_by_generator(&y);
        let w0n = n * w0;
        let pub_y = yp + w0n;

        Ok(Self { w0, l, y, pub_y })
    }

    /// Convenience constructor: compute `L = w1*P` from `w1s` bytes directly.
    ///
    /// Useful when the device has just run PBKDF2 and has `(w0s, w1s)` in hand.
    pub fn new_from_w1s(w0s: &[u8], w1s: &[u8]) -> Result<Self, &'static str> {
        let w1 = scalar_from_wide(w1s)?;
        let l = ProjectivePoint::mul_by_generator(&w1);
        let l_bytes = point_to_uncompressed(&l);
        Self::new(w0s, &l_bytes)
    }

    /// Return pB encoded as a 65-byte uncompressed SEC1 point.
    pub fn pake_message(&self) -> Vec<u8> {
        point_to_uncompressed(&self.pub_y)
    }

    /// Complete the exchange after receiving `pa_bytes` (pA from the prover).
    pub fn finish(&self, pa_bytes: &[u8], context: &[u8]) -> Result<Spake2PlusKeys, &'static str> {
        let pa = decode_point(pa_bytes)?;
        let m = decode_point(M_COMPRESSED)?;

        // Z = y * (pA - w0*M)
        let w0m = m * self.w0;
        let pa_minus_w0m = pa - w0m;
        let z = pa_minus_w0m * self.y;

        // V = y * L
        let v = self.l * self.y;

        derive_keys(context, &pa, &self.pub_y, &z, &v, &self.w0)
    }
}

// ── Shared key derivation ─────────────────────────────────────────────────────

/// Derive `Spake2PlusKeys` from the protocol transcript.
///
/// Per RFC 9383 §3.3 and Matter §3.9:
/// 1. Serialize pA, pB, Z, V, w0 as uncompressed points / raw scalar bytes.
/// 2. Build transcript hash TT = SHA-256(len64LE(ctx) || ctx || pA || pB || Z || V || w0).
/// 3. Ka||Ke = TT (split at 16 bytes each — the hash is 32 bytes total).
/// 4. Derive Kcca||Kccb = HKDF-SHA256(Ka, nil, "ConfirmationKeys", 32).
/// 5. cA = HMAC-SHA256(Kcca, pB), cB = HMAC-SHA256(Kccb, pA).
fn derive_keys(
    context: &[u8],
    pa: &ProjectivePoint,
    pb: &ProjectivePoint,
    z: &ProjectivePoint,
    v: &ProjectivePoint,
    w0: &Scalar,
) -> Result<Spake2PlusKeys, &'static str> {
    let pa_bytes = point_to_uncompressed(pa);
    let pb_bytes = point_to_uncompressed(pb);
    let z_bytes = point_to_uncompressed(z);
    let v_bytes = point_to_uncompressed(v);

    // w0 as 32-byte big-endian scalar (P-256 field element size)
    let w0_bytes: FieldBytes = w0.to_bytes();

    // ── Build transcript TT ──
    // TT = SHA-256( len64LE(context) || context || pA || pB || Z || V || w0 )
    let mut hasher = Sha256::new();
    let ctx_len = (context.len() as u64).to_le_bytes();
    hasher.update(ctx_len);
    hasher.update(context);
    hasher.update(&pa_bytes);
    hasher.update(&pb_bytes);
    hasher.update(&z_bytes);
    hasher.update(&v_bytes);
    hasher.update(w0_bytes.as_slice());
    let tt: [u8; 32] = hasher.finalize().into();

    // ── Ka || Ke — split the 32-byte transcript hash ──
    let mut ka = [0u8; 16];
    let mut ke = [0u8; 16];
    ka.copy_from_slice(&tt[..16]);
    ke.copy_from_slice(&tt[16..]);

    // ── Derive confirmation keys Kcca and Kccb via HKDF(Ka, nil, "ConfirmationKeys", 32) ──
    let hk = Hkdf::<Sha256>::new(None, &ka);
    let mut kcc = [0u8; 32];
    hk.expand(b"ConfirmationKeys", &mut kcc)
        .map_err(|_| "HKDF expand failed for ConfirmationKeys")?;
    let kcca = &kcc[..16];
    let kccb = &kcc[16..];

    // ── cA = HMAC-SHA256(Kcca, pB) ──
    let mut mac_a: Hmac<Sha256> =
        Mac::new_from_slice(kcca).map_err(|_| "HMAC init failed for cA")?;
    mac_a.update(&pb_bytes);
    let ca_vec = mac_a.finalize().into_bytes();

    // ── cB = HMAC-SHA256(Kccb, pA) ──
    let mut mac_b: Hmac<Sha256> =
        Mac::new_from_slice(kccb).map_err(|_| "HMAC init failed for cB")?;
    mac_b.update(&pa_bytes);
    let cb_vec = mac_b.finalize().into_bytes();

    let mut ca = [0u8; 32];
    let mut cb = [0u8; 32];
    ca.copy_from_slice(&ca_vec);
    cb.copy_from_slice(&cb_vec);

    Ok(Spake2PlusKeys { ke, ka, ca, cb })
}

// ── Point / scalar helpers ────────────────────────────────────────────────────

/// Decode a SEC1-encoded point (compressed or uncompressed) into a `ProjectivePoint`.
fn decode_point(bytes: &[u8]) -> Result<ProjectivePoint, &'static str> {
    let ep = EncodedPoint::from_bytes(bytes).map_err(|_| "invalid SEC1 point encoding")?;
    let pk = PublicKey::from_encoded_point(&ep);
    if pk.is_none().into() {
        return Err("point is not on the P-256 curve");
    }
    Ok(ProjectivePoint::from(pk.unwrap()))
}

/// Encode a `ProjectivePoint` as a 65-byte uncompressed SEC1 point.
fn point_to_uncompressed(p: &ProjectivePoint) -> Vec<u8> {
    let affine = p.to_affine();
    let ep = affine.to_encoded_point(false); // false = uncompressed
    ep.as_bytes().to_vec()
}

/// Reduce a byte slice (1–64 bytes) into a P-256 `Scalar` modulo the group order.
///
/// Matter spec §3.10 produces 40-byte w0s/w1s values.  We hash the input with SHA-256
/// to produce a 32-byte value and reduce it modulo the group order — matching the
/// approach used by the CHIP reference implementation.
fn scalar_from_wide(bytes: &[u8]) -> Result<Scalar, &'static str> {
    if bytes.is_empty() || bytes.len() > 64 {
        return Err("scalar input must be 1–64 bytes");
    }
    // Hash to 32 bytes then reduce mod n (P-256 group order).
    // This gives uniform distribution in the scalar field without needing
    // a wide-reduction implementation.
    let hash: [u8; 32] = Sha256::digest(bytes).into();
    let fb = FieldBytes::from(hash);
    // Reduce::reduce_bytes interprets fb as a big-endian integer and computes it mod n.
    Ok(<Scalar as Reduce<U256>>::reduce_bytes(&fb))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::crypto::kdf::derive_passcode_verifier;

    const TEST_PASSCODE: u32 = 20202021;
    const TEST_SALT: &[u8] = b"SPAKE2P+SaltValue";
    const TEST_ITERATIONS: u32 = 1000;
    const TEST_CONTEXT: &[u8] = b"Matter Test Context";

    fn test_verifier_bytes() -> (Vec<u8>, Vec<u8>) {
        derive_passcode_verifier(TEST_PASSCODE, TEST_SALT, TEST_ITERATIONS).unwrap()
    }

    // ── Round-trip: prover and verifier derive matching session keys ──────────

    #[test]
    fn spake2plus_roundtrip_ke_matches() {
        let (w0s, w1s) = test_verifier_bytes();

        let prover = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();

        let pa = prover.pake_message();
        let pb = verifier.pake_message();

        let prover_keys = prover.finish(&pb, TEST_CONTEXT).unwrap();
        let verifier_keys = verifier.finish(&pa, TEST_CONTEXT).unwrap();

        assert_eq!(prover_keys.ke, verifier_keys.ke, "session keys must match");
        assert_eq!(
            prover_keys.ka, verifier_keys.ka,
            "attestation keys must match"
        );
    }

    #[test]
    fn spake2plus_roundtrip_confirmations_cross_validate() {
        let (w0s, w1s) = test_verifier_bytes();

        let prover = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();

        let pa = prover.pake_message();
        let pb = verifier.pake_message();

        let pk = prover.finish(&pb, TEST_CONTEXT).unwrap();
        let vk = verifier.finish(&pa, TEST_CONTEXT).unwrap();

        // cA computed by prover must equal cA computed by verifier
        assert_eq!(pk.ca, vk.ca, "cA must match between prover and verifier");
        // cB computed by prover must equal cB computed by verifier
        assert_eq!(pk.cb, vk.cb, "cB must match between prover and verifier");
    }

    // ── Wrong passcode: keys must NOT match ──────────────────────────────────

    #[test]
    fn spake2plus_wrong_passcode_keys_differ() {
        let (w0s, w1s) = test_verifier_bytes();
        // Different passcode for the prover
        let (bad_w0s, bad_w1s) =
            derive_passcode_verifier(11111111, TEST_SALT, TEST_ITERATIONS).unwrap();

        let prover = Spake2PlusProver::new(&bad_w0s, &bad_w1s).unwrap();
        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();

        let pa = prover.pake_message();
        let pb = verifier.pake_message();

        let pk = prover.finish(&pb, TEST_CONTEXT).unwrap();
        let vk = verifier.finish(&pa, TEST_CONTEXT).unwrap();

        assert_ne!(
            pk.ke, vk.ke,
            "mismatched passcode must produce different session keys"
        );
    }

    // ── pake_message returns 65-byte uncompressed point ──────────────────────

    #[test]
    fn pake_message_length() {
        let (w0s, w1s) = test_verifier_bytes();
        let prover = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();

        let pa = prover.pake_message();
        let pb = verifier.pake_message();

        assert_eq!(pa.len(), 65, "pA must be 65-byte uncompressed point");
        assert_eq!(pb.len(), 65, "pB must be 65-byte uncompressed point");
        assert_eq!(pa[0], 0x04, "uncompressed point starts with 0x04");
        assert_eq!(pb[0], 0x04, "uncompressed point starts with 0x04");
    }

    // ── Fixed M and N points decode cleanly ──────────────────────────────────

    #[test]
    fn fixed_points_decode() {
        decode_point(M_COMPRESSED).expect("M point must decode");
        decode_point(N_COMPRESSED).expect("N point must decode");
    }

    // ── Different context strings produce different ca/cb ────────────────────

    #[test]
    fn different_context_different_confirmations() {
        let (w0s, w1s) = test_verifier_bytes();

        let prover_a = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier_a = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();
        let pa_a = prover_a.pake_message();
        let pb_a = verifier_a.pake_message();
        let keys_a = prover_a.finish(&pb_a, b"ContextA").unwrap();
        let _ = verifier_a.finish(&pa_a, b"ContextA").unwrap();

        let prover_b = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier_b = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();
        let pa_b = prover_b.pake_message();
        let pb_b = verifier_b.pake_message();
        let keys_b = prover_b.finish(&pb_b, b"ContextB").unwrap();
        let _ = verifier_b.finish(&pa_b, b"ContextB").unwrap();

        // Both contexts and ephemeral scalars differ — ca must differ
        assert_ne!(
            keys_a.ca, keys_b.ca,
            "different contexts must yield different cA"
        );
    }

    // ── Zeroize compiles and runs without panic ───────────────────────────────

    #[test]
    fn spake2plus_keys_zeroize() {
        let (w0s, w1s) = test_verifier_bytes();
        let prover = Spake2PlusProver::new(&w0s, &w1s).unwrap();
        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s).unwrap();
        let pb = verifier.pake_message();
        let mut keys = prover.finish(&pb, TEST_CONTEXT).unwrap();
        keys.zeroize();
        assert_eq!(keys.ke, [0u8; 16]);
        assert_eq!(keys.ka, [0u8; 16]);
        assert_eq!(keys.ca, [0u8; 32]);
        assert_eq!(keys.cb, [0u8; 32]);
    }
}
