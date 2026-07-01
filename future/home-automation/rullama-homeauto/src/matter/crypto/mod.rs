/// Matter 1.3 cryptographic primitives.
///
/// This module provides the cryptographic building blocks for the Matter protocol stack:
///
/// - `kdf`: Key-derivation functions — PBKDF2 passcode verifier and HKDF-SHA256 label expansion.
/// - `spake2plus`: SPAKE2+ password-authenticated key exchange per RFC 9383 / Matter §3.9.
///
/// All key material is zeroized on drop via the `zeroize` crate.
///
/// Key-derivation function helpers (PBKDF2, HKDF).
pub mod kdf;
/// SPAKE2+ PAKE — prover (commissioner) and verifier (commissionee) sides.
pub mod spake2plus;

pub use kdf::{derive_passcode_verifier, hkdf_expand_label};
pub use spake2plus::{Spake2PlusKeys, Spake2PlusProver, Spake2PlusVerifier};
