use anyhow::{Context, Result};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngExt;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Domain separator for network identity key derivation.
const KEY_DOMAIN: &str = "rullama-network-identity-v1:";

/// A signing key derived from a shared secret.
///
/// Uses ChaCha20-Poly1305 for authenticated encryption of messages,
/// reusing the same crypto primitives as the IPC layer (`ipc/crypto.rs`).
#[derive(Clone)]
pub struct SigningKey {
    cipher: ChaCha20Poly1305,
}

impl SigningKey {
    /// Derive a signing key from a shared secret (e.g. API key, session token).
    pub fn from_secret(secret: &str) -> Self {
        let key = derive_key(secret);
        let cipher = ChaCha20Poly1305::new(key.as_ref().into());
        Self { cipher }
    }

    /// Sign (encrypt + authenticate) a message payload.
    pub fn sign(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        rand::rng().fill(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow::anyhow!("signing failed: {e}"))?;

        // Wire format: [12-byte nonce][ciphertext + 16-byte auth tag]
        let mut output = Vec::with_capacity(12 + ciphertext.len());
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }
}

impl std::fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigningKey")
            .field("cipher", &"<redacted>")
            .finish()
    }
}

/// A verifying key that can authenticate signed messages.
///
/// Constructed from the same shared secret as the [`SigningKey`].
#[derive(Clone)]
pub struct VerifyingKey {
    cipher: ChaCha20Poly1305,
}

impl VerifyingKey {
    /// Derive a verifying key from the same shared secret used to create the
    /// [`SigningKey`].
    pub fn from_secret(secret: &str) -> Self {
        let key = derive_key(secret);
        let cipher = ChaCha20Poly1305::new(key.as_ref().into());
        Self { cipher }
    }

    /// Verify and decrypt a signed message.
    pub fn verify(&self, signed: &[u8]) -> Result<Vec<u8>> {
        if signed.len() < 12 {
            anyhow::bail!("signed message too short");
        }
        let (nonce_bytes, ciphertext) = signed.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("verification failed: {e}"))
            .context("message authentication failed")
    }
}

impl std::fmt::Debug for VerifyingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey")
            .field("cipher", &"<redacted>")
            .finish()
    }
}

/// Derive a 256-bit key from a secret string using SHA-256 with a domain
/// separator.
fn derive_key(secret: &str) -> Zeroizing<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(KEY_DOMAIN.as_bytes());
    hasher.update(secret.as_bytes());
    Zeroizing::new(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_roundtrip() {
        let secret = "test-secret-key";
        let signer = SigningKey::from_secret(secret);
        let verifier = VerifyingKey::from_secret(secret);

        let message = b"hello, agent network";
        let signed = signer.sign(message).unwrap();
        let recovered = verifier.verify(&signed).unwrap();

        assert_eq!(recovered, message);
    }

    #[test]
    fn wrong_secret_fails_verification() {
        let signer = SigningKey::from_secret("secret-a");
        let verifier = VerifyingKey::from_secret("secret-b");

        let signed = signer.sign(b"test").unwrap();
        assert!(verifier.verify(&signed).is_err());
    }

    #[test]
    fn tampered_message_fails() {
        let secret = "test-secret";
        let signer = SigningKey::from_secret(secret);
        let verifier = VerifyingKey::from_secret(secret);

        let mut signed = signer.sign(b"test").unwrap();
        // Flip a byte in the ciphertext
        if let Some(byte) = signed.last_mut() {
            *byte ^= 0xFF;
        }
        assert!(verifier.verify(&signed).is_err());
    }

    #[test]
    fn too_short_message_fails() {
        let verifier = VerifyingKey::from_secret("test");
        assert!(verifier.verify(&[0u8; 5]).is_err());
    }

    #[test]
    fn debug_redacts_key_material() {
        let signer = SigningKey::from_secret("secret");
        let debug = format!("{signer:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret"));
    }
}
