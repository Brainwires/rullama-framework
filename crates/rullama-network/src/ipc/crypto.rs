//! IPC Encryption using ChaCha20-Poly1305
//!
//! Provides authenticated encryption for IPC messages between CLI processes.
//! Uses a shared key derived from the session token for symmetric encryption.

use anyhow::{Context, Result};
use chacha20poly1305::{
    ChaCha20Poly1305, Nonce,
    aead::{Aead, KeyInit},
};
use rand::Rng;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Key size for ChaCha20-Poly1305 (256 bits)
const KEY_SIZE: usize = 32;
/// Nonce size for ChaCha20-Poly1305 (96 bits)
const NONCE_SIZE: usize = 12;

/// Encrypted message format:
/// [nonce (12 bytes)][ciphertext (variable)][auth tag (16 bytes, included in ciphertext)]
pub struct IpcCipher {
    cipher: ChaCha20Poly1305,
}

impl IpcCipher {
    /// Create a new cipher from a session token
    ///
    /// The session token is hashed with SHA-256 to derive the encryption key.
    /// This ensures a consistent 256-bit key regardless of token length.
    pub fn from_session_token(token: &str) -> Self {
        let key = derive_key_from_token(token);
        let cipher = ChaCha20Poly1305::new_from_slice(key.as_slice())
            .expect("Key is always 32 bytes from SHA-256");
        Self { cipher }
    }

    /// Encrypt a message
    ///
    /// Returns the encrypted message with nonce prepended.
    /// Format: [nonce (12 bytes)][ciphertext + auth tag]
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        rand::rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt with authentication
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        // Prepend nonce to ciphertext
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt a message
    ///
    /// Expects format: [nonce (12 bytes)][ciphertext + auth tag]
    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < NONCE_SIZE {
            anyhow::bail!("Encrypted message too short (missing nonce)");
        }

        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt and verify authentication
        let plaintext = self
            .cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| anyhow::anyhow!("Decryption failed (authentication failed)"))?;

        Ok(plaintext)
    }

    /// Encrypt a string message to base64
    ///
    /// Convenient method for encrypting string messages.
    pub fn encrypt_string(&self, plaintext: &str) -> Result<String> {
        let encrypted = self.encrypt(plaintext.as_bytes())?;
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &encrypted,
        ))
    }

    /// Decrypt a base64 message to string
    ///
    /// Convenient method for decrypting string messages.
    pub fn decrypt_string(&self, encrypted_b64: &str) -> Result<String> {
        let encrypted =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encrypted_b64)
                .context("Invalid base64 encoding")?;

        let plaintext = self.decrypt(&encrypted)?;
        String::from_utf8(plaintext).context("Decrypted data is not valid UTF-8")
    }
}

/// Derive a 256-bit key from a session token using SHA-256
fn derive_key_from_token(token: &str) -> Zeroizing<[u8; KEY_SIZE]> {
    let mut hasher = Sha256::new();
    // Add a domain separator to prevent key reuse across different contexts
    hasher.update(b"rullama-ipc-v1:");
    hasher.update(token.as_bytes());

    let result = hasher.finalize();
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    key.copy_from_slice(&result);
    key
}

/// Generate a random encryption key (for testing or direct use)
pub fn generate_random_key() -> Zeroizing<[u8; KEY_SIZE]> {
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    rand::rng().fill_bytes(&mut *key);
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let token = "test-session-token-12345";
        let cipher = IpcCipher::from_session_token(token);

        let plaintext = b"Hello, this is a secret message!";
        let encrypted = cipher.encrypt(plaintext).unwrap();

        // Encrypted should be different from plaintext
        assert_ne!(encrypted.as_slice(), plaintext);

        // Decrypt should recover original
        let decrypted = cipher.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_string() {
        let token = "string-test-token";
        let cipher = IpcCipher::from_session_token(token);

        let message = "This is a JSON message: {\"key\": \"value\"}";
        let encrypted = cipher.encrypt_string(message).unwrap();

        // Encrypted is base64
        assert!(
            encrypted
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
        );

        let decrypted = cipher.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, message);
    }

    #[test]
    fn test_different_tokens_different_ciphertext() {
        let cipher1 = IpcCipher::from_session_token("token1");
        let cipher2 = IpcCipher::from_session_token("token2");

        let plaintext = b"Same message";
        let encrypted1 = cipher1.encrypt(plaintext).unwrap();
        let encrypted2 = cipher2.encrypt(plaintext).unwrap();

        // Different tokens should produce different ciphertexts
        // (also nonces are random, so even same token would differ)
        assert_ne!(encrypted1, encrypted2);

        // Can't decrypt with wrong key
        assert!(cipher2.decrypt(&encrypted1).is_err());
        assert!(cipher1.decrypt(&encrypted2).is_err());
    }

    #[test]
    fn test_tamper_detection() {
        let token = "tamper-test";
        let cipher = IpcCipher::from_session_token(token);

        let plaintext = b"Original message";
        let mut encrypted = cipher.encrypt(plaintext).unwrap();

        // Tamper with the ciphertext
        if let Some(byte) = encrypted.last_mut() {
            *byte ^= 0xFF;
        }

        // Decryption should fail due to authentication failure
        assert!(cipher.decrypt(&encrypted).is_err());
    }

    #[test]
    fn test_empty_message() {
        let token = "empty-test";
        let cipher = IpcCipher::from_session_token(token);

        let plaintext = b"";
        let encrypted = cipher.encrypt(plaintext).unwrap();
        let decrypted = cipher.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_large_message() {
        let token = "large-test";
        let cipher = IpcCipher::from_session_token(token);

        // 1 MB message
        let plaintext: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let encrypted = cipher.encrypt(&plaintext).unwrap();
        let decrypted = cipher.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_derivation_deterministic() {
        let token = "deterministic-test";

        let cipher1 = IpcCipher::from_session_token(token);
        let cipher2 = IpcCipher::from_session_token(token);

        // Same token should allow decryption
        let plaintext = b"Test message";
        let encrypted = cipher1.encrypt(plaintext).unwrap();
        let decrypted = cipher2.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }
}
