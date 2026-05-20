//! Skill package signature verification using Ed25519.
//!
//! Provides cryptographic signing and verification for skill packages,
//! ensuring that skills come from trusted authors and have not been
//! tampered with.

use anyhow::{Context, Result};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::package::SkillPackage;

/// Ed25519-based skill package signer and verifier.
pub struct SkillVerifier;

impl SkillVerifier {
    /// Sign a skill package with a private key.
    ///
    /// Computes a deterministic digest of the package's manifest name,
    /// version, and content, then signs it with the provided Ed25519
    /// private key. Returns the hex-encoded signature.
    pub fn sign(package: &SkillPackage, private_key: &[u8]) -> Result<String> {
        let signing_key = SigningKey::try_from(private_key)
            .map_err(|e| anyhow::anyhow!("Invalid private key: {}", e))?;

        let digest = Self::package_digest(package);
        let signature = signing_key.sign(&digest);
        Ok(hex::encode(signature.to_bytes()))
    }

    /// Verify a skill package signature against a public key.
    ///
    /// Returns `Ok(true)` if the signature is valid, `Ok(false)` if it
    /// does not match. Returns an error only if the key or signature
    /// bytes are malformed.
    pub fn verify(package: &SkillPackage, signature: &str, public_key: &[u8]) -> Result<bool> {
        let verifying_key = VerifyingKey::try_from(public_key)
            .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;

        let sig_bytes = hex::decode(signature).context("Invalid hex signature")?;
        let signature = ed25519_dalek::Signature::from_slice(&sig_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid signature bytes: {}", e))?;

        let digest = Self::package_digest(package);
        Ok(verifying_key.verify(&digest, &signature).is_ok())
    }

    /// Generate a new Ed25519 keypair for skill signing.
    ///
    /// Returns `(private_key, public_key)` as raw byte vectors.
    pub fn generate_keypair() -> (Vec<u8>, Vec<u8>) {
        let mut csprng = rand_core::OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        (
            signing_key.to_bytes().to_vec(),
            verifying_key.to_bytes().to_vec(),
        )
    }

    /// Compute a deterministic digest of the signable parts of a package.
    fn package_digest(package: &SkillPackage) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(package.manifest.name.as_bytes());
        hasher.update(b":");
        hasher.update(package.manifest.version.to_string().as_bytes());
        hasher.update(b":");
        hasher.update(package.skill_content.as_bytes());
        hasher.finalize().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::SkillManifest;
    use chrono::Utc;

    fn sample_package() -> SkillPackage {
        SkillPackage {
            manifest: SkillManifest {
                name: "test-skill".to_string(),
                version: semver::Version::new(1, 0, 0),
                description: "A test skill".to_string(),
                author: "Test".to_string(),
                license: "MIT".to_string(),
                tags: vec![],
                dependencies: vec![],
                min_framework_version: None,
                repository: None,
                signing_key: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            skill_content: "# Test Skill\nDo things.".to_string(),
            checksum: "abc123".to_string(),
            signature: None,
        }
    }

    #[test]
    fn sign_and_verify_roundtrip() {
        let (private_key, public_key) = SkillVerifier::generate_keypair();
        let package = sample_package();

        let sig = SkillVerifier::sign(&package, &private_key).unwrap();
        let valid = SkillVerifier::verify(&package, &sig, &public_key).unwrap();
        assert!(valid);
    }

    #[test]
    fn invalid_signature_rejected() {
        let (_private_key, public_key) = SkillVerifier::generate_keypair();
        let package = sample_package();

        // Use a different keypair to produce a mismatched signature
        let (other_private, _) = SkillVerifier::generate_keypair();
        let bad_sig = SkillVerifier::sign(&package, &other_private).unwrap();

        let valid = SkillVerifier::verify(&package, &bad_sig, &public_key).unwrap();
        assert!(!valid);
    }

    #[test]
    fn tampered_content_rejected() {
        let (private_key, public_key) = SkillVerifier::generate_keypair();
        let mut package = sample_package();
        let sig = SkillVerifier::sign(&package, &private_key).unwrap();

        // Tamper with content
        package.skill_content = "# Tampered content".to_string();

        let valid = SkillVerifier::verify(&package, &sig, &public_key).unwrap();
        assert!(!valid);
    }

    #[test]
    fn keypair_generation_produces_valid_keys() {
        let (private_key, public_key) = SkillVerifier::generate_keypair();
        assert_eq!(private_key.len(), 32);
        assert_eq!(public_key.len(), 32);
    }

    #[test]
    fn invalid_private_key_returns_error() {
        let package = sample_package();
        let result = SkillVerifier::sign(&package, &[0u8; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_public_key_returns_error() {
        let package = sample_package();
        let result = SkillVerifier::verify(&package, "aabbcc", &[0u8; 5]);
        assert!(result.is_err());
    }
}
