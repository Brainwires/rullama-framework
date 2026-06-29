//! Skill Package Format
//!
//! A `SkillPackage` bundles a skill's SKILL.md content together with a
//! [`SkillManifest`] and a SHA-256 integrity checksum. Packages are the
//! unit of distribution — they can be serialized to bytes for transport
//! to / from a skill registry.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

use super::manifest::SkillManifest;

/// A distributable skill package containing manifest, content, and integrity checksum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPackage {
    /// Package manifest with metadata
    pub manifest: SkillManifest,
    /// Raw SKILL.md content
    pub skill_content: String,
    /// Hex-encoded SHA-256 checksum of `skill_content`
    pub checksum: String,
    /// Optional hex-encoded Ed25519 signature of the package content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl SkillPackage {
    /// Create a package by reading a SKILL.md file from disk.
    ///
    /// The caller must supply the manifest separately (e.g. parsed from
    /// a `skill-manifest.json` next to the SKILL.md, or constructed
    /// programmatically).
    pub fn from_skill_file(path: &Path, manifest: SkillManifest) -> Result<Self> {
        let skill_content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read skill file: {}", path.display()))?;

        let checksum = compute_checksum(&skill_content);

        Ok(Self {
            manifest,
            skill_content,
            checksum,
            signature: None,
        })
    }

    /// Verify that the stored checksum matches the content.
    pub fn verify_checksum(&self) -> bool {
        compute_checksum(&self.skill_content) == self.checksum
    }

    /// Serialize the package to bytes (JSON) for transport.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).context("Failed to serialize SkillPackage")
    }

    /// Deserialize a package from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("Failed to deserialize SkillPackage")
    }
}

/// Compute the hex-encoded SHA-256 digest of the given content.
fn compute_checksum(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::SkillManifest;
    use chrono::Utc;
    use tempfile::TempDir;

    fn sample_manifest() -> SkillManifest {
        SkillManifest {
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
        }
    }

    #[test]
    fn test_from_skill_file() {
        let dir = TempDir::new().unwrap();
        let skill_path = dir.path().join("SKILL.md");
        std::fs::write(&skill_path, "# Test Skill\nDo things.").unwrap();

        let pkg = SkillPackage::from_skill_file(&skill_path, sample_manifest()).unwrap();
        assert_eq!(pkg.skill_content, "# Test Skill\nDo things.");
        assert!(!pkg.checksum.is_empty());
    }

    #[test]
    fn test_verify_checksum_valid() {
        let dir = TempDir::new().unwrap();
        let skill_path = dir.path().join("SKILL.md");
        std::fs::write(&skill_path, "content").unwrap();

        let pkg = SkillPackage::from_skill_file(&skill_path, sample_manifest()).unwrap();
        assert!(pkg.verify_checksum());
    }

    #[test]
    fn test_verify_checksum_invalid() {
        let dir = TempDir::new().unwrap();
        let skill_path = dir.path().join("SKILL.md");
        std::fs::write(&skill_path, "content").unwrap();

        let mut pkg = SkillPackage::from_skill_file(&skill_path, sample_manifest()).unwrap();
        pkg.skill_content = "tampered".to_string();
        assert!(!pkg.verify_checksum());
    }

    #[test]
    fn test_serde_roundtrip() {
        let dir = TempDir::new().unwrap();
        let skill_path = dir.path().join("SKILL.md");
        std::fs::write(&skill_path, "roundtrip content").unwrap();

        let pkg = SkillPackage::from_skill_file(&skill_path, sample_manifest()).unwrap();
        let bytes = pkg.to_bytes().unwrap();
        let restored = SkillPackage::from_bytes(&bytes).unwrap();

        assert_eq!(restored.manifest.name, "test-skill");
        assert_eq!(restored.skill_content, "roundtrip content");
        assert_eq!(restored.checksum, pkg.checksum);
        assert!(restored.verify_checksum());
    }
}
