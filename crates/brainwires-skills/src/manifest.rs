//! Skill Package Manifest
//!
//! Defines the metadata structure for distributable skill packages.
//! A manifest contains all information needed to discover, resolve,
//! and install a skill from a registry.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Manifest describing a distributable skill package.
///
/// Contains all metadata needed for discovery, dependency resolution,
/// and compatibility checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique skill name (lowercase, hyphens allowed)
    pub name: String,
    /// Semantic version of this release
    pub version: semver::Version,
    /// Human-readable description
    pub description: String,
    /// Author or maintainer
    pub author: String,
    /// SPDX license expression (e.g. "MIT OR Apache-2.0")
    pub license: String,
    /// Discovery tags (e.g. ["code-review", "testing"])
    #[serde(default)]
    pub tags: Vec<String>,
    /// Skills this package depends on
    #[serde(default)]
    pub dependencies: Vec<SkillDependency>,
    /// Minimum brainwires-framework version required
    #[serde(default)]
    pub min_framework_version: Option<semver::VersionReq>,
    /// Source repository URL
    #[serde(default)]
    pub repository: Option<String>,
    /// Hex-encoded public key of the skill author for signature verification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_key: Option<String>,
    /// When this version was first published
    pub created_at: DateTime<Utc>,
    /// When the manifest was last modified
    pub updated_at: DateTime<Utc>,
}

/// A dependency on another skill package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDependency {
    /// Name of the required skill
    pub name: String,
    /// Acceptable version range
    pub version_req: semver::VersionReq,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_serde_roundtrip() {
        let manifest = SkillManifest {
            name: "review-pr".to_string(),
            version: semver::Version::new(1, 2, 3),
            description: "Reviews pull requests".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            tags: vec!["code-review".to_string(), "testing".to_string()],
            dependencies: vec![SkillDependency {
                name: "lint-code".to_string(),
                version_req: semver::VersionReq::parse(">=0.5.0").unwrap(),
            }],
            min_framework_version: Some(semver::VersionReq::parse(">=0.6.0").unwrap()),
            repository: Some("https://github.com/example/review-pr".to_string()),
            signing_key: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: SkillManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "review-pr");
        assert_eq!(deserialized.version, semver::Version::new(1, 2, 3));
        assert_eq!(deserialized.tags.len(), 2);
        assert_eq!(deserialized.dependencies.len(), 1);
        assert_eq!(deserialized.dependencies[0].name, "lint-code");
    }

    #[test]
    fn test_manifest_minimal() {
        let json = r#"{
            "name": "simple",
            "version": "0.10.0",
            "description": "A simple skill",
            "author": "Me",
            "license": "MIT",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }"#;

        let manifest: SkillManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.name, "simple");
        assert!(manifest.tags.is_empty());
        assert!(manifest.dependencies.is_empty());
        assert!(manifest.min_framework_version.is_none());
        assert!(manifest.repository.is_none());
    }

    #[test]
    fn test_semver_parsing() {
        let version = semver::Version::parse("1.0.0-beta.1").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);

        let req = semver::VersionReq::parse(">=0.6.0, <1.0.0").unwrap();
        assert!(req.matches(&semver::Version::new(0, 6, 0)));
        assert!(!req.matches(&semver::Version::new(1, 0, 0)));
    }

    #[test]
    fn test_skill_dependency_serde() {
        let dep = SkillDependency {
            name: "base-skill".to_string(),
            version_req: semver::VersionReq::parse("^1.0").unwrap(),
        };

        let json = serde_json::to_string(&dep).unwrap();
        let deserialized: SkillDependency = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "base-skill");
        assert!(
            deserialized
                .version_req
                .matches(&semver::Version::new(1, 5, 0))
        );
    }
}
