//! Tier-B adversarial cases for `brainwires_skills::verification::SkillVerifier`.
//!
//! Invariants:
//! - Tampering with `skill_content` after signing causes `verify` to fail.
//! - Changing `manifest.name` after signing causes `verify` to fail.
//! - Changing `manifest.version` after signing causes `verify` to fail.
//!   (Catches a regression where version is dropped from the digest.)
//! - A signature produced by a different keypair fails verification.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_skills::manifest::SkillManifest;
use brainwires_skills::package::SkillPackage;
use brainwires_skills::verification::SkillVerifier;
use chrono::Utc;

use crate::registry::SecurityCase;

fn sample_package() -> SkillPackage {
    SkillPackage {
        manifest: SkillManifest {
            name: "harness-skill".to_string(),
            version: semver::Version::new(1, 0, 0),
            description: "harness test package".to_string(),
            author: "harness".to_string(),
            license: "MIT".to_string(),
            tags: vec![],
            dependencies: vec![],
            min_framework_version: None,
            repository: None,
            signing_key: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        },
        skill_content: "# Sample skill\nDo a thing.".to_string(),
        checksum: "harness".to_string(),
        signature: None,
    }
}

// ── sec.skills.tampered_content_rejected ───────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.skills.tampered_content_rejected",
        crate_name: "brainwires-skills",
        invariant: "SkillVerifier::verify rejects a package whose skill_content was modified after signing",
        factory: || Box::new(TamperedContentCase),
    }
}

struct TamperedContentCase;

#[async_trait]
impl EvaluationCase for TamperedContentCase {
    fn name(&self) -> &str {
        "sec.skills.tampered_content_rejected"
    }
    fn category(&self) -> &str {
        "security.skills"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let (priv_key, pub_key) = SkillVerifier::generate_keypair();
        let mut pkg = sample_package();
        let sig = SkillVerifier::sign(&pkg, &priv_key)?;
        pkg.skill_content = "# Hijacked content — `rm -rf /`".to_string();
        let valid = SkillVerifier::verify(&pkg, &sig, &pub_key)?;
        if valid {
            return Ok(TrialResult::failure(
                0,
                0,
                "verify returned true after skill_content was tampered with",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.skills.tampered_name_rejected ──────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.skills.tampered_name_rejected",
        crate_name: "brainwires-skills",
        invariant: "SkillVerifier::verify rejects a package whose manifest.name was changed after signing",
        factory: || Box::new(TamperedNameCase),
    }
}

struct TamperedNameCase;

#[async_trait]
impl EvaluationCase for TamperedNameCase {
    fn name(&self) -> &str {
        "sec.skills.tampered_name_rejected"
    }
    fn category(&self) -> &str {
        "security.skills"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let (priv_key, pub_key) = SkillVerifier::generate_keypair();
        let mut pkg = sample_package();
        let sig = SkillVerifier::sign(&pkg, &priv_key)?;
        pkg.manifest.name = "some-other-skill".to_string();
        let valid = SkillVerifier::verify(&pkg, &sig, &pub_key)?;
        if valid {
            return Ok(TrialResult::failure(
                0,
                0,
                "verify returned true after manifest.name was changed",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.skills.tampered_version_rejected ───────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.skills.tampered_version_rejected",
        crate_name: "brainwires-skills",
        invariant: "SkillVerifier::verify rejects a package whose manifest.version was changed after signing (catches digest dropping version)",
        factory: || Box::new(TamperedVersionCase),
    }
}

struct TamperedVersionCase;

#[async_trait]
impl EvaluationCase for TamperedVersionCase {
    fn name(&self) -> &str {
        "sec.skills.tampered_version_rejected"
    }
    fn category(&self) -> &str {
        "security.skills"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let (priv_key, pub_key) = SkillVerifier::generate_keypair();
        let mut pkg = sample_package();
        let sig = SkillVerifier::sign(&pkg, &priv_key)?;
        pkg.manifest.version = semver::Version::new(9, 9, 9);
        let valid = SkillVerifier::verify(&pkg, &sig, &pub_key)?;
        if valid {
            return Ok(TrialResult::failure(
                0,
                0,
                "verify returned true after manifest.version was changed — version not in digest?",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.skills.wrong_pubkey_rejected ───────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.skills.wrong_pubkey_rejected",
        crate_name: "brainwires-skills",
        invariant: "SkillVerifier::verify rejects a valid signature against an unrelated public key",
        factory: || Box::new(WrongPubkeyCase),
    }
}

struct WrongPubkeyCase;

#[async_trait]
impl EvaluationCase for WrongPubkeyCase {
    fn name(&self) -> &str {
        "sec.skills.wrong_pubkey_rejected"
    }
    fn category(&self) -> &str {
        "security.skills"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let (priv_a, _pub_a) = SkillVerifier::generate_keypair();
        let (_priv_b, pub_b) = SkillVerifier::generate_keypair();
        let pkg = sample_package();
        let sig = SkillVerifier::sign(&pkg, &priv_a)?;
        let valid = SkillVerifier::verify(&pkg, &sig, &pub_b)?;
        if valid {
            return Ok(TrialResult::failure(
                0,
                0,
                "verify returned true against an unrelated public key",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
