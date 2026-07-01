//! Tier-B adversarial cases for `rullama_permission::FilesystemCapabilities`.
//!
//! Invariants:
//! - `FilesystemCapabilities::default()` includes deny patterns for
//!   `.env*`, `*credentials*`, and `*secret*`. Removing these is exactly
//!   the silent regression this case exists to catch.
//! - The deny patterns actually MATCH realistic adversarial paths
//!   (.env.production, /etc/openssl/credentials.json, /home/u/.secret_key).
//!   A "denied_paths is non-empty" check alone would miss a broken matcher.
//! - `FilesystemCapabilities::default().write_paths` is empty —
//!   write access must be opted in.
//! - `FilesystemCapabilities::default().can_delete` is false —
//!   delete must be opted in.

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_permission::FilesystemCapabilities;

use crate::registry::SecurityCase;

// ── sec.permission.default_denies_env_credentials_secrets ───────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.permission.default_denies_env_credentials_secrets",
        crate_name: "rullama-permission",
        invariant: "Default FilesystemCapabilities denies .env*, *credentials*, *secret* against realistic paths",
        factory: || Box::new(DefaultDeniesEnvCredentialsSecretsCase),
    }
}

struct DefaultDeniesEnvCredentialsSecretsCase;

#[async_trait]
impl EvaluationCase for DefaultDeniesEnvCredentialsSecretsCase {
    fn name(&self) -> &str {
        "sec.permission.default_denies_env_credentials_secrets"
    }
    fn category(&self) -> &str {
        "security.permission"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let caps = FilesystemCapabilities::default();
        if caps.denied_paths.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                "FilesystemCapabilities::default().denied_paths is empty",
            ));
        }
        // Each adversarial path must match at least one denied pattern.
        let adversarial = &[
            "/home/u/.env",
            "/home/u/.env.production",
            "./.env.local",
            "/etc/openssl/credentials.json",
            "/srv/app/aws_credentials",
            "/home/u/.aws/credentials",
            "/var/run/secrets/api_key",
            "/etc/openssl/secret_key.pem",
            "/home/u/.config/foo.secret",
        ];
        for p in adversarial {
            let matched = caps.denied_paths.iter().any(|pat| pat.matches(p));
            if !matched {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "default denied_paths failed to match adversarial path: {p}\n\
                         patterns: {:?}",
                        caps.denied_paths
                    ),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.permission.default_no_write_no_delete ──────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.permission.default_no_write_no_delete",
        crate_name: "rullama-permission",
        invariant: "Default FilesystemCapabilities has empty write_paths and can_delete=false — destructive ops are opt-in",
        factory: || Box::new(DefaultNoWriteNoDeleteCase),
    }
}

struct DefaultNoWriteNoDeleteCase;

#[async_trait]
impl EvaluationCase for DefaultNoWriteNoDeleteCase {
    fn name(&self) -> &str {
        "sec.permission.default_no_write_no_delete"
    }
    fn category(&self) -> &str {
        "security.permission"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let caps = FilesystemCapabilities::default();
        if !caps.write_paths.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "default write_paths must be empty (write is opt-in), got {:?}",
                    caps.write_paths
                ),
            ));
        }
        if caps.can_delete {
            return Ok(TrialResult::failure(
                0,
                0,
                "default can_delete must be false (delete is opt-in)",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.permission.full_caps_clears_denies ─────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.permission.full_caps_explicitly_clears_denies",
        crate_name: "rullama-permission",
        invariant: "FilesystemCapabilities::full() clears denied_paths to empty — `full` means full, not `full-except-default-denies`",
        factory: || Box::new(FullCapsClearsDeniesCase),
    }
}

struct FullCapsClearsDeniesCase;

#[async_trait]
impl EvaluationCase for FullCapsClearsDeniesCase {
    fn name(&self) -> &str {
        "sec.permission.full_caps_explicitly_clears_denies"
    }
    fn category(&self) -> &str {
        "security.permission"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let full = FilesystemCapabilities::full();
        if !full.denied_paths.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "FilesystemCapabilities::full() has unexpected denied_paths: {:?} \
                    — callers explicitly choosing `full` should not inherit silent denies",
                    full.denied_paths
                ),
            ));
        }
        if !full.can_delete {
            return Ok(TrialResult::failure(
                0,
                0,
                "FilesystemCapabilities::full() must allow delete",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
