//! Tier-B adversarial cases for `rullama_network::auth::AuthClient`.
//!
//! Invariants:
//! - `validate_api_key_format` REJECTS keys outside the canonical
//!   `bw_(prod|dev|test)_[a-z0-9]{32}` shape.
//! - SQL-injection and prompt-injection patterns embedded in API-key
//!   inputs do not get past the regex.
//! - Empty / whitespace / multi-line inputs are rejected.

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_network::auth::client::AuthClient;

use crate::registry::SecurityCase;

const CANONICAL_PATTERN: &str = r"^bw_(prod|dev|test)_[a-z0-9]{32}$";

fn make_client() -> AuthClient {
    AuthClient::new(
        "https://example.invalid".to_string(),
        "/auth".to_string(),
        CANONICAL_PATTERN,
    )
}

// ── sec.network.api_key_accepts_canonical ───────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.network.api_key_accepts_canonical",
        crate_name: "rullama-network",
        invariant: "validate_api_key_format accepts canonical `bw_{prod|dev|test}_[a-z0-9]{32}` keys",
        factory: || Box::new(ApiKeyAcceptsCanonicalCase),
    }
}

struct ApiKeyAcceptsCanonicalCase;

#[async_trait]
impl EvaluationCase for ApiKeyAcceptsCanonicalCase {
    fn name(&self) -> &str {
        "sec.network.api_key_accepts_canonical"
    }
    fn category(&self) -> &str {
        "security.network"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let c = make_client();
        let body = "0".repeat(32);
        for env in &["prod", "dev", "test"] {
            let key = format!("bw_{env}_{body}");
            if c.validate_api_key_format(&key).is_err() {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("validate rejected canonical key {key}"),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.network.api_key_rejects_adversarial ─────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.network.api_key_rejects_adversarial",
        crate_name: "rullama-network",
        invariant: "validate_api_key_format rejects malformed envs, wrong length, SQL/prompt injection, and whitespace",
        factory: || Box::new(ApiKeyRejectsAdversarialCase),
    }
}

struct ApiKeyRejectsAdversarialCase;

#[async_trait]
impl EvaluationCase for ApiKeyRejectsAdversarialCase {
    fn name(&self) -> &str {
        "sec.network.api_key_rejects_adversarial"
    }
    fn category(&self) -> &str {
        "security.network"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let c = make_client();
        // Each entry pairs an adversarial input with the reason it must be rejected.
        let body = "0".repeat(32);
        let too_short = "0".repeat(31);
        let too_long = "0".repeat(33);
        let upper = "A".repeat(32);
        let key_too_short = format!("bw_prod_{too_short}");
        let key_too_long = format!("bw_prod_{too_long}");
        let key_upper = format!("bw_prod_{upper}");
        let multiline = format!("bw_prod_{body}\nGET /admin");
        let key_canonical = format!("bw_prod_{body}");
        let surround_ws = format!("  {key_canonical}  ");
        let adversarial: Vec<(&str, &str)> = vec![
            ("", "empty string"),
            (" ", "single space"),
            ("bw_staging_00000000000000000000000000000000", "unknown env"),
            ("BW_PROD_00000000000000000000000000000000", "uppercase prefix"),
            (&key_too_short, "31-char body (one short)"),
            (&key_too_long, "33-char body (one long)"),
            (&key_upper, "uppercase body"),
            ("bw_prod_'; DROP TABLE keys;--", "SQL injection"),
            ("bw_prod_../../etc/passwd", "path traversal"),
            (&multiline, "multi-line trailing newline"),
            (&surround_ws, "valid key with surrounding whitespace"),
        ];
        for (key, reason) in adversarial {
            if c.validate_api_key_format(key).is_ok() {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("validate accepted adversarial key ({reason}): {key:?}"),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}
