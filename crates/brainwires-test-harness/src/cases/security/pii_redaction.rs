//! Tier-B `sec.tool_runtime.pii_redaction_catches_ssn_email_phone`. The
//! opt-in PII path in `brainwires_tool_runtime::sanitization` must:
//! - leave inputs untouched when `redact_pii=false` (preserving the
//!   historical default behaviour),
//! - strip every SSN / email / phone / credit-card pattern when
//!   `redact_pii=true`,
//! - not mutate the non-PII surrounding text.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::ContentSource;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_tool_runtime::{
    contains_pii, redact_pii, wrap_with_content_source, wrap_with_content_source_with_pii,
};

use crate::registry::SecurityCase;

pub struct PiiRedactionCatchesSsnEmailPhone;

const PII_HEAVY: &str = "Customer: alice@example.com, SSN 123-45-6789, \
                         phone +1 555-867-5309, card 4111 2222 3333 4444. \
                         Order ID #ord-9999 confirmed.";

#[async_trait]
impl EvaluationCase for PiiRedactionCatchesSsnEmailPhone {
    fn name(&self) -> &str {
        "sec.tool_runtime.pii_redaction_catches_ssn_email_phone"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();

        // 1) Detection
        if !contains_pii(PII_HEAVY) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "contains_pii returned false on PII-heavy input",
            ));
        }
        let clean = "Order ID #ord-9999 confirmed.";
        if contains_pii(clean) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "contains_pii returned true on PII-free input",
            ));
        }

        // 2) Redaction touches every pattern.
        let redacted = redact_pii(PII_HEAVY);
        for needle in [
            "alice@example.com",
            "123-45-6789",
            "555-867-5309",
            "4111 2222 3333 4444",
        ] {
            if redacted.contains(needle) {
                return Ok(TrialResult::failure(
                    trial_id,
                    started.elapsed().as_millis() as u64,
                    format!("redacted output still contains PII: {needle}"),
                ));
            }
        }
        // The non-PII text must survive.
        if !redacted.contains("Order ID #ord-9999 confirmed.") {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("redaction ate non-PII text: {redacted}"),
            ));
        }
        // Each pattern should leave a labelled marker behind.
        for label in ["pii-email", "ssn", "phone", "credit-card"] {
            let marker = format!("[REDACTED: {label}]");
            if !redacted.contains(&marker) {
                return Ok(TrialResult::failure(
                    trial_id,
                    started.elapsed().as_millis() as u64,
                    format!("missing redaction marker {marker} in: {redacted}"),
                ));
            }
        }

        // 3) Default wrap_with_content_source path is untouched for non-
        //    external sources (returns content verbatim).
        let default_wrap = wrap_with_content_source(PII_HEAVY, ContentSource::UserInput);
        if default_wrap != PII_HEAVY {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "wrap_with_content_source(UserInput) altered content with no opt-in",
            ));
        }
        // Non-external source with redact_pii=true should still redact even
        // though injection-sanitisation doesn't run.
        let opt_in = wrap_with_content_source_with_pii(
            PII_HEAVY,
            ContentSource::UserInput,
            true,
        );
        if opt_in.contains("alice@example.com") {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "opt-in PII path left email un-redacted",
            ));
        }
        // External-content path with opt-in also redacts.
        let opt_in_external = wrap_with_content_source_with_pii(
            PII_HEAVY,
            ContentSource::ExternalContent,
            true,
        );
        if opt_in_external.contains("123-45-6789") {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "external-content opt-in left SSN un-redacted",
            ));
        }
        if !opt_in_external.contains("[EXTERNAL CONTENT") {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "external-content wrapper missing on opt-in path",
            ));
        }

        let elapsed = started.elapsed().as_millis() as u64;
        Ok(TrialResult::success(trial_id, elapsed))
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.tool_runtime.pii_redaction_catches_ssn_email_phone",
        crate_name: "brainwires-tool-runtime",
        invariant: "opt-in PII redaction catches SSN, email, phone, credit-card without altering non-PII text",
        factory: || Box::new(PiiRedactionCatchesSsnEmailPhone),
    }
}
