//! Tier-B `sec.reasoning.schema_violation_triggers_retry`: when the producer
//! returns JSON that violates the schema, the retry orchestrator must invoke
//! the producer again (with the violations forwarded) instead of returning
//! the bad value to the caller. The third attempt succeeds; the test also
//! verifies the orchestrator gives up after the configured cap.

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_reasoning::{SchemaValidator, retry_until_valid};
use serde_json::json;

use crate::registry::SecurityCase;

pub struct SchemaViolationTriggersRetry;

#[async_trait]
impl EvaluationCase for SchemaViolationTriggersRetry {
    fn name(&self) -> &str {
        "sec.reasoning.schema_violation_triggers_retry"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let schema = SchemaValidator::new(&json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age":  { "type": "integer", "minimum": 0 }
            },
            "required": ["name", "age"]
        }))?;

        let started = std::time::Instant::now();

        // Branch 1: producer eventually returns a valid value on attempt 3,
        // after seeing two rounds of violations forwarded into its prompt.
        let mut call_count = 0usize;
        let mut errors_seen = 0usize;
        let val = retry_until_valid(&schema, 3, |errors| {
            call_count += 1;
            if errors.is_some() {
                errors_seen += 1;
            }
            let value = match call_count {
                1 => json!({"name": "Ada"}),                  // missing age
                2 => json!({"name": "Ada", "age": "thirty"}), // wrong type
                _ => json!({"name": "Ada", "age": 36}),       // valid
            };
            async move { Ok(value) }
        })
        .await?;

        let elapsed = started.elapsed().as_millis() as u64;

        if val != json!({"name": "Ada", "age": 36}) {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("expected valid object, got: {val}"),
            ));
        }
        if call_count != 3 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("producer should be invoked exactly 3 times, was {call_count}"),
            ));
        }
        if errors_seen != 2 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("producer should see errors on retries (2), saw {errors_seen}"),
            ));
        }

        // Branch 2: producer always returns invalid; orchestrator gives up
        // and returns Err after the configured retry count.
        let result = retry_until_valid(&schema, 2, |_errors| async {
            Ok(json!({"name": "Ada"})) // always missing age
        })
        .await;
        if result.is_ok() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "orchestrator should give up after max_attempts but returned Ok",
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("retries_seen", errors_seen as u64)
            .with_meta("converged_on_attempt", call_count as u64))
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.reasoning.schema_violation_triggers_retry",
        crate_name: "rullama-reasoning",
        invariant: "schema validator + retry orchestrator forwards violations and rejects after max_attempts",
        factory: || Box::new(SchemaViolationTriggersRetry),
    }
}
