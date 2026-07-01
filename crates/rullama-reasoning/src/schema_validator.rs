//! JSON Schema validation for LLM output, with a retry helper for agents
//! that want to enforce structured responses.
//!
//! The [`SchemaValidator`] wraps a compiled `jsonschema` validator and
//! exposes a single `validate` method that returns a list of human-readable
//! violation strings. The [`retry_until_valid`] free function is a thin
//! orchestrator for the common pattern: ask the producer for output,
//! validate it against the schema, and on violation re-invoke the producer
//! with a corrective instruction injected into the prompt.
//!
//! Only available behind the `schema-validation` feature so the reasoning
//! crate stays light when callers don't need it. The module is gated at its
//! declaration in `lib.rs`, so no inner `#![cfg]` is needed here.

use std::future::Future;

use anyhow::{Result, anyhow};
use jsonschema::Validator;
use serde_json::Value;

/// Compiled JSON Schema validator. Cheap to clone — internally `Arc`-shared.
#[derive(Clone)]
pub struct SchemaValidator {
    validator: std::sync::Arc<Validator>,
}

impl SchemaValidator {
    /// Compile a schema from a `serde_json::Value`. Returns an error if the
    /// schema itself is invalid (unparseable or not a valid draft).
    pub fn new(schema: &Value) -> Result<Self> {
        let validator =
            jsonschema::validator_for(schema).map_err(|e| anyhow!("invalid JSON Schema: {e}"))?;
        Ok(Self {
            validator: std::sync::Arc::new(validator),
        })
    }

    /// Validate `value` against the schema. Returns `Ok(())` if it passes;
    /// otherwise a list of one human-readable error per violating
    /// instance (`instance_path: error`).
    pub fn validate(&self, value: &Value) -> std::result::Result<(), Vec<String>> {
        let errors: Vec<String> = self
            .validator
            .iter_errors(value)
            .map(|e| format!("{}: {e}", e.instance_path))
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Call `producer` up to `max_attempts` times, validating each result
/// against `validator`. On violation, `producer` is invoked again with the
/// list of error strings — the closure is expected to surface those to the
/// LLM (e.g. via a corrective user message) so the next attempt converges.
///
/// Returns `Ok(value)` on the first attempt that validates. Returns the
/// final attempt's value wrapped in an error containing all the errors
/// after `max_attempts` exhaustion.
pub async fn retry_until_valid<F, Fut>(
    validator: &SchemaValidator,
    max_attempts: usize,
    mut producer: F,
) -> Result<Value>
where
    F: FnMut(Option<Vec<String>>) -> Fut,
    Fut: Future<Output = Result<Value>>,
{
    let mut last_errors: Option<Vec<String>> = None;
    for attempt in 0..max_attempts.max(1) {
        let value = producer(last_errors.clone()).await?;
        match validator.validate(&value) {
            Ok(()) => return Ok(value),
            Err(errors) => {
                tracing::warn!(
                    attempt = attempt + 1,
                    max = max_attempts,
                    "schema validation failed: {} violation(s)",
                    errors.len()
                );
                last_errors = Some(errors);
            }
        }
    }
    Err(anyhow!(
        "schema validation failed after {} attempts: {}",
        max_attempts,
        last_errors.unwrap_or_default().join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn person_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer", "minimum": 0 }
            },
            "required": ["name", "age"]
        })
    }

    #[test]
    fn valid_value_passes() {
        let v = SchemaValidator::new(&person_schema()).unwrap();
        assert!(v.validate(&json!({"name": "Ada", "age": 36})).is_ok());
    }

    #[test]
    fn missing_required_field_fails() {
        let v = SchemaValidator::new(&person_schema()).unwrap();
        let result = v.validate(&json!({"name": "Ada"}));
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("age")));
    }

    #[test]
    fn wrong_type_fails() {
        let v = SchemaValidator::new(&person_schema()).unwrap();
        let result = v.validate(&json!({"name": "Ada", "age": "old"}));
        assert!(result.is_err());
    }

    #[test]
    fn invalid_schema_returns_error() {
        // `type` must be a string or array — a number is invalid.
        let result = SchemaValidator::new(&json!({"type": 42}));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn retry_converges_on_valid() {
        let schema = SchemaValidator::new(&person_schema()).unwrap();
        let mut call = 0;
        let result = retry_until_valid(&schema, 3, |errors| {
            call += 1;
            let value = if errors.is_none() {
                json!({"name": "Ada"}) // missing age
            } else {
                json!({"name": "Ada", "age": 36})
            };
            async move { Ok(value) }
        })
        .await
        .unwrap();
        assert_eq!(result, json!({"name": "Ada", "age": 36}));
        assert_eq!(call, 2);
    }

    #[tokio::test]
    async fn retry_exhausts_then_errors() {
        let schema = SchemaValidator::new(&person_schema()).unwrap();
        let result = retry_until_valid(&schema, 2, |_errors| async {
            Ok(json!({"name": "Ada"})) // always missing age
        })
        .await;
        assert!(result.is_err());
    }
}
