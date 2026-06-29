//! D.14 — `live.openai.json_schema_structured_output`. Send a request with
//! OpenAI's `response_format: { type: "json_schema", json_schema: { strict: true, schema: ... } }`
//! and feed the resulting `content` through `SchemaValidator`. Verifies the
//! framework's schema-validation layer plays nicely with the provider's
//! native structured-output mode.
//!
//! This case talks to OpenAI directly via `reqwest` rather than through
//! `OpenAiChatProvider`, because the framework's `Provider::chat` doesn't
//! yet plumb `response_format` end-to-end (a deferred follow-up).

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_reasoning::SchemaValidator;
use serde_json::{Value, json};

use crate::live::{live_openai_key, live_openai_model};
use crate::registry::LiveCase;

pub struct OpenAiJsonSchemaStructuredOutput;

#[async_trait]
impl EvaluationCase for OpenAiJsonSchemaStructuredOutput {
    fn name(&self) -> &str {
        "live.openai.json_schema_structured_output"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(key) = live_openai_key() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "RULLAMA_LIVE_OPENAI_KEY not set",
            ));
        };
        let model = live_openai_model();
        let started = std::time::Instant::now();

        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age":  { "type": "integer", "minimum": 0 }
            },
            "required": ["name", "age"],
            "additionalProperties": false
        });

        let body = json!({
            "model": model,
            "messages": [
                {"role": "user", "content": "Make up a fictional person; return their name and age."}
            ],
            "max_completion_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": schema,
                }
            }
        });

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            let txt = String::from_utf8_lossy(&bytes).to_string();
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("OpenAI HTTP {status}: {txt}"),
            ));
        }

        let payload: Value = serde_json::from_slice(&bytes)?;
        let content = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing choices[0].message.content in response"))?;

        let parsed: Value = serde_json::from_str(content)?;
        let validator = SchemaValidator::new(&schema)?;
        let elapsed = started.elapsed().as_millis() as u64;
        match validator.validate(&parsed) {
            Ok(()) => Ok(TrialResult::success(trial_id, elapsed)
                .with_meta("parsed_name", parsed["name"].clone())
                .with_meta("parsed_age", parsed["age"].clone())),
            Err(errs) => Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!(
                    "schema_validation failed on supposedly-strict response: {}; raw content: {content}",
                    errs.join("; ")
                ),
            )),
        }
    }
}

inventory::submit! {
    LiveCase {
        id: "live.openai.json_schema_structured_output",
        provider: "openai",
        description: "OpenAI strict JSON-schema response_format produces SchemaValidator-clean output",
        factory: || Box::new(OpenAiJsonSchemaStructuredOutput),
    }
}
