//! Tier-A `feature.provider.anthropic_cache_control_emitted_per_strategy`.
//! Inspect the JSON body the Anthropic client produces for each
//! `CacheStrategy` variant and verify the right breakpoints land in the
//! right places:
//!
//! - `Off`            → no `cache_control` anywhere.
//! - `SystemOnly`     → cache_control on system; not on tools, not on tail.
//! - `SystemAndTools` → cache_control on system + last tool; not on tail.
//! - `SystemAndTailTurn { threshold_tokens }`:
//!    * if msgs total < threshold → behaves like `SystemAndTools`.
//!    * if msgs total ≥ threshold → also adds cache_control on the last
//!      message's tail content block.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::CacheStrategy;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider::AnthropicClient;
use serde_json::Value;

use crate::registry::TierACase;

pub struct AnthropicCacheControlEmittedPerStrategy;

/// Build a minimal Anthropic request shape via the framework's public
/// chat-provider path: serialise the result of `build_anthropic_request`
/// to JSON and inspect. We use the client's `messages` API indirectly by
/// calling `AnthropicClient::build_request_body_for_test`.
///
/// The client exposes that helper specifically for tests so this case can
/// avoid the network entirely.
fn build_body_for(strategy: CacheStrategy, large_history: bool) -> Result<Value> {
    use brainwires_provider::anthropic::{
        AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicTool,
    };
    let mut messages = vec![AnthropicMessage {
        role: "user".to_string(),
        content: vec![AnthropicContentBlock::Text {
            text: "hi".to_string(),
        }],
    }];
    if large_history {
        // Pad the user message past the 2000-token threshold (chars/4) by
        // making the content ~10_000 chars.
        messages[0].content = vec![AnthropicContentBlock::Text {
            text: "x".repeat(10_000),
        }];
    }
    let req = AnthropicRequest {
        model: "claude-haiku-4-5".to_string(),
        messages,
        system: Some("you are helpful".to_string()),
        max_tokens: 64,
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: Some(vec![
            AnthropicTool {
                name: "calculator".to_string(),
                description: "do math".to_string(),
                input_schema: Default::default(),
            },
            AnthropicTool {
                name: "search".to_string(),
                description: "search".to_string(),
                input_schema: Default::default(),
            },
        ]),
        stream: false,
        cache_strategy: strategy,
    };
    // Build via the client's public test helper.
    let client = AnthropicClient::new("fake-key".to_string(), "claude-haiku-4-5".to_string());
    Ok(client.build_request_body_for_test(&req))
}

fn system_has_cache_control(body: &Value) -> bool {
    body.get("system")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("cache_control"))
        .is_some()
}

fn last_tool_has_cache_control(body: &Value) -> bool {
    body.get("tools")
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.last())
        .and_then(|item| item.get("cache_control"))
        .is_some()
}

fn last_message_content_has_cache_control(body: &Value) -> bool {
    body.get("messages")
        .and_then(|m| m.as_array())
        .and_then(|arr| arr.last())
        .and_then(|msg| msg.get("content"))
        .and_then(|content| content.as_array())
        .and_then(|blocks| blocks.last())
        .and_then(|block| block.get("cache_control"))
        .is_some()
}

#[async_trait]
impl EvaluationCase for AnthropicCacheControlEmittedPerStrategy {
    fn name(&self) -> &str {
        "feature.provider.anthropic_cache_control_emitted_per_strategy"
    }
    fn category(&self) -> &str {
        "feature"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();

        // Off
        let body = build_body_for(CacheStrategy::Off, false)?;
        if system_has_cache_control(&body)
            || last_tool_has_cache_control(&body)
            || last_message_content_has_cache_control(&body)
        {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("Off emitted cache_control: {body}"),
            ));
        }

        // SystemOnly
        let body = build_body_for(CacheStrategy::SystemOnly, false)?;
        if !system_has_cache_control(&body) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemOnly missing system cache_control: {body}"),
            ));
        }
        if last_tool_has_cache_control(&body) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemOnly emitted tool cache_control: {body}"),
            ));
        }

        // SystemAndTools
        let body = build_body_for(CacheStrategy::SystemAndTools, false)?;
        if !system_has_cache_control(&body) || !last_tool_has_cache_control(&body) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemAndTools missing breakpoints: {body}"),
            ));
        }
        if last_message_content_has_cache_control(&body) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemAndTools emitted tail cache_control: {body}"),
            ));
        }

        // SystemAndTailTurn — short history (under threshold) — no tail.
        let body = build_body_for(
            CacheStrategy::SystemAndTailTurn {
                threshold_tokens: 2000,
            },
            false,
        )?;
        if last_message_content_has_cache_control(&body) {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemAndTailTurn(short) emitted tail cache_control: {body}"),
            ));
        }

        // SystemAndTailTurn — large history (past threshold) — tail must fire.
        let body = build_body_for(
            CacheStrategy::SystemAndTailTurn {
                threshold_tokens: 2000,
            },
            true,
        )?;
        if !system_has_cache_control(&body)
            || !last_tool_has_cache_control(&body)
            || !last_message_content_has_cache_control(&body)
        {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("SystemAndTailTurn(large) missing breakpoints: {body}"),
            ));
        }

        let elapsed = started.elapsed().as_millis() as u64;
        Ok(TrialResult::success(trial_id, elapsed))
    }
}

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::anthropic_cache_control::AnthropicCacheControlEmittedPerStrategy",
        crate_name: "brainwires-provider",
        description: "Anthropic request body emits cache_control breakpoints per CacheStrategy variant",
        factory: || Box::new(AnthropicCacheControlEmittedPerStrategy),
    }
}
