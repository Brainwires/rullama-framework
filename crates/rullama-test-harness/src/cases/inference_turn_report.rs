//! Tier-A: `ChatAgent::process_message_with_report` returns a TurnReport
//! whose token counts aggregate the provider calls in that single turn,
//! distinct from the cumulative_usage tracked across the session.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, ChatResponse, Message, Provider, ToolContext, Usage};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_inference::AgentBuilder;
use rullama_test_fixtures::ScriptedProvider;
use rullama_tool_builtins::BuiltinToolExecutor;
use rullama_tool_runtime::{ToolExecutor, ToolRegistry};

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::inference_turn_report::per_turn_report_aggregates_usage",
        crate_name: "rullama-inference",
        description: "process_message_with_report returns a TurnReport diffing cumulative_usage before/after; second turn's report does not include first turn's tokens",
        factory: || Box::new(TurnReportCase),
    }
}

struct TurnReportCase;

fn fake_executor() -> Arc<dyn ToolExecutor> {
    Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ))
}

#[async_trait]
impl EvaluationCase for TurnReportCase {
    fn name(&self) -> &str {
        "feature.inference.per_turn_report_aggregates_usage"
    }
    fn category(&self) -> &str {
        "feature.inference"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // ScriptedProvider::always_response lets us pin the Usage on every
        // response. Each turn returns 30+20 = 50 tokens.
        let canned = ChatResponse {
            message: Message::assistant("done"),
            usage: Usage::new(30, 20),
            finish_reason: Some("stop".into()),
        };
        let provider: Arc<dyn Provider> =
            Arc::new(ScriptedProvider::always_response("test", canned));
        let mut agent = AgentBuilder::new()
            .provider(provider)
            .tools(fake_executor())
            .options(ChatOptions::default())
            .max_iterations(3)
            .build_chat_agent()?;

        // First turn: one provider call → 50 tokens; duration > 0.
        let (_text1, r1) = agent.process_message_with_report("hi").await?;
        if r1.total_tokens != 50 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("turn 1: expected 50 total_tokens, got {}", r1.total_tokens),
            ));
        }
        if r1.prompt_tokens != 30 || r1.completion_tokens != 20 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "turn 1: expected (30,20), got ({},{})",
                    r1.prompt_tokens, r1.completion_tokens
                ),
            ));
        }
        // cost not plumbed yet (Tier 2.4)
        if r1.cost_usd_cents.is_some() {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "turn 1: cost_usd_cents should be None until Tier 2.4, got {:?}",
                    r1.cost_usd_cents
                ),
            ));
        }

        // Second turn: must NOT include first turn's tokens (delta, not cumulative).
        let (_text2, r2) = agent.process_message_with_report("again").await?;
        if r2.total_tokens != 50 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "turn 2: expected 50 total_tokens (delta), got {} — looks cumulative",
                    r2.total_tokens
                ),
            ));
        }

        // Cumulative still reflects both turns combined.
        let cum = agent.cumulative_usage();
        if cum.total_tokens != 100 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "cumulative_usage.total_tokens after 2 turns = {}, expected 100",
                    cum.total_tokens
                ),
            ));
        }

        Ok(TrialResult::success(0, 0))
    }
}
