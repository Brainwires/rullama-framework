//! Tier-B: ChatAgent auto-compact triggers before context-window overflow.
//!
//! Invariant: when `with_auto_compact_at(threshold)` is set and history
//! token-estimate exceeds `threshold`, `compact_history` runs BEFORE the
//! next provider call. Catches the "agent grew its own history past the
//! context window via tool loops" failure mode that surfaces as a hard
//! provider error today.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, Message, Provider, ToolContext};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_inference::{AgentBuilder, ChatAgent};
use rullama_test_fixtures::ScriptedProvider;
use rullama_tool_builtins::BuiltinToolExecutor;
use rullama_tool_runtime::{ToolExecutor, ToolRegistry};

use crate::registry::SecurityCase;

inventory::submit! {
    SecurityCase {
        id: "sec.inference.context_auto_compact_fires_before_overflow",
        crate_name: "rullama-inference",
        invariant: "ChatAgent::with_auto_compact_at(threshold) triggers compact_history before the next provider call once estimated tokens > threshold",
        factory: || Box::new(AutoCompactCase),
    }
}

struct AutoCompactCase;

fn fake_executor() -> Arc<dyn ToolExecutor> {
    Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ))
}

#[async_trait]
impl EvaluationCase for AutoCompactCase {
    fn name(&self) -> &str {
        "sec.inference.context_auto_compact_fires_before_overflow"
    }
    fn category(&self) -> &str {
        "security.inference"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // 1. Without a threshold set, agent never auto-compacts. Build a
        //    50-message history and confirm message_count stays at 51
        //    (50 + the user message we send) after one round.
        {
            let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::always_text("test", "ok"));
            let mut agent: ChatAgent = AgentBuilder::new()
                .provider(provider)
                .tools(fake_executor())
                .options(ChatOptions::default())
                .build_chat_agent()?;
            // Seed 50 messages directly. Each is ~80 chars → ~20 tokens.
            let mut seeded = Vec::new();
            for i in 0..50 {
                seeded.push(Message::user(format!(
                    "padding message {i} {}",
                    "x".repeat(64)
                )));
            }
            agent.restore_messages(seeded);
            let _ = agent.process_message("ping").await?;
            // 50 seeded + 1 new user + 1 assistant = 52
            if agent.message_count() != 52 {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "without auto-compact, expected 52 messages, got {}",
                        agent.message_count()
                    ),
                ));
            }
        }

        // 2. With threshold set BELOW the seeded estimate, the agent must
        //    compact before issuing the provider call. Without a summarizer
        //    attached, `compact_history` falls back to a 20-message trim.
        {
            let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::always_text("test", "ok"));
            let mut agent: ChatAgent = AgentBuilder::new()
                .provider(provider)
                .tools(fake_executor())
                .options(ChatOptions::default())
                .build_chat_agent()?;
            // Seed history same as case 1.
            let mut seeded = Vec::new();
            for i in 0..50 {
                seeded.push(Message::user(format!(
                    "padding message {i} {}",
                    "x".repeat(64)
                )));
            }
            agent.restore_messages(seeded);
            // 50 messages × ~80 chars ÷ 4 ≈ 1000 tokens. Set threshold low
            // so compact fires on the very first iteration.
            agent = agent.with_auto_compact_at(50);
            let _ = agent.process_message("ping").await?;
            // After compact_history (fallback trim keeps 20), we add user + assistant.
            // Result: 20 + 1 + 1 = 22 (with some slack — trim semantics).
            if agent.message_count() >= 52 {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "with auto-compact threshold=50 below ~1000-token history, expected message_count to drop below 52, got {}",
                        agent.message_count()
                    ),
                ));
            }
        }

        // 3. With threshold set ABOVE the estimate, compact must NOT fire.
        {
            let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::always_text("test", "ok"));
            let mut agent: ChatAgent = AgentBuilder::new()
                .provider(provider)
                .tools(fake_executor())
                .options(ChatOptions::default())
                .build_chat_agent()?;
            let mut seeded = Vec::new();
            for i in 0..50 {
                seeded.push(Message::user(format!(
                    "padding message {i} {}",
                    "x".repeat(64)
                )));
            }
            agent.restore_messages(seeded);
            // 1_000_000 tokens — way above the ~1000 estimate. Should not fire.
            agent = agent.with_auto_compact_at(1_000_000);
            let _ = agent.process_message("ping").await?;
            if agent.message_count() != 52 {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "with auto-compact threshold=1M, compact must not fire; expected 52 messages, got {}",
                        agent.message_count()
                    ),
                ));
            }
        }

        Ok(TrialResult::success(0, 0))
    }
}
