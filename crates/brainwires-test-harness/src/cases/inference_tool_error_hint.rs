//! Tier-A: ChatAgent injects a "try different inputs" hint when any tool
//! returns is_error=true, instead of looping on the same args.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{
    ChatOptions, ContentBlock, MessageContent, Provider, Tool, ToolContext, ToolResult, ToolUse,
};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_inference::AgentBuilder;
use brainwires_test_fixtures::ScriptedProvider;
use brainwires_tool_runtime::ToolExecutor;

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::inference_tool_error_hint::tool_error_rephrase_injects_hint",
        crate_name: "brainwires-inference",
        description: "ChatAgent: failing tool result appends a `[Tool error] ... try a different ...` cue into the next provider turn",
        factory: || Box::new(ToolErrorHintCase),
    }
}

struct ToolErrorHintCase;

/// Tool executor that always returns is_error=true with a known reason.
struct FailingTool;

#[async_trait]
impl ToolExecutor for FailingTool {
    async fn execute(&self, tool_use: &ToolUse, _ctx: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::error(
            tool_use.id.clone(),
            "EACCES: permission denied".to_string(),
        ))
    }
    fn available_tools(&self) -> Vec<Tool> {
        vec![Tool {
            name: "always_fails".to_string(),
            description: "Always returns an error".to_string(),
            ..Default::default()
        }]
    }
}

#[async_trait]
impl EvaluationCase for ToolErrorHintCase {
    fn name(&self) -> &str {
        "feature.inference.tool_error_rephrase_injects_hint"
    }
    fn category(&self) -> &str {
        "feature.inference"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // Script the provider to: call the failing tool, then say "done"
        // on the second turn. The agent loop should:
        //   1. Stream a tool call (always_fails)
        //   2. Execute it → ToolResult { is_error: true, content: "EACCES …" }
        //   3. Push the tool-result message with the appended [Tool error] hint
        //   4. Next iteration: emit "done"
        let provider: Arc<dyn Provider> = Arc::new(
            ScriptedProvider::new("test")
                .then_tool_call(
                    "call-1",
                    "always_fails",
                    serde_json::json!({"path": "/etc/shadow"}),
                )
                .then_text("done"),
        );
        let mut agent = AgentBuilder::new()
            .provider(provider)
            .tools(Arc::new(FailingTool))
            .options(ChatOptions::default())
            .max_iterations(5)
            .build_chat_agent()?;
        let _ = agent.process_message("read /etc/shadow").await?;

        // Find the tool-result message and confirm the hint block exists.
        let mut found_hint = false;
        for m in agent.messages() {
            if let MessageContent::Blocks(blocks) = &m.content {
                for b in blocks {
                    if let ContentBlock::Text { text } = b
                        && text.starts_with("[Tool error]")
                        && text.contains("EACCES")
                        && text.contains("Reconsider the inputs")
                    {
                        found_hint = true;
                    }
                }
            }
        }
        if !found_hint {
            // Dump the message history for debugging.
            let summary = agent
                .messages()
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.text().unwrap_or("<blocks>")))
                .collect::<Vec<_>>()
                .join("\n  ");
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "expected `[Tool error] EACCES … Reconsider the inputs …` hint in history, not found.\nMessages:\n  {summary}"
                ),
            ));
        }

        // Negative case: when ALL tools succeed, no hint is injected.
        let provider_ok: Arc<dyn Provider> = Arc::new(
            ScriptedProvider::new("test")
                .then_tool_call("call-1", "always_ok", serde_json::json!({}))
                .then_text("done"),
        );

        struct OkTool;
        #[async_trait]
        impl ToolExecutor for OkTool {
            async fn execute(&self, tool_use: &ToolUse, _: &ToolContext) -> Result<ToolResult> {
                Ok(ToolResult::success(tool_use.id.clone(), "ok".to_string()))
            }
            fn available_tools(&self) -> Vec<Tool> {
                vec![Tool {
                    name: "always_ok".to_string(),
                    description: "Always succeeds".to_string(),
                    ..Default::default()
                }]
            }
        }

        let mut agent_ok = AgentBuilder::new()
            .provider(provider_ok)
            .tools(Arc::new(OkTool))
            .options(ChatOptions::default())
            .max_iterations(5)
            .build_chat_agent()?;
        let _ = agent_ok.process_message("do a thing").await?;
        for m in agent_ok.messages() {
            if let MessageContent::Blocks(blocks) = &m.content {
                for b in blocks {
                    if let ContentBlock::Text { text } = b
                        && text.starts_with("[Tool error]")
                    {
                        return Ok(TrialResult::failure(
                            0,
                            0,
                            "[Tool error] hint emitted even though all tools succeeded",
                        ));
                    }
                }
            }
        }

        Ok(TrialResult::success(0, 0))
    }
}
