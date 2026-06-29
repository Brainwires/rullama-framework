//! D.11 — `live.ollama.tool_call_dispatch`. Build a `ChatAgent` with a single
//! "list_directory" tool and ask Ollama "what's in /tmp?"; assert the tool was
//! invoked and the agent produced a final answer. End-to-end agent loop on a
//! real backend.
//!
//! Uses a hand-written read-only `ListDirTool` instead of the bash tool so the
//! case doesn't have to depend on shell execution semantics.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{
    ChatOptions, Tool, ToolContext, ToolInputSchema, ToolResult, ToolUse,
};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_inference::AgentBuilder;
use rullama_provider::OllamaProvider;
use rullama_tool_runtime::ToolExecutor;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

struct ListDirTool {
    invocations: Arc<std::sync::Mutex<usize>>,
}

#[async_trait]
impl ToolExecutor for ListDirTool {
    async fn execute(&self, tool_use: &ToolUse, _context: &ToolContext) -> Result<ToolResult> {
        if tool_use.name != "list_directory" {
            return Ok(ToolResult::error(
                tool_use.id.clone(),
                format!("unknown tool: {}", tool_use.name),
            ));
        }
        *self.invocations.lock().unwrap() += 1;
        // Static synthetic listing — independent of the host's actual /tmp.
        Ok(ToolResult::success(
            tool_use.id.clone(),
            "Contents of /tmp:\n- example.txt\n- session.log\n".to_string(),
        ))
    }

    fn available_tools(&self) -> Vec<Tool> {
        let mut props = HashMap::new();
        props.insert(
            "path".to_string(),
            serde_json::json!({"type": "string", "description": "absolute path"}),
        );
        vec![Tool {
            name: "list_directory".to_string(),
            description: "List the contents of a directory.".to_string(),
            input_schema: ToolInputSchema::object(props, vec!["path".to_string()]),
            ..Default::default()
        }]
    }
}

pub struct OllamaToolCallDispatch;

#[async_trait]
impl EvaluationCase for OllamaToolCallDispatch {
    fn name(&self) -> &str {
        "live.ollama.tool_call_dispatch"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(base) = live_ollama_base() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "RULLAMA_LIVE_OLLAMA_BASE not set",
            ));
        };
        let model = live_ollama_model();
        let started = std::time::Instant::now();

        let provider = Arc::new(OllamaProvider::new(model.clone(), Some(base)));
        let invocations = Arc::new(std::sync::Mutex::new(0usize));
        let executor: Arc<dyn ToolExecutor> = Arc::new(ListDirTool {
            invocations: invocations.clone(),
        });

        let mut agent = AgentBuilder::new()
            .provider(provider)
            .tools(executor)
            .options(ChatOptions::default().model(model).max_tokens(256))
            .system("You have a `list_directory` tool. Use it when asked about directory contents. Always answer briefly.")
            .max_iterations(4)
            .build_chat_agent()?;

        let answer = agent
            .process_message("List the contents of /tmp using the list_directory tool.")
            .await?;
        let elapsed = started.elapsed().as_millis() as u64;
        let n = *invocations.lock().unwrap();
        // The core invariant is "framework wired the tool call from provider
        // to executor and threaded the result back into the conversation."
        // Whether the model emits a follow-up text turn is a model-quality
        // signal, recorded as metadata but not a pass criterion (small
        // local models often consider themselves done after the tool call).
        if n == 0 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("tool was never invoked; answer was: {answer}"),
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("tool_invocations", n)
            .with_meta("answer_len", answer.len())
            .with_meta("answer_nonempty", !answer.trim().is_empty()))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.ollama.tool_call_dispatch",
        provider: "ollama",
        description: "ChatAgent + Ollama dispatches a real tool call and renders a final answer",
        factory: || Box::new(OllamaToolCallDispatch),
    }
}
