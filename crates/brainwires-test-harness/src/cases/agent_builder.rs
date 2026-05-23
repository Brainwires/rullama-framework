//! Tier-A feature case for `brainwires_inference::AgentBuilder`.
//!
//! Three invariants:
//! 1. Missing `provider` produces a clear error mentioning the missing field.
//! 2. Missing `tools` produces a clear error mentioning the missing field.
//! 3. Happy path builds a working ChatAgent that completes a roundtrip
//!    against ScriptedProvider.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{ChatOptions, Provider, ToolContext};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_inference::AgentBuilder;
use brainwires_test_fixtures::ScriptedProvider;
use brainwires_tool_builtins::BuiltinToolExecutor;
use brainwires_tool_runtime::{ToolExecutor, ToolRegistry};

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::agent_builder::builder_facade",
        crate_name: "brainwires-inference",
        description: "AgentBuilder: missing provider/tools errors with named field; happy path builds a working ChatAgent",
        factory: || Box::new(BuilderFacadeCase),
    }
}

struct BuilderFacadeCase;

fn fake_executor() -> Arc<dyn ToolExecutor> {
    Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ))
}

#[async_trait]
impl EvaluationCase for BuilderFacadeCase {
    fn name(&self) -> &str {
        "feature.agent.builder_facade"
    }
    fn category(&self) -> &str {
        "feature.agent"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // 1. Missing provider → named-field error
        let r = AgentBuilder::new().tools(fake_executor()).build_chat_agent();
        match r {
            Err(e) if e.to_string().contains("`provider` is required") => {}
            Err(e) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("expected `provider` is required error, got: {e}"),
                ));
            }
            Ok(_) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    "AgentBuilder succeeded with no provider",
                ));
            }
        }

        // 2. Missing tools → named-field error
        let provider: Arc<dyn Provider> =
            Arc::new(ScriptedProvider::always_text("test", "hi"));
        let r = AgentBuilder::new()
            .provider(provider.clone())
            .build_chat_agent();
        match r {
            Err(e) if e.to_string().contains("`tools` is required") => {}
            Err(e) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("expected `tools` is required error, got: {e}"),
                ));
            }
            Ok(_) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    "AgentBuilder succeeded with no tools",
                ));
            }
        }

        // 3. Happy path — build + drive one turn against ScriptedProvider
        let mut agent = AgentBuilder::new()
            .provider(provider)
            .tools(fake_executor())
            .system("you are helpful")
            .max_iterations(5)
            .tool_concurrency(2)
            .options(ChatOptions::default())
            .build_chat_agent()?;
        let response = agent.process_message("ping").await?;
        if !response.contains("hi") {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("expected ScriptedProvider's canned response, got {response:?}"),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
