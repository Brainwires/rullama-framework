//! Chat Stream Processing
//!
//! Handles streaming chat responses with tool execution support.

use anyhow::Result;
use futures::StreamExt;
use indicatif::ProgressBar;
use std::sync::Arc;

use super::continuation::{default_logger, send_continuation_request};
use crate::providers::Provider;
use crate::tools::ToolExecutor;
use crate::types::agent::{AgentContext, PermissionMode};
use crate::types::message::StreamChunk;
use crate::types::provider::ChatOptions;
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};

/// Process chat stream with tool execution support
pub async fn process_chat_stream(
    provider: &Arc<dyn Provider>,
    context: &AgentContext,
    spinner: &Option<ProgressBar>,
    model: &str,
    chat_id: Option<String>,
) -> Result<String> {
    use crate::types::message::Role;

    let mut full_text = String::new();
    let tool_executor = ToolExecutor::new(PermissionMode::Auto);
    // Accumulate usage across the stream so we can record a single cost event
    // at the end. Some providers emit multiple `Usage` chunks (e.g. continuation
    // after tool calls); we sum them and then persist once to avoid hammering
    // the cost tracker file.
    let mut total_prompt_tokens: u32 = 0;
    let mut total_completion_tokens: u32 = 0;
    let mut got_usage = false;

    // Extract system prompt from conversation history
    let system_prompt = context
        .conversation_history
        .iter()
        .find(|m| m.role == Role::System)
        .and_then(|m| m.text().map(|s| s.to_string()));

    let options = ChatOptions {
        temperature: Some(0.7),
        max_tokens: Some(4096),
        top_p: None,
        stop: None,
        system: system_prompt,
        model: None,
        cache_strategy: Default::default(),
    };

    let mut stream = provider.stream_chat(
        &context.conversation_history,
        Some(&context.tools),
        &options,
    );

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        match chunk {
            StreamChunk::Text(text) => {
                full_text.push_str(&text);
            }
            StreamChunk::ToolCall {
                call_id,
                response_id,
                chat_id: tool_chat_id,
                tool_name,
                server,
                parameters,
            } => {
                // Tool call received from backend
                if let Some(s) = spinner {
                    s.set_message("Executing tool...");
                }

                eprintln!(
                    "\n🔧 Tool requested: {} (server: {})",
                    console::style(&tool_name).cyan().bold(),
                    console::style(&server).dim()
                );

                // Only execute if it's a cli-local tool
                if server == "cli-local" {
                    // Create ToolUse from the call
                    let tool_use = ToolUse {
                        id: call_id.clone(),
                        name: tool_name.clone(),
                        input: parameters.clone(),
                    };

                    // Execute tool locally
                    let tool_context = ToolContext::from_agent_context(context);

                    let result = tool_executor.execute(&tool_use, &tool_context).await?;

                    // Limit tool output to prevent context window overflow
                    const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
                    let truncated_output = if result.content.len() > MAX_TOOL_OUTPUT_CHARS {
                        let truncated = &result.content[..MAX_TOOL_OUTPUT_CHARS];
                        let lines_count = result.content.lines().count();
                        let truncated_lines = truncated.lines().count();
                        format!(
                            "{}\n\n[Output truncated: showing first {} of {} lines ({} of {} characters)]",
                            truncated,
                            truncated_lines,
                            lines_count,
                            MAX_TOOL_OUTPUT_CHARS,
                            result.content.len()
                        )
                    } else {
                        result.content.clone()
                    };

                    if result.is_error {
                        eprintln!(
                            "❌ Tool {} failed: {}\n",
                            console::style(&tool_name).red(),
                            console::style(&result.content).dim()
                        );
                    } else {
                        let preview = if truncated_output.len() > 200 {
                            format!("{}...", &truncated_output[..200])
                        } else {
                            truncated_output.clone()
                        };
                        eprintln!(
                            "✅ Tool {} completed: {}\n",
                            console::style(&tool_name).green(),
                            console::style(preview).dim()
                        );
                    }

                    // Send continuation request to backend with tool result
                    if let Some(s) = spinner {
                        s.set_message("Processing tool result...");
                    }

                    let continuation_text = send_continuation_request(
                        provider,
                        context,
                        model,
                        tool_chat_id.or_else(|| chat_id.clone()),
                        &response_id,
                        &call_id,
                        &tool_name,
                        &parameters,
                        &truncated_output,
                        &[], // Empty accumulated history for first tool call
                        default_logger(),
                    )
                    .await?;

                    full_text.push_str(&continuation_text);

                    // Tool execution complete - stop reading from original stream
                    break;
                } else {
                    eprintln!("⚠️  Ignoring tool from unknown server: {}\n", server);
                }
            }
            // The brainwires HTTP provider emits StreamChunk::Usage exactly once per
            // SSE stream inside the "complete" event, with cumulative totals for that
            // turn. Tool-use continuations open a new stream that emits its own cumulative
            // Usage — hence saturating_add across chunks correctly sums turn totals.
            // If counts ever look wrong, enable RUST_LOG=trace and compare per-turn
            // totals against actual reply length before touching this math.
            StreamChunk::Usage(usage) => {
                // Accumulate so `brainwires cost` has data to show.
                total_prompt_tokens = total_prompt_tokens.saturating_add(usage.prompt_tokens);
                total_completion_tokens =
                    total_completion_tokens.saturating_add(usage.completion_tokens);
                tracing::debug!(
                    prompt = total_prompt_tokens,
                    completion = total_completion_tokens,
                    "accumulated stream usage (cumulative across tool-continuations)"
                );
                got_usage = true;
            }
            StreamChunk::Done => {
                break;
            }
            StreamChunk::ToolUse { .. } | StreamChunk::ToolInputDelta { .. } => {
                // These are for other tool formats, ignore
            }
            StreamChunk::ContextCompacted { .. } => {
                // Context compaction is handled by the agent layer
            }
        }
    }

    // Persist usage to the cost tracker so `brainwires cost` can report it.
    // Best-effort: failures here should not fail the user's chat.
    if got_usage {
        let provider_name = provider.name().to_string();
        let model_name = model.to_string();
        if let Err(e) = record_usage_event(
            &provider_name,
            &model_name,
            total_prompt_tokens,
            total_completion_tokens,
        )
        .await
        {
            tracing::warn!("Failed to record usage to cost tracker: {}", e);
        }
    }

    Ok(full_text)
}

/// Load the persistent cost tracker, record a single usage event, and save.
///
/// This is the plumbing that makes `brainwires cost` non-empty after running
/// `brainwires chat --prompt ...`. The data file lives alongside other
/// Brainwires data (`~/.local/share/brainwires/cost_tracker.json` on Linux).
async fn record_usage_event(
    provider: &str,
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> anyhow::Result<()> {
    use crate::utils::cost_tracker::CostTracker;
    let mut tracker = CostTracker::load().await?;
    tracker.track_usage(provider, model, input_tokens, output_tokens);
    tracker.save().await?;
    Ok(())
}
