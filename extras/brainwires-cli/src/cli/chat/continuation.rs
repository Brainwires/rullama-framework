//! Continuation Request Handling
//!
//! Handles sending continuation requests to the backend with tool execution results
//! and processing chained tool calls.
//!
//! ## IMPORTANT: Stateless Responses API Usage
//!
//! This CLI uses the OpenAI Responses API in STATELESS mode. This means:
//!
//! 1. **NO `previousResponseId`** - We intentionally do NOT store response IDs.
//!    The server cannot look up previous tool calls by ID.
//!
//! 2. **NO `functionCallOutput`** - This field requires `previousResponseId` to work.
//!    Without it, the server returns "No tool call found for function call output".
//!
//! 3. **FULL CONVERSATION HISTORY** - Instead, we embed tool calls and results
//!    directly in the `conversationHistory` array using special roles:
//!    - `role: "function_call"` - The AI's request to call a tool
//!    - `role: "tool"` - The tool's execution result
//!
//! The backend (openai-helpers.ts) converts these to the Responses API `input` array:
//! - `function_call` -> `{ type: "function_call", call_id, name, arguments }`
//! - `tool` -> `{ type: "function_call_output", call_id, output }`
//!
//! This allows the full tool call chain to be reconstructed from conversation history
//! on every request, enabling truly stateless operation.

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

use crate::auth::SessionManager;
use crate::debug_log;
use crate::providers::Provider;
use crate::tools::ToolExecutor;
use crate::types::agent::{AgentContext, PermissionMode};
use crate::types::message::Role;
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};

/// Logging callback type for tool execution messages
pub type LogCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Default logger that writes to stderr (for CLI mode)
pub fn default_logger() -> LogCallback {
    Arc::new(|msg: &str| {
        eprintln!("{}", msg);
    })
}

/// Send continuation request to backend with tool execution result
/// Maintains conversation history including previous tool calls for chained execution
///
/// ## Stateless Operation
///
/// This function operates in STATELESS mode - it does NOT use `previousResponseId` or
/// `functionCallOutput`. Instead, the tool call and result are embedded directly in
/// `conversationHistory` using `role: "function_call"` and `role: "tool"`.
///
/// The `logger` callback is used to output tool execution status messages.
/// Use `default_logger()` for CLI mode, or provide a custom callback for TUI mode.
#[allow(clippy::too_many_arguments)]
pub fn send_continuation_request<'a>(
    _provider: &'a Arc<dyn Provider>,
    context: &'a AgentContext,
    model: &'a str,
    chat_id: Option<String>,
    _previous_response_id: &'a str, // UNUSED - kept for API compatibility, may be removed
    call_id: &'a str,
    tool_name: &'a str,
    tool_parameters: &'a serde_json::Value,
    tool_output: &'a str,
    accumulated_history: &'a [serde_json::Value],
    logger: LogCallback,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        // Get session for backend URL
        let session = SessionManager::load()?.context("No active session found")?;

        // Get API key from secure storage (keyring or fallback)
        let api_key = SessionManager::get_api_key()?
            .context("No API key found. Please re-authenticate with: brainwires auth")?;

        let http_client = Client::new();
        let url = format!("{}/api/chat/stream", session.backend);

        // Build conversation history using shared helper that properly serializes
        // tool calls and tool results (not just text content)
        let mut conversation_history =
            crate::types::message::serialize_messages_to_stateless_history(
                &context.conversation_history,
            );

        // Add accumulated tool call history (for chained calls)
        conversation_history.extend_from_slice(accumulated_history);

        // STATELESS MODE: Embed tool call and result in conversation history
        // DO NOT use `functionCallOutput` - it requires `previousResponseId` which we don't store.
        // Instead, add the function_call and tool result to the conversation history.
        // The backend (openai-helpers.ts) will convert these to the Responses API format.

        // Add the function_call (AI's request to call the tool)
        conversation_history.push(json!({
            "role": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": tool_parameters.to_string()
        }));

        // Add the tool result (output from executing the tool)
        conversation_history.push(json!({
            "role": "tool",
            "call_id": call_id,
            "name": tool_name,
            "content": tool_output
        }));

        // Convert tools to MCP format
        let mcp_tools: Vec<serde_json::Value> = context
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "server": "cli-local",
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                })
            })
            .collect();

        // Extract system prompt from conversation history
        let system_prompt = context
            .conversation_history
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()));

        // Build continuation request payload using Brainwires backend format
        // NOTE: We do NOT use `functionCallOutput` or `previousResponseId` here!
        // We operate in STATELESS mode - the full tool call chain is in conversationHistory.
        let mut request_body = json!({
            "chatId": chat_id,
            "content": "",
            "model": model,
            "timezone": "UTC",
            "conversationHistory": conversation_history
        });

        // Include system prompt if present
        if let Some(ref prompt) = system_prompt {
            request_body["systemPrompt"] = json!(prompt);
        }

        // Include tools to allow more tool calls
        if !mcp_tools.is_empty() {
            request_body["selectedMCPTools"] = json!(mcp_tools);
        }

        // Send request
        let response = http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key.as_str()))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send continuation request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Continuation request failed ({}): {}",
                status,
                error_text
            ));
        }

        // Parse SSE stream from continuation
        debug_log!("🔄 DEBUG: Parsing continuation SSE stream...");
        let (full_text, pending_tool_calls) = parse_sse_stream(response, logger.clone()).await?;
        debug_log!(
            "🔄 DEBUG: Continuation returned {} chars, {} pending tool calls",
            full_text.len(),
            pending_tool_calls.len()
        );

        // Execute any pending tool calls
        let final_text = execute_chained_tools(
            _provider,
            context,
            model,
            chat_id,
            call_id,
            tool_name,
            tool_parameters,
            tool_output,
            accumulated_history,
            &full_text,
            pending_tool_calls,
            logger,
        )
        .await?;

        Ok(final_text)
    })
}

/// Parse SSE stream and collect pending tool calls
async fn parse_sse_stream(
    response: reqwest::Response,
    logger: LogCallback,
) -> Result<(
    String,
    Vec<(
        String,
        String,
        Option<String>,
        String,
        String,
        serde_json::Value,
    )>,
)> {
    let mut full_text = String::new();
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut stream_done = false;
    let mut pending_tool_calls: Vec<(
        String,
        String,
        Option<String>,
        String,
        String,
        serde_json::Value,
    )> = Vec::new();

    loop {
        if stream_done {
            break;
        }

        let chunk_result = match stream.next().await {
            Some(result) => result,
            None => break,
        };

        let chunk = chunk_result.context("Failed to read stream chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events (delimited by \n\n)
        while let Some(pos) = buffer.find("\n\n") {
            if stream_done {
                break;
            }

            let event_block = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            // Parse SSE event block
            let mut event_type = None;
            let mut event_data = None;

            for line in event_block.lines() {
                if let Some(evt) = line.strip_prefix("event: ") {
                    event_type = Some(evt.to_string());
                } else if let Some(data) = line.strip_prefix("data: ") {
                    event_data = Some(data.to_string());
                }
            }

            if let (Some(evt_type), Some(data)) = (event_type, event_data) {
                match evt_type.as_str() {
                    "delta" => {
                        if let Ok(delta_data) = serde_json::from_str::<serde_json::Value>(&data)
                            && let Some(text) = delta_data.get("delta").and_then(|t| t.as_str())
                        {
                            full_text.push_str(text);
                        }
                    }
                    "toolCall" => {
                        debug_log!("🔄 DEBUG: Continuation received toolCall event!");
                        if let Ok(tool_data) = serde_json::from_str::<serde_json::Value>(&data) {
                            let next_call_id = tool_data
                                .get("callId")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let next_response_id = tool_data
                                .get("responseId")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let next_chat_id = tool_data
                                .get("chatId")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            let next_tool_name = tool_data
                                .get("toolName")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let next_server = tool_data
                                .get("server")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let next_parameters = tool_data
                                .get("parameters")
                                .cloned()
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                            logger(&format!("🔧 Chained tool requested: {}", next_tool_name));

                            pending_tool_calls.push((
                                next_call_id,
                                next_response_id,
                                next_chat_id,
                                next_tool_name,
                                next_server,
                                next_parameters,
                            ));

                            stream_done = true;
                            break;
                        }
                    }
                    "complete" => {
                        debug_log!(
                            "🔄 DEBUG: Continuation stream complete - {} chars text, {} pending tools",
                            full_text.len(),
                            pending_tool_calls.len()
                        );
                        logger("✅ Stream completed successfully");
                        stream_done = true;
                        break;
                    }
                    "error" => {
                        let error_msg = if let Ok(error_data) =
                            serde_json::from_str::<serde_json::Value>(&data)
                        {
                            error_data
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown error")
                                .to_string()
                        } else {
                            "Unknown error".to_string()
                        };
                        return Err(anyhow::anyhow!("Continuation stream error: {}", error_msg));
                    }
                    _ => {
                        // Ignore other event types (like "title")
                    }
                }
            }
        }
    }

    Ok((full_text, pending_tool_calls))
}

/// Execute chained tool calls
#[allow(clippy::too_many_arguments)]
async fn execute_chained_tools<'a>(
    _provider: &'a Arc<dyn Provider>,
    context: &'a AgentContext,
    model: &'a str,
    chat_id: Option<String>,
    call_id: &'a str,
    tool_name: &'a str,
    tool_parameters: &'a serde_json::Value,
    tool_output: &'a str,
    accumulated_history: &'a [serde_json::Value],
    full_text: &str,
    pending_tool_calls: Vec<(
        String,
        String,
        Option<String>,
        String,
        String,
        serde_json::Value,
    )>,
    logger: LogCallback,
) -> Result<String> {
    let mut result_text = full_text.to_string();

    if pending_tool_calls.is_empty() {
        tracing::debug!("No pending tool calls, returning text response");
        return Ok(result_text);
    }

    debug_log!(
        "🔄 DEBUG: Executing {} chained tool(s)",
        pending_tool_calls.len()
    );
    logger(&format!(
        "⚙️  Executing {} chained tool(s)...",
        pending_tool_calls.len()
    ));

    // Build accumulated history for chained calls
    // STATELESS MODE: Use function_call and tool roles (not assistant with tool_calls)
    let mut accumulated_history: Vec<serde_json::Value> = accumulated_history.to_vec();

    // Add any text the AI generated before the tool call
    if !full_text.is_empty() {
        accumulated_history.push(json!({
            "role": "assistant",
            "content": full_text
        }));
    }

    // Add the function_call (AI's request to call the tool)
    accumulated_history.push(json!({
        "role": "function_call",
        "call_id": call_id,
        "name": tool_name,
        "arguments": tool_parameters.to_string()
    }));

    // Add the tool result
    accumulated_history.push(json!({
        "role": "tool",
        "call_id": call_id,
        "name": tool_name,
        "content": tool_output
    }));

    for (
        next_call_id,
        next_response_id,
        next_chat_id,
        next_tool_name,
        next_server,
        next_parameters,
    ) in pending_tool_calls
    {
        if next_server != "cli-local" {
            logger(&format!(
                "⚠️  Skipping tool from non-local server: {}",
                next_server
            ));
            continue;
        }

        logger(&format!("🔧 Executing chained tool: {}", next_tool_name));

        // Execute the tool
        let tool_executor = ToolExecutor::new(PermissionMode::Auto);
        let tool_use = ToolUse {
            id: next_call_id.clone(),
            name: next_tool_name.clone(),
            input: next_parameters.clone(),
        };

        let tool_context = ToolContext::from_agent_context(context);

        let result = tool_executor.execute(&tool_use, &tool_context).await?;

        // Limit tool output
        const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
        let truncated_output = if result.content.len() > MAX_TOOL_OUTPUT_CHARS {
            format!(
                "{}\n\n[Output truncated: {} of {} chars]",
                &result.content[..MAX_TOOL_OUTPUT_CHARS],
                MAX_TOOL_OUTPUT_CHARS,
                result.content.len()
            )
        } else {
            result.content.clone()
        };

        logger(&format!(
            "✅ Chained tool {} executed successfully",
            next_tool_name
        ));

        // Recursively call continuation with accumulated history
        let nested_text = send_continuation_request(
            _provider,
            context,
            model,
            next_chat_id.or_else(|| chat_id.clone()),
            &next_response_id,
            &next_call_id,
            &next_tool_name,
            &next_parameters,
            &truncated_output,
            &accumulated_history,
            logger.clone(),
        )
        .await?;

        result_text.push_str(&nested_text);
    }

    Ok(result_text)
}
