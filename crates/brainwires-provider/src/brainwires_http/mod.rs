use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde_json::json;

use brainwires_core::message::{
    ChatResponse, ContentBlock, Message, MessageContent, Role, StreamChunk,
};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

/// Production backend URL.
pub const DEFAULT_BACKEND_URL: &str = "https://brainwires.studio";
/// Development backend URL.
pub const DEV_BACKEND_URL: &str = "https://dev.brainwires.net";

/// Determine the backend URL from an API key prefix.
///
/// Keys starting with `bw_dev_` route to the dev backend;
/// all others (including `bw_prod_` and `bw_test_`) route to production.
pub fn get_backend_from_api_key(api_key: &str) -> &'static str {
    if api_key.starts_with("bw_dev_") {
        DEV_BACKEND_URL
    } else {
        DEFAULT_BACKEND_URL
    }
}

/// HTTP-based Brainwires provider using Server-Sent Events for streaming.
///
/// Connects to the Brainwires Studio backend which routes requests to
/// the appropriate AI model (Claude, GPT, Gemini, etc.).
pub struct BrainwiresHttpProvider {
    api_key: String,
    backend_url: String,
    model: String,
    http_client: Client,
}

impl BrainwiresHttpProvider {
    /// Create a new Brainwires HTTP provider.
    pub fn new(api_key: String, backend_url: String, model: String) -> Self {
        Self {
            api_key,
            backend_url,
            model,
            http_client: Client::new(),
        }
    }

    fn _get_system_message(&self, messages: &[Message]) -> Option<String> {
        messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()))
    }
}

#[async_trait]
impl Provider for BrainwiresHttpProvider {
    fn name(&self) -> &str {
        "brainwires"
    }

    fn max_output_tokens(&self) -> Option<u32> {
        // Max output tokens per model family (provider specifications as of 2026-Q1).
        match self.model.as_str() {
            // Claude models
            s if s.contains("claude-3-5-sonnet") => Some(8192),
            s if s.contains("claude-3-opus") => Some(4096),
            s if s.contains("claude-3-haiku") => Some(4096),
            s if s.contains("claude") => Some(4096),

            // GPT models
            s if s.contains("gpt-5") => Some(32768),
            s if s.contains("gpt-4") => Some(8192),
            s if s.contains("gpt-3.5") => Some(4096),
            s if s.contains("o1") => Some(65536),

            // Gemini models
            s if s.contains("gemini-1.5-pro") => Some(8192),
            s if s.contains("gemini-1.5-flash") => Some(8192),
            s if s.contains("gemini") => Some(2048),

            // Default for unknown models
            _ => Some(8192),
        }
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        use futures::StreamExt;

        let mut stream = self.stream_chat(messages, tools, options);
        let mut full_text = String::new();
        let mut usage_data = None;
        let mut tool_calls = Vec::new();
        let mut last_response_id: Option<String> = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result? {
                StreamChunk::Text(text) => {
                    full_text.push_str(&text);
                }
                StreamChunk::Usage(usage) => {
                    usage_data = Some(usage);
                }
                StreamChunk::Done => break,
                StreamChunk::ToolCall {
                    call_id,
                    response_id,
                    tool_name,
                    parameters,
                    ..
                } => {
                    last_response_id = Some(response_id);
                    tool_calls.push(ContentBlock::ToolUse {
                        id: call_id,
                        name: tool_name,
                        input: parameters,
                    });
                }
                StreamChunk::ToolUse { .. } | StreamChunk::ToolInputDelta { .. } => {
                    // Not used by brainwires backend
                }
                StreamChunk::ContextCompacted { .. } => {
                    // Context compaction is handled by the agent layer; relay ignores it
                }
            }
        }

        let tool_call_count = tool_calls.len();
        let content = if tool_calls.is_empty() {
            MessageContent::Text(full_text)
        } else {
            let mut blocks = Vec::new();
            if !full_text.is_empty() {
                blocks.push(ContentBlock::Text { text: full_text });
            }
            blocks.extend(tool_calls);
            MessageContent::Blocks(blocks)
        };

        tracing::debug!("chat() collected {} tool calls", tool_call_count);

        let finish_reason = if tool_call_count > 0 {
            None
        } else {
            Some("stop".to_string())
        };

        let metadata = last_response_id.map(|rid| json!({"response_id": rid}));

        Ok(ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
                metadata,
            },
            usage: usage_data.unwrap_or_default(),
            finish_reason,
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        Box::pin(async_stream::stream! {
            let (current_content, conversation_history, function_call_output, previous_response_id) = if let Some(last_msg) = messages.last() {
                let mut func_output = None;

                // Check if last message contains ToolResult blocks
                if let MessageContent::Blocks(blocks) = &last_msg.content {
                    for block in blocks {
                        if let ContentBlock::ToolResult { tool_use_id, content, .. } = block {
                            if let Some(prev_msg) = messages.get(messages.len().saturating_sub(2))
                                && let MessageContent::Blocks(prev_blocks) = &prev_msg.content {
                                    for prev_block in prev_blocks {
                                        if let ContentBlock::ToolUse { id, name, .. } = prev_block
                                            && id == tool_use_id {
                                                func_output = Some(json!({
                                                    "call_id": tool_use_id,
                                                    "name": name,
                                                    "output": content
                                                }));
                                                break;
                                            }
                                    }
                                }
                            break;
                        }
                    }
                }

                if func_output.is_some() {
                    let assistant_msg_idx = messages.len().saturating_sub(2);
                    let assistant_msg = messages.get(assistant_msg_idx);

                    tracing::debug!(
                        "Looking for response_id: messages.len()={}, checking index={}, msg_role={:?}, has_metadata={}",
                        messages.len(),
                        assistant_msg_idx,
                        assistant_msg.map(|m| &m.role),
                        assistant_msg.and_then(|m| m.metadata.as_ref()).is_some()
                    );

                    let response_id_from_metadata = assistant_msg
                        .and_then(|m| m.metadata.as_ref())
                        .and_then(|meta| meta.get("response_id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    if response_id_from_metadata.is_none() {
                        tracing::warn!(
                            "No response_id found in metadata! Dumping message roles: {:?}",
                            messages.iter().map(|m| format!("{:?}", m.role)).collect::<Vec<_>>()
                        );
                    }

                    let history: Vec<serde_json::Value> = messages[..messages.len().saturating_sub(2)]
                        .iter()
                        .filter_map(|m| {
                            if m.role == Role::System {
                                return None;
                            }
                            let msg_content = m.text_or_summary();
                            if m.role == Role::Assistant && msg_content.trim().is_empty() {
                                return None;
                            }
                            Some(json!({
                                "role": match m.role {
                                    Role::User => "user",
                                    Role::Assistant => "assistant",
                                    Role::Tool => "user",
                                    Role::System => return None,
                                },
                                "content": msg_content,
                            }))
                        })
                        .collect();
                    ("".to_string(), history, func_output, response_id_from_metadata)
                } else {
                    let content = last_msg.text_or_summary();
                    let history: Vec<serde_json::Value> = messages[..messages.len().saturating_sub(1)]
                        .iter()
                        .filter_map(|m| {
                            if m.role == Role::System {
                                return None;
                            }
                            let msg_content = m.text_or_summary();
                            if m.role == Role::Assistant && msg_content.trim().is_empty() {
                                return None;
                            }
                            Some(json!({
                                "role": match m.role {
                                    Role::User => "user",
                                    Role::Assistant => "assistant",
                                    Role::Tool => "user",
                                    Role::System => return None,
                                },
                                "content": msg_content,
                            }))
                        })
                        .collect();
                    (content, history, None, None)
                }
            } else {
                yield Err(anyhow::anyhow!("No messages provided"));
                return;
            };

            let mut request_body = json!({
                "content": current_content,
                "model": self.model,
                "timezone": "UTC",
            });

            if !conversation_history.is_empty() {
                request_body["conversationHistory"] = json!(conversation_history);
            }

            if let Some(ref func_output) = function_call_output {
                request_body["functionCallOutput"] = func_output.clone();
                if let Some(resp_id) = &previous_response_id {
                    request_body["previousResponseId"] = json!(resp_id);
                    tracing::debug!(
                        "Sending request with: call_id={}, previousResponseId={}",
                        func_output.get("call_id").and_then(|v| v.as_str()).unwrap_or("?"),
                        resp_id
                    );
                } else {
                    tracing::warn!(
                        "Sending request WITHOUT previousResponseId: call_id={}",
                        func_output.get("call_id").and_then(|v| v.as_str()).unwrap_or("?")
                    );
                }
            }

            if let Some(system_msg) = &options.system {
                request_body["systemPrompt"] = json!(system_msg);
            }

            if let Some(temp) = options.temperature {
                request_body["temperature"] = json!(temp);
            }

            // Convert CLI tools to MCP tool format for backend
            if let Some(tools_list) = tools {
                let mcp_tools: Vec<serde_json::Value> = tools_list
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

                if !mcp_tools.is_empty() {
                    request_body["selectedMCPTools"] = json!(mcp_tools);
                }
            }

            let url = format!("{}/api/chat/stream", self.backend_url);

            tracing::debug!("Sending request to {}", url);
            tracing::debug!("Model: {}", self.model);
            tracing::debug!("Conversation history size: {} messages", conversation_history.len());

            if !conversation_history.is_empty() {
                if let Some(first) = conversation_history.first() {
                    let role = first.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                    let content = first.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    tracing::debug!("First msg [{}]: {}...", role, &content[..content.len().min(50)]);
                }
                if conversation_history.len() > 1
                    && let Some(last) = conversation_history.last() {
                        let role = last.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                        let content = last.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        tracing::debug!("Last msg [{}]: {}...", role, &content[..content.len().min(50)]);
                    }
            }
            if let Some(mcp_tools) = request_body.get("selectedMCPTools") {
                let tool_count = mcp_tools.as_array().map(|a| a.len()).unwrap_or(0);
                tracing::debug!("Sending {} tools to backend", tool_count);
            }

            let response = match self
                .http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!(
                    "Brainwires API error ({}): {}",
                    status,
                    error_text
                ));
                return;
            }

            // Parse SSE stream
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(e.into());
                        continue;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

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
                                    && let Some(text) = delta_data.get("delta").and_then(|t| t.as_str()) {
                                        yield Ok(StreamChunk::Text(text.to_string()));
                                    }
                            }
                            "complete" => {
                                if let Ok(complete_data) = serde_json::from_str::<serde_json::Value>(&data)
                                    && let Some(usage_data) = complete_data.get("usage")
                                        && let Ok(usage) = serde_json::from_value(usage_data.clone()) {
                                            yield Ok(StreamChunk::Usage(usage));
                                        }
                                yield Ok(StreamChunk::Done);
                            }
                            "error" => {
                                let error_msg = if let Ok(error_data) = serde_json::from_str::<serde_json::Value>(&data) {
                                    error_data.get("message")
                                        .and_then(|m| m.as_str())
                                        .unwrap_or("Unknown error")
                                        .to_string()
                                } else {
                                    "Unknown error".to_string()
                                };
                                yield Err(anyhow::anyhow!("Stream error: {}", error_msg));
                                return;
                            }
                            "toolCall" => {
                                if let Ok(tool_data) = serde_json::from_str::<serde_json::Value>(&data) {
                                    let call_id = tool_data.get("callId")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let response_id = tool_data.get("responseId")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let chat_id = tool_data.get("chatId")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string());
                                    let tool_name = tool_data.get("toolName")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let server = tool_data.get("server")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let parameters = tool_data.get("parameters")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                                    tracing::debug!(
                                        "Received toolCall: call_id={}, response_id={}, tool={}",
                                        call_id, response_id, tool_name
                                    );

                                    yield Ok(StreamChunk::ToolCall {
                                        call_id,
                                        response_id,
                                        chat_id,
                                        tool_name,
                                        server,
                                        parameters,
                                    });

                                    yield Ok(StreamChunk::Done);
                                    return;
                                }
                            }
                            "title" => {
                                tracing::debug!("Ignoring title event");
                            }
                            _ => {
                                tracing::debug!("Unknown event type: {}", evt_type);
                            }
                        }
                    }
                }
            }

            // Stream ended without explicit done signal
            yield Ok(StreamChunk::Done);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = BrainwiresHttpProvider::new(
            "test-key".to_string(),
            "http://localhost:3000".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
        );
        assert_eq!(provider.name(), "brainwires");
    }

    #[test]
    fn test_max_output_tokens() {
        let provider = BrainwiresHttpProvider::new(
            "test-key".to_string(),
            "http://localhost:3000".to_string(),
            "gpt-5-mini".to_string(),
        );
        assert_eq!(provider.max_output_tokens(), Some(32768));

        let provider = BrainwiresHttpProvider::new(
            "test-key".to_string(),
            "http://localhost:3000".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
        );
        assert_eq!(provider.max_output_tokens(), Some(8192));
    }

    #[test]
    fn test_get_system_message() {
        let provider = BrainwiresHttpProvider::new(
            "test-key".to_string(),
            "http://localhost:3000".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("You are a helpful assistant".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("Hello".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let system = provider._get_system_message(&messages);
        assert_eq!(system, Some("You are a helpful assistant".to_string()));
    }
}
