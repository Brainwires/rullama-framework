use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Role of the message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Message from the user.
    User,
    /// Message from the AI assistant.
    Assistant,
    /// System prompt or instruction.
    System,
    /// Tool result message.
    Tool,
}

/// Message content can be simple text or complex structured content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Array of content blocks (for multimodal messages)
    Blocks(Vec<ContentBlock>),
}

/// Content block for structured messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content block.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content block (base64 encoded).
    Image {
        /// The image source data.
        source: ImageSource,
    },
    /// Tool use request.
    ToolUse {
        /// Unique identifier for this tool invocation.
        id: String,
        /// Name of the tool to call.
        name: String,
        /// Input arguments for the tool.
        input: Value,
    },
    /// Tool result.
    ToolResult {
        /// ID of the tool use this result corresponds to.
        tool_use_id: String,
        /// Result content.
        content: String,
        /// Whether this result represents an error.
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Image source for image content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// MIME type (e.g. "image/png").
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message
    pub content: MessageContent,
    /// Optional name for the message sender (useful for multi-agent conversations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Message {
    /// Create a new user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            name: None,
            metadata: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            name: None,
            metadata: None,
        }
    }

    /// Create a new system message
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            name: None,
            metadata: None,
        }
    }

    /// Create a tool result message
    pub fn tool_result<S: Into<String>>(tool_use_id: S, content: S) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error: None,
            }]),
            name: None,
            metadata: None,
        }
    }

    /// Get the text content of a message (if it's simple text)
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(text) => Some(text),
            MessageContent::Blocks(_) => None,
        }
    }

    /// Get a text representation of the message content, including Blocks.
    /// For Text messages, returns the text directly.
    /// For Blocks messages, concatenates text from all blocks into a readable summary
    /// so that conversation history is preserved when serializing for API calls.
    pub fn text_or_summary(&self) -> String {
        match &self.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Blocks(blocks) => {
                let mut parts = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            parts.push(format!("[Called tool: {} with args: {}]", name, input));
                        }
                        ContentBlock::ToolResult {
                            content, is_error, ..
                        } => {
                            if is_error == &Some(true) {
                                parts.push(format!("[Tool error: {}]", content));
                            } else {
                                parts.push(format!("[Tool result: {}]", content));
                            }
                        }
                        ContentBlock::Image { .. } => {
                            parts.push("[Image]".to_string());
                        }
                    }
                }
                parts.join("\n")
            }
        }
    }

    /// Get mutable reference to the text content
    pub fn text_mut(&mut self) -> Option<&mut String> {
        match &mut self.content {
            MessageContent::Text(text) => Some(text),
            MessageContent::Blocks(_) => None,
        }
    }
}

/// Usage statistics for a chat completion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens
    pub total_tokens: u32,
    /// Tokens the provider charged to populate its prompt cache on this turn.
    ///
    /// Only meaningful for providers that support explicit caching (Anthropic
    /// Messages API). Zero for providers without prompt caching or when the
    /// cache is not in use.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from the provider's prompt cache on this turn — these are
    /// billed at a reduced rate and are the primary cost-savings signal.
    ///
    /// Zero when no cached bytes were hit.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_read_input_tokens: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

impl Usage {
    /// Create a new usage statistics (no cache activity).
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }

    /// Create a new usage statistics including cache accounting.
    pub fn with_cache(
        prompt_tokens: u32,
        completion_tokens: u32,
        cache_creation_input_tokens: u32,
        cache_read_input_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        }
    }
}

/// Response from a chat completion
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The generated message
    pub message: Message,
    /// Usage statistics
    pub usage: Usage,
    /// Optional finish reason
    pub finish_reason: Option<String>,
}

/// Serialize a slice of Messages into the STATELESS protocol format for conversation history.
///
/// Properly handles all message content types:
/// - `MessageContent::Text` → `{ "role": "user"|"assistant", "content": "..." }`
/// - `ContentBlock::ToolUse` → `{ "role": "function_call", "call_id", "name", "arguments" }`
/// - `ContentBlock::ToolResult` → `{ "role": "tool", "call_id", "content" }`
/// - `ContentBlock::Text` within Blocks → flushed as user/assistant text
/// - System messages and empty assistant messages are skipped
pub fn serialize_messages_to_stateless_history(messages: &[Message]) -> Vec<Value> {
    let mut history = Vec::new();

    for msg in messages {
        // Skip system messages
        if msg.role == Role::System {
            continue;
        }

        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::System => continue, // already handled above
        };

        match &msg.content {
            MessageContent::Text(text) => {
                // Skip empty assistant messages
                if msg.role == Role::Assistant && text.trim().is_empty() {
                    continue;
                }
                history.push(serde_json::json!({
                    "role": role_str,
                    "content": text,
                }));
            }
            MessageContent::Blocks(blocks) => {
                // Accumulate text blocks, emit tool entries individually
                let mut text_parts = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            // Flush accumulated text before tool entry
                            if !text_parts.is_empty() {
                                let combined = text_parts.join("\n");
                                if !(msg.role == Role::Assistant && combined.trim().is_empty()) {
                                    history.push(serde_json::json!({
                                        "role": role_str,
                                        "content": combined,
                                    }));
                                }
                                text_parts.clear();
                            }
                            history.push(serde_json::json!({
                                "role": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": input.to_string(),
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            // Flush accumulated text before tool entry
                            if !text_parts.is_empty() {
                                let combined = text_parts.join("\n");
                                if !(msg.role == Role::Assistant && combined.trim().is_empty()) {
                                    history.push(serde_json::json!({
                                        "role": role_str,
                                        "content": combined,
                                    }));
                                }
                                text_parts.clear();
                            }
                            history.push(serde_json::json!({
                                "role": "tool",
                                "call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                        ContentBlock::Image { .. } => {
                            // Images can't be serialized to stateless text format; skip
                        }
                    }
                }

                // Flush any remaining text
                if !text_parts.is_empty() {
                    let combined = text_parts.join("\n");
                    if !(msg.role == Role::Assistant && combined.trim().is_empty()) {
                        history.push(serde_json::json!({
                            "role": role_str,
                            "content": combined,
                        }));
                    }
                }
            }
        }
    }

    history
}

/// Streaming chunk from a chat completion
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Text delta
    Text(String),
    /// Tool use started.
    ToolUse {
        /// Unique tool use identifier.
        id: String,
        /// Name of the tool being invoked.
        name: String,
    },
    /// Tool input delta (partial JSON streaming).
    ToolInputDelta {
        /// Tool use identifier this delta belongs to.
        id: String,
        /// Partial JSON fragment.
        partial_json: String,
    },
    /// Tool call request from backend (for client-side execution).
    ToolCall {
        /// Unique call identifier.
        call_id: String,
        /// Response identifier for correlating results.
        response_id: String,
        /// Chat session identifier, if any.
        chat_id: Option<String>,
        /// Name of the tool to execute.
        tool_name: String,
        /// MCP server name hosting the tool.
        server: String,
        /// Parameters for the tool call.
        parameters: serde_json::Value,
    },
    /// Usage statistics (usually sent at the end)
    Usage(Usage),
    /// The model auto-compacted (summarised) the context window.
    ///
    /// Emitted by Claude 4.6+ when `context_window_management_event` fires.
    /// Agents should replace their message history with a synthetic assistant
    /// message containing the summary so future turns stay coherent.
    ContextCompacted {
        /// The model-generated summary that replaces the compacted messages.
        summary: String,
        /// Approximate number of tokens freed by compaction.
        tokens_freed: Option<u32>,
    },
    /// Stream completed
    Done,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), Some("Hello"));
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("Response");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), Some("Response"));
    }

    #[test]
    fn test_message_tool_result() {
        let msg = Message::tool_result("tool-1", "Result");
        assert_eq!(msg.role, Role::Tool);
    }

    #[test]
    fn test_usage_new() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_role_serialization() {
        let role = Role::User;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"user\"");
    }

    #[test]
    fn test_stateless_history_simple_text() {
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there")];
        let history = serialize_messages_to_stateless_history(&messages);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[1]["role"], "assistant");
    }

    #[test]
    fn test_stateless_history_skips_system() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let history = serialize_messages_to_stateless_history(&messages);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["role"], "user");
    }

    #[test]
    fn test_stateless_history_tool_round_trip() {
        let messages = vec![
            Message::user("Read the file"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "I'll check.".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "call-1".to_string(),
                        name: "read_file".to_string(),
                        input: json!({"path": "main.rs"}),
                    },
                ]),
                name: None,
                metadata: None,
            },
            Message::tool_result("call-1", "fn main() {}"),
            Message::assistant("The file contains a main function."),
        ];
        let history = serialize_messages_to_stateless_history(&messages);
        assert_eq!(history.len(), 5);
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[1]["role"], "assistant");
        assert_eq!(history[2]["role"], "function_call");
        assert_eq!(history[3]["role"], "tool");
        assert_eq!(history[4]["role"], "assistant");
    }
}
