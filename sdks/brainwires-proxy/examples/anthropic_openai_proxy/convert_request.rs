use std::collections::{HashMap, HashSet};

use crate::convert_tools::{
    convert_tool_choice_to_openai, convert_tools_to_openai, generate_tool_use_id, make_tool_call,
};
use crate::tool_name_mapper::ToolNameMapper;
use crate::types_anthropic::*;
use crate::types_openai::*;

/// Result of converting an Anthropic request to OpenAI format.
pub struct ConvertedRequest {
    pub request: OpenAIChatRequest,
    pub mapper: ToolNameMapper,
}

/// Convert an Anthropic Messages API request into an OpenAI Chat Completions request.
pub fn convert_request(
    req: &AnthropicRequest,
    target_model: &str,
) -> anyhow::Result<ConvertedRequest> {
    let mut mapper = ToolNameMapper::new();
    let mut messages: Vec<OpenAIMessage> = Vec::new();

    // 1. System prompt
    if let Some(ref system) = req.system {
        let text = flatten_system(system);
        if !text.is_empty() {
            messages.push(OpenAIMessage::System { content: text });
        }
    }

    // 2. Convert messages
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut id_mappings: HashMap<String, Vec<String>> = HashMap::new();

    for msg in &req.messages {
        let converted = convert_message(msg, &mut mapper, &mut seen_ids, &mut id_mappings)?;
        messages.extend(converted);
    }

    // 3. Tools
    let tools = req
        .tools
        .as_ref()
        .map(|t| convert_tools_to_openai(t, &mut mapper));

    let tool_choice = req
        .tool_choice
        .as_ref()
        .map(|tc| convert_tool_choice_to_openai(tc, &mut mapper));

    // 4. max_tokens handling
    let (max_tokens, max_completion_tokens) = normalize_max_tokens(req.max_tokens, target_model);

    let request = OpenAIChatRequest {
        model: target_model.to_string(),
        messages,
        max_tokens,
        max_completion_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences.clone(),
        tools,
        tool_choice,
        stream: false, // framework handles full bodies
    };

    Ok(ConvertedRequest { request, mapper })
}

/// Flatten system content (string or array of blocks) into a single string.
fn flatten_system(system: &SystemContent) -> String {
    match system {
        SystemContent::Text(s) => s.clone(),
        SystemContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

/// Normalize max_tokens: bump 1→32 (Azure compat), use max_completion_tokens for gpt-5*.
fn normalize_max_tokens(max_tokens: u32, model: &str) -> (Option<u32>, Option<u32>) {
    let tokens = if max_tokens == 1 { 32 } else { max_tokens };

    if model.starts_with("gpt-5") || model.starts_with("o1") || model.starts_with("o3") {
        (None, Some(tokens))
    } else {
        (Some(tokens), None)
    }
}

/// Convert a single Anthropic message into one or more OpenAI messages.
fn convert_message(
    msg: &AnthropicMessage,
    mapper: &mut ToolNameMapper,
    seen_ids: &mut HashSet<String>,
    id_mappings: &mut HashMap<String, Vec<String>>,
) -> anyhow::Result<Vec<OpenAIMessage>> {
    match msg.role.as_str() {
        "user" => convert_user_message(&msg.content, mapper, seen_ids, id_mappings),
        "assistant" => convert_assistant_message(&msg.content, mapper, seen_ids, id_mappings),
        other => anyhow::bail!("unsupported message role: {}", other),
    }
}

fn convert_user_message(
    content: &MessageContent,
    mapper: &mut ToolNameMapper,
    seen_ids: &mut HashSet<String>,
    id_mappings: &mut HashMap<String, Vec<String>>,
) -> anyhow::Result<Vec<OpenAIMessage>> {
    let mut result = Vec::new();

    match content {
        MessageContent::Text(text) => {
            result.push(OpenAIMessage::User {
                content: text.clone(),
            });
        }
        MessageContent::Blocks(blocks) => {
            // Tool results become separate tool messages; text goes into a user message.
            let mut text_parts: Vec<String> = Vec::new();

            for block in blocks {
                match block {
                    AnthropicContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let result_text = extract_tool_result_text(content, *is_error);

                        // Find the mapped ID(s) for this tool_use_id
                        let call_ids = resolve_tool_result_id(tool_use_id, id_mappings);
                        for call_id in call_ids {
                            result.push(OpenAIMessage::Tool {
                                content: result_text.clone(),
                                tool_call_id: call_id,
                            });
                        }
                    }
                    AnthropicContentBlock::Text { text } => {
                        text_parts.push(text.clone());
                    }
                    AnthropicContentBlock::ToolUse { .. } => {
                        // tool_use in a user message is unusual but handle gracefully
                    }
                }
            }

            if !text_parts.is_empty() {
                result.push(OpenAIMessage::User {
                    content: text_parts.join("\n"),
                });
            }
        }
    }

    let _ = (mapper, seen_ids); // used indirectly through id_mappings
    Ok(result)
}

fn convert_assistant_message(
    content: &MessageContent,
    mapper: &mut ToolNameMapper,
    seen_ids: &mut HashSet<String>,
    id_mappings: &mut HashMap<String, Vec<String>>,
) -> anyhow::Result<Vec<OpenAIMessage>> {
    match content {
        MessageContent::Text(text) => {
            // Skip assistant prefill tokens
            if is_prefill(text) {
                return Ok(Vec::new());
            }
            Ok(vec![OpenAIMessage::Assistant {
                content: Some(text.clone()),
                tool_calls: None,
            }])
        }
        MessageContent::Blocks(blocks) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();

            for block in blocks {
                match block {
                    AnthropicContentBlock::Text { text } => {
                        if !is_prefill(text) {
                            text_parts.push(text.clone());
                        }
                    }
                    AnthropicContentBlock::ToolUse { id, name, input } => {
                        let unique_id = dedup_id(id, seen_ids, id_mappings);
                        let short_name = mapper.get_short_name(name);
                        tool_calls.push(make_tool_call(&unique_id, &short_name, input));
                    }
                    AnthropicContentBlock::ToolResult { .. } => {
                        // tool_result in assistant message — skip
                    }
                }
            }

            let content_text = if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join("\n"))
            };

            let tc = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            };

            // Only emit the message if there's content or tool calls
            if content_text.is_some() || tc.is_some() {
                Ok(vec![OpenAIMessage::Assistant {
                    content: content_text,
                    tool_calls: tc,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }
}

/// Detect assistant prefill tokens that should be stripped.
fn is_prefill(text: &str) -> bool {
    let trimmed = text.trim();
    matches!(trimmed, "{" | "[" | "```" | "<")
        || trimmed.starts_with("<tool_code") && !trimmed.contains("</tool_code")
}

/// Deduplicate tool IDs: if this ID was already seen, generate a unique replacement
/// and record the mapping so tool_result can find it.
fn dedup_id(
    original: &str,
    seen_ids: &mut HashSet<String>,
    id_mappings: &mut HashMap<String, Vec<String>>,
) -> String {
    if seen_ids.insert(original.to_string()) {
        // First occurrence — use as-is, record in mappings
        id_mappings
            .entry(original.to_string())
            .or_default()
            .push(original.to_string());
        original.to_string()
    } else {
        // Duplicate — generate a unique ID
        let new_id = generate_tool_use_id();
        seen_ids.insert(new_id.clone());
        id_mappings
            .entry(original.to_string())
            .or_default()
            .push(new_id.clone());
        new_id
    }
}

/// Resolve which OpenAI call ID(s) correspond to an Anthropic tool_use_id for a tool_result.
fn resolve_tool_result_id(
    tool_use_id: &str,
    id_mappings: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    if let Some(ids) = id_mappings.get(tool_use_id) {
        // Return the last mapped ID (most recent tool_use with this ID)
        ids.last().cloned().into_iter().collect()
    } else {
        // No mapping found — use original
        vec![tool_use_id.to_string()]
    }
}

/// Extract text from a tool result content.
fn extract_tool_result_text(content: &Option<ToolResultContent>, is_error: Option<bool>) -> String {
    let text = match content {
        Some(ToolResultContent::Text(s)) => s.clone(),
        Some(ToolResultContent::Blocks(blocks)) => blocks
            .iter()
            .map(|b| match b {
                ToolResultBlock::Text { text } => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };

    if is_error == Some(true) && !text.is_empty() {
        format!("Error: {}", text)
    } else {
        text
    }
}
