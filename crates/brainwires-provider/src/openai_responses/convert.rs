//! Conversion helpers between brainwires-core types and Responses API types.

use anyhow::Result;
use serde_json::json;

use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, Message, MessageContent, Role, StreamChunk, Tool,
    Usage,
};

use super::types::{
    CreateResponseRequest, InputContent, OutputContentBlock, ResponseInput, ResponseInputItem,
    ResponseObject, ResponseOutputItem, ResponseTool, ResponseUsage, ToolChoice,
};

/// Convert brainwires-core messages to Responses API input items.
///
/// Returns `(input_items, system_prompt)` — the system message is extracted
/// and returned separately (used as `instructions`).
pub fn messages_to_input(messages: &[Message]) -> (Vec<ResponseInputItem>, Option<String>) {
    let mut items = Vec::new();
    let mut system_prompt = None;

    for msg in messages {
        match msg.role {
            Role::System => {
                if let Some(text) = msg.text() {
                    system_prompt = Some(text.to_string());
                }
            }
            Role::User | Role::Assistant => {
                let role = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    _ => "user",
                };

                if let Some(text) = msg.text() {
                    items.push(ResponseInputItem::Message {
                        role: role.to_string(),
                        content: InputContent::Text(text.to_string()),
                        status: None,
                    });
                }

                // Handle tool results in user messages
                if let MessageContent::Blocks(blocks) = &msg.content {
                    for block in blocks {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } = block
                        {
                            items.push(ResponseInputItem::FunctionCallOutput {
                                call_id: tool_use_id.clone(),
                                output: content.clone(),
                            });
                        }
                    }
                }
            }
            Role::Tool => {
                if let Some(text) = msg.text() {
                    items.push(ResponseInputItem::FunctionCallOutput {
                        call_id: msg.name.clone().unwrap_or_default(),
                        output: text.to_string(),
                    });
                }
            }
        }
    }

    (items, system_prompt)
}

/// Convert brainwires-core tools to Responses API tool definitions.
pub fn tools_to_response_tools(tools: &[Tool]) -> Vec<ResponseTool> {
    tools
        .iter()
        .map(|t| ResponseTool::Function {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: serde_json::to_value(&t.input_schema).unwrap_or(json!({})),
            strict: None,
        })
        .collect()
}

/// Parse a full ResponseObject into a brainwires-core ChatResponse.
pub fn response_to_chat_response(resp: &ResponseObject) -> Result<ChatResponse> {
    let mut content_blocks = Vec::new();

    for item in &resp.output {
        match item {
            ResponseOutputItem::Message { content, .. } => {
                for block in content {
                    match block {
                        OutputContentBlock::OutputText { text, .. } => {
                            content_blocks.push(ContentBlock::Text { text: text.clone() });
                        }
                        OutputContentBlock::Refusal { refusal } => {
                            content_blocks.push(ContentBlock::Text {
                                text: refusal.clone(),
                            });
                        }
                        OutputContentBlock::OutputAudio { transcript, .. } => {
                            if let Some(text) = transcript {
                                content_blocks.push(ContentBlock::Text { text: text.clone() });
                            }
                        }
                    }
                }
            }
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input: serde_json::Value = serde_json::from_str(arguments).unwrap_or(json!({}));
                content_blocks.push(ContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                });
            }
            // Other output item types are not mapped to ContentBlocks
            // (web search, file search, code interpreter, etc. are informational)
            _ => {}
        }
    }

    let content = if content_blocks.len() == 1 {
        if let Some(ContentBlock::Text { text }) = content_blocks.first() {
            MessageContent::Text(text.clone())
        } else {
            MessageContent::Blocks(content_blocks)
        }
    } else {
        MessageContent::Blocks(content_blocks)
    };

    let usage = convert_usage(resp.usage.as_ref());

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
            name: None,
            metadata: None,
        },
        usage,
        finish_reason: Some("stop".to_string()),
    })
}

/// Convert ResponseUsage to brainwires-core Usage.
pub fn convert_usage(usage: Option<&ResponseUsage>) -> Usage {
    usage.map_or(Usage::default(), |u| Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens.unwrap_or(u.input_tokens + u.output_tokens),
        ..Default::default()
    })
}

/// Map a streaming event to a StreamChunk (returns None for events we don't
/// surface to the Provider trait).
pub fn stream_event_to_chunk(
    event: &super::types::ResponseStreamEvent,
) -> Option<Vec<StreamChunk>> {
    use super::types::ResponseStreamEvent;

    match event {
        ResponseStreamEvent::OutputTextDelta { delta, .. } => {
            Some(vec![StreamChunk::Text(delta.clone())])
        }
        ResponseStreamEvent::OutputItemAdded { item, .. } => {
            if let ResponseOutputItem::FunctionCall { call_id, name, .. } = item {
                Some(vec![StreamChunk::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                }])
            } else {
                None
            }
        }
        ResponseStreamEvent::FunctionCallArgumentsDelta { delta, item_id, .. } => {
            Some(vec![StreamChunk::ToolInputDelta {
                id: item_id.clone(),
                partial_json: delta.clone(),
            }])
        }
        ResponseStreamEvent::ResponseCompleted { response } => {
            let usage = convert_usage(response.usage.as_ref());
            Some(vec![StreamChunk::Usage(usage), StreamChunk::Done])
        }
        ResponseStreamEvent::ResponseFailed { .. }
        | ResponseStreamEvent::ResponseIncomplete { .. } => Some(vec![StreamChunk::Done]),
        _ => None,
    }
}

/// Build a CreateResponseRequest from legacy-style arguments.
pub fn build_request(
    model: &str,
    input: Vec<ResponseInputItem>,
    instructions: Option<&str>,
    tools: Option<&[ResponseTool]>,
    options: &ChatOptions,
    previous_response_id: Option<&str>,
) -> CreateResponseRequest {
    let response_tools = tools.filter(|t| !t.is_empty()).map(|t| t.to_vec());

    let tool_choice = if response_tools.is_some() {
        Some(ToolChoice::Mode("auto".to_string()))
    } else {
        None
    };

    CreateResponseRequest {
        model: model.to_string(),
        input: ResponseInput::Items(input),
        instructions: instructions.map(|s| s.to_string()),
        tools: response_tools,
        tool_choice,
        parallel_tool_calls: None,
        max_output_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        stop: options.stop.clone(),
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
        previous_response_id: previous_response_id.map(|s| s.to_string()),
        store: None,
        metadata: None,
        truncation: None,
        reasoning: None,
        text: None,
        include: None,
        user: None,
        background: None,
        service_tier: None,
        conversation: None,
        context_management: None,
        modalities: None,
        audio: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ToolInputSchema;
    use std::collections::HashMap;

    #[test]
    fn test_messages_to_input_simple() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let (items, system) = messages_to_input(&messages);
        assert_eq!(items.len(), 1);
        assert!(system.is_none());
    }

    #[test]
    fn test_messages_to_input_with_system() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("You are helpful".to_string()),
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

        let (items, system) = messages_to_input(&messages);
        assert_eq!(items.len(), 1);
        assert_eq!(system, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_tools_to_response_tools() {
        let mut properties = HashMap::new();
        properties.insert("q".to_string(), json!({"type": "string"}));

        let tools = vec![Tool {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["q".to_string()]),
            requires_approval: false,
            ..Default::default()
        }];

        let converted = tools_to_response_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert!(matches!(&converted[0], ResponseTool::Function { name, .. } if name == "search"));
    }

    #[test]
    fn test_response_to_chat_response_text() {
        let resp = ResponseObject {
            id: "resp_123".to_string(),
            object: Some("response".to_string()),
            created_at: None,
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            instructions: None,
            model: Some("gpt-4o".to_string()),
            output: vec![ResponseOutputItem::Message {
                id: Some("msg_1".to_string()),
                role: "assistant".to_string(),
                content: vec![OutputContentBlock::OutputText {
                    text: "Hello!".to_string(),
                    annotations: vec![],
                }],
                status: Some("completed".to_string()),
            }],
            output_text: Some("Hello!".to_string()),
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            metadata: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            tool_choice: None,
            tools: None,
            text: None,
            truncation: None,
            store: None,
            usage: Some(ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: Some(15),
                output_tokens_details: None,
            }),
            user: None,
        };

        let chat_response = response_to_chat_response(&resp).unwrap();
        assert_eq!(chat_response.message.role, Role::Assistant);
        assert_eq!(chat_response.usage.prompt_tokens, 10);
        assert_eq!(chat_response.usage.completion_tokens, 5);

        if let MessageContent::Text(text) = &chat_response.message.content {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_response_to_chat_response_with_function_call() {
        let resp = ResponseObject {
            id: "resp_456".to_string(),
            object: None,
            created_at: None,
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            instructions: None,
            model: None,
            output: vec![
                ResponseOutputItem::Message {
                    id: None,
                    role: "assistant".to_string(),
                    content: vec![OutputContentBlock::OutputText {
                        text: "Let me search".to_string(),
                        annotations: vec![],
                    }],
                    status: None,
                },
                ResponseOutputItem::FunctionCall {
                    id: Some("fc_1".to_string()),
                    name: "search".to_string(),
                    arguments: r#"{"q":"test"}"#.to_string(),
                    call_id: "call_1".to_string(),
                    status: None,
                },
            ],
            output_text: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            metadata: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            tool_choice: None,
            tools: None,
            text: None,
            truncation: None,
            store: None,
            usage: None,
            user: None,
        };

        let chat_response = response_to_chat_response(&resp).unwrap();
        if let MessageContent::Blocks(blocks) = &chat_response.message.content {
            assert_eq!(blocks.len(), 2);
            assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "search"));
        } else {
            panic!("Expected blocks content");
        }
    }

    #[test]
    fn test_build_request() {
        let input = vec![ResponseInputItem::Message {
            role: "user".to_string(),
            content: InputContent::Text("Hello".to_string()),
            status: None,
        }];
        let options = ChatOptions::default();
        let req = build_request("gpt-4o", input, Some("Be helpful"), None, &options, None);
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.instructions, Some("Be helpful".to_string()));
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
    }

    #[test]
    fn test_convert_usage_some() {
        let usage = ResponseUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: Some(150),
            output_tokens_details: None,
        };
        let u = convert_usage(Some(&usage));
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(u.total_tokens, 150);
    }

    #[test]
    fn test_convert_usage_none() {
        let u = convert_usage(None);
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 0);
    }
}
