use crate::types_anthropic::*;

/// Format a complete Anthropic response as a series of SSE events.
///
/// When the client sent `stream: true`, we still get a complete response from the
/// upstream (since the framework forces `stream: false`). This function formats the
/// complete response as SSE events so the client sees the expected streaming format.
///
/// All events arrive at once (buffered SSE is valid per the spec).
pub fn format_as_sse(response: &AnthropicResponse) -> String {
    let mut events: Vec<String> = Vec::new();

    // 1. message_start — envelope with empty content
    let start = MessageStartEvent {
        event_type: "message_start".to_string(),
        message: MessageStartPayload {
            id: response.id.clone(),
            payload_type: "message".to_string(),
            role: "assistant".to_string(),
            model: response.model.clone(),
            content: vec![],
            stop_reason: serde_json::Value::Null,
            stop_sequence: serde_json::Value::Null,
            usage: AnthropicUsage {
                input_tokens: response.usage.input_tokens,
                output_tokens: 0,
            },
        },
    };
    events.push(sse_event("message_start", &start));

    // 2. ping
    let ping = PingEvent {
        event_type: "ping".to_string(),
    };
    events.push(sse_event("ping", &ping));

    // 3. Per content block: start → delta(s) → stop
    for (idx, block) in response.content.iter().enumerate() {
        match block {
            ResponseContentBlock::Text { text } => {
                // content_block_start
                let block_start = ContentBlockStartEvent {
                    event_type: "content_block_start".to_string(),
                    index: idx,
                    content_block: ContentBlockStartPayload::Text {
                        text: String::new(),
                    },
                };
                events.push(sse_event("content_block_start", &block_start));

                // content_block_delta (send entire text as one delta)
                if !text.is_empty() {
                    let delta = ContentBlockDeltaEvent {
                        event_type: "content_block_delta".to_string(),
                        index: idx,
                        delta: ContentBlockDelta::TextDelta { text: text.clone() },
                    };
                    events.push(sse_event("content_block_delta", &delta));
                }

                // content_block_stop
                let block_stop = ContentBlockStopEvent {
                    event_type: "content_block_stop".to_string(),
                    index: idx,
                };
                events.push(sse_event("content_block_stop", &block_stop));
            }
            ResponseContentBlock::ToolUse { id, name, input } => {
                // content_block_start
                let block_start = ContentBlockStartEvent {
                    event_type: "content_block_start".to_string(),
                    index: idx,
                    content_block: ContentBlockStartPayload::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::json!({}),
                    },
                };
                events.push(sse_event("content_block_start", &block_start));

                // content_block_delta with the full input JSON
                let json_str = serde_json::to_string(input).unwrap_or_default();
                if !json_str.is_empty() && json_str != "{}" {
                    let delta = ContentBlockDeltaEvent {
                        event_type: "content_block_delta".to_string(),
                        index: idx,
                        delta: ContentBlockDelta::InputJsonDelta {
                            partial_json: json_str,
                        },
                    };
                    events.push(sse_event("content_block_delta", &delta));
                }

                // content_block_stop
                let block_stop = ContentBlockStopEvent {
                    event_type: "content_block_stop".to_string(),
                    index: idx,
                };
                events.push(sse_event("content_block_stop", &block_stop));
            }
        }
    }

    // 4. message_delta — stop_reason + output_tokens
    let msg_delta = MessageDeltaEvent {
        event_type: "message_delta".to_string(),
        delta: MessageDelta {
            stop_reason: response.stop_reason.clone(),
            stop_sequence: response.stop_sequence.clone(),
        },
        usage: MessageDeltaUsage {
            output_tokens: response.usage.output_tokens,
        },
    };
    events.push(sse_event("message_delta", &msg_delta));

    // 5. message_stop
    let msg_stop = MessageStopEvent {
        event_type: "message_stop".to_string(),
    };
    events.push(sse_event("message_stop", &msg_stop));

    events.join("")
}

fn sse_event<T: serde::Serialize>(event_name: &str, data: &T) -> String {
    let json = serde_json::to_string(data).unwrap_or_default();
    format!("event: {}\ndata: {}\n\n", event_name, json)
}
