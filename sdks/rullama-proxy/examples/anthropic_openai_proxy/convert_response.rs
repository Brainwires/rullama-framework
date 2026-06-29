use crate::convert_tools::generate_tool_use_id;
use crate::tool_name_mapper::ToolNameMapper;
use crate::types_anthropic::*;
use crate::types_openai::OpenAIChatResponse;

/// Convert an OpenAI Chat Completions response into an Anthropic Messages response.
pub fn convert_response(
    openai: &OpenAIChatResponse,
    original_model: &str,
    mapper: &ToolNameMapper,
) -> anyhow::Result<AnthropicResponse> {
    let choice = openai
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("OpenAI response has no choices"))?;

    let mut content: Vec<ResponseContentBlock> = Vec::new();

    // Text content
    if let Some(ref text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ResponseContentBlock::Text { text: text.clone() });
    }

    // Tool calls → tool_use blocks
    if let Some(ref tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let original_name = mapper.get_original_name(&tc.function.name);
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));

            content.push(ResponseContentBlock::ToolUse {
                id: if tc.id.is_empty() {
                    generate_tool_use_id()
                } else {
                    tc.id.clone()
                },
                name: original_name,
                input,
            });
        }
    }

    // Map finish_reason
    let stop_reason = choice.finish_reason.as_deref().map(map_finish_reason);

    // Map usage
    let usage = openai
        .usage
        .as_ref()
        .map(|u| AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        })
        .unwrap_or(AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
        });

    Ok(AnthropicResponse {
        id: format!("msg_{}", openai.id),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: original_model.to_string(),
        content,
        stop_reason: stop_reason.map(String::from),
        stop_sequence: None,
        usage,
    })
}

fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "end_turn",
        _ => "end_turn",
    }
}
