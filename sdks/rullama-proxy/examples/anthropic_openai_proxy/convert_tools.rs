use crate::tool_name_mapper::ToolNameMapper;
use crate::types_anthropic::{AnthropicToolChoice, AnthropicToolDefinition};
use crate::types_openai::{
    OpenAIFunction, OpenAIFunctionCall, OpenAITool, OpenAIToolCall, OpenAIToolChoice,
    OpenAIToolChoiceFunction,
};
use rand::RngExt;

/// Convert Anthropic tool definitions to OpenAI function-calling format.
pub fn convert_tools_to_openai(
    tools: &[AnthropicToolDefinition],
    mapper: &mut ToolNameMapper,
) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|tool| {
            let name = mapper.get_short_name(&tool.name);
            let mut schema = tool.input_schema.clone();

            // Ensure `properties` field exists (OpenAI requires it)
            if let Some(obj) = schema.as_object_mut() {
                obj.entry("properties")
                    .or_insert_with(|| serde_json::json!({}));
            }

            OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name,
                    description: tool.description.clone(),
                    parameters: schema,
                },
            }
        })
        .collect()
}

/// Convert Anthropic tool_choice to OpenAI format.
pub fn convert_tool_choice_to_openai(
    choice: &AnthropicToolChoice,
    mapper: &mut ToolNameMapper,
) -> OpenAIToolChoice {
    match choice.choice_type.as_str() {
        "auto" => OpenAIToolChoice::String("auto".to_string()),
        "any" => OpenAIToolChoice::String("required".to_string()),
        "tool" => {
            let name = choice
                .name
                .as_ref()
                .map(|n| mapper.get_short_name(n))
                .unwrap_or_default();
            OpenAIToolChoice::Named {
                choice_type: "function".to_string(),
                function: OpenAIToolChoiceFunction { name },
            }
        }
        other => OpenAIToolChoice::String(other.to_string()),
    }
}

/// Generate an Anthropic-style tool use ID: `toolu_` + 24 random alphanumeric chars.
pub fn generate_tool_use_id() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    let suffix: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    format!("toolu_{}", suffix)
}

/// Build an OpenAI tool_call from parts.
pub fn make_tool_call(id: &str, name: &str, arguments: &serde_json::Value) -> OpenAIToolCall {
    OpenAIToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: OpenAIFunctionCall {
            name: name.to_string(),
            arguments: serde_json::to_string(arguments).unwrap_or_default(),
        },
    }
}
