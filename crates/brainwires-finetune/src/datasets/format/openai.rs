use serde_json::json;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{TrainingExample, TrainingMessage, TrainingRole};
use super::FormatConverter;

/// OpenAI chat fine-tuning JSONL format.
///
/// Format: `{"messages": [{"role": "...", "content": "..."}]}`
pub struct OpenAiFormat;

impl FormatConverter for OpenAiFormat {
    fn name(&self) -> &str {
        "openai"
    }

    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value> {
        let messages: Vec<serde_json::Value> = example
            .messages
            .iter()
            .map(|msg| {
                let mut obj = json!({
                    "role": msg.role.to_string(),
                    "content": msg.content,
                });
                if let Some(ref tool_calls) = msg.tool_calls {
                    obj["tool_calls"] = json!(tool_calls);
                }
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    obj["tool_call_id"] = json!(tool_call_id);
                }
                if let Some(ref name) = msg.name {
                    obj["name"] = json!(name);
                }
                obj
            })
            .collect();

        Ok(json!({ "messages": messages }))
    }

    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample> {
        let messages_value =
            value
                .get("messages")
                .ok_or_else(|| DatasetError::FormatConversion {
                    message: "Missing 'messages' field".to_string(),
                })?;

        let messages_arr =
            messages_value
                .as_array()
                .ok_or_else(|| DatasetError::FormatConversion {
                    message: "'messages' must be an array".to_string(),
                })?;

        let mut messages = Vec::with_capacity(messages_arr.len());
        for msg_value in messages_arr {
            let role_str = msg_value
                .get("role")
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::FormatConversion {
                    message: "Message missing 'role'".to_string(),
                })?;

            let role = match role_str {
                "system" => TrainingRole::System,
                "user" => TrainingRole::User,
                "assistant" => TrainingRole::Assistant,
                "tool" => TrainingRole::Tool,
                other => {
                    return Err(DatasetError::FormatConversion {
                        message: format!("Unknown role: {}", other),
                    });
                }
            };

            let content = msg_value
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let tool_calls = msg_value
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned();

            let tool_call_id = msg_value
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            let name = msg_value
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from);

            messages.push(TrainingMessage {
                role,
                content,
                tool_calls,
                tool_call_id,
                name,
            });
        }

        Ok(TrainingExample::new(messages))
    }
}

use super::super::types::PreferencePair;
use super::PreferenceConverter;

impl PreferenceConverter for OpenAiFormat {
    fn name(&self) -> &str {
        "openai"
    }

    fn preference_to_json(&self, pair: &PreferencePair) -> DatasetResult<serde_json::Value> {
        let to_msgs = |msgs: &[TrainingMessage]| -> Vec<serde_json::Value> {
            msgs.iter()
                .map(|msg| json!({ "role": msg.role.to_string(), "content": msg.content }))
                .collect()
        };

        let mut result = json!({
            "prompt": to_msgs(&pair.prompt),
            "chosen": to_msgs(&pair.chosen),
            "rejected": to_msgs(&pair.rejected),
        });

        if !pair.metadata.is_empty() {
            result["metadata"] = json!(pair.metadata);
        }

        Ok(result)
    }

    fn parse_preference_json(&self, value: &serde_json::Value) -> DatasetResult<PreferencePair> {
        let parse_msgs = |key: &str| -> DatasetResult<Vec<TrainingMessage>> {
            let arr = value.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
                DatasetError::FormatConversion {
                    message: format!("Missing or invalid '{}' field", key),
                }
            })?;
            let mut msgs = Vec::new();
            for msg in arr {
                let role = match msg.get("role").and_then(|v| v.as_str()) {
                    Some("system") => TrainingRole::System,
                    Some("user") => TrainingRole::User,
                    Some("assistant") => TrainingRole::Assistant,
                    Some("tool") => TrainingRole::Tool,
                    _ => {
                        return Err(DatasetError::FormatConversion {
                            message: format!("Invalid role in '{}' messages", key),
                        });
                    }
                };
                let content = msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                msgs.push(TrainingMessage::new(role, content));
            }
            Ok(msgs)
        };

        let prompt = parse_msgs("prompt")?;
        let chosen = parse_msgs("chosen")?;
        let rejected = parse_msgs("rejected")?;

        let mut pair = PreferencePair::new(prompt, chosen, rejected);
        if let Some(meta) = value.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                pair.metadata.insert(k.clone(), v.clone());
            }
        }

        Ok(pair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_roundtrip() {
        let format = OpenAiFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::system("You are helpful"),
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi there!"),
        ]);

        let json = format.to_json(&example).unwrap();
        let parsed = format.parse_json(&json).unwrap();

        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[0].role, TrainingRole::System);
        assert_eq!(parsed.messages[1].content, "Hello");
        assert_eq!(parsed.messages[2].content, "Hi there!");
    }

    #[test]
    fn test_openai_format_structure() {
        let format = OpenAiFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::user("Q"),
            TrainingMessage::assistant("A"),
        ]);

        let json = format.to_json(&example).unwrap();
        assert!(json.get("messages").is_some());
        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn test_openai_preference_roundtrip() {
        use super::PreferenceConverter;
        use crate::datasets::types::PreferencePair;
        let format = OpenAiFormat;
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("What is 2+2?")],
            vec![TrainingMessage::assistant("4")],
            vec![TrainingMessage::assistant("22")],
        );
        let json = format.preference_to_json(&pair).unwrap();
        let parsed = format.parse_preference_json(&json).unwrap();
        assert_eq!(parsed.prompt[0].content, "What is 2+2?");
        assert_eq!(parsed.chosen[0].content, "4");
        assert_eq!(parsed.rejected[0].content, "22");
    }
}
