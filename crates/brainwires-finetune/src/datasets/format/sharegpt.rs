use serde_json::json;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{TrainingExample, TrainingMessage, TrainingRole};
use super::FormatConverter;

/// ShareGPT conversation format.
///
/// Format: `{"conversations": [{"from": "human|gpt|system", "value": "..."}]}`
pub struct ShareGptFormat;

impl FormatConverter for ShareGptFormat {
    fn name(&self) -> &str {
        "sharegpt"
    }

    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value> {
        let conversations: Vec<serde_json::Value> = example
            .messages
            .iter()
            .map(|msg| {
                let from = match msg.role {
                    TrainingRole::System => "system",
                    TrainingRole::User => "human",
                    TrainingRole::Assistant => "gpt",
                    TrainingRole::Tool => "tool",
                };
                json!({
                    "from": from,
                    "value": msg.content,
                })
            })
            .collect();

        Ok(json!({ "conversations": conversations }))
    }

    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample> {
        let conversations = value
            .get("conversations")
            .and_then(|v| v.as_array())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing or invalid 'conversations' field".to_string(),
            })?;

        let mut messages = Vec::with_capacity(conversations.len());
        for conv in conversations {
            let from = conv.get("from").and_then(|v| v.as_str()).ok_or_else(|| {
                DatasetError::FormatConversion {
                    message: "Conversation entry missing 'from'".to_string(),
                }
            })?;

            let role = match from {
                "system" => TrainingRole::System,
                "human" | "user" => TrainingRole::User,
                "gpt" | "assistant" | "chatgpt" => TrainingRole::Assistant,
                "tool" => TrainingRole::Tool,
                other => {
                    return Err(DatasetError::FormatConversion {
                        message: format!("Unknown ShareGPT role: {}", other),
                    });
                }
            };

            let content = conv
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            messages.push(TrainingMessage::new(role, content));
        }

        Ok(TrainingExample::new(messages))
    }
}

use super::super::types::PreferencePair;
use super::PreferenceConverter;

impl PreferenceConverter for ShareGptFormat {
    fn name(&self) -> &str {
        "sharegpt"
    }

    fn preference_to_json(&self, pair: &PreferencePair) -> DatasetResult<serde_json::Value> {
        let to_convs = |msgs: &[TrainingMessage]| -> Vec<serde_json::Value> {
            msgs.iter()
                .map(|msg| {
                    let from = match msg.role {
                        TrainingRole::System => "system",
                        TrainingRole::User => "human",
                        TrainingRole::Assistant => "gpt",
                        TrainingRole::Tool => "tool",
                    };
                    json!({ "from": from, "value": msg.content })
                })
                .collect()
        };

        let mut result = json!({
            "conversations": to_convs(&pair.prompt),
            "chosen": to_convs(&pair.chosen),
            "rejected": to_convs(&pair.rejected),
        });

        if !pair.metadata.is_empty() {
            result["metadata"] = json!(pair.metadata);
        }

        Ok(result)
    }

    fn parse_preference_json(&self, value: &serde_json::Value) -> DatasetResult<PreferencePair> {
        let parse_convs = |key: &str| -> DatasetResult<Vec<TrainingMessage>> {
            let arr = value.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
                DatasetError::FormatConversion {
                    message: format!("Missing or invalid '{}' field", key),
                }
            })?;
            let mut msgs = Vec::new();
            for conv in arr {
                let from = conv.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let role = match from {
                    "system" => TrainingRole::System,
                    "human" | "user" => TrainingRole::User,
                    "gpt" | "assistant" | "chatgpt" => TrainingRole::Assistant,
                    "tool" => TrainingRole::Tool,
                    other => {
                        return Err(DatasetError::FormatConversion {
                            message: format!("Unknown role: {}", other),
                        });
                    }
                };
                let content = conv
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                msgs.push(TrainingMessage::new(role, content));
            }
            Ok(msgs)
        };

        let prompt = parse_convs("conversations")?;
        let chosen = parse_convs("chosen")?;
        let rejected = parse_convs("rejected")?;

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
    fn test_sharegpt_roundtrip() {
        let format = ShareGptFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::system("You are helpful"),
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi!"),
        ]);

        let json = format.to_json(&example).unwrap();
        let convs = json["conversations"].as_array().unwrap();
        assert_eq!(convs[0]["from"], "system");
        assert_eq!(convs[1]["from"], "human");
        assert_eq!(convs[2]["from"], "gpt");

        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[1].role, TrainingRole::User);
    }

    #[test]
    fn test_sharegpt_alternate_roles() {
        let format = ShareGptFormat;
        let json = json!({
            "conversations": [
                {"from": "user", "value": "Hello"},
                {"from": "chatgpt", "value": "Hi!"},
            ]
        });
        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages[0].role, TrainingRole::User);
        assert_eq!(parsed.messages[1].role, TrainingRole::Assistant);
    }

    #[test]
    fn test_sharegpt_preference_roundtrip() {
        use super::PreferenceConverter;
        use crate::datasets::types::PreferencePair;
        let format = ShareGptFormat;
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        );
        let json = format.preference_to_json(&pair).unwrap();
        let parsed = format.parse_preference_json(&json).unwrap();
        assert_eq!(parsed.prompt[0].content, "Q");
        assert_eq!(parsed.chosen[0].content, "Good");
    }
}
