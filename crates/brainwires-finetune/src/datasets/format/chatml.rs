use serde_json::json;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{TrainingExample, TrainingMessage, TrainingRole};
use super::FormatConverter;

/// ChatML template format.
///
/// Format: `{"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n..."}`
pub struct ChatMlFormat;

impl ChatMlFormat {
    fn messages_to_chatml(messages: &[TrainingMessage]) -> String {
        let mut text = String::new();
        for msg in messages {
            let role = msg.role.to_string();
            text.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                role, msg.content
            ));
        }
        text
    }

    fn parse_chatml(text: &str) -> DatasetResult<Vec<TrainingMessage>> {
        let mut messages = Vec::new();
        let mut remaining = text;

        while let Some(start) = remaining.find("<|im_start|>") {
            remaining = &remaining[start + 12..]; // skip "<|im_start|>"

            let end =
                remaining
                    .find("<|im_end|>")
                    .ok_or_else(|| DatasetError::FormatConversion {
                        message: "Unclosed <|im_start|> tag".to_string(),
                    })?;

            let block = &remaining[..end];
            let newline_pos = block.find('\n').unwrap_or(block.len());
            let role_str = block[..newline_pos].trim();
            let content = if newline_pos < block.len() {
                block[newline_pos + 1..].trim().to_string()
            } else {
                String::new()
            };

            let role = match role_str {
                "system" => TrainingRole::System,
                "user" => TrainingRole::User,
                "assistant" => TrainingRole::Assistant,
                "tool" => TrainingRole::Tool,
                other => {
                    return Err(DatasetError::FormatConversion {
                        message: format!("Unknown ChatML role: {}", other),
                    });
                }
            };

            messages.push(TrainingMessage::new(role, content));
            remaining = &remaining[end + 10..]; // skip "<|im_end|>"
        }

        if messages.is_empty() {
            return Err(DatasetError::FormatConversion {
                message: "No ChatML messages found".to_string(),
            });
        }

        Ok(messages)
    }
}

impl FormatConverter for ChatMlFormat {
    fn name(&self) -> &str {
        "chatml"
    }

    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value> {
        let text = Self::messages_to_chatml(&example.messages);
        Ok(json!({ "text": text }))
    }

    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample> {
        let text = value.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
            DatasetError::FormatConversion {
                message: "Missing 'text' field for ChatML format".to_string(),
            }
        })?;

        let messages = Self::parse_chatml(text)?;
        Ok(TrainingExample::new(messages))
    }
}

use super::super::types::PreferencePair;
use super::PreferenceConverter;

impl PreferenceConverter for ChatMlFormat {
    fn name(&self) -> &str {
        "chatml"
    }

    fn preference_to_json(&self, pair: &PreferencePair) -> DatasetResult<serde_json::Value> {
        let chosen_text = Self::messages_to_chatml(&pair.chosen);
        let rejected_text = Self::messages_to_chatml(&pair.rejected);
        let prompt_text = Self::messages_to_chatml(&pair.prompt);

        let mut result = json!({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        });

        if !pair.metadata.is_empty() {
            result["metadata"] = json!(pair.metadata);
        }

        Ok(result)
    }

    fn parse_preference_json(&self, value: &serde_json::Value) -> DatasetResult<PreferencePair> {
        let prompt_text = value
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'prompt' field for ChatML preference".to_string(),
            })?;

        let chosen_text = value
            .get("chosen")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'chosen' field for ChatML preference".to_string(),
            })?;

        let rejected_text = value
            .get("rejected")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'rejected' field for ChatML preference".to_string(),
            })?;

        let prompt = Self::parse_chatml(prompt_text)?;
        let chosen = Self::parse_chatml(chosen_text)?;
        let rejected = Self::parse_chatml(rejected_text)?;

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
    fn test_chatml_roundtrip() {
        let format = ChatMlFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::system("You are helpful"),
            TrainingMessage::user("What is Rust?"),
            TrainingMessage::assistant("Rust is a systems programming language."),
        ]);

        let json = format.to_json(&example).unwrap();
        let text = json["text"].as_str().unwrap();
        assert!(text.contains("<|im_start|>system"));
        assert!(text.contains("<|im_start|>user"));
        assert!(text.contains("<|im_start|>assistant"));
        assert!(text.contains("<|im_end|>"));

        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[0].role, TrainingRole::System);
        assert_eq!(
            parsed.messages[2].content,
            "Rust is a systems programming language."
        );
    }

    #[test]
    fn test_chatml_format_structure() {
        let text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n";
        let messages = ChatMlFormat::parse_chatml(text).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "Hi!");
    }

    #[test]
    fn test_chatml_preference_roundtrip() {
        use super::PreferenceConverter;
        use crate::datasets::types::PreferencePair;
        let format = ChatMlFormat;
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
