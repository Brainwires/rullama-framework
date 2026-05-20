use serde_json::json;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{TrainingExample, TrainingMessage, TrainingRole};
use super::FormatConverter;

/// Together AI fine-tuning format.
///
/// Uses OpenAI-compatible chat format but with `text` wrapper:
/// `{"text": "<s>[INST] ... [/INST] ..."}`
///
/// For chat format (preferred), same as OpenAI: `{"messages": [...]}`
pub struct TogetherFormat {
    /// If true, use chat messages format (OpenAI-compatible). If false, use text template.
    pub use_chat_format: bool,
}

impl Default for TogetherFormat {
    fn default() -> Self {
        Self {
            use_chat_format: true,
        }
    }
}

impl TogetherFormat {
    /// Create a Together format using chat messages (OpenAI-compatible).
    pub fn chat() -> Self {
        Self {
            use_chat_format: true,
        }
    }

    /// Create a Together format using text template wrapping.
    pub fn text() -> Self {
        Self {
            use_chat_format: false,
        }
    }

    fn messages_to_text(messages: &[TrainingMessage]) -> String {
        let mut text = String::new();
        for msg in messages {
            match msg.role {
                TrainingRole::System => {
                    text.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
                }
                TrainingRole::User => {
                    text.push_str(&format!("[INST] {} [/INST] ", msg.content));
                }
                TrainingRole::Assistant => {
                    text.push_str(&format!("{}\n", msg.content));
                }
                TrainingRole::Tool => {
                    text.push_str(&format!("[TOOL] {} [/TOOL] ", msg.content));
                }
            }
        }
        format!("<s>{}</s>", text.trim())
    }
}

impl FormatConverter for TogetherFormat {
    fn name(&self) -> &str {
        "together"
    }

    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value> {
        if self.use_chat_format {
            // Same as OpenAI format
            let messages: Vec<serde_json::Value> = example
                .messages
                .iter()
                .map(|msg| {
                    json!({
                        "role": msg.role.to_string(),
                        "content": msg.content,
                    })
                })
                .collect();
            Ok(json!({ "messages": messages }))
        } else {
            let text = Self::messages_to_text(&example.messages);
            Ok(json!({ "text": text }))
        }
    }

    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample> {
        // Prefer chat format parsing
        if let Some(messages) = value.get("messages") {
            let arr = messages
                .as_array()
                .ok_or_else(|| DatasetError::FormatConversion {
                    message: "'messages' must be an array".to_string(),
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
                            message: "Invalid or missing role".to_string(),
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
            Ok(TrainingExample::new(msgs))
        } else if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            // Basic text format parsing — extract user/assistant turns
            let mut messages = Vec::new();
            let text = text
                .trim_start_matches("<s>")
                .trim_end_matches("</s>")
                .trim();

            // Extract system message if present
            if let Some(sys_start) = text.find("<<SYS>>")
                && let Some(sys_end) = text.find("<</SYS>>")
            {
                let system_content = text[sys_start + 7..sys_end].trim().to_string();
                messages.push(TrainingMessage::system(system_content));
            }

            // Extract [INST]...[/INST] pairs
            let mut remaining = text;
            while let Some(inst_start) = remaining.find("[INST]") {
                if let Some(inst_end) = remaining.find("[/INST]") {
                    let user_content = remaining[inst_start + 6..inst_end].trim().to_string();
                    messages.push(TrainingMessage::user(user_content));

                    remaining = &remaining[inst_end + 7..];
                    // Everything until next [INST] or end is assistant
                    let assistant_end = remaining.find("[INST]").unwrap_or(remaining.len());
                    let assistant_content = remaining[..assistant_end].trim().to_string();
                    if !assistant_content.is_empty() {
                        messages.push(TrainingMessage::assistant(assistant_content));
                    }
                    remaining = &remaining[assistant_end..];
                } else {
                    break;
                }
            }

            if messages.is_empty() {
                return Err(DatasetError::FormatConversion {
                    message: "Could not parse Together text format".to_string(),
                });
            }

            Ok(TrainingExample::new(messages))
        } else {
            Err(DatasetError::FormatConversion {
                message: "Expected 'messages' or 'text' field".to_string(),
            })
        }
    }
}

use super::super::types::PreferencePair;
use super::PreferenceConverter;

impl PreferenceConverter for TogetherFormat {
    fn name(&self) -> &str {
        "together"
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
    fn test_together_chat_roundtrip() {
        let format = TogetherFormat::chat();
        let example = TrainingExample::new(vec![
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi!"),
        ]);

        let json = format.to_json(&example).unwrap();
        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages.len(), 2);
    }

    #[test]
    fn test_together_text_format() {
        let format = TogetherFormat::text();
        let example = TrainingExample::new(vec![
            TrainingMessage::system("Be helpful"),
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi!"),
        ]);

        let json = format.to_json(&example).unwrap();
        let text = json["text"].as_str().unwrap();
        assert!(text.starts_with("<s>"));
        assert!(text.ends_with("</s>"));
        assert!(text.contains("<<SYS>>"));
        assert!(text.contains("[INST]"));
    }

    #[test]
    fn test_together_preference_roundtrip() {
        use super::PreferenceConverter;
        use crate::datasets::types::PreferencePair;
        let format = TogetherFormat::chat();
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        );
        let json = format.preference_to_json(&pair).unwrap();
        let parsed = format.parse_preference_json(&json).unwrap();
        assert_eq!(parsed.chosen[0].content, "Good");
    }
}
