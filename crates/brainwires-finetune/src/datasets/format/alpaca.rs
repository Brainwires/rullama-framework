use serde_json::json;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{TrainingExample, TrainingMessage, TrainingRole};
use super::FormatConverter;

/// Stanford Alpaca format.
///
/// Format: `{"instruction": "...", "input": "...", "output": "..."}`
pub struct AlpacaFormat;

impl FormatConverter for AlpacaFormat {
    fn name(&self) -> &str {
        "alpaca"
    }

    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value> {
        // Map multi-turn to Alpaca: system -> instruction context, user -> instruction, assistant -> output
        let system = example
            .messages
            .iter()
            .find(|m| m.role == TrainingRole::System)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let user_messages: Vec<_> = example
            .messages
            .iter()
            .filter(|m| m.role == TrainingRole::User)
            .collect();

        let assistant_messages: Vec<_> = example
            .messages
            .iter()
            .filter(|m| m.role == TrainingRole::Assistant)
            .collect();

        let instruction = user_messages
            .first()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let output = assistant_messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let mut result = json!({
            "instruction": instruction,
            "output": output,
        });

        if !system.is_empty() {
            result["input"] = json!(system);
        } else {
            result["input"] = json!("");
        }

        Ok(result)
    }

    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample> {
        let instruction = value
            .get("instruction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'instruction' field".to_string(),
            })?;

        let input = value.get("input").and_then(|v| v.as_str()).unwrap_or("");

        let output = value
            .get("output")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'output' field".to_string(),
            })?;

        let mut messages = Vec::new();

        if !input.is_empty() {
            messages.push(TrainingMessage::system(input));
        }

        messages.push(TrainingMessage::user(instruction));
        messages.push(TrainingMessage::assistant(output));

        Ok(TrainingExample::new(messages))
    }
}

use super::super::types::PreferencePair;
use super::PreferenceConverter;

impl PreferenceConverter for AlpacaFormat {
    fn name(&self) -> &str {
        "alpaca"
    }

    fn preference_to_json(&self, pair: &PreferencePair) -> DatasetResult<serde_json::Value> {
        let instruction = pair
            .prompt
            .iter()
            .find(|m| m.role == TrainingRole::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let chosen_output = pair
            .chosen
            .iter()
            .find(|m| m.role == TrainingRole::Assistant)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let rejected_output = pair
            .rejected
            .iter()
            .find(|m| m.role == TrainingRole::Assistant)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let mut result = json!({
            "instruction": instruction,
            "chosen_output": chosen_output,
            "rejected_output": rejected_output,
        });

        if !pair.metadata.is_empty() {
            result["metadata"] = json!(pair.metadata);
        }

        Ok(result)
    }

    fn parse_preference_json(&self, value: &serde_json::Value) -> DatasetResult<PreferencePair> {
        let instruction = value
            .get("instruction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'instruction' field".to_string(),
            })?;

        let chosen_output = value
            .get("chosen_output")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'chosen_output' field".to_string(),
            })?;

        let rejected_output = value
            .get("rejected_output")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::FormatConversion {
                message: "Missing 'rejected_output' field".to_string(),
            })?;

        let mut pair = PreferencePair::new(
            vec![TrainingMessage::user(instruction)],
            vec![TrainingMessage::assistant(chosen_output)],
            vec![TrainingMessage::assistant(rejected_output)],
        );

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
    fn test_alpaca_roundtrip() {
        let format = AlpacaFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::system("You are a math tutor"),
            TrainingMessage::user("What is 2+2?"),
            TrainingMessage::assistant("4"),
        ]);

        let json = format.to_json(&example).unwrap();
        assert_eq!(json["instruction"], "What is 2+2?");
        assert_eq!(json["input"], "You are a math tutor");
        assert_eq!(json["output"], "4");

        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[0].role, TrainingRole::System);
    }

    #[test]
    fn test_alpaca_no_system() {
        let format = AlpacaFormat;
        let example = TrainingExample::new(vec![
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi!"),
        ]);

        let json = format.to_json(&example).unwrap();
        assert_eq!(json["input"], "");

        let parsed = format.parse_json(&json).unwrap();
        assert_eq!(parsed.messages.len(), 2);
    }

    #[test]
    fn test_alpaca_preference_roundtrip() {
        use super::PreferenceConverter;
        use crate::datasets::types::PreferencePair;
        let format = AlpacaFormat;
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("What is 2+2?")],
            vec![TrainingMessage::assistant("4")],
            vec![TrainingMessage::assistant("22")],
        );
        let json = format.preference_to_json(&pair).unwrap();
        assert_eq!(json["instruction"], "What is 2+2?");
        assert_eq!(json["chosen_output"], "4");
        assert_eq!(json["rejected_output"], "22");
        let parsed = format.parse_preference_json(&json).unwrap();
        assert_eq!(parsed.prompt[0].content, "What is 2+2?");
    }
}
