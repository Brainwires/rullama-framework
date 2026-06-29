/// Alpaca instruction-following format converter.
pub mod alpaca;
/// ChatML template format converter.
pub mod chatml;
/// OpenAI fine-tuning format converter.
pub mod openai;
/// ShareGPT conversation format converter.
pub mod sharegpt;
/// Together AI fine-tuning format converter.
pub mod together;

use super::error::DatasetResult;
use super::types::{DataFormat, PreferencePair, TrainingExample};

/// Convert training examples to/from a specific provider format.
pub trait FormatConverter: Send + Sync {
    /// Name of this format (e.g., "openai", "alpaca").
    fn name(&self) -> &str;

    /// Convert a TrainingExample to this format's JSON representation.
    fn to_json(&self, example: &TrainingExample) -> DatasetResult<serde_json::Value>;

    /// Parse this format's JSON back into a TrainingExample.
    fn parse_json(&self, value: &serde_json::Value) -> DatasetResult<TrainingExample>;

    /// Convert a batch of examples to this format.
    fn to_json_batch(&self, examples: &[TrainingExample]) -> DatasetResult<Vec<serde_json::Value>> {
        examples.iter().map(|e| self.to_json(e)).collect()
    }

    /// Parse a batch of JSON values into training examples.
    fn parse_json_batch(
        &self,
        values: &[serde_json::Value],
    ) -> DatasetResult<Vec<TrainingExample>> {
        values.iter().map(|v| self.parse_json(v)).collect()
    }
}

/// Convert preference pairs to/from a specific provider format.
pub trait PreferenceConverter: Send + Sync {
    /// Name of this format.
    fn name(&self) -> &str;

    /// Convert a PreferencePair to this format's JSON representation.
    fn preference_to_json(&self, pair: &PreferencePair) -> DatasetResult<serde_json::Value>;

    /// Parse this format's JSON back into a PreferencePair.
    fn parse_preference_json(&self, value: &serde_json::Value) -> DatasetResult<PreferencePair>;

    /// Convert a batch of preference pairs to this format.
    fn preference_to_json_batch(
        &self,
        pairs: &[PreferencePair],
    ) -> DatasetResult<Vec<serde_json::Value>> {
        pairs.iter().map(|p| self.preference_to_json(p)).collect()
    }

    /// Parse a batch of JSON values into preference pairs.
    fn parse_preference_json_batch(
        &self,
        values: &[serde_json::Value],
    ) -> DatasetResult<Vec<PreferencePair>> {
        values
            .iter()
            .map(|v| self.parse_preference_json(v))
            .collect()
    }
}

/// Auto-detect the format of a JSON value.
pub fn detect_format(value: &serde_json::Value) -> Option<DataFormat> {
    if value.get("messages").is_some() {
        return Some(DataFormat::OpenAI);
    }
    if value.get("instruction").is_some() && value.get("output").is_some() {
        return Some(DataFormat::Alpaca);
    }
    if value.get("conversations").is_some() {
        return Some(DataFormat::ShareGpt);
    }
    if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        if text.contains("<|im_start|>") {
            return Some(DataFormat::ChatMl);
        }
        return Some(DataFormat::Together);
    }
    None
}

pub use alpaca::AlpacaFormat;
pub use chatml::ChatMlFormat;
pub use openai::OpenAiFormat;
pub use sharegpt::ShareGptFormat;
pub use together::TogetherFormat;
