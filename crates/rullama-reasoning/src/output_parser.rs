//! Structured output parsing for LLM responses
//!
//! Provides parsers that extract structured data from raw LLM text output.
//! Supports JSON extraction, regex-based parsing, and retry-on-invalid patterns.
//!
//! # Example
//!
//! ```rust
//! use rullama_reasoning::output_parser::{JsonOutputParser, OutputParser};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Review {
//!     sentiment: String,
//!     score: f32,
//! }
//!
//! let parser = JsonOutputParser::<Review>::new();
//! let raw = r#"Here's my analysis: {"sentiment": "positive", "score": 0.9}"#;
//! let review = parser.parse(raw).unwrap();
//! assert_eq!(review.sentiment, "positive");
//! ```

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

/// Trait for parsing structured output from LLM text responses.
pub trait OutputParser: Send + Sync {
    /// The output type produced by this parser.
    type Output;

    /// Parse the raw LLM response text into structured output.
    fn parse(&self, text: &str) -> Result<Self::Output>;

    /// Return format instructions to inject into the prompt.
    ///
    /// These instructions tell the LLM how to format its response so this
    /// parser can extract structured data from it.
    fn format_instructions(&self) -> String;
}

/// Extracts JSON from LLM responses and deserializes into `T`.
///
/// Handles common LLM quirks:
/// - JSON wrapped in markdown code fences
/// - JSON embedded in surrounding prose
/// - Leading/trailing whitespace
pub struct JsonOutputParser<T> {
    _phantom: PhantomData<T>,
}

impl<T> JsonOutputParser<T> {
    /// Create a new JSON output parser.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for JsonOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DeserializeOwned + Send + Sync> OutputParser for JsonOutputParser<T> {
    type Output = T;

    fn parse(&self, text: &str) -> Result<T> {
        let json_str = extract_json(text).context("No JSON found in LLM response")?;
        serde_json::from_str(&json_str).context("Failed to parse JSON from LLM response")
    }

    fn format_instructions(&self) -> String {
        "Respond with valid JSON only. Do not include any other text before or after the JSON."
            .to_string()
    }
}

/// Extracts a list of items from a JSON array in the LLM response.
pub struct JsonListParser<T> {
    _phantom: PhantomData<T>,
}

impl<T> JsonListParser<T> {
    /// Create a new JSON list parser.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for JsonListParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DeserializeOwned + Send + Sync> OutputParser for JsonListParser<T> {
    type Output = Vec<T>;

    fn parse(&self, text: &str) -> Result<Vec<T>> {
        let json_str = extract_json(text).context("No JSON array found in LLM response")?;
        serde_json::from_str(&json_str).context("Failed to parse JSON array from LLM response")
    }

    fn format_instructions(&self) -> String {
        "Respond with a valid JSON array only. Do not include any other text.".to_string()
    }
}

/// Parses LLM output using a regex pattern with named capture groups.
pub struct RegexOutputParser {
    pattern: regex::Regex,
}

impl RegexOutputParser {
    /// Create a new regex parser.
    ///
    /// The pattern should use named capture groups like `(?P<name>...)`.
    pub fn new(pattern: &str) -> Result<Self> {
        let regex = regex::Regex::new(pattern).context("Invalid regex pattern")?;
        Ok(Self { pattern: regex })
    }
}

impl OutputParser for RegexOutputParser {
    type Output = std::collections::HashMap<String, String>;

    fn parse(&self, text: &str) -> Result<Self::Output> {
        let caps = self
            .pattern
            .captures(text)
            .context("Regex pattern did not match LLM output")?;

        let mut result = std::collections::HashMap::new();
        for name in self.pattern.capture_names().flatten() {
            if let Some(m) = caps.name(name) {
                result.insert(name.to_string(), m.as_str().to_string());
            }
        }
        Ok(result)
    }

    fn format_instructions(&self) -> String {
        format!(
            "Format your response to match this pattern: {}",
            self.pattern.as_str()
        )
    }
}

/// Extract JSON from text that may contain markdown fences or surrounding prose.
fn extract_json(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Try direct parse first
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        return Some(trimmed.to_string());
    }

    // Try markdown code fence: ```json ... ``` or ``` ... ```
    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        // Skip optional language tag
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            let json_str = content[..end].trim();
            if !json_str.is_empty() {
                return Some(json_str.to_string());
            }
        }
    }

    // Try to find first { or [ and match to last } or ]
    let obj_start = trimmed.find('{');
    let arr_start = trimmed.find('[');

    let start_idx = match (obj_start, arr_start) {
        (Some(o), Some(a)) => Some(o.min(a)),
        (Some(o), None) => Some(o),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    }?;

    let close_char = if trimmed.as_bytes()[start_idx] == b'{' {
        '}'
    } else {
        ']'
    };

    let end_idx = trimmed.rfind(close_char)?;
    if end_idx > start_idx {
        Some(trimmed[start_idx..=end_idx].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestStruct {
        name: String,
        value: i32,
    }

    #[test]
    fn test_json_parser_clean() {
        let parser = JsonOutputParser::<TestStruct>::new();
        let result = parser.parse(r#"{"name": "test", "value": 42}"#).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_json_parser_with_prose() {
        let parser = JsonOutputParser::<TestStruct>::new();
        let input = r#"Here is the result: {"name": "test", "value": 42} Hope that helps!"#;
        let result = parser.parse(input).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_json_parser_with_code_fence() {
        let parser = JsonOutputParser::<TestStruct>::new();
        let input = "Here's the JSON:\n```json\n{\"name\": \"test\", \"value\": 42}\n```";
        let result = parser.parse(input).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_json_list_parser() {
        let parser = JsonListParser::<TestStruct>::new();
        let input = r#"[{"name": "a", "value": 1}, {"name": "b", "value": 2}]"#;
        let result = parser.parse(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "a");
        assert_eq!(result[1].name, "b");
    }

    #[test]
    fn test_regex_parser() {
        let parser =
            RegexOutputParser::new(r"sentiment: (?P<sentiment>\w+), score: (?P<score>[\d.]+)")
                .unwrap();
        let result = parser
            .parse("The sentiment: positive, score: 0.95 overall")
            .unwrap();
        assert_eq!(result["sentiment"], "positive");
        assert_eq!(result["score"], "0.95");
    }

    #[test]
    fn test_json_parser_no_json() {
        let parser = JsonOutputParser::<TestStruct>::new();
        assert!(parser.parse("no json here at all").is_err());
    }

    #[test]
    fn test_format_instructions() {
        let parser = JsonOutputParser::<TestStruct>::new();
        let instructions = parser.format_instructions();
        assert!(instructions.contains("JSON"));
    }

    #[test]
    fn test_extract_json_array_in_prose() {
        let input = r#"Here are the items: [1, 2, 3] done."#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, "[1, 2, 3]");
    }
}
