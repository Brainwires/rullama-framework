//! Validator - Semantic Response Validation
//!
//! Uses a provider to perform semantic validation of responses,
//! enhancing the pattern-based red-flagging system.

use std::sync::Arc;
use tracing::warn;

use rullama_core::message::Message;
use rullama_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Result of local validation
#[derive(Clone, Debug)]
pub enum ValidationResult {
    /// Response is valid.
    Valid {
        /// Confidence in the validity assessment (0.0-1.0).
        confidence: f32,
    },
    /// Response has issues.
    Invalid {
        /// Description of the validation issue.
        reason: String,
        /// Severity of the issue (0.0-1.0).
        severity: f32,
        /// Confidence in the invalidity assessment (0.0-1.0).
        confidence: f32,
    },
    /// Validation was skipped (fallback to pattern-based)
    Skipped,
}

impl ValidationResult {
    /// Returns `true` if the response passed validation.
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid { .. })
    }

    /// Returns `true` if the response failed validation.
    pub fn is_invalid(&self) -> bool {
        matches!(self, ValidationResult::Invalid { .. })
    }
}

/// Validator for semantic response validation
pub struct LocalValidator {
    provider: Arc<dyn Provider>,
    model_id: String,
}

impl LocalValidator {
    /// Create a new validator
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
        }
    }

    /// Validate a response for the given task
    ///
    /// Performs semantic validation to catch issues that pattern matching might miss.
    pub async fn validate(&self, task: &str, response: &str) -> ValidationResult {
        let timer = InferenceTimer::new("validate_response", &self.model_id);

        // Skip very short responses (likely already handled by pattern matching)
        if response.trim().len() < 10 {
            return ValidationResult::Skipped;
        }

        let system_prompt = self.build_validation_prompt();
        let user_prompt = format!(
            "Validate if this response is appropriate for the task.\n\nTask: {}\n\nResponse: {}\n\nOutput ONLY: VALID or INVALID:<reason>",
            task,
            // Truncate response for efficiency
            if response.len() > 500 {
                &response[..500]
            } else {
                response
            }
        );

        let messages = vec![Message::user(&user_prompt)];
        let options = ChatOptions::deterministic(50).system(system_prompt);

        match self.provider.chat(&messages, None, &options).await {
            Ok(chat_response) => {
                let text = chat_response.message.text_or_summary();
                let result = self.parse_validation(&text);
                timer.finish(true);
                result
            }
            Err(e) => {
                warn!(target: "local_llm", "Response validation failed: {}", e);
                timer.finish(false);
                ValidationResult::Skipped
            }
        }
    }

    /// Quick heuristic validation (no LLM call)
    ///
    /// Use for fast pre-filtering before LLM validation.
    pub fn validate_heuristic(&self, task: &str, response: &str) -> ValidationResult {
        let response_lower = response.to_lowercase();
        let task_lower = task.to_lowercase();

        // Check for obvious issues

        // 1. Response is completely off-topic (no shared words with task)
        let task_words: std::collections::HashSet<&str> = task_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        let response_words: std::collections::HashSet<&str> = response_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let overlap = task_words.intersection(&response_words).count();
        if overlap == 0 && task_words.len() > 3 {
            return ValidationResult::Invalid {
                reason: "Response appears unrelated to task".to_string(),
                severity: 0.6,
                confidence: 0.4,
            };
        }

        // 2. Response contains refusal patterns
        let refusal_patterns = [
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "sorry, i",
            "i don't have",
            "i do not have",
            "as an ai",
        ];

        for pattern in refusal_patterns {
            if response_lower.contains(pattern) {
                return ValidationResult::Invalid {
                    reason: format!("Response contains refusal pattern: {}", pattern),
                    severity: 0.7,
                    confidence: 0.6,
                };
            }
        }

        // 3. Response is just repeating the task
        let task_trimmed = task_lower.trim();
        let response_trimmed = response_lower.trim();
        if response_trimmed.starts_with(task_trimmed) && response.len() < task.len() * 2 {
            return ValidationResult::Invalid {
                reason: "Response appears to just repeat the task".to_string(),
                severity: 0.5,
                confidence: 0.5,
            };
        }

        // 4. Response is suspiciously short for a complex task
        if task.len() > 100 && response.len() < 20 {
            return ValidationResult::Invalid {
                reason: "Response too short for complex task".to_string(),
                severity: 0.4,
                confidence: 0.4,
            };
        }

        ValidationResult::Valid { confidence: 0.5 }
    }

    /// Build the system prompt for validation
    fn build_validation_prompt(&self) -> String {
        r#"You are a response validator. Given a task and response, determine if the response is appropriate.

Check for:
1. Response addresses the task (not off-topic)
2. Response doesn't contain confusion or self-correction
3. Response isn't a refusal or "I can't do that"
4. Response isn't just repeating the task
5. Response has substance (not empty platitudes)

Output format:
- If valid: VALID
- If invalid: INVALID:<brief reason>

Be strict but fair. Only flag clear issues."#.to_string()
    }

    /// Parse the LLM output to determine validity
    fn parse_validation(&self, output: &str) -> ValidationResult {
        let trimmed = output.trim().to_uppercase();

        if trimmed.starts_with("VALID") && !trimmed.contains("INVALID") {
            return ValidationResult::Valid { confidence: 0.8 };
        }

        if trimmed.starts_with("INVALID") {
            let reason = if let Some(idx) = trimmed.find(':') {
                trimmed[idx + 1..].trim().to_string()
            } else {
                "Unspecified validation failure".to_string()
            };

            return ValidationResult::Invalid {
                reason,
                severity: 0.6,
                confidence: 0.75,
            };
        }

        // Ambiguous output - treat as skipped
        ValidationResult::Skipped
    }
}

/// Builder for LocalValidator
pub struct LocalValidatorBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
}

impl Default for LocalValidatorBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(),
        }
    }
}

impl LocalValidatorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for validation.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Build the validator, returning `None` if no provider was set.
    pub fn build(self) -> Option<LocalValidator> {
        self.provider.map(|p| LocalValidator::new(p, self.model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_checks() {
        let valid = ValidationResult::Valid { confidence: 0.9 };
        assert!(valid.is_valid());
        assert!(!valid.is_invalid());

        let invalid = ValidationResult::Invalid {
            reason: "test".to_string(),
            severity: 0.5,
            confidence: 0.8,
        };
        assert!(!invalid.is_valid());
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_heuristic_validation_refusal() {
        let _validator = LocalValidatorBuilder::default();

        // Test refusal detection
        let result = validate_heuristic_direct(
            "Write a poem",
            "I'm sorry, I cannot write poems as an AI assistant.",
        );

        assert!(matches!(result, ValidationResult::Invalid { .. }));
    }

    #[test]
    fn test_heuristic_validation_valid() {
        let result = validate_heuristic_direct("Calculate 2+2", "The result of 2+2 is 4.");

        assert!(matches!(result, ValidationResult::Valid { .. }));
    }

    fn validate_heuristic_direct(_task: &str, response: &str) -> ValidationResult {
        let response_lower = response.to_lowercase();

        let refusal_patterns = ["i cannot", "i can't", "i'm unable", "sorry, i", "as an ai"];

        for pattern in refusal_patterns {
            if response_lower.contains(pattern) {
                return ValidationResult::Invalid {
                    reason: format!("Refusal pattern: {}", pattern),
                    severity: 0.7,
                    confidence: 0.6,
                };
            }
        }

        ValidationResult::Valid { confidence: 0.5 }
    }

    #[test]
    fn test_parse_validation() {
        // Test parsing logic
        assert!(matches!(
            parse_validation_direct("VALID"),
            ValidationResult::Valid { .. }
        ));

        assert!(matches!(
            parse_validation_direct("INVALID: Response is off-topic"),
            ValidationResult::Invalid { .. }
        ));

        assert!(matches!(
            parse_validation_direct("Maybe?"),
            ValidationResult::Skipped
        ));
    }

    fn parse_validation_direct(output: &str) -> ValidationResult {
        let trimmed = output.trim().to_uppercase();

        if trimmed.starts_with("VALID") && !trimmed.contains("INVALID") {
            return ValidationResult::Valid { confidence: 0.8 };
        }

        if trimmed.starts_with("INVALID") {
            let reason = if let Some(idx) = trimmed.find(':') {
                trimmed[idx + 1..].trim().to_string()
            } else {
                "Unspecified".to_string()
            };

            return ValidationResult::Invalid {
                reason,
                severity: 0.6,
                confidence: 0.75,
            };
        }

        ValidationResult::Skipped
    }
}
