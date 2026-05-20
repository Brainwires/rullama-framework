use super::super::error::DatasetResult;
use super::super::types::{PreferencePair, TrainingExample, TrainingRole};

/// Validation issue severity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueSeverity {
    /// A blocking error that makes the example invalid.
    Error,
    /// A non-blocking warning about potential issues.
    Warning,
}

/// A single validation issue found in a dataset example.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// ID of the example where the issue was found.
    pub example_id: String,
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Human-readable description of the issue.
    pub message: String,
    /// Optional line number where the issue was found.
    pub line_number: Option<usize>,
    /// Optional suggestion for how to fix the issue.
    pub suggestion: Option<String>,
}

/// Result of validating a dataset.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
    /// Total number of examples validated.
    pub total_examples: usize,
    /// Number of examples that passed without errors.
    pub valid_examples: usize,
}

impl ValidationReport {
    /// Return true if any error-level issues exist.
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    /// Count the number of error-level issues.
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Count the number of warning-level issues.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }
}

/// Configuration for dataset validation.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Minimum messages per example.
    pub min_messages: usize,
    /// Maximum messages per example.
    pub max_messages: usize,
    /// Maximum tokens per example (estimated).
    pub max_tokens: usize,
    /// Require the last message to be from assistant.
    pub require_assistant_last: bool,
    /// Require a system message.
    pub require_system_message: bool,
    /// Reject empty content.
    pub reject_empty_content: bool,
    /// Require alternating user/assistant turns after system.
    pub require_alternating_turns: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            min_messages: 2,
            max_messages: 1000,
            max_tokens: 32768,
            require_assistant_last: true,
            require_system_message: false,
            reject_empty_content: true,
            require_alternating_turns: false,
        }
    }
}

/// Validates training examples against configurable rules.
pub struct DataValidator {
    config: ValidatorConfig,
}

impl DataValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Create a new validator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Validate a single training example.
    pub fn validate_example(&self, example: &TrainingExample) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let id = &example.id;

        // Check message count
        if example.messages.len() < self.config.min_messages {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Error,
                message: format!(
                    "Too few messages: {} (min: {})",
                    example.messages.len(),
                    self.config.min_messages
                ),
                line_number: None,
                suggestion: None,
            });
        }

        if example.messages.len() > self.config.max_messages {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Warning,
                message: format!(
                    "Too many messages: {} (max: {})",
                    example.messages.len(),
                    self.config.max_messages
                ),
                line_number: None,
                suggestion: None,
            });
        }

        // Check token count
        let tokens = example.estimated_tokens();
        if tokens > self.config.max_tokens {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Warning,
                message: format!(
                    "Estimated tokens ({}) exceeds max ({})",
                    tokens, self.config.max_tokens
                ),
                line_number: None,
                suggestion: None,
            });
        }

        // Check system message requirement
        if self.config.require_system_message && !example.has_system_message() {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Warning,
                message: "Missing system message".to_string(),
                line_number: None,
                suggestion: None,
            });
        }

        // Check last message is assistant
        if self.config.require_assistant_last && !example.ends_with_assistant() {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Error,
                message: "Last message must be from assistant".to_string(),
                line_number: None,
                suggestion: None,
            });
        }

        // Check empty content
        if self.config.reject_empty_content {
            for (i, msg) in example.messages.iter().enumerate() {
                if msg.content.trim().is_empty() && msg.tool_calls.is_none() {
                    issues.push(ValidationIssue {
                        example_id: id.clone(),
                        severity: IssueSeverity::Error,
                        message: format!("Message {} has empty content", i),
                        line_number: None,
                        suggestion: None,
                    });
                }
            }
        }

        // Check alternating turns
        if self.config.require_alternating_turns {
            let non_system: Vec<_> = example
                .messages
                .iter()
                .filter(|m| m.role != TrainingRole::System && m.role != TrainingRole::Tool)
                .collect();
            for window in non_system.windows(2) {
                if window[0].role == window[1].role {
                    issues.push(ValidationIssue {
                        example_id: id.clone(),
                        severity: IssueSeverity::Warning,
                        message: format!(
                            "Consecutive {} messages (expected alternating)",
                            window[0].role
                        ),
                        line_number: None,
                        suggestion: None,
                    });
                    break;
                }
            }
        }

        issues
    }

    /// Validate a preference pair.
    pub fn validate_preference(&self, pair: &PreferencePair) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let id = &pair.id;

        if pair.prompt.is_empty() {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Error,
                message: "Preference pair has empty prompt".to_string(),
                line_number: None,
                suggestion: Some("Add at least one prompt message".to_string()),
            });
        }

        if pair.chosen.is_empty() {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Error,
                message: "Preference pair has empty chosen response".to_string(),
                line_number: None,
                suggestion: Some("Add at least one chosen response message".to_string()),
            });
        }

        if pair.rejected.is_empty() {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Error,
                message: "Preference pair has empty rejected response".to_string(),
                line_number: None,
                suggestion: Some("Add at least one rejected response message".to_string()),
            });
        }

        // Check empty content in messages
        if self.config.reject_empty_content {
            for (i, msg) in pair.prompt.iter().enumerate() {
                if msg.content.trim().is_empty() {
                    issues.push(ValidationIssue {
                        example_id: id.clone(),
                        severity: IssueSeverity::Error,
                        message: format!("Prompt message {} has empty content", i),
                        line_number: None,
                        suggestion: None,
                    });
                }
            }
            for (i, msg) in pair.chosen.iter().enumerate() {
                if msg.content.trim().is_empty() {
                    issues.push(ValidationIssue {
                        example_id: id.clone(),
                        severity: IssueSeverity::Error,
                        message: format!("Chosen message {} has empty content", i),
                        line_number: None,
                        suggestion: None,
                    });
                }
            }
            for (i, msg) in pair.rejected.iter().enumerate() {
                if msg.content.trim().is_empty() {
                    issues.push(ValidationIssue {
                        example_id: id.clone(),
                        severity: IssueSeverity::Error,
                        message: format!("Rejected message {} has empty content", i),
                        line_number: None,
                        suggestion: None,
                    });
                }
            }
        }

        // Warn if chosen == rejected
        if !pair.chosen.is_empty() && !pair.rejected.is_empty() {
            let chosen_text: String = pair
                .chosen
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("");
            let rejected_text: String = pair
                .rejected
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("");
            if chosen_text == rejected_text {
                issues.push(ValidationIssue {
                    example_id: id.clone(),
                    severity: IssueSeverity::Warning,
                    message: "Chosen and rejected responses are identical".to_string(),
                    line_number: None,
                    suggestion: Some("Ensure chosen and rejected responses differ".to_string()),
                });
            }

            // Warn if length ratio > 10x
            let chosen_len = chosen_text.len().max(1);
            let rejected_len = rejected_text.len().max(1);
            let ratio = chosen_len.max(rejected_len) as f64 / chosen_len.min(rejected_len) as f64;
            if ratio > 10.0 {
                issues.push(ValidationIssue {
                    example_id: id.clone(),
                    severity: IssueSeverity::Warning,
                    message: format!(
                        "Length ratio between chosen and rejected is {:.1}x (>10x)",
                        ratio
                    ),
                    line_number: None,
                    suggestion: Some(
                        "Large length differences may indicate data quality issues".to_string(),
                    ),
                });
            }
        }

        // Token count check
        let tokens = pair.estimated_tokens();
        if tokens > self.config.max_tokens {
            issues.push(ValidationIssue {
                example_id: id.clone(),
                severity: IssueSeverity::Warning,
                message: format!(
                    "Estimated tokens ({}) exceeds max ({})",
                    tokens, self.config.max_tokens
                ),
                line_number: None,
                suggestion: None,
            });
        }

        issues
    }

    /// Validate a full preference dataset, producing a report.
    pub fn validate_preference_dataset(
        &self,
        pairs: &[PreferencePair],
    ) -> DatasetResult<ValidationReport> {
        let mut all_issues = Vec::new();
        let mut valid_count = 0;

        for pair in pairs {
            let issues = self.validate_preference(pair);
            if issues.iter().all(|i| i.severity != IssueSeverity::Error) {
                valid_count += 1;
            }
            all_issues.extend(issues);
        }

        tracing::debug!(
            "Validated {} preference pairs: {} valid, {} issues",
            pairs.len(),
            valid_count,
            all_issues.len()
        );

        Ok(ValidationReport {
            issues: all_issues,
            total_examples: pairs.len(),
            valid_examples: valid_count,
        })
    }

    /// Validate a full dataset, producing a report.
    pub fn validate_dataset(
        &self,
        examples: &[TrainingExample],
    ) -> DatasetResult<ValidationReport> {
        let mut all_issues = Vec::new();
        let mut valid_count = 0;

        for example in examples {
            let issues = self.validate_example(example);
            if issues.iter().all(|i| i.severity != IssueSeverity::Error) {
                valid_count += 1;
            }
            all_issues.extend(issues);
        }

        tracing::debug!(
            "Validated {} examples: {} valid, {} issues",
            examples.len(),
            valid_count,
            all_issues.len()
        );

        Ok(ValidationReport {
            issues: all_issues,
            total_examples: examples.len(),
            valid_examples: valid_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::types::TrainingMessage;

    #[test]
    fn test_valid_example() {
        let validator = DataValidator::with_defaults();
        let example = TrainingExample::with_id(
            "test",
            vec![
                TrainingMessage::user("Hello"),
                TrainingMessage::assistant("Hi!"),
            ],
        );
        let issues = validator.validate_example(&example);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_too_few_messages() {
        let validator = DataValidator::with_defaults();
        let example = TrainingExample::with_id("test", vec![TrainingMessage::user("Hello")]);
        let issues = validator.validate_example(&example);
        assert!(issues.iter().any(|i| i.message.contains("Too few")));
        assert!(
            issues
                .iter()
                .any(|i| i.message.contains("must be from assistant"))
        );
    }

    #[test]
    fn test_empty_content_rejected() {
        let validator = DataValidator::with_defaults();
        let example = TrainingExample::with_id(
            "test",
            vec![TrainingMessage::user(""), TrainingMessage::assistant("Hi")],
        );
        let issues = validator.validate_example(&example);
        assert!(issues.iter().any(|i| i.message.contains("empty content")));
    }

    #[test]
    fn test_validation_report() {
        let validator = DataValidator::with_defaults();
        let examples = vec![
            TrainingExample::with_id(
                "good",
                vec![TrainingMessage::user("Q"), TrainingMessage::assistant("A")],
            ),
            TrainingExample::with_id("bad", vec![TrainingMessage::user("Q")]),
        ];
        let report = validator.validate_dataset(&examples).unwrap();
        assert_eq!(report.total_examples, 2);
        assert_eq!(report.valid_examples, 1);
        assert!(report.has_errors());
    }

    #[test]
    fn test_preference_validation_identical() {
        let validator = DataValidator::with_defaults();
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Same")],
            vec![TrainingMessage::assistant("Same")],
        );
        let issues = validator.validate_preference(&pair);
        assert!(issues.iter().any(|i| i.message.contains("identical")));
    }

    #[test]
    fn test_preference_validation_empty_content() {
        let validator = DataValidator::with_defaults();
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        );
        let issues = validator.validate_preference(&pair);
        assert!(issues.iter().any(|i| i.message.contains("empty content")));
    }

    #[test]
    fn test_validate_preference_dataset() {
        let validator = DataValidator::with_defaults();
        let pairs = vec![
            PreferencePair::new(
                vec![TrainingMessage::user("Q")],
                vec![TrainingMessage::assistant("Good")],
                vec![TrainingMessage::assistant("Bad")],
            ),
            PreferencePair::new(
                vec![],
                vec![TrainingMessage::assistant("Good")],
                vec![TrainingMessage::assistant("Bad")],
            ),
        ];
        let report = validator.validate_preference_dataset(&pairs).unwrap();
        assert_eq!(report.total_examples, 2);
        assert_eq!(report.valid_examples, 1);
    }
}
