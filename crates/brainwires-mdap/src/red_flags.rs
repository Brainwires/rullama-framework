//! Red-Flagging System
//!
//! Implements Algorithm 3's red-flag validation from the MAKER paper.
//! Red-flagging strictly discards outputs that signal unreliability:
//! - Responses exceeding token limits (paper: ~750 tokens)
//! - Invalid format responses
//! - Self-correction patterns indicating model confusion
//! - Confused reasoning patterns
//!
//! The paper's approach is STRICT: discard on any red flag, no repair attempts.

use regex::Regex;

use super::voting::ResponseMetadata;

/// Red-flag configuration following the paper's strict approach
#[derive(Clone, Debug)]
pub struct RedFlagConfig {
    /// Maximum response tokens before flagging (paper: ~750)
    pub max_response_tokens: u32,
    /// Require exact format match (no repair attempts - paper's approach)
    pub require_exact_format: bool,
    /// Flag responses with self-correction patterns
    pub flag_self_correction: bool,
    /// Patterns indicating confused reasoning (to discard)
    pub confusion_patterns: Vec<String>,
    /// Minimum response length (to catch empty/truncated responses)
    pub min_response_length: u32,
    /// Maximum empty line ratio (indicates formatting issues)
    pub max_empty_line_ratio: f32,
}

impl Default for RedFlagConfig {
    fn default() -> Self {
        Self::strict()
    }
}

impl RedFlagConfig {
    /// Paper's strict configuration - the recommended approach
    pub fn strict() -> Self {
        Self {
            max_response_tokens: 750,
            require_exact_format: true,
            flag_self_correction: true,
            confusion_patterns: vec![
                "Wait,".to_string(),
                "Actually,".to_string(),
                "Let me reconsider".to_string(),
                "I made a mistake".to_string(),
                "On second thought".to_string(),
                "Hmm,".to_string(),
                "I think I".to_string(),
                "Let me correct".to_string(),
                "Sorry, I meant".to_string(),
                "That's not right".to_string(),
            ],
            min_response_length: 1,
            max_empty_line_ratio: 0.5,
        }
    }

    /// Relaxed configuration for less critical tasks
    pub fn relaxed() -> Self {
        Self {
            max_response_tokens: 1500,
            require_exact_format: false,
            flag_self_correction: false,
            confusion_patterns: vec![],
            min_response_length: 0,
            max_empty_line_ratio: 0.8,
        }
    }

    /// Custom configuration builder
    pub fn builder() -> RedFlagConfigBuilder {
        RedFlagConfigBuilder::default()
    }
}

/// Builder for RedFlagConfig
#[derive(Default)]
pub struct RedFlagConfigBuilder {
    config: RedFlagConfig,
}

impl RedFlagConfigBuilder {
    /// Set the maximum response token count.
    pub fn max_response_tokens(mut self, tokens: u32) -> Self {
        self.config.max_response_tokens = tokens;
        self
    }

    /// Set whether exact format matching is required.
    pub fn require_exact_format(mut self, require: bool) -> Self {
        self.config.require_exact_format = require;
        self
    }

    /// Set whether to flag self-correction patterns.
    pub fn flag_self_correction(mut self, flag: bool) -> Self {
        self.config.flag_self_correction = flag;
        self
    }

    /// Add a confusion pattern to detect.
    pub fn add_confusion_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.config.confusion_patterns.push(pattern.into());
        self
    }

    /// Set all confusion patterns.
    pub fn confusion_patterns(mut self, patterns: Vec<String>) -> Self {
        self.config.confusion_patterns = patterns;
        self
    }

    /// Set the minimum response length.
    pub fn min_response_length(mut self, length: u32) -> Self {
        self.config.min_response_length = length;
        self
    }

    /// Set the maximum empty line ratio.
    pub fn max_empty_line_ratio(mut self, ratio: f32) -> Self {
        self.config.max_empty_line_ratio = ratio;
        self
    }

    /// Build the red-flag configuration.
    pub fn build(self) -> RedFlagConfig {
        self.config
    }
}

/// Result of red-flag validation
#[derive(Clone, Debug)]
pub enum RedFlagResult {
    /// Response passed validation
    Valid,
    /// Response was flagged and should be discarded.
    Flagged {
        /// The reason for flagging.
        reason: RedFlagReason,
        /// Severity from 0.0 (minor) to 1.0 (critical)
        severity: f32,
    },
}

impl RedFlagResult {
    /// Check if the response is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, RedFlagResult::Valid)
    }

    /// Check if the response was flagged
    pub fn is_flagged(&self) -> bool {
        matches!(self, RedFlagResult::Flagged { .. })
    }
}

/// Reasons for red-flagging a response
#[derive(Clone, Debug)]
pub enum RedFlagReason {
    /// Response exceeded token limit (paper: error rate increases past ~700 tokens).
    ResponseTooLong {
        /// Actual token count.
        tokens: u32,
        /// Maximum allowed tokens.
        limit: u32,
    },
    /// Response was too short (possibly truncated or incomplete).
    ResponseTooShort {
        /// Actual response length.
        length: u32,
        /// Minimum required length.
        minimum: u32,
    },
    /// Response format didn't match expected format.
    InvalidFormat {
        /// Expected format.
        expected: String,
        /// Actual format received.
        got: String,
    },
    /// Self-correction detected (indicates model confusion).
    SelfCorrectionDetected {
        /// Detected pattern.
        pattern: String,
    },
    /// Confused reasoning pattern detected.
    ConfusedReasoning {
        /// Detected confusion pattern.
        pattern: String,
    },
    /// Failed to parse response.
    ParseError {
        /// Parse error message.
        message: String,
    },
    /// Empty response.
    EmptyResponse,
    /// Response has too many empty lines (formatting issue).
    TooManyEmptyLines {
        /// Actual empty line ratio.
        ratio: f32,
        /// Maximum allowed ratio.
        max: f32,
    },
    /// Invalid JSON structure.
    InvalidJson {
        /// Error message.
        message: String,
    },
    /// Missing required field in structured output.
    MissingField {
        /// Name of the missing field.
        field: String,
    },
    /// Response was truncated (finish_reason indicates truncation).
    Truncated {
        /// Truncation reason.
        reason: String,
    },
}

impl std::fmt::Display for RedFlagReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RedFlagReason::ResponseTooLong { tokens, limit } => {
                write!(f, "Response too long: {} tokens > {} limit", tokens, limit)
            }
            RedFlagReason::ResponseTooShort { length, minimum } => {
                write!(
                    f,
                    "Response too short: {} chars < {} minimum",
                    length, minimum
                )
            }
            RedFlagReason::InvalidFormat { expected, got } => {
                write!(f, "Invalid format: expected {}, got {}", expected, got)
            }
            RedFlagReason::SelfCorrectionDetected { pattern } => {
                write!(f, "Self-correction detected: '{}'", pattern)
            }
            RedFlagReason::ConfusedReasoning { pattern } => {
                write!(f, "Confused reasoning: '{}'", pattern)
            }
            RedFlagReason::ParseError { message } => {
                write!(f, "Parse error: {}", message)
            }
            RedFlagReason::EmptyResponse => write!(f, "Empty response"),
            RedFlagReason::TooManyEmptyLines { ratio, max } => {
                write!(
                    f,
                    "Too many empty lines: {:.1}% > {:.1}% max",
                    ratio * 100.0,
                    max * 100.0
                )
            }
            RedFlagReason::InvalidJson { message } => {
                write!(f, "Invalid JSON: {}", message)
            }
            RedFlagReason::MissingField { field } => {
                write!(f, "Missing required field: {}", field)
            }
            RedFlagReason::Truncated { reason } => {
                write!(f, "Response truncated: {}", reason)
            }
        }
    }
}

/// Trait for red-flag validation
pub trait RedFlagValidator: Send + Sync {
    /// Validate a response against red-flag criteria
    fn validate(&self, response: &str, metadata: &ResponseMetadata) -> RedFlagResult;
}

/// Standard red-flag validator implementing the paper's approach
pub struct StandardRedFlagValidator {
    config: RedFlagConfig,
    expected_format: Option<OutputFormat>,
    confusion_regexes: Vec<Regex>,
}

impl StandardRedFlagValidator {
    /// Create a new validator with the given config
    pub fn new(config: RedFlagConfig, expected_format: Option<OutputFormat>) -> Self {
        // Pre-compile confusion pattern regexes for efficiency
        let confusion_regexes = config
            .confusion_patterns
            .iter()
            .filter_map(|p| {
                // Escape the pattern for regex matching
                Regex::new(&regex::escape(p)).ok()
            })
            .collect();

        Self {
            config,
            expected_format,
            confusion_regexes,
        }
    }

    /// Create a strict validator (paper's recommended approach)
    pub fn strict() -> Self {
        Self::new(RedFlagConfig::strict(), None)
    }

    /// Create a validator with expected format
    pub fn with_format(format: OutputFormat) -> Self {
        Self::new(RedFlagConfig::strict(), Some(format))
    }

    /// Set the expected output format
    pub fn set_expected_format(&mut self, format: Option<OutputFormat>) {
        self.expected_format = format;
    }

    /// Check response length constraints
    fn check_length(&self, response: &str, metadata: &ResponseMetadata) -> Option<RedFlagResult> {
        // Check empty
        if response.trim().is_empty() {
            return Some(RedFlagResult::Flagged {
                reason: RedFlagReason::EmptyResponse,
                severity: 1.0,
            });
        }

        // Check minimum length
        if (response.len() as u32) < self.config.min_response_length {
            return Some(RedFlagResult::Flagged {
                reason: RedFlagReason::ResponseTooShort {
                    length: response.len() as u32,
                    minimum: self.config.min_response_length,
                },
                severity: 0.9,
            });
        }

        // Check maximum tokens (paper: error rate increases past ~700 tokens)
        if metadata.token_count > self.config.max_response_tokens {
            return Some(RedFlagResult::Flagged {
                reason: RedFlagReason::ResponseTooLong {
                    tokens: metadata.token_count,
                    limit: self.config.max_response_tokens,
                },
                severity: 0.8,
            });
        }

        None
    }

    /// Check for self-correction patterns (indicates model confusion)
    fn check_self_correction(&self, response: &str) -> Option<RedFlagResult> {
        if !self.config.flag_self_correction {
            return None;
        }

        for (regex, pattern) in self
            .confusion_regexes
            .iter()
            .zip(&self.config.confusion_patterns)
        {
            if regex.is_match(response) {
                return Some(RedFlagResult::Flagged {
                    reason: RedFlagReason::SelfCorrectionDetected {
                        pattern: pattern.clone(),
                    },
                    severity: 0.7,
                });
            }
        }

        None
    }

    /// Check format validity
    fn check_format(&self, response: &str) -> Option<RedFlagResult> {
        if !self.config.require_exact_format {
            return None;
        }

        if let Some(ref format) = self.expected_format
            && !format.matches(response)
        {
            return Some(RedFlagResult::Flagged {
                reason: RedFlagReason::InvalidFormat {
                    expected: format.description(),
                    got: self.extract_format_sample(response),
                },
                severity: 0.9,
            });
        }

        None
    }

    /// Check for truncation
    fn check_truncation(&self, metadata: &ResponseMetadata) -> Option<RedFlagResult> {
        if let Some(ref reason) = metadata.finish_reason {
            let reason_lower = reason.to_lowercase();
            if reason_lower.contains("length") || reason_lower.contains("max_tokens") {
                return Some(RedFlagResult::Flagged {
                    reason: RedFlagReason::Truncated {
                        reason: reason.clone(),
                    },
                    severity: 0.85,
                });
            }
        }
        None
    }

    /// Check empty line ratio
    fn check_empty_lines(&self, response: &str) -> Option<RedFlagResult> {
        let lines: Vec<&str> = response.lines().collect();
        if lines.is_empty() {
            return None;
        }

        let empty_count = lines.iter().filter(|l| l.trim().is_empty()).count();
        let ratio = empty_count as f32 / lines.len() as f32;

        if ratio > self.config.max_empty_line_ratio {
            return Some(RedFlagResult::Flagged {
                reason: RedFlagReason::TooManyEmptyLines {
                    ratio,
                    max: self.config.max_empty_line_ratio,
                },
                severity: 0.6,
            });
        }

        None
    }

    /// Extract a sample of the response format for error messages
    fn extract_format_sample(&self, response: &str) -> String {
        let trimmed = response.trim();
        if trimmed.len() <= 50 {
            trimmed.to_string()
        } else {
            format!("{}...", &trimmed[..50])
        }
    }
}

impl RedFlagValidator for StandardRedFlagValidator {
    fn validate(&self, response: &str, metadata: &ResponseMetadata) -> RedFlagResult {
        // Check in order of severity/importance:

        // 1. Check length constraints (including empty)
        if let Some(result) = self.check_length(response, metadata) {
            return result;
        }

        // 2. Check for truncation
        if let Some(result) = self.check_truncation(metadata) {
            return result;
        }

        // 3. Check format validity (strict - no repair per paper)
        if let Some(result) = self.check_format(response) {
            return result;
        }

        // 4. Check for self-correction patterns (indicates confusion)
        if let Some(result) = self.check_self_correction(response) {
            return result;
        }

        // 5. Check empty line ratio
        if let Some(result) = self.check_empty_lines(response) {
            return result;
        }

        RedFlagResult::Valid
    }
}

/// Expected output format for validation
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OutputFormat {
    /// Must match exact string
    Exact(String),
    /// Must match regex pattern
    Pattern(String),
    /// Must parse as valid JSON
    Json,
    /// Must parse as valid JSON with specific structure
    JsonWithFields(Vec<String>),
    /// Must contain specific markers
    Markers {
        /// Required start marker.
        start: String,
        /// Required end marker.
        end: String,
    },
    /// Must be one of specific values
    OneOf(Vec<String>),
    /// Custom validator function (stored as description)
    Custom {
        /// Human-readable description of the custom validator.
        description: String,
        /// Identifier for the custom validator function.
        validator_id: String,
    },
}

impl OutputFormat {
    /// Check if the response matches this format
    pub fn matches(&self, response: &str) -> bool {
        let trimmed = response.trim();
        match self {
            OutputFormat::Exact(s) => trimmed == s.trim(),
            OutputFormat::Pattern(pattern) => Regex::new(pattern)
                .map(|re| re.is_match(trimmed))
                .unwrap_or(false),
            OutputFormat::Json => serde_json::from_str::<serde_json::Value>(trimmed).is_ok(),
            OutputFormat::JsonWithFields(fields) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
                    && let Some(obj) = value.as_object()
                {
                    return fields.iter().all(|f| obj.contains_key(f));
                }
                false
            }
            OutputFormat::Markers { start, end } => {
                trimmed.contains(start) && trimmed.contains(end)
            }
            OutputFormat::OneOf(options) => options.iter().any(|o| trimmed == o.trim()),
            OutputFormat::Custom { .. } => {
                // Custom validators need external validation logic
                // By default, accept if we can't validate
                true
            }
        }
    }

    /// Get a description of this format for error messages
    pub fn description(&self) -> String {
        match self {
            OutputFormat::Exact(s) => format!("exact: '{}'", s),
            OutputFormat::Pattern(p) => format!("pattern: {}", p),
            OutputFormat::Json => "valid JSON".to_string(),
            OutputFormat::JsonWithFields(fields) => {
                format!("JSON with fields: {}", fields.join(", "))
            }
            OutputFormat::Markers { start, end } => format!("markers: {}...{}", start, end),
            OutputFormat::OneOf(options) => format!("one of: {}", options.join(", ")),
            OutputFormat::Custom { description, .. } => description.clone(),
        }
    }
}

/// Always-accept validator for testing or when red-flagging is disabled
pub struct AcceptAllValidator;

impl RedFlagValidator for AcceptAllValidator {
    fn validate(&self, _response: &str, _metadata: &ResponseMetadata) -> RedFlagResult {
        RedFlagResult::Valid
    }
}

/// Validator that combines multiple validators
pub struct CompositeValidator {
    validators: Vec<Box<dyn RedFlagValidator>>,
}

impl CompositeValidator {
    /// Create a new empty composite validator.
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to this composite.
    pub fn with_validator(mut self, validator: Box<dyn RedFlagValidator>) -> Self {
        self.validators.push(validator);
        self
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl RedFlagValidator for CompositeValidator {
    fn validate(&self, response: &str, metadata: &ResponseMetadata) -> RedFlagResult {
        for validator in &self.validators {
            let result = validator.validate(response, metadata);
            if result.is_flagged() {
                return result;
            }
        }
        RedFlagResult::Valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata(tokens: u32) -> ResponseMetadata {
        ResponseMetadata {
            token_count: tokens,
            response_time_ms: 100,
            format_valid: true,
            finish_reason: None,
            model: None,
        }
    }

    #[test]
    fn test_valid_response() {
        let validator = StandardRedFlagValidator::strict();
        let result = validator.validate("This is a valid response.", &make_metadata(50));
        assert!(result.is_valid());
    }

    #[test]
    fn test_empty_response() {
        let validator = StandardRedFlagValidator::strict();
        let result = validator.validate("", &make_metadata(0));
        assert!(result.is_flagged());
        if let RedFlagResult::Flagged { reason, .. } = result {
            assert!(matches!(reason, RedFlagReason::EmptyResponse));
        }
    }

    #[test]
    fn test_response_too_long() {
        let validator = StandardRedFlagValidator::strict();
        let result = validator.validate("Some response", &make_metadata(800)); // > 750
        assert!(result.is_flagged());
        if let RedFlagResult::Flagged { reason, .. } = result {
            assert!(matches!(reason, RedFlagReason::ResponseTooLong { .. }));
        }
    }

    #[test]
    fn test_self_correction_detected() {
        let validator = StandardRedFlagValidator::strict();
        let result = validator.validate(
            "Wait, I think I made an error. Let me reconsider.",
            &make_metadata(50),
        );
        assert!(result.is_flagged());
        if let RedFlagResult::Flagged { reason, .. } = result {
            assert!(matches!(
                reason,
                RedFlagReason::SelfCorrectionDetected { .. }
            ));
        }
    }

    #[test]
    fn test_confused_reasoning() {
        let validator = StandardRedFlagValidator::strict();
        let result = validator.validate(
            "Actually, that's not right. On second thought...",
            &make_metadata(50),
        );
        assert!(result.is_flagged());
    }

    #[test]
    fn test_format_validation_exact() {
        let validator =
            StandardRedFlagValidator::with_format(OutputFormat::Exact("hello".to_string()));

        assert!(validator.validate("hello", &make_metadata(10)).is_valid());
        assert!(
            validator
                .validate("  hello  ", &make_metadata(10))
                .is_valid()
        ); // Trimmed
        assert!(validator.validate("world", &make_metadata(10)).is_flagged());
    }

    #[test]
    fn test_format_validation_json() {
        let validator = StandardRedFlagValidator::with_format(OutputFormat::Json);

        assert!(
            validator
                .validate(r#"{"key": "value"}"#, &make_metadata(20))
                .is_valid()
        );
        assert!(
            validator
                .validate("not json", &make_metadata(10))
                .is_flagged()
        );
    }

    #[test]
    fn test_format_validation_json_with_fields() {
        let validator = StandardRedFlagValidator::with_format(OutputFormat::JsonWithFields(vec![
            "name".to_string(),
            "value".to_string(),
        ]));

        assert!(
            validator
                .validate(r#"{"name": "test", "value": 42}"#, &make_metadata(30))
                .is_valid()
        );
        assert!(
            validator
                .validate(r#"{"name": "test"}"#, &make_metadata(20))
                .is_flagged()
        ); // Missing "value"
    }

    #[test]
    fn test_format_validation_markers() {
        let validator = StandardRedFlagValidator::with_format(OutputFormat::Markers {
            start: "```".to_string(),
            end: "```".to_string(),
        });

        assert!(
            validator
                .validate("```code here```", &make_metadata(20))
                .is_valid()
        );
        assert!(
            validator
                .validate("no markers", &make_metadata(10))
                .is_flagged()
        );
    }

    #[test]
    fn test_format_validation_one_of() {
        let validator = StandardRedFlagValidator::with_format(OutputFormat::OneOf(vec![
            "yes".to_string(),
            "no".to_string(),
            "maybe".to_string(),
        ]));

        assert!(validator.validate("yes", &make_metadata(5)).is_valid());
        assert!(validator.validate("no", &make_metadata(5)).is_valid());
        assert!(
            validator
                .validate("perhaps", &make_metadata(10))
                .is_flagged()
        );
    }

    #[test]
    fn test_truncation_detection() {
        let validator = StandardRedFlagValidator::strict();
        let mut metadata = make_metadata(50);
        metadata.finish_reason = Some("length".to_string());

        let result = validator.validate("Truncated response", &metadata);
        assert!(result.is_flagged());
        if let RedFlagResult::Flagged { reason, .. } = result {
            assert!(matches!(reason, RedFlagReason::Truncated { .. }));
        }
    }

    #[test]
    fn test_relaxed_config() {
        let config = RedFlagConfig::relaxed();
        let validator = StandardRedFlagValidator::new(config, None);

        // Self-correction shouldn't be flagged in relaxed mode
        let result = validator.validate("Wait, let me reconsider this.", &make_metadata(50));
        assert!(result.is_valid());
    }

    #[test]
    fn test_config_builder() {
        let config = RedFlagConfig::builder()
            .max_response_tokens(500)
            .flag_self_correction(false)
            .add_confusion_pattern("Oops")
            .build();

        assert_eq!(config.max_response_tokens, 500);
        assert!(!config.flag_self_correction);
        assert!(config.confusion_patterns.contains(&"Oops".to_string()));
    }

    #[test]
    fn test_accept_all_validator() {
        let validator = AcceptAllValidator;

        assert!(validator.validate("", &make_metadata(0)).is_valid());
        assert!(
            validator
                .validate("anything", &make_metadata(10000))
                .is_valid()
        );
    }

    #[test]
    fn test_composite_validator() {
        let validator =
            CompositeValidator::new().with_validator(Box::new(StandardRedFlagValidator::strict()));

        assert!(validator.validate("valid", &make_metadata(10)).is_valid());
        assert!(validator.validate("", &make_metadata(0)).is_flagged());
    }

    #[test]
    fn test_red_flag_reason_display() {
        let reason = RedFlagReason::ResponseTooLong {
            tokens: 800,
            limit: 750,
        };
        assert_eq!(
            reason.to_string(),
            "Response too long: 800 tokens > 750 limit"
        );

        let reason = RedFlagReason::SelfCorrectionDetected {
            pattern: "Wait,".to_string(),
        };
        assert!(reason.to_string().contains("Wait,"));
    }

    #[test]
    fn test_empty_line_ratio() {
        let config = RedFlagConfig::builder()
            .max_empty_line_ratio(0.3)
            .flag_self_correction(false)
            .build();
        let validator = StandardRedFlagValidator::new(config, None);

        // Response with too many empty lines
        let response = "line1\n\n\n\nline2";
        let result = validator.validate(response, &make_metadata(10));
        assert!(result.is_flagged());
    }
}
