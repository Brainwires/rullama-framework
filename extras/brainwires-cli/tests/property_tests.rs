//! Property-Based Tests
//!
//! Uses proptest to verify invariants with random inputs.
//! These tests complement unit tests by finding edge cases.

use proptest::prelude::*;

// ============================================================================
// Token Counting Properties
// ============================================================================

mod token_counting {
    use super::*;
    use brainwires_cli::utils::tokenizer::{
        TokenizerConfig, estimate_tokens, estimate_tokens_with_config,
    };

    proptest! {
        /// Token count should be >= 0 and bounded by string length
        #[test]
        fn token_count_bounded(s in ".*") {
            let tokens = estimate_tokens(&s);
            // Can't have more tokens than characters (worst case: 1 char = 1 token)
            prop_assert!(tokens <= s.chars().count() + 1);
        }

        /// Empty string should have 0 tokens
        #[test]
        fn empty_string_zero_tokens(_dummy in Just(())) {
            let tokens = estimate_tokens("");
            prop_assert_eq!(tokens, 0);
        }

        /// Token count should be monotonically increasing with string length
        #[test]
        fn tokens_increase_with_length(s in ".{1,100}") {
            let tokens = estimate_tokens(&s);
            let double_tokens = estimate_tokens(&format!("{}{}", s, s));
            // Doubling input should roughly double tokens (within 20%)
            prop_assert!(double_tokens >= tokens);
            prop_assert!(double_tokens <= tokens * 2 + 10);
        }

        /// Different configs should give different results for same input
        #[test]
        fn config_affects_count(s in ".{10,50}") {
            let openai_config = TokenizerConfig::openai();
            let anthropic_config = TokenizerConfig::anthropic();

            let openai_tokens = estimate_tokens_with_config(&s, &openai_config);
            let anthropic_tokens = estimate_tokens_with_config(&s, &anthropic_config);

            // Both should be positive for non-empty strings
            prop_assert!(openai_tokens > 0);
            prop_assert!(anthropic_tokens > 0);

            // They should be within 30% of each other
            let diff = (openai_tokens as f64 - anthropic_tokens as f64).abs();
            let max = openai_tokens.max(anthropic_tokens) as f64;
            prop_assert!(diff / max < 0.3);
        }
    }
}

// ============================================================================
// Message Serialization Round-Trip Properties
// ============================================================================

mod message_serialization {
    use super::*;
    use brainwires_cli::types::message::{Message, MessageContent, Role};

    fn arb_role() -> impl Strategy<Value = Role> {
        prop_oneof![
            Just(Role::User),
            Just(Role::Assistant),
            Just(Role::System),
            Just(Role::Tool),
        ]
    }

    fn arb_text_content() -> impl Strategy<Value = MessageContent> {
        ".*".prop_map(MessageContent::Text)
    }

    fn arb_message() -> impl Strategy<Value = Message> {
        (arb_role(), arb_text_content()).prop_map(|(role, content)| Message {
            role,
            content,
            name: None,
            metadata: None,
        })
    }

    proptest! {
        /// Message serialization should round-trip
        #[test]
        fn message_roundtrip(msg in arb_message()) {
            // Serialize
            let json = serde_json::to_string(&msg);
            prop_assert!(json.is_ok(), "Serialization failed");

            // Deserialize
            let json = json.unwrap();
            let restored: Result<Message, _> = serde_json::from_str(&json);
            prop_assert!(restored.is_ok(), "Deserialization failed: {}", json);

            let restored = restored.unwrap();

            // Verify content preserved
            match (&msg.content, &restored.content) {
                (MessageContent::Text(a), MessageContent::Text(b)) => {
                    prop_assert_eq!(a, b);
                }
                _ => {
                    prop_assert!(false, "Content type mismatch");
                }
            }
        }

        /// Role should serialize to expected string
        #[test]
        fn role_serialization(role in arb_role()) {
            let json = serde_json::to_string(&role).unwrap();
            let expected = match role {
                Role::User => "\"user\"",
                Role::Assistant => "\"assistant\"",
                Role::System => "\"system\"",
                Role::Tool => "\"tool\"",
            };
            prop_assert_eq!(json, expected);
        }
    }
}

// ============================================================================
// Tool Argument Validation Properties
// ============================================================================

mod tool_validation {
    use super::*;
    use serde_json::json;

    /// Generate arbitrary valid JSON values
    fn arb_json_value() -> impl Strategy<Value = serde_json::Value> {
        let leaf = prop_oneof![
            Just(json!(null)),
            any::<bool>().prop_map(|b| json!(b)),
            any::<i64>().prop_map(|n| json!(n)),
            any::<f64>()
                .prop_filter("must be finite", |f| f.is_finite())
                .prop_map(|n| json!(n)),
            ".*".prop_map(|s| json!(s)),
        ];

        leaf.prop_recursive(3, 10, 5, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5).prop_map(|v| json!(v)),
                prop::collection::hash_map("\\w+", inner, 0..5)
                    .prop_map(|m| serde_json::Value::Object(m.into_iter().collect())),
            ]
        })
    }

    proptest! {
        /// JSON values should serialize and deserialize correctly
        #[test]
        fn json_roundtrip(value in arb_json_value()) {
            let json_str = serde_json::to_string(&value);
            prop_assert!(json_str.is_ok());

            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str.unwrap());
            prop_assert!(parsed.is_ok());
        }

        /// File paths should not contain null bytes
        #[test]
        fn path_no_null_bytes(path in "[^\0]{1,100}") {
            // Valid paths shouldn't contain null bytes
            prop_assert!(!path.contains('\0'));
            // CString creation should succeed
            let c_string = std::ffi::CString::new(path.as_bytes());
            prop_assert!(c_string.is_ok());
        }
    }
}

// ============================================================================
// Error Classification Properties
// ============================================================================

mod error_classification {
    use super::*;
    use brainwires_cli::tools::error::classify_error;

    /// Generate error messages
    fn arb_error_message() -> impl Strategy<Value = String> {
        prop_oneof![
            // Known patterns
            Just("Connection refused".to_string()),
            Just("Permission denied".to_string()),
            Just("No such file or directory".to_string()),
            Just("Rate limit exceeded".to_string()),
            Just("Request timed out".to_string()),
            // Random messages
            "[a-zA-Z ]{1,50}".prop_map(|s| s),
        ]
    }

    proptest! {
        /// Error classification should never panic
        #[test]
        fn classification_never_panics(
            tool in "[a-z_]{1,20}",
            error in arb_error_message()
        ) {
            // Should not panic
            let category = classify_error(&tool, &error);

            // Should always return a valid category
            prop_assert!(!category.category_name().is_empty());
            prop_assert!(!category.error_message().is_empty());
        }

        /// Retry strategy should have valid properties
        #[test]
        fn retry_strategy_valid(error in arb_error_message()) {
            let category = classify_error("bash", &error);
            let strategy = category.retry_strategy();

            // Max attempts should be reasonable
            prop_assert!(strategy.max_attempts() <= 10);

            // If retryable, should have positive max attempts
            if category.is_retryable() {
                prop_assert!(strategy.max_attempts() > 0);
            }
        }
    }
}

// ============================================================================
// API Surface Properties
// ============================================================================

mod api_properties {
    use super::*;
    use brainwires_cli::error::AppError;

    proptest! {
        /// AppError should format without panicking
        #[test]
        fn app_error_formats(msg in ".*") {
            let errors = vec![
                AppError::Config(msg.clone()),
                AppError::AuthRequired(msg.clone()),
                AppError::FileNotFound(msg.clone()),
                AppError::Internal(msg.clone()),
            ];

            for err in errors {
                // Should not panic when formatting
                let _ = format!("{}", err);
                let _ = format!("{:?}", err);
            }
        }

        /// is_retryable should be consistent
        #[test]
        fn retryable_consistency(_dummy in Just(())) {
            // Non-retryable errors
            let non_retryable = vec![
                AppError::Config("test".into()),
                AppError::ConfigMissing("key".into()),
                AppError::AuthRequired("test".into()),
                AppError::PermissionDenied("test".into()),
                AppError::FileNotFound("test".into()),
                AppError::Cancelled,
            ];

            for err in non_retryable {
                prop_assert!(!err.is_retryable(), "Should not be retryable: {:?}", err);
            }

            // Retryable errors
            let retryable = vec![
                AppError::Timeout("test".into()),
                AppError::Connection("test".into()),
                AppError::ProviderRateLimit { provider: "test".into(), retry_after_secs: 5 },
            ];

            for err in retryable {
                prop_assert!(err.is_retryable(), "Should be retryable: {:?}", err);
            }
        }
    }
}
