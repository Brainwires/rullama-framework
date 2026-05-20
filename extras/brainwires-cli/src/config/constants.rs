/// Backend URL constants
pub const DEFAULT_BACKEND_URL: &str = "https://brainwires.studio";
pub const DEV_BACKEND_URL: &str = "https://dev.brainwires.net";
pub const API_CLI_AUTH_ENDPOINT: &str = "/api/cli/auth";
pub const API_CLI_KEYS_ENDPOINT: &str = "/api/cli/keys";
pub const API_MCP_EXECUTE_ENDPOINT: &str = "/api/mcp/execute";
pub const API_MCP_TOOLS_ENDPOINT: &str = "/api/mcp/tools";

/// API key format validation regex pattern
pub const API_KEY_PATTERN: &str = r"^bw_(prod|dev|test)_[a-z0-9]{32}$";

/// Determine backend URL from API key prefix
pub fn get_backend_from_api_key(api_key: &str) -> &'static str {
    if api_key.starts_with("bw_dev_") {
        DEV_BACKEND_URL
    } else {
        DEFAULT_BACKEND_URL
    }
}

/// Session expiration (7 days in seconds)
///
/// **DEPRECATED**: Sessions no longer expire. API keys persist until
/// explicitly logged out. Kept for backward compatibility only.
#[deprecated(note = "Sessions no longer expire. API keys persist until explicitly logged out.")]
pub const SESSION_EXPIRATION_SECS: i64 = 7 * 24 * 60 * 60;

/// Maximum context length (100k tokens)
pub const MAX_CONTEXT_TOKENS: usize = 100_000;

/// Maximum file size to inject directly into context (~2000 tokens)
/// Files larger than this will be chunked and only relevant portions injected
pub const MAX_DIRECT_FILE_CHARS: usize = 8_000;

/// Target chunk size for large files (characters)
pub const LARGE_FILE_CHUNK_SIZE: usize = 1_500;

/// Maximum chunks to retrieve for a large file query
pub const MAX_FILE_CHUNKS: usize = 5;

/// Context compaction threshold (80k tokens)
///
/// **DEPRECATED**: Manual compaction is deprecated. The system now uses
/// automatic context management via TieredMemory, MessageStore, and
/// ContextBuilder. This constant is kept for backward compatibility only.
#[deprecated(note = "Manual compaction is deprecated. Use automatic context management instead.")]
pub const COMPACTION_THRESHOLD_TOKENS: usize = 80_000;

/// Maximum worker iterations
pub const MAX_WORKER_ITERATIONS: u32 = 15;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_urls_format() {
        assert!(DEFAULT_BACKEND_URL.starts_with("https://"));
        assert!(DEV_BACKEND_URL.starts_with("https://"));
        assert!(!DEFAULT_BACKEND_URL.ends_with('/'));
        assert!(!DEV_BACKEND_URL.ends_with('/'));
    }

    #[test]
    fn test_api_endpoints_format() {
        assert!(API_CLI_AUTH_ENDPOINT.starts_with('/'));
        assert!(API_CLI_KEYS_ENDPOINT.starts_with('/'));
        assert!(API_MCP_EXECUTE_ENDPOINT.starts_with('/'));
        assert!(API_MCP_TOOLS_ENDPOINT.starts_with('/'));
    }

    #[test]
    fn test_api_key_pattern_valid() {
        // Valid API keys
        assert!(
            regex::Regex::new(API_KEY_PATTERN)
                .unwrap()
                .is_match("bw_prod_12345678901234567890123456789012")
        );
        assert!(
            regex::Regex::new(API_KEY_PATTERN)
                .unwrap()
                .is_match("bw_dev_abcdefghijklmnopqrstuvwxyz123456")
        );
        assert!(
            regex::Regex::new(API_KEY_PATTERN)
                .unwrap()
                .is_match("bw_test_00000000000000000000000000000000")
        );
    }

    #[test]
    fn test_api_key_pattern_invalid() {
        let pattern = regex::Regex::new(API_KEY_PATTERN).unwrap();
        // Invalid patterns
        assert!(!pattern.is_match("invalid"));
        assert!(!pattern.is_match("bw_invalid_12345678901234567890123456789012"));
        assert!(!pattern.is_match("bw_prod_short"));
        assert!(!pattern.is_match("bw_prod_1234567890123456789012345678901")); // 31 chars
        assert!(!pattern.is_match("bw_prod_123456789012345678901234567890123")); // 33 chars
    }

    #[test]
    #[allow(deprecated)]
    fn test_session_expiration_constant() {
        // Constant is deprecated but kept for compatibility
        // 7 days in seconds
        assert_eq!(SESSION_EXPIRATION_SECS, 604_800);
        assert_eq!(SESSION_EXPIRATION_SECS, 7 * 24 * 60 * 60);
    }

    #[test]
    fn test_context_token_limits() {
        assert_eq!(MAX_CONTEXT_TOKENS, 100_000);
        #[allow(deprecated)]
        {
            assert_eq!(COMPACTION_THRESHOLD_TOKENS, 80_000);
            const _: () = assert!(COMPACTION_THRESHOLD_TOKENS < MAX_CONTEXT_TOKENS);
            // Threshold should be 80% of max
            assert_eq!(COMPACTION_THRESHOLD_TOKENS, (MAX_CONTEXT_TOKENS * 80) / 100);
        }
    }

    #[test]
    fn test_worker_iterations() {
        assert_eq!(MAX_WORKER_ITERATIONS, 15);
        const _: () = assert!(MAX_WORKER_ITERATIONS > 0);
    }

    #[test]
    fn test_get_backend_from_api_key_dev() {
        assert_eq!(
            get_backend_from_api_key("bw_dev_12345678901234567890123456789012"),
            DEV_BACKEND_URL
        );
    }

    #[test]
    fn test_get_backend_from_api_key_prod() {
        assert_eq!(
            get_backend_from_api_key("bw_prod_12345678901234567890123456789012"),
            DEFAULT_BACKEND_URL
        );
    }

    #[test]
    fn test_get_backend_from_api_key_test() {
        assert_eq!(
            get_backend_from_api_key("bw_test_12345678901234567890123456789012"),
            DEFAULT_BACKEND_URL
        );
    }

    #[test]
    fn test_get_backend_from_api_key_plain_bw() {
        // Keys starting with just "bw_" (not bw_dev_) should use production
        assert_eq!(
            get_backend_from_api_key("bw_12345678901234567890123456789012"),
            DEFAULT_BACKEND_URL
        );
    }
}
