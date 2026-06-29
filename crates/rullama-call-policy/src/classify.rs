//! Error classification for retry decisions.
//!
//! LLM providers in this workspace surface errors as `anyhow::Error`. We don't
//! have a typed error envelope to match on, so we classify by string-matching
//! the error message against common transient-failure signatures. This is
//! pragmatic: providers return text like "status: 429 Too Many Requests" or
//! "connection reset", which is stable enough to key on.

use std::time::Duration;

/// Coarse error classification used by retry and circuit-breaker logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// HTTP 429 / rate-limited. Retryable with backoff. `Retry-After` hint may
    /// be embedded in the error string.
    RateLimited,
    /// Transient network / IO error (connection reset, DNS, TLS). Retryable.
    Network,
    /// HTTP 5xx server error. Retryable.
    Server5xx,
    /// HTTP 4xx other than 429 (400/401/403/404). Not retryable.
    Client4xx,
    /// Authentication failure. Not retryable.
    Auth,
    /// Unknown / unclassified. Treated as non-retryable by default.
    Unknown,
}

impl ErrorClass {
    /// Whether errors in this class should be retried.
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::RateLimited | Self::Network | Self::Server5xx)
    }
}

/// Classify an `anyhow::Error` by string inspection.
pub fn classify_error(err: &anyhow::Error) -> ErrorClass {
    let s = format!("{err:#}").to_lowercase();

    if s.contains("429") || s.contains("rate limit") || s.contains("too many requests") {
        return ErrorClass::RateLimited;
    }
    if s.contains(" 401") || s.contains("unauthorized") || s.contains("invalid api key") {
        return ErrorClass::Auth;
    }
    if s.contains(" 500")
        || s.contains(" 502")
        || s.contains(" 503")
        || s.contains(" 504")
        || s.contains("internal server error")
        || s.contains("bad gateway")
        || s.contains("service unavailable")
        || s.contains("gateway timeout")
    {
        return ErrorClass::Server5xx;
    }
    if s.contains("connection reset")
        || s.contains("connection refused")
        || s.contains("connection closed")
        || s.contains("broken pipe")
        || s.contains("timed out")
        || s.contains("timeout")
        || s.contains("dns")
        || s.contains("tls")
        || s.contains("hyper")
        || s.contains("reqwest")
        || s.contains("io error")
    {
        return ErrorClass::Network;
    }
    if s.contains(" 400")
        || s.contains(" 403")
        || s.contains(" 404")
        || s.contains("bad request")
        || s.contains("forbidden")
        || s.contains("not found")
    {
        return ErrorClass::Client4xx;
    }

    ErrorClass::Unknown
}

/// Parse a `Retry-After` hint out of an error string, if present.
///
/// Looks for patterns like `retry-after: 30`, `retry after 30s`, or
/// `retry-after: 30 seconds`. Returns `None` if no hint can be extracted.
pub fn parse_retry_after(err: &anyhow::Error) -> Option<Duration> {
    let s = format!("{err:#}").to_lowercase();
    let idx = s.find("retry-after").or_else(|| s.find("retry after"))?;
    let tail = &s[idx..];

    let mut num = String::new();
    let mut seen_digit = false;
    for c in tail.chars() {
        if c.is_ascii_digit() {
            num.push(c);
            seen_digit = true;
        } else if seen_digit {
            break;
        }
    }

    let secs: u64 = num.parse().ok()?;
    if secs == 0 || secs > 3600 {
        return None;
    }
    Some(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_429() {
        let err = anyhow::anyhow!("status 429 Too Many Requests");
        assert_eq!(classify_error(&err), ErrorClass::RateLimited);
        assert!(classify_error(&err).is_retryable());
    }

    #[test]
    fn classify_5xx() {
        let err = anyhow::anyhow!("status 503 Service Unavailable");
        assert_eq!(classify_error(&err), ErrorClass::Server5xx);
    }

    #[test]
    fn classify_network() {
        let err = anyhow::anyhow!("connection reset by peer");
        assert_eq!(classify_error(&err), ErrorClass::Network);
    }

    #[test]
    fn classify_auth_not_retryable() {
        let err = anyhow::anyhow!("401 Unauthorized: invalid API key");
        assert_eq!(classify_error(&err), ErrorClass::Auth);
        assert!(!classify_error(&err).is_retryable());
    }

    #[test]
    fn parse_retry_after_seconds() {
        let err = anyhow::anyhow!("429 Too Many Requests; retry-after: 42");
        assert_eq!(parse_retry_after(&err), Some(Duration::from_secs(42)));
    }

    #[test]
    fn parse_retry_after_absent() {
        let err = anyhow::anyhow!("generic failure");
        assert_eq!(parse_retry_after(&err), None);
    }

    #[test]
    fn parse_retry_after_rejects_absurd() {
        let err = anyhow::anyhow!("retry-after: 99999");
        assert_eq!(parse_retry_after(&err), None);
    }
}
