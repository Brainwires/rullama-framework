//! FFI-safe error types.

/// Error type exposed across the FFI boundary.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiAudioError {
    /// Provider API error (network, auth, rate limit, etc.).
    #[error("provider error: {message}")]
    Provider { message: String },

    /// Invalid provider handle.
    #[error("invalid handle: {message}")]
    InvalidHandle { message: String },

    /// Unsupported operation for this provider.
    #[error("unsupported: {message}")]
    Unsupported { message: String },

    /// Hardware audio error (device, capture, playback).
    #[error("hardware error: {message}")]
    Hardware { message: String },

    /// Unknown provider name.
    #[error("unknown provider: {message}")]
    UnknownProvider { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_display() {
        let e = FfiAudioError::Provider {
            message: "rate limited".to_string(),
        };
        assert_eq!(e.to_string(), "provider error: rate limited");
    }

    #[test]
    fn invalid_handle_display() {
        let e = FfiAudioError::InvalidHandle {
            message: "handle 99".to_string(),
        };
        assert_eq!(e.to_string(), "invalid handle: handle 99");
    }

    #[test]
    fn unsupported_display() {
        let e = FfiAudioError::Unsupported {
            message: "editing not supported".to_string(),
        };
        assert_eq!(e.to_string(), "unsupported: editing not supported");
    }

    #[test]
    fn hardware_display() {
        let e = FfiAudioError::Hardware {
            message: "no device found".to_string(),
        };
        assert_eq!(e.to_string(), "hardware error: no device found");
    }

    #[test]
    fn unknown_provider_display() {
        let e = FfiAudioError::UnknownProvider {
            message: "fake-provider".to_string(),
        };
        assert_eq!(e.to_string(), "unknown provider: fake-provider");
    }
}
