use thiserror::Error;

/// Errors that can occur during audio operations.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Audio device not found or unavailable.
    #[error("device error: {0}")]
    Device(String),

    /// Audio capture failed.
    #[error("capture error: {0}")]
    Capture(String),

    /// Audio playback failed.
    #[error("playback error: {0}")]
    Playback(String),

    /// Speech-to-text transcription failed.
    #[error("transcription error: {0}")]
    Transcription(String),

    /// Text-to-speech synthesis failed.
    #[error("synthesis error: {0}")]
    Synthesis(String),

    /// Audio format conversion failed.
    #[error("format error: {0}")]
    Format(String),

    /// API communication error.
    #[error("api error: {0}")]
    Api(String),

    /// Audio stream was interrupted or closed.
    #[error("stream closed: {0}")]
    StreamClosed(String),

    /// Unsupported configuration requested.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// IO error.
    #[error("io error: {source}")]
    Io {
        /// The underlying IO error.
        #[from]
        source: std::io::Error,
    },
}

/// Result alias for audio operations.
pub type AudioResult<T> = Result<T, AudioError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_device_error() {
        let err = AudioError::Device("no mic found".into());
        assert_eq!(err.to_string(), "device error: no mic found");
    }

    #[test]
    fn display_capture_error() {
        let err = AudioError::Capture("buffer overrun".into());
        assert_eq!(err.to_string(), "capture error: buffer overrun");
    }

    #[test]
    fn display_api_error() {
        let err = AudioError::Api("401 unauthorized".into());
        assert_eq!(err.to_string(), "api error: 401 unauthorized");
    }

    #[test]
    fn display_format_error() {
        let err = AudioError::Format("unsupported codec".into());
        assert_eq!(err.to_string(), "format error: unsupported codec");
    }

    #[test]
    fn display_stream_closed_error() {
        let err = AudioError::StreamClosed("peer hung up".into());
        assert_eq!(err.to_string(), "stream closed: peer hung up");
    }
}
