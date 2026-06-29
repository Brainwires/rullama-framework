use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::error::AudioResult;
use crate::audio::types::{AudioBuffer, TtsOptions, Voice};

/// Converts text to audio (text-to-speech / synthesis).
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    /// Get the name of this TTS backend (e.g., "openai-tts", "piper-local").
    fn name(&self) -> &str;

    /// List available voices.
    async fn list_voices(&self) -> AudioResult<Vec<Voice>>;

    /// Synthesize text to a complete audio buffer.
    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer>;

    /// Synthesize text to a stream of audio chunks for real-time playback.
    ///
    /// Yields audio buffers as they become available, enabling low-latency
    /// playback while synthesis continues.
    fn synthesize_stream(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> BoxStream<'static, AudioResult<AudioBuffer>>;
}
