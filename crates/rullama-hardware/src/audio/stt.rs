use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::error::AudioResult;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript};

/// Converts audio to text (speech-to-text / transcription).
#[async_trait]
pub trait SpeechToText: Send + Sync {
    /// Get the name of this STT backend (e.g., "openai-whisper", "whisper-local").
    fn name(&self) -> &str;

    /// Transcribe a complete audio buffer to text.
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript>;

    /// Transcribe audio from a stream in real-time, yielding partial transcripts.
    ///
    /// Each yielded [`Transcript`] may be a partial result that gets refined
    /// as more audio arrives. The final transcript has the complete text.
    ///
    /// Not all backends support streaming; those that don't should buffer
    /// internally and yield a single final result.
    fn transcribe_stream(
        &self,
        audio_stream: BoxStream<'static, AudioResult<AudioBuffer>>,
        options: &SttOptions,
    ) -> BoxStream<'static, AudioResult<Transcript>>;
}
